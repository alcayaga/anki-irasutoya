"""Vector embedding client, catalog post vector indexing, and hybrid pre-search engine."""
import array
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except ImportError:
    np = None

from addon.catalog import CatalogIndex
from addon.ranking import has_title_illustration_match
from addon.text_utils import clean_search_query
from scripts.synonyms.common import (
    _KeepAliveJSONClient,
    _readonly_catalog_uri,
    filter_relevant_illustrations,
)

logger = logging.getLogger("synonyms.ollama_embed")

QWEN3_SEARCH_INSTRUCT = (
    "Given a Japanese vocabulary term, retrieve Japanese illustration descriptions "
    "that visually depict the concept or a concrete scene for it"
)

__all__ = [
    "CatalogVectorIndex",
    "OllamaEmbedClient",
    "QWEN3_SEARCH_INSTRUCT",
    "clean_post_for_embedding",
    "compute_catalog_posts_sha256",
    "format_candidate_context_block",
    "format_query_for_embedding",
    "l2_normalize",
    "resolve_valid_post_target",
    "search_hybrid_presearch",
]


def compute_catalog_posts_sha256(catalog_path: str) -> str:
    """Compute deterministic SHA-256 fingerprint of catalog posts embedding fields.

    Args:
        catalog_path: Path to catalog.sqlite.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    try:
        h = hashlib.sha256()
        conn = sqlite3.connect(_readonly_catalog_uri(catalog_path), uri=True)
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, title, description FROM posts ORDER BY id")
            for r in cur.fetchall():
                h.update(str(r[0]).encode("utf-8"))
                h.update(str(r[1] or "").encode("utf-8"))
                h.update(str(r[2] or "").encode("utf-8"))
        finally:
            conn.close()
        return h.hexdigest()
    except Exception as exc:
        logger.warning("Failed to fingerprint catalog posts in %s: %s", catalog_path, exc)
        return ""


def clean_post_for_embedding(title: str, description: str) -> str:
    """Strip repetitive Irasutoya boilerplate from post title and description.

    Args:
        title: Post title (e.g. 'スクール水着のイラスト（上下セット）').
        description: Post description text.

    Returns:
        Concise, semantically rich string combining title and description.
    """
    t = re.sub(r"のイラスト", "", title or "").strip()
    d = (description or "").strip()
    d = re.sub(r"を描いたイラストです。?$", "", d)
    d = re.sub(r"の(イラスト|キャラクター)です。?$", "", d)
    d = re.sub(r"イラストです。?$", "", d)
    d = re.sub(r"です。?$", "", d).strip()
    if t and d:
        return f"{t}: {d}"
    return t or d or (title or "").strip()


def format_query_for_embedding(
    term: str,
    english_glosses: Optional[Sequence[str]] = None,
    instruct: str = QWEN3_SEARCH_INSTRUCT,
) -> str:
    """Format a runtime search query with Qwen3-Embedding's instruction prefix.

    Args:
        term: Japanese vocabulary headword.
        english_glosses: Optional list of English gloss definitions.
        instruct: Task instruction header for asymmetric retrieval.

    Returns:
        Formatted instruction-aware query string.
    """
    clean_term = term.strip()
    if not clean_term:
        return ""
    gloss_str = ", ".join(g.strip() for g in (english_glosses or [])[:3] if g and g.strip())
    query_body = f"{clean_term} ({gloss_str})" if gloss_str else clean_term
    if instruct:
        return f"Instruct: {instruct}\nQuery: {query_body}"
    return query_body


def l2_normalize(vector: Sequence[float]) -> List[float]:
    """Normalize a floating point vector to unit L2 norm.

    Args:
        vector: Sequence of numeric weights.

    Returns:
        List of normalized float values with Euclidean norm equal to 1.0.
    """
    if not vector:
        return []
    sq_sum = sum(x * x for x in vector)
    if sq_sum <= 0.0:
        return [0.0] * len(vector)
    inv_norm = 1.0 / math.sqrt(sq_sum)
    return [float(x * inv_norm) for x in vector]


class OllamaEmbedClient:
    """Client for generating dense text embeddings via Ollama /api/embed."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 30.0) -> None:
        """Initialize embedding client.

        Args:
            base_url: Ollama server base URL.
            timeout: Network request timeout in seconds.
        """
        parsed = urllib.parse.urlparse(base_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/api/embed"):
            path = path[:-len("/api/embed")]
        elif path.endswith("/api/chat"):
            path = path[:-len("/api/chat")]
        elif path.endswith("/api"):
            path = path[:-len("/api")]
        clean_base = f"{parsed.scheme}://{parsed.netloc}{path}" if parsed.netloc else "http://localhost:11434"
        self._embed_url = f"{clean_base.rstrip('/')}/api/embed"
        self._timeout = timeout
        self._http_client = _KeepAliveJSONClient(timeout=timeout)

    def _embed_chunk(
        self,
        chunk: Sequence[str],
        model: str,
        dimensions: int,
        options: Dict[str, Any],
    ) -> List[List[float]]:
        """Embed a single chunk with bisect-retry on HTTP or network errors."""
        payload: Dict[str, Any] = {
            "model": model,
            "input": list(chunk),
            "dimensions": dimensions,
            "options": options,
        }
        try:
            resp_data = self._http_client.post_json(self._embed_url, payload, timeout=self._timeout)
            embeddings = resp_data.get("embeddings", [])
            if isinstance(embeddings, list) and len(embeddings) == len(chunk):
                return [
                    (l2_normalize(vec[:dimensions]) if isinstance(vec, list) else [])
                    for vec in embeddings
                ]
            logger.warning("Malformed 'embeddings' response for chunk of size %d", len(chunk))
        except Exception as exc:
            if len(chunk) > 1:
                mid = len(chunk) // 2
                logger.info(
                    "Bisecting failed batch (%d items -> %d + %d): %s",
                    len(chunk),
                    mid,
                    len(chunk) - mid,
                    exc,
                )
                left = self._embed_chunk(chunk[:mid], model=model, dimensions=dimensions, options=options)
                right = self._embed_chunk(chunk[mid:], model=model, dimensions=dimensions, options=options)
                return left + right
            logger.warning("Ollama /api/embed failed for single item '%s': %s", chunk[0][:50], exc)

        return [[] for _ in chunk]

    def embed_batch(
        self,
        texts: Sequence[str],
        model: str = "qwen3-embedding:4b",
        batch_size: int = 100,
        dimensions: int = 512,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        """Compute normalized vector embeddings for a batch of strings.

        Args:
            texts: Sequence of Japanese or English phrases to embed.
            model: Ollama embedding model name.
            batch_size: Maximum items per /api/embed HTTP request.
            dimensions: Output embedding dimensionality (e.g. 512 for Matryoshka).
            options: Optional Ollama options dict (defaults to {'num_ctx': 4096}).

        Returns:
            List of L2-normalized float embedding vectors.
        """
        clean_texts = [str(t).strip() for t in texts]
        if not clean_texts:
            return []

        eff_options = {"num_ctx": 4096}
        if options:
            eff_options.update(options)

        results: List[List[float]] = []
        total_chunks = (len(clean_texts) + batch_size - 1) // batch_size
        for chunk_idx, i in enumerate(range(0, len(clean_texts), batch_size), start=1):
            chunk = clean_texts[i : i + batch_size]
            chunk_vectors = self._embed_chunk(
                chunk,
                model=model,
                dimensions=dimensions,
                options=eff_options,
            )
            results.extend(chunk_vectors)
            if total_chunks > 5 and (chunk_idx % 5 == 0 or chunk_idx == total_chunks):
                logger.info("Embedded batch %d/%d (%d items)", chunk_idx, total_chunks, len(results))

        return results

    def embed_single(
        self,
        text: str,
        model: str = "qwen3-embedding:4b",
        dimensions: int = 512,
        options: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[float]]:
        """Compute normalized embedding vector for a single string.

        Args:
            text: Text to embed.
            model: Ollama embedding model name.
            dimensions: Output embedding dimensionality.
            options: Optional Ollama options dict.

        Returns:
            L2-normalized float embedding vector, or None on failure.
        """
        clean = text.strip()
        if not clean:
            return None
        res = self.embed_batch(
            [clean],
            model=model,
            batch_size=1,
            dimensions=dimensions,
            options=options,
        )
        return res[0] if (res and len(res) == 1 and res[0]) else None

    def close(self) -> None:
        """Close persistent HTTP client connections."""
        self._http_client.close()


def _save_partial_checkpoint(
    base_path: str,
    embeddings: Sequence[Sequence[float]],
    dim: int,
    catalog_sha256: str,
    model: str,
) -> None:
    """Save interim progress checkpoint to .partial.bin and .partial.json.

    Args:
        base_path: Base filesystem path without extension.
        embeddings: Sequence of normalized embedding float vectors.
        dim: Dimensionality of each vector.
        catalog_sha256: Hash of the catalog SQLite file.
        model: Embedding model name.
    """
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
    meta_path = f"{base_path}.json"
    bin_path = f"{base_path}.bin"

    meta: Dict[str, Any] = {
        "count": len(embeddings),
        "dim": dim,
        "catalog_sha256": catalog_sha256,
        "model": model,
    }
    tmp_meta = f"{meta_path}.tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_meta, meta_path)

    tmp_bin = f"{bin_path}.tmp"
    with open(tmp_bin, "wb") as f_bin:
        arr = array.array("f")
        for vec in embeddings:
            arr.extend(vec)
        arr.tofile(f_bin)
    os.replace(tmp_bin, bin_path)


class CatalogVectorIndex:
    """In-memory vector index for Irasutoya catalog post embeddings."""

    def __init__(
        self,
        post_ids: Sequence[int],
        vectors: Any,
        dim: int,
    ) -> None:
        """Initialize vector index with post identifiers and dense vectors.

        Args:
            post_ids: Sequence of integer SQLite post primary key IDs.
            vectors: 2D array of shape (N, dim), either numpy.ndarray or array.array('f').
            dim: Embedding vector dimensionality.
        """
        self.post_ids: List[int] = list(post_ids)
        self.dim: int = dim
        self._vectors = vectors

    @classmethod
    def load(cls, cache_path: str) -> Optional["CatalogVectorIndex"]:
        """Load cached vector index from binary and metadata files.

        Args:
            cache_path: Base path to binary vector file (without extension or with .bin).

        Returns:
            Loaded CatalogVectorIndex instance, or None if files are missing/corrupted.
        """
        base = cache_path[:-4] if cache_path.endswith(".bin") else cache_path
        bin_path = f"{base}.bin"
        meta_path = f"{base}.json"

        if not os.path.isfile(bin_path) or not os.path.isfile(meta_path):
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            post_ids = meta.get("post_ids", [])
            dim = int(meta.get("dim", 0))
            count = int(meta.get("count", 0))
            if not post_ids or dim <= 0 or count <= 0 or len(post_ids) != count:
                return None

            expected_bytes = count * dim * 4
            if os.path.getsize(bin_path) != expected_bytes:
                logger.warning("Vector cache file %s size mismatch; invalidating cache", bin_path)
                return None

            if np is not None:
                vectors = np.fromfile(bin_path, dtype=np.float32).reshape((count, dim))
            else:
                vectors = array.array("f")
                with open(bin_path, "rb") as f_bin:
                    vectors.fromfile(f_bin, count * dim)

            return cls(post_ids=post_ids, vectors=vectors, dim=dim)
        except Exception as exc:
            logger.warning("Failed to load cached vector index from %s: %s", base, exc)
            return None

    def save(self, cache_path: str, meta_extra: Optional[Dict[str, Any]] = None) -> None:
        """Persist vector index to binary and metadata files.

        Args:
            cache_path: Base path for vector cache files.
            meta_extra: Optional extra metadata dictionary (e.g. catalog sha256).
        """
        base = cache_path[:-4] if cache_path.endswith(".bin") else cache_path
        bin_path = f"{base}.bin"
        meta_path = f"{base}.json"
        os.makedirs(os.path.dirname(os.path.abspath(bin_path)), exist_ok=True)

        meta: Dict[str, Any] = {
            "count": len(self.post_ids),
            "dim": self.dim,
            "post_ids": self.post_ids,
        }
        if meta_extra:
            meta.update(meta_extra)

        tmp_meta = f"{meta_path}.tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_meta, meta_path)

        tmp_bin = f"{bin_path}.tmp"
        with open(tmp_bin, "wb") as f_bin:
            if np is not None and isinstance(self._vectors, np.ndarray):
                self._vectors.astype(np.float32).tofile(f_bin)
            elif isinstance(self._vectors, array.array):
                self._vectors.tofile(f_bin)
            else:
                # Flat sequence fallback
                arr = array.array("f", self._vectors)
                arr.tofile(f_bin)
        os.replace(tmp_bin, bin_path)

    @classmethod
    def build_or_load(
        cls,
        catalog_path: str,
        client: OllamaEmbedClient,
        model: str = "qwen3-embedding:4b",
        cache_dir: str = ".catalog_cache",
        batch_size: int = 100,
        dimensions: int = 512,
        build_if_missing: bool = False,
    ) -> Optional["CatalogVectorIndex"]:
        """Load cached post embeddings or compute and index catalog posts.

        Args:
            catalog_path: Path to destination catalog.sqlite database.
            client: Configured OllamaEmbedClient instance.
            model: Ollama embedding model tag.
            cache_dir: Directory for storing vector cache.
            batch_size: Embedding batch size.
            dimensions: Target embedding vector dimensionality.
            build_if_missing: If True, builds vector index from scratch if cache is missing.

        Returns:
            CatalogVectorIndex instance if available/built, None otherwise.
        """
        clean_model = re.sub(r"[^\w\-.]+", "_", model)
        cache_base = os.path.join(cache_dir, f"catalog_post_embeddings_{clean_model}")

        # Compute deterministic fingerprint of embedding inputs (posts id, title, description)
        catalog_sha256 = compute_catalog_posts_sha256(catalog_path)

        cached = cls.load(cache_base)
        if cached is not None and catalog_sha256:
            meta_path = f"{cache_base}.json"
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if (
                    meta.get("catalog_sha256") == catalog_sha256
                    and meta.get("model") == model
                    and cached.dim == dimensions
                ):
                    logger.info(
                        "Loaded cached vector index (%d posts, dim=%d) from %s",
                        len(cached.post_ids),
                        cached.dim,
                        cache_base,
                    )
                    return cached
                logger.info("Catalog SHA256 or dimension changed; vector cache at %s is stale", cache_base)
            except Exception:
                pass

        if not build_if_missing:
            logger.info(
                "Vector index cache not found at %s; using lexical pre-search (build with --build-vector-index)",
                cache_base,
            )
            return None

        # Build vectors from scratch
        logger.info(
            "Building post vector index using model '%s' (dim=%d) from %s...",
            model,
            dimensions,
            catalog_path,
        )
        conn = sqlite3.connect(_readonly_catalog_uri(catalog_path), uri=True)
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, title, description FROM posts ORDER BY id ASC")
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            logger.warning("No posts found in %s to embed", catalog_path)
            return None

        post_ids = [r[0] for r in rows]
        texts = [clean_post_for_embedding(r[1], r[2]) for r in rows]
        total_posts = len(post_ids)

        partial_base = f"{cache_base}.partial"
        partial_bin = f"{partial_base}.bin"
        partial_meta = f"{partial_base}.json"
        embeddings: List[List[float]] = []

        if os.path.isfile(partial_bin) and os.path.isfile(partial_meta):
            try:
                with open(partial_meta, "r", encoding="utf-8") as f_pm:
                    p_meta = json.load(f_pm)
                p_sha = p_meta.get("catalog_sha256")
                p_model = p_meta.get("model")
                p_dim = int(p_meta.get("dim", 0))
                p_count = int(p_meta.get("count", 0))
                if (
                    p_sha == catalog_sha256
                    and p_model == model
                    and p_dim == dimensions
                    and 0 < p_count < total_posts
                    and os.path.getsize(partial_bin) == p_count * p_dim * 4
                ):
                    with open(partial_bin, "rb") as f_pbin:
                        arr = array.array("f")
                        arr.fromfile(f_pbin, p_count * p_dim)
                    for idx in range(p_count):
                        embeddings.append(list(arr[idx * p_dim : (idx + 1) * p_dim]))
                    logger.info(
                        "Resuming vector index build from checkpoint: %d/%d posts completed",
                        len(embeddings),
                        total_posts,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to load partial checkpoint from %s: %s; restarting build",
                    partial_base,
                    exc,
                )
                embeddings.clear()

        start_idx = len(embeddings)
        t_start = time.monotonic()
        posts_done_this_run = 0

        for i in range(start_idx, total_posts, batch_size):
            chunk_texts = texts[i : i + batch_size]
            chunk_embs = client.embed_batch(
                chunk_texts,
                model=model,
                batch_size=batch_size,
                dimensions=dimensions,
            )
            if not chunk_embs or len(chunk_embs) != len(chunk_texts) or any(len(v) != dimensions for v in chunk_embs):
                logger.error("Failed to compute valid embeddings for batch starting at index %d; stopping build", i)
                return None
            embeddings.extend(chunk_embs)
            posts_done_this_run += len(chunk_texts)

            elapsed = time.monotonic() - t_start
            rate = posts_done_this_run / elapsed if elapsed > 0 else 0.0
            remaining_posts = total_posts - len(embeddings)
            eta_mins = (remaining_posts / rate) / 60.0 if rate > 0 else 0.0
            current_batch = (len(embeddings) + batch_size - 1) // batch_size
            total_batches = (total_posts + batch_size - 1) // batch_size
            pct = (len(embeddings) / total_posts) * 100.0
            logger.info(
                "[%d/%d] Embedded %d/%d posts (%.1f%%) | %.1f posts/s | ETA %.1fm",
                current_batch,
                total_batches,
                len(embeddings),
                total_posts,
                pct,
                rate,
                eta_mins,
            )

            if current_batch % 5 == 0 or len(embeddings) == total_posts:
                _save_partial_checkpoint(
                    partial_base,
                    embeddings=embeddings,
                    dim=dimensions,
                    catalog_sha256=catalog_sha256,
                    model=model,
                )

        if np is not None:
            vectors = np.array(embeddings, dtype=np.float32)
        else:
            vectors = array.array("f")
            for vec in embeddings:
                vectors.extend(vec)

        idx = cls(post_ids=post_ids, vectors=vectors, dim=dimensions)
        try:
            idx.save(cache_base, meta_extra={"catalog_sha256": catalog_sha256, "model": model})
            logger.info("Saved vector index (%d posts, dim=%d) to %s", len(post_ids), dimensions, cache_base)
            for p_file in (f"{partial_base}.bin", f"{partial_base}.json"):
                try:
                    if os.path.isfile(p_file):
                        os.remove(p_file)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Failed to save vector index cache: %s", exc)

        return idx

    def search_similar(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Search top-k most similar catalog posts using cosine dot product.

        Args:
            query_vector: L2-normalized query embedding vector.
            top_k: Number of highest-similarity posts to return.

        Returns:
            List of (post_id, similarity_score) sorted by descending similarity.
        """
        if not self.post_ids or top_k <= 0:
            return []

        norm_q = l2_normalize(query_vector)
        if not norm_q or len(norm_q) != self.dim:
            return []

        count = len(self.post_ids)
        if np is not None and isinstance(self._vectors, np.ndarray):
            q_arr = np.array(norm_q, dtype=np.float32)
            scores = np.dot(self._vectors, q_arr)
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [(self.post_ids[int(i)], float(scores[int(i)])) for i in top_idx]

        # Pure Python fallback
        scores_list: List[Tuple[int, float]] = []
        dim = self.dim
        vecs = self._vectors
        for i in range(count):
            offset = i * dim
            dot = sum(norm_q[j] * vecs[offset + j] for j in range(dim))
            scores_list.append((self.post_ids[i], dot))

        scores_list.sort(key=lambda item: item[1], reverse=True)
        return scores_list[:top_k]


def resolve_valid_post_target(
    post_id: int,
    post_title: str,
    index: Optional[CatalogIndex],
    conn: sqlite3.Connection,
) -> str:
    """Resolve a clean, runtime-valid search target string for a catalog post.

    Inspects the post title and associated image titles, canonicalizing candidates
    via clean_search_query() and validating that the target produces title
    illustration matches in the catalog index.

    Args:
        post_id: SQLite integer posts.id.
        post_title: Title of the Blogger post.
        index: Optional pre-loaded CatalogIndex for runtime validation.
        conn: Open SQLite connection to catalog.sqlite.

    Returns:
        A validated search target string if one passes, or fallback clean query.
    """
    m = re.match(r"^(.+?)のイラスト(?:$|（)", post_title)
    primary = m.group(1) if m else post_title
    clean_primary = clean_search_query(primary)

    if index is None:
        return clean_primary or clean_search_query(post_title) or post_title

    if clean_primary:
        res = filter_relevant_illustrations(index.search(clean_primary), clean_primary)
        if res and has_title_illustration_match(res, clean_primary):
            return clean_primary

    clean_full = clean_search_query(post_title)
    if clean_full:
        res = filter_relevant_illustrations(index.search(clean_full), clean_full)
        if res and has_title_illustration_match(res, clean_full):
            return clean_full

    try:
        cur = conn.cursor()
        cur.execute("SELECT title FROM images WHERE post_id = ? ORDER BY id ASC LIMIT 5", (post_id,))
        for (img_t,) in cur.fetchall():
            m_img = re.match(r"^(.+?)のイラスト(?:$|（)", img_t)
            c_img = clean_search_query(m_img.group(1) if m_img else img_t)
            if c_img:
                res = filter_relevant_illustrations(index.search(c_img), c_img)
                if res and has_title_illustration_match(res, c_img):
                    return c_img
    except Exception as exc:
        logger.debug("Failed image title lookup for post %d: %s", post_id, exc)

    return clean_primary or clean_full or post_title


def search_hybrid_presearch(
    term: str,
    catalog_path: str,
    index: Optional[CatalogIndex] = None,
    vector_index: Optional[CatalogVectorIndex] = None,
    embed_client: Optional[OllamaEmbedClient] = None,
    embed_model: str = "qwen3-embedding:4b",
    conn: Optional[sqlite3.Connection] = None,
    limit: int = 5,
    english_glosses: Optional[Sequence[str]] = None,
    query_vector: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Perform upfront hybrid lexical + semantic pre-search over catalog posts.

    Args:
        term: Search headword or Japanese concept.
        catalog_path: Path to catalog.sqlite.
        index: Optional pre-loaded CatalogIndex.
        vector_index: Optional pre-loaded CatalogVectorIndex.
        embed_client: Optional OllamaEmbedClient instance.
        embed_model: Ollama embedding model tag.
        conn: Optional SQLite connection.
        limit: Max combined candidates to return.
        english_glosses: Optional sequence of English gloss definitions.
        query_vector: Optional pre-computed query embedding vector.

    Returns:
        List of candidate post dictionaries with 'post_id', 'title', 'description',
        'suggested_target', and 'source'.
    """
    clean_q = clean_search_query(term)
    if not clean_q:
        return []

    close_conn = False
    if conn is None:
        conn = sqlite3.connect(_readonly_catalog_uri(catalog_path), uri=True)
        close_conn = True

    hits: List[Dict[str, Any]] = []
    seen_post_ids: Set[int] = set()
    seen_blogger_ids: Set[str] = set()

    try:
        cur = conn.cursor()

        # 1. Lexical Substring & Exact Match on Post Title First, then Description
        has_semantic = vector_index is not None and (query_vector is not None or embed_client is not None)
        lexical_cap = max(1, limit - 2) if has_semantic else limit

        escaped_q = clean_q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        # 1a. Match post title first (direct high-precision hits)
        cur.execute(
            "SELECT id, blogger_id, title, description FROM posts WHERE title LIKE ? ESCAPE '\\' LIMIT ?",
            (f"%{escaped_q}%", lexical_cap),
        )
        for pid, bid, ptitle, pdesc in cur.fetchall():
            if pid not in seen_post_ids:
                seen_post_ids.add(pid)
                if bid:
                    seen_blogger_ids.add(str(bid))
                suggested = resolve_valid_post_target(pid, ptitle, index, conn)
                hits.append({
                    "post_id": pid,
                    "title": ptitle,
                    "description": (pdesc[:120] if pdesc else ""),
                    "suggested_target": suggested,
                    "source": "lexical",
                })

        # 1b. If fewer than lexical_cap, search post description
        if len(hits) < lexical_cap:
            cur.execute(
                "SELECT id, blogger_id, title, description FROM posts WHERE description LIKE ? ESCAPE '\\' LIMIT ?",
                (f"%{escaped_q}%", lexical_cap - len(hits)),
            )
            for pid, bid, ptitle, pdesc in cur.fetchall():
                if pid not in seen_post_ids:
                    seen_post_ids.add(pid)
                    if bid:
                        seen_blogger_ids.add(str(bid))
                    suggested = resolve_valid_post_target(pid, ptitle, index, conn)
                    hits.append({
                        "post_id": pid,
                        "title": ptitle,
                        "description": (pdesc[:120] if pdesc else ""),
                        "suggested_target": suggested,
                        "source": "lexical",
                    })

        # Also inspect in-memory index if lexical query yielded few hits
        if len(hits) < lexical_cap and index is not None:
            records = index.search(clean_q)
            for rec in records:
                if rec.post_id and rec.post_id not in seen_blogger_ids:
                    seen_blogger_ids.add(rec.post_id)
                    # Look up post details by blogger_id
                    cur.execute("SELECT id, title, description FROM posts WHERE blogger_id = ?", (rec.post_id,))
                    p_row = cur.fetchone()
                    if p_row and p_row[0] not in seen_post_ids:
                        seen_post_ids.add(p_row[0])
                        suggested = resolve_valid_post_target(p_row[0], p_row[1], index, conn)
                        hits.append({
                            "post_id": p_row[0],
                            "title": p_row[1],
                            "description": (p_row[2][:120] if p_row[2] else ""),
                            "suggested_target": suggested,
                            "source": "lexical",
                        })
                        if len(hits) >= lexical_cap:
                            break

        # 2. Semantic Vector Similarity Pre-Search
        if has_semantic and len(hits) < limit and vector_index is not None:
            q_vec = query_vector
            if q_vec is None and embed_client is not None:
                formatted_q = format_query_for_embedding(clean_q, english_glosses=english_glosses)
                q_vec = embed_client.embed_single(
                    formatted_q,
                    model=embed_model,
                    dimensions=vector_index.dim,
                )
            if q_vec:
                vector_hits = vector_index.search_similar(q_vec, top_k=limit * 2)
                for pid, _score in vector_hits:
                    if pid not in seen_post_ids:
                        cur.execute("SELECT id, blogger_id, title, description FROM posts WHERE id = ?", (pid,))
                        p_row = cur.fetchone()
                        if p_row:
                            seen_post_ids.add(pid)
                            if p_row[1]:
                                seen_blogger_ids.add(str(p_row[1]))
                            suggested = resolve_valid_post_target(p_row[0], p_row[2], index, conn)
                            hits.append({
                                "post_id": p_row[0],
                                "title": p_row[2],
                                "description": (p_row[3][:120] if p_row[3] else ""),
                                "suggested_target": suggested,
                                "source": "semantic",
                            })
                            if len(hits) >= limit:
                                break

        return hits[:limit]
    finally:
        if close_conn:
            conn.close()


def format_candidate_context_block(hits: Sequence[Dict[str, Any]]) -> str:
    """Format candidate catalog illustration hits into prompt context string.

    Args:
        hits: Sequence of hit dictionaries returned by search_hybrid_presearch.

    Returns:
        Formatted multi-line markdown/text string to inject into LLM user prompt.
    """
    if not hits:
        return ""

    lines = ["Context: Candidate Catalog Illustrations:"]
    for i, h in enumerate(hits, 1):
        target = h.get("suggested_target", "")
        title = h.get("title", "")
        desc = h.get("description", "")
        desc_part = f" — {desc}" if desc else ""
        lines.append(f"{i}. Target: 「{target}」 (Post: {title}{desc_part})")

    return "\n".join(lines)
