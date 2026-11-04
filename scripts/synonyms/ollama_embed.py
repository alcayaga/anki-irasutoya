"""Vector embedding client, catalog post vector indexing, and hybrid pre-search engine."""
import array
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    np = None

from addon.catalog import CatalogIndex
from addon.text_utils import clean_search_query
from scripts.synonyms.common import _readonly_catalog_uri

logger = logging.getLogger("synonyms.ollama_embed")

__all__ = [
    "CatalogVectorIndex",
    "OllamaEmbedClient",
    "format_candidate_context_block",
    "l2_normalize",
    "search_hybrid_presearch",
]


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

    def embed_batch(
        self,
        texts: Sequence[str],
        model: str = "qwen3-embedding:4b",
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Compute normalized vector embeddings for a batch of strings.

        Args:
            texts: Sequence of Japanese or English phrases to embed.
            model: Ollama embedding model name.
            batch_size: Maximum items per /api/embed HTTP request.

        Returns:
            List of L2-normalized float embedding vectors.
        """
        clean_texts = [str(t).strip() for t in texts]
        if not clean_texts:
            return []

        results: List[List[float]] = []
        total_chunks = (len(clean_texts) + batch_size - 1) // batch_size
        for chunk_idx, i in enumerate(range(0, len(clean_texts), batch_size), start=1):
            chunk = clean_texts[i : i + batch_size]
            payload = json.dumps({"model": model, "input": chunk}).encode("utf-8")
            req = urllib.request.Request(
                self._embed_url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "anki-irasutoya/1.0"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                logger.warning("Ollama /api/embed failed for batch of %d items: %s", len(chunk), exc)
                return []

            if not isinstance(resp_data, dict):
                logger.warning("Malformed Ollama /api/embed response (not a dict)")
                return []

            embeddings = resp_data.get("embeddings", [])
            if not isinstance(embeddings, list):
                logger.warning("Malformed 'embeddings' field in /api/embed response")
                return []

            for raw_vec in embeddings:
                if isinstance(raw_vec, list):
                    results.append(l2_normalize(raw_vec))
                else:
                    results.append([])

            if total_chunks > 5 and (chunk_idx % 5 == 0 or chunk_idx == total_chunks):
                logger.info("Embedded batch %d/%d (%d posts)", chunk_idx, total_chunks, len(results))

        return results

    def embed_single(
        self,
        text: str,
        model: str = "qwen3-embedding:4b",
    ) -> Optional[List[float]]:
        """Compute normalized embedding vector for a single string.

        Args:
            text: Text to embed.
            model: Ollama embedding model name.

        Returns:
            L2-normalized float embedding vector, or None on failure.
        """
        clean = text.strip()
        if not clean:
            return None
        res = self.embed_batch([clean], model=model, batch_size=1)
        return res[0] if (res and len(res) == 1 and res[0]) else None


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
        build_if_missing: bool = False,
    ) -> Optional["CatalogVectorIndex"]:
        """Load cached post embeddings or compute and index catalog posts.

        Args:
            catalog_path: Path to destination catalog.sqlite database.
            client: Configured OllamaEmbedClient instance.
            model: Ollama embedding model tag.
            cache_dir: Directory for storing vector cache.
            batch_size: Embedding batch size.
            build_if_missing: If True, builds vector index from scratch if cache is missing.

        Returns:
            CatalogVectorIndex instance if available/built, None otherwise.
        """
        clean_model = re.sub(r"[^\w\-.]+", "_", model)
        cache_base = os.path.join(cache_dir, f"catalog_post_embeddings_{clean_model}")

        # Compute catalog sha256 to ensure cache validity
        try:
            h = hashlib.sha256()
            with open(catalog_path, "rb") as f_cat:
                while chunk := f_cat.read(131072):
                    h.update(chunk)
            catalog_sha256 = h.hexdigest()
        except Exception:
            catalog_sha256 = ""

        cached = cls.load(cache_base)
        if cached is not None and catalog_sha256:
            meta_path = f"{cache_base}.json"
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("catalog_sha256") == catalog_sha256:
                    logger.info("Loaded cached vector index (%d posts, dim=%d) from %s", len(cached.post_ids), cached.dim, cache_base)
                    return cached
                logger.info("Catalog SHA256 changed; vector cache at %s is stale", cache_base)
            except Exception:
                pass

        if not build_if_missing:
            logger.info("Vector index cache not found at %s; using lexical pre-search (build with --build-vector-index)", cache_base)
            return None

        # Build vectors from scratch
        logger.info("Building post vector index using model '%s' from %s...", model, catalog_path)
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
        texts = [(f"{r[1]} — {r[2]}" if r[2] else r[1]) for r in rows]

        embeddings = client.embed_batch(texts, model=model, batch_size=batch_size)
        if not embeddings or len(embeddings) != len(post_ids):
            logger.warning("Failed to compute complete embeddings for %d posts", len(post_ids))
            return None

        dim = len(embeddings[0])
        if dim <= 0 or any(len(vec) != dim for vec in embeddings):
            logger.warning("Inconsistent or empty embedding dimensions; skipping vector index build")
            return None

        if np is not None:
            vectors = np.array(embeddings, dtype=np.float32)
        else:
            vectors = array.array("f")
            for vec in embeddings:
                vectors.extend(vec)

        idx = cls(post_ids=post_ids, vectors=vectors, dim=dim)
        try:
            idx.save(cache_base, meta_extra={"catalog_sha256": catalog_sha256, "model": model})
            logger.info("Saved vector index (%d posts, dim=%d) to %s", len(post_ids), dim, cache_base)
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


def search_hybrid_presearch(
    term: str,
    catalog_path: str,
    index: Optional[CatalogIndex] = None,
    vector_index: Optional[CatalogVectorIndex] = None,
    embed_client: Optional[OllamaEmbedClient] = None,
    embed_model: str = "qwen3-embedding:4b",
    conn: Optional[sqlite3.Connection] = None,
    limit: int = 5,
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
    seen_post_ids = set()

    try:
        cur = conn.cursor()

        # 1. Lexical Substring & Exact Match on Post Title / Description
        escaped_q = clean_q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        cur.execute(
            "SELECT id, blogger_id, title, description FROM posts WHERE title LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' LIMIT ?",
            (f"%{escaped_q}%", f"%{escaped_q}%", limit),
        )
        for pid, _bid, ptitle, pdesc in cur.fetchall():
            if pid not in seen_post_ids:
                seen_post_ids.add(pid)
                m = re.match(r"^(.+?)のイラスト(?:$|（)", ptitle)
                suggested = m.group(1) if m else ptitle
                hits.append({
                    "post_id": pid,
                    "title": ptitle,
                    "description": (pdesc[:120] if pdesc else ""),
                    "suggested_target": suggested,
                    "source": "lexical",
                })

        # Also inspect in-memory index if lexical query yielded few hits
        if len(hits) < limit and index is not None:
            records = index.search(clean_q)
            for rec in records:
                if rec.post_id and rec.post_id not in seen_post_ids:
                    # Look up post details by blogger_id
                    cur.execute("SELECT id, title, description FROM posts WHERE blogger_id = ?", (rec.post_id,))
                    p_row = cur.fetchone()
                    if p_row and p_row[0] not in seen_post_ids:
                        seen_post_ids.add(p_row[0])
                        m = re.match(r"^(.+?)のイラスト(?:$|（)", p_row[1])
                        suggested = m.group(1) if m else p_row[1]
                        hits.append({
                            "post_id": p_row[0],
                            "title": p_row[1],
                            "description": (p_row[2][:120] if p_row[2] else ""),
                            "suggested_target": suggested,
                            "source": "lexical",
                        })
                        if len(hits) >= limit:
                            break

        # 2. Semantic Vector Similarity Pre-Search
        if vector_index is not None and embed_client is not None and len(hits) < limit:
            q_vec = embed_client.embed_single(clean_q, model=embed_model)
            if q_vec:
                vector_hits = vector_index.search_similar(q_vec, top_k=limit)
                for pid, _score in vector_hits:
                    if pid not in seen_post_ids:
                        cur.execute("SELECT id, title, description FROM posts WHERE id = ?", (pid,))
                        p_row = cur.fetchone()
                        if p_row:
                            seen_post_ids.add(pid)
                            m = re.match(r"^(.+?)のイラスト(?:$|（)", p_row[1])
                            suggested = m.group(1) if m else p_row[1]
                            hits.append({
                                "post_id": p_row[0],
                                "title": p_row[1],
                                "description": (p_row[2][:120] if p_row[2] else ""),
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
