"""Ollama LLM agent loop, search_catalog tool execution, and concept compilation."""
import inspect
import json
import logging
import math
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from addon.catalog import CatalogStore
from addon.ranking import has_title_illustration_match
from addon.text_utils import clean_search_query
from scripts.synonyms.common import (
    DEFAULT_LLM_CONCEPTS_PATH,
    TEXT_ASSET_TERM_KEYWORDS,
    UNVERIFIED_REASON_PREFIX,
    _is_disallowed_text_callout_target,
    _KeepAliveJSONClient,
    _readonly_catalog_uri,
    filter_relevant_illustrations,
)
from scripts.synonyms.ollama_embed import (
    CatalogVectorIndex,
    OllamaEmbedClient,
    format_candidate_context_block,
    format_query_for_embedding,
    search_hybrid_presearch,
)
from scripts.synonyms.ollama_prompt import (
    JAPANESE_CHAR_RE,
    OLLAMA_SYSTEM_PROMPT,
    SEARCH_CATALOG_TOOL,
    VALID_JAPANESE_LATIN_ACRONYMS,
    _contains_english_word,
    _parse_llm_json_response,
    search_catalog_for_agent,
)
from scripts.synonyms.yomitan_jpdb import (
    _fetch_yomitan_term_entries,
    _parse_yomitan_metadata,
    fetch_jpdb_freq_github,
    load_curated_terms,
    load_jpdb_candidates,
    propagate_kana_uk_concepts,
)

logger = logging.getLogger("synonyms.ollama_agent")

__all__ = [
    "JAPANESE_CHAR_RE",
    "OLLAMA_SYSTEM_PROMPT",
    "SEARCH_CATALOG_TOOL",
    "TEXT_ASSET_TERM_KEYWORDS",
    "VALID_JAPANESE_LATIN_ACRONYMS",
    "_KeepAliveJSONClient",
    "_contains_english_word",
    "_flush_llm_concepts",
    "_is_disallowed_text_callout_target",
    "_parse_llm_json_response",
    "check_ollama_availability",
    "run_ollama_agent_batch",
    "run_ollama_compile",
    "search_catalog_for_agent",
]


def check_ollama_availability(
    ollama_url: str = "http://localhost:11434/api/chat",
    model: Optional[str] = None,
    timeout: float = 3.0,
) -> Tuple[bool, Optional[str]]:
    """Verify that the Ollama service is reachable and responsive.

    Args:
        ollama_url: Chat endpoint URL.
        model: Optional model name to check against installed models.
        timeout: Network probe timeout in seconds.

    Returns:
        Tuple of (is_available, error_message_if_any).
    """
    parsed = urllib.parse.urlparse(ollama_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "http://localhost:11434"
    path = parsed.path or ""
    path_prefix = path[:-len("/api/chat")] if path.endswith("/api/chat") else (path[:-len("/chat")] if path.endswith("/chat") else "")
    tags_url = f"{base_url.rstrip('/')}{path_prefix.rstrip('/')}/api/tags"

    req = urllib.request.Request(
        tags_url,
        headers={"User-Agent": "anki-irasutoya/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        root_url = f"{base_url.rstrip('/')}{path_prefix.rstrip('/')}/"
        try:
            req_root = urllib.request.Request(root_url, headers={"User-Agent": "anki-irasutoya/1.0"})
            with urllib.request.urlopen(req_root, timeout=timeout) as resp:
                if resp.status == 200:
                    body = resp.read().decode("utf-8", errors="replace").strip()
                    if "Ollama is running" in body:
                        return True, None
        except Exception:
            pass
        err_detail = exc.reason if hasattr(exc, "reason") else exc
        return False, f"Cannot connect to Ollama at {base_url} ({err_detail})"
    except Exception as exc:
        return False, f"Cannot connect to Ollama at {base_url} ({exc})"

    if model and isinstance(data, dict):
        models_list = data.get("models", [])
        if isinstance(models_list, list) and models_list:
            installed: Set[str] = set()
            for m in models_list:
                if isinstance(m, dict) and "name" in m:
                    full = str(m["name"]).strip()
                    installed.add(full)
                    installed.add(full.split(":")[0])
            model_base = model.split(":")[0]
            if model not in installed and model_base not in installed:
                logger.warning(
                    "Model '%s' was not found in Ollama installed models: %s. (Ollama may attempt auto-pull on first request).",
                    model,
                    sorted(installed),
                )

    return True, None


def get_model_options_from_manifest(
    model_name: str,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve model generation options from declarative models manifest if available.

    Args:
        model_name: Name of the Ollama model.
        manifest_path: Optional path to models.yml.

    Returns:
        Dictionary of generation options.
    """
    path = manifest_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models.yml",
    )
    if not os.path.isfile(path):
        return {}
    try:
        from scripts.pull_models import load_models_manifest

        manifest = load_models_manifest(path)
        cfg = manifest.get("ollama", manifest) if isinstance(manifest.get("ollama"), dict) else manifest
        models = cfg.get("models", [])
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict) and m.get("name") == model_name:
                    opts = m.get("options")
                    if isinstance(opts, dict):
                        return dict(opts)
    except Exception as exc:
        logger.debug("Could not read options for model %s from %s: %s", model_name, path, exc)
    return {}


def run_ollama_agent_batch(
    candidates: Sequence[Dict[str, Any]],
    catalog_path: str,
    ollama_url: str = "http://localhost:11434/api/chat",
    model: str = "gemma4:26b",
    max_turns: int = 3,
    timeout: float = 120.0,
    store: Optional[CatalogStore] = None,
    vector_index: Optional[Any] = None,
    embed_client: Optional[Any] = None,
    embed_model: str = "qwen3-embedding:4b",
    on_match: Optional[Callable[..., None]] = None,
    on_reject: Optional[Callable[..., None]] = None,
    on_unverified: Optional[Callable[[str, str], None]] = None,
) -> Tuple[Dict[str, List[str]], int]:
    """Run interactive agentic search loop via Ollama /api/chat with search_catalog tool.

    Args:
        candidates: Sequence of candidate dicts with term, reading, frequency, glosses, pos_tags.
        catalog_path: Path to catalog.sqlite.
        ollama_url: Ollama chat endpoint URL.
        model: Ollama model identifier.
        max_turns: Maximum conversation turns per item (default 3).
        timeout: HTTP request timeout in seconds.
        store: Optional pre-loaded CatalogStore.
        vector_index: Optional CatalogVectorIndex for semantic vector similarity pre-search.
        embed_client: Optional OllamaEmbedClient for query embeddings.
        embed_model: Ollama embedding model tag.
        on_match: Optional callback invoked immediately with (term, targets, tier).
        on_reject: Optional callback invoked with (term, reason, tier).
        on_unverified: Optional callback invoked with (term, reason) when visualizable targets are unverified in catalog.

    Returns:
        Tuple of (mappings_dict, failed_candidates_count).
    """
    active_store = store or CatalogStore(catalog_path)
    index = active_store.get_index()
    if index is None:
        logger.error("Catalog index is unavailable at %s", catalog_path)
        return {}, 0

    conn = sqlite3.connect(_readonly_catalog_uri(catalog_path), uri=True)
    mappings: Dict[str, List[str]] = {}
    failed_candidates = 0
    consecutive_connection_failures = 0
    max_consecutive_connection_failures = 3

    http_client = _KeepAliveJSONClient(timeout=timeout)

    def _determine_arity(fn: Optional[Callable[..., Any]]) -> int:
        """Inspect callable to determine whether it accepts 3 arguments (with tier) or 2."""
        if fn is None:
            return 0
        try:
            sig = inspect.signature(fn)
            params = [
                p for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
            return 3 if has_varargs else len(params)
        except Exception:
            return 3

    on_match_arity = _determine_arity(on_match)
    on_reject_arity = _determine_arity(on_reject)

    model_opts = get_model_options_from_manifest(model)

    def _opt(name: str, cast: Callable[[Any], Any], default: Any) -> Any:
        """Return cast manifest option value with safe fallback to default on invalid or missing entries."""
        raw_val = model_opts.get(name)
        if raw_val is None:
            return default
        try:
            val = cast(raw_val)
            if isinstance(val, (int, float)) and not math.isfinite(val):
                logger.warning(
                    "Non-finite manifest option '%s'=%r for model '%s'; falling back to default %r",
                    name,
                    raw_val,
                    model,
                    default,
                )
                return default
            return val
        except (TypeError, ValueError, OverflowError):
            logger.warning(
                "Invalid manifest option '%s'=%r for model '%s'; falling back to default %r",
                name,
                raw_val,
                model,
                default,
            )
            return default

    temp_t0 = _opt("temperature_turn0", float, 0.1)
    temp_t1 = _opt("temperature_turn1", float, 0.3)
    opt_top_p = _opt("top_p", float, 0.9)
    opt_seed = _opt("seed", int, 42)
    opt_num_ctx = _opt("num_ctx", int, 4096)
    opt_num_predict = _opt("num_predict", int, 200)

    # Batch-embed all candidate queries up front if vector index is enabled
    precomputed_vectors: List[Optional[List[float]]] = [None] * len(candidates)
    if vector_index is not None and embed_client is not None and candidates:
        logger.info("Pre-computing query embeddings for %d candidates...", len(candidates))
        query_texts = [
            format_query_for_embedding(c["term"], c.get("english_glosses"))
            for c in candidates
        ]
        try:
            raw_vectors = embed_client.embed_batch(
                query_texts,
                model=embed_model,
                dimensions=vector_index.dim,
            )
            for i, vec in enumerate(raw_vectors):
                precomputed_vectors[i] = vec if vec else None
        except Exception as embed_err:
            logger.warning(
                "Upfront batch embedding failed: %s; falling back to per-candidate embedding",
                embed_err,
            )

    try:
        total = len(candidates)
        for idx, cand in enumerate(candidates, 1):
            t_start = time.monotonic()
            term = cand["term"]
            reading = cand.get("reading", "")
            freq = cand.get("frequency", "")
            glosses = ", ".join(cand.get("english_glosses", [])[:5]) or "None"

            logger.info("[%d/%d] Compiling '%s' (rank %s)...", idx, total, term, freq)

            # Upfront Turn 0 Hybrid Lexical + Semantic Pre-Search
            try:
                presearch_hits = search_hybrid_presearch(
                    term=term,
                    catalog_path=catalog_path,
                    index=index,
                    vector_index=vector_index,
                    embed_client=embed_client,
                    embed_model=embed_model,
                    conn=conn,
                    limit=5,
                    english_glosses=cand.get("english_glosses"),
                    query_vector=precomputed_vectors[idx - 1] if idx - 1 < len(precomputed_vectors) else None,
                )
            except Exception as pre_exc:
                logger.warning(
                    "Hybrid pre-search failed for '%s': %s; continuing without context",
                    term,
                    pre_exc,
                )
                presearch_hits = []
            context_block = format_candidate_context_block(presearch_hits)

            user_content_lines = [
                f"Japanese term: {term}",
                f"Reading: {reading}",
                f"Frequency Rank: {freq}",
                f"English Glosses: {glosses}",
            ]
            if context_block:
                user_content_lines.append(f"\n{context_block}")
            user_content_lines.append(
                "\nInstructions:\n"
                "Determine if this concept is visualizable in Irasutoya (as a direct concept or as a representative visual exemplar/scene), "
                "and select 1-3 concrete visual exemplar targets using the reason-first JSON schema."
            )
            user_content = "\n".join(user_content_lines)

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            try:
                for _turn in range(max_turns):
                    payload: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "tools": [SEARCH_CATALOG_TOOL],
                        "stream": False,
                        "think": False,
                        "options": {
                            "temperature": temp_t0 if _turn == 0 else temp_t1,
                            "top_p": opt_top_p,
                            "seed": opt_seed,
                            "num_ctx": opt_num_ctx,
                            "num_predict": opt_num_predict,
                        },
                    }
                    try:
                        resp_data = http_client.post_json(ollama_url, payload, timeout=timeout)
                        consecutive_connection_failures = 0
                    except Exception as exc:
                        logger.warning("Ollama API request failed for '%s': %s", term, exc)
                        failed_candidates += 1
                        consecutive_connection_failures += 1
                        if consecutive_connection_failures >= max_consecutive_connection_failures:
                            remaining = total - idx
                            logger.error(
                                "Ollama connection failed %d consecutive times (%s); aborting batch (%d candidates skipped)",
                                max_consecutive_connection_failures,
                                exc,
                                remaining,
                            )
                            failed_candidates += remaining
                            return mappings, failed_candidates
                        break

                    if not isinstance(resp_data, dict):
                        logger.warning("Malformed Ollama response for '%s' (not a dict); skipping", term)
                        failed_candidates += 1
                        break

                    msg = resp_data.get("message")
                    if not isinstance(msg, dict):
                        logger.warning("Malformed Ollama response for '%s' (message missing or not a dict); skipping", term)
                        failed_candidates += 1
                        break

                    done_reason = resp_data.get("done_reason")
                    if done_reason == "length":
                        logger.warning(
                            "Ollama response for '%s' was truncated (done_reason='length', eval_count=%s, num_predict=%s)",
                            term,
                            resp_data.get("eval_count"),
                            opt_num_predict,
                        )

                    messages.append(msg)
                    tool_calls = msg.get("tool_calls")
                    if isinstance(tool_calls, list) and tool_calls:
                        for tc in tool_calls:
                            if not isinstance(tc, dict):
                                continue
                            fn = tc.get("function")
                            if not isinstance(fn, dict):
                                fn = {}
                            fn_name = fn.get("name", "search_catalog")
                            fn_args = fn.get("arguments", {})
                            if isinstance(fn_args, str):
                                try:
                                    fn_args = json.loads(fn_args)
                                except Exception:
                                    fn_args = {}
                            if not isinstance(fn_args, dict):
                                fn_args = {}
                            raw_queries = fn_args.get("queries", [])
                            if isinstance(raw_queries, str):
                                raw_queries = [raw_queries]
                            elif not isinstance(raw_queries, list):
                                raw_queries = []
                            queries = [str(q).strip() for q in raw_queries if str(q).strip()][:3]
                            logger.info(
                                "Fallback search_catalog (turn %d) for '%s': queries=%s",
                                _turn + 1,
                                term,
                                queries,
                            )
                            tool_results = []
                            for q in queries:
                                tool_results.append(
                                    search_catalog_for_agent(
                                        catalog_path=catalog_path,
                                        query=q,
                                        limit=5,
                                        index=index,
                                        conn=conn,
                                        vector_index=vector_index,
                                        embed_client=embed_client,
                                        embed_model=embed_model,
                                    )
                                )
                            tool_msg: Dict[str, Any] = {
                                "role": "tool",
                                "name": fn_name,
                                "tool_name": fn_name,
                                "content": json.dumps(tool_results, ensure_ascii=False),
                            }
                            if "id" in tc:
                                tool_msg["tool_call_id"] = tc["id"]
                            messages.append(tool_msg)
                    else:
                        content = msg.get("content", "")
                        parsed = _parse_llm_json_response(content)
                        matched = False
                        is_malformed = False
                        if parsed:
                            raw_vis = parsed.get("visualizable")
                            if not isinstance(raw_vis, bool):
                                logger.warning(
                                    "Malformed non-boolean 'visualizable' in response for '%s': %r; skipping rejection to allow retry",
                                    term,
                                    raw_vis,
                                )
                                is_malformed = True
                                break

                            reason = str(parsed.get("reason", "")).strip()
                            raw_tier = parsed.get("tier", 2)
                            tier = int(raw_tier) if str(raw_tier) in ("1", "2") else 2

                            if raw_vis is True:
                                targets = parsed.get("targets", [])
                                if isinstance(targets, str):
                                    targets = [targets]
                                elif not isinstance(targets, list):
                                    logger.warning(
                                        "Malformed non-list targets in visualizable response for '%s': %r; skipping rejection to allow retry",
                                        term,
                                        targets,
                                    )
                                    is_malformed = True
                                    break
                                valid_targets = []
                                for t in targets:
                                    clean_t = clean_search_query(str(t))
                                    if (
                                        not clean_t
                                        or clean_search_query(clean_t) != clean_t
                                        or not JAPANESE_CHAR_RE.search(clean_t)
                                        or _contains_english_word(clean_t)
                                    ):
                                        continue
                                    if _is_disallowed_text_callout_target(clean_t, term):
                                        continue
                                    results = filter_relevant_illustrations(index.search(clean_t), clean_t)
                                    if has_title_illustration_match(results, clean_t):
                                        if clean_t not in valid_targets:
                                            valid_targets.append(clean_t)
                                if valid_targets:
                                    mappings[term] = valid_targets
                                    elapsed = time.monotonic() - t_start
                                    logger.info(
                                        "Matched '%s' (Tier %d, %d turns, %.2fs) -> %s\n  Reason: %s",
                                        term,
                                        tier,
                                        _turn + 1,
                                        elapsed,
                                        valid_targets,
                                        reason,
                                    )
                                    if on_match:
                                        if on_match_arity >= 3:
                                            on_match(term, valid_targets, tier)
                                        else:
                                            on_match(term, valid_targets)
                                    matched = True

                        if not matched and not is_malformed:
                            elapsed = time.monotonic() - t_start
                            if not parsed:
                                logger.warning(
                                    "Unparsable model response for '%s' (%.2fs); skipping rejection to allow retry on next run",
                                    term,
                                    elapsed,
                                )
                            else:
                                if parsed.get("visualizable") is True:
                                    reason = f"{UNVERIFIED_REASON_PREFIX} proposed targets {parsed.get('targets', [])!r} (LLM reason: {parsed.get('reason', '')})"
                                    logger.info(
                                        "Unverified target miss for '%s' (%d turns, %.2fs)\n  Reason: %s",
                                        term,
                                        _turn + 1,
                                        elapsed,
                                        reason,
                                    )
                                    if on_unverified:
                                        on_unverified(term, reason)
                                    elif on_reject:
                                        if on_reject_arity >= 3:
                                            on_reject(term, reason, 2)
                                        else:
                                            on_reject(term, reason)
                                else:
                                    reason = str(parsed.get("reason", "non-visualizable")).strip() or "non-visualizable"
                                    tier = int(parsed.get("tier", 2)) if str(parsed.get("tier")) in ("1", "2") else 2
                                    logger.info(
                                        "Rejected '%s' (%d turns, %.2fs)\n  Reason: %s",
                                        term,
                                        _turn + 1,
                                        elapsed,
                                        reason,
                                    )
                                    if on_reject:
                                        if on_reject_arity >= 3:
                                            on_reject(term, reason, tier)
                                        else:
                                            on_reject(term, reason)
                        break
                else:
                    elapsed = time.monotonic() - t_start
                    logger.warning(
                        "Exhausted %d turns for '%s' without a final answer (%.2fs); leaving it for the next run",
                        max_turns,
                        term,
                        elapsed,
                    )
            except Exception as cand_err:
                logger.warning("Unexpected error compiling candidate '%s': %s", term, cand_err)
                failed_candidates += 1
    finally:
        http_client.close()
        conn.close()

    return mappings, failed_candidates


def _flush_llm_concepts(
    concepts_map: Dict[str, List[str]],
    out_path: str,
    model: str,
    tier2_map: Optional[Dict[str, List[str]]] = None,
    rejected_map: Optional[Dict[str, str]] = None,
    stale_map: Optional[Dict[str, Any]] = None,
    unverified_map: Optional[Dict[str, str]] = None,
) -> None:
    """Atomically write compiled LLM concepts, Tier 2 concepts, rejected terms, and pending stale entries to JSON.

    Args:
        concepts_map: Mapping of Japanese term to list of verified catalog targets (Tier 1).
        out_path: Destination JSON file path.
        model: Identifier of the LLM model used.
        tier2_map: Optional mapping of Japanese term to Tier 2 visual exemplars.
        rejected_map: Optional mapping of rejected terms to explanation strings.
        stale_map: Optional mapping of unevaluated concepts with stale catalog targets.
        unverified_map: Optional mapping of visualizable target misses for diagnostics.
    """
    rej = rejected_map or {}
    unver = unverified_map or {}
    stale = stale_map or {}
    t2 = tier2_map or {}
    out_data = {
        "schema_version": 1,
        "source": "llm_concept",
        "model": model,
        "generated_at": time.strftime("%Y-%m-%d"),
        "concepts": {k: concepts_map[k] for k in sorted(concepts_map)},
        "tier2_concepts": {k: t2[k] for k in sorted(t2)},
        "rejected": {k: rej[k] for k in sorted(rej)},
    }
    if unver:
        out_data["unverified"] = {k: unver[k] for k in sorted(unver)}
    if stale:
        out_data["stale"] = {k: stale[k] for k in sorted(stale)}
    abs_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    tmp_path = f"{abs_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, abs_path)


def run_ollama_compile(
    catalog_path: str,
    jpdb_path: Optional[str] = None,
    limit: int = 2000,
    mode: str = "non-tier1",
    yomitan_url: str = "http://127.0.0.1:19633",
    ollama_url: str = "http://localhost:11434/api/chat",
    ollama_model: str = "gemma4:26b",
    embed_model: str = "qwen3-embedding:4b",
    llm_out_path: str = DEFAULT_LLM_CONCEPTS_PATH,
    cache_dir: str = ".catalog_cache",
    terms: Optional[Union[str, Sequence[str]]] = None,
    clear_yomitan_cache: bool = False,
    recheck_rejected: bool = False,
    no_embed: bool = False,
    build_vector_index: bool = False,
) -> bool:
    """Run the complete build-time JPDB + Yomitan + Ollama visual concept compilation pipeline.

    Args:
        catalog_path: Path to catalog.sqlite.
        jpdb_path: Optional path to JPDB v2.2 frequency CSV (downloads from GitHub if None).
        limit: Max JPDB frequency entries to inspect.
        mode: Filtering mode ('non-tier1', 'unmatched', or 'all').
        yomitan_url: Yomitan API base URL.
        ollama_url: Ollama chat endpoint URL.
        ollama_model: Ollama model name (default: gemma4:26b).
        embed_model: Ollama embedding model tag (default: qwen3-embedding:4b).
        llm_out_path: Output JSON path to save compiled concepts.
        cache_dir: Directory for cached files.
        terms: Optional comma-separated string or sequence of explicit terms to compile.
        clear_yomitan_cache: If True, discards existing cached Yomitan lookups.
        recheck_rejected: If True, re-evaluates terms previously marked as rejected.
        no_embed: If True, disables vector pre-search and runs pure lexical pre-search.
        build_vector_index: If True, computes catalog vector index from scratch if missing.

    Returns:
        True if concepts were compiled and saved successfully, False otherwise.
    """
    store = CatalogStore(catalog_path)
    index = store.get_index()
    if index is None:
        logger.error("Catalog index unavailable at %s; aborting Ollama compilation", catalog_path)
        return False

    is_available, err_msg = check_ollama_availability(
        ollama_url=ollama_url,
        model=ollama_model,
    )
    if not is_available:
        logger.error("Ollama service unavailable: %s; aborting compilation", err_msg)
        return False

    if terms:
        if isinstance(terms, str):
            raw_list = [t.strip() for t in terms.split(",") if t.strip()]
        else:
            raw_list = [str(t).strip() for t in terms if str(t).strip()]
        candidates: List[Dict[str, Any]] = []
        for idx, term in enumerate(raw_list):
            entries, fetched = _fetch_yomitan_term_entries(term, yomitan_url=yomitan_url)
            if fetched == "connection_error" or fetched is False:
                logger.error("Yomitan API connection error for explicit term '%s'; aborting compilation", term)
                return False
            if fetched == "term_error":
                logger.warning(
                    "Yomitan returned term-level error for explicit term '%s'; continuing without Yomitan metadata",
                    term,
                )
                entries = []
            meta = _parse_yomitan_metadata(entries, term=term) if entries else None
            reading = ""
            if entries:
                for entry in entries:
                    if isinstance(entry, dict):
                        hws = entry.get("headwords")
                        if not isinstance(hws, list):
                            continue
                        for hw in hws:
                            if isinstance(hw, dict) and hw.get("term") == term and hw.get("reading"):
                                reading = str(hw["reading"])
                                break
                    if reading:
                        break
            candidates.append({
                "term": term,
                "reading": reading,
                "frequency": idx + 1,
                "pos_tags": sorted(meta.pos_tags) if meta else [],
                "english_glosses": meta.english_glosses if meta else [],
                "is_uk": meta.is_uk if meta else False,
            })
        logger.info("Loaded %d explicit terms for Ollama compilation", len(candidates))
    else:
        if not jpdb_path:
            try:
                jpdb_path = fetch_jpdb_freq_github(cache_dir=cache_dir)
            except Exception as exc:
                logger.error("Failed to fetch JPDB frequency list: %s", exc)
                return False

        candidates, deferred_count = load_jpdb_candidates(
            catalog_path=catalog_path,
            jpdb_path=jpdb_path,
            limit=limit,
            mode=mode,
            yomitan_url=yomitan_url,
            cache_dir=cache_dir,
            clear_yomitan_cache=clear_yomitan_cache,
        )
        logger.info("Loaded %d %s candidates from JPDB v2.2 for compilation", len(candidates), mode)
        if deferred_count > 0:
            logger.error(
                "%d JPDB candidates were deferred because the Yomitan API was unavailable; aborting compilation",
                deferred_count,
            )
            return False

    existing_concepts: Dict[str, List[str]] = {}
    existing_tier2_concepts: Dict[str, List[str]] = {}
    existing_rejected: Dict[str, str] = {}
    existing_unverified: Dict[str, str] = {}
    stale_raw_concepts: Dict[str, Any] = {}
    migrated_unverified = False
    has_dropped_targets = False
    if os.path.isfile(llm_out_path):
        try:
            with open(llm_out_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            if not isinstance(prev_data, dict):
                logger.error("Root JSON value in %s is not an object; refusing to overwrite", llm_out_path)
                return False
            raw_concepts = prev_data.get("concepts") or prev_data.get("mappings") or {}
            raw_tier2 = prev_data.get("tier2_concepts") or {}
            if not isinstance(raw_concepts, dict):
                logger.error("Invalid 'concepts' structure in %s; refusing to overwrite", llm_out_path)
                return False
            if not isinstance(raw_tier2, dict):
                logger.error("Invalid 'tier2_concepts' structure in %s; refusing to overwrite", llm_out_path)
                return False
            raw_stale = prev_data.get("stale", {})
            if isinstance(raw_stale, dict):
                stale_raw_concepts.update({str(k): v for k, v in raw_stale.items()})

            def _validate_concept_dict(src_dict: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Set[str]]:
                """Validate saved concept targets against the catalog index.

                Args:
                    src_dict: Mapping of term to saved target list.

                Returns:
                    Tuple of (valid term-to-targets map, set of stale terms).
                """
                valid_map: Dict[str, List[str]] = {}
                stale_set: Set[str] = set()
                for c_term, c_targets in src_dict.items():
                    if not isinstance(c_targets, list) or not c_targets:
                        stale_set.add(c_term)
                        continue
                    valid_c_targets: List[str] = []
                    for ct in c_targets:
                        clean_ct = clean_search_query(str(ct))
                        if not clean_ct or clean_search_query(clean_ct) != clean_ct:
                            continue
                        if _is_disallowed_text_callout_target(clean_ct, c_term):
                            continue
                        res = filter_relevant_illustrations(index.search(clean_ct), clean_ct)
                        if res and has_title_illustration_match(res, clean_ct):
                            if clean_ct not in valid_c_targets:
                                valid_c_targets.append(clean_ct)
                    if valid_c_targets:
                        valid_map[c_term] = valid_c_targets
                    else:
                        stale_set.add(c_term)
                return valid_map, stale_set

            existing_concepts, stale1 = _validate_concept_dict(raw_concepts)
            existing_tier2_concepts, stale2 = _validate_concept_dict(raw_tier2)
            has_dropped_targets = bool(
                existing_concepts != raw_concepts or existing_tier2_concepts != raw_tier2
            )

            all_stale = stale1 | stale2
            if all_stale:
                stale_raw_concepts.update({
                    t: raw_concepts.get(t, raw_tier2.get(t)) for t in all_stale
                })
                logger.warning(
                    "Re-queuing %d saved concepts with stale/invalid catalog targets for re-evaluation: %s",
                    len(all_stale),
                    sorted(all_stale),
                )
            raw_unverified = prev_data.get("unverified", {})
            if isinstance(raw_unverified, dict):
                existing_unverified = {
                    k: str(v) for k, v in raw_unverified.items()
                    if k not in existing_concepts and k not in existing_tier2_concepts
                }
            raw_rej = prev_data.get("rejected", {})
            if isinstance(raw_rej, dict):
                for k, v in raw_rej.items():
                    if k in existing_concepts or k in existing_tier2_concepts:
                        continue
                    if not isinstance(v, str):
                        continue
                    clean_v = v.strip()
                    if not clean_v:
                        continue
                    if clean_v.startswith(UNVERIFIED_REASON_PREFIX):
                        migrated_unverified = True
                        if k not in existing_unverified:
                            existing_unverified[k] = clean_v
                    else:
                        existing_rejected[k] = clean_v
            elif isinstance(raw_rej, list):
                existing_rejected = {
                    k: "rejected" for k in raw_rej
                    if k not in existing_concepts and k not in existing_tier2_concepts
                }
        except Exception as exc:
            logger.error(
                "Could not read previous LLM concepts from %s: %s; refusing to overwrite",
                llm_out_path,
                exc,
            )
            return False

    def _pending_stale() -> Dict[str, Any]:
        """Return unevaluated stale concepts outside the active 'concepts' key."""
        return {
            t: v for t, v in stale_raw_concepts.items()
            if t not in existing_concepts and t not in existing_tier2_concepts and t not in existing_rejected and t not in existing_unverified
        }

    # Initialize Hybrid Vector Index if enabled
    embed_client: Optional[OllamaEmbedClient] = None
    vector_index: Optional[CatalogVectorIndex] = None
    if not no_embed:
        parsed_base = urllib.parse.urlparse(ollama_url)
        path = parsed_base.path.rstrip("/")
        if path.endswith("/api/chat"):
            path = path[:-len("/api/chat")]
        elif path.endswith("/api"):
            path = path[:-len("/api")]
        base_url = f"{parsed_base.scheme}://{parsed_base.netloc}{path}" if parsed_base.netloc else "http://localhost:11434"
        embed_client = OllamaEmbedClient(base_url=base_url)
        try:
            vector_index = CatalogVectorIndex.build_or_load(
                catalog_path=catalog_path,
                client=embed_client,
                model=embed_model,
                cache_dir=cache_dir,
                build_if_missing=build_vector_index,
            )
        except Exception as vec_exc:
            logger.warning("Could not initialize vector index (%s); using pure lexical pre-search", vec_exc)
            vector_index = None

    propagated_covered_terms: Set[str] = set()
    if jpdb_path and os.path.isfile(jpdb_path) and not terms:
        curated_terms = load_curated_terms(catalog_path)
        temp_c = dict(existing_concepts)
        temp_t2 = dict(existing_tier2_concepts)
        temp_rej = dict(existing_rejected)
        propagate_kana_uk_concepts(
            concepts=temp_c,
            tier2_concepts=temp_t2,
            rejected=temp_rej,
            jpdb_path=jpdb_path,
            unverified=existing_unverified,
            curated_terms=curated_terms,
        )
        propagated_covered_terms.update(temp_c.keys())
        propagated_covered_terms.update(temp_t2.keys())
        if not recheck_rejected:
            propagated_covered_terms.update(temp_rej.keys())

    has_artifact_updates = bool(stale_raw_concepts or migrated_unverified or has_dropped_targets)
    if not candidates:
        logger.warning("No candidates found to compile.")
        if has_artifact_updates:
            try:
                _flush_llm_concepts(
                    existing_concepts,
                    llm_out_path,
                    ollama_model,
                    tier2_map=existing_tier2_concepts,
                    rejected_map=existing_rejected,
                    stale_map=_pending_stale(),
                    unverified_map=existing_unverified,
                )
            except Exception as exc:
                logger.error("Failed to persist revalidated concepts on empty candidates exit to %s: %s", llm_out_path, exc)
                return False
        if not os.path.isfile(llm_out_path):
            logger.error("No candidates compiled and no output artifact exists at %s; aborting", llm_out_path)
            return False
        return True

    remaining_candidates: List[Dict[str, Any]] = []
    seen_terms: Set[str] = set()
    force_explicit = bool(terms)
    for c in candidates:
        c_term = c["term"]
        if c_term in seen_terms:
            continue
        if not force_explicit:
            if (
                c_term in existing_concepts
                or c_term in existing_tier2_concepts
                or c_term in propagated_covered_terms
            ):
                continue
            if not recheck_rejected and (
                c_term in existing_rejected
                or c_term in existing_unverified
            ):
                continue
        seen_terms.add(c_term)
        remaining_candidates.append(c)

    logger.info(
        "Candidate evaluation pool: %d candidates (%d Tier 1 concepts, %d Tier 2 concepts, %d rejected, %d unverified in %s, %d remaining)",
        len(candidates),
        len(existing_concepts),
        len(existing_tier2_concepts),
        len(existing_rejected),
        len(existing_unverified),
        llm_out_path,
        len(remaining_candidates),
    )
    if not remaining_candidates:
        logger.info("All %d candidates already evaluated. Nothing to do.", len(candidates))
        if has_artifact_updates:
            try:
                _flush_llm_concepts(
                    existing_concepts,
                    llm_out_path,
                    ollama_model,
                    tier2_map=existing_tier2_concepts,
                    rejected_map=existing_rejected,
                    stale_map=_pending_stale(),
                    unverified_map=existing_unverified,
                )
            except Exception as exc:
                logger.error("Failed to persist revalidated concepts on early exit to %s: %s", llm_out_path, exc)
                return False
        return True

    evaluated_count = 0

    def _handle_match(term: str, targets: List[str], tier: int = 1) -> None:
        """Persist newly compiled concept mapping immediately to disk."""
        nonlocal evaluated_count
        evaluated_count += 1
        if tier == 1:
            existing_concepts[term] = targets
            existing_tier2_concepts.pop(term, None)
        else:
            existing_tier2_concepts[term] = targets
            existing_concepts.pop(term, None)
        existing_rejected.pop(term, None)
        existing_unverified.pop(term, None)
        stale_raw_concepts.pop(term, None)
        try:
            _flush_llm_concepts(
                existing_concepts,
                llm_out_path,
                ollama_model,
                tier2_map=existing_tier2_concepts,
                rejected_map=existing_rejected,
                stale_map=_pending_stale(),
                unverified_map=existing_unverified,
            )
        except Exception as exc:
            logger.warning("Interim write failed for '%s': %s", term, exc)
        logger.info(
            "Saved concept '%s' (Tier %d) -> %s to %s (total: %d T1, %d T2)",
            term,
            tier,
            targets,
            llm_out_path,
            len(existing_concepts),
            len(existing_tier2_concepts),
        )

    def _handle_reject(term: str, reason: str, tier: int = 2) -> None:
        """Persist rejected term immediately to disk."""
        nonlocal evaluated_count
        if not force_explicit and (term in existing_concepts or term in existing_tier2_concepts):
            return
        evaluated_count += 1
        existing_concepts.pop(term, None)
        existing_tier2_concepts.pop(term, None)
        existing_unverified.pop(term, None)
        stale_raw_concepts.pop(term, None)
        existing_rejected[term] = reason
        try:
            _flush_llm_concepts(
                existing_concepts,
                llm_out_path,
                ollama_model,
                tier2_map=existing_tier2_concepts,
                rejected_map=existing_rejected,
                stale_map=_pending_stale(),
                unverified_map=existing_unverified,
            )
        except Exception as exc:
            logger.warning("Interim write failed for rejection '%s': %s", term, exc)

    def _handle_unverified(term: str, reason: str) -> None:
        """Persist unverified target miss immediately to disk for diagnostics."""
        nonlocal evaluated_count
        if not force_explicit and (term in existing_concepts or term in existing_tier2_concepts):
            return
        evaluated_count += 1
        existing_concepts.pop(term, None)
        existing_tier2_concepts.pop(term, None)
        existing_rejected.pop(term, None)
        stale_raw_concepts.pop(term, None)
        existing_unverified[term] = reason
        try:
            _flush_llm_concepts(
                existing_concepts,
                llm_out_path,
                ollama_model,
                tier2_map=existing_tier2_concepts,
                rejected_map=existing_rejected,
                stale_map=_pending_stale(),
                unverified_map=existing_unverified,
            )
        except Exception as exc:
            logger.warning("Interim write failed for unverified '%s': %s", term, exc)

    compiled, failed_count = run_ollama_agent_batch(
        candidates=remaining_candidates,
        catalog_path=catalog_path,
        ollama_url=ollama_url,
        model=ollama_model,
        vector_index=vector_index,
        embed_client=embed_client,
        embed_model=embed_model,
        on_match=_handle_match,
        on_reject=_handle_reject,
        on_unverified=_handle_unverified,
        store=store,
    )
    if failed_count > 0:
        logger.error(
            "Ollama compilation incomplete: %d candidates encountered request or network failures",
            failed_count,
        )
        return False

    if not compiled and evaluated_count == 0:
        logger.error(
            "Ollama compilation produced 0 new mappings or rejections for %d candidates; keeping previous %s",
            len(remaining_candidates),
            llm_out_path,
        )
        return False

    try:
        _flush_llm_concepts(
            existing_concepts,
            llm_out_path,
            ollama_model,
            tier2_map=existing_tier2_concepts,
            rejected_map=existing_rejected,
            stale_map=_pending_stale(),
            unverified_map=existing_unverified,
        )
    except Exception as exc:
        logger.error("Failed to write final compiled concepts artifact to %s: %s", llm_out_path, exc)
        return False

    logger.info(
        "Successfully evaluated candidates (%d new matches, %d T1, %d T2 concepts, %d rejected, %d unverified in %s)",
        len(compiled),
        len(existing_concepts),
        len(existing_tier2_concepts),
        len(existing_rejected),
        len(existing_unverified),
        llm_out_path,
    )
    return True
