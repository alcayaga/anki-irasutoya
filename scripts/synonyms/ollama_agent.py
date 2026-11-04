"""Ollama LLM agent loop, search_catalog tool execution, and concept compilation."""
import json
import logging
import os
import sqlite3
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from addon.catalog import CatalogStore
from addon.ranking import has_title_illustration_match
from scripts.synonyms.common import (
    DEFAULT_LLM_CONCEPTS_PATH,
    _readonly_catalog_uri,
    filter_relevant_illustrations,
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
    load_jpdb_candidates,
)

logger = logging.getLogger("synonyms.ollama_agent")

__all__ = [
    "JAPANESE_CHAR_RE",
    "OLLAMA_SYSTEM_PROMPT",
    "SEARCH_CATALOG_TOOL",
    "VALID_JAPANESE_LATIN_ACRONYMS",
    "_contains_english_word",
    "_flush_llm_concepts",
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


def run_ollama_agent_batch(
    candidates: Sequence[Dict[str, Any]],
    catalog_path: str,
    ollama_url: str = "http://localhost:11434/api/chat",
    model: str = "gemma4:31b",
    max_turns: int = 4,
    timeout: float = 120.0,
    store: Optional[CatalogStore] = None,
    on_match: Optional[Callable[[str, List[str]], None]] = None,
    on_reject: Optional[Callable[[str, str], None]] = None,
) -> Tuple[Dict[str, List[str]], int]:
    """Run interactive agentic search loop via Ollama /api/chat with search_catalog tool.

    Args:
        candidates: Sequence of candidate dicts with term, reading, frequency, glosses, pos_tags.
        catalog_path: Path to catalog.sqlite.
        ollama_url: Ollama chat endpoint URL.
        model: Ollama model identifier.
        max_turns: Maximum conversation turns per item.
        timeout: HTTP request timeout in seconds.
        store: Optional pre-loaded CatalogStore.
        on_match: Optional callback invoked immediately with (term, targets) upon successful compilation.
        on_reject: Optional callback invoked with (term, reason) when a candidate is rejected or unverified.

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

    try:
        total = len(candidates)
        for idx, cand in enumerate(candidates, 1):
            term = cand["term"]
            reading = cand.get("reading", "")
            freq = cand.get("frequency", "")
            glosses = ", ".join(cand.get("english_glosses", [])[:5]) or "None"
            pos_tags = ", ".join(sorted(cand.get("pos_tags", []))) or "None"

            logger.info("[%d/%d] Compiling '%s' (rank %s)...", idx, total, term, freq)
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Japanese term: {term}\n"
                        f"Reading: {reading}\n"
                        f"Frequency Rank: {freq}\n"
                        f"English Glosses: {glosses}\n"
                        f"POS: {pos_tags}\n\n"
                        "Search the catalog using `search_catalog` and determine if this concept is visualizable in Irasutoya, "
                        "and if so, select 1-2 optimal catalog targets."
                    ),
                },
            ]

            try:
                for _turn in range(max_turns):
                    payload: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "tools": [SEARCH_CATALOG_TOOL],
                        "stream": False,
                        "think": False,
                        "options": {"temperature": 0.0},
                    }
                    req = urllib.request.Request(
                        ollama_url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json", "User-Agent": "anki-irasutoya/1.0"},
                    )
                    try:
                        with urllib.request.urlopen(req, timeout=timeout) as resp:
                            resp_data = json.loads(resp.read().decode("utf-8"))
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
                            tool_results = []
                            for q in queries:
                                tool_results.append(
                                    search_catalog_for_agent(
                                        catalog_path=catalog_path,
                                        query=q,
                                        limit=5,
                                        index=index,
                                        conn=conn,
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
                        if parsed and parsed.get("visualizable") is True:
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
                                clean_t = str(t).strip()
                                if not clean_t or not JAPANESE_CHAR_RE.search(clean_t) or _contains_english_word(clean_t):
                                    continue
                                results = filter_relevant_illustrations(index.search(clean_t), clean_t)
                                if has_title_illustration_match(results, clean_t):
                                    valid_targets.append(clean_t)
                            if valid_targets:
                                mappings[term] = valid_targets
                                logger.info("Matched '%s' -> %s", term, valid_targets)
                                if on_match:
                                    on_match(term, valid_targets)
                                matched = True
                        if not matched and not is_malformed:
                            if not parsed:
                                logger.warning(
                                    "Unparsable model response for '%s'; skipping rejection to allow retry on next run",
                                    term,
                                )
                            else:
                                if parsed.get("visualizable") is True:
                                    reason = f"unverified in catalog: proposed targets {parsed.get('targets', [])!r}"
                                else:
                                    reason = str(parsed.get("reason", "non-visualizable"))
                                logger.info("Rejected '%s' (%s)", term, reason)
                                if on_reject:
                                    on_reject(term, reason)
                        break
                else:
                    logger.warning(
                        "Exhausted %d turns for '%s' without a final answer; leaving it for the next run",
                        max_turns,
                        term,
                    )
            except Exception as cand_err:
                logger.warning("Unexpected error compiling candidate '%s': %s", term, cand_err)
                failed_candidates += 1
    finally:
        conn.close()

    return mappings, failed_candidates


def _flush_llm_concepts(
    concepts_map: Dict[str, List[str]],
    out_path: str,
    model: str,
    rejected_map: Optional[Dict[str, str]] = None,
    stale_map: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically write compiled LLM concepts, rejected terms, and pending stale entries to JSON.

    Args:
        concepts_map: Mapping of Japanese term to list of verified catalog targets.
        out_path: Destination JSON file path.
        model: Identifier of the LLM model used.
        rejected_map: Optional mapping of rejected terms to explanation strings.
        stale_map: Optional mapping of unevaluated concepts with stale catalog targets.
    """
    rej = rejected_map or {}
    stale = stale_map or {}
    out_data = {
        "schema_version": 1,
        "source": "llm_concept",
        "model": model,
        "generated_at": time.strftime("%Y-%m-%d"),
        "concepts": {k: concepts_map[k] for k in sorted(concepts_map)},
        "rejected": {k: rej[k] for k in sorted(rej)},
    }
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
    ollama_model: str = "gemma4:31b",
    llm_out_path: str = DEFAULT_LLM_CONCEPTS_PATH,
    cache_dir: str = ".catalog_cache",
    terms: Optional[Union[str, Sequence[str]]] = None,
    clear_yomitan_cache: bool = False,
) -> bool:
    """Run the complete build-time JPDB + Yomitan + Ollama visual concept compilation pipeline.

    Args:
        catalog_path: Path to catalog.sqlite.
        jpdb_path: Optional path to JPDB v2.2 frequency CSV (downloads from GitHub if None).
        limit: Max JPDB frequency entries to inspect.
        mode: Filtering mode ('non-tier1', 'unmatched', or 'all').
        yomitan_url: Yomitan API base URL.
        ollama_url: Ollama chat endpoint URL.
        ollama_model: Ollama model name.
        llm_out_path: Output JSON path to save compiled concepts.
        cache_dir: Directory for cached files.
        terms: Optional comma-separated string or sequence of explicit terms to compile.
        clear_yomitan_cache: If True, discards existing cached Yomitan lookups.

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
    existing_rejected: Dict[str, str] = {}
    stale_raw_concepts: Dict[str, Any] = {}
    if os.path.isfile(llm_out_path):
        try:
            with open(llm_out_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            if not isinstance(prev_data, dict):
                logger.error("Root JSON value in %s is not an object; refusing to overwrite", llm_out_path)
                return False
            raw_concepts = prev_data.get("concepts") or prev_data.get("mappings") or {}
            if not isinstance(raw_concepts, dict):
                logger.error("Invalid 'concepts' structure in %s; refusing to overwrite", llm_out_path)
                return False
            raw_stale = prev_data.get("stale", {})
            if isinstance(raw_stale, dict):
                stale_raw_concepts.update({str(k): v for k, v in raw_stale.items()})
            valid_existing_concepts: Dict[str, List[str]] = {}
            stale_concept_terms: Set[str] = set()
            for c_term, c_targets in raw_concepts.items():
                if not isinstance(c_targets, list) or not c_targets:
                    stale_concept_terms.add(c_term)
                    continue
                valid_c_targets: List[str] = []
                for ct in c_targets:
                    clean_ct = str(ct).strip()
                    if not clean_ct:
                        continue
                    res = filter_relevant_illustrations(index.search(clean_ct), clean_ct)
                    if res and has_title_illustration_match(res, clean_ct):
                        valid_c_targets.append(clean_ct)
                if valid_c_targets:
                    valid_existing_concepts[c_term] = valid_c_targets
                else:
                    stale_concept_terms.add(c_term)

            existing_concepts = valid_existing_concepts
            if stale_concept_terms:
                stale_raw_concepts.update({
                    t: raw_concepts[t] for t in stale_concept_terms if t in raw_concepts
                })
                logger.warning(
                    "Re-queuing %d saved concepts with stale/invalid catalog targets for re-evaluation: %s",
                    len(stale_concept_terms),
                    sorted(stale_concept_terms),
                )
            raw_rej = prev_data.get("rejected", {})
            if isinstance(raw_rej, dict):
                existing_rejected = {k: str(v) for k, v in raw_rej.items() if k not in existing_concepts}
            elif isinstance(raw_rej, list):
                existing_rejected = {k: "rejected" for k in raw_rej if k not in existing_concepts}
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
            if t not in existing_concepts and t not in existing_rejected
        }

    if not candidates:
        logger.warning("No candidates found to compile.")
        if stale_raw_concepts:
            try:
                _flush_llm_concepts(
                    existing_concepts,
                    llm_out_path,
                    ollama_model,
                    rejected_map=existing_rejected,
                    stale_map=_pending_stale(),
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
        if not force_explicit and (c_term in existing_concepts or c_term in existing_rejected):
            continue
        seen_terms.add(c_term)
        remaining_candidates.append(c)

    logger.info(
        "Candidate evaluation pool: %d candidates (%d concepts, %d rejected in %s, %d remaining)",
        len(candidates),
        len(existing_concepts),
        len(existing_rejected),
        llm_out_path,
        len(remaining_candidates),
    )
    if not remaining_candidates:
        logger.info("All %d candidates already evaluated. Nothing to do.", len(candidates))
        if stale_raw_concepts:
            try:
                _flush_llm_concepts(
                    existing_concepts,
                    llm_out_path,
                    ollama_model,
                    rejected_map=existing_rejected,
                    stale_map=_pending_stale(),
                )
            except Exception as exc:
                logger.error("Failed to persist revalidated concepts on early exit to %s: %s", llm_out_path, exc)
                return False
        return True

    evaluated_count = 0

    def _handle_match(term: str, targets: List[str]) -> None:
        """Persist newly compiled concept mapping immediately to disk."""
        nonlocal evaluated_count
        evaluated_count += 1
        existing_concepts[term] = targets
        existing_rejected.pop(term, None)
        stale_raw_concepts.pop(term, None)
        try:
            _flush_llm_concepts(
                existing_concepts,
                llm_out_path,
                ollama_model,
                rejected_map=existing_rejected,
                stale_map=_pending_stale(),
            )
        except Exception as exc:
            logger.warning("Interim write failed for '%s': %s", term, exc)
        logger.info(
            "Saved concept '%s' -> %s to %s (total: %d concepts)",
            term,
            targets,
            llm_out_path,
            len(existing_concepts),
        )

    def _handle_reject(term: str, reason: str) -> None:
        """Persist rejected term immediately to disk."""
        nonlocal evaluated_count
        if not force_explicit and term in existing_concepts:
            return
        evaluated_count += 1
        existing_concepts.pop(term, None)
        stale_raw_concepts.pop(term, None)
        existing_rejected[term] = reason
        try:
            _flush_llm_concepts(
                existing_concepts,
                llm_out_path,
                ollama_model,
                rejected_map=existing_rejected,
                stale_map=_pending_stale(),
            )
        except Exception as exc:
            logger.warning("Interim write failed for rejection '%s': %s", term, exc)

    compiled, failed_count = run_ollama_agent_batch(
        candidates=remaining_candidates,
        catalog_path=catalog_path,
        ollama_url=ollama_url,
        model=ollama_model,
        on_match=_handle_match,
        on_reject=_handle_reject,
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
            rejected_map=existing_rejected,
            stale_map=_pending_stale(),
        )
    except Exception as exc:
        logger.error("Failed to write final compiled concepts artifact to %s: %s", llm_out_path, exc)
        return False

    logger.info(
        "Successfully evaluated candidates (%d new matches, %d total concepts, %d rejected in %s)",
        len(compiled),
        len(existing_concepts),
        len(existing_rejected),
        llm_out_path,
    )
    return True
