"""JPDB v2.2 frequency list candidate extraction and Yomitan API metadata enrichment."""
import csv
import itertools
import json
import logging
import os
import posixpath
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from addon.catalog import CatalogStore
from addon.ranking import has_title_illustration_match
from addon.text_utils import generate_search_candidates, is_grammar_function_word
from addon.yomitan_parser import VERB_POS_PATTERN, YomitanMetadata, is_grammar_sense
from scripts.synonyms.common import (
    _readonly_catalog_uri,
    filter_relevant_illustrations,
)

logger = logging.getLogger("synonyms.yomitan_jpdb")

JPDB_FREQ_RAW_URL = (
    "https://raw.githubusercontent.com/Kuuuube/yomitan-dictionaries/main/data/jpdb_v2.2_freq_list_2024-10-13.csv"
)


def _is_valid_jpdb_header_or_row(first_line: str) -> bool:
    """Validate whether the first line represents a valid JPDB header or data row.

    Args:
        first_line: First line from the cached JPDB frequency file.

    Returns:
        True if the line starts with a standard 'term' header or has valid delimited columns.
    """
    if first_line.startswith(("term\t", "term,")):
        return True
    delim = "\t" if "\t" in first_line else ("," if "," in first_line else None)
    if not delim:
        return False
    cols = [c.strip() for c in first_line.split(delim)]
    if len(cols) >= 3 and cols[2].isdigit():
        return True
    return False


def fetch_jpdb_freq_github(
    cache_dir: str = ".catalog_cache",
    url: str = JPDB_FREQ_RAW_URL,
) -> str:
    """Download and cache the JPDB v2.2 frequency list TSV/CSV from GitHub.

    Args:
        cache_dir: Directory where the cached CSV is stored.
        url: Raw URL to the JPDB v2.2 frequency CSV file.

    Returns:
        Absolute filesystem path to the downloaded/cached file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    parsed_path = urllib.parse.urlparse(url).path
    filename = posixpath.basename(parsed_path) or "jpdb_v2.2_freq_list_2024-10-13.csv"
    target_path = os.path.join(cache_dir, filename)
    if os.path.isfile(target_path) and os.path.getsize(target_path) > 1000:
        try:
            with open(target_path, "r", encoding="utf-8-sig") as check_f:
                header = check_f.readline()
                if _is_valid_jpdb_header_or_row(header):
                    logger.info("Using cached JPDB frequency list at %s", target_path)
                    return os.path.abspath(target_path)
                logger.warning(
                    "Cached JPDB file at %s has invalid header/row %r; re-downloading...",
                    target_path,
                    header[:50],
                )
        except Exception as exc:
            logger.warning("Failed to validate cached JPDB file at %s: %s; re-downloading...", target_path, exc)

    logger.info("Downloading JPDB v2.2 frequency list from %s...", url)
    tmp_path = f"{target_path}.part"
    req = urllib.request.Request(url, headers={"User-Agent": "anki-irasutoya/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as out_f:
        while chunk := resp.read(65536):
            out_f.write(chunk)
    os.replace(tmp_path, target_path)
    logger.info("Downloaded JPDB frequency list to %s (%d bytes)", target_path, os.path.getsize(target_path))
    return os.path.abspath(target_path)


def _fetch_yomitan_term_entries(
    term: str,
    yomitan_url: str = "http://127.0.0.1:19633",
    timeout: float = 5.0,
) -> Tuple[Optional[List[Dict[str, Any]]], Union[str, bool]]:
    """Fetch raw dictionary entries from local Yomitan HTTP API /termEntries.

    Args:
        term: Vocabulary expression to look up.
        yomitan_url: Base URL of the Yomitan API server.
        timeout: HTTP request timeout in seconds.

    Returns:
        Tuple of (entries_list, status_or_bool).
        status is "ok", "term_error", or "connection_error".
    """
    url = f"{yomitan_url.rstrip('/')}/termEntries"
    payload = json.dumps({"term": term}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "anki-irasutoya/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("Yomitan returned HTTP %s for '%s'", exc.code, term)
        return None, "term_error"
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        logger.debug("Yomitan API connection failed for '%s': %s", term, exc)
        return None, "connection_error"
    except (ValueError, UnicodeDecodeError) as exc:
        logger.warning("Undecodable Yomitan response for '%s': %s", term, exc)
        return None, "term_error"
    except Exception as exc:
        logger.debug("Yomitan API unexpected error for '%s': %s", term, exc)
        return None, "term_error"

    if not isinstance(data, dict):
        logger.debug("Yomitan returned non-object JSON for '%s': %r", term, type(data).__name__)
        return None, "term_error"
    if "dictionaryEntries" not in data:
        logger.debug("Yomitan response missing 'dictionaryEntries' key for '%s'", term)
        return None, "term_error"
    raw_entries = data.get("dictionaryEntries")
    if not isinstance(raw_entries, list):
        logger.debug("Yomitan returned non-list dictionaryEntries for '%s': %r", term, raw_entries)
        return None, "term_error"
    return raw_entries, "ok"


def _parse_yomitan_metadata(
    dictionary_entries: List[Dict[str, Any]],
    term: str,
    reading: Optional[str] = None,
) -> Optional[YomitanMetadata]:
    """Parse structured JMdict metadata from Yomitan dictionary entries.

    Args:
        dictionary_entries: Raw dictionary entries list from Yomitan /termEntries.
        term: Search headword.
        reading: Optional kana reading.

    Returns:
        YomitanMetadata instance if valid entry found, None otherwise.
    """
    if not isinstance(dictionary_entries, list) or not dictionary_entries:
        return None

    def _is_english_jmdict(d: Dict[str, Any]) -> bool:
        """Return True when the definition comes from an English JMdict dictionary."""
        dict_name = str(d.get("dictionary", "")).strip()
        if "JMdict" not in dict_name:
            return False
        if dict_name in ("JMdict", "JMdict (English)", "JMdict [en]", "JMdict (eng)"):
            return True
        non_english = (
            "Spanish", "French", "German", "Russian", "Dutch", "Hungarian",
            "Swedish", "Italian", "Portuguese", "Español", "Français", "Deutsch",
        )
        if any(lang.lower() in dict_name.lower() for lang in non_english):
            return False
        if "english" in dict_name.lower() or dict_name == "JMdict":
            return True
        m = re.search(r"\(([^)]+)\)", dict_name)
        if m and "english" not in m.group(1).lower():
            return False
        return True

    def _collect_strings(node: Any, out: List[str]) -> None:
        """Append each non-empty trimmed string under a node's content to out."""
        if isinstance(node, str):
            s = node.strip()
            if s:
                out.append(s)
        elif isinstance(node, dict):
            if "content" in node:
                _collect_strings(node["content"], out)
        elif isinstance(node, list):
            for item in node:
                _collect_strings(item, out)

    def _extract_glosses(node: Any) -> List[str]:
        """Return the gloss strings found in a Yomitan structured-content node."""
        glosses: List[str] = []
        if isinstance(node, str):
            s = node.strip()
            if s:
                glosses.append(s)
        elif isinstance(node, dict):
            if node.get("type") == "text" and isinstance(node.get("text"), str):
                s = node["text"].strip()
                if s:
                    glosses.append(s)
                return glosses
            data = node.get("data")
            if isinstance(data, dict) and data.get("content") == "glossary":
                _collect_strings(node.get("content"), glosses)
            elif "content" in node:
                glosses.extend(_extract_glosses(node["content"]))
        elif isinstance(node, list):
            for item in node:
                glosses.extend(_extract_glosses(item))
        return glosses

    best_entry: Optional[Dict[str, Any]] = None
    for entry in dictionary_entries:
        if not isinstance(entry, dict):
            continue
        defs = entry.get("definitions", [])
        if not isinstance(defs, list) or not any(_is_english_jmdict(d) for d in defs if isinstance(d, dict)):
            continue
        hws = entry.get("headwords", [])
        if not isinstance(hws, list):
            continue
        for hw in hws:
            if not isinstance(hw, dict):
                continue
            if hw.get("term") == term and (not reading or hw.get("reading") == reading):
                best_entry = entry
                break
        if best_entry:
            break

    if not best_entry:
        for entry in dictionary_entries:
            if not isinstance(entry, dict):
                continue
            defs = entry.get("definitions", [])
            if not isinstance(defs, list) or not any(_is_english_jmdict(d) for d in defs if isinstance(d, dict)):
                continue
            hws = entry.get("headwords", [])
            if not isinstance(hws, list):
                continue
            for hw in hws:
                if not isinstance(hw, dict):
                    continue
                if hw.get("term") == term and (not reading or not hw.get("reading") or hw.get("reading") == reading):
                    best_entry = entry
                    break
            if best_entry:
                break

    if not best_entry:
        target_read = reading or term
        # Prefer entries with 'uk' (usually kana) or where headword term matches reading
        for entry in dictionary_entries:
            if not isinstance(entry, dict):
                continue
            defs = entry.get("definitions", [])
            if not isinstance(defs, list) or not any(_is_english_jmdict(d) for d in defs if isinstance(d, dict)):
                continue
            has_uk = any(
                any(isinstance(t, dict) and t.get("name") == "uk" for t in d.get("tags", []) if isinstance(t, dict))
                for d in defs
                if isinstance(d, dict) and isinstance(d.get("tags"), list)
            )
            hws = entry.get("headwords", [])
            if not isinstance(hws, list):
                continue
            for hw in hws:
                if not isinstance(hw, dict):
                    continue
                if hw.get("reading") == target_read and (has_uk or hw.get("term") == target_read):
                    best_entry = entry
                    break
            if best_entry:
                break

    if not best_entry:
        target_read = reading or term
        reading_matches: List[Dict[str, Any]] = []
        matched_terms: Set[str] = set()
        for entry in dictionary_entries:
            if not isinstance(entry, dict):
                continue
            defs = entry.get("definitions", [])
            if not isinstance(defs, list) or not any(_is_english_jmdict(d) for d in defs if isinstance(d, dict)):
                continue
            hws = entry.get("headwords", [])
            if not isinstance(hws, list):
                continue
            for hw in hws:
                if not isinstance(hw, dict):
                    continue
                if hw.get("reading") == target_read:
                    reading_matches.append(entry)
                    matched_terms.add(str(hw.get("term", "")))
                    break

        # Only accept an unambiguous reading match (single distinct headword term)
        if reading_matches and len(matched_terms) == 1:
            best_entry = reading_matches[0]

    if not best_entry:
        return None

    pos_tags: Set[str] = set()
    is_uk = False
    english_glosses: List[str] = []
    verb_class: Optional[str] = None
    per_definition_tags: List[Set[str]] = []

    definitions = best_entry.get("definitions", [])
    if not isinstance(definitions, list):
        return None

    for d in definitions:
        if not isinstance(d, dict) or not _is_english_jmdict(d):
            continue
        def_pos: Set[str] = set()
        tags = d.get("tags", [])
        if isinstance(tags, list):
            for t in tags:
                if not isinstance(t, dict):
                    continue
                tag_name = t.get("name", "")
                if t.get("category") == "partOfSpeech":
                    pos_tags.add(tag_name)
                    def_pos.add(tag_name)
                    if not verb_class and VERB_POS_PATTERN.match(tag_name):
                        verb_class = tag_name
                elif tag_name == "uk":
                    is_uk = True
        if def_pos:
            per_definition_tags.append(def_pos)
        entries_list = d.get("entries", [])
        if isinstance(entries_list, list):
            english_glosses.extend(_extract_glosses(entries_list))

    is_particle_or_aux = bool(per_definition_tags) and all(
        is_grammar_sense(tags) for tags in per_definition_tags
    )
    return YomitanMetadata(
        pos_tags=pos_tags,
        is_uk=is_uk,
        is_particle_or_aux=is_particle_or_aux,
        verb_class=verb_class,
        english_glosses=english_glosses,
    )


def query_yomitan_term_metadata(
    term: str,
    reading: Optional[str] = None,
    yomitan_url: str = "http://127.0.0.1:19633",
    timeout: float = 5.0,
) -> Optional[YomitanMetadata]:
    """Query local Yomitan HTTP API /termEntries for structured JMdict POS and glosses.

    Args:
        term: Vocabulary expression to look up.
        reading: Optional kana reading of the term.
        yomitan_url: Base URL of the running Yomitan API server.
        timeout: HTTP request timeout in seconds.

    Returns:
        YomitanMetadata instance if successfully resolved, or None if unavailable/error.
    """
    entries, ok = _fetch_yomitan_term_entries(term, yomitan_url=yomitan_url, timeout=timeout)
    if ok != "ok" and ok is not True:
        return None
    if not entries:
        return None
    return _parse_yomitan_metadata(entries, term=term, reading=reading)


def _save_yomitan_cache(cache: Dict[str, Any], path: str) -> None:
    """Atomically write the Yomitan lookup cache to path using a temp file.

    Args:
        cache: Dictionary of cached Yomitan term lookups.
        path: Destination JSON file path.
    """
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    tmp_path = f"{abs_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as cache_f:
        json.dump(cache, cache_f, ensure_ascii=False, indent=2)
        cache_f.write("\n")
    os.replace(tmp_path, abs_path)


def load_jpdb_candidates(
    catalog_path: str,
    jpdb_path: str,
    limit: int = 2000,
    mode: str = "non-tier1",
    yomitan_url: str = "http://127.0.0.1:19633",
    cache_dir: str = ".catalog_cache",
    clear_yomitan_cache: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    """Load, filter, and enrich candidates from JPDB v2.2 frequency list.

    Args:
        catalog_path: Path to catalog.sqlite to check existing matches.
        jpdb_path: Path to local JPDB v2.2 frequency CSV/TSV.
        limit: Number of top frequency entries to evaluate.
        mode: Filtering mode: 'non-tier1' (default), 'unmatched', or 'all'.
        yomitan_url: Local Yomitan HTTP API URL.
        cache_dir: Directory to cache Yomitan API responses.
        clear_yomitan_cache: If True, discards existing cached Yomitan lookups.

    Returns:
        Tuple of (candidates_list, deferred_count).
    """
    store = CatalogStore(catalog_path)
    index = store.get_index()
    if index is None:
        logger.error("Catalog index unavailable for JPDB candidate loading at %s", catalog_path)
        return [], 0

    os.makedirs(cache_dir, exist_ok=True)
    yomitan_cache_path = os.path.join(cache_dir, "yomitan_api_cache.json")
    yomitan_cache: Dict[str, Any] = {}
    if not clear_yomitan_cache and os.path.isfile(yomitan_cache_path):
        try:
            with open(yomitan_cache_path, "r", encoding="utf-8") as f:
                loaded_cache = json.load(f)
            if isinstance(loaded_cache, dict):
                yomitan_cache = loaded_cache
            else:
                logger.warning("Ignoring Yomitan cache at %s: root is not a JSON object.", yomitan_cache_path)
        except Exception as exc:
            logger.warning("Could not read Yomitan cache from %s: %s", yomitan_cache_path, exc)

    existing_manual: Set[str] = set()
    existing_kaishi: Set[str] = set()
    try:
        conn = sqlite3.connect(_readonly_catalog_uri(catalog_path), uri=True)
        try:
            cur = conn.cursor()
            cur.execute("SELECT term FROM synonyms WHERE source = 'manual'")
            existing_manual = {r[0] for r in cur.fetchall()}
            cur.execute("SELECT term FROM synonyms WHERE source = 'donkuri/kaishi'")
            existing_kaishi = {r[0] for r in cur.fetchall()}
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("Failed to read existing catalog synonyms: %s", exc)

    candidates: List[Dict[str, Any]] = []
    deferred_count = 0
    cache_dirty = False
    consecutive_yomitan_failures = 0
    max_consecutive_yomitan_failures = 5

    with open(jpdb_path, "r", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t"
        if sample:
            first_line = sample.splitlines()[0] if sample.splitlines() else ""
            if "\t" in first_line:
                delimiter = "\t"
            elif "," in first_line:
                delimiter = ","
        reader: Any = csv.reader(f, delimiter=delimiter)
        first_row = next(reader, None)
        if first_row is not None and first_row[:1] != ["term"]:
            reader = itertools.chain([first_row], reader)
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            if idx > 0 and idx % 250 == 0:
                logger.info("Inspecting JPDB candidate %d/%d (%d candidates queued)...", idx, limit, len(candidates))
                if cache_dirty:
                    try:
                        _save_yomitan_cache(yomitan_cache, yomitan_cache_path)
                        cache_dirty = False
                    except Exception as exc:
                        logger.warning("Failed to save Yomitan cache checkpoint: %s", exc)
            if not row or not row[0].strip():
                continue
            term = row[0].strip()
            reading = row[1].strip() if len(row) > 1 else term
            try:
                freq = int(row[2].strip()) if len(row) > 2 and row[2].strip() else idx + 1
            except ValueError:
                freq = idx + 1

            if is_grammar_function_word(term):
                continue

            if term in existing_manual or term in existing_kaishi:
                continue

            yomitan_meta: Optional[YomitanMetadata] = None
            cache_key = f"{yomitan_url}\t{term}\t{reading}"
            cached_val = yomitan_cache.get(cache_key)
            pos_tags_val = cached_val.get("pos_tags") if isinstance(cached_val, dict) else None
            is_valid_cache_entry = (
                isinstance(cached_val, dict)
                and isinstance(pos_tags_val, list)
                and all(isinstance(tag, str) for tag in pos_tags_val)
                and isinstance(cached_val.get("is_particle_or_aux"), bool)
            )
            if cache_key in yomitan_cache and cached_val is None:
                yomitan_meta = None
            elif is_valid_cache_entry:
                assert isinstance(cached_val, dict)
                assert isinstance(pos_tags_val, list)
                pos_tags_raw = pos_tags_val
                glosses_raw = cached_val.get("english_glosses")
                gloss_list = [str(g) for g in glosses_raw if isinstance(g, str)] if isinstance(glosses_raw, (list, tuple)) else []
                yomitan_meta = YomitanMetadata(
                    pos_tags=set(pos_tags_raw),
                    is_uk=bool(cached_val.get("is_uk", False)),
                    is_particle_or_aux=cached_val["is_particle_or_aux"],
                    verb_class=cached_val.get("verb_class"),
                    english_glosses=gloss_list,
                )
            else:
                if consecutive_yomitan_failures >= max_consecutive_yomitan_failures:
                    logger.debug(
                        "Deferring candidate '%s' due to Yomitan outage (%d consecutive failures).",
                        term,
                        consecutive_yomitan_failures,
                    )
                    deferred_count += 1
                    continue
                else:
                    entries, status = _fetch_yomitan_term_entries(term, yomitan_url=yomitan_url)
                    if status == "ok" or status is True:
                        consecutive_yomitan_failures = 0
                        if entries:
                            yomitan_meta = _parse_yomitan_metadata(entries, term=term, reading=reading)
                            if yomitan_meta:
                                yomitan_cache[cache_key] = {
                                    "pos_tags": sorted(yomitan_meta.pos_tags),
                                    "is_uk": yomitan_meta.is_uk,
                                    "is_particle_or_aux": yomitan_meta.is_particle_or_aux,
                                    "verb_class": yomitan_meta.verb_class,
                                    "english_glosses": yomitan_meta.english_glosses,
                                }
                            else:
                                yomitan_cache[cache_key] = None
                        else:
                            yomitan_cache[cache_key] = None
                        cache_dirty = True
                    elif status == "connection_error" or status is False:
                        consecutive_yomitan_failures += 1
                        if consecutive_yomitan_failures >= max_consecutive_yomitan_failures:
                            logger.warning(
                                "Yomitan API reached %d consecutive failures; skipping further Yomitan lookups for this run.",
                                max_consecutive_yomitan_failures,
                            )
                        logger.debug("Deferring candidate '%s' due to Yomitan fetch failure.", term)
                        deferred_count += 1
                        continue
                    else:  # status == "term_error"
                        consecutive_yomitan_failures += 1
                        if consecutive_yomitan_failures >= max_consecutive_yomitan_failures:
                            logger.warning(
                                "Yomitan API reached %d consecutive failures; skipping further Yomitan lookups for this run.",
                                max_consecutive_yomitan_failures,
                            )
                        logger.warning(
                            "Skipping candidate '%s' due to Yomitan term-level error (not cached, not deferred).",
                            term,
                        )
                        continue

            if yomitan_meta and yomitan_meta.is_particle_or_aux:
                continue

            if mode != "all":
                if mode == "non-tier1":
                    t1_cands = generate_search_candidates(term, allow_tier2=False, store=store)
                    has_t1 = any(
                        has_title_illustration_match(filter_relevant_illustrations(index.search(c[0]), c[0]), c[0])
                        for c in t1_cands
                    )
                    if has_t1:
                        continue
                elif mode == "unmatched":
                    all_cands = generate_search_candidates(term, allow_tier2=True, store=store)
                    has_any = any(
                        has_title_illustration_match(filter_relevant_illustrations(index.search(c[0]), c[0]), c[0])
                        for c in all_cands
                    )
                    if has_any:
                        continue

            candidates.append({
                "term": term,
                "reading": reading,
                "frequency": freq,
                "pos_tags": sorted(yomitan_meta.pos_tags) if yomitan_meta else [],
                "english_glosses": yomitan_meta.english_glosses if yomitan_meta else [],
                "is_uk": yomitan_meta.is_uk if yomitan_meta else False,
            })

    if cache_dirty:
        try:
            _save_yomitan_cache(yomitan_cache, yomitan_cache_path)
        except Exception as exc:
            logger.warning("Failed to save Yomitan cache to %s: %s", yomitan_cache_path, exc)

    return candidates, deferred_count
