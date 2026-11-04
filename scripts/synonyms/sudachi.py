"""SudachiDict synonym compilation and filtering into catalog.sqlite."""
import csv
import hashlib
import logging
import os
import sqlite3
import time
import unicodedata
import urllib.request
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from addon.catalog import (
    CatalogStore,
    canonicalize_synonym_term,
    create_schema,
    write_synonyms,
)
from addon.ranking import (
    classify_match_boundary,
    has_title_illustration_match,
    is_callout_asset,
    is_decorative_asset,
    is_title_match,
    score_candidate_image,
)
from addon.text_utils import GRAMMAR_FUNCTION_WORDS, clean_search_query
from scripts.synonyms.common import (
    filter_relevant_illustrations,
    update_catalog_sha256,
)

logger = logging.getLogger("synonyms.sudachi")

SUDACHI_DICT_RAW_URL = (
    "https://raw.githubusercontent.com/WorksApplications/SudachiDict/v20261015/src/main/text/synonyms.txt"
)
SUDACHI_MIN_BYTES = 1024 * 1024
CANONICAL_MATCH_BOUNDARIES = frozenset({"exact_quoted", "standalone", "polite_prefix", "inflected"})


def fetch_sudachi_github(
    cache_dir: str = ".catalog_cache",
    url: str = SUDACHI_DICT_RAW_URL,
    max_age_days: float = 7.0,
    min_bytes: int = SUDACHI_MIN_BYTES,
) -> str:
    """Fetch SudachiDict synonyms.txt from GitHub raw content into cache_dir.

    Args:
        cache_dir: Directory to cache fetched synonyms.txt file.
        url: Remote URL for SudachiDict synonyms.txt.
        max_age_days: Maximum age of cached file in days before refreshing.
        min_bytes: Minimum accepted file size in bytes for cache reuse and downloads.

    Returns:
        Path to local cached synonyms.txt file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    out_path = os.path.join(cache_dir, f"sudachi_synonyms_{url_hash}.txt")
    if (
        os.path.isfile(out_path)
        and os.path.getsize(out_path) >= min_bytes
        and (time.time() - os.path.getmtime(out_path) < max_age_days * 86400)
    ):
        logger.info("Reusing cached SudachiDict synonyms at %s", out_path)
        return out_path

    logger.info("Downloading SudachiDict synonyms from %s...", url)
    req = urllib.request.Request(url, headers={"User-Agent": "anki-irasutoya/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read()

    if not content or len(content) < min_bytes:
        raise ValueError(f"Downloaded SudachiDict file too small ({len(content)} bytes)")

    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(content)
    os.replace(tmp_path, out_path)
    logger.info("Saved %d bytes to %s", len(content), out_path)
    return out_path


def clean_sudachi_headword(raw: str) -> str:
    """Return cleaned headword, or empty string if cleaning truncated or structurally altered it.

    Args:
        raw: Raw headword string from Sudachi dictionary row.

    Returns:
        Cleaned headword matching its NFKC-normalized form, or empty string if discarded.
    """
    cleaned = clean_search_query(raw)
    norm = unicodedata.normalize("NFKC", raw).strip()
    if not cleaned or cleaned != norm:
        return ""
    return cleaned


def populate_sudachi_synonyms(
    catalog_path: str,
    sudachi_file_path: str,
) -> bool:
    """Compile SudachiDict synonym records into catalog.sqlite synonyms table.

    Args:
        catalog_path: Path to destination catalog.sqlite database.
        sudachi_file_path: Path to local SudachiDict synonyms.txt CSV file.

    Returns:
        True if compilation and population succeeded, False otherwise.
    """
    catalog_path = os.path.abspath(catalog_path)
    sudachi_file_path = os.path.abspath(sudachi_file_path)

    if not os.path.isfile(catalog_path):
        logger.error("Catalog database not found: %s", catalog_path)
        return False

    if not os.path.isfile(sudachi_file_path):
        logger.error("Sudachi synonyms file not found: %s", sudachi_file_path)
        return False

    logger.info("Loading catalog index from %s...", catalog_path)
    store = CatalogStore(catalog_path)
    index = store.get_index()
    if index is None:
        logger.error("Catalog index is unavailable at %s", catalog_path)
        return False

    # Find existing manual / kaishi terms and pre-construct titles blob from SQLite
    conn = sqlite3.connect(catalog_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='synonyms'")
        has_syns = cur.fetchone() is not None
        existing_covered: Set[str] = set()
        if has_syns:
            cur.execute(
                "SELECT DISTINCT term FROM synonyms WHERE source IN ('manual', 'donkuri/kaishi')"
            )
            existing_covered = {
                canonicalize_synonym_term(r[0])
                for r in cur.fetchall()
            }

        cur.execute("SELECT i.title, p.title FROM images i JOIN posts p ON i.post_id = p.id")
        title_rows = cur.fetchall()
    finally:
        conn.close()

    titles_blob = "\x00".join(f"{r[0]}\x00{r[1]}" for r in title_rows).casefold()

    # Parse SudachiDict CSV records into groups
    groups: Dict[str, List[List[str]]] = defaultdict(list)
    with open(sudachi_file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_line = line.strip()
            if not raw_line or raw_line.startswith("#"):
                continue
            try:
                parts = list(csv.reader([raw_line]))[0]
            except Exception:
                continue
            if len(parts) >= 9:
                groups[parts[0]].append(parts)

    if not groups:
        logger.warning("No valid Sudachi synonym groups found in %s", sudachi_file_path)
        return False

    logger.info("Parsing %d Sudachi synonym groups against %s...", len(groups), catalog_path)

    target_cache: Dict[str, Tuple[bool, bool, float]] = {}

    def validate_target(target: str) -> Tuple[bool, bool, float]:
        """Validate whether target matches catalog illustrations with quality score."""
        if target in target_cache:
            return target_cache[target]
        if target.casefold() not in titles_blob:
            target_cache[target] = (False, False, 0.0)
            return (False, False, 0.0)
        results = filter_relevant_illustrations(index.search(target), target)
        if not has_title_illustration_match(results, target):
            target_cache[target] = (False, False, 0.0)
            return (False, False, 0.0)
        title_imgs = [img for img in results if not is_callout_asset(img, target) and is_title_match(img, target)]
        has_irasuto = any(f"{target}のイラスト" in img.title for img in title_imgs)
        top_score = max((score_candidate_image(img, target) for img in title_imgs), default=0.0)
        res = (True, has_irasuto, top_score)
        target_cache[target] = res
        return res

    source_canonical_cache: Dict[str, int] = {}

    def count_canonical_matches(term: str) -> int:
        """Count canonical illustrations matching term without decorative noise."""
        if term in source_canonical_cache:
            return source_canonical_cache[term]
        if term.casefold() not in titles_blob:
            source_canonical_cache[term] = 0
            return 0
        imgs = [
            img for img in index.search(term)
            if not is_decorative_asset(img, term) and not is_callout_asset(img, term)
        ]
        cnt = 0
        for img in imgs:
            if (
                classify_match_boundary(img.title, term) in CANONICAL_MATCH_BOUNDARIES
                or (img.post_title and classify_match_boundary(img.post_title, term) in CANONICAL_MATCH_BOUNDARIES)
            ):
                cnt += 1
        source_canonical_cache[term] = cnt
        return cnt

    clean_mapping: Dict[str, List[str]] = {}
    for _gid, g_rows in groups.items():
        valid_rows = []
        for r in g_rows:
            reg = r[4].strip() if len(r) > 4 and r[4] else "0"
            field = r[7].strip() if len(r) > 7 else ""
            exp = r[2].strip() if len(r) > 2 and r[2] else "0"
            orth = r[6].strip() if len(r) > 6 and r[6] else "0"
            hw = clean_sudachi_headword(r[8])
            if not hw or hw in GRAMMAR_FUNCTION_WORDS:
                continue
            # Exclude archaic/obsolete, misuse, misspelling, derogatory, and expansion-disabled
            if exp == "2" or reg in ("3", "4") or orth == "3" or "不快" in field:
                continue
            valid_rows.append(r)

        if len(valid_rows) < 2:
            continue

        for s_row in valid_rows:
            s_term = clean_sudachi_headword(s_row[8])
            s_canonical = canonicalize_synonym_term(s_term)
            if not s_canonical or s_canonical in existing_covered:
                continue
            s_exp = s_row[2].strip() if len(s_row) > 2 and s_row[2] else "0"
            if s_exp == "1":  # Cannot trigger expansion
                continue
            s_is_noun = s_row[1].strip() if len(s_row) > 1 and s_row[1] else "1"

            # Candidate targets from the same group
            cands: List[Tuple[str, bool, bool, float]] = []
            for t_row in valid_rows:
                t_term = clean_sudachi_headword(t_row[8])
                if not t_term or canonicalize_synonym_term(t_term) == s_canonical:
                    continue
                t_is_noun = t_row[1].strip() if len(t_row) > 1 and t_row[1] else "1"
                # Predicates (is_noun=2) must not cross-map to nouns (is_noun=1)
                if s_is_noun == "2" and t_is_noun != "2":
                    continue

                ok, has_irasuto, score = validate_target(t_term)
                if not ok:
                    continue

                t_reg = int(t_row[4]) if len(t_row) > 4 and t_row[4].isdigit() else 0
                cands.append((t_term, t_reg == 0, has_irasuto, score))

            if not cands:
                continue

            # Source Need & Delta Gate: Skip if source already has >= 3 canonical title matches
            if count_canonical_matches(s_term) >= 3:
                continue

            # Rank targets: register == 0 preferred, then f"{T}のイラスト", then score_candidate_image
            cands.sort(key=lambda c: (1 if c[1] else 0, 1 if c[2] else 0, c[3]), reverse=True)

            # Deduplicate and cap at max 3 targets
            seen_targets: Set[str] = set()
            final_targets: List[str] = []
            for cand in cands:
                if cand[0] not in seen_targets:
                    seen_targets.add(cand[0])
                    final_targets.append(cand[0])
                    if len(final_targets) >= 3:
                        break

            if final_targets:
                merged = clean_mapping.get(s_canonical, [])
                for tgt in final_targets:
                    if tgt not in merged and len(merged) < 3:
                        merged.append(tgt)
                clean_mapping[s_canonical] = merged

    new_rows = sum(len(t) for t in clean_mapping.values())
    if new_rows == 0:
        logger.error("Sudachi compilation produced 0 synonym rows; refusing to overwrite existing data.")
        return False

    logger.info("Writing %d Sudachi synonym mappings into %s...", len(clean_mapping), catalog_path)
    conn = sqlite3.connect(catalog_path)
    committed = False
    try:
        create_schema(conn)
        existing = 0
        if has_syns:
            legacy_targets: Dict[str, Set[str]] = {}
            for legacy_term, legacy_target in conn.execute(
                "SELECT term, target FROM synonyms WHERE source = 'sudachi'"
            ):
                legacy_key = canonicalize_synonym_term(legacy_term)
                legacy_target_key = canonicalize_synonym_term(legacy_target)
                if legacy_key and legacy_target_key != legacy_key:
                    legacy_targets.setdefault(legacy_key, set()).add(legacy_target)
            existing = sum(min(3, len(targets)) for targets in legacy_targets.values())

        if existing > 0 and new_rows < existing * 0.8:
            logger.error(
                "Sudachi import produced %d rows, significantly fewer than %d existing rows; refusing replacement.",
                new_rows,
                existing,
            )
            return False
        inserted = write_synonyms(conn, source="sudachi", mapping=clean_mapping, commit=False)
        conn.commit()
        committed = True
        logger.info("Inserted %d synonym rows into catalog (source='sudachi')", inserted)
        try:
            conn.execute("PRAGMA journal_mode = DELETE")
            conn.execute("VACUUM")
            conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning("VACUUM failed (ignored): %s", e)
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to write Sudachi synonyms: %s", exc)
        if committed:
            update_catalog_sha256(catalog_path)
        return False
    finally:
        conn.close()

    update_catalog_sha256(catalog_path)
    return True
