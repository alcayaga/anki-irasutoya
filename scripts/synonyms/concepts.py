"""JSON concept mapping validation, population, and catalog synonym verification."""
import json
import logging
import os
import sqlite3
import sys
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

from addon.catalog import (
    CatalogStore,
    canonicalize_synonym_term,
    default_store,
    write_synonyms,
)
from addon.ranking import has_title_illustration_match
from addon.text_utils import clean_search_query
from addon.yomitan_parser import normalize_english_gloss
from scripts.synonyms.common import (
    DEFAULT_CATALOG_PATH,
    _is_disallowed_text_callout_target,
    filter_relevant_illustrations,
    update_catalog_sha256,
)

logger = logging.getLogger("synonyms.concepts")

CONCEPT_MIN_VALID_RATIO = 0.95
BILINGUAL_MIN_MATCHED = 50


def _populate_concept_json_mapping(
    catalog_path: str,
    json_path: str,
    source: str,
    min_matched: int,
    concept_key: str = "concepts",
    validate_canonical_headwords: bool = False,
    strict: bool = True,
    validate_only: bool = False,
) -> bool:
    """Validate JSON visual concepts against catalog illustration titles and populate synonyms.

    Args:
        catalog_path: Path to destination catalog.sqlite database.
        json_path: Path to source JSON file containing concept definitions.
        source: Source attribution string for synonyms table.
        min_matched: Minimum required valid concept mappings before writing.
        concept_key: Key in the JSON dictionary to extract concepts from (default: 'concepts').
        validate_canonical_headwords: If True, validates headwords match canonical English glosses.
        strict: If True, requires 100% of targets to validate against the catalog.
        validate_only: If True, validates concepts without modifying the catalog database.

    Returns:
        True if all targets validated and populated successfully, False otherwise.
    """
    catalog_path = os.path.abspath(catalog_path)
    json_path = os.path.abspath(json_path)

    if not os.path.isfile(catalog_path):
        logger.error("Catalog database not found: %s", catalog_path)
        return False

    if not os.path.isfile(json_path):
        logger.error("Concepts file not found: %s", json_path)
        return False

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        logger.error("Root JSON value in %s is not an object", json_path)
        return False

    if concept_key == "concepts":
        raw_concepts = data.get("concepts") or data.get("mappings") or {}
    else:
        raw_concepts = data.get(concept_key) or {}

    if not isinstance(raw_concepts, dict):
        logger.error("Invalid '%s' structure in %s: expected object", concept_key, json_path)
        return False
    if not raw_concepts or len(raw_concepts) < min_matched:
        logger.error(
            "Too few concepts (%d < %d) found in %s",
            len(raw_concepts),
            min_matched,
            json_path,
        )
        return False

    logger.info("Validating %d %s concepts against %s...", len(raw_concepts), source, catalog_path)
    store = CatalogStore(catalog_path)
    index = store.get_index()
    if index is None:
        logger.error("Catalog index is unavailable at %s", catalog_path)
        return False

    for hw, tgts in raw_concepts.items():
        if not isinstance(tgts, (list, tuple)):
            logger.error(
                "Invalid concept value for '%s' in %s: expected list, got %s",
                hw,
                source,
                type(tgts).__name__,
            )
            return False

    total_targets = sum(len(tgts) for tgts in raw_concepts.values())

    # Pre-validate all target keywords against catalog titles
    validation_failures: List[Tuple[str, str]] = []
    headword_failures = 0
    valid_target_count = 0
    clean_mapping: Dict[str, List[str]] = {}
    for headword, targets in sorted(raw_concepts.items()):
        clean_hw = headword.strip()
        if not clean_hw:
            logger.error("Validation FAILED: concept headword is empty or whitespace-only (%r)", headword)
            headword_failures += 1
            continue
        if validate_canonical_headwords:
            norm_hw = normalize_english_gloss(clean_hw)
            if norm_hw != clean_hw:
                logger.error(
                    "Validation FAILED: '%s' is not in canonical normalized form (expected '%s')",
                    clean_hw,
                    norm_hw,
                )
                headword_failures += 1
                if strict:
                    continue
        valid_targets: List[str] = []
        for target in targets:
            clean_tgt = clean_search_query(str(target))
            if not clean_tgt:
                if strict:
                    validation_failures.append((clean_hw, "<empty>"))
                continue
            if clean_search_query(clean_tgt) != clean_tgt:
                if strict:
                    validation_failures.append((clean_hw, f"<non-idempotent: {clean_tgt}>"))
                else:
                    logger.warning(
                        "Skipping non-idempotent target '%s' for headword '%s'",
                        clean_tgt,
                        clean_hw,
                    )
                continue
            if _is_disallowed_text_callout_target(clean_tgt, clean_hw):
                validation_failures.append((clean_hw, f"<disallowed text callout: {clean_tgt}>"))
                continue
            results = filter_relevant_illustrations(index.search(clean_tgt), clean_tgt)
            if not has_title_illustration_match(results, clean_tgt):
                validation_failures.append((clean_hw, clean_tgt))
            else:
                valid_targets.append(clean_tgt)
        if valid_targets:
            valid_target_count += len(valid_targets)
            merged = clean_mapping.setdefault(clean_hw, [])
            merged.extend(t for t in valid_targets if t not in merged)

    if strict:
        if validation_failures or headword_failures:
            for hw, tgt in validation_failures:
                logger.error(
                    "Validation FAILED: '%s' -> target '%s' has 0 title illustration matches in catalog",
                    hw,
                    tgt,
                )
            logger.error(
                "Aborting populate: %d targets and %d headwords failed validation.",
                len(validation_failures),
                headword_failures,
            )
            return False
    else:
        valid_ratio = (valid_target_count / total_targets) if total_targets > 0 else 0.0
        if valid_ratio >= CONCEPT_MIN_VALID_RATIO and len(clean_mapping) >= min_matched and headword_failures == 0:
            for hw, tgt in validation_failures:
                logger.warning(
                    "Non-strict mode: skipping invalid target '%s' for '%s' (0 title illustration matches)",
                    tgt,
                    hw,
                )
        else:
            for hw, tgt in validation_failures:
                logger.error(
                    "Validation FAILED: '%s' -> target '%s' has 0 title illustration matches in catalog",
                    hw,
                    tgt,
                )
            logger.error(
                "Aborting populate: valid ratio %.1f%% < %.1f%% or %d headword failures.",
                valid_ratio * 100,
                CONCEPT_MIN_VALID_RATIO * 100,
                headword_failures,
            )
            return False

    if not clean_mapping or len(clean_mapping) < min_matched:
        logger.error(
            "Too few valid concepts (%d < %d) after validation in %s",
            len(clean_mapping),
            min_matched,
            json_path,
        )
        return False

    if validate_only:
        return True

    # Connect to catalog and write synonyms
    logger.info("Writing %d validated %s concept mappings into %s...", len(clean_mapping), source, catalog_path)
    try:
        conn = sqlite3.connect(catalog_path)
        try:
            inserted = write_synonyms(conn, source=source, mapping=clean_mapping)
            logger.info("Inserted %d synonym rows into catalog (source='%s')", inserted, source)
            conn.execute("PRAGMA journal_mode = DELETE")
            conn.execute("VACUUM")
            conn.commit()
        finally:
            conn.close()
    finally:
        # Re-compute and refresh the SHA256 sidecar checksum after writing synonyms
        update_catalog_sha256(catalog_path)
    return True


def populate_manual_concepts(
    catalog_path: str,
    manual_json_path: str,
    min_matched: int = 1,
    strict: bool = True,
) -> bool:
    """Compile human-curated manual concepts into catalog.sqlite synonyms table.

    Args:
        catalog_path: Path to catalog.sqlite to update.
        manual_json_path: Path to source JSON file containing manual concepts.
        min_matched: Minimum required concept mappings before writing (defaults to 1).
        strict: If True, requires 100% of targets to validate against the catalog.

    Returns:
        True if all targets validated and populated successfully, False otherwise.
    """
    return _populate_concept_json_mapping(
        catalog_path=catalog_path,
        json_path=manual_json_path,
        source="manual",
        min_matched=min_matched,
        validate_canonical_headwords=False,
        strict=strict,
    )


def populate_bilingual_glosses(
    catalog_path: str,
    bilingual_file_path: str,
    min_matched: Optional[int] = None,
    strict: bool = True,
) -> bool:
    """Compile human-curated and validated bilingual definition glosses into catalog.sqlite synonyms table.

    Args:
        catalog_path: Path to catalog.sqlite to update.
        bilingual_file_path: Path to source JSON file containing bilingual concepts.
        min_matched: Minimum required concept mappings before writing (defaults to BILINGUAL_MIN_MATCHED).
        strict: If True, requires 100% of targets to validate against the catalog.

    Returns:
        True if all targets validated and populated successfully, False otherwise.
    """
    if min_matched is None:
        build_mod = sys.modules.get("scripts.build_synonyms")
        min_matched = getattr(build_mod, "BILINGUAL_MIN_MATCHED", BILINGUAL_MIN_MATCHED)
    return _populate_concept_json_mapping(
        catalog_path=catalog_path,
        json_path=bilingual_file_path,
        source="bilingual_gloss",
        min_matched=min_matched,
        validate_canonical_headwords=True,
        strict=strict,
    )


def _clear_synonym_source(catalog_path: str, source: str) -> bool:
    """Delete all synonym rows for a source and refresh the catalog SHA256 sidecar.

    Args:
        catalog_path: Path to catalog.sqlite to update.
        source: Source identifier to delete from synonyms table.

    Returns:
        True if rows were cleared (or already empty) and sidecar refreshed, False on error.
    """
    if not os.path.isfile(catalog_path):
        logger.error("Catalog database not found: %s", catalog_path)
        return False
    try:
        conn = sqlite3.connect(catalog_path)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='synonyms'"
            ).fetchone()
            if has_table:
                conn.execute("DELETE FROM synonyms WHERE source = ?", (source,))
                # Recompute meta.synonym_sources within the same transaction
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT source FROM synonyms ORDER BY source ASC")
                sources = [r[0] for r in cur.fetchall()]
                cur.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('synonym_sources', ?)",
                    (",".join(sources),),
                )
                conn.commit()
            conn.execute("PRAGMA journal_mode = DELETE")
            conn.execute("VACUUM")
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.error("Failed to clear '%s' synonyms in %s: %s", source, catalog_path, exc)
        return False
    update_catalog_sha256(catalog_path)
    return True


def populate_llm_concepts(
    catalog_path: str,
    llm_json_path: str,
    min_matched: int = 1,
    strict: bool = True,
) -> bool:
    """Compile LLM-curated visual concepts (Tier 1 & Tier 2) into catalog.sqlite synonyms table.

    Args:
        catalog_path: Path to catalog.sqlite to update.
        llm_json_path: Path to source JSON file containing LLM concepts.
        min_matched: Minimum required concept mappings before writing (defaults to 1).
        strict: If True, requires 100% of targets to validate against the catalog.

    Returns:
        True if all targets validated and populated successfully, False otherwise.
    """
    try:
        with open(llm_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Failed to read LLM concepts JSON from %s: %s", llm_json_path, exc)
        return False

    if not isinstance(data, dict):
        logger.error("Root JSON structure in %s is not an object", llm_json_path)
        return False

    for field in ("concepts", "mappings", "tier2_concepts"):
        if field in data and not isinstance(data[field], dict):
            logger.error("Invalid '%s' structure in %s: expected object", field, llm_json_path)
            return False

    has_t1 = bool(data.get("concepts") or data.get("mappings"))
    has_t2 = bool(data.get("tier2_concepts"))
    raw_rej = data.get("rejected")

    positive_keys: Set[str] = set()
    for field in ("concepts", "mappings", "tier2_concepts"):
        field_dict = data.get(field)
        if isinstance(field_dict, dict):
            for term in field_dict:
                k = canonicalize_synonym_term(term)
                if k:
                    positive_keys.add(k)

    # Extract semantic rejections, filtering out unverified target misses and accepted concepts
    rejection_mapping: Dict[str, List[str]] = {}
    if isinstance(raw_rej, dict):
        for term, reason in sorted(raw_rej.items()):
            if not isinstance(reason, str):
                continue
            clean_term = canonicalize_synonym_term(term)
            clean_reason = reason.strip()
            if (
                clean_term
                and clean_term not in positive_keys
                and clean_reason
                and not clean_reason.startswith("unverified in catalog:")
            ):
                rejection_mapping[clean_term] = [clean_reason]

    has_rej = bool(rejection_mapping)

    if not has_t1 and not has_t2 and not has_rej:
        logger.error("No 'concepts', 'tier2_concepts', or usable 'rejected' found in %s", llm_json_path)
        return False

    # Phase 1: Validate both Tier 1 and Tier 2 (if present) before mutating the catalog
    if has_t1:
        ok_t1 = _populate_concept_json_mapping(
            catalog_path=catalog_path,
            json_path=llm_json_path,
            source="llm_concept",
            min_matched=min_matched,
            concept_key="concepts",
            validate_canonical_headwords=False,
            strict=strict,
            validate_only=True,
        )
        if not ok_t1:
            return False

    if has_t2:
        ok_t2 = _populate_concept_json_mapping(
            catalog_path=catalog_path,
            json_path=llm_json_path,
            source="llm_concept_tier2",
            min_matched=1,
            concept_key="tier2_concepts",
            validate_canonical_headwords=False,
            strict=strict,
            validate_only=True,
        )
        if not ok_t2:
            return False

    # Phase 2: Apply mutations (write or clear) once both tiers have passed validation
    if has_t1:
        ok_t1 = _populate_concept_json_mapping(
            catalog_path=catalog_path,
            json_path=llm_json_path,
            source="llm_concept",
            min_matched=min_matched,
            concept_key="concepts",
            validate_canonical_headwords=False,
            strict=strict,
            validate_only=False,
        )
        if not ok_t1:
            return False
    else:
        if not _clear_synonym_source(catalog_path, "llm_concept"):
            return False

    if has_t2:
        ok_t2 = _populate_concept_json_mapping(
            catalog_path=catalog_path,
            json_path=llm_json_path,
            source="llm_concept_tier2",
            min_matched=1,
            concept_key="tier2_concepts",
            validate_canonical_headwords=False,
            strict=strict,
            validate_only=False,
        )
        if not ok_t2:
            return False
    else:
        if not _clear_synonym_source(catalog_path, "llm_concept_tier2"):
            return False

    if rejection_mapping:
        try:
            conn = sqlite3.connect(catalog_path)
            try:
                write_synonyms(conn, source="llm_rejected", mapping=rejection_mapping)
                logger.info(
                    "Compiled %d semantic rejections into synonyms (source='llm_rejected')",
                    len(rejection_mapping),
                )
                conn.execute("PRAGMA journal_mode = DELETE")
                conn.execute("VACUUM")
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.error("Failed to populate llm_rejected in %s: %s", catalog_path, exc)
            return False
        finally:
            update_catalog_sha256(catalog_path)
    else:
        if not _clear_synonym_source(catalog_path, "llm_rejected"):
            return False

    return True


def validate_catalog_synonyms(catalog_path: Optional[str] = None) -> bool:
    """Validate that 100% of target keywords in catalog.sqlite synonyms have title matches.

    Args:
        catalog_path: Optional explicit catalog database path.

    Returns:
        True if all target keywords match successfully, False otherwise.
    """
    target_catalog = catalog_path or DEFAULT_CATALOG_PATH
    logger.info("Loading catalog index from %s...", target_catalog)
    store = CatalogStore(target_catalog) if catalog_path else default_store()
    index = store.get_index()
    if index is None:
        logger.error("Catalog index is unavailable. Build %s first.", target_catalog)
        return False

    # Query all synonyms directly from sqlite
    uri_path = urllib.request.pathname2url(os.path.abspath(target_catalog))
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='synonyms'")
        if cur.fetchone() is None:
            logger.warning("No synonyms table found in %s", target_catalog)
            return True

        cur.execute("SELECT term, target, source, ord FROM synonyms ORDER BY term, source, ord")
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        logger.warning("Synonyms table is empty in %s", target_catalog)
        return True

    logger.info("Validating %d synonym rows in catalog.sqlite...", len(rows))
    all_valid = True
    failed_rows: List[Tuple[str, str, str]] = []

    target_ok: Dict[str, bool] = {}
    for term, target, source, _ord in rows:
        if source == "llm_rejected":
            continue
        if target not in target_ok:
            results = filter_relevant_illustrations(index.search(target), target)
            target_ok[target] = has_title_illustration_match(results, target)
        if not target_ok[target]:
            logger.error(
                "Validation FAILED [%s]: '%s' -> target '%s' has 0 title illustration matches in catalog",
                source,
                term,
                target,
            )
            failed_rows.append((term, target, source))
            all_valid = False

    if all_valid:
        logger.info(
            "SUCCESS: All %d synonym targets across sources validated against catalog.sqlite (100%% title match).",
            len(rows),
        )
    else:
        logger.error(
            "FAILURE: %d of %d synonym targets failed catalog title validation.",
            len(failed_rows),
            len(rows),
        )

    return all_valid


def prune_stale_synonyms(
    catalog_path: Optional[str] = None,
    source: str = "llm_concept",
) -> int:
    """Prune stale rows for a source whose target has 0 title illustration matches in catalog.

    Args:
        catalog_path: Filesystem path to catalog.sqlite.
        source: Synonym source name to prune (e.g. 'llm_concept').

    Returns:
        Number of stale rows removed from the synonyms table, or -1 if index unavailable.
    """
    target_catalog = catalog_path or DEFAULT_CATALOG_PATH
    store = CatalogStore(target_catalog) if catalog_path else default_store()
    index = store.get_index()
    if index is None:
        logger.error("Catalog index unavailable for pruning at %s", target_catalog)
        return -1

    conn = sqlite3.connect(target_catalog)
    pruned_count = 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='synonyms'")
        if cur.fetchone() is None:
            logger.info("No synonyms table found in %s; nothing to prune.", target_catalog)
            return 0

        cur.execute("SELECT DISTINCT target FROM synonyms WHERE source = ?", (source,))
        distinct_targets = [row[0] for row in cur.fetchall()]
        stale_targets: List[str] = []
        for tgt in distinct_targets:
            results = filter_relevant_illustrations(index.search(tgt), tgt)
            if not has_title_illustration_match(results, tgt):
                stale_targets.append(tgt)

        if not stale_targets:
            logger.info("No stale %s synonyms found to prune in %s.", source, target_catalog)
            return 0

        logger.info(
            "Pruning %d stale targets (%s) from %s...",
            len(stale_targets),
            source,
            target_catalog,
        )
        cur.executemany(
            "DELETE FROM synonyms WHERE source = ? AND target = ?",
            [(source, tgt) for tgt in stale_targets],
        )
        pruned_count = cur.rowcount
        cur.execute(
            "INSERT OR REPLACE INTO meta (key, value) "
            "SELECT 'synonym_sources', COALESCE(group_concat(source, ','), '') "
            "FROM (SELECT DISTINCT source FROM synonyms ORDER BY source)"
        )
        conn.commit()
        logger.info("Pruned %d stale %s rows from %s", pruned_count, source, target_catalog)
    finally:
        conn.close()

    if pruned_count > 0:
        update_catalog_sha256(target_catalog)

    return pruned_count
