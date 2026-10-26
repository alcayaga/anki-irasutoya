#!/usr/bin/env python3
"""Build and validation CLI for Irasutoya concept maps and synonym dictionaries."""
import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from addon.catalog import CatalogStore, default_store, write_synonyms  # noqa: E402
from addon.feed import IrasutoyaImage  # noqa: E402
from addon.ranking import (  # noqa: E402
    has_title_illustration_match,
    is_decorative_asset,
    is_image_relevant,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_synonyms")

DEFAULT_CATALOG_PATH = os.path.join(REPO_ROOT, "addon", "data", "catalog.sqlite")


def filter_relevant_illustrations(
    images: Sequence[IrasutoyaImage],
    query: str,
) -> List[IrasutoyaImage]:
    """Filter out decorative assets and images not relevant to query, matching runtime client logic."""
    return [
        img for img in images
        if not is_decorative_asset(img, query) and is_image_relevant(img, query)
    ]


def update_catalog_sha256(catalog_path: str) -> str:
    """Compute and write SHA256 checksum for catalog database.

    Args:
        catalog_path: Path to catalog.sqlite.

    Returns:
        Hex-encoded SHA256 digest string.
    """
    hasher = hashlib.sha256()
    with open(catalog_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    digest = hasher.hexdigest()

    sha_path = f"{catalog_path}.sha256"
    basename = os.path.basename(catalog_path)
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {basename}\n")
    logger.info("Updated SHA256 checksum at %s", sha_path)
    return digest


def populate_manual_concepts(
    catalog_path: str,
    manual_json_path: str,
) -> bool:
    """Compile human-curated manual concepts into catalog.sqlite synonyms table.

    Args:
        catalog_path: Path to catalog.sqlite to update.
        manual_json_path: Path to source JSON file containing manual concepts.

    Returns:
        True if all targets validated and populated successfully, False otherwise.
    """
    if not os.path.isfile(manual_json_path):
        logger.error("Manual concepts file not found: %s", manual_json_path)
        return False

    with open(manual_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_concepts: Dict[str, List[str]] = data.get("concepts", {})
    if not raw_concepts:
        logger.error("No concepts found in %s", manual_json_path)
        return False

    logger.info("Validating %d manual concepts against %s...", len(raw_concepts), catalog_path)
    store = CatalogStore(catalog_path)
    index = store.get_index()
    if index is None:
        logger.error("Catalog index is unavailable at %s", catalog_path)
        return False

    # Pre-validate all target keywords against catalog titles
    validation_failures: List[Tuple[str, str]] = []
    clean_mapping: Dict[str, List[str]] = {}
    for headword, targets in sorted(raw_concepts.items()):
        clean_hw = headword.strip()
        valid_targets: List[str] = []
        for target in targets:
            clean_tgt = str(target).strip()
            if not clean_tgt:
                continue
            results = filter_relevant_illustrations(index.search(clean_tgt), clean_tgt)
            if not has_title_illustration_match(results, clean_tgt):
                validation_failures.append((clean_hw, clean_tgt))
            else:
                valid_targets.append(clean_tgt)
        if valid_targets:
            clean_mapping[clean_hw] = valid_targets

    if validation_failures:
        for hw, tgt in validation_failures:
            logger.error("Validation FAILED: '%s' -> target '%s' has 0 title illustration matches in catalog", hw, tgt)
        logger.error("Aborting populate: %d targets failed validation.", len(validation_failures))
        return False

    # Connect to catalog and write synonyms
    logger.info("Writing %d validated manual concept mappings into %s...", len(clean_mapping), catalog_path)
    conn = sqlite3.connect(catalog_path)
    try:
        inserted = write_synonyms(conn, source="manual", mapping=clean_mapping)
        logger.info("Inserted %d synonym rows into catalog (source='manual')", inserted)
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()

    update_catalog_sha256(catalog_path)
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

    for term, target, source, _ord in rows:
        results = filter_relevant_illustrations(index.search(target), target)
        if not has_title_illustration_match(results, target):
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


def main(argv: Sequence[str]) -> int:
    """Main CLI entrypoint for synonym building and validation."""
    parser = argparse.ArgumentParser(description="Build and validate Irasutoya concept maps and synonyms in SQLite.")
    parser.add_argument(
        "--catalog",
        default=DEFAULT_CATALOG_PATH,
        help="Path to destination catalog.sqlite database",
    )
    parser.add_argument(
        "--manual",
        help="Path to manual concepts JSON source file to populate into catalog.sqlite",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all synonyms in catalog.sqlite against illustration titles",
    )

    args = parser.parse_args(argv)

    if args.manual:
        ok = populate_manual_concepts(catalog_path=args.catalog, manual_json_path=args.manual)
        if not ok:
            return 1

    if args.validate:
        ok = validate_catalog_synonyms(catalog_path=args.catalog)
        return 0 if ok else 1

    if not args.manual and not args.validate:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
