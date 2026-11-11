#!/usr/bin/env python3
"""Build and validation CLI for Irasutoya concept maps and synonym dictionaries in SQLite."""
import argparse
import http.client
import logging
import os
import sys
import urllib.error
import zipfile
from typing import Sequence

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.synonyms.common import (  # noqa: E402
    DEFAULT_BILINGUAL_CONCEPTS_PATH,
    DEFAULT_CATALOG_PATH,
    DEFAULT_LLM_CONCEPTS_PATH,
    _readonly_catalog_uri,
    filter_relevant_illustrations,
    update_catalog_sha256,
)
from scripts.synonyms.concepts import (  # noqa: E402
    BILINGUAL_MIN_MATCHED,
    CONCEPT_MIN_VALID_RATIO,
    _populate_concept_json_mapping,
    populate_bilingual_glosses,
    populate_llm_concepts,
    populate_manual_concepts,
    prune_stale_synonyms,
    validate_catalog_synonyms,
)
from scripts.synonyms.kaishi import (  # noqa: E402
    KAISHI_MIN_MATCHED,
    clean_alt_text,
    extract_collection_from_apkg,
    fetch_kaishi_github_slice,
    normalize_stem,
    populate_kaishi_concepts,
    zstandard,
)
from scripts.synonyms.sudachi import (  # noqa: E402
    clean_sudachi_headword,
    fetch_sudachi_github,
    populate_sudachi_synonyms,
)
from scripts.synonyms.yomitan_jpdb import (  # noqa: E402
    fetch_jpdb_freq_github,
    load_curated_terms,
    propagate_kana_uk_concepts,
    resolve_uk_aliases,
)

__all__ = [
    "BILINGUAL_MIN_MATCHED",
    "CONCEPT_MIN_VALID_RATIO",
    "DEFAULT_BILINGUAL_CONCEPTS_PATH",
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_LLM_CONCEPTS_PATH",
    "KAISHI_MIN_MATCHED",
    "_populate_concept_json_mapping",
    "_readonly_catalog_uri",
    "clean_alt_text",
    "clean_sudachi_headword",
    "extract_collection_from_apkg",
    "fetch_jpdb_freq_github",
    "fetch_kaishi_github_slice",
    "fetch_sudachi_github",
    "filter_relevant_illustrations",
    "load_curated_terms",
    "main",
    "normalize_stem",
    "populate_bilingual_glosses",
    "populate_kaishi_concepts",
    "populate_llm_concepts",
    "populate_manual_concepts",
    "populate_sudachi_synonyms",
    "propagate_kana_uk_concepts",
    "prune_stale_synonyms",
    "resolve_uk_aliases",
    "update_catalog_sha256",
    "validate_catalog_synonyms",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_synonyms")


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
        "--bilingual",
        nargs="?",
        const=DEFAULT_BILINGUAL_CONCEPTS_PATH,
        help=f"Path to bilingual concepts JSON file to compile into catalog (default: {DEFAULT_BILINGUAL_CONCEPTS_PATH})",
    )
    parser.add_argument(
        "--llm",
        nargs="?",
        const=DEFAULT_LLM_CONCEPTS_PATH,
        help=f"Path to LLM concepts JSON file to compile into catalog (default: {DEFAULT_LLM_CONCEPTS_PATH})",
    )
    parser.add_argument(
        "--jpdb",
        help="Path to local JPDB v2.2 frequency list CSV/TSV to propagate uk aliases",
    )
    parser.add_argument(
        "--jpdb-github",
        action="store_true",
        help="Fetch JPDB v2.2 frequency list from GitHub and propagate uk aliases",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        dest="strict",
        help="Enforce 100%% strict target validation when populating JSON concept files (default)",
    )
    parser.add_argument(
        "--non-strict",
        action="store_false",
        dest="strict",
        help="Allow soft validation (minimum ratio threshold) when populating JSON concept files",
    )
    parser.set_defaults(strict=True)
    parser.add_argument(
        "--kaishi-github",
        action="store_true",
        help="Fetch Kaishi 1.5k latest release from GitHub and populate synonyms and image_keywords",
    )
    parser.add_argument(
        "--kaishi-apkg",
        help="Path to local Kaishi 1.5k .apkg file to populate synonyms and image_keywords",
    )
    parser.add_argument(
        "--sudachi",
        help="Path to local SudachiDict synonyms.txt CSV file to populate into catalog.sqlite",
    )
    parser.add_argument(
        "--sudachi-github",
        action="store_true",
        help="Fetch latest SudachiDict synonyms.txt from GitHub and populate into catalog.sqlite",
    )
    parser.add_argument(
        "--cache-dir",
        default=".catalog_cache",
        help="Directory to cache fetched and extracted SQLite files",
    )
    parser.add_argument(
        "--min-matched",
        type=int,
        default=KAISHI_MIN_MATCHED,
        help=f"Minimum required matched notes before populating into catalog (default: {KAISHI_MIN_MATCHED})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all synonyms in catalog.sqlite against illustration titles",
    )
    parser.add_argument(
        "--prune-stale",
        metavar="SOURCE",
        help="Prune stale synonym rows for specified source that lack catalog title matches",
    )

    args = parser.parse_args(argv)

    jpdb_path = args.jpdb
    if args.jpdb_github:
        if not args.llm:
            parser.error("--jpdb-github requires --llm")
        try:
            jpdb_path = fetch_jpdb_freq_github(cache_dir=args.cache_dir)
        except Exception as exc:
            logger.error("Failed to fetch JPDB frequency list from GitHub: %s", exc)
            return 1
    elif args.jpdb:
        if not args.llm:
            parser.error("--jpdb requires --llm")
        if not os.path.isfile(args.jpdb):
            logger.error("JPDB frequency list not found: %s", args.jpdb)
            return 1

    if args.prune_stale:
        pruned = prune_stale_synonyms(catalog_path=args.catalog, source=args.prune_stale)
        if pruned < 0:
            logger.error("Failed to prune stale synonyms for source '%s'.", args.prune_stale)
            return 1
        logger.info("Pruned %d stale rows for source '%s'.", pruned, args.prune_stale)

    if args.manual:
        ok = populate_manual_concepts(
            catalog_path=args.catalog,
            manual_json_path=args.manual,
            strict=args.strict,
        )
        if not ok:
            return 1

    if args.bilingual:
        ok = populate_bilingual_glosses(
            catalog_path=args.catalog,
            bilingual_file_path=args.bilingual,
            strict=args.strict,
        )
        if not ok:
            return 1

    if args.llm:
        ok = populate_llm_concepts(
            catalog_path=args.catalog,
            llm_json_path=args.llm,
            strict=args.strict,
            jpdb_path=jpdb_path,
        )
        if not ok:
            return 1

    kaishi_errors = [OSError, ValueError, ImportError, zipfile.BadZipFile]
    if zstandard is not None and hasattr(zstandard, "ZstdError"):
        kaishi_errors.append(zstandard.ZstdError)
    kaishi_error_types = tuple(kaishi_errors)

    if args.kaishi_apkg:
        try:
            kaishi_db = extract_collection_from_apkg(args.kaishi_apkg, cache_dir=args.cache_dir)
        except kaishi_error_types as exc:
            logger.error("Failed to extract Kaishi database from apkg: %s", exc)
            return 1
        ok = populate_kaishi_concepts(
            catalog_path=args.catalog, kaishi_db_path=kaishi_db, min_matched=args.min_matched
        )
        if not ok:
            return 1

    if args.kaishi_github:
        try:
            kaishi_db = fetch_kaishi_github_slice(cache_dir=args.cache_dir, allow_full_download=True)
        except kaishi_error_types as exc:
            logger.error("Failed to fetch or extract Kaishi database from GitHub: %s", exc)
            return 1
        ok = populate_kaishi_concepts(
            catalog_path=args.catalog, kaishi_db_path=kaishi_db, min_matched=args.min_matched
        )
        if not ok:
            return 1

    if args.sudachi:
        ok = populate_sudachi_synonyms(catalog_path=args.catalog, sudachi_file_path=args.sudachi)
        if not ok:
            return 1

    if args.sudachi_github:
        try:
            sudachi_path = fetch_sudachi_github(cache_dir=args.cache_dir)
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
            logger.error("Failed to fetch SudachiDict synonyms from GitHub: %s", exc)
            return 1
        ok = populate_sudachi_synonyms(catalog_path=args.catalog, sudachi_file_path=sudachi_path)
        if not ok:
            return 1

    if args.validate:
        ok = validate_catalog_synonyms(catalog_path=args.catalog)
        return 0 if ok else 1

    if (
        not args.manual
        and not args.kaishi_apkg
        and not args.kaishi_github
        and not args.sudachi
        and not args.sudachi_github
        and not args.bilingual
        and not args.llm
        and not args.validate
        and not args.prune_stale
    ):
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
