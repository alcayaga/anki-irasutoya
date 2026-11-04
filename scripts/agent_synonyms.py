#!/usr/bin/env python3
"""Local LLM visual concept compilation CLI for Irasutoya catalog."""
import argparse
import logging
import os
import sys
from typing import Sequence

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.synonyms.common import (  # noqa: E402
    DEFAULT_CATALOG_PATH,
    DEFAULT_LLM_CONCEPTS_PATH,
)
from scripts.synonyms.ollama_agent import (  # noqa: E402
    OLLAMA_SYSTEM_PROMPT,
    SEARCH_CATALOG_TOOL,
    VALID_JAPANESE_LATIN_ACRONYMS,
    _contains_english_word,
    _flush_llm_concepts,
    _parse_llm_json_response,
    check_ollama_availability,
    run_ollama_agent_batch,
    run_ollama_compile,
    search_catalog_for_agent,
)
from scripts.synonyms.yomitan_jpdb import (  # noqa: E402
    JPDB_FREQ_RAW_URL,
    _fetch_yomitan_term_entries,
    _parse_yomitan_metadata,
    _save_yomitan_cache,
    fetch_jpdb_freq_github,
    load_jpdb_candidates,
    query_yomitan_term_metadata,
)

__all__ = [
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_LLM_CONCEPTS_PATH",
    "JPDB_FREQ_RAW_URL",
    "OLLAMA_SYSTEM_PROMPT",
    "SEARCH_CATALOG_TOOL",
    "VALID_JAPANESE_LATIN_ACRONYMS",
    "_contains_english_word",
    "_fetch_yomitan_term_entries",
    "_flush_llm_concepts",
    "_parse_llm_json_response",
    "_parse_yomitan_metadata",
    "_save_yomitan_cache",
    "check_ollama_availability",
    "fetch_jpdb_freq_github",
    "load_jpdb_candidates",
    "main",
    "query_yomitan_term_metadata",
    "run_ollama_agent_batch",
    "run_ollama_compile",
    "search_catalog_for_agent",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent_synonyms")


def main(argv: Sequence[str]) -> int:
    """Main CLI entrypoint for local Ollama visual concept compilation."""
    parser = argparse.ArgumentParser(
        description="Run local JPDB + Yomitan + Ollama (gemma4:31b) visual concept compiler for Irasutoya."
    )
    parser.add_argument(
        "--catalog",
        default=DEFAULT_CATALOG_PATH,
        help="Path to destination catalog.sqlite database",
    )
    parser.add_argument(
        "--out",
        "--llm-out",
        dest="out",
        default=DEFAULT_LLM_CONCEPTS_PATH,
        help=f"Output path for compiled concepts JSON (default: {DEFAULT_LLM_CONCEPTS_PATH})",
    )
    parser.add_argument(
        "--jpdb",
        help="Path to local JPDB v2.2 frequency list CSV/TSV file",
    )
    parser.add_argument(
        "--jpdb-limit",
        type=int,
        default=2000,
        help="Number of top JPDB v2.2 words to evaluate (default: 2000)",
    )
    parser.add_argument(
        "--jpdb-mode",
        choices=["non-tier1", "unmatched", "all"],
        default="non-tier1",
        help="Candidate selection mode for JPDB evaluation (default: non-tier1)",
    )
    parser.add_argument(
        "--yomitan-url",
        default="http://127.0.0.1:19633",
        help="Base URL of local Yomitan HTTP API (default: http://127.0.0.1:19633)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/chat",
        help="Ollama chat API endpoint URL (default: http://localhost:11434/api/chat)",
    )
    parser.add_argument(
        "--ollama-model",
        default="gemma4:31b",
        help="Ollama model name (default: gemma4:31b)",
    )
    parser.add_argument(
        "--terms",
        help="Optional comma-separated explicit terms to compile with Ollama",
    )
    parser.add_argument(
        "--cache-dir",
        default=".catalog_cache",
        help="Directory to cache fetched and extracted files",
    )
    parser.add_argument(
        "--clear-yomitan-cache",
        action="store_true",
        help="Clear/discard local cached Yomitan term lookups and re-fetch from API",
    )

    args = parser.parse_args(argv)

    ok = run_ollama_compile(
        catalog_path=args.catalog,
        jpdb_path=args.jpdb,
        limit=args.jpdb_limit,
        mode=args.jpdb_mode,
        yomitan_url=args.yomitan_url,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        llm_out_path=args.out,
        cache_dir=args.cache_dir,
        terms=args.terms,
        clear_yomitan_cache=args.clear_yomitan_cache,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
