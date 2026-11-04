#!/usr/bin/env python3
"""Build and validation CLI for Irasutoya concept maps and synonym dictionaries."""
import argparse
import csv
import hashlib
import http.client
import itertools
import json
import logging
import os
import posixpath
import re
import sqlite3
import struct
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    import zstandard
except ImportError:
    zstandard = None

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from addon.catalog import (  # noqa: E402
    CatalogIndex,
    CatalogStore,
    canonicalize_synonym_term,
    create_schema,
    default_store,
    write_image_keywords,
    write_synonyms,
)
from addon.feed import IrasutoyaImage  # noqa: E402
from addon.ranking import (  # noqa: E402
    classify_match_boundary,
    has_title_illustration_match,
    is_callout_asset,
    is_decorative_asset,
    is_image_relevant,
    is_title_match,
    rank_relevant_images,
    score_candidate_image,
)
from addon.text_utils import (  # noqa: E402
    GRAMMAR_FUNCTION_WORDS,
    clean_search_query,
    generate_search_candidates,
    is_grammar_function_word,
)
from addon.yomitan_parser import (  # noqa: E402
    VERB_POS_PATTERN,
    YomitanMetadata,
    is_grammar_sense,
    normalize_english_gloss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_synonyms")

DEFAULT_CATALOG_PATH = os.path.join(REPO_ROOT, "addon", "data", "catalog.sqlite")
DEFAULT_BILINGUAL_CONCEPTS_PATH = os.path.join(REPO_ROOT, "scripts", "sources", "bilingual_concepts.json")
DEFAULT_LLM_CONCEPTS_PATH = os.path.join(REPO_ROOT, "scripts", "sources", "llm_concepts.json")
JPDB_FREQ_RAW_URL = (
    "https://raw.githubusercontent.com/Kuuuube/yomitan-dictionaries/main/data/jpdb_v2.2_freq_list_2024-10-13.csv"
)
CONCEPT_MIN_VALID_RATIO = 0.95
JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf々〆〇]")
VALID_JAPANESE_LATIN_ACRONYMS: Set[str] = {
    "AI", "AR", "ATM", "BGM", "CD", "CM", "DIY", "DJ", "DVD", "FAQ", "GPS",
    "IT", "LAN", "LED", "MC", "MV", "NG", "OK", "OS", "PC", "PR", "PTA",
    "PV", "QR", "SD", "SDGS", "SIM", "SNS", "TV", "URL", "USB", "VIP", "VR",
    "WEB", "WI", "FI", "WIFI",
}


def _contains_english_word(text: str) -> bool:
    """Check whether a candidate target contains embedded English/Latin words.

    Distinguishes English words (e.g. 'cat', 'car') from valid Japanese concepts
    written with Latin letters or initialisms (e.g. 'Tシャツ', 'Wi-Fi', 'QRコード').

    Args:
        text: Target concept or illustration topic candidate.

    Returns:
        True if text contains an embedded English word, False otherwise.
    """
    latin_tokens = re.findall(r"[A-Za-z]+", text)
    for token in latin_tokens:
        if len(token) == 1:
            continue
        if token.upper() in VALID_JAPANESE_LATIN_ACRONYMS:
            continue
        return True
    return False

SEARCH_CATALOG_TOOL = {
    "type": "function",
    "function": {
        "name": "search_catalog",
        "description": (
            "Search the Irasutoya catalog by Japanese query keywords. "
            "Returns matching Blogger posts (title, description, suggested_target) "
            "and the runtime top-1 illustration ('runtime_top1_if_queried') that runtime ranking selects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 1-3 Japanese search candidate queries to evaluate in the catalog.",
                }
            },
            "required": ["queries"],
        },
    },
}

OLLAMA_SYSTEM_PROMPT = """You are an expert Japanese visual concept compiler for the Irasutoya (いらすとや) illustration dictionary.
Your task is to evaluate a Japanese vocabulary term and map it to 1-2 concrete, specific Japanese Irasutoya illustration topics or canonical headwords.

CRITICAL RULES:
1. ONLY return Japanese concepts or illustration topics (NEVER English words).
2. Reject non-visual functional, discourse, grammatical, temporal, or purely abstract concepts (e.g. そのまま, 確かに, 方針, 概念, 数日, 翌日, つもり, わけがない) by setting "visualizable": false.
3. You have access to the `search_catalog` tool. Use it to search the catalog (which checks post titles and descriptions) and inspect `suggested_target` and `runtime_top1_if_queried`.
4. Inspect `runtime_top1_if_queried` carefully! Avoid choosing overly broad targets whose #1 ranked illustration is unrepresentative or unrelated to the intended meaning. Pick specific, representative targets.
5. Every target you choose MUST be confirmed to exist in the catalog with a relevant title match.
6. When done, output ONLY a JSON object:
{
  "visualizable": true or false,
  "reason": "Brief explanation",
  "targets": ["JapaneseTarget1", "JapaneseTarget2"]
}"""


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


def _populate_concept_json_mapping(
    catalog_path: str,
    json_path: str,
    source: str,
    min_matched: int,
    validate_canonical_headwords: bool = False,
    strict: bool = True,
) -> bool:
    """Validate JSON visual concepts against catalog illustration titles and populate synonyms.

    Args:
        catalog_path: Path to destination catalog.sqlite database.
        json_path: Path to source JSON file containing concept definitions.
        source: Source attribution string for synonyms table.
        min_matched: Minimum required valid concept mappings before writing.
        validate_canonical_headwords: If True, validates headwords match canonical English glosses.
        strict: If True, requires 100% of targets to validate against the catalog.

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

    raw_concepts: Dict[str, List[str]] = data.get("concepts") or data.get("mappings") or {}
    if not isinstance(raw_concepts, dict):
        logger.error("Invalid 'concepts' structure in %s: expected object", json_path)
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
    clean_mapping: Dict[str, List[str]] = {}
    for headword, targets in sorted(raw_concepts.items()):
        clean_hw = headword.strip()
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
            clean_tgt = str(target).strip()
            if not clean_tgt:
                if strict:
                    validation_failures.append((clean_hw, "<empty>"))
                continue
            results = filter_relevant_illustrations(index.search(clean_tgt), clean_tgt)
            if not has_title_illustration_match(results, clean_tgt):
                validation_failures.append((clean_hw, clean_tgt))
            else:
                valid_targets.append(clean_tgt)
        if valid_targets:
            clean_mapping[clean_hw] = valid_targets

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
        valid_count = sum(len(tgts) for tgts in clean_mapping.values())
        valid_ratio = (valid_count / total_targets) if total_targets > 0 else 0.0
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


BILINGUAL_MIN_MATCHED = 50


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
        min_matched = BILINGUAL_MIN_MATCHED
    return _populate_concept_json_mapping(
        catalog_path=catalog_path,
        json_path=bilingual_file_path,
        source="bilingual_gloss",
        min_matched=min_matched,
        validate_canonical_headwords=True,
        strict=strict,
    )


def populate_llm_concepts(
    catalog_path: str,
    llm_json_path: str,
    min_matched: int = 1,
    strict: bool = True,
) -> bool:
    """Compile LLM-curated visual concepts into catalog.sqlite synonyms table.

    Args:
        catalog_path: Path to catalog.sqlite to update.
        llm_json_path: Path to source JSON file containing LLM concepts.
        min_matched: Minimum required concept mappings before writing (defaults to 1).
        strict: If True, requires 100% of targets to validate against the catalog.

    Returns:
        True if all targets validated and populated successfully, False otherwise.
    """
    return _populate_concept_json_mapping(
        catalog_path=catalog_path,
        json_path=llm_json_path,
        source="llm_concept",
        min_matched=min_matched,
        validate_canonical_headwords=False,
        strict=strict,
    )


KAISHI_RELEASES_API = "https://api.github.com/repos/donkuri/kaishi/releases/latest"
KAISHI_FALLBACK_URL = "https://github.com/donkuri/kaishi/releases/download/v2.4.3/Kaishi.1.5k.v2.4.3.apkg"
KAISHI_MIN_MATCH_RATIO = 0.5


def normalize_stem(src: str) -> str:
    """Normalize image source filename stem by stripping extensions and trailing Anki media hashes."""
    base = re.sub(r"(\.(?:png|jpg|jpeg|webp|gif))+$", "", src, flags=re.I)
    base = re.sub(r"-[0-9a-fA-F]{8,}$", "", base)
    return base


def clean_alt_text(alt: str) -> str:
    """Clean alt text from image tags by stripping Irasutoya website branding suffixes."""
    return re.sub(r"\s*\|\s*かわいいフリー素材集.*$", "", alt).strip()


MAX_COLLECTION_SIZE = 50 * 1024 * 1024


def extract_collection_from_apkg(apkg_path: str, cache_dir: str = ".catalog_cache") -> str:
    """Extract SQLite database from local .apkg file, decompressing collection.anki21b if needed.

    Args:
        apkg_path: Path to local .apkg file.
        cache_dir: Directory to cache extracted SQLite file.

    Returns:
        Path to extracted SQLite database file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(apkg_path))[0]
    out_path = os.path.join(cache_dir, f"{basename}_collection.sqlite")

    with zipfile.ZipFile(apkg_path, "r") as z:
        names = set(z.namelist())
        if "collection.anki21b" in names:
            if z.getinfo("collection.anki21b").file_size > MAX_COLLECTION_SIZE:
                raise ValueError(
                    f"collection.anki21b in {apkg_path} exceeds the {MAX_COLLECTION_SIZE // (1024 * 1024)} MB limit"
                )
            if zstandard is None:
                raise ImportError(
                    "zstandard library is required to decompress collection.anki21b from .apkg. "
                    "Install with 'pip install zstandard'."
                )
            payload = zstandard.ZstdDecompressor().decompress(
                z.read("collection.anki21b"), max_output_size=MAX_COLLECTION_SIZE
            )
        elif "collection.anki21" in names or "collection.anki2" in names:
            member = "collection.anki21" if "collection.anki21" in names else "collection.anki2"
            if z.getinfo(member).file_size > MAX_COLLECTION_SIZE:
                raise ValueError(
                    f"{member} in {apkg_path} exceeds the {MAX_COLLECTION_SIZE // (1024 * 1024)} MB limit"
                )
            payload = z.read(member)
        else:
            raise ValueError(f"No collection database found in {apkg_path}")

    tmp_out = out_path + ".part"
    with open(tmp_out, "wb") as f:
        f.write(payload)
    os.replace(tmp_out, out_path)
    return out_path


def fetch_kaishi_github_slice(
    cache_dir: str = ".catalog_cache",
    allow_full_download: bool = False,
) -> str:
    """Fetch collection.anki21b from Kaishi latest release using a ~512KB HTTP Range slice.

    Args:
        cache_dir: Directory to cache extracted SQLite file.
        allow_full_download: Whether to fall back to downloading full .apkg if slice parse fails.

    Returns:
        Path to extracted SQLite database file.
    """
    os.makedirs(cache_dir, exist_ok=True)

    tag = "latest"
    asset_url = KAISHI_FALLBACK_URL
    try:
        req = urllib.request.Request(KAISHI_RELEASES_API, headers={"User-Agent": "anki-irasutoya"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
            tag = re.sub(r"[^A-Za-z0-9._-]", "_", str(data.get("tag_name", "latest")))
            for a in data.get("assets", []):
                if a.get("name", "").endswith(".apkg"):
                    asset_url = a.get("browser_download_url", asset_url)
                    break
    except Exception as exc:
        logger.warning("Could not query GitHub releases API (%s); using fallback URL %s", exc, asset_url)

    if tag == "latest":
        v_match = re.search(r"/download/([^/]+)/", asset_url)
        if v_match:
            tag = re.sub(r"[^A-Za-z0-9._-]", "_", v_match.group(1))

    url_hash = hashlib.sha256(asset_url.encode("utf-8")).hexdigest()[:10]
    cached_sqlite = os.path.join(cache_dir, f"kaishi_{tag}_{url_hash}_collection.sqlite")
    if os.path.isfile(cached_sqlite) and os.path.getsize(cached_sqlite) > 100000:
        logger.info("Using cached Kaishi database at %s", cached_sqlite)
        return cached_sqlite

    logger.info("Fetching Kaishi 512KB slice from %s...", asset_url)
    req = urllib.request.Request(
        asset_url,
        headers={"Range": "bytes=0-524287", "User-Agent": "anki-irasutoya"},
    )
    last_err: Optional[Exception] = None
    slice_bytes = b""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                slice_bytes = resp.read(524288)
            last_err = None
            break
        except OSError as exc:
            last_err = exc
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    if last_err is not None:
        raise last_err

    try:
        idx = slice_bytes.find(b"collection.anki21b")
        if idx == -1:
            raise ValueError("collection.anki21b not found in fetched slice from GitHub")

        header_start = idx - 30
        if header_start < 0:
            raise ValueError("Malformed ZIP local header for collection.anki21b in fetched slice")

        _sig, _ver, flags, comp, _mtime, _mdate, _crc, csize, _usize, fn_len, extra_len = struct.unpack(
            "<4sHHHHHIIIHH", slice_bytes[header_start : header_start + 30]
        )
        if _sig != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header signature for collection.anki21b")
        if comp != 0 or (flags & 0x08) or csize == 0:
            raise ValueError("Unsupported ZIP entry layout (compressed or data-descriptor) for collection.anki21b")

        file_data_start = header_start + 30 + fn_len + extra_len
        compressed_data = slice_bytes[file_data_start : file_data_start + csize]
        if len(compressed_data) != csize:
            raise ValueError("collection.anki21b is larger than the fetched 512KB slice")

        if zstandard is None:
            raise ImportError(
                "zstandard library is required to decompress collection.anki21b from GitHub slice. "
                "Install with 'pip install zstandard'."
            )

        dctx = zstandard.ZstdDecompressor()
        try:
            decomp = dctx.decompress(compressed_data, max_output_size=50 * 1024 * 1024)
        except Exception as z_err:
            raise ValueError(f"Zstandard decompression failed: {z_err}") from z_err

        tmp_sqlite = cached_sqlite + ".part"
        with open(tmp_sqlite, "wb") as f:
            f.write(decomp)
        os.replace(tmp_sqlite, cached_sqlite)
        logger.info("Extracted and cached Kaishi collection SQLite to %s", cached_sqlite)
        return cached_sqlite
    except (ValueError, struct.error, zipfile.BadZipFile) as slice_err:
        if not allow_full_download:
            raise
        logger.warning(
            "Range slice optimization failed (%s); falling back to full .apkg download from %s...",
            slice_err,
            asset_url,
        )
        apkg_dest = os.path.join(cache_dir, f"kaishi_{tag}_{url_hash}.apkg")
        if not os.path.isfile(apkg_dest):
            tmp_apkg = apkg_dest + ".part"
            with urllib.request.urlopen(
                urllib.request.Request(asset_url, headers={"User-Agent": "anki-irasutoya"}),
                timeout=120,
            ) as r_full:
                with open(tmp_apkg, "wb") as f_out:
                    while True:
                        chunk = r_full.read(65536)
                        if not chunk:
                            break
                        f_out.write(chunk)
            os.replace(tmp_apkg, apkg_dest)
        return extract_collection_from_apkg(apkg_dest, cache_dir=cache_dir)


KAISHI_MIN_MATCHED = 1000


def populate_kaishi_concepts(
    catalog_path: str,
    kaishi_db_path: str,
    min_matched: int = KAISHI_MIN_MATCHED,
) -> bool:
    """Compile ground-truth Kaishi 1.5k concepts into synonyms and image_keywords tables.

    Args:
        catalog_path: Path to destination catalog.sqlite database.
        kaishi_db_path: Path to extracted Kaishi collection SQLite database.
        min_matched: Minimum required matched notes before writing (default 1000).

    Returns:
        True if population succeeded, False otherwise.
    """
    catalog_path = os.path.abspath(catalog_path)
    kaishi_db_path = os.path.abspath(kaishi_db_path)

    if not os.path.isfile(catalog_path):
        logger.error("Catalog database not found: %s", catalog_path)
        return False
    if not os.path.isfile(kaishi_db_path):
        logger.error("Kaishi database not found: %s", kaishi_db_path)
        return False

    store = CatalogStore(catalog_path)
    index = store.get_index()
    if index is None:
        logger.error("Catalog index is unavailable at %s", catalog_path)
        return False

    # Read notes from Kaishi SQLite
    k_conn = sqlite3.connect(kaishi_db_path)
    try:
        k_cur = k_conn.cursor()
        k_cur.execute("SELECT flds FROM notes")
        kaishi_rows = k_cur.fetchall()
    finally:
        k_conn.close()

    # Load images from catalog to build stem and title lookup maps
    cat_conn = sqlite3.connect(catalog_path)
    try:
        cat_cur = cat_conn.cursor()
        cat_cur.execute(
            """
            SELECT i.id, i.url, i.title, p.title
            FROM images i
            JOIN posts p ON i.post_id = p.id
            """
        )
        catalog_images = cat_cur.fetchall()
    finally:
        cat_conn.close()

    url_stems: Dict[str, Tuple[int, str, str]] = {}
    title_to_rec: Dict[str, Tuple[int, str, str]] = {}
    post_title_to_rec: Dict[str, Tuple[int, str, str]] = {}
    for img_id, url, title, p_title in catalog_images:
        path = urllib.parse.urlparse(url).path
        fname = os.path.basename(path)
        stem = normalize_stem(fname)
        if stem not in url_stems:
            url_stems[stem] = (img_id, title, p_title)
        if title not in title_to_rec:
            title_to_rec[title] = (img_id, title, p_title)
        if p_title not in post_title_to_rec:
            post_title_to_rec[p_title] = (img_id, title, p_title)

    image_keywords: List[Tuple[int, str, float]] = []
    synonyms_dict: Dict[str, List[str]] = {}
    matched_count = 0
    unmatched_words: List[str] = []

    for r in kaishi_rows:
        flds = r[0].split("\x1f")
        raw_word = flds[0]
        if not raw_word or "Welcome to Kaishi" in raw_word:
            continue
        word = clean_search_query(raw_word)
        if not word:
            continue

        pic = next((f.strip() for f in flds[1:] if "<img" in f), "")
        m = re.search(r"<img\s+([^>]*?)>", pic)
        if not m:
            unmatched_words.append(word)
            continue

        attrs = m.group(1)
        src_m = re.search(r'src=["\']([^"\']+)["\']', attrs)
        alt_m = re.search(r'alt=["\']([^"\']+)["\']', attrs)
        src = src_m.group(1) if src_m else ""
        alt = alt_m.group(1) if alt_m else ""

        stem = normalize_stem(src)
        rec = url_stems.get(stem)
        if not rec and alt:
            c_alt = clean_alt_text(alt)
            rec = title_to_rec.get(c_alt) or post_title_to_rec.get(c_alt)

        if not rec:
            unmatched_words.append(word)
            continue

        img_id, title, _post_title = rec
        matched_count += 1
        image_keywords.append((img_id, word, 1.0))

        if word in GRAMMAR_FUNCTION_WORDS:
            continue
        b = classify_match_boundary(title, word)
        if b in ("exact_quoted", "standalone", "polite_prefix", "inflected"):
            continue

        target = re.sub(r"のイラスト.*$", "", title)
        target = re.sub(r"^(?:色々な|いろいろな)", "", target).strip()
        if not target or clean_search_query(target) != target or target == word:
            continue

        res = filter_relevant_illustrations(index.search(target), target)
        if not has_title_illustration_match(res, target):
            continue

        if word not in synonyms_dict:
            synonyms_dict[word] = [target]
        elif target not in synonyms_dict[word]:
            synonyms_dict[word].append(target)

    logger.info(
        "Kaishi mapping: %d matched notes (%d unmatched). Generated %d image_keywords, %d synonym terms (%d rows).",
        matched_count,
        len(unmatched_words),
        len(image_keywords),
        len(synonyms_dict),
        sum(len(t) for t in synonyms_dict.values()),
    )

    eligible = matched_count + len(unmatched_words)
    match_ratio = matched_count / eligible if eligible else 0.0
    if matched_count == 0 or match_ratio < KAISHI_MIN_MATCH_RATIO or matched_count < min_matched:
        logger.error(
            "Only %d of %d Kaishi notes matched catalog images (%.1f%%, min required: %d); refusing to overwrite existing '%s' data.",
            matched_count,
            eligible,
            match_ratio * 100,
            min_matched,
            "donkuri/kaishi",
        )
        return False

    committed = False
    conn = sqlite3.connect(catalog_path)
    try:
        create_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM synonyms WHERE source = 'donkuri/kaishi'")
        existing_syns_row = cur.fetchone()
        existing_syns = existing_syns_row[0] if existing_syns_row else 0
        cur.execute("SELECT count(*) FROM image_keywords WHERE source = 'donkuri/kaishi'")
        existing_kws_row = cur.fetchone()
        existing_kws = existing_kws_row[0] if existing_kws_row else 0

        new_syn_rows = sum(len(t) for t in synonyms_dict.values())
        if existing_syns > 0 and new_syn_rows < existing_syns * 0.8:
            logger.error(
                "Kaishi import produced %d synonym rows, which is significantly fewer than %d existing rows; refusing replacement.",
                new_syn_rows,
                existing_syns,
            )
            return False
        if existing_kws > 0 and len(image_keywords) < existing_kws * 0.8:
            logger.error(
                "Kaishi import produced %d keywords, which is significantly fewer than %d existing keywords; refusing replacement.",
                len(image_keywords),
                existing_kws,
            )
            return False

        conn.execute("BEGIN TRANSACTION")
        write_synonyms(conn, source="donkuri/kaishi", mapping=synonyms_dict, commit=False)
        write_image_keywords(conn, source="donkuri/kaishi", keywords=image_keywords, commit=False)
        conn.commit()
        committed = True
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("VACUUM")
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to write Kaishi tables: %s", exc)
        if committed:
            update_catalog_sha256(catalog_path)
        return False
    finally:
        conn.close()

    # Refresh the checksum sidecar ONLY after the update transaction succeeds
    update_catalog_sha256(catalog_path)
    return True


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
        logger.info("Using cached JPDB frequency list at %s", target_path)
        return os.path.abspath(target_path)

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
                if hw.get("term") == term:
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


def _readonly_catalog_uri(catalog_path: str) -> str:
    """Return an encoded read-only immutable SQLite URI for catalog_path.

    Args:
        catalog_path: Filesystem path to the catalog database.

    Returns:
        Properly encoded SQLite URI string with mode=ro&immutable=1.
    """
    uri_path = urllib.request.pathname2url(os.path.abspath(catalog_path))
    return f"file:{uri_path}?mode=ro&immutable=1"


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

    with open(jpdb_path, "r", encoding="utf-8") as f:
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


def search_catalog_for_agent(
    catalog_path: str,
    query: str,
    limit: int = 5,
    index: Optional[CatalogIndex] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """Search catalog posts and runtime top-1 preview for an LLM agent query.

    Args:
        catalog_path: Path to catalog.sqlite.
        query: Search term or concept query.
        limit: Max post results to return.
        index: Optional pre-loaded CatalogIndex.
        conn: Optional SQLite connection.

    Returns:
        Dict with 'query', 'posts', and 'runtime_top1_if_queried'.
    """
    clean_q = clean_search_query(query)
    if not clean_q:
        return {"query": query, "posts": [], "runtime_top1_if_queried": None}

    close_conn = False
    if conn is None:
        conn = sqlite3.connect(_readonly_catalog_uri(catalog_path), uri=True)
        close_conn = True

    try:
        cur = conn.cursor()
        escaped_q = (
            clean_q.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        cur.execute(
            "SELECT id, title, description FROM posts WHERE title LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' LIMIT ?",
            (f"%{escaped_q}%", f"%{escaped_q}%", limit),
        )
        posts = []
        for pid, ptitle, pdesc in cur.fetchall():
            m = re.match(r"^(.+?)のイラスト(?:$|（)", ptitle)
            suggested = m.group(1) if m else ptitle
            posts.append({
                "post_id": pid,
                "title": ptitle,
                "description": pdesc[:120] if pdesc else "",
                "suggested_target": suggested,
            })

        top1 = None
        if clean_q:
            if index is None:
                store = CatalogStore(catalog_path)
                index = store.get_index()
            if index is not None:
                results = filter_relevant_illustrations(index.search(clean_q), clean_q)
                if results and has_title_illustration_match(results, clean_q):
                    best_img = rank_relevant_images(results, clean_q)[0]
                    cur.execute("SELECT title, description FROM posts WHERE blogger_id = ?", (best_img.post_id,))
                    p_row = cur.fetchone()
                    top1 = {
                        "matched_image_title": best_img.title,
                        "post_title": p_row[0] if p_row else "",
                        "post_description": (p_row[1][:120] if p_row and p_row[1] else ""),
                    }
        return {"query": query, "posts": posts, "runtime_top1_if_queried": top1}
    finally:
        if close_conn:
            conn.close()


def _parse_llm_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON object from LLM response text, handling markdown fences and commentary.

    Args:
        text: Raw LLM response string.

    Returns:
        Parsed dictionary if successful, None otherwise.
    """
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


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
        "--ollama-compile",
        action="store_true",
        help="Run the build-time JPDB + Yomitan + Ollama (gemma4:31b) visual concept compiler",
    )
    parser.add_argument(
        "--llm-out",
        default=DEFAULT_LLM_CONCEPTS_PATH,
        help=f"Output path for --ollama-compile (default: {DEFAULT_LLM_CONCEPTS_PATH})",
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
        "--clear-yomitan-cache",
        action="store_true",
        help="Clear/discard local cached Yomitan term lookups and re-fetch from API",
    )
    parser.add_argument(
        "--prune-stale",
        metavar="SOURCE",
        help="Prune stale synonym rows for specified source that lack catalog title matches",
    )

    args = parser.parse_args(argv)

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

    if args.ollama_compile:
        ok = run_ollama_compile(
            catalog_path=args.catalog,
            jpdb_path=args.jpdb,
            limit=args.jpdb_limit,
            mode=args.jpdb_mode,
            yomitan_url=args.yomitan_url,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            llm_out_path=args.llm_out,
            cache_dir=args.cache_dir,
            terms=args.terms,
            clear_yomitan_cache=args.clear_yomitan_cache,
        )
        if not ok:
            return 1

    if args.llm:
        ok = populate_llm_concepts(
            catalog_path=args.catalog,
            llm_json_path=args.llm,
            strict=args.strict,
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
        and not args.ollama_compile
        and not args.validate
        and not args.prune_stale
    ):
        parser.print_help()

    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
