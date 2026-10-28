#!/usr/bin/env python3
"""Build and validation CLI for Irasutoya concept maps and synonym dictionaries."""
import argparse
import csv
import hashlib
import http.client
import json
import logging
import os
import re
import sqlite3
import struct
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    import zstandard
except ImportError:
    zstandard = None

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from addon.catalog import (  # noqa: E402
    CatalogStore,
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
    score_candidate_image,
)
from addon.text_utils import GRAMMAR_FUNCTION_WORDS, clean_search_query  # noqa: E402

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
    catalog_path = os.path.abspath(catalog_path)
    manual_json_path = os.path.abspath(manual_json_path)

    if not os.path.isfile(catalog_path):
        logger.error("Catalog database not found: %s", catalog_path)
        return False

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
    try:
        conn = sqlite3.connect(catalog_path)
        try:
            inserted = write_synonyms(conn, source="manual", mapping=clean_mapping)
            logger.info("Inserted %d synonym rows into catalog (source='manual')", inserted)
            conn.execute("PRAGMA journal_mode = DELETE")
            conn.execute("VACUUM")
            conn.commit()
        finally:
            conn.close()
    finally:
        # Re-compute and refresh the SHA256 sidecar checksum after writing synonyms
        update_catalog_sha256(catalog_path)
    return True


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
            existing_covered = {r[0] for r in cur.fetchall()}

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
            if not s_term or s_term in existing_covered:
                continue
            s_exp = s_row[2].strip() if len(s_row) > 2 and s_row[2] else "0"
            if s_exp == "1":  # Cannot trigger expansion
                continue
            s_is_noun = s_row[1].strip() if len(s_row) > 1 and s_row[1] else "1"

            # Candidate targets from the same group
            cands: List[Tuple[str, bool, bool, float]] = []
            for t_row in valid_rows:
                t_term = clean_sudachi_headword(t_row[8])
                if not t_term or t_term == s_term:
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
                merged = clean_mapping.get(s_term, [])
                for tgt in final_targets:
                    if tgt not in merged and len(merged) < 3:
                        merged.append(tgt)
                clean_mapping[s_term] = merged

    new_rows = sum(len(t) for t in clean_mapping.values())
    if new_rows == 0:
        logger.error("Sudachi compilation produced 0 synonym rows; refusing to overwrite existing data.")
        return False

    logger.info("Writing %d Sudachi synonym mappings into %s...", len(clean_mapping), catalog_path)
    conn = sqlite3.connect(catalog_path)
    committed = False
    try:
        create_schema(conn)
        existing = (
            conn.execute("SELECT count(*) FROM synonyms WHERE source = 'sudachi'").fetchone()[0]
            if has_syns
            else 0
        )
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

    args = parser.parse_args(argv)

    if args.manual:
        ok = populate_manual_concepts(catalog_path=args.catalog, manual_json_path=args.manual)
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
        and not args.validate
    ):
        parser.print_help()

    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
