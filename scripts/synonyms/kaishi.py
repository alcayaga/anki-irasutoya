"""Kaishi 1.5k deck extraction and catalog synonym / image_keywords population."""
import hashlib
import json
import logging
import os
import re
import sqlite3
import struct
import time
import urllib.parse
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

try:
    import zstandard
except ImportError:
    zstandard = None

from addon.catalog import (
    CatalogStore,
    create_schema,
    write_image_keywords,
    write_synonyms,
)
from addon.ranking import classify_match_boundary, has_title_illustration_match
from addon.text_utils import GRAMMAR_FUNCTION_WORDS, clean_search_query
from scripts.synonyms.common import (
    filter_relevant_illustrations,
    update_catalog_sha256,
)

logger = logging.getLogger("synonyms.kaishi")

KAISHI_RELEASES_API = "https://api.github.com/repos/donkuri/kaishi/releases/latest"
KAISHI_FALLBACK_URL = "https://github.com/donkuri/kaishi/releases/download/v2.4.3/Kaishi.1.5k.v2.4.3.apkg"
KAISHI_MIN_MATCH_RATIO = 0.5
KAISHI_MIN_MATCHED = 1000
MAX_COLLECTION_SIZE = 50 * 1024 * 1024


def normalize_stem(src: str) -> str:
    """Normalize image source filename stem by stripping extensions and trailing Anki media hashes.

    Args:
        src: Image source path or URL.

    Returns:
        Cleaned base filename without extension or hash suffix.
    """
    base = re.sub(r"(\.(?:png|jpg|jpeg|webp|gif))+$", "", src, flags=re.I)
    base = re.sub(r"-[0-9a-fA-F]{8,}$", "", base)
    return base


def clean_alt_text(alt: str) -> str:
    """Clean alt text from image tags by stripping Irasutoya website branding suffixes.

    Args:
        alt: Image alt attribute string.

    Returns:
        Cleaned description without branding.
    """
    return re.sub(r"\s*\|\s*かわいいフリー素材集.*$", "", alt).strip()


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
        distinct_kws_count = len({(img_id, kw) for img_id, kw, _ in image_keywords})
        if existing_kws > 0 and distinct_kws_count < existing_kws * 0.8:
            logger.error(
                "Kaishi import produced %d distinct keywords, which is significantly fewer than %d existing keywords; refusing replacement.",
                distinct_kws_count,
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
