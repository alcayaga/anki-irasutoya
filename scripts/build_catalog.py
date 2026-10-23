#!/usr/bin/env python3
"""Build and validate the offline SQLite metadata catalog for Irasutoya."""
import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from typing import List, Optional, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from addon.catalog import CatalogStore, write_posts  # noqa: E402
from addon.feed import ParsedPost, parse_entry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_catalog")

BASE_FEED_URL = "https://www.irasutoya.com/feeds/posts/default"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36 (Anki-Irasutoya-CatalogBuilder/1.0)"
)


def fetch_page(
    url: str,
    cache_dir: Optional[str] = None,
    max_retries: int = 3,
    max_age_sec: float = 86400.0,
) -> dict:
    """Fetch a JSON feed page with local file caching, atomic writes, and exponential backoff."""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"page_{url_hash}.json")
        if os.path.isfile(cache_path):
            file_age = time.time() - os.path.getmtime(cache_path)
            if file_age <= max_age_sec:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning("Corrupt cache file %s: %s; treating as miss", cache_path, e)

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if cache_path:
                    tmp_cache = f"{cache_path}.tmp.{os.getpid()}"
                    with open(tmp_cache, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                    os.replace(tmp_cache, cache_path)
                return data
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                backoff = (attempt + 1) * 2.0
                logger.warning("Error fetching %s: %s. Retrying in %.1fs...", url, e, backoff)
                time.sleep(backoff)
            else:
                raise last_error
    raise RuntimeError(f"Failed to fetch {url}")


def crawl_live_feed(
    cache_dir: Optional[str] = None,
    delay: float = 1.0,
    page_size: int = 150,
) -> Tuple[List[ParsedPost], int]:
    """Crawl Blogger feed from newest to oldest using published-max cursor pagination."""
    posts: List[ParsedPost] = []
    seen_ids: Set[str] = set()
    total_results: Optional[int] = None
    published_max: Optional[str] = None

    page_num = 0
    while True:
        page_num += 1
        params = {
            "alt": "json",
            "orderby": "published",
            "max-results": str(page_size),
        }
        if published_max:
            params["published-max"] = published_max

        url = f"{BASE_FEED_URL}?{urllib.parse.urlencode(params)}"
        logger.info("Fetching page %d (cursor: %s)...", page_num, published_max or "start")

        data = fetch_page(url, cache_dir=cache_dir)
        feed = data.get("feed", {})
        if total_results is None:
            total_raw = feed.get("openSearch$totalResults", {}).get("$t")
            if total_raw:
                total_results = int(total_raw)
                logger.info("Blogger feed reports total results: %d", total_results)

        entries = feed.get("entry", [])
        if not entries:
            logger.info("No more entries found. Crawl complete.")
            break

        new_on_page = 0
        oldest_pub = None
        for entry in entries:
            post = parse_entry(entry)
            if post.blogger_id in seen_ids:
                continue
            seen_ids.add(post.blogger_id)
            posts.append(post)
            new_on_page += 1
            if post.published:
                if oldest_pub is None or post.published < oldest_pub:
                    oldest_pub = post.published

        logger.info(
            "Page %d: %d entries (%d new). Total collected: %d",
            page_num,
            len(entries),
            new_on_page,
            len(posts),
        )

        if not oldest_pub or oldest_pub == published_max or new_on_page == 0:
            logger.info("Pagination boundary reached.")
            break

        published_max = oldest_pub
        if delay > 0:
            time.sleep(delay)

    return posts, total_results or len(posts)


def load_from_fixtures(fixtures_dir: str) -> Tuple[List[ParsedPost], int]:
    """Load feed entries from local JSON fixture files."""
    posts: List[ParsedPost] = []
    seen_ids: Set[str] = set()

    files = sorted([f for f in os.listdir(fixtures_dir) if f.endswith(".json")])
    for fname in files:
        fpath = os.path.join(fixtures_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            entries = data.get("feed", {}).get("entry", [])
            if not entries and "entry" in data:
                entries = [data["entry"]]
            for entry in entries:
                post = parse_entry(entry)
                if post.blogger_id not in seen_ids:
                    seen_ids.add(post.blogger_id)
                    posts.append(post)

    return posts, len(posts)


def merge_keywords(conn: sqlite3.Connection, keywords_dir: str) -> None:
    """Read .jsonl files in keywords_dir and insert keywords into image_keywords table."""
    if not os.path.isdir(keywords_dir):
        return

    cursor = conn.cursor()
    cursor.execute("SELECT id, key FROM images")
    key_to_id = {row[1]: row[0] for row in cursor.fetchall()}

    files = sorted([f for f in os.listdir(keywords_dir) if f.endswith(".jsonl")])
    total_inserted = 0
    unknown_keys = 0

    for fname in files:
        fpath = os.path.join(keywords_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    img_key = record.get("key", "")
                    keywords = record.get("keywords", [])
                    source = record.get("source", fname)
                    weight = float(record.get("weight", 1.0))

                    if img_key not in key_to_id:
                        unknown_keys += 1
                        continue

                    img_id = key_to_id[img_key]
                    for kw in keywords:
                        kw_clean = kw.strip()
                        if kw_clean:
                            cursor.execute(
                                """
                                INSERT OR REPLACE INTO image_keywords (image_id, keyword, source, weight)
                                VALUES (?, ?, ?, ?)
                                """,
                                (img_id, kw_clean, source, weight),
                            )
                            total_inserted += 1
                except Exception as e:
                    logger.warning("Error parsing line in %s: %s", fname, e)

    conn.commit()
    logger.info("Merged %d keywords (%d unknown keys skipped)", total_inserted, unknown_keys)


def compare_with_old(conn: sqlite3.Connection, old_path: str) -> None:
    """Compare new database with old SQLite database and print diff."""
    if not os.path.isfile(old_path):
        logger.info("Previous catalog not found at %s. Skipping comparison.", old_path)
        return

    old_conn = sqlite3.connect(old_path)
    try:
        old_cur = old_conn.cursor()
        new_cur = conn.cursor()

        try:
            old_cur.execute("SELECT blogger_id, updated FROM posts")
            old_posts = {row[0]: row[1] for row in old_cur.fetchall()}
        except sqlite3.Error as e:
            logger.warning("Cannot read previous catalog %s: %s. Skipping comparison.", old_path, e)
            return

        new_cur.execute("SELECT blogger_id, updated FROM posts")
        new_posts = {row[0]: row[1] for row in new_cur.fetchall()}

        added_ids = set(new_posts.keys()) - set(old_posts.keys())
        updated_ids = {
            bid for bid in set(new_posts.keys()) & set(old_posts.keys())
            if new_posts[bid] > old_posts[bid]
        }

        print("=== Catalog Comparison ===")
        print(f"Old database: {len(old_posts):,} posts")
        print(f"New database: {len(new_posts):,} posts")
        print(f"Posts added:   {len(added_ids):,}")
        print(f"Posts updated: {len(updated_ids):,}")
        print("==========================")
    finally:
        old_conn.close()


def validate_catalog(
    out_path: str,
    total_results: int,
    require_probes: bool = True,
    max_size_mb: float = 40.0,
) -> None:
    """Execute validation gates against generated catalog. Raises RuntimeError on gate failure."""
    logger.info("Running validation gates on %s...", out_path)

    # Gate 1: Integrity check
    conn = sqlite3.connect(out_path)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        row = cursor.fetchone()
        if not row or row[0] != "ok":
            raise RuntimeError(f"Gate 1 Failed: PRAGMA integrity_check returned: {row}")

        # Gate 2: Completeness check
        cursor.execute("SELECT COUNT(*) FROM posts")
        post_count = cursor.fetchone()[0]
        min_allowed = int(total_results * 0.98) if total_results > 0 else 1
        if post_count < min_allowed:
            raise RuntimeError(
                f"Gate 2 Failed: post_count {post_count} is less than 98% of total {total_results} ({min_allowed})"
            )

        cursor.execute("SELECT COUNT(*) FROM images")
        image_count = cursor.fetchone()[0]
        if image_count < post_count:
            raise RuntimeError(
                f"Gate 2 Failed: image_count {image_count} is less than post_count {post_count}"
            )
    finally:
        conn.close()

    # Gate 3: In-memory store and probe verification
    store = CatalogStore(out_path)
    index = store.get_index()
    if index is None:
        raise RuntimeError("Gate 3 Failed: CatalogStore.get_index() returned None")

    if require_probes:
        probes = ["猫", "りんご", "走る"]
        for probe in probes:
            results = index.search(probe)
            if not results:
                raise RuntimeError(f"Gate 3 Failed: Probe search for '{probe}' returned 0 results")
            logger.info("Probe search for '%s': %d results found", probe, len(results))

    # Gate 4: File size limit check
    size_bytes = os.path.getsize(out_path)
    size_mb = size_bytes / (1024.0 * 1024.0)
    logger.info("Catalog file size: %.2f MB (max allowed: %.1f MB)", size_mb, max_size_mb)
    if size_mb > max_size_mb:
        raise RuntimeError(f"Gate 4 Failed: Catalog size {size_mb:.2f} MB exceeds {max_size_mb:.1f} MB")

    logger.info("All validation gates passed successfully!")


def main() -> None:
    """CLI entrypoint for building and validating the Irasutoya SQLite catalog."""
    parser = argparse.ArgumentParser(description="Build offline Irasutoya SQLite catalog.")
    parser.add_argument("--out", required=True, help="Destination path for catalog.sqlite")
    parser.add_argument("--delay", type=float, default=1.0, help="Polite delay between HTTP requests in seconds")
    parser.add_argument("--cache", default=".catalog_cache", help="Directory for HTTP response cache")
    parser.add_argument("--keywords", help="Directory containing keyword .jsonl enrichment files")
    parser.add_argument("--from-fixtures", help="Load entries from local JSON fixtures directory instead of crawling")
    parser.add_argument("--compare-with", help="Path to previous SQLite catalog to compare against")
    parser.add_argument("--skip-probes", action="store_true", help="Skip probe verification (for small fixture runs)")
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if args.from_fixtures:
        fixtures_dir = args.from_fixtures
        logger.info("Loading entries from fixtures directory: %s", fixtures_dir)
        posts, total_results = load_from_fixtures(fixtures_dir)
        require_probes = not args.skip_probes and len(posts) > 100
    else:
        logger.info("Starting live crawl of Irasutoya Blogger feeds...")
        posts, total_results = crawl_live_feed(cache_dir=args.cache, delay=args.delay)
        require_probes = not args.skip_probes

    logger.info("Writing %d posts to SQLite database at %s...", len(posts), out_path)
    if os.path.exists(out_path):
        os.remove(out_path)

    conn = sqlite3.connect(out_path)
    try:
        write_posts(conn, posts, source_total_results=total_results)
        if args.keywords:
            merge_keywords(conn, args.keywords)

        if args.compare_with:
            compare_with_old(conn, args.compare_with)

        logger.info("Optimizing database with VACUUM and journal_mode=DELETE...")
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()

    # Compute sha256 checksum
    hasher = hashlib.sha256()
    with open(out_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    digest = hasher.hexdigest()

    sha_path = f"{out_path}.sha256"
    basename = os.path.basename(out_path)
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {basename}\n")
    logger.info("Wrote SHA256 checksum to %s", sha_path)

    # Validate output
    validate_catalog(out_path, total_results, require_probes=require_probes)


if __name__ == "__main__":
    main()
