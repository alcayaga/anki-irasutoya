"""Offline SQLite metadata catalog store and in-memory search index for Irasutoya."""
import datetime
import logging
import os
import sqlite3
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .feed import IrasutoyaImage, ParsedPost, to_thumbnail_url, upgrade_image_url

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CatalogInfo:
    """Metadata summary of the bundled catalog."""

    built_at: str
    post_count: int
    image_count: int
    schema_version: int
    max_updated: str


@dataclass(frozen=True)
class _CatalogImageRecord:
    """Internal lightweight record representing a single catalog illustration image."""

    title: str
    url_s0: str
    post_title: str
    description: str
    is_single: bool
    post_id: str
    categories: Tuple[str, ...]
    haystack: str


class CatalogIndex:
    """Immutable, in-memory index for fast substring search across Irasutoya metadata."""

    def __init__(self, records: Sequence[_CatalogImageRecord]) -> None:
        """Initialize index with precomputed image records."""
        self._records = tuple(records)

    def search(self, query: str, image_size: str = "s800") -> List[IrasutoyaImage]:
        """Search illustrations by query keyword substring matching against title/post_title/description."""
        clean_q = query.strip()
        if not clean_q:
            return []

        matching: List[IrasutoyaImage] = []
        for rec in self._records:
            if clean_q in rec.haystack:
                matching.append(
                    IrasutoyaImage(
                        title=rec.title,
                        image_url=upgrade_image_url(rec.url_s0, image_size),
                        thumb_url=to_thumbnail_url(rec.url_s0, "s120-c"),
                        post_title=rec.post_title,
                        description=rec.description,
                        is_single_image_post=rec.is_single,
                        post_id=rec.post_id,
                        categories=rec.categories,
                    )
                )
        return matching

    def __len__(self) -> int:
        """Return the total number of images indexed."""
        return len(self._records)


class CatalogStore:
    """Manages reading and caching the immutable SQLite catalog."""

    def __init__(self, path: str) -> None:
        """Initialize catalog store pointing to a SQLite database file."""
        self.path = os.path.abspath(path)
        self._lock = threading.Lock()
        self._index: Optional[CatalogIndex] = None
        self._info: Optional[CatalogInfo] = None

    def get_index(self) -> Optional[CatalogIndex]:
        """Lazy, load all records into memory once under a lock, then immediately close connection.

        Returns None if missing, corrupt, or schema version mismatch.
        """
        if self._index is not None:
            return self._index

        with self._lock:
            if self._index is not None:
                return self._index

            if not os.path.isfile(self.path):
                return None

            try:
                # Open read-only and immutable to avoid any write locks
                uri_path = urllib.request.pathname2url(self.path)
                try:
                    conn = sqlite3.connect(f"file:{uri_path}?mode=ro&immutable=1", uri=True)
                except Exception:
                    conn = sqlite3.connect(self.path)

                try:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()

                    # Check schema version
                    cursor.execute("SELECT value FROM meta WHERE key = 'schema_version'")
                    row = cursor.fetchone()
                    if not row or int(row["value"]) != SCHEMA_VERSION:
                        logger.warning(
                            "Catalog schema mismatch in %s: expected %d, got %s",
                            self.path,
                            SCHEMA_VERSION,
                            row["value"] if row else "missing",
                        )
                        return None

                    # Cache info
                    cursor.execute("SELECT key, value FROM meta")
                    meta_dict = {r["key"]: r["value"] for r in cursor.fetchall()}
                    self._info = CatalogInfo(
                        built_at=meta_dict.get("built_at", ""),
                        post_count=int(meta_dict.get("post_count", 0)),
                        image_count=int(meta_dict.get("image_count", 0)),
                        schema_version=int(meta_dict.get("schema_version", SCHEMA_VERSION)),
                        max_updated=meta_dict.get("max_updated", ""),
                    )

                    # Query all images joined with post details and aggregated categories
                    cursor.execute(
                        """
                        SELECT
                            i.id,
                            p.blogger_id,
                            i.title,
                            i.url,
                            p.title AS post_title,
                            p.description,
                            p.image_count,
                            GROUP_CONCAT(c.name, '\x1f') AS categories
                        FROM images i
                        JOIN posts p ON i.post_id = p.id
                        LEFT JOIN post_categories pc ON pc.post_id = p.id
                        LEFT JOIN categories c ON pc.category_id = c.id
                        GROUP BY i.id
                        ORDER BY p.id ASC, i.ord ASC
                        """
                    )
                    rows = cursor.fetchall()

                    records: List[_CatalogImageRecord] = []
                    for r in rows:
                        title = r["title"]
                        post_title = r["post_title"]
                        desc = r["description"] or ""
                        cats_raw = r["categories"]
                        cats = tuple(cats_raw.split("\x1f")) if cats_raw else ()
                        haystack = f"{title}\x00{post_title}\x00{desc}\x00{' '.join(cats)}"

                        records.append(
                            _CatalogImageRecord(
                                title=title,
                                url_s0=r["url"],
                                post_title=post_title,
                                description=desc,
                                is_single=(r["image_count"] == 1),
                                post_id=r["blogger_id"],
                                categories=cats,
                                haystack=haystack,
                            )
                        )

                    self._index = CatalogIndex(records)
                    return self._index
                finally:
                    conn.close()
            except Exception as e:
                logger.warning("Failed to load catalog index from %s: %s", self.path, e)
                return None

    def warm_async(self) -> None:
        """Asynchronously load the catalog into memory in a background daemon thread."""
        if self._index is not None or not os.path.isfile(self.path):
            return
        thread = threading.Thread(target=self.get_index, name="CatalogWarmThread", daemon=True)
        thread.start()

    def info(self) -> Optional[CatalogInfo]:
        """Return catalog metadata summary, reading from disk if not yet loaded."""
        if self._info is not None:
            return self._info
        # Attempting get_index populates _info
        self.get_index()
        return self._info


_DEFAULT_STORE: Optional[CatalogStore] = None
_DEFAULT_STORE_LOCK = threading.Lock()


def default_store() -> CatalogStore:
    """Return the global default CatalogStore instance pointing to addon/data/catalog.sqlite."""
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is None:
            catalog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "catalog.sqlite")
            _DEFAULT_STORE = CatalogStore(catalog_path)
        return _DEFAULT_STORE


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all catalog tables and indexes on an open SQLite connection."""
    cursor = conn.cursor()
    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY,
            blogger_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            published TEXT NOT NULL,
            updated TEXT NOT NULL,
            image_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            post_id INTEGER NOT NULL REFERENCES posts(id),
            key TEXT NOT NULL UNIQUE,
            ord INTEGER NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS post_categories (
            post_id INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            PRIMARY KEY (post_id, category_id)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS image_keywords (
            image_id INTEGER NOT NULL REFERENCES images(id),
            keyword TEXT NOT NULL,
            source TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (image_id, keyword, source)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS images_post ON images(post_id);
        """
    )
    conn.commit()


def write_posts(
    conn: sqlite3.Connection,
    posts: Iterable[ParsedPost],
    source_total_results: Optional[int] = None,
    keyword_sources: Optional[str] = None,
) -> None:
    """Populate catalog database with parsed posts, normalizing image URLs to /s0/ and stamping meta."""
    create_schema(conn)
    cursor = conn.cursor()

    category_id_map: Dict[str, int] = {}
    post_count = 0
    image_count = 0
    max_updated = ""
    seen_image_keys: Set[str] = set()

    for post in posts:
        post_count += 1
        if post.updated and post.updated > max_updated:
            max_updated = post.updated

        cursor.execute(
            """
            INSERT INTO posts (blogger_id, title, description, published, updated, image_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                post.blogger_id,
                post.title,
                post.description,
                post.published,
                post.updated,
                len(post.images),
            ),
        )
        post_id = cursor.lastrowid

        # Insert and link categories
        for cat_name in post.categories:
            cat_clean = cat_name.strip()
            if not cat_clean:
                continue
            if cat_clean not in category_id_map:
                cursor.execute(
                    "INSERT OR IGNORE INTO categories (name) VALUES (?)",
                    (cat_clean,),
                )
                cursor.execute("SELECT id FROM categories WHERE name = ?", (cat_clean,))
                row = cursor.fetchone()
                category_id_map[cat_clean] = row[0]
            cat_id = category_id_map[cat_clean]
            cursor.execute(
                "INSERT OR IGNORE INTO post_categories (post_id, category_id) VALUES (?, ?)",
                (post_id, cat_id),
            )

        # Insert images with normalized /s0/ size and stable key
        for ord_idx, img in enumerate(post.images):
            image_count += 1
            norm_url = upgrade_image_url(img.image_url, "s0")
            parsed_path = urllib.parse.urlparse(norm_url).path
            basename = parsed_path.rstrip("/").split("/")[-1] if parsed_path else f"image_{ord_idx}.png"

            img_key = f"{post.blogger_id}/{basename}"
            if img_key in seen_image_keys:
                img_key = f"{post.blogger_id}/{ord_idx}_{basename}"
            seen_image_keys.add(img_key)

            cursor.execute(
                """
                INSERT INTO images (post_id, key, ord, title, url)
                VALUES (?, ?, ?, ?, ?)
                """,
                (post_id, img_key, ord_idx, img.title, norm_url),
            )

    # Stamp meta table
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta_rows = [
        ("schema_version", str(SCHEMA_VERSION)),
        ("built_at", now_iso),
        ("max_updated", max_updated),
        ("post_count", str(post_count)),
        ("image_count", str(image_count)),
    ]
    if source_total_results is not None:
        meta_rows.append(("source_total_results", str(source_total_results)))
    if keyword_sources is not None:
        meta_rows.append(("keyword_sources", keyword_sources))

    cursor.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        meta_rows,
    )
    conn.commit()
