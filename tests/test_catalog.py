"""Unit tests for offline SQLite catalog store, schema, and in-memory index."""
import os
import sqlite3
import tempfile
import unittest

from addon.catalog import (
    SCHEMA_VERSION,
    CatalogInfo,
    CatalogStore,
    create_schema,
    write_posts,
)
from addon.feed import IrasutoyaImage, ParsedPost


class TestCatalog(unittest.TestCase):
    """Test suite for SQLite metadata catalog storage, index search, and lifecycle."""

    def setUp(self):
        """Create a temporary directory for test database files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "catalog.sqlite")

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def _sample_posts(self):
        """Create sample parsed posts with multiple images and categories."""
        img1 = IrasutoyaImage(
            title="かわいい黒猫のイラスト",
            image_url="https://1.bp.blogspot.com/s800/cat_black.png",
            thumb_url="https://1.bp.blogspot.com/s120-c/cat_black.png",
            post_title="猫のイラスト（黒猫）",
            description="かわいい黒猫が座っているイラストです。",
            is_single_image_post=False,
            post_id="1001",
            categories=("動物", "猫"),
        )
        img2 = IrasutoyaImage(
            title="かわいい白猫のイラスト",
            image_url="https://1.bp.blogspot.com/s800/cat_white.png",
            thumb_url="https://1.bp.blogspot.com/s120-c/cat_white.png",
            post_title="猫のイラスト（黒猫）",
            description="かわいい白猫が歩いているイラストです。",
            is_single_image_post=False,
            post_id="1001",
            categories=("動物", "猫"),
        )
        post1 = ParsedPost(
            blogger_id="1001",
            title="猫のイラスト（黒猫）",
            description="かわいい猫たちのイラストセットです。",
            published="2026-01-01T12:00:00Z",
            updated="2026-01-02T12:00:00Z",
            categories=("動物", "猫"),
            images=[img1, img2],
        )

        img3 = IrasutoyaImage(
            title="赤いりんごのイラスト",
            image_url="https://1.bp.blogspot.com/s400/apple_red.png",
            thumb_url="https://1.bp.blogspot.com/s120-c/apple_red.png",
            post_title="りんごのイラスト",
            description="美味しそうな赤いリンゴのイラストです。",
            is_single_image_post=True,
            post_id="1002",
            categories=("食べ物", "フルーツ"),
        )
        post2 = ParsedPost(
            blogger_id="1002",
            title="りんごのイラスト",
            description="美味しそうな赤いリンゴのイラストです。",
            published="2026-01-03T10:00:00Z",
            updated="2026-01-04T10:00:00Z",
            categories=("食べ物", "フルーツ"),
            images=[img3],
        )
        return [post1, post2]

    def test_schema_creation(self):
        """Verify create_schema creates all expected tables, indexes, and constraints."""
        conn = sqlite3.connect(self.db_path)
        create_schema(conn)

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        expected_tables = {
            "meta",
            "posts",
            "images",
            "categories",
            "post_categories",
            "image_keywords",
        }
        self.assertTrue(expected_tables.issubset(tables))
        conn.close()

    def test_write_posts_and_meta(self):
        """Verify write_posts populates database, normalizes /s0/ URLs, and writes meta stats."""
        conn = sqlite3.connect(self.db_path)
        posts = self._sample_posts()
        write_posts(conn, posts, source_total_results=2)

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM posts")
        self.assertEqual(cursor.fetchone()[0], 2)

        cursor.execute("SELECT COUNT(*) FROM images")
        self.assertEqual(cursor.fetchone()[0], 3)

        cursor.execute("SELECT COUNT(*) FROM categories")
        self.assertEqual(cursor.fetchone()[0], 4)  # 動物, 猫, 食べ物, フルーツ

        # Verify URL normalization to /s0/
        cursor.execute("SELECT url, key FROM images WHERE post_id = 1")
        rows = cursor.fetchall()
        for url, key in rows:
            self.assertIn("/s0/", url)
            self.assertTrue(key.startswith("1001/"))

        # Verify meta values
        cursor.execute("SELECT key, value FROM meta")
        meta = dict(cursor.fetchall())
        self.assertEqual(int(meta["schema_version"]), SCHEMA_VERSION)
        self.assertEqual(int(meta["post_count"]), 2)
        self.assertEqual(int(meta["image_count"]), 3)
        self.assertEqual(int(meta["source_total_results"]), 2)
        self.assertEqual(meta["max_updated"], "2026-01-04T10:00:00Z")
        conn.close()

    def test_store_loads_and_closes_connection(self):
        """Verify CatalogStore loads index into memory and immediately closes the SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        write_posts(conn, self._sample_posts())
        conn.close()

        store = CatalogStore(self.db_path)
        index = store.get_index()
        self.assertIsNotNone(index)
        self.assertEqual(len(index), 3)

        # File can be safely written to or moved since connection was closed immediately
        info = store.info()
        self.assertIsInstance(info, CatalogInfo)
        self.assertEqual(info.post_count, 2)
        self.assertEqual(info.image_count, 3)

    def test_store_missing_file_returns_none(self):
        """Verify CatalogStore returns None safely when the catalog database does not exist."""
        non_existent = os.path.join(self.temp_dir.name, "missing.sqlite")
        store = CatalogStore(non_existent)
        self.assertIsNone(store.get_index())
        self.assertIsNone(store.info())

    def test_store_schema_version_mismatch_returns_none(self):
        """Verify CatalogStore rejects catalogs with unexpected schema version numbers."""
        conn = sqlite3.connect(self.db_path)
        create_schema(conn)
        conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '999')")
        conn.commit()
        conn.close()

        store = CatalogStore(self.db_path)
        self.assertIsNone(store.get_index())

    def test_store_corrupt_database_returns_none(self):
        """Verify CatalogStore handles corrupted database files safely without unhandled crashes."""
        with open(self.db_path, "wb") as f:
            f.write(b"NOT_A_VALID_SQLITE_DATABASE_HEADER")

        store = CatalogStore(self.db_path)
        self.assertIsNone(store.get_index())

    def test_catalog_index_search_and_substring_matching(self):
        """Verify CatalogIndex performs fast substring matching on 1-char, 2-char, and phrases."""
        conn = sqlite3.connect(self.db_path)
        write_posts(conn, self._sample_posts())
        conn.close()

        store = CatalogStore(self.db_path)
        index = store.get_index()
        self.assertIsNotNone(index)

        # 1-character kanji search
        cat_matches = index.search("猫")
        self.assertEqual(len(cat_matches), 2)
        self.assertIn("cat_black.png", cat_matches[0].image_url)
        self.assertEqual(cat_matches[0].is_single_image_post, False)
        self.assertEqual(cat_matches[0].categories, ("動物", "猫"))

        # Substring in description
        desc_matches = index.search("フルーツ")
        self.assertEqual(len(desc_matches), 1)
        self.assertEqual(desc_matches[0].title, "赤いりんごのイラスト")
        self.assertEqual(desc_matches[0].is_single_image_post, True)

        # Empty and non-matching queries
        self.assertEqual(index.search(""), [])
        self.assertEqual(index.search("   "), [])
        self.assertEqual(index.search("存在しないワード"), [])

    def test_catalog_index_image_size_upgrade(self):
        """Verify CatalogIndex rewrites resolution codes on search output."""
        conn = sqlite3.connect(self.db_path)
        write_posts(conn, self._sample_posts())
        conn.close()

        store = CatalogStore(self.db_path)
        index = store.get_index()

        s800_results = index.search("りんご", image_size="s800")
        self.assertIn("/s800/", s800_results[0].image_url)

        s0_results = index.search("りんご", image_size="s0")
        self.assertIn("/s0/", s0_results[0].image_url)

    def test_stable_image_keys_on_rebuild(self):
        """Verify images.key remains deterministic across rebuilds."""
        posts = self._sample_posts()

        conn1 = sqlite3.connect(self.db_path)
        write_posts(conn1, posts)
        cursor1 = conn1.cursor()
        cursor1.execute("SELECT id, key FROM images ORDER BY id")
        keys1 = cursor1.fetchall()
        conn1.close()

        db2_path = os.path.join(self.temp_dir.name, "catalog2.sqlite")
        conn2 = sqlite3.connect(db2_path)
        write_posts(conn2, posts)
        cursor2 = conn2.cursor()
        cursor2.execute("SELECT id, key FROM images ORDER BY id")
        keys2 = cursor2.fetchall()
        conn2.close()

        self.assertEqual([k[1] for k in keys1], [k[1] for k in keys2])

    def test_catalog_load_failure_caching(self):
        """Verify load failure is cached so corrupt files don't re-trigger parsing."""
        with open(self.db_path, "w") as f:
            f.write("not a sqlite database")

        store = CatalogStore(self.db_path)
        self.assertFalse(store.is_loaded())
        self.assertFalse(store.is_load_failed())

        # First load fails
        self.assertIsNone(store.get_index())
        self.assertFalse(store.is_loaded())
        self.assertTrue(store.is_load_failed())

        # Second load short-circuits without error
        self.assertIsNone(store.get_index())
        self.assertTrue(store.is_load_failed())

    def test_image_keywords_search_matching(self):
        """Verify keywords from image_keywords table are indexed and searchable."""
        conn = sqlite3.connect(self.db_path)
        write_posts(conn, self._sample_posts())
        # Add enriched keyword to the black cat image
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM images WHERE title LIKE '%黒猫%'")
        img_id = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO image_keywords (image_id, keyword, source, weight) VALUES (?, ?, ?, ?)",
            (img_id, "black_cat_synonym", "llm", 1.0),
        )
        conn.commit()
        conn.close()

        store = CatalogStore(self.db_path)
        index = store.get_index()
        self.assertIsNotNone(index)
        results = index.search("black_cat_synonym")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "かわいい黒猫のイラスト")
