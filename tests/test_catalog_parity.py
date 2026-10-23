"""Parity tests ensuring catalog index results match live Blogger feed parser output."""
import os
import sqlite3
import tempfile
import unittest

from addon.catalog import CatalogStore, write_posts
from addon.feed import parse_entries, parse_entry


class TestCatalogParity(unittest.TestCase):
    """Verify complete output parity between live Blogger feed parsing and offline catalog search."""

    def setUp(self):
        """Create a temporary directory for test catalog."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "catalog.sqlite")

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def _sample_feed_entries(self):
        """Generate realistic feed entries covering single-image, multi-image, and category metadata."""
        return [
            {
                "id": {"$t": "tag:blogger.com,1999:blog-1.post-81234567"},
                "title": {"$t": "可愛い柴犬のイラスト"},
                "summary": {"$t": "柴犬が元気にお散歩しているイラストです。"},
                "published": {"$t": "2026-03-01T10:00:00Z"},
                "updated": {"$t": "2026-03-01T12:00:00Z"},
                "category": [
                    {"term": "犬"},
                    {"term": "ペット"},
                    {"term": "動物"},
                ],
                "content": {
                    "$t": (
                        '<a href="https://1.bp.blogspot.com/s1600/shiba_inu.png">'
                        '<img alt="走る柴犬" src="https://1.bp.blogspot.com/s320/shiba_inu.png" />'
                        '</a>'
                    )
                },
            },
            {
                "id": {"$t": "tag:blogger.com,1999:blog-1.post-99887766"},
                "title": {"$t": "果物のイラストセット"},
                "summary": {"$t": "色々な果物のイラスト詰め合わせです。"},
                "published": {"$t": "2026-03-02T10:00:00Z"},
                "updated": {"$t": "2026-03-02T14:00:00Z"},
                "category": [
                    {"term": "果物"},
                    {"term": "食べ物"},
                ],
                "content": {
                    "$t": (
                        '<div class="separator">'
                        '<a href="https://2.bp.blogspot.com/s1600/apple_fruit.png">'
                        '<img alt="りんご" src="https://2.bp.blogspot.com/s320/apple_fruit.png" />'
                        '</a>'
                        '</div>'
                        '<div class="separator">'
                        '<a href="https://2.bp.blogspot.com/s1600/orange_fruit.png">'
                        '<img alt="みかん" src="https://2.bp.blogspot.com/s320/orange_fruit.png" />'
                        '</a>'
                        '</div>'
                    )
                },
            },
        ]

    def test_live_vs_catalog_parity(self):
        """Verify CatalogIndex output matches parse_entries live parsing output exactly."""
        raw_entries = self._sample_feed_entries()

        # Live feed parse
        live_images = parse_entries(raw_entries, image_size="s800")
        self.assertEqual(len(live_images), 3)

        # Build SQLite catalog from the same entries
        parsed_posts = [parse_entry(e) for e in raw_entries]
        conn = sqlite3.connect(self.db_path)
        write_posts(conn, parsed_posts)
        conn.close()

        # Query catalog
        store = CatalogStore(self.db_path)
        index = store.get_index()
        self.assertIsNotNone(index)

        # Search for each illustration title and verify exact equality
        for live_img in live_images:
            cat_results = index.search(live_img.title, image_size="s800")
            self.assertGreaterEqual(len(cat_results), 1)

            # Find matching result
            matched = next((c for c in cat_results if c.image_url == live_img.image_url), None)
            self.assertIsNotNone(matched)
            self.assertEqual(matched.title, live_img.title)
            self.assertEqual(matched.image_url, live_img.image_url)
            self.assertEqual(matched.thumb_url, live_img.thumb_url)
            self.assertEqual(matched.post_title, live_img.post_title)
            self.assertEqual(matched.description, live_img.description)
            self.assertEqual(matched.is_single_image_post, live_img.is_single_image_post)
            self.assertEqual(matched.post_id, live_img.post_id)
            self.assertEqual(matched.categories, live_img.categories)
