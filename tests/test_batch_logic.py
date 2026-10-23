"""Unit tests for batch processing logic, filename generation, and note evaluation."""
import unittest
import urllib.parse

from addon.batch import BatchStats, make_safe_media_filename, process_note_record
from addon.feed import IrasutoyaImage


class MockIrasutoyaClient:
    """Mock Irasutoya client for deterministic unit testing."""

    def __init__(self, search_results=None, image_data=b"PNG_FAKE_DATA", search_mapping=None):
        """Initialize mock client with optional preset search results, binary payload, or per-query mapping."""
        self.search_results = search_results or []
        self.image_data = image_data
        self.search_mapping = search_mapping or {}

    def search(self, query: str, max_results: int = 3, image_size: str = "s800", orderby: str = "relevance"):
        """Return preset search results."""
        if query in self.search_mapping:
            return self.search_mapping[query]
        return self.search_results

    def download_image(self, url: str, timeout: int = 15):
        """Return preset binary image data."""
        return self.image_data


class TestBatchLogic(unittest.TestCase):
    """Test suite for batch processing, filename safety, and skip/overwrite decisions."""

    def test_safe_media_filename(self):
        """Test sanitizing queries and titles into valid Anki media filenames with proper extension."""
        fn = make_safe_media_filename("猫", "猫の日の女の子/キャラ")
        self.assertTrue(fn.startswith("irasutoya_猫_猫の日の女の子_キャラ"))
        self.assertTrue(fn.endswith(".png"))
        self.assertNotIn("/", fn)
        self.assertNotIn(":", fn)

        # Long filename truncation
        fn_long = make_safe_media_filename("very_long_query_term_" * 4, "very_long_title_" * 4)
        self.assertTrue(fn_long.endswith(".png"))
        self.assertNotIn(".pn.png", fn_long)
        self.assertLessEqual(len(fn_long), 64)

        # Pre-existing .png in title
        fn_ext = make_safe_media_filename("dog", "dog_image.png")
        self.assertTrue(fn_ext.endswith(".png"))
        self.assertNotIn(".png.png", fn_ext)

    def test_process_note_skip_when_not_empty(self):
        """Verify note is skipped when target field is non-empty and overwrite is set to 'skip'."""
        note = {"Expression": "猫", "DefinitionPicture": "<div>existing_img.png</div>"}
        client = MockIrasutoyaClient()
        status, mod = process_note_record(
            note=note,
            source_field="Expression",
            target_field="DefinitionPicture",
            overwrite_mode="skip",
            client=client,
            write_media_func=lambda f, d: f
        )
        self.assertEqual(status, "skipped_not_empty")
        self.assertIsNone(mod)

    def test_process_note_overwrite(self):
        """Verify note target field is overwritten when overwrite mode is set to 'overwrite'."""
        note = {"Expression": "猫", "DefinitionPicture": "old_content"}
        img = IrasutoyaImage(
            title="可愛い猫",
            image_url="https://example.com/cat.png",
            thumb_url="https://example.com/cat_thumb.png",
            post_title="猫のイラスト"
        )
        client = MockIrasutoyaClient(search_results=[img])

        written_files = []
        def mock_writer(filename, data):
            """Mock Anki media writer storing write arguments in memory."""
            written_files.append((filename, data))
            return "stored_" + filename

        status, mod = process_note_record(
            note=note,
            source_field="Expression",
            target_field="DefinitionPicture",
            overwrite_mode="overwrite",
            client=client,
            write_media_func=mock_writer
        )

        self.assertEqual(status, "added")
        self.assertIsNotNone(mod)
        self.assertIn(f'<img src="{urllib.parse.quote("stored_irasutoya_猫_可愛い猫.png")}">', mod["DefinitionPicture"])
        self.assertEqual(len(written_files), 1)

    def test_process_note_not_found(self):
        """Verify appropriate status is returned when client yields no search results."""
        note = {"Expression": "unknown_word_xyz", "DefinitionPicture": ""}
        client = MockIrasutoyaClient(search_results=[])
        status, mod = process_note_record(
            note=note,
            source_field="Expression",
            target_field="DefinitionPicture",
            overwrite_mode="skip",
            client=client,
            write_media_func=lambda f, d: f
        )
        self.assertEqual(status, "not_found")
        self.assertIsNone(mod)

    def test_batch_stats_summary(self):
        """Verify BatchStats summary message formatting."""
        stats = BatchStats(total=5, added=3, skipped=1, not_found=1)
        self.assertIn("Added: 3", stats.summary_message)
        self.assertIn("Skipped: 1", stats.summary_message)
        self.assertIn("Not found: 1", stats.summary_message)

        stats_with_err = BatchStats(total=6, added=2, skipped=1, not_found=1, errors=2, canceled=True)
        self.assertIn("Canceled.", stats_with_err.summary_message)
        self.assertIn("Errors: 2", stats_with_err.summary_message)

    def test_batch_stats_fallback_summary(self):
        """Verify BatchStats summary formatting when fallback matches occur."""
        stats = BatchStats(total=5, added=3, direct_matches=2, fallback_matches=1, skipped=1, not_found=1)
        self.assertIn("Added: 3 (2 direct, 1 via fallback)", stats.summary_message)

    def test_process_note_reading_fallback_and_tagging(self):
        """Verify process_note_record falls back to note reading field and applies audit tag."""
        class MockNote(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tags = []

            def add_tag(self, tag):
                if tag not in self.tags:
                    self.tags.append(tag)

            def remove_tag(self, tag):
                if tag in self.tags:
                    self.tags.remove(tag)

        note = MockNote({"Expression": "其の儘", "Reading": "そのまま", "DefinitionPicture": ""})

        reading_img = IrasutoyaImage(
            title="そのままのイラスト",
            image_url="https://example.com/sonomama.png",
            thumb_url="https://example.com/thumb.png",
            post_title="そのままのイラスト"
        )
        # Primary query returns empty, fallback query "そのまま" returns image
        client = MockIrasutoyaClient(search_mapping={"そのまま": [reading_img]})

        status, mod = process_note_record(
            note=note,
            source_field="Expression",
            target_field="DefinitionPicture",
            overwrite_mode="skip",
            client=client,
            write_media_func=lambda f, d: f,
            allow_tier2=True
        )

        self.assertEqual(status, "added")
        self.assertIsNotNone(mod)
        self.assertIn("irasutoya::fallback_reading", note.tags)
        self.assertIn(urllib.parse.quote("irasutoya_そのまま_そのままのイラスト.png"), mod["DefinitionPicture"])

    def test_batch_removes_stale_fallback_tag_on_direct_match(self):
        """Verify direct Tier 1 match removes any stale fallback audit tag from previous runs."""
        class MockNote(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tags = ["irasutoya::fallback_reading", "other_tag"]

            def add_tag(self, tag):
                if tag not in self.tags:
                    self.tags.append(tag)

            def remove_tag(self, tag):
                if tag in self.tags:
                    self.tags.remove(tag)

        note = MockNote({"Expression": "猫", "DefinitionPicture": "<img src='old.png'>"})
        cat_img = IrasutoyaImage(
            title="かわいい猫のイラスト",
            image_url="https://example.com/cat.png",
            thumb_url="https://example.com/thumb.png",
            post_title="猫のイラスト"
        )
        client = MockIrasutoyaClient(search_mapping={"猫": [cat_img]})

        status, mod = process_note_record(
            note=note,
            source_field="Expression",
            target_field="DefinitionPicture",
            overwrite_mode="overwrite",
            client=client,
            write_media_func=lambda f, d: f,
            allow_tier2=True
        )

        self.assertEqual(status, "added")
        self.assertNotIn("irasutoya::fallback_reading", note.tags)
        self.assertIn("other_tag", note.tags)

    def test_batch_skips_decorative_assets(self):
        """Verify batch processor excludes decorative frames and continues to fallback illustration."""
        frame_img = IrasutoyaImage("フレームのイラスト", "https://example.com/f.png", "https://example.com/t.png", "フレーム")
        real_img = IrasutoyaImage("猫のイラスト", "https://example.com/c.png", "https://example.com/t.png", "猫")

        note = {"Expression": "其の儘", "Reading": "猫", "DefinitionPicture": ""}
        client = MockIrasutoyaClient(search_mapping={
            "其の儘": [frame_img],
            "猫": [real_img]
        })

        status, mod = process_note_record(
            note=note,
            source_field="Expression",
            target_field="DefinitionPicture",
            overwrite_mode="skip",
            client=client,
            write_media_func=lambda f, d: f,
            allow_tier2=True
        )
        self.assertEqual(status, "added")
        self.assertIn("猫のイラスト", urllib.parse.unquote(mod["DefinitionPicture"]))

    def test_batch_legacy_path_caps_at_max_candidates(self):
        """Verify legacy search path does not iterate past BATCH_MAX_CANDIDATES queries."""
        queried = []

        class QueryTrackingClient(MockIrasutoyaClient):
            def search(self, query: str, **kwargs):
                queried.append(query)
                return []

        note = {"Expression": "1杯", "Reading": "いっぱい", "Kana": "イッパイ", "DefinitionPicture": ""}
        client = QueryTrackingClient()

        status, mod = process_note_record(
            note=note,
            source_field="Expression",
            target_field="DefinitionPicture",
            overwrite_mode="skip",
            client=client,
            write_media_func=lambda f, d: f,
            allow_tier2=True
        )
        self.assertEqual(status, "not_found")
        self.assertLessEqual(len(queried), 4)


if __name__ == "__main__":
    unittest.main()
