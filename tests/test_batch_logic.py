"""Unit tests for batch processing logic, filename generation, and note evaluation."""
import unittest
import urllib.parse
from typing import Any, List
from unittest.mock import MagicMock, patch

try:
    import aqt  # noqa: F401
    HAS_AQT = True
except ImportError:
    HAS_AQT = False

from addon.batch import (
    BatchStats,
    make_safe_media_filename,
    process_note_record,
    run_batch_fill,
    safe_extract_note_pos_type,
)
from addon.feed import IrasutoyaImage


class MockNote(dict):
    """Mock dictionary-based note with tag management."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize mock note with empty tags list."""
        super().__init__(*args, **kwargs)
        self.tags: List[str] = []

    def add_tag(self, tag: str) -> None:
        """Add a tag to the mock note if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the mock note if present."""
        if tag in self.tags:
            self.tags.remove(tag)


class MockIrasutoyaClient:
    """Mock Irasutoya client for deterministic unit testing."""

    def __init__(self, search_results=None, image_data=b"PNG_FAKE_DATA", search_mapping=None):
        """Initialize mock client with optional preset search results, binary payload, or per-query mapping."""
        self.search_results = search_results or []
        self.image_data = image_data
        self.search_mapping = search_mapping or {}
        self.searched_queries: List[str] = []

    def search(self, query: str, max_results: int = 3, image_size: str = "s800", orderby: str = "relevance"):
        """Return preset search results."""
        self.searched_queries.append(query)
        if query in self.search_mapping:
            return self.search_mapping[query]
        return self.search_results

    def download_image(self, url: str, timeout: int = 15, *, image=None):
        """Return preset binary image data."""
        return self.image_data


class TestBatchLogic(unittest.TestCase):
    """Test suite for batch processing, filename safety, and skip/overwrite decisions."""

    def setUp(self) -> None:
        """Isolate batch tests from the shipped catalog's LLM rejection data."""
        patcher = patch("addon.batch.is_llm_rejected", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

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
        self.assertEqual(status, "no_match")
        self.assertIsNotNone(mod)
        self.assertEqual(mod["DefinitionPicture"], "")

    def test_batch_stats_summary(self):
        """Verify BatchStats summary message formatting."""
        stats = BatchStats(total=5, added=3, skipped=1, no_match=1)
        self.assertIn("Added: 3", stats.summary_message)
        self.assertIn("Skipped: 1", stats.summary_message)
        self.assertIn("No match: 1", stats.summary_message)

        stats_with_err = BatchStats(total=6, added=2, skipped=1, no_match=1, errors=2, canceled=True)
        self.assertIn("Canceled.", stats_with_err.summary_message)
        self.assertIn("Errors: 2", stats_with_err.summary_message)

        stats_with_details = BatchStats(total=3, errors=1, details=["Offline catalog unavailable"])
        self.assertIn("(Offline catalog unavailable)", stats_with_details.summary_message)

    def test_batch_stats_fallback_summary(self):
        """Verify BatchStats summary formatting when fallback matches occur."""
        stats = BatchStats(total=5, added=3, direct_matches=2, fallback_matches=1, skipped=1, not_found=1)
        self.assertIn("Added: 3 (2 direct, 1 approximate)", stats.summary_message)

    def test_process_note_reading_fallback_and_tagging(self):
        """Verify process_note_record holds back phonetic reading match with review status and tag."""
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

        self.assertEqual(status, "review")
        self.assertIsNotNone(mod)
        self.assertEqual(mod["DefinitionPicture"], "")
        self.assertIn("irasutoya::review", note.tags)

    def test_process_note_tier2_exemplar_auto_fills_and_tags_approximate(self):
        """Verify Tier 2 visual exemplar / deinflection auto-fills and tags irasutoya::approximate."""
        class MockNote(dict):
            """Mock dictionary-based note with tag management."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                """Initialize mock note with empty tags list."""
                super().__init__(*args, **kwargs)
                self.tags: List[str] = []

            def add_tag(self, tag: str) -> None:
                """Add a tag to the mock note if not already present."""
                if tag not in self.tags:
                    self.tags.append(tag)

            def remove_tag(self, tag: str) -> None:
                """Remove a tag from the mock note if present."""
                if tag in self.tags:
                    self.tags.remove(tag)

        note = MockNote({"Expression": "走っている", "DefinitionPicture": ""})

        deinflected_img = IrasutoyaImage(
            title="走る人のイラスト",
            image_url="https://example.com/run.png",
            thumb_url="https://example.com/thumb.png",
            post_title="走る人のイラスト"
        )
        # Primary "走っている" returns empty, deinflection "走る" returns image
        client = MockIrasutoyaClient(search_mapping={"走る": [deinflected_img]})

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
        self.assertIn("irasutoya::approximate", note.tags)
        self.assertIn(urllib.parse.quote("irasutoya_走る_走る人のイラスト.png"), mod["DefinitionPicture"])

    def test_batch_removes_stale_fallback_tag_on_direct_match(self):
        """Verify direct Tier 1 match removes any stale fallback audit tag from previous runs."""
        class MockNote(dict):
            """Mock dictionary-based note with tag management."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                """Initialize mock note with empty tags list."""
                super().__init__(*args, **kwargs)
                self.tags: List[str] = ["irasutoya::fallback_reading", "irasutoya::approximate", "irasutoya::review", "other_tag"]

            def add_tag(self, tag: str) -> None:
                """Add a tag to the mock note if not already present."""
                if tag not in self.tags:
                    self.tags.append(tag)

            def remove_tag(self, tag: str) -> None:
                """Remove a tag from the mock note if present."""
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
        self.assertNotIn("irasutoya::approximate", note.tags)
        self.assertNotIn("irasutoya::review", note.tags)
        self.assertIn("other_tag", note.tags)

    def test_batch_skips_decorative_assets(self):
        """Verify batch processor excludes decorative frames and continues to valid illustration."""
        frame_img = IrasutoyaImage("フレームのイラスト", "https://example.com/f.png", "https://example.com/t.png", "フレーム")
        real_img = IrasutoyaImage("猫のイラスト", "https://example.com/c.png", "https://example.com/t.png", "猫")

        note = {"Expression": "猫", "DefinitionPicture": ""}
        client = MockIrasutoyaClient(search_mapping={
            "猫": [frame_img, real_img]
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

    def test_batch_legacy_path_evaluates_all_candidates(self):
        """Verify legacy search path iterates through generated candidates without artificial cap."""
        from addon.text_utils import generate_search_candidates

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
        self.assertEqual(status, "no_match")
        expected = [
            c[0] for c in generate_search_candidates(
                note["Expression"], note=note, allow_tier2=True, allow_deinflection=True
            )
        ]
        self.assertEqual(queried, expected)


    def test_safe_extract_note_pos_type(self):
        """Verify safe_extract_note_pos_type safely handles malformed notes and extracts POS hints."""
        # Malformed note object that raises on key access
        class FaultyNote:
            def __getitem__(self, item):
                raise RuntimeError("Corrupted note data")

        self.assertIsNone(safe_extract_note_pos_type(FaultyNote()))
        self.assertIsNone(safe_extract_note_pos_type(None))
        self.assertIsNone(safe_extract_note_pos_type({}))

        # Note with valid Yomitan HTML definition
        valid_verb_note = {
            "MainDefinition": '<ol><li><i>(v1, JMdict)</i> to eat</li></ol>'
        }
        self.assertEqual(safe_extract_note_pos_type(valid_verb_note), "verb")

    @unittest.skipUnless(HAS_AQT, "aqt not available in headless test environment")
    def test_run_batch_fill_aborts_when_catalog_missing(self):
        """Verify run_batch_fill aborts cleanly, excluding skipped notes from error count."""
        mock_store = MagicMock()
        mock_store.get_index.return_value = None

        mock_col = MagicMock()
        notes = {
            101: {"Expression": "猫", "DefinitionPicture": ""},  # would process -> error
            102: {"Expression": "犬", "DefinitionPicture": "dog.png"},  # target exists in skip mode -> skipped
        }
        mock_col.get_note.side_effect = lambda nid: notes[nid]

        mock_mw = MagicMock()
        mock_mw.col = mock_col

        def fake_with_progress(task: Any, on_done: Any, **kwargs: Any) -> None:
            """Simulate taskman with_progress execution."""
            res = task()
            fut = MagicMock()
            fut.exception.return_value = None
            fut.result.return_value = res
            on_done(fut)

        mock_mw.taskman.with_progress = fake_with_progress

        class MockQueryOp:
            """Mock Anki QueryOp executing synchronous operations."""

            def __init__(self, parent: Any = None, op: Any = None, success: Any = None) -> None:
                """Initialize mock QueryOp."""
                self.op = op
                self.success_cb = success

            def with_backend(self) -> "MockQueryOp":
                """Return self for chained builder call."""
                return self

            def run_in_background(self) -> None:
                """Execute op synchronously."""
                res = self.op(mock_col)
                if self.success_cb:
                    self.success_cb(res)

        finished_stats: list = []

        with patch("addon.batch.default_store", return_value=mock_store), \
             patch("aqt.operations.QueryOp", MockQueryOp), \
             patch("aqt.mw", mock_mw), \
             patch("aqt.utils.tooltip"):
            run_batch_fill(
                parent_window=None,
                note_ids=[101, 102],
                source_field="Expression",
                target_field="DefinitionPicture",
                overwrite_mode="skip",
                on_finish=lambda s: finished_stats.append(s),
            )

        self.assertEqual(len(finished_stats), 1)
        self.assertEqual(finished_stats[0].total, 2)
        self.assertEqual(finished_stats[0].errors, 1)
        self.assertEqual(finished_stats[0].skipped, 1)
        self.assertEqual(finished_stats[0].added, 0)
        self.assertIn("Offline catalog unavailable", finished_stats[0].summary_message)

    def test_resolve_surface_term_prefers_primary_over_curated(self) -> None:
        """Verify resolve_surface_term picks the candidate marked 'primary' even when curated targets precede it."""
        from addon.batch import resolve_surface_term

        # Curated candidate in Group 1 precedes primary
        candidates = [
            ("怪しい儀式のイラスト", "llm_concept", 1),
            ("儀式", "primary", 1),
            ("ぎしき", "reading", 2),
        ]
        self.assertEqual(resolve_surface_term(candidates, raw_source="儀式"), "儀式")

        # When primary is absent, falls back to cleaned raw_source
        candidates_no_primary = [
            ("怪しい儀式のイラスト", "llm_concept", 1),
            ("ぎしき", "reading", 2),
        ]
        self.assertEqual(resolve_surface_term(candidates_no_primary, raw_source="儀式"), "儀式")

        # When raw_source is empty, falls back to first candidate
        self.assertEqual(resolve_surface_term(candidates_no_primary, raw_source=""), "怪しい儀式のイラスト")
        self.assertEqual(resolve_surface_term([], raw_source=""), "")

    def test_batch_processing_applies_non_visual_tag_on_rejected_term(self) -> None:
        """Verify process_note_record tags rejected terms as irasutoya::non_visual without searching."""
        from addon.batch import TAG_NON_VISUAL, process_note_record

        note = MockNote({"Expression": "外見", "DefinitionPicture": ""})
        mock_client = MockIrasutoyaClient()

        with patch("addon.batch.is_llm_rejected", return_value=True):
            status, modified = process_note_record(
                note=note,
                source_field="Expression",
                target_field="DefinitionPicture",
                overwrite_mode="skip",
                client=mock_client,
                write_media_func=lambda fn, data: fn,
            )
            self.assertEqual(status, "non_visual")
            self.assertIn(TAG_NON_VISUAL, note.tags)
            self.assertEqual(len(mock_client.searched_queries), 0)


if __name__ == "__main__":
    unittest.main()
