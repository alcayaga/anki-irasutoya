"""Hermetic tests for scripts/eval_deck_search.py benchmarking logic."""
import io
import unittest
from unittest.mock import MagicMock, patch

from addon.feed import IrasutoyaImage
from scripts.eval_deck_search import (
    detect_source_field,
    fetch_deck_notes,
    legacy_clean_query,
    run_evaluation,
)


class TestEvalDeckSearch(unittest.TestCase):
    """Test suite for deck search evaluation CLI helper functions and report generation."""

    def test_legacy_clean_query(self) -> None:
        """Verify legacy query cleaning replicates previous add-on behavior."""
        self.assertEqual(legacy_clean_query("猫[ねこ]"), "猫")
        self.assertEqual(legacy_clean_query("犬（いぬ）"), "犬")
        self.assertEqual(legacy_clean_query("<b>リンゴ</b>"), "リンゴ")
        self.assertEqual(legacy_clean_query("リンゴ / りんご"), "リンゴ")
        self.assertEqual(legacy_clean_query(""), "")
        self.assertEqual(legacy_clean_query(None), "")

    def test_detect_source_field(self) -> None:
        """Verify auto-detection of source vocabulary fields."""
        self.assertEqual(detect_source_field({"Expression": "猫", "Meaning": "cat"}), "Expression")
        self.assertEqual(detect_source_field({"Word": "犬", "Meaning": "dog"}), "Word")
        self.assertEqual(detect_source_field({"Front": "鳥"}, requested_field="Front"), "Front")
        self.assertEqual(detect_source_field({"Custom": "魚"}, requested_field="NonExistent"), "Custom")

    @patch("scripts.eval_deck_search.anki_invoke")
    def test_fetch_deck_notes_sorting_and_chunking(self, mock_invoke: MagicMock) -> None:
        """Verify fetch_deck_notes retrieves, sorts, and chunks note IDs."""
        mock_invoke.side_effect = [
            # 1. findCards
            [101, 102],
            # 2. cardsInfo
            [
                {"cardId": 101, "note": 1001, "fields": {"FreqSort": {"value": "20"}}},
                {"cardId": 102, "note": 1002, "fields": {"FreqSort": {"value": "10"}}},
            ],
            # 3. notesInfo chunk
            [
                {"noteId": 1002, "fields": {"Expression": {"value": "猫"}}},
                {"noteId": 1001, "fields": {"Expression": {"value": "犬"}}},
            ],
        ]

        notes = fetch_deck_notes("TestDeck", sort_field="FreqSort", limit=10)
        self.assertEqual(len(notes), 2)
        # Should be sorted by FreqSort ascending (1002 with '10' first, then 1001 with '20')
        self.assertEqual(notes[0]["noteId"], 1002)
        self.assertEqual(notes[1]["noteId"], 1001)

    @patch("scripts.eval_deck_search.anki_invoke")
    def test_fetch_deck_notes_custom_query(self, mock_invoke: MagicMock) -> None:
        """Verify fetch_deck_notes queries Anki with custom query when provided."""
        mock_invoke.side_effect = [
            [101],
            [{"cardId": 101, "note": 1001, "fields": {"FreqSort": {"value": "1"}}}],
            [{"noteId": 1001, "fields": {"Expression": {"value": "猫"}}}],
        ]
        notes = fetch_deck_notes("Mining", query="flag:4")
        self.assertEqual(len(notes), 1)
        mock_invoke.assert_any_call("findCards", anki_url="http://localhost:8765", query="flag:4")

    @patch("scripts.eval_deck_search.anki_invoke")
    def test_fetch_deck_notes_501_chunk_boundary(self, mock_invoke: MagicMock) -> None:
        """Verify fetch_deck_notes splits 501 note IDs into chunks of 500 and 1."""
        cards = list(range(1, 502))
        cards_info = [{"cardId": cid, "note": cid, "fields": {"FreqSort": {"value": str(cid)}}} for cid in cards]
        chunk_1 = [{"noteId": nid, "fields": {"Expression": {"value": f"Word{nid}"}}} for nid in range(1, 501)]
        chunk_2 = [{"noteId": 501, "fields": {"Expression": {"value": "Word501"}}}]

        mock_invoke.side_effect = [
            cards,       # findCards
            cards_info,  # cardsInfo
            chunk_1,     # notesInfo batch 1 (500)
            chunk_2,     # notesInfo batch 2 (1)
        ]

        notes = fetch_deck_notes("BigDeck", sort_field="FreqSort", limit=600)
        self.assertEqual(len(notes), 501)
        self.assertEqual(notes[0]["noteId"], 1)
        self.assertEqual(notes[500]["noteId"], 501)
        self.assertEqual(mock_invoke.call_count, 4)
        self.assertEqual(len(mock_invoke.call_args_list[2][1]["notes"]), 500)
        self.assertEqual(len(mock_invoke.call_args_list[3][1]["notes"]), 1)

    @patch("scripts.eval_deck_search.anki_invoke")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_evaluation_catalog_stages(self, mock_stdout: io.StringIO, mock_invoke: MagicMock) -> None:
        """Verify run_evaluation executes in-memory catalog benchmarking and prints incremental stages."""
        mock_invoke.side_effect = [
            [101],
            [{"cardId": 101, "note": 1001, "fields": {"FreqSort": {"value": "1"}}}],
            [{
                "noteId": 1001,
                "fields": {
                    "Expression": {"value": "猫"},
                    "Meaning": {"value": "cat"},
                },
            }],
        ]

        fake_img = IrasutoyaImage(
            title="猫のイラスト",
            image_url="https://1.bp.blogspot.com/s800/cat.png",
            thumb_url="https://1.bp.blogspot.com/s120-c/cat.png",
            post_title="猫のイラスト",
        )

        with patch("scripts.eval_deck_search.default_store") as mock_store:
            mock_index = MagicMock()
            mock_store.return_value.get_index.return_value = mock_index
            mock_index.search.return_value = [fake_img]

            run_evaluation(
                deck_name="TestDeck",
                limit=1,
                delay=0.0,
                backend="catalog",
                track_stages=True,
                show_bilingual_hits=True,
            )

        output = mock_stdout.getvalue()
        self.assertIn("IRASUTOYA SEARCH RECALL BENCHMARK", output)
        self.assertIn("INCREMENTAL RECALL & STRATEGY PROGRESSION", output)
        self.assertIn("1. Legacy (Naive Direct Query)", output)
        self.assertIn("4. Bilingual Gloss (Definition Bridge)", output)

    @patch("scripts.eval_deck_search.anki_invoke")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_evaluation_stage_baseline_error_preserves_failure(
        self, mock_stdout: io.StringIO, mock_invoke: MagicMock
    ) -> None:
        """Verify errors during stage baseline calculation mark the note as failed and preserve failure status."""
        mock_invoke.side_effect = [
            [101],
            [{"cardId": 101, "note": 1001, "fields": {"FreqSort": {"value": "1"}}}],
            [{
                "noteId": 1001,
                "fields": {
                    "Expression": {"value": "猫"},
                    "Meaning": {"value": "cat"},
                },
            }],
        ]

        with patch("scripts.eval_deck_search.default_store") as mock_store:
            mock_client = MagicMock()
            fake_img = IrasutoyaImage(
                title="猫のイラスト",
                image_url="https://1.bp.blogspot.com/s800/cat.png",
                thumb_url="https://1.bp.blogspot.com/s120-c/cat.png",
                post_title="猫のイラスト",
            )
            # Stage baseline search raises an error, but subsequent evaluated search succeeds
            mock_client.search_with_fallbacks.side_effect = [
                RuntimeError("Stage baseline network failure"),
                ([fake_img], ("猫", "exact", 1), fake_img),
            ]
            mock_store.return_value.get_index.return_value = MagicMock()

            with patch("scripts.eval_deck_search.IrasutoyaClient", return_value=mock_client):
                with self.assertRaises(SystemExit) as cm:
                    run_evaluation(
                        deck_name="TestDeck",
                        limit=1,
                        delay=0.0,
                        backend="catalog",
                        track_stages=True,
                    )
                self.assertEqual(cm.exception.code, 1)

        output = mock_stdout.getvalue()
        self.assertIn("[WARNING] Stage baseline search failed", output)
        self.assertIn("⚠", output)
        self.assertIn("had failed search requests and were excluded from recall", output)


if __name__ == "__main__":
    unittest.main()
