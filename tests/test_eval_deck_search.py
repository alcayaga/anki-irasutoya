"""Hermetic tests for scripts/eval_deck_search.py benchmarking logic."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from addon.feed import IrasutoyaImage
from scripts.eval_deck_search import (
    build_search_query,
    detect_source_field,
    fetch_deck_notes,
    legacy_clean_query,
    main,
    run_evaluation,
)


class TestEvalDeckSearch(unittest.TestCase):
    """Test suite for deck search evaluation CLI helper functions and report generation."""

    def setUp(self) -> None:
        """Keep every benchmark run artifact inside a temporary directory."""
        self.logs = tempfile.TemporaryDirectory()
        self.addCleanup(self.logs.cleanup)
        # Redirect the default repository path without changing custom paths.
        self.default_path = patch(
            "scripts.eval_deck_search.__file__",
            str(Path(self.logs.name) / "scripts" / "eval_deck_search.py"),
        )
        self.default_path.start()
        self.addCleanup(self.default_path.stop)

    def read_log(self, output: str) -> str:
        """Read the UTF-8 artifact named by the console report."""
        path = output.rsplit("Log file: ", 1)[1].splitlines()[0].strip()
        return Path(path).read_text(encoding="utf-8")

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

    def test_build_search_query(self) -> None:
        """Verify build_search_query correctly composes deck and custom queries."""
        self.assertEqual(build_search_query("Mining"), 'deck:"Mining"')
        self.assertEqual(build_search_query('My "Mining" Deck'), 'deck:"My \\"Mining\\" Deck"')
        self.assertEqual(build_search_query("Mining", "flag:4"), 'deck:"Mining" (flag:4)')
        self.assertEqual(build_search_query("Mining", "flag:4 or flag:3"), 'deck:"Mining" (flag:4 or flag:3)')
        self.assertEqual(build_search_query("Mining", 'deck:"Other" tag:marked'), 'deck:"Other" tag:marked')
        self.assertEqual(build_search_query("Mining", "DECK:Special"), "DECK:Special")
        # Standalone fully quoted deck selectors
        self.assertEqual(build_search_query("Mining", '"deck:Other Deck" flag:4'), '"deck:Other Deck" flag:4')
        self.assertEqual(build_search_query("Mining", 'flag:4 ("deck:Other Deck")'), 'flag:4 ("deck:Other Deck")')
        self.assertEqual(build_search_query("Mining", '-"deck:Other Deck" flag:4'), 'deck:"Mining" (-"deck:Other Deck" flag:4)')
        self.assertEqual(build_search_query("Mining", '"deck:Mining"'), '"deck:Mining"')
        # Negative deck selectors must not suppress the requested deck scope
        self.assertEqual(build_search_query("Mining", "-deck:Other flag:4"), 'deck:"Mining" (-deck:Other flag:4)')
        self.assertEqual(build_search_query("Mining", "flag:4 (-deck:Other)"), 'deck:"Mining" (flag:4 (-deck:Other))')
        self.assertEqual(build_search_query("Mining", "-(deck:Other) flag:4"), 'deck:"Mining" (-(deck:Other) flag:4)')
        self.assertEqual(build_search_query("Mining", '-(deck:Other or deck:Sub) flag:4'), 'deck:"Mining" (-(deck:Other or deck:Sub) flag:4)')
        self.assertEqual(build_search_query("Mining", '-((deck:Other) or tag:foo) flag:4'), 'deck:"Mining" (-((deck:Other) or tag:foo) flag:4)')
        # Deck selectors inside double-quoted field values or unquoted values must not be treated as deck operands
        self.assertEqual(build_search_query("Mining", 'Front:"deck:Other" flag:4'), 'deck:"Mining" (Front:"deck:Other" flag:4)')
        self.assertEqual(build_search_query("Mining", r'Front:"say \"deck:Other\" now" flag:4'), r'deck:"Mining" (Front:"say \"deck:Other\" now" flag:4)')
        self.assertEqual(build_search_query("Mining", 'Front:deck:Other flag:4'), 'deck:"Mining" (Front:deck:Other flag:4)')
        self.assertEqual(build_search_query("Mining", '-Front:"say deck:Other now" flag:4'), 'deck:"Mining" (-Front:"say deck:Other now" flag:4)')
        # Deck-like text inside quoted general search phrases must not be treated as deck selectors
        self.assertEqual(build_search_query("Mining", '"w:see (deck:x) now" flag:4'), 'deck:"Mining" ("w:see (deck:x) now" flag:4)')
        self.assertEqual(build_search_query("Mining", r'"say \"hello\" (deck:x) now" flag:4'), r'deck:"Mining" ("say \"hello\" (deck:x) now" flag:4)')
        # Explicit positive deck selectors inside parens
        self.assertEqual(build_search_query("Mining", 'flag:4 (deck:Other)'), 'flag:4 (deck:Other)')
        # Combined positive and negative deck selectors preserve raw query
        self.assertEqual(build_search_query("Mining", "deck:Mining -deck:Mining::Subdeck"), "deck:Mining -deck:Mining::Subdeck")
        self.assertEqual(build_search_query("", "flag:4"), "flag:4")
        self.assertEqual(build_search_query("", None), "")

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
        """Verify fetch_deck_notes queries Anki with custom query combined with deck scope."""
        mock_invoke.side_effect = [
            [101],
            [{"cardId": 101, "note": 1001, "fields": {"FreqSort": {"value": "1"}}}],
            [{"noteId": 1001, "fields": {"Expression": {"value": "猫"}}}],
            [102],
            [{"cardId": 102, "note": 1002, "fields": {"FreqSort": {"value": "2"}}}],
            [{"noteId": 1002, "fields": {"Expression": {"value": "犬"}}}],
        ]
        # 1. Custom query scoped to deck with parentheses
        notes = fetch_deck_notes("Mining", query="flag:4")
        self.assertEqual(len(notes), 1)
        mock_invoke.assert_any_call("findCards", anki_url="http://localhost:8765", query='deck:"Mining" (flag:4)')

        # 2. Custom query with existing deck: specifier preserves raw query
        notes_raw = fetch_deck_notes("Mining", query='deck:"Other" flag:4')
        self.assertEqual(len(notes_raw), 1)
        mock_invoke.assert_any_call("findCards", anki_url="http://localhost:8765", query='deck:"Other" flag:4')

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

    @patch("scripts.eval_deck_search.fetch_deck_notes", return_value=[])
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_evaluation_empty_query_scope_message(self, mock_stdout: io.StringIO, mock_fetch: MagicMock) -> None:
        """Verify run_evaluation names custom query in empty-result message."""
        run_evaluation("Mining", query="flag:4")
        self.assertEqual(mock_fetch.call_args[1]["query"], "flag:4")
        output = mock_stdout.getvalue()
        self.assertIn('Query:              deck:"Mining" (flag:4)', self.read_log(output))
        self.assertIn("No notes available to evaluate in query 'flag:4'.", output)

        mock_stdout.truncate(0)
        mock_stdout.seek(0)
        run_evaluation("Mining", query=None)
        output_deck = mock_stdout.getvalue()
        self.assertIn("No notes available to evaluate in deck 'Mining'.", output_deck)

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
                verbose=True,
            )

        output = mock_stdout.getvalue()
        self.assertNotIn("IRASUTOYA SEARCH RECALL BENCHMARK", output)
        self.assertIn("IRASUTOYA SEARCH RECALL BENCHMARK", self.read_log(output))
        self.assertIn("猫", self.read_log(output))
        self.assertNotIn("[ 1/ 1]", output)
        self.assertIn("↳ Matched:", self.read_log(output))
        self.assertNotIn("↳ Matched:", output)
        self.assertIn("INCREMENTAL RECALL & STRATEGY PROGRESSION", output)
        self.assertIn("1. Legacy (Naive Direct Query)", output)
        self.assertIn("4. Bilingual Gloss (Definition Bridge)", output)

    @patch("scripts.eval_deck_search.anki_invoke")
    @patch("sys.stdout", new_callable=io.StringIO)
    @patch("sys.stderr", new_callable=io.StringIO)
    def test_run_evaluation_stage_baseline_error_preserves_failure(
        self, mock_stderr: io.StringIO, mock_stdout: io.StringIO, mock_invoke: MagicMock
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
        self.assertNotIn("[WARNING] Stage baseline search failed", output)
        self.assertIn("[WARNING] Stage baseline search failed", self.read_log(output))
        self.assertIn("⚠", self.read_log(output))
        error = "had failed search requests and were excluded from recall"
        self.assertNotIn(error, output)
        self.assertIn(error, mock_stderr.getvalue())
        self.assertIn(error, self.read_log(output))
        self.assertIn("EVALUATION SUMMARY REPORT", output)
        self.assertIn("Valid / Excluded Notes:   0 / 1", output)

    @patch("scripts.eval_deck_search.fetch_deck_notes", return_value=[])
    def test_custom_directory_unique_logs(self, mock_fetch: MagicMock) -> None:
        """Create distinct complete logs for repeated runs in a new directory."""
        directory = Path(self.logs.name) / "custom" / "nested"
        for _ in range(2):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                run_evaluation("日本語", log_dir=str(directory))
            self.assertIn("Log file: ", output.getvalue())
        paths = list(directory.glob("*.log"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            content = path.read_text(encoding="utf-8")
            self.assertEqual(content.count("IRASUTOYA SEARCH RECALL BENCHMARK"), 1)
            self.assertIn("日本語", content)
            self.assertIn("No notes available", content)
        self.assertEqual(mock_fetch.call_count, 2)

    @patch("scripts.eval_deck_search.fetch_deck_notes", return_value=[])
    def test_log_directory_resolution(self, mock_fetch: MagicMock) -> None:
        """Anchor default logs to the script and relative overrides to cwd."""
        with patch("os.getcwd", return_value=self.logs.name):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                run_evaluation(log_dir="relative")
        custom = Path(output.getvalue().rsplit("Log file: ", 1)[1].splitlines()[0].strip())
        self.assertEqual(custom.parent.resolve(), Path(self.logs.name).resolve() / "relative")
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            run_evaluation()
        default = Path(output.getvalue().rsplit("Log file: ", 1)[1].splitlines()[0].strip())
        self.assertEqual(default.parent, Path(self.logs.name).resolve() / "scratch" / "eval_deck_search")

    @patch("scripts.eval_deck_search.fetch_deck_notes")
    def test_log_initialization_failure(self, mock_fetch: MagicMock) -> None:
        """Fail visibly before querying Anki if the log directory is unusable."""
        with patch("scripts.eval_deck_search.Path.mkdir", side_effect=PermissionError("denied")):
            with patch("sys.stderr", new_callable=io.StringIO) as errors:
                with self.assertRaises(SystemExit) as result:
                    run_evaluation(log_dir=self.logs.name)
        self.assertEqual(result.exception.code, 1)
        self.assertIn("Cannot initialize benchmark log", errors.getvalue())
        mock_fetch.assert_not_called()

    def test_log_path_visible_before_evaluation(self) -> None:
        """Expose and flush the log location before querying Anki."""
        output = io.StringIO()

        def check_startup(*args: object, **kwargs: object) -> list:
            """Inspect console output at the first benchmark operation."""
            self.assertTrue(output.getvalue().startswith("Log file: "))
            path = Path(output.getvalue().split("Log file: ", 1)[1].strip())
            self.assertTrue(path.is_file())
            flush.assert_called()
            return []

        with patch("sys.stdout", output), patch.object(output, "flush") as flush:
            with patch("scripts.eval_deck_search.fetch_deck_notes", side_effect=check_startup):
                run_evaluation()
        self.assertEqual(output.getvalue().count("Log file: "), 1)

    @patch("scripts.eval_deck_search.fetch_deck_notes", side_effect=RuntimeError("offline"))
    def test_fatal_error_logged_and_visible(self, mock_fetch: MagicMock) -> None:
        """Preserve startup failure status with a durable error and log path."""
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            with patch("sys.stderr", new_callable=io.StringIO) as errors:
                with self.assertRaises(SystemExit) as result:
                    run_evaluation()
        self.assertEqual(result.exception.code, 1)
        self.assertIn("offline", errors.getvalue())
        self.assertIn("offline", self.read_log(output.getvalue()))
        self.assertNotIn("IRASUTOYA SEARCH", output.getvalue())

    @patch("scripts.eval_deck_search.run_evaluation")
    @patch("sys.argv", ["scripts/eval_deck_search.py", "--query", "flag:4", "--deck", "Mining"])
    def test_main_cli_forwards_custom_query(self, mock_run: MagicMock) -> None:
        """Verify main CLI parser forwards custom --query argument to run_evaluation."""
        main()
        self.assertEqual(mock_run.call_args[1]["query"], "flag:4")
        self.assertEqual(mock_run.call_args[1]["deck_name"], "Mining")


if __name__ == "__main__":
    unittest.main()
