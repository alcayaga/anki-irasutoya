"""Hermetic coverage for deck-driven concept compilation."""
import unittest
from unittest.mock import patch

from scripts.agent_synonyms import load_deck_terms, main


class DeckTermsTests(unittest.TestCase):
    """Verify deck selection independently of artificial sorting ranks."""

    def test_selection(self) -> None:
        """Keep all five empirical terms and exclude ineligible notes."""
        words = ["鑑識課", "ネズミ捕り", "止む無し", "完全に", "一言で言うと", "鑑識課"]
        notes = [
            {"noteId": i, "fields": {"Expression": {"value": word},
                                   "Frequency": {"value": ""},
                                   "FreqSort": {"value": str(999999 - i)}}}
            for i, word in enumerate(words)
        ]
        notes.extend([
            {"noteId": 10, "fields": {"Expression": {"value": "猫"}}},
            {"noteId": 11, "fields": {"Expression": {"value": "犬"},
                                    "Frequency": {"value": "123"}}},
        ])
        with patch("scripts.eval_deck_search.anki_invoke",
                   side_effect=[list(range(8)), list(reversed(notes))]) as invoke:
            self.assertEqual(load_deck_terms("Mining", "http://localhost:8765"), words[:5])
        self.assertEqual([call.args[0] for call in invoke.call_args_list], ["findNotes", "notesInfo"])

    def test_flags(self) -> None:
        """Reject incomplete deck input and conflicting sources before connecting."""
        for args in (["--deck", "Mining"], ["--missing-frequency"],
                     ["--deck", "Mining", "--missing-frequency", "--terms", "猫"]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                main(args)

    def test_empty_and_failure(self) -> None:
        """Avoid compilation for empty results and report connection failures."""
        args = ["--deck", "Mining", "--missing-frequency"]
        with patch("scripts.agent_synonyms.load_deck_terms", return_value=[]),                 patch("scripts.agent_synonyms.run_ollama_compile") as compile_mock:
            self.assertEqual(main(args), 0)
            compile_mock.assert_not_called()
        with patch("scripts.agent_synonyms.load_deck_terms", side_effect=ConnectionError):
            self.assertEqual(main(args), 1)

    def test_dispatch(self) -> None:
        """Dispatch selected terms without artificial frequency metadata."""
        with patch("scripts.agent_synonyms.load_deck_terms", return_value=["鑑識課"]),                 patch("scripts.agent_synonyms.run_ollama_compile", return_value=True) as compile_mock:
            self.assertEqual(main(["--deck", "Mining", "--missing-frequency"]), 0)
        self.assertEqual(compile_mock.call_args.kwargs["terms"], ["鑑識課"])
        self.assertFalse(compile_mock.call_args.kwargs["include_frequency"])
