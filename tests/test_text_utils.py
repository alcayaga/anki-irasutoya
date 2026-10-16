"""Unit tests for text query extraction and sanitization utilities."""
import unittest
from text_utils import clean_search_query


class TestTextUtils(unittest.TestCase):
    """Test suite for sanitizing and stripping noise from Anki field values."""

    def test_empty_and_none(self):
        """Test handling of None and empty string inputs."""
        self.assertEqual(clean_search_query(None), "")
        self.assertEqual(clean_search_query(""), "")
        self.assertEqual(clean_search_query("   "), "")

    def test_strip_html(self):
        """Test stripping HTML tags and entities."""
        self.assertEqual(clean_search_query("<div>猫</div>"), "猫")
        self.assertEqual(clean_search_query("<span style='color:red;'>犬</span>"), "犬")
        self.assertEqual(clean_search_query("<b>リンゴ</b>&nbsp;"), "リンゴ")

    def test_strip_sound_tags(self):
        """Test stripping Anki sound tags from field content."""
        self.assertEqual(clean_search_query("猫 [sound:cat.mp3]"), "猫")
        self.assertEqual(clean_search_query("[sound:taberu_audio.mp3] 食べる"), "食べる")

    def test_strip_furigana_brackets(self):
        """Test removing bracketed furigana readings while preserving kanji."""
        self.assertEqual(clean_search_query("食べる[たべる]"), "食べる")
        self.assertEqual(clean_search_query("日本語[にほんご]"), "日本語")

    def test_strip_parenthesized_readings(self):
        """Test removing parenthesized kana readings in Japanese or English parentheses."""
        self.assertEqual(clean_search_query("猫（ねこ）"), "猫")
        self.assertEqual(clean_search_query("犬(いぬ)"), "犬")
        self.assertEqual(clean_search_query("走る（はしる） [sound:run.mp3]"), "走る")

    def test_multiple_definitions_and_separators(self):
        """Test splitting multi-definition glosses and keeping the primary headword."""
        self.assertEqual(clean_search_query("食べる / 飲む"), "食べる")
        self.assertEqual(clean_search_query("猫・ねこ"), "猫")
        self.assertEqual(clean_search_query("リンゴ, りんご"), "リンゴ")
        self.assertEqual(clean_search_query("犬；いぬ"), "犬")

    def test_combined_complex_field(self):
        """Test cleaning a complex field containing HTML, furigana, sound tags, and glosses."""
        raw = "<div><b>勉強</b>[べんきょう]（する） / study [sound:audio123.mp3]</div>"
        self.assertEqual(clean_search_query(raw), "勉強")


if __name__ == "__main__":
    unittest.main()
