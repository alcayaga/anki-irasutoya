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
        """Test stripping HTML tags and unescaping entities."""
        self.assertEqual(clean_search_query("<div>猫</div>"), "猫")
        self.assertEqual(clean_search_query("<span style='color:red;'>犬</span>"), "犬")
        self.assertEqual(clean_search_query("<b>リンゴ</b>&nbsp;"), "リンゴ")
        self.assertEqual(clean_search_query("&#29483;"), "猫")
        self.assertEqual(clean_search_query("Tom &amp; Jerry"), "Tom & Jerry")

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

    def test_nfkc_normalization(self):
        """Test NFKC normalization of full-width numbers and characters."""
        self.assertEqual(clean_search_query("１杯"), "1杯")
        self.assertEqual(clean_search_query("２人"), "2人")
        self.assertEqual(clean_search_query("ＡＩ"), "AI")
        self.assertEqual(clean_search_query("ｶﾒﾗ"), "カメラ")

    def test_extract_furigana_terms(self):
        """Test extracting surface kanji and bracketed readings."""
        from text_utils import extract_furigana_terms
        self.assertEqual(extract_furigana_terms("林檎[りんご]"), ("林檎", "りんご"))
        self.assertEqual(extract_furigana_terms("動[うご]かす"), ("動かす", "うごかす"))
        self.assertEqual(extract_furigana_terms("１杯[いっぱい]"), ("1杯", "いっぱい"))
        self.assertEqual(extract_furigana_terms("猫"), ("猫", None))
        self.assertEqual(extract_furigana_terms(""), ("", None))
        self.assertEqual(extract_furigana_terms(None), ("", None))

    def test_kana_conversions(self):
        """Test Hiragana and Katakana bidirectional character conversions."""
        from text_utils import to_katakana, to_hiragana
        self.assertEqual(to_katakana("とまと"), "トマト")
        self.assertEqual(to_katakana("ぱん"), "パン")
        self.assertEqual(to_katakana("かめら"), "カメラ")
        self.assertEqual(to_hiragana("リンゴ"), "りんご")
        self.assertEqual(to_hiragana("コーヒー"), "こーひー")

    def test_expand_katakana_chouon(self):
        """Test expanding Katakana terms missing trailing vowel elongation marks."""
        from text_utils import expand_katakana_chouon
        self.assertEqual(expand_katakana_chouon("メンバ"), ["メンバー"])
        self.assertEqual(expand_katakana_chouon("サーバ"), ["サーバー"])
        self.assertEqual(expand_katakana_chouon("フォルダ"), ["フォルダー"])
        self.assertEqual(expand_katakana_chouon("コーヒー"), [])  # Already has ー
        self.assertEqual(expand_katakana_chouon("猫"), [])       # Not Katakana

    def test_convert_to_kanji_numerals(self):
        """Test converting Arabic numerals with Japanese counters into Kanji numerals."""
        from text_utils import convert_to_kanji_numerals
        self.assertEqual(convert_to_kanji_numerals("1杯"), "一杯")
        self.assertEqual(convert_to_kanji_numerals("2人"), "二人")
        self.assertEqual(convert_to_kanji_numerals("3つ"), "三つ")
        self.assertEqual(convert_to_kanji_numerals("1本"), "一本")
        self.assertIsNone(convert_to_kanji_numerals("abc"))
        self.assertIsNone(convert_to_kanji_numerals("9999"))

    def test_preserve_middle_dot_compound_loanwords(self):
        """Test preserving middle dots in compound loanwords while splitting definitions."""
        self.assertEqual(clean_search_query("メッセージ・カード"), "メッセージ・カード")
        self.assertEqual(clean_search_query("オート・ロック"), "オート・ロック")
        self.assertEqual(clean_search_query("猫・ねこ"), "猫")

    def test_deinflect_japanese(self):
        """Test heuristic deinflection of verb and compound forms."""
        from text_utils import deinflect_japanese
        self.assertIn("帰る", deinflect_japanese("帰ってくる"))
        self.assertIn("走る", deinflect_japanese("走っている"))
        self.assertIn("読む", deinflect_japanese("読める"))
        self.assertIn("結果", deinflect_japanese("その結果"))
        self.assertIn("方", deinflect_japanese("方がいい"))

    def test_generate_search_candidates(self):
        """Test candidate generation priority, tiers, and note field inspection."""
        from text_utils import generate_search_candidates
        
        # Test 1: Direct furigana in primary term (All Tier 1)
        cands1 = generate_search_candidates("林檎[りんご]")
        terms1 = [c[0] for c in cands1]
        self.assertIn("林檎", terms1)
        self.assertIn("りんご", terms1)
        self.assertIn("リンゴ", terms1)
        self.assertEqual(cands1[0], ("林檎", "primary", 1))

        # Test 2: Separate note reading field (Tier 2)
        note = {
            "Expression": "其の儘",
            "ExpressionReading": "そのまま"
        }
        cands2 = generate_search_candidates("其の儘", note=note, allow_tier2=True)
        self.assertEqual(cands2[0], ("其の儘", "primary", 1))
        # Note reading should be present as Tier 2
        reading_cand = next(c for c in cands2 if c[0] == "そのまま")
        self.assertEqual(reading_cand[2], 2)
        self.assertIn("note_reading", reading_cand[1])

        # Test 3: Numeral counter expansion
        cands3 = generate_search_candidates("１杯")
        terms3 = [c[0] for c in cands3]
        self.assertIn("1杯", terms3)
        self.assertIn("一杯", terms3)

        # Test 4: Katakana elongation
        cands4 = generate_search_candidates("メンバ")
        terms4 = [c[0] for c in cands4]
        self.assertIn("メンバー", terms4)


if __name__ == "__main__":
    unittest.main()

