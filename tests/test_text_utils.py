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
        self.assertEqual(extract_furigana_terms("其の儘[そのまま]"), ("其の儘", "そのまま"))
        self.assertEqual(extract_furigana_terms("食べる[たべる]"), ("食べる", "たべる"))
        self.assertEqual(extract_furigana_terms("お茶[おちゃ]"), ("お茶", "おちゃ"))
        self.assertEqual(extract_furigana_terms("ご飯[ごはん]"), ("ご飯", "ごはん"))
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
        self.assertEqual(expand_katakana_chouon("パン"), [])      # Not an omission ending
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
        self.assertEqual(clean_search_query("猫・ネコ・キャット"), "猫")
        self.assertEqual(clean_search_query("メッセージ・カード・手紙"), "メッセージ・カード")

    def test_deinflect_japanese(self):
        """Test heuristic deinflection of verb and compound forms."""
        from text_utils import deinflect_japanese
        kaeru_forms = deinflect_japanese("帰ってくる")
        self.assertIn("帰る", kaeru_forms)
        self.assertNotIn("帰っる", kaeru_forms)
        hashiru_forms = deinflect_japanese("走っている")
        self.assertIn("走る", hashiru_forms)
        self.assertNotIn("走っる", hashiru_forms)
        mite_forms = deinflect_japanese("走ってみる")
        self.assertIn("走る", mite_forms)
        self.assertNotIn("走っる", mite_forms)
        self.assertIn("結果", deinflect_japanese("その結果"))
        self.assertIn("方", deinflect_japanese("方がいい"))

    def test_generate_search_candidates(self):
        """Test candidate generation priority, tiers, and note field inspection."""
        from text_utils import generate_search_candidates
        
        # Test 1: Direct furigana in primary term (Primary is Tier 1, readings are Tier 2 for audit)
        cands1 = generate_search_candidates("林檎[りんご]")
        terms1 = [c[0] for c in cands1]
        self.assertIn("林檎", terms1)
        self.assertIn("りんご", terms1)
        self.assertIn("リンゴ", terms1)
        self.assertEqual(cands1[0], ("林檎", "primary", 1))
        self.assertEqual(cands1[1], ("りんご", "same_field_reading", 2))
        self.assertEqual(cands1[2], ("リンゴ", "same_field_reading_katakana", 2))

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

        # Test 4: Katakana elongation (Tier 2)
        cands4 = generate_search_candidates("メンバ")
        terms4 = [c[0] for c in cands4]
        self.assertIn("メンバー", terms4)
        self.assertEqual(cands4[1], ("メンバー", "katakana_chouon", 2))

        # Test 5: Separate reading field containing bracketed furigana
        note_furigana = {
            "Expression": "其の儘",
            "Reading": "其[そ]の儘[まま]"
        }
        cands5 = generate_search_candidates("其の儘", note=note_furigana, allow_tier2=True)
        terms5 = [c[0] for c in cands5]
        self.assertIn("そのまま", terms5)
        self.assertIn("ソノママ", terms5)

        # Test 6: Mixed-script verb (e.g. 食べる) does not produce pseudo-katakana (食ベル)
        cands_taberu = generate_search_candidates("食べる")
        terms_taberu = [c[0] for c in cands_taberu]
        self.assertNotIn("食ベル", terms_taberu)

        # Pure Hiragana loanword (とまと) converts to Katakana (トマト)
        cands_tomato = generate_search_candidates("とまと")
        terms_tomato = [c[0] for c in cands_tomato]
        self.assertIn("トマト", terms_tomato)

        # Test 7: Non-string furigana field value safely handled
        note_non_str = {"Expression": "本", "Furigana": 12345}
        cands_non_str = generate_search_candidates("本", note=note_non_str, allow_tier2=True)
        self.assertEqual(cands_non_str[0], ("本", "primary", 1))

        # Test 8: Note reading fields take precedence over heuristic deinflections in Tier 2
        note_kaeru = {
            "Expression": "帰ってくる",
            "Reading": "かえってくる"
        }
        cands_kaeru = generate_search_candidates("帰ってくる", note=note_kaeru, allow_tier2=True, allow_deinflection=True)
        terms_kaeru = [c[0] for c in cands_kaeru]
        self.assertIn("かえってくる", terms_kaeru)
        self.assertIn("帰る", terms_kaeru)
        self.assertLess(terms_kaeru.index("かえってくる"), terms_kaeru.index("帰る"))
        self.assertNotIn("帰っる", terms_kaeru)

    def test_verb_conjugations(self):
        """Test Kanji-preserving verb inflection generation for Irasutoya titles."""
        from text_utils import conjugate_japanese_verb
        # Godan verbs
        katsu = conjugate_japanese_verb("勝つ")
        self.assertIn("勝った", katsu)
        self.assertIn("勝ち", katsu)
        self.assertIn("勝とう", katsu)

        uru = conjugate_japanese_verb("売る")
        self.assertIn("売った", uru)
        self.assertIn("売り", uru)
        self.assertIn("売ろう", uru)

        yobu = conjugate_japanese_verb("呼ぶ")
        self.assertIn("呼んだ", yobu)
        self.assertIn("呼び", yobu)

        # Ichidan verbs
        tsukareru = conjugate_japanese_verb("疲れる")
        self.assertIn("疲れた", tsukareru)
        self.assertIn("疲れ", tsukareru)
        self.assertNotIn("疲れった", tsukareru)
        self.assertNotIn("疲れり", tsukareru)

        mitsukeru = conjugate_japanese_verb("見つける")
        self.assertIn("見つけた", mitsukeru)
        self.assertIn("見つけ", mitsukeru)
        self.assertNotIn("見つけった", mitsukeru)

        taberu = conjugate_japanese_verb("食べる")
        self.assertEqual(taberu, ["食べた", "食べ"])
        self.assertNotIn("食べった", taberu)
        self.assertNotIn("食べり", taberu)

        # Godan verbs ending in る must not produce Ichidan forms
        hashiru = conjugate_japanese_verb("走る")
        self.assertIn("走った", hashiru)
        self.assertIn("走り", hashiru)
        self.assertNotIn("走た", hashiru)

        self.assertNotIn("売た", uru)

        # Single-kanji Ichidan verbs (omit bare single-kanji stem to prevent false-positive collisions)
        miru = conjugate_japanese_verb("見る")
        self.assertEqual(miru, ["見た"])
        self.assertNotIn("見", miru)
        self.assertNotIn("見った", miru)

        deru = conjugate_japanese_verb("出る")
        self.assertEqual(deru, ["出た"])
        self.assertNotIn("出", deru)

        # Irregular verbs
        benkyou = conjugate_japanese_verb("勉強する")
        self.assertEqual(benkyou, ["勉強した", "勉強し"])
        self.assertNotIn("勉強すった", benkyou)

        kuru = conjugate_japanese_verb("来る")
        self.assertEqual(kuru, ["来た"])
        self.assertNotIn("来", kuru)

        iku = conjugate_japanese_verb("行く")
        self.assertIn("行った", iku)
        self.assertNotIn("行いた", iku)

        # Compound auxiliaries
        kaette = conjugate_japanese_verb("帰ってくる")
        self.assertEqual(kaette, ["帰ってきた"])

        hashitte = conjugate_japanese_verb("走っている")
        self.assertEqual(hashitte, [])

        # Numeral counters must not be conjugated as verbs
        self.assertEqual(conjugate_japanese_verb("一つ"), [])
        self.assertEqual(conjugate_japanese_verb("二つ"), [])

        # Non-kanji or short inputs should yield empty list
        self.assertEqual(conjugate_japanese_verb("する"), [])
        self.assertEqual(conjugate_japanese_verb("いく"), [])
        self.assertEqual(conjugate_japanese_verb(""), [])

    def test_bikago_prefix(self):
        """Test canonical polite お prefix expansion for everyday nouns."""
        from text_utils import expand_bikago_prefix
        self.assertEqual(expand_bikago_prefix("肉"), "お肉")
        self.assertEqual(expand_bikago_prefix("肌"), "お肌")
        self.assertEqual(expand_bikago_prefix("茶"), "お茶")
        self.assertEqual(expand_bikago_prefix("風呂"), "お風呂")
        self.assertEqual(expand_bikago_prefix("金"), "お金")
        self.assertIsNone(expand_bikago_prefix("猫"))
        self.assertIsNone(expand_bikago_prefix("自動車"))

    def test_grammar_function_words(self):
        """Test identification of pure grammar particles and function words."""
        from text_utils import is_grammar_function_word
        self.assertTrue(is_grammar_function_word("よ"))
        self.assertTrue(is_grammar_function_word("ね"))
        self.assertTrue(is_grammar_function_word("から"))
        self.assertTrue(is_grammar_function_word("まで"))
        self.assertTrue(is_grammar_function_word("ので"))
        self.assertTrue(is_grammar_function_word("なら"))
        self.assertTrue(is_grammar_function_word("もん"))
        self.assertTrue(is_grammar_function_word("その"))
        self.assertTrue(is_grammar_function_word("これ"))
        self.assertFalse(is_grammar_function_word("本"))
        self.assertFalse(is_grammar_function_word("猫"))
        self.assertFalse(is_grammar_function_word("走る"))
        self.assertFalse(is_grammar_function_word("トマト"))

    def test_segmented_and_prefixed_furigana(self):
        """Test extracting segmented furigana with prefixes and spaces."""
        from text_utils import extract_furigana_terms
        self.assertEqual(extract_furigana_terms("お 前[まえ]"), ("お前", "おまえ"))
        self.assertEqual(extract_furigana_terms("この 世[よ]"), ("この世", "このよ"))
        self.assertEqual(extract_furigana_terms("其[そ]の 儘[まま]"), ("其の儘", "そのまま"))
        self.assertEqual(extract_furigana_terms("其[そ]の儘[まま]"), ("其の儘", "そのまま"))
        self.assertEqual(extract_furigana_terms("これ 以上[いじょう]"), ("これ以上", "これいじょう"))

    def test_candidate_tiers_morphology(self):
        """Test that verb inflections and bikago run in Tier 1 while readings run in Tier 2."""
        from text_utils import generate_search_candidates
        # Verb past/volitional inflection is Tier 1, while masu-stem is Tier 2
        cands_verb = generate_search_candidates("勝つ")
        cand_map_verb = {term: (reason, tier) for term, reason, tier in cands_verb}
        self.assertIn("勝った", cand_map_verb)
        self.assertEqual(cand_map_verb["勝った"][1], 1)
        self.assertEqual(cand_map_verb["勝った"][0], "verb_inflection")
        self.assertIn("勝とう", cand_map_verb)
        self.assertEqual(cand_map_verb["勝とう"][1], 1)
        self.assertEqual(cand_map_verb["勝とう"][0], "verb_inflection")
        self.assertIn("勝ち", cand_map_verb)
        self.assertEqual(cand_map_verb["勝ち"][1], 2)
        self.assertEqual(cand_map_verb["勝ち"][0], "verb_stem")

        # Bikago prefix is Tier 1
        cands_niku = generate_search_candidates("肉")
        cand_map_niku = {term: (reason, tier) for term, reason, tier in cands_niku}
        self.assertIn("お肉", cand_map_niku)
        self.assertEqual(cand_map_niku["お肉"][1], 1)
        self.assertEqual(cand_map_niku["お肉"][0], "bikago_prefix")

        # Grammar particle suppresses unanchored search fallbacks
        cands_particle = generate_search_candidates("まで")
        self.assertEqual(len(cands_particle), 1)
        self.assertEqual(cands_particle[0], ("まで", "primary", 1))


if __name__ == "__main__":
    unittest.main()

