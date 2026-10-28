"""Unit tests for text query extraction and sanitization utilities."""
import os
import re
import sqlite3
import tempfile
import unittest
from typing import List
from unittest.mock import patch

from addon.catalog import CatalogStore, create_schema, write_synonyms
from addon.synonyms import lookup_sudachi_synonyms
from addon.text_utils import clean_search_query, generate_search_candidates


class TestTextUtils(unittest.TestCase):
    """Test suite for sanitizing and stripping noise from Anki field values."""

    def setUp(self) -> None:
        """Isolate candidate generator tests from external catalog concept lookups."""
        patcher = patch("addon.text_utils.lookup_concept_map", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)
        k_patcher = patch("addon.text_utils.lookup_kaishi_concept", return_value=[])
        k_patcher.start()
        self.addCleanup(k_patcher.stop)
        s_patcher = patch("addon.text_utils.lookup_sudachi_synonyms", return_value=[])
        s_patcher.start()
        self.addCleanup(s_patcher.stop)

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
        from addon.text_utils import extract_furigana_terms
        self.assertEqual(extract_furigana_terms("林檎[りんご]"), ("林檎", "りんご"))
        self.assertEqual(extract_furigana_terms("動[うご]かす"), ("動かす", "うごかす"))
        self.assertEqual(extract_furigana_terms("１杯[いっぱい]"), ("1杯", "いっぱい"))
        self.assertEqual(extract_furigana_terms("其の儘[そのまま]"), ("其の儘", "そのまま"))
        self.assertEqual(extract_furigana_terms("食べる[たべる]"), ("食べる", "たべる"))
        self.assertEqual(extract_furigana_terms("お茶[おちゃ]"), ("お茶", "おちゃ"))
        self.assertEqual(extract_furigana_terms("ご飯[ごはん]"), ("ご飯", "ごはん"))
        # Fields with HTML wrappers or trailing sound tags
        self.assertEqual(extract_furigana_terms("<div>其の儘[そのまま]</div>"), ("其の儘", "そのまま"))
        self.assertEqual(extract_furigana_terms("食べ物[たべもの] [sound:x.mp3]"), ("食べ物", "たべもの"))
        self.assertEqual(extract_furigana_terms("<div>林檎[りんご]</div> [sound:audio.mp3]"), ("林檎", "りんご"))
        self.assertEqual(extract_furigana_terms("猫"), ("猫", None))
        self.assertEqual(extract_furigana_terms(""), ("", None))
        self.assertEqual(extract_furigana_terms(None), ("", None))

    def test_kana_conversions(self):
        """Test Hiragana and Katakana bidirectional character conversions."""
        from addon.text_utils import to_hiragana, to_katakana
        self.assertEqual(to_katakana("とまと"), "トマト")
        self.assertEqual(to_katakana("ぱん"), "パン")
        self.assertEqual(to_katakana("かめら"), "カメラ")
        self.assertEqual(to_hiragana("リンゴ"), "りんご")
        self.assertEqual(to_hiragana("コーヒー"), "こーひー")

    def test_expand_katakana_chouon(self):
        """Test expanding Katakana terms missing trailing vowel elongation marks."""
        from addon.text_utils import expand_katakana_chouon
        self.assertEqual(expand_katakana_chouon("メンバ"), ["メンバー"])
        self.assertEqual(expand_katakana_chouon("サーバ"), ["サーバー"])
        self.assertEqual(expand_katakana_chouon("フォルダ"), ["フォルダー"])
        self.assertEqual(expand_katakana_chouon("コーヒー"), [])  # Already has ー
        self.assertEqual(expand_katakana_chouon("パン"), [])      # Not an omission ending
        self.assertEqual(expand_katakana_chouon("猫"), [])       # Not Katakana

    def test_convert_to_kanji_numerals(self):
        """Test converting Arabic numerals with Japanese counters into Kanji numerals."""
        from addon.text_utils import convert_to_kanji_numerals
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
        from addon.text_utils import deinflect_japanese
        kaeru_forms = deinflect_japanese("帰ってくる")
        self.assertIn("帰る", kaeru_forms)
        self.assertNotIn("帰っる", kaeru_forms)
        hashiru_forms = deinflect_japanese("走っている")
        self.assertIn("走る", hashiru_forms)
        self.assertNotIn("走っる", hashiru_forms)
        mite_forms = deinflect_japanese("走ってみる")
        self.assertIn("走る", mite_forms)
        self.assertNotIn("走っる", mite_forms)
        # 行 geminate exception: 行ってくる -> 行く (not 行る), but 流行っている -> 流行る (not 流行く)
        iku_forms = deinflect_japanese("行ってくる")
        self.assertEqual(iku_forms[0], "行く")
        self.assertNotIn("行る", iku_forms)
        hayaru_forms = deinflect_japanese("流行っている")
        self.assertEqual(hayaru_forms[0], "流行る")
        self.assertNotIn("流行く", hayaru_forms)
        # Voiced reductions: 読んでいる -> 読む (not 読んぐ), 泳いでいる -> 泳ぐ (not 泳いぐ), 遊んでいる -> 遊ぶ
        yomu_forms = deinflect_japanese("読んでいる")
        self.assertEqual(yomu_forms[0], "読む")
        self.assertNotIn("読んぐ", yomu_forms)
        oyogu_forms = deinflect_japanese("泳いでいる")
        self.assertEqual(oyogu_forms[0], "泳ぐ")
        self.assertNotIn("泳いぐ", oyogu_forms)
        asobu_forms = deinflect_japanese("遊んでいる")
        self.assertEqual(asobu_forms[0], "遊ぶ")
        shinu_forms = deinflect_japanese("死んでいる")
        self.assertEqual(shinu_forms[0], "死ぬ")
        # Voiced reductions with でいく and でみる: 飛んでいく -> 飛ぶ, 読んでみる -> 読む
        tonde_forms = deinflect_japanese("飛んでいく")
        self.assertEqual(tonde_forms[0], "飛ぶ")
        yonde_forms = deinflect_japanese("読んでみる")
        self.assertEqual(yonde_forms[0], "読む")
        # Regular te-stems: 勉強している -> 勉強する, 話している -> 話す, 書いている -> 書く, 食べている -> 食べる
        self.assertEqual(deinflect_japanese("勉強している")[0], "勉強する")
        self.assertEqual(deinflect_japanese("話している")[0], "話す")
        self.assertEqual(deinflect_japanese("書いている")[0], "書く")
        self.assertEqual(deinflect_japanese("食べている")[0], "食べる")
        self.assertIn("結果", deinflect_japanese("その結果"))
        self.assertIn("方", deinflect_japanese("方がいい"))

    def test_generate_search_candidates(self):
        """Test candidate generation priority, tiers, and note field inspection."""
        from addon.text_utils import generate_search_candidates

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

        # Test 8: Surface-derived validated deinflected bases precede phonetic readings in Tier 2
        # to ensure semantic bases are evaluated ahead of unanchored phonetic readings
        note_kaeru = {
            "Expression": "帰ってくる",
            "Reading": "かえってくる"
        }
        cands_kaeru = generate_search_candidates("帰ってくる", note=note_kaeru, allow_tier2=True, allow_deinflection=True)
        terms_kaeru = [c[0] for c in cands_kaeru]
        self.assertIn("帰る", terms_kaeru[:5])
        self.assertIn("かえってくる", terms_kaeru[:5])
        self.assertLess(terms_kaeru.index("帰る"), terms_kaeru.index("かえってくる"))
        self.assertNotIn("帰っる", terms_kaeru)
        # Speculative alternatives (帰う, 帰つ) appear after note reading
        before_reading = terms_kaeru[:terms_kaeru.index("かえってくる")]
        self.assertNotIn("帰う", before_reading)
        self.assertNotIn("帰つ", before_reading)

        note_matsu = {
            "Expression": "待っている",
            "Reading": "まっている"
        }
        cands_matsu = generate_search_candidates("待っている", note=note_matsu, allow_tier2=True, allow_deinflection=True)
        terms_matsu = [c[0] for c in cands_matsu]
        self.assertIn("待つ", terms_matsu[:5])
        self.assertIn("まっている", terms_matsu[:5])
        self.assertLess(terms_matsu.index("待つ"), terms_matsu.index("まっている"))
        self.assertNotIn("待る", terms_matsu[:terms_matsu.index("まっている")])

    def test_verb_conjugations(self):
        """Test Kanji-preserving verb inflection generation for Irasutoya titles."""
        from addon.text_utils import conjugate_japanese_verb
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
        self.assertIn("疲れよう", tsukareru)
        self.assertIn("疲れ", tsukareru)
        self.assertNotIn("疲れった", tsukareru)
        self.assertNotIn("疲れり", tsukareru)

        mitsukeru = conjugate_japanese_verb("見つける")
        self.assertIn("見つけた", mitsukeru)
        self.assertIn("見つけよう", mitsukeru)
        self.assertIn("見つけ", mitsukeru)
        self.assertNotIn("見つけった", mitsukeru)

        taberu = conjugate_japanese_verb("食べる")
        self.assertEqual(taberu, ["食べた", "食べよう", "食べ"])
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
        self.assertEqual(miru, ["見た", "見よう"])
        self.assertNotIn("見", miru)
        self.assertNotIn("見った", miru)

        deru = conjugate_japanese_verb("出る")
        self.assertEqual(deru, ["出た", "出よう"])
        self.assertNotIn("出", deru)

        # Single-kanji Ichidan homographs of 見る (観る, 診る, 看る, 視る)
        kanmiru = conjugate_japanese_verb("観る")
        self.assertEqual(kanmiru, ["観た", "観よう"])
        self.assertNotIn("観った", kanmiru)
        shinmiru = conjugate_japanese_verb("診る")
        self.assertEqual(shinmiru, ["診た", "診よう"])
        self.assertNotIn("診った", shinmiru)

        # Irregular verbs
        benkyou = conjugate_japanese_verb("勉強する")
        self.assertEqual(benkyou, ["勉強した", "勉強しよう", "勉強し"])
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
        self.assertEqual(conjugate_japanese_verb("飛んでいく"), [])
        self.assertEqual(conjugate_japanese_verb("読んでみる"), [])

        # Numeral counters must not be conjugated as verbs
        self.assertEqual(conjugate_japanese_verb("一つ"), [])
        self.assertEqual(conjugate_japanese_verb("二つ"), [])

        # Adverbial i-adjective forms ending in く must not be conjugated as verbs
        self.assertEqual(conjugate_japanese_verb("早く"), [])
        self.assertEqual(conjugate_japanese_verb("遅く"), [])
        self.assertEqual(conjugate_japanese_verb("多く"), [])
        self.assertEqual(conjugate_japanese_verb("美しく"), [])
        self.assertEqual(conjugate_japanese_verb("大きく"), [])
        # Non-verb adverbs and interrogative counters must not be conjugated as verbs
        self.assertEqual(conjugate_japanese_verb("全く"), [])
        self.assertEqual(conjugate_japanese_verb("暫く"), [])
        self.assertEqual(conjugate_japanese_verb("漸く"), [])
        self.assertEqual(conjugate_japanese_verb("恐らく"), [])
        self.assertEqual(conjugate_japanese_verb("悉く"), [])
        self.assertEqual(conjugate_japanese_verb("如く"), [])
        self.assertEqual(conjugate_japanese_verb("幾つ"), [])

        # Non-kanji or short inputs should yield empty list
        self.assertEqual(conjugate_japanese_verb("する"), [])
        self.assertEqual(conjugate_japanese_verb("いく"), [])
        self.assertEqual(conjugate_japanese_verb(""), [])

    def test_bikago_prefix(self):
        """Test canonical polite お prefix expansion for everyday nouns."""
        from addon.text_utils import expand_bikago_prefix
        self.assertEqual(expand_bikago_prefix("肉"), "お肉")
        self.assertEqual(expand_bikago_prefix("肌"), "お肌")
        self.assertEqual(expand_bikago_prefix("茶"), "お茶")
        self.assertEqual(expand_bikago_prefix("風呂"), "お風呂")
        self.assertEqual(expand_bikago_prefix("金"), "お金")
        self.assertIsNone(expand_bikago_prefix("猫"))
        self.assertIsNone(expand_bikago_prefix("自動車"))

    def test_grammar_function_words(self):
        """Test identification of pure grammar particles and function words."""
        from addon.text_utils import is_grammar_function_word
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
        from addon.text_utils import extract_furigana_terms
        self.assertEqual(extract_furigana_terms("お 前[まえ]"), ("お前", "おまえ"))
        self.assertEqual(extract_furigana_terms("この 世[よ]"), ("この世", "このよ"))
        self.assertEqual(extract_furigana_terms("其[そ]の 儘[まま]"), ("其の儘", "そのまま"))
        self.assertEqual(extract_furigana_terms("其[そ]の儘[まま]"), ("其の儘", "そのまま"))
        self.assertEqual(extract_furigana_terms("これ 以上[いじょう]"), ("これ以上", "これいじょう"))

    def test_candidate_tiers_morphology(self):
        """Test that verb inflections and bikago run in Tier 1 while readings run in Tier 2."""
        from addon.text_utils import generate_search_candidates
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

    def test_yomitan_uk_reading_promotion(self):
        """Test that words tagged 'uk' promote their Kana and Katakana readings to Tier 1."""
        from addon.text_utils import generate_search_candidates
        note = {
            "Expression": "其の儘",
            "ExpressionReading": "そのまま",
            "MainDefinition": (
                '<div class="yomitan-glossary"><ol><li data-dictionary="JMdict [2025-11-01]">'
                '<i>(1, adj-no, adv, n, uk, JMdict [2025-11-01])</i> '
                '<span><ul data-sc-content="glossary"><li>as is</li></ul></span></li></ol></div>'
            )
        }
        cands = generate_search_candidates(note["Expression"], note=note, allow_tier2=True)
        cand_map = {term: (reason, tier) for term, reason, tier in cands}
        self.assertIn("そのまま", cand_map)
        self.assertEqual(cand_map["そのまま"], ("yomitan_uk_reading", 1))
        self.assertIn("ソノママ", cand_map)
        self.assertEqual(cand_map["ソノママ"], ("yomitan_uk_katakana", 1))

    def test_yomitan_secondary_uk_not_promoted(self):
        """Test that uk on a secondary sense does NOT promote kana reading to Tier 1."""
        from addon.text_utils import generate_search_candidates
        note = {
            "Expression": "漢字",
            "ExpressionReading": "かんじ",
            "MainDefinition": (
                '<div class="yomitan-glossary"><ol>'
                '<li data-dictionary="JMdict [2025-11-01]"><i>(1, n, JMdict [2025-11-01])</i> <span>Chinese characters</span></li>'
                '<li data-dictionary="JMdict [2025-11-01]"><i>(2, n, uk, JMdict [2025-11-01])</i> <span>secondary sense</span></li>'
                '</ol></div>'
            )
        }
        cands = generate_search_candidates(note["Expression"], note=note, allow_tier2=True)
        cand_map = {term: (reason, tier) for term, reason, tier in cands}
        # Kana reading should be Tier 2 fallback, NOT promoted to Tier 1
        self.assertIn("かんじ", cand_map)
        self.assertEqual(cand_map["かんじ"][1], 2)
        reasons = [reason for term, reason, tier in cands]
        self.assertNotIn("yomitan_uk_reading", reasons)

    def test_yomitan_forms_star_promotion(self):
        """Test that ★ forms from formsTable are promoted to Tier 1."""
        from addon.text_utils import generate_search_candidates
        note = {
            "Expression": "交代",
            "ExpressionReading": "こうたい",
            "MainDefinition": (
                '<table data-sc-content="formsTable">'
                '<tbody><tr><th></th><th>交代</th><th>交替</th></tr>'
                '<tr><th>こうたい</th><td>★</td><td>★</td></tr></tbody></table>'
            )
        }
        cands = generate_search_candidates(note["Expression"], note=note, allow_tier2=True)
        cand_map = {term: (reason, tier) for term, reason, tier in cands}
        self.assertIn("交替", cand_map)
        self.assertEqual(cand_map["交替"], ("yomitan_star_form", 1))

    def test_yomitan_deterministic_verb_conjugation(self):
        """Test that exact JMdict verb class (v5r) fixes Godan verbs misclassified by heuristic."""
        from addon.text_utils import generate_search_candidates
        # Without Yomitan metadata, 見くびる would be misclassified as Ichidan (stem ends in 'び')
        note = {
            "Expression": "見くびる",
            "ExpressionReading": "みくびる",
            "MainDefinition": (
                '<ol><li data-dictionary="JMdict [2025-11-01]">'
                '<i>(1, v5r, vt, JMdict [2025-11-01])</i> '
                '<span>to underrate</span></li></ol>'
            )
        }
        cands = generate_search_candidates(note["Expression"], note=note, allow_tier2=True)
        cand_map = {term: (reason, tier) for term, reason, tier in cands}
        # Correct Godan past tense (v5r: 見くびった) must be present in Tier 1
        self.assertIn("見くびった", cand_map)
        self.assertEqual(cand_map["見くびった"], ("verb_inflection", 1))
        # Godan volitional form (見くびろう)
        self.assertIn("見くびろう", cand_map)
        self.assertEqual(cand_map["見くびろう"], ("verb_inflection", 1))
        # Masu-stem (見くびり) in Tier 2
        self.assertIn("見くびり", cand_map)
        self.assertEqual(cand_map["見くびり"], ("verb_stem", 2))
        # Wrong Ichidan past tense (見くびた) must NOT be present
        self.assertNotIn("見くびた", cand_map)

    def test_yomitan_references_and_sense_kanji(self):
        """Test that 'see:' references and sense-specific kanji are emitted as Tier 2 fallbacks."""
        from addon.text_utils import generate_search_candidates
        note = {
            "Expression": "総理",
            "ExpressionReading": "そうり",
            "MainDefinition": (
                '<div class="yomitan-glossary">'
                '<ol><li data-dictionary="JMdict [2025-11-01]">'
                '<i>(1, vs, n, JMdict [2025-11-01])</i> '
                '<ul data-sc-content="references"><li>see: <a lang="ja" href="#"><span>内閣総理大臣</span></a>'
                '<span data-sc-content="refGlosses"> prime minister</span></li></ul>'
                '</li></ol></div>'
            )
        }
        cands = generate_search_candidates(note["Expression"], note=note, allow_tier2=True)
        cand_map = {term: (reason, tier) for term, reason, tier in cands}
        self.assertIn("内閣総理大臣", cand_map)
        self.assertEqual(cand_map["内閣総理大臣"], ("yomitan_see_reference", 2))

        # Sense-specific kanji test (止める -> esp. 停める, 留める)
        note_tomeru = {
            "Expression": "止める",
            "ExpressionReading": "とめる",
            "MainDefinition": (
                '<div class="yomitan-glossary">'
                '<ol><li data-dictionary="JMdict [2025-11-01]">'
                '<i>(1, v1, vt, JMdict [2025-11-01])</i> '
                '<ul data-sc-content="notes"><li>esp. 止める, 停める</li></ul>'
                '<ul data-sc-content="notes"><li>esp. 留める</li></ul>'
                '</li></ol></div>'
            )
        }
        cands_tomeru = generate_search_candidates(note_tomeru["Expression"], note=note_tomeru, allow_tier2=True)
        cand_map_tomeru = {term: (reason, tier) for term, reason, tier in cands_tomeru}
        self.assertIn("停める", cand_map_tomeru)
        self.assertEqual(cand_map_tomeru["停める"], ("yomitan_sense_kanji", 2))
        self.assertIn("留める", cand_map_tomeru)
        self.assertEqual(cand_map_tomeru["留める"], ("yomitan_sense_kanji", 2))

    def test_yomitan_particle_gating(self):
        """Test that notes tagged 'prt' or 'conj' suppress unanchored fallbacks."""
        from addon.text_utils import generate_search_candidates
        # 及び is not in GRAMMAR_FUNCTION_WORDS, so gating is driven purely by Yomitan metadata
        note_oyobi = {
            "Expression": "及び",
            "ExpressionReading": "および",
            "MainDefinition": (
                '<ol><li data-dictionary="JMdict [2025-11-01]">'
                '<i>(1, conj, JMdict [2025-11-01])</i> '
                '<span>and; as well as</span></li></ol>'
            )
        }
        cands = generate_search_candidates(note_oyobi["Expression"], note=note_oyobi, allow_tier2=True)
        # Should only have the primary term
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0], ("及び", "primary", 1))

    def test_yomitan_vk_compound_conjugation(self):
        """Test that ~てくる compounds with vk tag still conjugate correctly to ~てきた."""
        from addon.text_utils import generate_search_candidates
        note_kaeru = {
            "Expression": "帰ってくる",
            "ExpressionReading": "かえってくる",
            "MainDefinition": (
                '<ol><li data-dictionary="JMdict [2025-11-01]">'
                '<i>(1, vk, vi, JMdict [2025-11-01])</i> '
                '<span>to come back</span></li></ol>'
            )
        }
        cands = generate_search_candidates(note_kaeru["Expression"], note=note_kaeru, allow_tier2=True)
        cand_map = {term: (reason, tier) for term, reason, tier in cands}
        self.assertIn("帰ってきた", cand_map)
        self.assertEqual(cand_map["帰ってきた"], ("verb_inflection", 1))
        self.assertNotIn("帰ってくるた", cand_map)

    def test_yomitan_forms_all_orthography_promotion(self):
        """Test that orthographic Kanji variants from forms_all are promoted to Tier 1."""
        from addon.text_utils import generate_search_candidates
        note_kankeri = {
            "Expression": "缶けり",
            "Reading": "かんけり",
            "MainDefinition": (
                '<ol><li data-dictionary="JMdict [2025-11-01]">'
                '<i>(forms, JMdict [2025-11-01])</i> '
                '<ul><li>缶蹴り</li><li>缶けり</li></ul></li></ol>'
            ),
        }
        cands = generate_search_candidates(note_kankeri["Expression"], note=note_kankeri, allow_tier2=True)
        cand_map = {term: (reason, tier) for term, reason, tier in cands}
        self.assertIn("缶蹴り", cand_map)
        self.assertEqual(cand_map["缶蹴り"], ("yomitan_form", 1))

    def test_tier2_semantic_stems_precede_phonetic_readings(self):
        """Test that verb_stem (控え) precedes phonetic readings (ひかえる, ヒカエル) in Tier 2."""
        from addon.text_utils import generate_search_candidates
        note_hikaeru = {
            "Expression": "控える",
            "Reading": "ひかえる",
        }
        cands = generate_search_candidates(note_hikaeru["Expression"], note=note_hikaeru, allow_tier2=True)
        cand_terms = [term for term, _, _ in cands]
        self.assertIn("控え", cand_terms)
        self.assertIn("ひかえる", cand_terms)
        self.assertIn("ヒカエル", cand_terms)
        # 控え must appear before both phonetic readings
        self.assertLess(cand_terms.index("控え"), cand_terms.index("ひかえる"))
        self.assertLess(cand_terms.index("控え"), cand_terms.index("ヒカエル"))

    def test_pronoun_katakana_conversion_suppression(self):
        """Test that Japanese pronouns and demonstratives are suppressed from Katakana conversion."""
        from addon.text_utils import generate_search_candidates
        # 1. Blocklisted pronoun わし must not emit ワシ (which would match 鷲 "eagle")
        note_washi = {
            "Expression": "わし",
            "Reading": "わし",
        }
        cands_washi = generate_search_candidates(note_washi["Expression"], note=note_washi, allow_tier2=True)
        cand_terms_washi = {term for term, _, _ in cands_washi}
        self.assertIn("わし", cand_terms_washi)
        self.assertNotIn("ワシ", cand_terms_washi)

        # 2. Blocklisted pronouns/demonstratives not covered by grammar gating must not emit Katakana
        for dem in ("こっち", "あっち", "わたし", "かのじょ"):
            cands_dem = generate_search_candidates(dem, allow_tier2=True)
            cand_terms_dem = {term for term, _, _ in cands_dem}
            self.assertIn(dem, cand_terms_dem)
            self.assertFalse(any(re.fullmatch(r"[\u30a0-\u30ff]+", t) for t in cand_terms_dem))

        # 3. Dynamic suppression via Yomitan 'pn' POS tag
        note_custom_pn = {
            "Expression": "わちき",
            "Reading": "わちき",
            "MainDefinition": '<ol><li><i>(pn, JMdict)</i> I; me</li></ol>',
        }
        cands_custom = generate_search_candidates(note_custom_pn["Expression"], note=note_custom_pn, allow_tier2=True)
        cand_terms_custom = {term for term, _, _ in cands_custom}
        self.assertIn("わちき", cand_terms_custom)
        self.assertNotIn("ワチキ", cand_terms_custom)

    def test_uk_reading_precedes_forms_all_kanji(self):
        """Test that for 'uk' words, canonical kana reading precedes alternative Kanji variants from forms_all."""
        from addon.text_utils import generate_search_candidates
        note_sonomama = {
            "Expression": "其の儘",
            "Reading": "そのまま",
            "MainDefinition": (
                '<div class="yomitan-glossary">'
                '<ol>'
                '<li data-dictionary="JMdict [2025-11-01]">'
                '<i>(1, adj-no, adv, n, uk, JMdict [2025-11-01])</i> '
                '<span>as is</span>'
                '</li>'
                '<li data-dictionary="JMdict [2025-11-01]">'
                '<i>(forms, JMdict [2025-11-01])</i> '
                '<ul><li>其の侭</li><li>その侭</li></ul>'
                '</li>'
                '</ol></div>'
            ),
        }
        cands = generate_search_candidates(note_sonomama["Expression"], note=note_sonomama, allow_tier2=True)
        cand_terms = [t for t, _, _ in cands]
        self.assertIn("そのまま", cand_terms)
        self.assertIn("其の侭", cand_terms)
        # Canonical kana reading must precede alternative Kanji form
        self.assertLess(cand_terms.index("そのまま"), cand_terms.index("其の侭"))

    def test_rare_starred_forms_table_cells_excluded_from_candidates(self):
        """Test that formsTable headers with rare-marked starred cells are not emitted as candidates."""
        from addon.text_utils import generate_search_candidates
        note = {
            "Expression": "手袋",
            "Reading": "てぶくろ",
            "MainDefinition": (
                '<div class="yomitan-glossary">'
                '<table data-sc-content="formsTable">'
                '<tbody>'
                '<tr><th></th><th>手袋</th><th>〈手裹〉</th><th>腕袋</th><th>籠手</th></tr>'
                '<tr><th>てぶくろ</th><td>★</td><td>★</td><td>★⛬</td><td class="rare">★</td></tr>'
                '</tbody></table>'
                '</div>'
            ),
        }
        cands = generate_search_candidates(note["Expression"], note=note, allow_tier2=True)
        cand_map = {term: (reason, tier) for term, reason, tier in cands}
        # Standard form 手袋 is primary / star form
        self.assertIn("手袋", cand_map)
        # Marked header 〈手裹〉, rare-starred cell ★⛬ (腕袋), and rare-class cell (籠手) must not be generated
        self.assertNotIn("手裹", cand_map)
        self.assertNotIn("腕袋", cand_map)
        self.assertNotIn("籠手", cand_map)

    def test_yomitan_forms_all_capped_at_three(self):
        """Test that Tier 1 candidates from forms_all are capped at 3 newly added Kanji forms."""
        from addon.text_utils import generate_search_candidates
        # Expression is 見本一, which is already in seen as primary
        note = {
            "Expression": "見本一",
            "Reading": "みほんいち",
            "MainDefinition": (
                '<div class="yomitan-glossary">'
                '<ol><li data-dictionary="JMdict [2025-11-01]">'
                '<i>(forms, JMdict [2025-11-01])</i> '
                '<ul>'
                '<li>見本一</li>'
                '<li>見本二</li>'
                '<li>見本三</li>'
                '<li>見本四</li>'
                '<li>見本五</li>'
                '</ul></li></ol>'
                '</div>'
            ),
        }
        cands = generate_search_candidates(note["Expression"], note=note, allow_tier2=True)
        form_cands = [term for term, reason, _ in cands if reason == "yomitan_form"]
        # 見本一 was already seen, so exactly 3 new forms (見本二, 見本三, 見本四) are added
        self.assertEqual(len(form_cands), 3)
        self.assertEqual(form_cands, ["見本二", "見本三", "見本四"])

    def test_generate_search_candidates_emits_concept_map(self) -> None:
        """Verify generate_search_candidates emits manual concept map targets in Tier 2."""
        with patch("addon.text_utils.lookup_concept_map", return_value=["お葬式", "墓"]):
            cands = generate_search_candidates("亡くなる", allow_tier2=True)
            concept_cands = [c for c in cands if c[1] == "concept_map"]
            self.assertEqual(len(concept_cands), 2)
            self.assertEqual(concept_cands[0], ("お葬式", "concept_map", 2))
            self.assertEqual(concept_cands[1], ("墓", "concept_map", 2))

    def test_generate_search_candidates_deinflected_concept_map(self) -> None:
        """Verify generate_search_candidates deinflects surface term to match concept map."""
        with patch("addon.text_utils.lookup_concept_map", side_effect=lambda term: ["獲得", "宝箱"] if term == "手に入れる" else []):
            cands = generate_search_candidates("手に入れている", allow_tier2=True, allow_deinflection=True)
            concept_cands = [c for c in cands if c[1] == "concept_map"]
            self.assertEqual(len(concept_cands), 2)
            self.assertEqual(concept_cands[0], ("獲得", "concept_map", 2))
            self.assertEqual(concept_cands[1], ("宝箱", "concept_map", 2))

    def test_generate_search_candidates_suppressed_when_tier2_false(self) -> None:
        """Verify concept map targets are suppressed when allow_tier2 is False."""
        with patch("addon.text_utils.lookup_concept_map", return_value=["お葬式", "墓"]):
            cands = generate_search_candidates("亡くなる", allow_tier2=False)
            concept_cands = [c for c in cands if c[1] == "concept_map"]
            self.assertEqual(len(concept_cands), 0)

    def test_generate_search_candidates_injected_concept_lookup(self) -> None:
        """Verify generate_search_candidates uses injected concept_lookup callable without patching."""
        cands = generate_search_candidates(
            "亡くなる",
            allow_tier2=True,
            concept_lookup=lambda term: ["お葬式", "墓"] if term == "亡くなる" else [],
        )
        concept_cands = [c for c in cands if c[1] == "concept_map"]
        self.assertEqual(len(concept_cands), 2)
        self.assertEqual(concept_cands[0], ("お葬式", "concept_map", 2))
        self.assertEqual(concept_cands[1], ("墓", "concept_map", 2))

    def test_generate_search_candidates_emits_kaishi_concept(self) -> None:
        """Verify generate_search_candidates emits Kaishi concept map targets in Tier 2."""
        with patch("addon.text_utils.lookup_kaishi_concept", return_value=["「ダメ」のポーズをする人"]):
            cands = generate_search_candidates("全然", allow_tier2=True)
            kaishi_cands = [c for c in cands if c[1] == "kaishi_concept"]
            self.assertEqual(len(kaishi_cands), 1)
            self.assertEqual(kaishi_cands[0], ("「ダメ」のポーズをする人", "kaishi_concept", 2))

    def test_generate_search_candidates_ordering_manual_then_kaishi_then_reading(self) -> None:
        """Verify candidate ordering: concept_map (manual) -> kaishi_concept -> note_reading."""
        class MockNote(dict):
            pass

        mock_note = MockNote({"Word": "走る", "Reading": "はしる"})
        with patch("addon.text_utils.lookup_concept_map", return_value=["マニュアル走る"]):
            with patch("addon.text_utils.lookup_kaishi_concept", return_value=["カイシ走る"]):
                cands = generate_search_candidates("走る", note=mock_note, allow_tier2=True)
                reasons = [c[1] for c in cands]
                # concept_map must appear before kaishi_concept
                # kaishi_concept must appear before note_reading
                self.assertIn("concept_map", reasons)
                self.assertIn("kaishi_concept", reasons)
                self.assertIn("note_reading_Reading", reasons)
                idx_manual = reasons.index("concept_map")
                idx_kaishi = reasons.index("kaishi_concept")
                idx_reading = reasons.index("note_reading_Reading")
                self.assertLess(idx_manual, idx_kaishi)
                self.assertLess(idx_kaishi, idx_reading)

    def test_generate_search_candidates_deinflected_kaishi_concept(self) -> None:
        """Verify generate_search_candidates deinflects surface term to match kaishi concept map."""
        with patch(
            "addon.text_utils.lookup_kaishi_concept",
            side_effect=lambda term: ["スタートダッシュ"] if term == "始まる" else [],
        ):
            cands = generate_search_candidates("始まっている", allow_tier2=True, allow_deinflection=True)
            kaishi_cands = [c for c in cands if c[1] == "kaishi_concept"]
            self.assertEqual(len(kaishi_cands), 1)
            self.assertEqual(kaishi_cands[0], ("スタートダッシュ", "kaishi_concept", 2))

    def test_generate_search_candidates_injected_kaishi_lookup(self) -> None:
        """Verify generate_search_candidates uses injected kaishi_lookup callable."""
        cands = generate_search_candidates(
            "全然",
            allow_tier2=True,
            kaishi_lookup=lambda term: ["「ダメ」のポーズをする人"] if term == "全然" else [],
        )
        kaishi_cands = [c for c in cands if c[1] == "kaishi_concept"]
        self.assertEqual(len(kaishi_cands), 1)
        self.assertEqual(kaishi_cands[0], ("「ダメ」のポーズをする人", "kaishi_concept", 2))

    def test_generate_search_candidates_emits_sudachi_synonym(self) -> None:
        """Verify generate_search_candidates emits Sudachi synonym targets in Tier 2."""
        with patch("addon.text_utils.lookup_sudachi_synonyms", return_value=["寝る", "睡眠"]):
            cands = generate_search_candidates("就寝", allow_tier2=True)
            sudachi_cands = [c for c in cands if c[1] == "sudachi_synonym"]
            self.assertEqual(len(sudachi_cands), 2)
            self.assertEqual(sudachi_cands[0], ("寝る", "sudachi_synonym", 2))
            self.assertEqual(sudachi_cands[1], ("睡眠", "sudachi_synonym", 2))

    def test_generate_search_candidates_deinflected_sudachi_synonym(self) -> None:
        """Verify generate_search_candidates deinflects surface term to match Sudachi synonym."""
        with patch(
            "addon.text_utils.lookup_sudachi_synonyms",
            side_effect=lambda term: ["駆ける"] if term == "走る" else [],
        ):
            cands = generate_search_candidates("走っている", allow_tier2=True, allow_deinflection=True)
            sudachi_cands = [c for c in cands if c[1] == "sudachi_synonym"]
            self.assertEqual(len(sudachi_cands), 1)
            self.assertEqual(sudachi_cands[0], ("駆ける", "sudachi_synonym", 2))

    def test_generate_search_candidates_sudachi_suppressed_when_tier2_false(self) -> None:
        """Verify Sudachi synonym targets are suppressed when allow_tier2 is False."""
        with patch("addon.text_utils.lookup_sudachi_synonyms", return_value=["寝る"]):
            cands = generate_search_candidates("就寝", allow_tier2=False)
            sudachi_cands = [c for c in cands if c[1] == "sudachi_synonym"]
            self.assertEqual(len(sudachi_cands), 0)

    def test_generate_search_candidates_injected_sudachi_lookup(self) -> None:
        """Verify generate_search_candidates uses injected sudachi_lookup callable."""
        cands = generate_search_candidates(
            "就寝",
            allow_tier2=True,
            sudachi_lookup=lambda term: ["寝る"] if term == "就寝" else [],
        )
        sudachi_cands = [c for c in cands if c[1] == "sudachi_synonym"]
        self.assertEqual(len(sudachi_cands), 1)
        self.assertEqual(sudachi_cands[0], ("寝る", "sudachi_synonym", 2))

    def test_generate_search_candidates_case_insensitive_sudachi_synonym(self) -> None:
        """Verify generate_search_candidates resolves Sudachi synonyms for Latin terms regardless of query case."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_catalog.sqlite")
            conn = sqlite3.connect(db_path)
            create_schema(conn)
            conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '1')")
            write_synonyms(conn, source="sudachi", mapping={"Cupid": ["キューピッド"]})
            conn.commit()
            conn.close()

            test_store = CatalogStore(db_path)

            def real_lookup(term: str) -> List[str]:
                """Look up Sudachi synonyms from isolated test store."""
                return lookup_sudachi_synonyms(term, store=test_store)

            for q in ("cupid", "Cupid", "CUPID"):
                cands = generate_search_candidates(
                    q,
                    allow_tier2=True,
                    sudachi_lookup=real_lookup,
                )
                sudachi_cands = [c for c in cands if c[1] == "sudachi_synonym"]
                self.assertEqual(len(sudachi_cands), 1)
                self.assertEqual(sudachi_cands[0], ("キューピッド", "sudachi_synonym", 2))

    def test_candidate_ordering_tier2_hierarchy_with_sudachi(self) -> None:
        """Verify candidate ordering: concept_map -> kaishi_concept -> note_reading -> same_field -> sudachi_synonym."""
        mock_note = {"Reading": "そうこう"}
        with patch("addon.text_utils.lookup_concept_map", return_value=["マニュアル走る"]):
            with patch("addon.text_utils.lookup_kaishi_concept", return_value=["カイシ走る"]):
                with patch("addon.text_utils.lookup_sudachi_synonyms", return_value=["スダチ走る"]):
                    cands = generate_search_candidates("走る[はしる]", note=mock_note, allow_tier2=True)
                    reasons = [c[1] for c in cands]
                    self.assertIn("concept_map", reasons)
                    self.assertIn("kaishi_concept", reasons)
                    self.assertIn("note_reading_Reading", reasons)
                    self.assertIn("same_field_reading_katakana", reasons)
                    self.assertIn("sudachi_synonym", reasons)

                    idx_manual = reasons.index("concept_map")
                    idx_kaishi = reasons.index("kaishi_concept")
                    idx_reading = reasons.index("note_reading_Reading")
                    idx_same_kana = reasons.index("same_field_reading_katakana")
                    idx_sudachi = reasons.index("sudachi_synonym")

                    self.assertLess(idx_manual, idx_kaishi)
                    self.assertLess(idx_kaishi, idx_reading)
                    self.assertLess(idx_reading, idx_same_kana)
                    self.assertLess(idx_same_kana, idx_sudachi)

    def test_generate_search_candidates_emits_bilingual_gloss(self) -> None:
        """Verify generate_search_candidates emits Tier 2 candidates from bilingual definition glosses."""
        mock_note = {"Expression": "亡くなる", "Meaning": "to pass away"}
        with patch(
            "addon.synonyms.lookup_bilingual_glosses",
            return_value=["死ぬ", "葬式"],
        ):
            cands = generate_search_candidates("亡くなる", note=mock_note, allow_tier2=True)
            b_cands = [c for c in cands if c[1] == "bilingual_gloss"]
            self.assertEqual(len(b_cands), 2)
            self.assertEqual(b_cands[0], ("死ぬ", "bilingual_gloss", 2))
            self.assertEqual(b_cands[1], ("葬式", "bilingual_gloss", 2))

    def test_generate_search_candidates_bilingual_suppressed_when_tier2_false(self) -> None:
        """Verify bilingual gloss candidates are suppressed when allow_tier2 is False."""
        mock_note = {"Expression": "亡くなる", "Meaning": "to pass away"}
        with patch(
            "addon.synonyms.lookup_bilingual_glosses",
            return_value=["死ぬ"],
        ):
            cands = generate_search_candidates("亡くなる", note=mock_note, allow_tier2=False)
            b_cands = [c for c in cands if c[1] == "bilingual_gloss"]
            self.assertEqual(len(b_cands), 0)

    def test_generate_search_candidates_injected_bilingual_lookup(self) -> None:
        """Verify generate_search_candidates uses injected bilingual_lookup callable."""
        mock_note = {"Expression": "疾走", "Meaning": "sprint"}
        cands = generate_search_candidates(
            "疾走",
            note=mock_note,
            allow_tier2=True,
            bilingual_lookup=lambda note, meta: ["全力疾走", "ランニング"],
        )
        b_cands = [c for c in cands if c[1] == "bilingual_gloss"]
        self.assertEqual(len(b_cands), 2)
        self.assertEqual(b_cands[0], ("全力疾走", "bilingual_gloss", 2))
        self.assertEqual(b_cands[1], ("ランニング", "bilingual_gloss", 2))

    def test_generate_search_candidates_grammar_pos_suppresses_bilingual(self) -> None:
        """Verify grammar particle/auxiliary notes suppress Tier 2 bilingual glosses."""
        mock_note = {
            "Expression": "及び",
            "MainDefinition": '<ol><li><i>(conj, JMdict)</i> and</li></ol>',
        }
        cands = generate_search_candidates(
            "及び",
            note=mock_note,
            allow_tier2=True,
            bilingual_lookup=lambda note, meta: ["出発"],
        )
        b_cands = [c for c in cands if c[1] == "bilingual_gloss"]
        self.assertEqual(len(b_cands), 0)


if __name__ == "__main__":
    unittest.main()


