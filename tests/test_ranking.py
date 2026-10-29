"""Unit tests for boundary classification, relevance scoring, and archetype ranking."""
import unittest
from unittest.mock import patch

from addon.feed import IrasutoyaImage
from addon.ranking import (
    PRONOUNS_DESCRIPTION_BLOCKLIST,
    _is_byproduct_modifier_title,
    calculate_category_score,
    classify_match_boundary,
    has_title_illustration_match,
    infer_note_pos_type,
    infer_pos_type,
    is_callout_asset,
    is_decorative_asset,
    is_image_relevant,
    is_title_match,
    rank_relevant_images,
    score_candidate_image,
    select_best_image,
)
from addon.text_utils import generate_search_candidates


class TestRanking(unittest.TestCase):
    """Test suite for boundary classification, image relevance heuristics, and ranking."""

    def setUp(self) -> None:
        """Isolate ranking and candidate evaluation from external catalog concept lookups."""
        patcher = patch("addon.text_utils.lookup_concept_map", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)
        k_patcher = patch("addon.text_utils.lookup_kaishi_concept", return_value=[])
        k_patcher.start()
        self.addCleanup(k_patcher.stop)

    def test_score_candidate_image(self):
        """Test archetype scoring engine penalties and boosts."""
        img_exact = IrasutoyaImage(
            title="本",
            image_url="https://example.com/book.png",
            thumb_url="https://example.com/thumb.png",
            post_title="本のイラスト",
        )
        score_exact = score_candidate_image(img_exact, "本")

        img_tax = IrasutoyaImage(
            title="確定申告で税金を払う人のイラスト",
            image_url="https://example.com/tax.png",
            thumb_url="https://example.com/thumb.png",
            post_title="税金・確定申告のイラスト",
        )
        score_tax = score_candidate_image(img_tax, "本")

        self.assertGreater(score_exact, score_tax)
        self.assertGreater(score_exact, 100)
        self.assertLess(score_tax, 50)

    def test_select_best_image(self):
        """Test selecting the highest scoring image among candidates."""
        img_bad = IrasutoyaImage(
            title="本を投げ捨てる問題児のイラスト",
            image_url="https://example.com/bad.png",
            thumb_url="https://example.com/thumb.png",
            post_title="トラブルのイラスト",
        )
        img_good = IrasutoyaImage(
            title="開いた本のイラスト",
            image_url="https://example.com/good.png",
            thumb_url="https://example.com/thumb.png",
            post_title="本のイラスト",
        )
        best = select_best_image([img_bad, img_good], "本")
        self.assertEqual(best, img_good)

    def test_rank_relevant_images(self):
        """Verify rank_relevant_images orders non-callout illustrations before callouts even when callout scores higher."""
        callout = IrasutoyaImage("合格のイラスト文字", "https://example.com/c.png", "https://example.com/tc.png", "合格")
        scene_archetype = IrasutoyaImage("合格のイラスト", "https://example.com/s1.png", "https://example.com/ts1.png", "合格のイラスト")
        scene_low = IrasutoyaImage("合格発表を見に行って落ち込む人のイラスト", "https://example.com/s2.png", "https://example.com/ts2.png", "合格")

        self.assertTrue(is_callout_asset(callout, "合格"))
        self.assertFalse(is_callout_asset(scene_low, "合格"))

        ranked = rank_relevant_images([callout, scene_low, scene_archetype], "合格")
        self.assertEqual(ranked, [scene_archetype, scene_low, callout])

    def test_is_decorative_asset(self):
        """Verify decorative noise detection for frames, borders, and templates."""
        frame_img = IrasutoyaImage("花のフレーム", "https://example.com/f.png", "https://example.com/t.png", "花")
        self.assertTrue(is_decorative_asset(frame_img, "花"))
        # Explicit search for frame should not be treated as unwanted decorative noise
        self.assertFalse(is_decorative_asset(frame_img, "フレーム"))

        cat_img = IrasutoyaImage("かわいい猫のイラスト", "https://example.com/c.png", "https://example.com/t.png", "猫")
        self.assertFalse(is_decorative_asset(cat_img, "猫"))

    def test_is_image_relevant(self):
        """Verify image relevance checking against queries, preventing multi-image post mismatches."""
        # Relevant: image title directly contains query
        cat_img = IrasutoyaImage("かわいい猫のイラスト", "https://example.com/c.png", "https://example.com/t.png", "色々な動物のイラスト")
        self.assertTrue(is_image_relevant(cat_img, "猫"))

        # Unrelated: post matched query but individual image is unrelated
        dog_in_cat_post = IrasutoyaImage("犬のイラスト", "https://example.com/d.png", "https://example.com/t.png", "猫のイラスト")
        self.assertFalse(is_image_relevant(dog_in_cat_post, "猫"))

        # Inherited title: image lacked alt so title is identical to post title
        single_post_img = IrasutoyaImage("猫のイラスト", "https://example.com/c2.png", "https://example.com/t.png", "猫のイラスト")
        self.assertTrue(is_image_relevant(single_post_img, "猫"))

        # Multi-word query
        multi_img = IrasutoyaImage("黒い猫のイラスト", "https://example.com/b.png", "https://example.com/t.png", "黒い猫")
        self.assertTrue(is_image_relevant(multi_img, "黒い 猫"))
        self.assertFalse(is_image_relevant(multi_img, "白い 猫"))

        # Okurigana verb stems fused with Kanji must be rejected (売り場 != 売り, 食べ物 != 食べ, 買い物 != 買い)
        uriba_img = IrasutoyaImage("売り場のイラスト", "https://example.com/u.png", "https://example.com/t.png", "売り場")
        self.assertFalse(is_image_relevant(uriba_img, "売り"))
        tabemono_img = IrasutoyaImage("食べ物のイラスト", "https://example.com/tb.png", "https://example.com/t.png", "食べ物")
        self.assertFalse(is_image_relevant(tabemono_img, "食べ"))
        kaimono_img = IrasutoyaImage("買い物のイラスト", "https://example.com/km.png", "https://example.com/t.png", "買い物")
        self.assertFalse(is_image_relevant(kaimono_img, "買い"))

        # Single-Kanji noun compounds must be rejected as irrelevant (乾燥肌 != 肌, 北京ダック != 北)
        hada_img = IrasutoyaImage("乾燥肌のイラスト", "https://example.com/h.png", "https://example.com/t.png", "乾燥肌")
        self.assertFalse(is_image_relevant(hada_img, "肌"))
        # Polite-prefix nouns (お肌) remain relevant for single-Kanji nouns
        ohada_img = IrasutoyaImage("つるつるのお肌の女性のイラスト", "https://example.com/oh.png", "https://example.com/t.png", "お肌のイラスト")
        self.assertTrue(is_image_relevant(ohada_img, "肌"))
        matsuri_img = IrasutoyaImage("祭り会場のイラスト", "https://example.com/m.png", "https://example.com/t.png", "祭り会場")
        self.assertTrue(is_image_relevant(matsuri_img, "祭り"))
        summer_matsuri_img = IrasutoyaImage("夏祭りのイラスト", "https://example.com/sm.png", "https://example.com/t.png", "夏祭り")
        self.assertTrue(is_image_relevant(summer_matsuri_img, "祭り"))
        akai_car_img = IrasutoyaImage("赤い車のイラスト", "https://example.com/ac.png", "https://example.com/t.png", "赤い車")
        self.assertTrue(is_image_relevant(akai_car_img, "赤い"))
        ookii_tree_img = IrasutoyaImage("大きい木のイラスト", "https://example.com/ot.png", "https://example.com/t.png", "大きい木")
        self.assertTrue(is_image_relevant(ookii_tree_img, "大きい"))
        # Masu-stems preceded by Kanji should remain relevant (夏休み is relevant to 休み)
        yasumi_img = IrasutoyaImage("夏休みのイラスト", "https://example.com/y.png", "https://example.com/t.png", "夏休み")
        self.assertTrue(is_image_relevant(yasumi_img, "休み"))

        # Case-insensitive Latin queries
        cupid_img = IrasutoyaImage("Cupidのイラスト", "https://example.com/cupid.png", "https://example.com/thumb.png", "Cupidのイラスト")
        self.assertTrue(is_image_relevant(cupid_img, "cupid"))
        self.assertTrue(is_image_relevant(cupid_img, "Cupid"))
        self.assertFalse(is_image_relevant(cupid_img, "cup"))
        smr_img = IrasutoyaImage("SMRのキャラクター", "https://example.com/smr.png", "https://example.com/thumb.png", "SMRのイラスト")
        self.assertTrue(is_image_relevant(smr_img, "smr"))
        e_img = IrasutoyaImage("Éのイラスト", "https://example.com/e.png", "https://example.com/t.png", "Éのイラスト")
        self.assertTrue(is_image_relevant(e_img, "é"))

    def test_classify_match_boundary(self):
        """Test fine-grained morphological boundary classification."""
        # Quoted concepts
        self.assertEqual(classify_match_boundary("交通安全のイラスト「聞く」", "聞く"), "exact_quoted")
        self.assertEqual(classify_match_boundary("地球のイラスト（笑顔）", "笑顔"), "exact_quoted")
        self.assertEqual(classify_match_boundary("本のイラスト", "本"), "exact_quoted")

        # Polite prefix
        self.assertEqual(classify_match_boundary("お肉のイラスト", "肉"), "polite_prefix")
        self.assertEqual(classify_match_boundary("お茶のイラスト", "茶"), "polite_prefix")

        # Standalone
        self.assertEqual(classify_match_boundary("赤い本のイラスト", "本"), "standalone")
        # Bounded 2-mora Katakana queries
        self.assertEqual(classify_match_boundary("焼きたてのパンのイラスト", "パン"), "standalone")
        self.assertEqual(classify_match_boundary("黒いネコのイラスト", "ネコ"), "standalone")
        self.assertEqual(classify_match_boundary("おいしいピザのイラスト", "ピザ"), "standalone")
        # Bounded 2-mora Hiragana loanwords in SAFE_2MORA_LOANWORDS
        self.assertEqual(classify_match_boundary("焼きたてのぱんのイラスト", "ぱん"), "standalone")
        # Hiragana queries preceded by case particles
        self.assertEqual(classify_match_boundary("犬がごはんを食べているイラスト", "ごはん"), "standalone")
        # Hiragana queries with trailing mid-word collision
        self.assertEqual(classify_match_boundary("さくらんぼのイラスト", "さくら"), "none")
        self.assertEqual(classify_match_boundary("綺麗なさくらのイラスト", "さくら"), "standalone")
        # Hiragana queries preceded by attributive endings (い, る, た, だ)
        self.assertEqual(classify_match_boundary("赤いりんごのイラスト", "りんご"), "standalone")
        self.assertEqual(classify_match_boundary("走るおとこのこのイラスト", "おとこのこ"), "standalone")
        # Unbounded / collision Katakana queries
        self.assertEqual(classify_match_boundary("フライパンのイラスト", "パン"), "none")

        # Verb inflections
        self.assertEqual(classify_match_boundary("勝った人のイラスト", "勝つ"), "inflected")
        self.assertEqual(classify_match_boundary("不用品を売ろうとする人のイラスト", "売る"), "inflected")

        # Kanji compound affixes
        self.assertEqual(classify_match_boundary("乾燥肌のイラスト", "肌"), "compound")
        self.assertEqual(classify_match_boundary("中性的な人のイラスト", "中"), "compound")

        # Mid-word collisions should be classified as none
        self.assertEqual(classify_match_boundary("カフェオレのイラスト", "オレ"), "none")
        self.assertEqual(classify_match_boundary("芋けんぴのイラスト", "けん"), "none")

        # Katakana compounds separated by middle dot (・) should match as standalone
        self.assertEqual(classify_match_boundary("ハンバーガー・ポテトのイラスト", "ポテト"), "standalone")
        self.assertEqual(classify_match_boundary("ハンバーガー・ポテトのイラスト", "ハンバーガー"), "standalone")

        # Mixed kana/kanji query ending in Kanji followed by Kanji should be classified as compound
        self.assertEqual(classify_match_boundary("お金持ちのイラスト", "お金"), "compound")
        self.assertEqual(classify_match_boundary("お茶碗のイラスト", "お茶"), "compound")
        self.assertEqual(classify_match_boundary("たくさんのお金のイラスト", "お金"), "standalone")
        self.assertEqual(classify_match_boundary("パンを食べる人のイラスト", "食べる"), "standalone")

        # Inflected matches should ignore bare stems forming Kanji compound nouns
        self.assertEqual(classify_match_boundary("食べ物のイラスト", "食べる"), "none")
        self.assertEqual(classify_match_boundary("売り場のイラスト", "売る"), "none")
        self.assertEqual(classify_match_boundary("食べ物とご飯を食べた人のイラスト", "食べる"), "inflected")

        # Verb stems ending in okurigana forming nouns should be classified as compound (not standalone)
        self.assertEqual(classify_match_boundary("買い物のイラスト", "買い"), "compound")
        self.assertEqual(classify_match_boundary("食べ物のイラスト", "食べ"), "compound")
        self.assertEqual(classify_match_boundary("売り場のイラスト", "売り"), "compound")
        # True adjectives modifying nouns should be classified as standalone
        self.assertEqual(classify_match_boundary("赤い車のイラスト", "赤い"), "standalone")
        self.assertEqual(classify_match_boundary("美しい風景のイラスト", "美しい"), "standalone")
        self.assertEqual(classify_match_boundary("大きい木のイラスト", "大きい"), "standalone")
        self.assertEqual(classify_match_boundary("小さい犬のイラスト", "小さい"), "standalone")
        # Established okurigana nouns forming compounds with following Kanji
        self.assertEqual(classify_match_boundary("祭り会場のイラスト", "祭り"), "compound")
        self.assertEqual(classify_match_boundary("夏祭りのイラスト", "祭り"), "compound")

        # Adjectives with な particle preceding Kanji nouns
        self.assertEqual(classify_match_boundary("大きな犬のイラスト", "犬"), "standalone")
        self.assertEqual(classify_match_boundary("きれいな花のイラスト", "花"), "standalone")

        # Attributive plain-form verbs preceding pure Kanji nouns
        self.assertEqual(classify_match_boundary("歌う人のイラスト", "人"), "standalone")
        self.assertEqual(classify_match_boundary("歩く犬のイラスト", "犬"), "standalone")
        self.assertEqual(classify_match_boundary("飲む人のイラスト", "人"), "standalone")

        # Hiragana て-form and multi-char particles/connectives preceding pure Kanji nouns
        self.assertEqual(classify_match_boundary("座って本を読む人のイラスト", "本"), "standalone")
        self.assertEqual(classify_match_boundary("ここから本を借りる人のイラスト", "本"), "standalone")
        self.assertEqual(classify_match_boundary("笑いながら本を読む人のイラスト", "本"), "standalone")
        self.assertEqual(classify_match_boundary("机の上など本を置く人のイラスト", "本"), "standalone")
        self.assertEqual(classify_match_boundary("自分だけ本を読む人のイラスト", "本"), "standalone")
        self.assertEqual(classify_match_boundary("山ほど本がある部屋のイラスト", "本"), "standalone")
        self.assertEqual(classify_match_boundary("少しばかり本を読む人のイラスト", "本"), "standalone")
        self.assertEqual(classify_match_boundary("夜遅くまで本を読む人のイラスト", "本"), "standalone")

        # Compounds with preceding Katakana or non-particle Hiragana
        self.assertEqual(classify_match_boundary("くるりポイ禁止のイラスト", "禁止"), "compound")
        self.assertEqual(classify_match_boundary("ポイ捨て禁止のイラスト", "禁止"), "compound")
        # Canonical mark titles matching query
        self.assertEqual(classify_match_boundary("禁止マークのイラスト", "禁止"), "exact_quoted")
        self.assertEqual(classify_match_boundary("初心者マークのイラスト", "初心者"), "exact_quoted")

        # Case-insensitive Latin queries
        self.assertEqual(classify_match_boundary("Cupidのイラスト", "cupid"), "exact_quoted")
        self.assertEqual(classify_match_boundary("cupidのイラスト", "Cupid"), "exact_quoted")
        self.assertEqual(classify_match_boundary("SMRのイラスト", "smr"), "exact_quoted")
        self.assertEqual(classify_match_boundary("AIに取り込まれた人のイラスト", "ai"), "standalone")
        self.assertEqual(classify_match_boundary("リアルタイムPCRのイラスト", "pcr"), "compound")
        # Embedded Latin word collision should be rejected
        self.assertEqual(classify_match_boundary("Cupidのイラスト", "cup"), "none")
        self.assertEqual(classify_match_boundary("Catching upのイラスト", "cat"), "none")

        # Expanding characters in casefold (e.g. ß -> ss) must preserve adjacent offsets
        self.assertEqual(classify_match_boundary("Straße Catのイラスト", "cat"), "standalone")

        # Untrimmed leading whitespace must not shift prefix evaluation
        self.assertEqual(classify_match_boundary("  はくちょうのイラスト", "くちょう"), "none")

        # Accented Latin characters (e.g. É -> é, Café -> café)
        self.assertEqual(classify_match_boundary("Éのイラスト", "é"), "exact_quoted")
        self.assertEqual(classify_match_boundary("Caféのイラスト", "café"), "exact_quoted")

        # Mixed-script Latin+Japanese inflections
        self.assertEqual(classify_match_boundary("PCを使った人のイラスト", "pcを使う"), "inflected")
        self.assertEqual(classify_match_boundary("AIを使ったPCのイラスト", "AIを使う"), "inflected")
        self.assertEqual(classify_match_boundary("SDX化した人のイラスト", "dx化する"), "none")

    def test_score_callout_text_heavy_penalty(self):
        """Test that illustrated text callouts receive heavy penalty instead of hard rejection."""
        img_callout = IrasutoyaImage(
            title="「決定！」のイラスト文字",
            image_url="https://example.com/kettei_text.png",
            thumb_url="https://example.com/thumb.png",
            post_title="「決定！」のイラスト文字",
        )
        img_scene = IrasutoyaImage(
            title="決定を下す人のイラスト",
            image_url="https://example.com/kettei_scene.png",
            thumb_url="https://example.com/thumb.png",
            post_title="決定のイラスト",
        )

        # Callout text is not hard-rejected by is_decorative_asset
        self.assertFalse(is_decorative_asset(img_callout, "決定"))
        self.assertTrue(is_image_relevant(img_callout, "決定"))

        # But scene illustration scores significantly higher than text callout
        score_callout = score_candidate_image(img_callout, "決定")
        score_scene = score_candidate_image(img_scene, "決定")
        self.assertGreater(score_scene, score_callout)

        # Even when the callout has an exact quoted match and canonical post title,
        # select_best_image prefers the scene illustration
        img_callout_canonical = IrasutoyaImage(
            title="「決定」のイラスト文字",
            image_url="https://example.com/kettei_text2.png",
            thumb_url="https://example.com/thumb.png",
            post_title="決定のイラスト",
        )
        self.assertGreater(score_scene, score_candidate_image(img_callout_canonical, "決定"))
        best = select_best_image([img_callout_canonical, img_scene], "決定")
        self.assertEqual(best, img_scene)

        # Verify exact callout penalty (-65.0) as specified in Issue #7
        img_plain_callout = IrasutoyaImage(
            title="決定のイラスト文字",
            image_url="https://example.com/plain_callout.png",
            thumb_url="https://example.com/thumb.png",
            post_title="その他のイラスト",
        )
        with patch("addon.ranking.is_callout_asset", side_effect=[True, False]):
            score_callout = score_candidate_image(img_plain_callout, "決定")
            score_non_callout = score_candidate_image(img_plain_callout, "決定")
            self.assertEqual(score_non_callout - score_callout, 65.0)

    def test_single_image_post_description_relevance(self):
        """Test accepting relevant single-image posts whose description matches query."""
        single_img = IrasutoyaImage(
            title="ガッツポーズをする人のイラスト",
            image_url="https://example.com/win.png",
            thumb_url="https://example.com/thumb.png",
            post_title="ガッツポーズをする人のイラスト",
            description="勝った人や勝利を喜んでいる人のイラストです。",
        )
        single_img.is_single_image_post = True

        # Matches '勝利' in caption of single image post
        self.assertTrue(is_image_relevant(single_img, "勝利"))

        # On multi-image posts, description alone is not sufficient
        single_img.is_single_image_post = False
        self.assertFalse(is_image_relevant(single_img, "勝利"))

    def test_candidate_regression_fixture(self):
        """Automated regression fixture verifying candidate generation and boundary matching on Mining and Kaishi 1.5k decks."""
        # Mining vocabulary dataset: (card_term, reading, target_illustration_title)
        mining_dataset = [
            ("1杯", "いっぱい", "一杯のイラスト"),
            ("肉", "にく", "お肉のイラスト"),
            ("茶", "ちゃ", "お茶のイラスト"),
            ("勝つ", "かつ", "勝った人のイラスト"),
            ("疲れる", "つかれる", "疲れた人のイラスト"),
            ("走る", "はしる", "走る人のイラスト"),
            ("とまと", "とまと", "トマトのイラスト"),
            ("猫", "ねこ", "かわいい猫のイラスト"),
            ("本", "ほん", "本のイラスト"),
            ("犬", "いぬ", "犬のイラスト"),
        ]

        # Kaishi 1.5k vocabulary dataset: (card_term, reading, target_illustration_title)
        kaishi_dataset = [
            ("其の儘[そのまま]", None, "そのままのイラスト"),
            ("動[うご]かす", "うごかす", "動かす人のイラスト"),
            ("帰ってくる", "かえってくる", "帰る人のイラスト"),
            ("お 前[まえ]", "おまえ", "お前のイラスト"),
            ("林檎[りんご]", "りんご", "りんごのイラスト"),
            ("檸檬[れもん]", "れもん", "レモンのイラスト"),
            ("食べる[たべる]", "たべる", "食べる人のイラスト"),
            ("見る", "みる", "見る人のイラスト"),
            ("売る", "うる", "売る人のイラスト"),
            ("車", "くるま", "車のイラスト"),
        ]

        for deck_name, dataset in [("Mining", mining_dataset), ("Kaishi 1.5k", kaishi_dataset)]:
            for term, reading, target_title in dataset:
                with self.subTest(deck=deck_name, term=term):
                    note = {"Expression": term, "Reading": reading} if reading else {"Expression": term}
                    candidates = generate_search_candidates(
                        term, note=note, allow_tier2=True, allow_deinflection=True
                    )
                    self.assertTrue(
                        any(
                            classify_match_boundary(target_title, c[0])
                            in ("exact_quoted", "standalone", "inflected", "polite_prefix")
                            for c in candidates
                        ),
                        f"No candidate for '{term}' matches '{target_title}'",
                    )

        # Boundary safety verification (preventing homophone / substring collisions)
        negative_cases = [
            ("さくらんぼのイラスト", "さくら"),
            ("カフェオレのイラスト", "オレ"),
            ("芋けんぴのイラスト", "けん"),
            ("食べ物のイラスト", "食べる"),
            ("食べ物のイラスト", "食べ"),
            ("売り場のイラスト", "売る"),
            ("売り場のイラスト", "売り"),
            ("勝ち組のイラスト", "勝ち"),
            ("買い物のイラスト", "買い"),
            ("中性的な人のイラスト", "中"),
        ]
        for title, query in negative_cases:
            with self.subTest(negative_case=query):
                quality = classify_match_boundary(title, query)
                self.assertIn(quality, ("none", "compound"), f"Query '{query}' in '{title}' must not match as standalone/inflected")

    def test_category_weights_boost_action_for_verbs(self):
        """Verify action/pose categories boost human movement illustrations when query is a verb."""
        img_human = IrasutoyaImage(
            title="走る女の子のイラスト",
            image_url="https://example.com/girl.png",
            thumb_url="https://example.com/t.png",
            post_title="走る女の子のイラスト",
            categories=("こども", "ポーズ", "運動"),
        )
        img_animal = IrasutoyaImage(
            title="走る犬のイラスト",
            image_url="https://example.com/dog.png",
            thumb_url="https://example.com/t.png",
            post_title="走る犬のイラスト",
            categories=("犬",),
        )

        score_human_verb = score_candidate_image(img_human, "走る", pos_type="verb")
        score_animal_verb = score_candidate_image(img_animal, "走る", pos_type="verb")
        self.assertGreater(score_human_verb, score_animal_verb)

        best = select_best_image([img_animal, img_human], "走る", pos_type="verb")
        self.assertEqual(best, img_human)

    def test_category_weights_boost_objects_for_nouns(self):
        """Verify physical object categories boost tangible entities when query is a noun."""
        img_obj = IrasutoyaImage(
            title="青い車のイラスト",
            image_url="https://example.com/car.png",
            thumb_url="https://example.com/t.png",
            post_title="車のイラスト",
            categories=("車", "乗り物"),
        )
        img_action = IrasutoyaImage(
            title="車を運転する人のイラスト",
            image_url="https://example.com/driver.png",
            thumb_url="https://example.com/t.png",
            post_title="ドライブのイラスト",
            categories=("ポーズ", "ドライブ"),
        )

        score_obj_noun = score_candidate_image(img_obj, "車", pos_type="noun")
        score_action_noun = score_candidate_image(img_action, "車", pos_type="noun")
        self.assertGreater(score_obj_noun, score_action_noun)

        best = select_best_image([img_action, img_obj], "車", pos_type="noun")
        self.assertEqual(best, img_obj)

        # Latin query case-insensitive category boost eligibility
        img_cat = IrasutoyaImage(
            title="CATのイラスト",
            image_url="https://example.com/cat.png",
            thumb_url="https://example.com/t.png",
            post_title="CATのイラスト",
            categories=("動物", "ペット"),
        )
        score_cat = score_candidate_image(img_cat, "cat", pos_type="noun")
        score_cat_nocat = score_candidate_image(
            IrasutoyaImage("CATのイラスト", "https://example.com/cat.png", "https://example.com/t.png", "CATのイラスト"),
            "cat",
            pos_type="noun",
        )
        self.assertGreater(score_cat, score_cat_nocat)

        # Near-miss Latin substring (e.g. "cat" inside "Catching up") without valid boundary must NOT receive category boost
        img_catching = IrasutoyaImage(
            title="Catching upのイラスト",
            image_url="https://example.com/c.png",
            thumb_url="https://example.com/t.png",
            post_title="Catching upのイラスト",
            categories=("動物", "ペット"),
        )
        score_catching_with_cats = score_candidate_image(img_catching, "cat", pos_type="noun")
        score_catching_no_cats = score_candidate_image(
            IrasutoyaImage("Catching upのイラスト", "https://example.com/c.png", "https://example.com/t.png", "Catching upのイラスト"),
            "cat",
            pos_type="noun",
        )
        self.assertEqual(score_catching_with_cats, score_catching_no_cats)

        # Case-insensitive category match for Latin category (e.g. "POP")
        score_pop_upper = calculate_category_score(
            ("POP",), "POP", title="POPのイラスト", allow_boost=True
        )
        score_pop_lower = calculate_category_score(
            ("POP",), "pop", title="POPのイラスト", allow_boost=True
        )
        self.assertEqual(score_pop_lower, score_pop_upper)
        self.assertGreater(score_pop_lower, 0.0)

    def test_anti_archetype_category_penalties(self):
        """Verify anti-archetype categories (文字, マーク, フレーム) receive negative adjustments."""
        img_frame = IrasutoyaImage(
            title="桜の枠",
            image_url="https://example.com/frame.png",
            thumb_url="https://example.com/t.png",
            post_title="春の素材",
            categories=("フレーム", "春"),
        )
        img_mark = IrasutoyaImage(
            title="桜のマーク",
            image_url="https://example.com/mark.png",
            thumb_url="https://example.com/t.png",
            post_title="春の素材",
            categories=("マーク", "春"),
        )
        img_flower = IrasutoyaImage(
            title="桜の花のイラスト",
            image_url="https://example.com/flower.png",
            thumb_url="https://example.com/t.png",
            post_title="桜のイラスト",
            categories=("花", "植物", "春"),
        )

        score_frame = score_candidate_image(img_frame, "桜", pos_type="noun")
        score_mark = score_candidate_image(img_mark, "桜", pos_type="noun")
        score_flower = score_candidate_image(img_flower, "桜", pos_type="noun")

        # Directly assert that anti-archetype categories return negative adjustments
        self.assertLess(calculate_category_score(("フレーム",), "桜", pos_type="noun"), 0.0)
        self.assertLess(calculate_category_score(("マーク",), "桜", pos_type="noun"), 0.0)

        self.assertGreater(score_flower, score_frame)
        self.assertGreater(score_flower, score_mark)

        best = select_best_image([img_frame, img_mark, img_flower], "桜", pos_type="noun")
        self.assertEqual(best, img_flower)

    def test_avoid_duplicate_title_and_category_penalties(self):
        """Verify that when a title already receives a keyword noise penalty, category scoring skips duplicate penalty."""
        img_title_accident = IrasutoyaImage(
            title="車の事故のイラスト",
            image_url="https://example.com/car.png",
            thumb_url="https://example.com/t.png",
            post_title="車の事故のイラスト",
            categories=("事故",),
        )
        img_title_accident_no_cat = IrasutoyaImage(
            title="車の事故のイラスト",
            image_url="https://example.com/car.png",
            thumb_url="https://example.com/t.png",
            post_title="車の事故のイラスト",
            categories=(),
        )
        # Score should not subtract 25 twice for 事故
        score_with_cat = score_candidate_image(img_title_accident, "車")
        score_without_cat = score_candidate_image(img_title_accident_no_cat, "車")
        self.assertEqual(score_with_cat, score_without_cat)

        # But when hazard keyword is NOT in title, category penalty applies
        img_plain_with_hazard_cat = IrasutoyaImage(
            title="車のイラスト",
            image_url="https://example.com/car.png",
            thumb_url="https://example.com/t.png",
            post_title="車のイラスト",
            categories=("事故",),
        )
        img_plain_clean = IrasutoyaImage(
            title="車のイラスト",
            image_url="https://example.com/car.png",
            thumb_url="https://example.com/t.png",
            post_title="車のイラスト",
            categories=(),
        )
        score_hazard_cat = score_candidate_image(img_plain_with_hazard_cat, "車")
        score_clean = score_candidate_image(img_plain_clean, "車")
        self.assertEqual(score_hazard_cat - score_clean, -25.0)

        # Hazard presence must suppress positive category boosts even when category penalty was deduplicated
        img_accident_with_boost_cats = IrasutoyaImage(
            title="車の事故のイラスト",
            image_url="https://example.com/car.png",
            thumb_url="https://example.com/t.png",
            post_title="車の事故のイラスト",
            categories=("事故", "車", "乗り物"),
        )
        score_accident_boosted = score_candidate_image(img_accident_with_boost_cats, "車", pos_type="noun")
        self.assertEqual(score_accident_boosted, score_with_cat)

        # Line dedup: title containing 'ライン' without 'ライン素材' still gets category penalty
        img_line_no_sozai = IrasutoyaImage(
            title="波線のラインのイラスト",
            image_url="https://example.com/line.png",
            thumb_url="https://example.com/t.png",
            post_title="波線のラインのイラスト",
            categories=("ライン",),
        )
        img_line_clean = IrasutoyaImage(
            title="波線のラインのイラスト",
            image_url="https://example.com/line.png",
            thumb_url="https://example.com/t.png",
            post_title="波線のラインのイラスト",
            categories=(),
        )
        self.assertEqual(
            score_candidate_image(img_line_no_sozai, "波線") - score_candidate_image(img_line_clean, "波線"),
            -50.0,
        )

    def test_graceful_degradation_without_categories(self):
        """Verify images with empty categories produce identical scores regardless of pos_type."""
        img_empty = IrasutoyaImage(
            title="猫のイラスト",
            image_url="https://example.com/cat.png",
            thumb_url="https://example.com/t.png",
            post_title="猫のイラスト",
            categories=(),
        )

        score_none = score_candidate_image(img_empty, "猫", pos_type=None)
        score_noun = score_candidate_image(img_empty, "猫", pos_type="noun")
        score_verb = score_candidate_image(img_empty, "猫", pos_type="verb")

        self.assertEqual(score_none, score_noun)
        self.assertEqual(score_none, score_verb)
        self.assertEqual(calculate_category_score((), "猫"), 0.0)

        # Hermetic regression fixture: candidates with empty categories retain identical Phase 1 ranking order
        candidates = [
            IrasutoyaImage("猫のイラスト", "https://example.com/1.png", "https://example.com/t1.png", "猫のイラスト", categories=()),
            IrasutoyaImage("黒猫のイラスト", "https://example.com/2.png", "https://example.com/t2.png", "黒猫のイラスト", categories=()),
            IrasutoyaImage("かわいい猫のイラスト", "https://example.com/3.png", "https://example.com/t3.png", "かわいい猫のイラスト", categories=()),
            IrasutoyaImage("「猫」のイラスト文字", "https://example.com/4.png", "https://example.com/t4.png", "「猫」のイラスト文字", categories=()),
        ]
        best = select_best_image(candidates, "猫", pos_type="noun")
        self.assertEqual(best.title, "猫のイラスト")

        ranked = rank_relevant_images(candidates, "猫", pos_type="noun")
        ranked_titles = [img.title for img in ranked]
        self.assertEqual(
            ranked_titles,
            [
                "猫のイラスト",
                "かわいい猫のイラスト",
                "黒猫のイラスト",
                "「猫」のイラスト文字",
            ],
        )

    def test_situational_hazard_penalties_and_exemption(self):
        """Verify hazard categories are penalized for normal queries but exempted when query is hazard."""
        img_accident = IrasutoyaImage(
            title="交通事故のイラスト",
            image_url="https://example.com/accident.png",
            thumb_url="https://example.com/t.png",
            post_title="事故のイラスト",
            categories=("事故", "車", "乗り物"),
        )

        # Searching for normal word "車" applies penalty for "事故" and suppresses noun boosts
        score_car = calculate_category_score(img_accident.categories, "車", pos_type="noun")
        self.assertEqual(score_car, -25.0)

        # Searching for "事故" directly exempts the "事故" penalty
        score_accident = calculate_category_score(img_accident.categories, "事故", pos_type="noun")
        self.assertGreaterEqual(score_accident, 0.0)

    def test_infer_note_pos_type(self):
        """Verify infer_note_pos_type extracts POS only when explicit Yomitan dictionary metadata exists."""
        class MockMeta:
            def __init__(self, verb_class=None, pos_tags=None):
                self.verb_class = verb_class
                self.pos_tags = pos_tags or set()

        self.assertIsNone(infer_note_pos_type(None))
        self.assertIsNone(infer_note_pos_type(MockMeta()))
        self.assertIsNone(infer_note_pos_type(MockMeta(pos_tags={"exp"})))
        self.assertEqual(infer_note_pos_type(MockMeta(verb_class="v1")), "verb")
        self.assertEqual(infer_note_pos_type(MockMeta(pos_tags={"v5k"})), "verb")
        self.assertEqual(infer_note_pos_type(MockMeta(pos_tags={"vs"})), "verb")
        self.assertEqual(infer_note_pos_type(MockMeta(pos_tags={"vk"})), "verb")
        self.assertEqual(infer_note_pos_type(MockMeta(pos_tags={"adj-i"})), "adjective")
        self.assertEqual(infer_note_pos_type(MockMeta(pos_tags={"n"})), "noun")
        self.assertEqual(infer_note_pos_type(MockMeta(pos_tags={"n-adv"})), "noun")
        # Noun-like or pre-noun adj tags (adj-no, adj-f, adj-pn) must not override noun classification
        self.assertEqual(infer_note_pos_type(MockMeta(pos_tags={"n", "adj-no"})), "noun")
        self.assertEqual(infer_note_pos_type(MockMeta(pos_tags={"n", "adj-f"})), "noun")
        self.assertEqual(infer_note_pos_type(MockMeta(pos_tags={"adj-na"})), "adjective")
        self.assertIsNone(infer_note_pos_type(MockMeta(pos_tags={"adj-no"})))
        self.assertIsNone(infer_note_pos_type(MockMeta(pos_tags={"adj-f"})))
        self.assertIsNone(infer_note_pos_type(MockMeta(pos_tags={"adj-pn"})))
        # Ensure unrelated tags starting with 'n' or 'v' (num, vulg) are excluded
        self.assertIsNone(infer_note_pos_type(MockMeta(pos_tags={"num"})))
        self.assertIsNone(infer_note_pos_type(MockMeta(pos_tags={"vulg"})))

    def test_infer_pos_type(self):
        """Verify infer_pos_type accurately detects verbs, nouns, and adjectives."""
        # 1. Yomitan metadata detection
        class MockMeta:
            def __init__(self, verb_class=None, pos_tags=None):
                self.verb_class = verb_class
                self.pos_tags = pos_tags or set()

        self.assertEqual(infer_pos_type("食べる", yomitan_meta=MockMeta(verb_class="v1")), "verb")
        self.assertEqual(infer_pos_type("走る", yomitan_meta=MockMeta(pos_tags={"v5r"})), "verb")
        self.assertEqual(infer_pos_type("勉強", yomitan_meta=MockMeta(pos_tags={"vs"})), "verb")
        self.assertEqual(infer_pos_type("嬉しい", yomitan_meta=MockMeta(pos_tags={"adj-i"})), "adjective")
        self.assertEqual(infer_pos_type("本", yomitan_meta=MockMeta(pos_tags={"n"})), "noun")

        # 2. Candidate fallback reason detection
        self.assertEqual(infer_pos_type("食べた", cand_reason="verb_past"), "verb")
        self.assertEqual(infer_pos_type("食べ", cand_reason="yomitan_masu_stem"), "verb")

        # 3. Surface heuristic detection
        self.assertEqual(infer_pos_type("走る"), "verb")
        self.assertEqual(infer_pos_type("泳ぐ"), "verb")
        self.assertEqual(infer_pos_type("はしる"), "verb")
        self.assertEqual(infer_pos_type("たべる"), "verb")
        self.assertEqual(infer_pos_type("悲しい"), "adjective")
        self.assertEqual(infer_pos_type("猫"), "noun")
        self.assertEqual(infer_pos_type("車"), "noun")
        self.assertEqual(infer_pos_type("パン"), "noun")
        self.assertEqual(infer_pos_type("お肉"), "noun")
        self.assertEqual(infer_pos_type("食べ物"), "noun")

        # 4. Pure Hiragana noun regression cases (must not be classified as verbs from ending kana)
        self.assertEqual(infer_pos_type("こたつ"), "noun")
        self.assertEqual(infer_pos_type("くつ"), "noun")
        self.assertEqual(infer_pos_type("ねずみ"), "noun")
        self.assertEqual(infer_pos_type("さる"), "noun")
        self.assertEqual(infer_pos_type("たまご"), "noun")

        # Okurigana counters and adverbs (ending in okurigana つ/く)
        self.assertEqual(infer_pos_type("一つ"), "noun")
        self.assertEqual(infer_pos_type("二つ"), "noun")
        self.assertEqual(infer_pos_type("近く"), "noun")
        self.assertEqual(infer_pos_type("遠く"), "noun")
        self.assertEqual(infer_pos_type("多く"), "noun")

        # Non-verb godan lookalikes (adverbs/nouns ending in godan kana) must not be classified as verbs
        self.assertNotEqual(infer_pos_type("全く"), "verb")
        self.assertNotEqual(infer_pos_type("幾つ"), "verb")
        self.assertNotEqual(infer_pos_type("滅多に"), "verb")

        # 5. Unclassified ambiguous kana terms without metadata or verb reason remain neutral
        self.assertIsNone(infer_pos_type("かけぬける"))

    def test_canonical_mark_illustrations_and_abstract_nouns(self):
        """Verify canonical mark illustrations (e.g. 禁止マーク) rank highest and abstract nouns do not get object boosts."""
        from addon.ranking import _is_canonical_mark_title

        self.assertTrue(_is_canonical_mark_title("禁止マークのイラスト", "禁止"))
        self.assertTrue(_is_canonical_mark_title("初心者マーク", "初心者"))
        self.assertTrue(_is_canonical_mark_title("交通安全のマーク", "交通安全"))
        self.assertTrue(_is_canonical_mark_title("「禁止」マーク", "禁止"))
        self.assertFalse(_is_canonical_mark_title("猫のイラスト", "猫"))
        # Substring lookalike: 日本のマーク for 本 must not match as canonical mark of 本
        self.assertFalse(_is_canonical_mark_title("日本のマーク", "本"))
        # Scene depicting a mark attached to an object must not match as a canonical mark title
        self.assertFalse(_is_canonical_mark_title("初心者マークをつけた車のイラスト", "初心者"))
        self.assertFalse(_is_canonical_mark_title("禁止マークを持つ人のイラスト", "禁止"))

        img_mark = IrasutoyaImage(
            title="禁止マークのイラスト",
            image_url="https://example.com/kinshi_mark.png",
            thumb_url="https://example.com/t.png",
            post_title="禁止マークのイラスト",
            categories=("マーク", "禁止"),
        )
        img_kururi = IrasutoyaImage(
            title="くるりポイ禁止のイラスト",
            image_url="https://example.com/kururi.png",
            thumb_url="https://example.com/t.png",
            post_title="くるりポイ禁止のイラスト",
            categories=("ショッピング", "マナー", "禁止", "食材"),
        )
        img_poisute = IrasutoyaImage(
            title="ポイ捨て禁止のイラスト",
            image_url="https://example.com/poisute.png",
            thumb_url="https://example.com/t.png",
            post_title="ポイ捨て禁止のイラスト",
            categories=("マナー", "禁止", "ゴミ"),
        )

        score_mark = score_candidate_image(img_mark, "禁止")
        score_kururi = score_candidate_image(img_kururi, "禁止")
        score_poisute = score_candidate_image(img_poisute, "禁止")

        # 禁止マーク must score far above compound situational prohibitions
        self.assertGreater(score_mark, 100.0)
        self.assertLess(score_kururi, 0.0)
        self.assertLess(score_poisute, 0.0)
        self.assertGreater(score_mark, score_kururi)
        self.assertGreater(score_mark, score_poisute)

        # Abstract nouns like 禁止 must not receive object archetype boosts (e.g. 食材)
        cat_score_kururi = calculate_category_score(img_kururi.categories, "禁止", pos_type="noun")
        self.assertEqual(cat_score_kururi, 25.0)  # Gets +25 for "禁止" in cat_set, but 0 for "食材"

        # Gating allow_boost suppresses positive category boosts (e.g. description-only matches)
        cat_score_desc = calculate_category_score(img_kururi.categories, "禁止", pos_type="noun", allow_boost=False)
        self.assertEqual(cat_score_desc, 0.0)

        # Marks in scenes (e.g. 初心者マークをつけた車) must not receive canonical mark scoring
        img_scene = IrasutoyaImage(
            title="初心者マークをつけた車のイラスト",
            image_url="https://example.com/car.png",
            thumb_url="https://example.com/t.png",
            post_title="初心者マークをつけた車のイラスト",
            categories=("マーク", "乗り物"),
        )
        img_mark_clean = IrasutoyaImage(
            title="初心者マークのイラスト",
            image_url="https://example.com/mark.png",
            thumb_url="https://example.com/t.png",
            post_title="初心者マークのイラスト",
            categories=("マーク",),
        )
        score_scene = score_candidate_image(img_scene, "初心者")
        score_mark_pure = score_candidate_image(img_mark_clean, "初心者")
        self.assertGreater(score_mark_pure, score_scene + 50.0)

    def test_inflected_title_preserves_action_category_boost(self):
        """Verify that inflected title matches (e.g. 勝った人のイラスト for 勝つ) receive action category boost."""
        img_inflected = IrasutoyaImage(
            title="勝った人のイラスト",
            image_url="https://example.com/katta.png",
            thumb_url="https://example.com/t.png",
            post_title="勝った人のイラスト",
            categories=("ポーズ", "スポーツ"),
        )
        img_neutral = IrasutoyaImage(
            title="勝った人のイラスト",
            image_url="https://example.com/katta.png",
            thumb_url="https://example.com/t.png",
            post_title="勝った人のイラスト",
            categories=(),
        )
        score_inflected = score_candidate_image(img_inflected, "勝つ", pos_type="verb")
        score_neutral = score_candidate_image(img_neutral, "勝つ", pos_type="verb")
        # Action category boost (+35.0) should be included because qual_title == "inflected" gates allow_boost=True
        self.assertAlmostEqual(score_inflected - score_neutral, 35.0)

    def test_single_kanji_compound_rejection_and_kata_boundary(self) -> None:
        """Verify 1-Kanji queries reject compound matches while preserving standalone and polite forms."""
        # Rejected compounds for single-Kanji queries
        compound_cases = [
            ("北", "北京ダックのイラスト"),
            ("北", "北海道のキャラクター"),
            ("北", "南北線"),
            ("彼", "彼岸花のイラスト（花）"),
            ("身", "刺身のイラスト"),
            ("口", "人口ピラミッドのイラスト"),
            ("手", "空手のイラスト"),
            ("手", "切手のイラスト"),
            ("手", "大手のイラスト"),
            ("会", "大会のイラスト"),
            ("聞", "新聞のイラスト"),
            ("猫", "猫背のイラスト"),
            ("猫", "猫耳のイラスト"),
            ("本", "三本のバラのイラスト"),
        ]
        for query, title in compound_cases:
            with self.subTest(query=query, title=title):
                img = IrasutoyaImage(title, "https://example.com/i.png", "https://example.com/t.png", title)
                self.assertFalse(
                    is_image_relevant(img, query),
                    f"Compound title '{title}' must not match 1-Kanji query '{query}'",
                )

        # Boundary test: Kanji followed by Katakana (e.g. 北アメリカ) must be classified as compound
        self.assertEqual(classify_match_boundary("北アメリカ", "北"), "compound")
        desc_img = IrasutoyaImage(
            "リョコウバトのイラスト",
            "https://example.com/i.png",
            "https://example.com/t.png",
            "リョコウバトのイラスト",
            description="かつて北アメリカに生息していた動物",
        )
        self.assertFalse(is_image_relevant(desc_img, "北"))

        # Preserved genuine subtype compounds where single-character query is the head noun
        subtype_cases = [
            ("猫", "黒猫のイラスト", ()),
            ("猫", "かわいい黒猫のイラスト", ()),
            ("犬", "柴犬のイラスト", ("犬",)),
            ("犬", "柴犬のイラスト", ()),
            ("犬", "子犬のイラスト", ()),
            ("犬", "走る子犬のイラスト", ()),
            ("犬", "盲導犬のイラスト", ("犬",)),
            ("犬", "犬小屋と柴犬のイラスト", ()),
            ("犬", "犬小屋と柴犬のイラスト", ("犬",)),
        ]
        for query, title, cats in subtype_cases:
            with self.subTest(subtype=title, query=query):
                img = IrasutoyaImage(title, "https://example.com/i.png", "https://example.com/t.png", title, categories=cats)
                self.assertTrue(
                    is_image_relevant(img, query),
                    f"Subtype compound '{title}' must remain relevant for query '{query}'",
                )

        # Multi-occurrence title where all occurrences are fused non-subtypes
        img_fused_multi = IrasutoyaImage(
            "犬小屋と犬走りのイラスト", "https://example.com/i.png", "https://example.com/t.png", "犬小屋と犬走りのイラスト"
        )
        self.assertFalse(is_image_relevant(img_fused_multi, "犬"))

        # Preserved standalone, polite prefix, and exact quoted matches for 1-Kanji queries
        valid_cases = [
            ("手", "手のイラスト", "exact_quoted"),
            ("口", "口のイラスト", "exact_quoted"),
            ("手", "お手のイラスト", "polite_prefix"),
            ("肌", "つるつるのお肌の女性のイラスト", "polite_prefix"),
            ("北", "「北」のイラスト文字", "exact_quoted"),
            ("本", "赤い本のイラスト", "standalone"),
            ("挨拶", "「こんにちは」の挨拶のイラスト", "standalone"),
        ]
        for query, title, expected_boundary in valid_cases:
            with self.subTest(query=query, title=title):
                img = IrasutoyaImage(title, "https://example.com/i.png", "https://example.com/t.png", title)
                self.assertEqual(classify_match_boundary(title, query), expected_boundary)
                self.assertTrue(is_image_relevant(img, query))

    def test_natural_disaster_category_exemption(self) -> None:
        """Verify natural disaster queries are exempted from the '災害' category penalty."""
        img_typhoon_aerial = IrasutoyaImage(
            title="上から見た台風のイラスト",
            image_url="https://example.com/t1.png",
            thumb_url="https://example.com/th1.png",
            post_title="上から見た台風のイラスト",
            categories=("天気", "災害", "地図"),
        )
        img_typhoon_disaster = IrasutoyaImage(
            title="台風のイラスト（自然災害）",
            image_url="https://example.com/t2.png",
            thumb_url="https://example.com/th2.png",
            post_title="台風のイラスト（自然災害）",
            categories=("災害",),
        )

        score_aerial = score_candidate_image(img_typhoon_aerial, "台風", pos_type="noun")
        score_disaster = score_candidate_image(img_typhoon_disaster, "台風", pos_type="noun")

        # Must not receive the -25.0 hazard penalty for '災害'
        self.assertAlmostEqual(score_aerial, 25.4, delta=0.1)
        self.assertAlmostEqual(score_disaster, 84.6, delta=0.1)

        # Other disaster terms in NATURAL_DISASTER_QUERIES also exempt '災害' penalty
        for term in ("地震", "津波", "噴火", "竜巻", "土砂崩れ", "雪崩", "豪雨"):
            with self.subTest(term=term):
                cat_score = calculate_category_score(("災害",), term, pos_type="noun")
                self.assertEqual(cat_score, 0.0)

        # Non-disaster queries still receive the '災害' penalty
        car_disaster_score = calculate_category_score(("災害", "車"), "車", pos_type="noun")
        self.assertEqual(car_disaster_score, -25.0)

    def test_hiragana_particle_prefix_and_kanji_suffix_guard(self) -> None:
        """Verify Hiragana boundary guards reject false particle matches and kanji collisions."""
        # 1. Reading 'くちょう' for '口調' must not match 'はくちょう座'
        title_constellation = "はくちょう座のイラスト（星座）"
        self.assertEqual(classify_match_boundary(title_constellation, "くちょう"), "none")
        self.assertEqual(classify_match_boundary("、はくちょう座", "くちょう"), "none")
        img_constellation = IrasutoyaImage(
            title=title_constellation,
            image_url="https://example.com/c.png",
            thumb_url="https://example.com/t.png",
            post_title=title_constellation,
        )
        self.assertFalse(is_image_relevant(img_constellation, "くちょう"))

        # 2. Adjective and verb attributive forms modifying a Kanji noun must remain standalone
        self.assertEqual(classify_match_boundary("かわいい猫のイラスト", "かわいい"), "standalone")
        self.assertEqual(classify_match_boundary("おいしい水のイラスト", "おいしい"), "standalone")
        self.assertEqual(classify_match_boundary("かけぬける人のイラスト", "かけぬける"), "standalone")
        self.assertEqual(classify_match_boundary("走る人のイラスト", "走る"), "standalone")

        # 3. Known Okurigana nouns ending in kana (like 'こたつ') modifying Kanji are compounds
        self.assertEqual(classify_match_boundary("こたつ布団のイラスト", "こたつ"), "compound")

    def test_archetype_collection_boost(self) -> None:
        """Verify archetype collection boost prioritizes 'いろいろな{term}' over narrow subtypes."""
        img_collection = IrasutoyaImage(
            title="いろいろなネジのイラスト",
            image_url="https://example.com/col.png",
            thumb_url="https://example.com/t.png",
            post_title="いろいろなネジのイラスト",
            categories=("道具",),
        )
        img_resin = IrasutoyaImage(
            title="樹脂のネジのイラスト",
            image_url="https://example.com/res.png",
            thumb_url="https://example.com/t.png",
            post_title="樹脂のネジのイラスト",
            categories=("道具",),
        )

        score_col = score_candidate_image(img_collection, "ネジ", pos_type="noun")
        score_res = score_candidate_image(img_resin, "ネジ", pos_type="noun")

        # Collection illustration (+15.0 boost) beats the narrower resin subtype
        self.assertAlmostEqual(score_col, 60.4, delta=0.1)
        self.assertAlmostEqual(score_res, 47.0, delta=0.1)
        self.assertGreater(score_col, score_res)

        # Single canonical title still beats collection title
        img_exact = IrasutoyaImage(
            title="ネジのイラスト",
            image_url="https://example.com/ex.png",
            thumb_url="https://example.com/t.png",
            post_title="ネジのイラスト",
            categories=("道具",),
        )
        score_exact = score_candidate_image(img_exact, "ネジ", pos_type="noun")
        self.assertGreater(score_exact, score_col)

        # Other collection prefixes also receive the boost
        for prefix in ("色々な", "様々な", "さまざまな"):
            with self.subTest(prefix=prefix):
                img_alt = IrasutoyaImage(
                    title=f"{prefix}ネジのイラスト",
                    image_url="https://example.com/alt.png",
                    thumb_url="https://example.com/t.png",
                    post_title=f"{prefix}ネジのイラスト",
                    categories=("道具",),
                )
                score_alt = score_candidate_image(img_alt, "ネジ", pos_type="noun")
                self.assertGreater(score_alt, score_res)

    def test_attributive_byproduct_modifier_penalty(self) -> None:
        """Verify byproduct modifier titles receive penalty and do not claim headword bonus."""
        img_bundle = IrasutoyaImage(
            title="髪の毛の束のイラスト",
            image_url="https://example.com/bundle.png",
            thumb_url="https://example.com/t.png",
            post_title="髪の毛の束のイラスト",
            categories=("髪", "リクエスト"),
        )
        img_brush = IrasutoyaImage(
            title="髪の毛のついたヘアブラシのイラスト（黒い毛）",
            image_url="https://example.com/brush.png",
            thumb_url="https://example.com/t.png",
            post_title="髪の毛のついたヘアブラシのイラスト",
            categories=("ペット", "髪", "道具"),
        )
        img_rice = IrasutoyaImage(
            title="髪の毛の入ったご飯のイラスト",
            image_url="https://example.com/rice.png",
            thumb_url="https://example.com/t.png",
            post_title="髪の毛の入ったご飯のイラスト",
            categories=("食事", "髪", "リクエスト"),
        )

        score_bundle = score_candidate_image(img_bundle, "髪の毛", pos_type="noun")
        score_brush = score_candidate_image(img_brush, "髪の毛", pos_type="noun")
        score_rice = score_candidate_image(img_rice, "髪の毛", pos_type="noun")

        self.assertAlmostEqual(score_bundle, 87.0, delta=0.1)
        self.assertAlmostEqual(score_brush, 17.4, delta=0.1)
        self.assertAlmostEqual(score_rice, 3.8, delta=0.1)
        self.assertGreater(score_bundle, score_brush)
        self.assertGreater(score_bundle, score_rice)

        # Case-insensitive byproduct modifier detection
        self.assertTrue(_is_byproduct_modifier_title("Cupidのついた箱のイラスト", "cupid"))
        self.assertTrue(_is_byproduct_modifier_title("cupidのついた箱のイラスト", "Cupid"))
        img_cupid_box = IrasutoyaImage(
            title="Cupidのついた箱のイラスト",
            image_url="https://example.com/box.png",
            thumb_url="https://example.com/t.png",
            post_title="Cupidのついた箱のイラスト",
        )
        score_cupid_box = score_candidate_image(img_cupid_box, "cupid", pos_type="noun")
        self.assertLess(score_cupid_box, 30.0)

    def test_compound_prefix_synonym_headword_tolerance(self) -> None:
        """Verify compound headwords with standalone synonyms receive positive adjustment."""
        img_masquerade = IrasutoyaImage(
            title="仮面舞踏会のマスクのイラスト",
            image_url="https://example.com/mask.png",
            thumb_url="https://example.com/t.png",
            post_title="仮面舞踏会のマスクのイラスト",
            categories=("ファッション", "パーティ"),
        )
        img_noh = IrasutoyaImage(
            title="能面のイラスト",
            image_url="https://example.com/noh.png",
            thumb_url="https://example.com/t.png",
            post_title="能面のイラスト",
            description="能楽で使われる仮面、能面（女面）のイラストです。",
            categories=("おもちゃ", "日本", "道具", "美術", "リクエスト"),
        )
        img_ronin = IrasutoyaImage(
            title="仮面浪人のイラスト",
            image_url="https://example.com/ronin.png",
            thumb_url="https://example.com/t.png",
            post_title="仮面浪人のイラスト",
            categories=("リクエスト", "受験"),
        )

        score_masquerade = score_candidate_image(img_masquerade, "仮面", pos_type="noun")
        score_noh = score_candidate_image(img_noh, "仮面", pos_type="noun")
        score_ronin = score_candidate_image(img_ronin, "仮面", pos_type="noun")

        self.assertAlmostEqual(score_masquerade, 23.8, delta=0.1)
        self.assertAlmostEqual(score_noh, 9.4, delta=0.1)
        self.assertAlmostEqual(score_ronin, -42.2, delta=0.1)

        self.assertGreater(score_masquerade, score_noh)
        self.assertGreater(score_noh, score_ronin)
        best = select_best_image([img_ronin, img_noh, img_masquerade], "仮面", pos_type="noun")
        self.assertEqual(best, img_masquerade)

    def test_is_title_match(self) -> None:
        """Verify is_title_match accurately identifies matches in title or post_title."""
        img_title = IrasutoyaImage("お葬式のイラスト", "https://example.com/1.png", "https://example.com/t.png", "お葬式")
        self.assertTrue(is_title_match(img_title, "お葬式"))
        self.assertTrue(is_title_match(img_title, "葬式"))

        img_post = IrasutoyaImage("イラスト", "https://example.com/2.png", "https://example.com/t.png", "警察官のイラスト")
        self.assertTrue(is_title_match(img_post, "警察官"))

        # Description-only match has False for is_title_match
        img_desc = IrasutoyaImage(
            title="セロリのイラスト",
            image_url="https://example.com/3.png",
            thumb_url="https://example.com/t.png",
            post_title="セロリのイラスト",
            description="料理に使われるセロリのイラストです。",
            is_single_image_post=True,
        )
        self.assertFalse(is_title_match(img_desc, "料理"))

        # Empty/whitespace inputs
        self.assertFalse(is_title_match(img_title, ""))
        self.assertFalse(is_title_match(img_title, "   "))

    def test_has_title_illustration_match(self) -> None:
        """Verify has_title_illustration_match filters out decorative, callout, and description-only images."""
        callout = IrasutoyaImage("イラスト文字「今」", "https://example.com/1.png", "https://example.com/t.png", "今")
        decorative = IrasutoyaImage("ライン素材", "https://example.com/2.png", "https://example.com/t.png", "ライン素材")
        desc_only = IrasutoyaImage(
            title="ベニクラゲのイラスト",
            image_url="https://example.com/3.png",
            thumb_url="https://example.com/t.png",
            post_title="ベニクラゲのイラスト",
            description="死ぬほどではありません。",
            is_single_image_post=True,
        )
        real_scene = IrasutoyaImage("お墓のイラスト", "https://example.com/4.png", "https://example.com/t.png", "お墓のイラスト")

        self.assertFalse(has_title_illustration_match([callout], "今"))
        self.assertFalse(has_title_illustration_match([decorative], "今"))
        self.assertFalse(has_title_illustration_match([desc_only], "死ぬ"))
        self.assertTrue(has_title_illustration_match([callout, desc_only, real_scene], "墓"))
        self.assertFalse(has_title_illustration_match([], "テスト"))

    def test_pronouns_description_blocklist(self) -> None:
        """Verify personal pronouns in PRONOUNS_DESCRIPTION_BLOCKLIST do not match via post descriptions."""
        self.assertIn("私", PRONOUNS_DESCRIPTION_BLOCKLIST)
        self.assertIn("僕", PRONOUNS_DESCRIPTION_BLOCKLIST)
        self.assertIn("彼", PRONOUNS_DESCRIPTION_BLOCKLIST)

        img_celery = IrasutoyaImage(
            title="セロリのイラスト",
            image_url="https://example.com/celery.png",
            thumb_url="https://example.com/t.png",
            post_title="セロリのイラスト",
            description="僕はセロリが大好きです。",
            is_single_image_post=True,
        )
        # "僕" is blocked from description-only matches
        self.assertFalse(is_image_relevant(img_celery, "僕"))

        # Non-pronoun word matching only the description is allowed
        self.assertTrue(is_image_relevant(img_celery, "大好き"))
        img_salary = IrasutoyaImage(
            title="女性のイラスト",
            image_url="https://example.com/woman.png",
            thumb_url="https://example.com/t.png",
            post_title="女性のイラスト",
            description="私の年収は普通です。",
            is_single_image_post=True,
        )
        self.assertFalse(is_image_relevant(img_salary, "私"))


if __name__ == "__main__":
    unittest.main()
