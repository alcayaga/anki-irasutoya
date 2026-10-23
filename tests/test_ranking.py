"""Unit tests for boundary classification, relevance scoring, and archetype ranking."""
import unittest
from unittest.mock import patch

from addon.feed import IrasutoyaImage
from addon.ranking import (
    classify_match_boundary,
    is_callout_asset,
    is_decorative_asset,
    is_image_relevant,
    rank_relevant_images,
    score_candidate_image,
    select_best_image,
)
from addon.text_utils import BATCH_MAX_CANDIDATES, generate_search_candidates


class TestRanking(unittest.TestCase):
    """Test suite for boundary classification, image relevance heuristics, and ranking."""

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

        # Genuine Kanji-noun compounds, okurigana nouns, and adjectives modifying nouns should still be accepted
        hada_img = IrasutoyaImage("乾燥肌のイラスト", "https://example.com/h.png", "https://example.com/t.png", "乾燥肌")
        self.assertTrue(is_image_relevant(hada_img, "肌"))
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
                    )[:BATCH_MAX_CANDIDATES]
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


if __name__ == "__main__":
    unittest.main()
