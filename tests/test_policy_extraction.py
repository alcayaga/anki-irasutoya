"""Behavioral regression checks for the runtime policy extraction."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

from addon.feed import IrasutoyaImage
from addon.ranking import rank_relevant_images, score_candidate_image


class TestPolicyExtraction(unittest.TestCase):
    """Preserve scores and headless imports while relocating policy data."""

    def test_recorded_scores(self) -> None:
        """Retain pre-refactor scores for canonical, hazard, and decorative cases."""
        cases = [
            ("本のイラスト", "本", ("本",), 185.2),
            ("お肉のイラスト", "肉", ("食べ物",), 159.4),
            ("禁止マークのイラスト", "禁止", ("マーク",), 142.0),
            ("猫のフレーム", "猫", ("フレーム",), 30.200000000000003),
            ("走る人のイラスト", "走る", ("運動", "スポーツ"), 63.6),
            ("髪の毛のついたブラシ", "髪の毛", (), 7.0),
            ("合格のイラスト文字", "合格", ("文字",), 22.799999999999997),
        ]
        for title, query, categories, expected in cases:
            with self.subTest(title=title):
                image = IrasutoyaImage(title, "https://example.com/a.png", "", title, categories=categories)
                self.assertEqual(score_candidate_image(image, query), expected)

    def test_stable_order_for_equal_scores(self) -> None:
        """Keep input ordering when equally relevant illustrations have tied scores."""
        images = [
            IrasutoyaImage("本のイラスト", f"https://example.com/{i}.png", "", "本のイラスト")
            for i in range(3)
        ]
        self.assertEqual(rank_relevant_images(images, "本"), images)
        self.assertEqual(rank_relevant_images(images[::-1], "本"), images[::-1])

    def test_headless_import_orders(self) -> None:
        """Import each entry point first in fresh processes without Anki or Qt."""
        root = Path(__file__).resolve().parents[1]
        for first in ("ranking", "synonyms", "text_utils", "ranking_constants", "synonym_constants"):
            with self.subTest(first=first):
                code = (
                    "import importlib, sys; "
                    "sys.modules['aqt'] = None; sys.modules['PyQt6'] = None; "
                    f"importlib.import_module('addon.{first}'); "
                    "from addon import ranking, synonyms; "
                    "assert synonyms._is_action_target('踊る'); "
                    "assert not synonyms._is_action_target('机'); "
                    "assert ranking.classify_match_boundary('本のイラスト', '本') == 'exact_quoted'"
                )
                result = subprocess.run(
                    [sys.executable, "-c", code], cwd=root, env=dict(os.environ),
                    capture_output=True, text=True, timeout=30, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
