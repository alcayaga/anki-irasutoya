"""Unit tests for the Irasutoya HTTP client and fallback search pipeline."""
import json
import unittest
import urllib.error

from addon.client import InsecureRedirectError, IrasutoyaClient, _HttpsOnlyRedirectHandler
from addon.feed import IrasutoyaImage


class TestIrasutoyaClient(unittest.TestCase):
    """Test suite for IrasutoyaClient searching, HTTP handling, and fallback pipeline."""

    def setUp(self):
        """Initialize client instance for tests."""
        self.client = IrasutoyaClient()

    def test_empty_query(self):
        """Test that empty or whitespace-only search queries return an empty list."""
        self.assertEqual(self.client.search(""), [])
        self.assertEqual(self.client.search("   "), [])

    def test_search_mocked(self):
        """Test searching with stubbed HTTP response to ensure deterministic execution."""
        sample_feed = {
            "feed": {
                "entry": [
                    {
                        "title": {"$t": "りんごのイラスト"},
                        "summary": {"$t": "美味しいりんごのイラスト"},
                        "content": {
                            "$t": (
                                '<a href="https://1.bp.blogspot.com/s400/apple.png">'
                                '<img alt="赤色のりんご" src="https://1.bp.blogspot.com/s200/apple.png" />'
                                '</a>'
                            )
                        },
                    }
                ]
            }
        }
        self.client._make_request = lambda url, timeout=10, max_retries=2: json.dumps(sample_feed).encode("utf-8")

        results = self.client.search("りんご", max_results=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "赤色のりんご")
        self.assertEqual(results[0].image_url, "https://1.bp.blogspot.com/s800/apple.png")
        self.assertEqual(results[0].thumb_url, "https://1.bp.blogspot.com/s120-c/apple.png")

    def test_download_image_scheme_validation(self):
        """Verify non-HTTPS or non-allowlisted image download URLs are rejected with ValueError."""
        with self.assertRaises(ValueError):
            self.client.download_image("file:///etc/passwd")
        with self.assertRaises(ValueError):
            self.client.download_image("http://example.com/insecure.png")
        with self.assertRaises(ValueError):
            self.client.download_image("https://evil-untrusted-domain.com/cat.png")

    def test_https_only_redirect_handler(self):
        """Verify that HTTPRedirectHandler rejects non-HTTPS or untrusted redirect targets."""
        handler = _HttpsOnlyRedirectHandler()
        with self.assertRaises(InsecureRedirectError):
            handler.redirect_request(None, None, 302, "Found", {}, "http://insecure.target/img.png")
        with self.assertRaises(InsecureRedirectError):
            handler.redirect_request(None, None, 302, "Found", {}, "file:///etc/passwd")
        with self.assertRaises(InsecureRedirectError):
            handler.redirect_request(None, None, 302, "Found", {}, "https://evil-untrusted-site.com/image.png")

    def test_search_with_fallbacks(self):
        """Test candidate fallback execution when primary returns 0 results."""
        client = IrasutoyaClient()

        def mock_search(q, max_results=3, image_size="s800", orderby="relevance"):
            if q == "そのまま":
                return [
                    IrasutoyaImage("そのままのイラスト", "https://example.com/img.png", "https://example.com/t.png", "そのまま")
                ]
            return []

        client.search = mock_search

        candidates = [
            ("其の儘", "primary", 1),
            ("そのまま", "note_reading_Reading", 2),
        ]
        images, matched_cand, best_img = client.search_with_fallbacks(candidates)
        self.assertEqual(len(images), 1)
        self.assertIsNotNone(matched_cand)
        self.assertEqual(matched_cand[0], "そのまま")
        self.assertEqual(matched_cand[2], 2)
        self.assertIsNotNone(best_img)
        self.assertEqual(best_img.title, "そのままのイラスト")

    def test_search_with_fallbacks_propagates_search_exception(self):
        """Verify search_with_fallbacks propagates network/HTTP errors rather than falling back to Tier 2."""
        client = IrasutoyaClient()

        def mock_failing_search(q, **kwargs):
            if q == "其の儘":
                raise urllib.error.HTTPError("https://example.com", 429, "Too Many Requests", {}, None)
            return [
                IrasutoyaImage("そのままのイラスト", "https://example.com/img.png", "https://example.com/t.png", "そのまま")
            ]

        client.search = mock_failing_search
        candidates = [
            ("其の儘", "primary", 1),
            ("そのまま", "note_reading_Reading", 2),
        ]

        with self.assertRaises(urllib.error.HTTPError):
            client.search_with_fallbacks(candidates)

    def test_search_with_fallbacks_skips_decorative_only_results(self):
        """Verify search_with_fallbacks continues to fallback candidate if initial candidate only yields decorative assets."""
        client = IrasutoyaClient()

        def mock_search(q, max_results=3, image_size="s800", orderby="relevance"):
            if q == "追いかける":
                # Only decorative frame returned
                return [
                    IrasutoyaImage("フレームのイラスト", "https://example.com/frame.png", "https://example.com/t.png", "フレーム")
                ]
            if q == "走る":
                # Actual illustration returned
                return [
                    IrasutoyaImage("走る人のイラスト", "https://example.com/run.png", "https://example.com/t.png", "走る")
                ]
            return []

        client.search = mock_search
        candidates = [
            ("追いかける", "primary", 1),
            ("走る", "note_reading", 2),
        ]
        images, matched_cand, best_img = client.search_with_fallbacks(candidates)
        self.assertIsNotNone(matched_cand)
        self.assertEqual(matched_cand[0], "走る")

    def test_search_with_fallbacks_skips_unrelated_images(self):
        """Verify search_with_fallbacks skips candidates that only return unrelated images from multi-image posts."""
        client = IrasutoyaClient()

        def mock_search(q, max_results=3, image_size="s800", orderby="relevance"):
            if q == "猫":
                # Returns only a dog image from a post that mentioned cat
                return [
                    IrasutoyaImage("犬のイラスト", "https://example.com/dog.png", "https://example.com/t.png", "猫のイラスト")
                ]
            if q == "ネコ":
                # Returns an actual cat illustration
                return [
                    IrasutoyaImage("ネコのイラスト", "https://example.com/cat.png", "https://example.com/t.png", "ネコのイラスト")
                ]
            return []

        client.search = mock_search
        candidates = [
            ("猫", "primary", 1),
            ("ネコ", "to_katakana", 1),
        ]
        images, matched_cand, best_img = client.search_with_fallbacks(candidates)
        self.assertIsNotNone(matched_cand)
        self.assertEqual(matched_cand[0], "ネコ")
        self.assertIsNotNone(best_img)
        self.assertEqual(best_img.title, "ネコのイラスト")

    def test_search_with_fallbacks_prefers_scene_over_callout(self):
        """Verify search_with_fallbacks selects non-callout scene illustration before text callout."""
        client = IrasutoyaClient()

        def mock_search(q, max_results=3, image_size="s800", orderby="relevance"):
            if q == "肉":
                # Returns both a text callout (with high quoted/post title bonus) and a relevant scene
                callout = IrasutoyaImage("「肉」のイラスト文字", "https://example.com/c.png", "https://example.com/t.png", "肉のイラスト")
                scene = IrasutoyaImage("培養肉のイラスト", "https://example.com/s.png", "https://example.com/t.png", "培養肉のイラスト")
                return [callout, scene]
            return []

        client.search = mock_search
        candidates = [("肉", "primary", 1)]
        images, matched_cand, best_img = client.search_with_fallbacks(candidates)
        self.assertIsNotNone(best_img)
        self.assertEqual(best_img.title, "培養肉のイラスト")
        self.assertEqual(images[0].title, "培養肉のイラスト")
        self.assertEqual(images[1].title, "「肉」のイラスト文字")


if __name__ == "__main__":
    unittest.main()
