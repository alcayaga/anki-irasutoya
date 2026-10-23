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

    def test_client_uses_catalog_when_available(self):
        """Verify client uses offline CatalogIndex when available."""
        from unittest.mock import MagicMock
        mock_store = MagicMock()
        mock_index = MagicMock()
        mock_store.get_index.return_value = mock_index
        mock_index.search.return_value = [
            IrasutoyaImage("猫のイラスト", "https://example.com/cat.png", "https://example.com/thumb.png", "猫")
        ]

        client = IrasutoyaClient(catalog=mock_store, backend="auto")
        self.assertTrue(client.uses_catalog())

        results = client.search("猫")
        self.assertEqual(len(results), 1)
        mock_index.search.assert_called_once_with("猫", image_size="s800")

    def test_client_falls_back_to_live_when_configured_or_missing(self):
        """Verify client respects backend flag and falls back to live Blogger search if catalog missing."""
        from unittest.mock import MagicMock
        mock_store = MagicMock()
        mock_store.get_index.return_value = None

        client_missing = IrasutoyaClient(catalog=mock_store, backend="auto")
        self.assertFalse(client_missing.uses_catalog())

        client_live_forced = IrasutoyaClient(catalog=mock_store, backend="live")
        self.assertFalse(client_live_forced.uses_catalog())

        client_none = IrasutoyaClient(catalog=None, backend="auto")
        self.assertFalse(client_none.uses_catalog())

    def test_client_catalog_mode_never_falls_back_to_live(self):
        """Verify backend='catalog' returns empty list and never falls through to live HTTP search."""
        from unittest.mock import MagicMock
        mock_store = MagicMock()
        mock_store.get_index.return_value = None

        client = IrasutoyaClient(catalog=mock_store, backend="catalog")
        client._make_request = MagicMock()

        results = client.search("猫")
        self.assertEqual(results, [])
        client._make_request.assert_not_called()

    def test_search_with_fallbacks_skips_pacing_in_catalog_mode(self):
        """Verify pace_callback is bypassed in offline catalog mode and invoked in live search."""
        from unittest.mock import MagicMock
        mock_store = MagicMock()
        mock_index = MagicMock()
        mock_store.get_index.return_value = mock_index
        mock_index.search.return_value = [
            IrasutoyaImage("猫のイラスト", "https://example.com/cat.png", "https://example.com/thumb.png", "猫")
        ]

        client_cat = IrasutoyaClient(catalog=mock_store, backend="auto")
        pace_called_cat = []
        client_cat.search_with_fallbacks([("猫", "primary", 1)], pace_callback=lambda: pace_called_cat.append(True))
        self.assertEqual(len(pace_called_cat), 0)

        # In live mode, pace_callback is invoked
        client_live = IrasutoyaClient(catalog=None, backend="live")
        client_live.search = lambda q, **kwargs: [
            IrasutoyaImage("猫のイラスト", "https://example.com/cat.png", "https://example.com/thumb.png", "猫")
        ]
        pace_called_live = []
        client_live.search_with_fallbacks([("猫", "primary", 1)], pace_callback=lambda: pace_called_live.append(True))
        self.assertEqual(len(pace_called_live), 1)

    def test_download_image_self_heals_on_404(self):
        """Verify download_image executes a single self-heal refetch on HTTP 404/410 when post_id is present."""
        client = IrasutoyaClient()

        image = IrasutoyaImage(
            title="更新された猫",
            image_url="https://1.bp.blogspot.com/s800/stale_cat.png",
            thumb_url="https://1.bp.blogspot.com/s120-c/stale_cat.png",
            post_title="猫のイラスト",
            post_id="9999",
        )

        stale_url = image.image_url
        new_url = "https://1.bp.blogspot.com/s800/fresh_cat.png"
        refetched_post_json = {
            "entry": {
                "id": {"$t": "tag:blogger.com,1999:blog-1.post-9999"},
                "title": {"$t": "猫のイラスト"},
                "content": {
                    "$t": f'<a href="{new_url}"><img alt="更新された猫" src="{new_url}" /></a>'
                },
            }
        }

        call_log = []

        def mock_make_request(url, timeout=15):
            call_log.append(url)
            if url == stale_url:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            if "feeds/posts/default/9999" in url:
                return json.dumps(refetched_post_json).encode("utf-8")
            if url == new_url:
                return b"FRESH_CAT_BINARY_DATA"
            raise ValueError(f"Unexpected URL: {url}")

        client._make_request = mock_make_request

        result = client.download_image(stale_url, image=image)
        self.assertEqual(result, b"FRESH_CAT_BINARY_DATA")
        self.assertIn(stale_url, call_log)
        self.assertIn("https://www.irasutoya.com/feeds/posts/default/9999?alt=json", call_log)
        self.assertIn(new_url, call_log)

        # Test thumbnail URL healing preserves /s120-c/ size token
        stale_thumb = image.thumb_url
        new_thumb = "https://1.bp.blogspot.com/s120-c/fresh_cat.png"
        call_log.clear()

        def mock_thumb_request(url, timeout=15):
            call_log.append(url)
            if url == stale_thumb:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            if "feeds/posts/default/9999" in url:
                return json.dumps(refetched_post_json).encode("utf-8")
            if url == new_thumb:
                return b"FRESH_THUMB_BINARY"
            raise ValueError(f"Unexpected URL: {url}")

        client._make_request = mock_thumb_request
        thumb_res = client.download_image(stale_thumb, image=image)
        self.assertEqual(thumb_res, b"FRESH_THUMB_BINARY")
        self.assertIn(new_thumb, call_log)

    def test_self_heal_matches_filename_before_title(self):
        """Verify _self_heal_image_url prioritizes matching image filename over non-unique titles."""
        client = IrasutoyaClient()
        post_json = {
            "entry": {
                "id": {"$t": "tag:blogger.com,1999:blog-1.post-123"},
                "title": {"$t": "猫のイラスト"},
                "content": {
                    "$t": (
                        '<a href="https://1.bp.blogspot.com/s800/img_a.png"><img alt="猫" src="https://1.bp.blogspot.com/s800/img_a.png" /></a>'
                        '<a href="https://1.bp.blogspot.com/s800/img_b.png"><img alt="猫" src="https://1.bp.blogspot.com/s800/img_b.png" /></a>'
                    )
                },
            }
        }
        client._make_request = lambda url, timeout=10: json.dumps(post_json).encode("utf-8")

        # Looking for img_b.png even though both share alt="猫"
        target_img = IrasutoyaImage("猫", "https://old.example.com/s800/img_b.png", "t", "猫", post_id="123")
        healed = client._self_heal_image_url(target_img, target_img.image_url)
        self.assertEqual(healed, "https://1.bp.blogspot.com/s800/img_b.png")

    def test_download_image_reraises_on_non_404_or_failed_heal(self):
        """Verify download_image re-raises errors when status is not 404/410 or healing is impossible."""
        client = IrasutoyaClient()

        # HTTP 500 is never healed
        def mock_server_error(url, timeout=15):
            raise urllib.error.HTTPError(url, 500, "Internal Server Error", {}, None)

        client._make_request = mock_server_error
        img_with_id = IrasutoyaImage("猫", "https://1.bp.blogspot.com/cat.png", "t.png", "猫", post_id="123")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            client.download_image(img_with_id.image_url, image=img_with_id)
        self.assertEqual(cm.exception.code, 500)

        # HTTP 404 with missing post_id is immediately raised
        def mock_not_found(url, timeout=15):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        client._make_request = mock_not_found
        img_no_id = IrasutoyaImage("猫", "https://1.bp.blogspot.com/cat.png", "t.png", "猫", post_id="")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            client.download_image(img_no_id.image_url, image=img_no_id)
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
