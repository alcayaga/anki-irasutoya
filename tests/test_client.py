"""Unit tests for the Irasutoya scraper client."""
import json
import unittest
import urllib.error
from client import IrasutoyaClient, IrasutoyaImage


class TestIrasutoyaClient(unittest.TestCase):
    """Test suite for IrasutoyaClient URL upgrading, parsing, and searching."""

    def setUp(self):
        """Initialize client instance for tests."""
        self.client = IrasutoyaClient()

    def test_upgrade_image_url(self):
        """Test replacing Blogger image thumbnail codes with high-res dimensions."""
        url1 = "https://blogger.googleusercontent.com/img/b/R29/s72-c/image.png"
        self.assertEqual(
            self.client._upgrade_image_url(url1, "s800"),
            "https://blogger.googleusercontent.com/img/b/R29/s800/image.png"
        )
        self.assertEqual(
            self.client._upgrade_image_url(url1, "s0"),
            "https://blogger.googleusercontent.com/img/b/R29/s0/image.png"
        )

        url2 = "https://blogger.googleusercontent.com/img/b/R29/s320/cat.jpg"
        self.assertEqual(
            self.client._upgrade_image_url(url2, "s800"),
            "https://blogger.googleusercontent.com/img/b/R29/s800/cat.jpg"
        )

    def test_empty_query(self):
        """Test that empty or whitespace-only search queries return an empty list."""
        self.assertEqual(self.client.search(""), [])
        self.assertEqual(self.client.search("   "), [])

    def test_parse_entries_multi_image(self):
        """Test parsing Blogger feed entry with multiple illustration images."""
        sample_entries = [
            {
                "title": {"$t": "猫のイラスト"},
                "summary": {"$t": "可愛い猫のイラストです。"},
                "content": {
                    "$t": (
                        '<div class="separator">'
                        '<a href="https://example.com/s400/cat_white.png">'
                        '<img alt="白猫のイラスト" src="https://example.com/s200/cat_white.png" />'
                        '</a>'
                        '<a href="https://example.com/s400/cat_black.png">'
                        '<img alt="黒猫のイラスト" src="https://example.com/s200/cat_black.png" />'
                        '</a>'
                        '</div>'
                    )
                }
            }
        ]

        images = self.client.parse_entries(sample_entries, image_size="s800")
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].title, "白猫のイラスト")
        self.assertEqual(images[0].image_url, "https://example.com/s800/cat_white.png")
        self.assertEqual(images[0].thumb_url, "https://example.com/s120-c/cat_white.png")
        self.assertEqual(images[0].post_title, "猫のイラスト")

        self.assertEqual(images[1].title, "黒猫のイラスト")
        self.assertEqual(images[1].image_url, "https://example.com/s800/cat_black.png")

    def test_parse_entries_thumbnail_fallback(self):
        """Test fallback to post thumbnail when entry content lacks explicit <a><img> links."""
        sample_entries = [
            {
                "title": {"$t": "リンゴのイラスト"},
                "summary": {"$t": "リンゴです。"},
                "content": {"$t": "<div>テキストのみ</div>"},
                "media$thumbnail": {
                    "url": "https://example.com/s72-c/apple.png"
                }
            }
        ]

        images = self.client.parse_entries(sample_entries, image_size="s800")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].title, "リンゴのイラスト")
        self.assertEqual(images[0].image_url, "https://example.com/s800/apple.png")
        self.assertEqual(images[0].thumb_url, "https://example.com/s120-c/apple.png")

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
                                '<a href="https://example.com/s400/apple.png">'
                                '<img alt="赤色のりんご" src="https://example.com/s200/apple.png" />'
                                '</a>'
                            )
                        }
                    }
                ]
            }
        }
        self.client._make_request = lambda url, timeout=10, max_retries=2: json.dumps(sample_feed).encode("utf-8")

        results = self.client.search("りんご", max_results=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "赤色のりんご")
        self.assertEqual(results[0].image_url, "https://example.com/s800/apple.png")
        self.assertEqual(results[0].thumb_url, "https://example.com/s120-c/apple.png")

    def test_download_image_scheme_validation(self):
        """Verify non-HTTPS image download URLs are rejected with ValueError."""
        with self.assertRaises(ValueError):
            self.client.download_image("file:///etc/passwd")
        with self.assertRaises(ValueError):
            self.client.download_image("http://example.com/insecure.png")

    def test_https_only_redirect_handler(self):
        """Verify that HTTPRedirectHandler rejects non-HTTPS redirect targets."""
        from client import _HttpsOnlyRedirectHandler
        handler = _HttpsOnlyRedirectHandler()
        with self.assertRaises(urllib.error.URLError):
            handler.redirect_request(None, None, 302, "Found", {}, "http://insecure.target/img.png")
        with self.assertRaises(urllib.error.URLError):
            handler.redirect_request(None, None, 302, "Found", {}, "file:///etc/passwd")


if __name__ == "__main__":
    unittest.main()
