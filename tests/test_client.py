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

        # Ensure preceding path segments like /s100/ are preserved
        url_multi = "https://blogger.googleusercontent.com/img/b/s100/s72-c/image.png"
        self.assertEqual(
            self.client._upgrade_image_url(url_multi, "s800"),
            "https://blogger.googleusercontent.com/img/b/s100/s800/image.png"
        )
        self.assertEqual(
            self.client._to_thumbnail_url(url_multi, "s120-c"),
            "https://blogger.googleusercontent.com/img/b/s100/s120-c/image.png"
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
                        '<a href="https://1.bp.blogspot.com/s400/cat_white.png">'
                        '<img alt="白猫のイラスト" src="https://1.bp.blogspot.com/s200/cat_white.png" />'
                        '</a>'
                        '<a href="https://1.bp.blogspot.com/s400/cat_black.png">'
                        '<img alt="黒猫のイラスト" src="https://1.bp.blogspot.com/s200/cat_black.png" />'
                        '</a>'
                        '</div>'
                    )
                }
            }
        ]

        images = self.client.parse_entries(sample_entries, image_size="s800")
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].title, "白猫のイラスト")
        self.assertEqual(images[0].image_url, "https://1.bp.blogspot.com/s800/cat_white.png")
        self.assertEqual(images[0].thumb_url, "https://1.bp.blogspot.com/s120-c/cat_white.png")
        self.assertEqual(images[0].post_title, "猫のイラスト")

        self.assertEqual(images[1].title, "黒猫のイラスト")
        self.assertEqual(images[1].image_url, "https://1.bp.blogspot.com/s800/cat_black.png")

    def test_parse_entries_thumbnail_fallback(self):
        """Test fallback to post thumbnail when entry content lacks explicit <a><img> links."""
        sample_entries = [
            {
                "title": {"$t": "リンゴのイラスト"},
                "summary": {"$t": "リンゴです。"},
                "content": {"$t": "<div>テキストのみ</div>"},
                "media$thumbnail": {
                    "url": "https://1.bp.blogspot.com/s72-c/apple.png"
                }
            }
        ]

        images = self.client.parse_entries(sample_entries, image_size="s800")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].title, "リンゴのイラスト")
        self.assertEqual(images[0].image_url, "https://1.bp.blogspot.com/s800/apple.png")
        self.assertEqual(images[0].thumb_url, "https://1.bp.blogspot.com/s120-c/apple.png")

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
                        }
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
        # Allowed host should not fail host validation check
        self.assertTrue(self.client._is_allowed_image_host("blogger.googleusercontent.com"))
        self.assertTrue(self.client._is_allowed_image_host("1.bp.blogspot.com"))
        self.assertTrue(self.client._is_allowed_image_host("www.irasutoya.com"))
        self.assertFalse(self.client._is_allowed_image_host("evilgoogleusercontent.com"))
        self.assertFalse(self.client._is_allowed_image_host("not-irasutoya.com"))
        self.assertFalse(self.client._is_allowed_image_host("blogger.com"))

    def test_https_only_redirect_handler(self):
        """Verify that HTTPRedirectHandler rejects non-HTTPS or untrusted redirect targets."""
        from client import _HttpsOnlyRedirectHandler, InsecureRedirectError
        handler = _HttpsOnlyRedirectHandler()
        with self.assertRaises(InsecureRedirectError):
            handler.redirect_request(None, None, 302, "Found", {}, "http://insecure.target/img.png")
        with self.assertRaises(InsecureRedirectError):
            handler.redirect_request(None, None, 302, "Found", {}, "file:///etc/passwd")
        with self.assertRaises(InsecureRedirectError):
            handler.redirect_request(None, None, 302, "Found", {}, "https://evil-untrusted-site.com/image.png")

    def test_parse_entries_rejects_non_image_extension_urls(self):
        """Verify that links ending with .png.html or other non-image suffixes are not treated as images."""
        sample_entry = [
            {
                "title": {"$t": "HTML Link Post"},
                "content": {
                    "$t": '<a href="https://1.bp.blogspot.com/photo.png.html"><img src="thumb.jpg"></a>'
                }
            }
        ]
        results = self.client.parse_entries(sample_entry)
        self.assertEqual(len(results), 0)

    def test_parse_entries_mixed_hosts(self):
        """Verify that entries containing untrusted image hosts filter out the untrusted URLs."""
        sample_entry = [
            {
                "title": {"$t": "Mixed Host Post"},
                "content": {
                    "$t": (
                        '<a href="https://evil-site.com/bad.png">'
                        '<img alt="Bad image" src="https://evil-site.com/bad_thumb.png">'
                        '</a>'
                        '<a href="https://1.bp.blogspot.com/good.png">'
                        '<img alt="Good image" src="https://1.bp.blogspot.com/good_thumb.png">'
                        '</a>'
                    )
                }
            }
        ]
        results = self.client.parse_entries(sample_entry)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Good image")
        self.assertEqual(results[0].image_url, "https://1.bp.blogspot.com/good.png")

    def test_parse_entries_rejects_http_links(self):
        """Verify that HTTP (non-HTTPS) image links and thumbnails are rejected during feed parsing."""
        sample_entry = [
            {
                "title": {"$t": "Insecure Link Post"},
                "content": {
                    "$t": (
                        '<a href="http://1.bp.blogspot.com/insecure.png">'
                        '<img alt="Insecure" src="http://1.bp.blogspot.com/insecure_thumb.png">'
                        '</a>'
                        '<a href="https://1.bp.blogspot.com/secure.png">'
                        '<img alt="Secure" src="https://1.bp.blogspot.com/secure_thumb.png">'
                        '</a>'
                    )
                }
            }
        ]
        results = self.client.parse_entries(sample_entry)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Secure")
        self.assertEqual(results[0].image_url, "https://1.bp.blogspot.com/secure.png")

        # Test thumbnail fallback with HTTP
        fallback_entry = [
            {
                "title": {"$t": "Insecure Thumbnail"},
                "content": {"$t": "text only"},
                "media$thumbnail": {"url": "http://1.bp.blogspot.com/thumb.png"}
            }
        ]
        self.assertEqual(len(self.client.parse_entries(fallback_entry)), 0)

    def test_parse_entries_unescapes_html_and_respects_attribute_boundaries(self):
        """Verify that HTML entities are unescaped and attributes like data-src don't match."""
        sample_entry = [
            {
                "title": {"$t": "Fish &amp; Chips"},
                "summary": {"$t": "Delicious &lt;snack&gt; &amp; meal"},
                "content": {
                    "$t": (
                        '<a data-href="https://1.bp.blogspot.com/fake.png" href="https://1.bp.blogspot.com/real.png">'
                        '<img data-src="https://1.bp.blogspot.com/fake_thumb.png" '
                        'src="https://1.bp.blogspot.com/real_thumb.png" '
                        'data-alt="Fake &amp; Old" '
                        'alt="Tom &amp; Jerry&#39;s Boat">'
                        '</a>'
                    )
                }
            }
        ]
        results = self.client.parse_entries(sample_entry)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Tom & Jerry's Boat")
        self.assertEqual(results[0].post_title, "Fish & Chips")
        self.assertEqual(results[0].description, "Delicious <snack> & meal")
        self.assertEqual(results[0].image_url, "https://1.bp.blogspot.com/real.png")
        self.assertEqual(results[0].thumb_url, "https://1.bp.blogspot.com/real_thumb.png")

    def test_parse_entries_thumbnail_untrusted_src_fallback(self):
        """Verify that if img src is untrusted or non-https, thumbnail falls back to high_res."""
        sample_entry = [
            {
                "title": {"$t": "Fallback Test"},
                "content": {
                    "$t": (
                        '<a href="https://1.bp.blogspot.com/s1600/photo.png">'
                        '<img src="http://insecure-domain.com/photo.png">'
                        '</a>'
                    )
                }
            }
        ]
        results = self.client.parse_entries(sample_entry)
        self.assertEqual(len(results), 1)
        # Should fallback to high_res converted to thumbnail s120-c
        self.assertEqual(results[0].thumb_url, "https://1.bp.blogspot.com/s120-c/photo.png")

    def test_parse_entries_handles_whitespace_and_misleading_title_attribute(self):
        """Verify HTMLParser handles attributes with whitespace and ignores fake href in title."""
        sample_entry = [
            {
                "title": {"$t": "Attribute Edge Case"},
                "content": {
                    "$t": (
                        '<a title="Click here href=\'https://evil.com/fake.png\' for info" '
                        '   href  =  "https://1.bp.blogspot.com/real.png?v=1&amp;w=2" >'
                        '  <span><img src  =  "https://1.bp.blogspot.com/real_thumb.png" alt="Card Image"></span>'
                        '</a>'
                    )
                }
            }
        ]
        results = self.client.parse_entries(sample_entry)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Card Image")
        self.assertEqual(results[0].image_url, "https://1.bp.blogspot.com/real.png?v=1&w=2")
        self.assertEqual(results[0].thumb_url, "https://1.bp.blogspot.com/real_thumb.png")

    def test_parse_entries_thumbnail_decodes_html_entities(self):
        """Verify fallback thumbnail URL is properly HTML decoded."""
        fallback_entry = [
            {
                "title": {"$t": "Entity Thumbnail"},
                "content": {"$t": "text only"},
                "media$thumbnail": {"url": "https://1.bp.blogspot.com/thumb.png?a=1&amp;b=2"}
            }
        ]
        results = self.client.parse_entries(fallback_entry)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].thumb_url, "https://1.bp.blogspot.com/thumb.png?a=1&b=2")


if __name__ == "__main__":
    unittest.main()
