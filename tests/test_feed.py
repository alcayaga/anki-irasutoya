"""Unit tests for Irasutoya Blogger feed parsing, HTML extraction, and URL helpers."""
import unittest
from unittest.mock import patch

from addon.feed import (
    is_allowed_image_host,
    parse_entries,
    parse_entry,
    to_thumbnail_url,
    upgrade_image_url,
)
from addon.ranking import is_image_relevant


class TestFeedParser(unittest.TestCase):
    """Test suite for Blogger feed HTML parsing, image URL upgrading, and extraction."""

    def test_upgrade_image_url(self):
        """Test replacing Blogger image thumbnail codes with high-res dimensions."""
        url1 = "https://blogger.googleusercontent.com/img/b/R29/s72-c/image.png"
        self.assertEqual(
            upgrade_image_url(url1, "s800"),
            "https://blogger.googleusercontent.com/img/b/R29/s800/image.png",
        )
        self.assertEqual(
            upgrade_image_url(url1, "s0"),
            "https://blogger.googleusercontent.com/img/b/R29/s0/image.png",
        )

        url2 = "https://blogger.googleusercontent.com/img/b/R29/s320/cat.jpg"
        self.assertEqual(
            upgrade_image_url(url2, "s800"),
            "https://blogger.googleusercontent.com/img/b/R29/s800/cat.jpg",
        )

        # Ensure preceding path segments like /s100/ are preserved
        url_multi = "https://blogger.googleusercontent.com/img/b/s100/s72-c/image.png"
        self.assertEqual(
            upgrade_image_url(url_multi, "s800"),
            "https://blogger.googleusercontent.com/img/b/s100/s800/image.png",
        )
        self.assertEqual(
            to_thumbnail_url(url_multi, "s120-c"),
            "https://blogger.googleusercontent.com/img/b/s100/s120-c/image.png",
        )

    def test_is_allowed_image_host(self):
        """Verify host whitelist enforcement."""
        self.assertTrue(is_allowed_image_host("blogger.googleusercontent.com"))
        self.assertTrue(is_allowed_image_host("1.bp.blogspot.com"))
        self.assertTrue(is_allowed_image_host("www.irasutoya.com"))
        self.assertFalse(is_allowed_image_host("evilgoogleusercontent.com"))
        self.assertFalse(is_allowed_image_host("not-irasutoya.com"))
        self.assertFalse(is_allowed_image_host("blogger.com"))

    def test_parse_entries_multi_image(self):
        """Test parsing Blogger feed entry with multiple illustration images."""
        sample_entries = [
            {
                "id": {"$t": "tag:blogger.com,1999:blog.post-101"},
                "title": {"$t": "猫のイラスト"},
                "summary": {"$t": "可愛い猫のイラストです。"},
                "published": {"$t": "2024-01-01T00:00:00Z"},
                "updated": {"$t": "2024-01-02T00:00:00Z"},
                "category": [{"term": "動物"}],
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
                },
            }
        ]

        images = parse_entries(sample_entries, image_size="s800")
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].title, "白猫のイラスト")
        self.assertEqual(images[0].image_url, "https://1.bp.blogspot.com/s800/cat_white.png")
        self.assertEqual(images[0].thumb_url, "https://1.bp.blogspot.com/s120-c/cat_white.png")
        self.assertEqual(images[0].post_title, "猫のイラスト")

        self.assertEqual(images[1].title, "黒猫のイラスト")
        self.assertEqual(images[1].image_url, "https://1.bp.blogspot.com/s800/cat_black.png")

        # Also test parse_entry directly
        parsed_post = parse_entry(sample_entries[0], image_size="s800")
        self.assertEqual(parsed_post.blogger_id, "101")
        self.assertEqual(parsed_post.categories, ("動物",))
        self.assertEqual(len(parsed_post.images), 2)

    def test_parse_entries_thumbnail_fallback(self):
        """Test fallback to post thumbnail when entry content lacks explicit <a><img> links."""
        sample_entries = [
            {
                "title": {"$t": "リンゴのイラスト"},
                "summary": {"$t": "リンゴです。"},
                "content": {"$t": "<div>テキストのみ</div>"},
                "media$thumbnail": {
                    "url": "https://1.bp.blogspot.com/s72-c/apple.png"
                },
            }
        ]

        images = parse_entries(sample_entries, image_size="s800")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].title, "リンゴのイラスト")
        self.assertEqual(images[0].image_url, "https://1.bp.blogspot.com/s800/apple.png")
        self.assertEqual(images[0].thumb_url, "https://1.bp.blogspot.com/s120-c/apple.png")

    def test_parse_entries_rejects_non_image_extension_urls(self):
        """Verify that links ending with .png.html or other non-image suffixes are not treated as images."""
        sample_entry = [
            {
                "title": {"$t": "HTML Link Post"},
                "content": {
                    "$t": '<a href="https://1.bp.blogspot.com/photo.png.html"><img src="thumb.jpg"></a>'
                },
            }
        ]
        results = parse_entries(sample_entry)
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
                },
            }
        ]
        results = parse_entries(sample_entry)
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
                },
            }
        ]
        results = parse_entries(sample_entry)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Secure")
        self.assertEqual(results[0].image_url, "https://1.bp.blogspot.com/secure.png")

        # Test thumbnail fallback with HTTP
        fallback_entry = [
            {
                "title": {"$t": "Insecure Thumbnail"},
                "content": {"$t": "text only"},
                "media$thumbnail": {"url": "http://1.bp.blogspot.com/thumb.png"},
            }
        ]
        self.assertEqual(len(parse_entries(fallback_entry)), 0)

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
                },
            }
        ]
        results = parse_entries(sample_entry)
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
                },
            }
        ]
        results = parse_entries(sample_entry)
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
                },
            }
        ]
        results = parse_entries(sample_entry)
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
                "media$thumbnail": {"url": "https://1.bp.blogspot.com/thumb.png?a=1&amp;b=2"},
            }
        ]
        results = parse_entries(fallback_entry)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].thumb_url, "https://1.bp.blogspot.com/thumb.png?a=1&b=2")

    def test_parse_entries_standalone_img(self):
        """Test parsing Blogger feed entry where images are standalone <img> tags without wrapping <a>."""
        sample_entries = [
            {
                "title": {"$t": "犬のイラスト"},
                "summary": {"$t": "可愛い犬のイラストです。"},
                "content": {
                    "$t": (
                        '<div class="separator">'
                        '<img alt="柴犬のイラスト" src="https://1.bp.blogspot.com/s400/shiba.png" />'
                        '</div>'
                    )
                },
            }
        ]
        images = parse_entries(sample_entries, image_size="s800")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].title, "柴犬のイラスト")
        self.assertEqual(images[0].image_url, "https://1.bp.blogspot.com/s800/shiba.png")
        self.assertEqual(images[0].thumb_url, "https://1.bp.blogspot.com/s120-c/shiba.png")

    def test_parse_entries_falls_back_when_alt_is_placeholder_bullet(self):
        """Verify that meaningless placeholder alt tags like '■' or '・' fall back to post_title."""
        sample_entries = [
            {
                "title": {"$t": "ネズミを追いかける猫のイラスト"},
                "content": {
                    "$t": (
                        '<div class="separator">'
                        '<img alt="■" src="https://1.bp.blogspot.com/s400/cat.png" />'
                        '</div>'
                    )
                },
            }
        ]
        images = parse_entries(sample_entries)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].title, "ネズミを追いかける猫のイラスト")

    def test_parse_entries_excludes_script_and_style_data_from_description(self):
        """Verify script and style blocks do not leak into post descriptions or cause false matches."""
        sample_entries = [
            {
                "title": {"$t": "敗北のイラスト"},
                "content": {
                    "$t": (
                        '<script>var label = "勝利";</script>'
                        '<style>.shori { color: red; }</style>'
                        '<div class="separator">'
                        '<img alt="敗北のイラスト" src="https://1.bp.blogspot.com/s400/loss.png" />'
                        '</div>'
                        '<p>試合に負けて悔しがっている人のイラストです。</p>'
                    )
                },
            }
        ]
        images = parse_entries(sample_entries)
        self.assertEqual(len(images), 1)
        img = images[0]
        self.assertNotIn("勝利", img.description)
        self.assertIn("試合に負けて悔しがっている人のイラストです。", img.description)
        # Verify an otherwise unrelated single-image post is not matched using script text
        self.assertFalse(is_image_relevant(img, "勝利"))

    def test_parse_entries_ignores_tags_inside_noscript(self):
        """Verify images inside noscript blocks are ignored and do not misclassify single-image posts."""
        sample_entries = [
            {
                "title": {"$t": "走る人のイラスト"},
                "content": {
                    "$t": (
                        '<div class="separator">'
                        '<a href="https://1.bp.blogspot.com/s1600/run.png">'
                        '<img alt="走る人のイラスト" src="https://1.bp.blogspot.com/s400/run.png" />'
                        '</a>'
                        '<noscript>'
                        '<a href="https://1.bp.blogspot.com/s1600/ignored.png">'
                        '<img alt="無視される画像" src="https://1.bp.blogspot.com/s400/ignored.png" />'
                        '</a>'
                        '</noscript>'
                        '</div>'
                        '<p>元気に走っている人のイラストです。</p>'
                    )
                },
            }
        ]
        images = parse_entries(sample_entries)
        self.assertEqual(len(images), 1)
        img = images[0]
        self.assertEqual(img.title, "走る人のイラスト")
        self.assertEqual(img.image_url, "https://1.bp.blogspot.com/s800/run.png")
        self.assertTrue(img.is_single_image_post)
        # Description matching remains active for genuine single-image posts
        self.assertTrue(is_image_relevant(img, "元気"))

    def test_parse_entries_tracks_multi_image_posts_even_with_cross_post_duplicates(self):
        """Verify that multi-image posts are not misclassified as single-image posts when duplicate images are deduplicated."""
        entries = [
            {
                "title": {"$t": "Post A"},
                "content": {"$t": '<img src="https://1.bp.blogspot.com/img1.png" alt="img 1">'},
                "summary": {"$t": "Summary A"},
            },
            {
                "title": {"$t": "Post B"},
                "content": {
                    "$t": (
                        '<img src="https://1.bp.blogspot.com/img1.png" alt="img 1">'
                        '<img src="https://1.bp.blogspot.com/img2.png" alt="img 2">'
                    )
                },
                "summary": {"$t": "Summary B"},
            },
        ]
        images = parse_entries(entries, "s800")
        self.assertEqual(len(images), 2)
        # img 1 from Post A is a single image post
        self.assertTrue(images[0].is_single_image_post)
        # img 2 from Post B should NOT be marked single image post because Post B originally had 2 images
        self.assertFalse(images[1].is_single_image_post)

    def test_parse_entries_separates_adjacent_html_elements_with_spaces(self):
        """Verify text from separate HTML elements is space-delimited to prevent fusing words into compound Kanji runs."""
        entries = [
            {
                "title": {"$t": "Post A"},
                "content": {
                    "$t": '<div>猫</div><div>犬の説明</div><img src="https://1.bp.blogspot.com/img1.png" alt="img 1">'
                },
                "summary": {"$t": ""},
            }
        ]
        images = parse_entries(entries, "s800")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].description, "猫 犬の説明")

    def test_parse_entries_skips_malformed_entries_and_retains_valid(self):
        """Verify parse_entries catches exceptions on malformed entries and continues processing valid entries."""
        valid_entry = {
            "title": {"$t": "有効なポスト"},
            "content": {"$t": '<img src="https://1.bp.blogspot.com/valid.png" alt="Valid Image">'},
            "summary": {"$t": "説明"},
        }
        malformed_entry_1 = {"content": None}  # Missing title and non-dict content
        malformed_entry_2 = "not a dict"

        entries = [malformed_entry_1, valid_entry, malformed_entry_2]
        images = parse_entries(entries, "s800")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].title, "Valid Image")

    def test_parse_entries_does_not_commit_seen_urls_on_failed_entry(self):
        """Verify that seen URLs from a failed entry are discarded and not blocked in later entries."""
        # Entry 1 adds its image URL to entry_seen, then fails while building the thumbnail
        failing_entry = {
            "title": {"$t": "失敗ポスト"},
            "content": {
                "$t": '<img src="https://1.bp.blogspot.com/shared.png" alt="Shared Image">'
            },
            "summary": {"$t": ""},
        }
        # Entry 2 is valid and reuses the same image URL
        valid_entry = {
            "title": {"$t": "成功ポスト"},
            "content": {
                "$t": '<img src="https://1.bp.blogspot.com/shared.png" alt="Shared Image">'
            },
            "summary": {"$t": ""},
        }

        real_thumb = to_thumbnail_url
        calls = {"n": 0}

        def flaky_thumb(url, size="s120-c"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Temporary thumbnail error")
            return real_thumb(url, size)

        with patch("addon.feed.to_thumbnail_url", side_effect=flaky_thumb):
            images = parse_entries([failing_entry, valid_entry], "s800")

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].image_url, "https://1.bp.blogspot.com/shared.png")
        self.assertEqual(images[0].post_title, "成功ポスト")


if __name__ == "__main__":
    unittest.main()
