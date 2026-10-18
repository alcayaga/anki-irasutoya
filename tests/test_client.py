"""Unit tests for the Irasutoya scraper client."""
import json
import unittest
import urllib.error
from client import IrasutoyaClient, IrasutoyaImage, is_decorative_asset, is_image_relevant, select_best_image


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
                }
            }
        ]
        images = self.client.parse_entries(sample_entries, image_size="s800")
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
                }
            }
        ]
        images = self.client.parse_entries(sample_entries)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].title, "ネズミを追いかける猫のイラスト")

    def test_score_candidate_image(self):
        """Test archetype scoring engine penalties and boosts."""
        from client import score_candidate_image

        img_exact = IrasutoyaImage(
            title="本",
            image_url="https://example.com/book.png",
            thumb_url="https://example.com/thumb.png",
            post_title="本のイラスト"
        )
        score_exact = score_candidate_image(img_exact, "本")

        img_tax = IrasutoyaImage(
            title="確定申告で税金を払う人のイラスト",
            image_url="https://example.com/tax.png",
            thumb_url="https://example.com/thumb.png",
            post_title="税金・確定申告のイラスト"
        )
        score_tax = score_candidate_image(img_tax, "本")

        self.assertGreater(score_exact, score_tax)
        self.assertGreater(score_exact, 100)
        self.assertLess(score_tax, 50)

    def test_select_best_image(self):
        """Test selecting the highest scoring image among candidates."""
        from client import select_best_image

        img_bad = IrasutoyaImage(
            title="本を投げ捨てる問題児のイラスト",
            image_url="https://example.com/bad.png",
            thumb_url="https://example.com/thumb.png",
            post_title="トラブルのイラスト"
        )
        img_good = IrasutoyaImage(
            title="開いた本のイラスト",
            image_url="https://example.com/good.png",
            thumb_url="https://example.com/thumb.png",
            post_title="本のイラスト"
        )
        best = select_best_image([img_bad, img_good], "本")
        self.assertEqual(best, img_good)

    def test_search_with_fallbacks(self):
        """Test candidate fallback execution when primary returns 0 results."""
        client = IrasutoyaClient()

        def mock_search(q, max_results=3, image_size="s800", orderby="relevance"):
            if q == "そのまま":
                return [IrasutoyaImage("そのままのイラスト", "https://example.com/img.png", "https://example.com/t.png", "そのまま")]
            return []

        client.search = mock_search

        candidates = [
            ("其の儘", "primary", 1),
            ("そのまま", "note_reading_Reading", 2)
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
        import urllib.error
        client = IrasutoyaClient()

        def mock_failing_search(q, **kwargs):
            if q == "其の儘":
                raise urllib.error.HTTPError("https://example.com", 429, "Too Many Requests", {}, None)
            return [IrasutoyaImage("そのままのイラスト", "https://example.com/img.png", "https://example.com/t.png", "そのまま")]

        client.search = mock_failing_search
        candidates = [
            ("其の儘", "primary", 1),
            ("そのまま", "note_reading_Reading", 2)
        ]

        with self.assertRaises(urllib.error.HTTPError):
            client.search_with_fallbacks(candidates)

    def test_is_decorative_asset(self):
        """Verify decorative noise detection for frames, borders, and templates."""
        frame_img = IrasutoyaImage("花のフレーム", "https://example.com/f.png", "https://example.com/t.png", "花")
        self.assertTrue(is_decorative_asset(frame_img, "花"))
        # Explicit search for frame should not be treated as unwanted decorative noise
        self.assertFalse(is_decorative_asset(frame_img, "フレーム"))

        cat_img = IrasutoyaImage("かわいい猫のイラスト", "https://example.com/c.png", "https://example.com/t.png", "猫")
        self.assertFalse(is_decorative_asset(cat_img, "猫"))

    def test_search_with_fallbacks_skips_decorative_only_results(self):
        """Verify search_with_fallbacks continues to fallback candidate if initial candidate only yields decorative assets."""
        client = IrasutoyaClient()

        def mock_search(q, max_results=3, image_size="s800", orderby="relevance"):
            if q == "追いかける":
                # Only decorative frame returned
                return [IrasutoyaImage("フレームのイラスト", "https://example.com/frame.png", "https://example.com/t.png", "フレーム")]
            if q == "走る":
                # Actual illustration returned
                return [IrasutoyaImage("走る人のイラスト", "https://example.com/run.png", "https://example.com/t.png", "走る")]
            return []

        client.search = mock_search
        candidates = [
            ("追いかける", "primary", 1),
            ("走る", "note_reading", 2),
        ]
        images, matched_cand, best_img = client.search_with_fallbacks(candidates)
        self.assertIsNotNone(matched_cand)
        self.assertEqual(matched_cand[0], "走る")
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

        # Okurigana verb stems fused with Kanji must be rejected (売り場 != 売り, 食べ物 != 食べ)
        uriba_img = IrasutoyaImage("売り場のイラスト", "https://example.com/u.png", "https://example.com/t.png", "売り場")
        self.assertFalse(is_image_relevant(uriba_img, "売り"))
        tabemono_img = IrasutoyaImage("食べ物のイラスト", "https://example.com/tb.png", "https://example.com/t.png", "食べ物")
        self.assertFalse(is_image_relevant(tabemono_img, "食べ"))

        # Genuine Kanji-noun compounds should still be accepted
        hada_img = IrasutoyaImage("乾燥肌のイラスト", "https://example.com/h.png", "https://example.com/t.png", "乾燥肌")
        self.assertTrue(is_image_relevant(hada_img, "肌"))

    def test_search_with_fallbacks_skips_unrelated_images(self):
        """Verify search_with_fallbacks skips candidates that only return unrelated images from multi-image posts."""
        client = IrasutoyaClient()

        def mock_search(q, max_results=3, image_size="s800", orderby="relevance"):
            if q == "猫":
                # Returns only a dog image from a post that mentioned cat
                return [IrasutoyaImage("犬のイラスト", "https://example.com/dog.png", "https://example.com/t.png", "猫のイラスト")]
            if q == "ネコ":
                # Returns an actual cat illustration
                return [IrasutoyaImage("ネコのイラスト", "https://example.com/cat.png", "https://example.com/t.png", "ネコのイラスト")]
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

    def test_classify_match_boundary(self):
        """Test fine-grained morphological boundary classification."""
        from client import classify_match_boundary

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

    def test_score_callout_text_heavy_penalty(self):
        """Test that illustrated text callouts receive heavy penalty instead of hard rejection."""
        from client import score_candidate_image, is_image_relevant, is_decorative_asset

        img_callout = IrasutoyaImage(
            title="「決定！」のイラスト文字",
            image_url="https://example.com/kettei_text.png",
            thumb_url="https://example.com/thumb.png",
            post_title="「決定！」のイラスト文字"
        )
        img_scene = IrasutoyaImage(
            title="決定を下す人のイラスト",
            image_url="https://example.com/kettei_scene.png",
            thumb_url="https://example.com/thumb.png",
            post_title="決定のイラスト"
        )

        # Callout text is not hard-rejected by is_decorative_asset
        self.assertFalse(is_decorative_asset(img_callout, "決定"))
        self.assertTrue(is_image_relevant(img_callout, "決定"))

        # But scene illustration scores significantly higher than text callout
        score_callout = score_candidate_image(img_callout, "決定")
        score_scene = score_candidate_image(img_scene, "決定")
        self.assertGreater(score_scene, score_callout)

    def test_single_image_post_description_relevance(self):
        """Test accepting relevant single-image posts whose description matches query."""
        from client import is_image_relevant

        single_img = IrasutoyaImage(
            title="ガッツポーズをする人のイラスト",
            image_url="https://example.com/win.png",
            thumb_url="https://example.com/thumb.png",
            post_title="ガッツポーズをする人のイラスト",
            description="勝った人や勝利を喜んでいる人のイラストです。"
        )
        single_img.is_single_image_post = True

        # Matches '勝利' in caption of single image post
        self.assertTrue(is_image_relevant(single_img, "勝利"))

        # On multi-image posts, description alone is not sufficient
        single_img.is_single_image_post = False
        self.assertFalse(is_image_relevant(single_img, "勝利"))

    def test_parse_entries_tracks_multi_image_posts_even_with_cross_post_duplicates(self):
        """Verify that multi-image posts are not misclassified as single-image posts when duplicate images are deduplicated."""
        client = IrasutoyaClient()
        entries = [
            {
                "title": {"$t": "Post A"},
                "content": {"$t": '<img src="https://1.bp.blogspot.com/img1.png" alt="img 1">'},
                "summary": {"$t": "Summary A"}
            },
            {
                "title": {"$t": "Post B"},
                "content": {"$t": '<img src="https://1.bp.blogspot.com/img1.png" alt="img 1"><img src="https://1.bp.blogspot.com/img2.png" alt="img 2">'},
                "summary": {"$t": "Summary B"}
            }
        ]
        images = client.parse_entries(entries, "s800")
        self.assertEqual(len(images), 2)
        # img 1 from Post A is a single image post
        self.assertTrue(images[0].is_single_image_post)
        # img 2 from Post B should NOT be marked single image post because Post B originally had 2 images
        self.assertFalse(images[1].is_single_image_post)

    def test_parse_entries_separates_adjacent_html_elements_with_spaces(self):
        """Verify text from separate HTML elements is space-delimited to prevent fusing words into compound Kanji runs."""
        client = IrasutoyaClient()
        entries = [
            {
                "title": {"$t": "Post A"},
                "content": {"$t": '<div>猫</div><div>犬の説明</div><img src="https://1.bp.blogspot.com/img1.png" alt="img 1">'},
                "summary": {"$t": ""}
            }
        ]
        images = client.parse_entries(entries, "s800")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].description, "猫 犬の説明")

    def test_candidate_regression_fixture(self):
        """Automated regression fixture verifying candidate generation and boundary matching on Mining and Kaishi 1.5k decks."""
        from text_utils import generate_search_candidates, BATCH_MAX_CANDIDATES
        from client import classify_match_boundary

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
                    candidates = generate_search_candidates(term, note=note, allow_tier2=True, allow_deinflection=True)[:BATCH_MAX_CANDIDATES]
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
            ("中性的な人のイラスト", "中"),
        ]
        for title, query in negative_cases:
            with self.subTest(negative_case=query):
                quality = classify_match_boundary(title, query)
                self.assertIn(quality, ("none", "compound"), f"Query '{query}' in '{title}' must not match as standalone/inflected")


if __name__ == "__main__":
    unittest.main()

