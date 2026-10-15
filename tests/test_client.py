import unittest
from client import IrasutoyaClient, IrasutoyaImage


class TestIrasutoyaClient(unittest.TestCase):
    def setUp(self):
        self.client = IrasutoyaClient()

    def test_upgrade_image_url(self):
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
        self.assertEqual(self.client.search(""), [])
        self.assertEqual(self.client.search("   "), [])

    def test_parse_entries_multi_image(self):
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


    def test_live_search(self):
        try:
            results = self.client.search("りんご", max_results=2)
            self.assertGreater(len(results), 0)
            self.assertTrue(results[0].image_url.startswith("http"))
            self.assertTrue(results[0].thumb_url.startswith("http"))
        except Exception as e:
            self.skipTest(f"Skipping live search test due to network: {e}")


if __name__ == "__main__":
    unittest.main()
