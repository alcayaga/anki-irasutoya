"""Unit tests for the local LLM visual concept compiler (scripts/agent_synonyms.py)."""
import io
import json
import os
import sqlite3
import tempfile
import unittest
import urllib.error
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from addon.catalog import CatalogStore, write_posts
from addon.feed import IrasutoyaImage, ParsedPost
from scripts.agent_synonyms import (
    _contains_english_word,
    _fetch_yomitan_term_entries,
    _parse_yomitan_metadata,
    _save_yomitan_cache,
    check_ollama_availability,
    fetch_jpdb_freq_github,
    load_jpdb_candidates,
    main,
    query_yomitan_term_metadata,
    run_ollama_agent_batch,
    run_ollama_compile,
    search_catalog_for_agent,
)


class TestYomitanJpdbCandidates(unittest.TestCase):
    """Test suite for JPDB v2.2 candidate extraction and Yomitan API enrichment."""

    def setUp(self) -> None:
        """Create an isolated temporary catalog and environment for hermetic testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_catalog.sqlite")

        # Create basic catalog database with one post
        conn = sqlite3.connect(self.db_path)
        img = IrasutoyaImage("可愛い猫", "https://1.bp.blogspot.com/s800/cat.png", "t.png", "猫", post_id="blogger_99999")
        post = ParsedPost("blogger_99999", "猫のイラスト", "desc", "pub", "upd", (), [img])
        write_posts(conn, [post])
        conn.close()

        # Hermetic default: never probe a live Ollama server from unit tests
        ollama_patcher = patch(
            "scripts.synonyms.ollama_agent.check_ollama_availability",
            return_value=(True, None),
        )
        ollama_patcher.start()
        self.addCleanup(ollama_patcher.stop)

    def tearDown(self) -> None:
        """Clean up temporary test workspace."""
        self.temp_dir.cleanup()

    def test_fetch_jpdb_freq_github_uses_cache(self) -> None:
        """Verify fetch_jpdb_freq_github reuses existing cached file when valid."""
        cache_dir = os.path.join(self.temp_dir.name, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(cache_dir, "jpdb_v2.2_freq_list_2024-10-13.csv")
        with open(cached_file, "wb") as f:
            f.write(b"term\treading\tfreq\tkana_freq\n" * 100)

        with patch("urllib.request.urlopen") as mock_url:
            path = fetch_jpdb_freq_github(cache_dir=cache_dir)
            mock_url.assert_not_called()
        self.assertEqual(path, os.path.abspath(cached_file))

    def test_fetch_jpdb_freq_github_derives_cache_filename(self) -> None:
        """Verify fetch_jpdb_freq_github extracts cache filename from the URL path."""
        custom_url = "https://example.com/data/custom_vocab_list.csv?token=secret123"
        cached_file = os.path.join(self.temp_dir.name, "custom_vocab_list.csv")
        with open(cached_file, "w", encoding="utf-8") as f:
            f.write("test content" * 100)

        with patch("urllib.request.urlopen") as mock_url:
            path = fetch_jpdb_freq_github(cache_dir=self.temp_dir.name, url=custom_url)
            mock_url.assert_not_called()
        self.assertEqual(os.path.basename(path), "custom_vocab_list.csv")

    def test_query_yomitan_term_metadata(self) -> None:
        """Verify query_yomitan_term_metadata extracts structured JMdict POS and glosses."""
        mock_resp_data = {
            "dictionaryEntries": [
                {
                    "headwords": [{"term": "読む", "reading": "よむ"}],
                    "definitions": [
                        {
                            "dictionary": "JMdict (Spanish)",
                            "tags": [{"name": "v5m", "category": "partOfSpeech"}],
                            "entries": ["leer"],
                        },
                        {
                            "dictionary": "JMdict [2026-08-20]",
                            "tags": [
                                {"name": "v5m", "category": "partOfSpeech"},
                                {"name": "vt", "category": "partOfSpeech"},
                            ],
                            "entries": [
                                {
                                    "type": "structured-content",
                                    "content": {
                                        "tag": "ul",
                                        "data": {"content": "glossary"},
                                        "content": [{"tag": "li", "content": "to read"}],
                                    },
                                }
                            ],
                        },
                    ],
                }
            ]
        }
        mock_bytes = json.dumps(mock_resp_data).encode("utf-8")

        class MockResponse:
            """Mock HTTP response."""

            def __enter__(self) -> "MockResponse":
                """Enter context manager and return self."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self) -> bytes:
                """Read response body bytes."""
                return mock_bytes

        def fake_urlopen(req: Any, *args: Any, **kwargs: Any) -> MockResponse:
            """Simulate urllib.request.urlopen returning MockResponse."""
            return MockResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            meta = query_yomitan_term_metadata("読む", reading="よむ")
            self.assertIsNotNone(meta)
            assert meta is not None
            self.assertIn("v5m", meta.pos_tags)
            self.assertIn("vt", meta.pos_tags)
            self.assertEqual(meta.verb_class, "v5m")
            self.assertFalse(meta.is_particle_or_aux)
            self.assertFalse(meta.is_uk)
            self.assertEqual(meta.english_glosses, ["to read"])

    def test_parse_yomitan_metadata_disambiguation_guards(self) -> None:
        """Verify _parse_yomitan_metadata guards against prefix mismatch and prioritizes uk."""
        # 1. Prefix mismatch: Yomitan returns '弟子' (reading 'でし') for query 'でした'
        deshi_entry = {
            "headwords": [{"term": "弟子", "reading": "でし"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [{"name": "n", "category": "partOfSpeech"}],
                    "entries": ["pupil", "disciple"],
                }
            ],
        }
        res_mismatch = _parse_yomitan_metadata([deshi_entry], term="でした", reading="でした")
        self.assertIsNone(res_mismatch)

        # 2. Kana homophone disambiguation: prefer '達' (is_uk=True) over '太刀' (is_uk=False) for 'たち'
        tachi_sword_entry = {
            "headwords": [{"term": "太刀", "reading": "たち"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [{"name": "n", "category": "partOfSpeech"}],
                    "entries": ["long sword"],
                }
            ],
        }
        tachi_suffix_entry = {
            "headwords": [{"term": "達", "reading": "たち"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [
                        {"name": "suf", "category": "partOfSpeech"},
                        {"name": "uk", "category": "default"},
                    ],
                    "entries": ["pluralizing suffix"],
                }
            ],
        }
        # Order: sword first, suffix second
        res_kana = _parse_yomitan_metadata(
            [tachi_sword_entry, tachi_suffix_entry],
            term="たち",
            reading="たち",
        )
        self.assertIsNotNone(res_kana)
        assert res_kana is not None
        self.assertTrue(res_kana.is_uk)
        self.assertIn("suf", res_kana.pos_tags)
        self.assertTrue(res_kana.is_particle_or_aux)
        self.assertEqual(res_kana.english_glosses, ["pluralizing suffix"])

        # 3. Ambiguous kana homophones without uk: 'はし' matching both '橋' and '箸'
        hashi_bridge_entry = {
            "headwords": [{"term": "橋", "reading": "はし"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [{"name": "n", "category": "partOfSpeech"}],
                    "entries": ["bridge"],
                }
            ],
        }
        hashi_chopsticks_entry = {
            "headwords": [{"term": "箸", "reading": "はし"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [{"name": "n", "category": "partOfSpeech"}],
                    "entries": ["chopsticks"],
                }
            ],
        }
        # Ambiguous homophones without uk -> rejected as None to prevent homophone trap
        res_ambiguous = _parse_yomitan_metadata(
            [hashi_bridge_entry, hashi_chopsticks_entry],
            term="はし",
            reading="はし",
        )
        self.assertIsNone(res_ambiguous)

        # 4. Unambiguous reading match: single distinct headword term
        res_unambiguous = _parse_yomitan_metadata(
            [hashi_bridge_entry],
            term="はし",
            reading="はし",
        )
        self.assertIsNotNone(res_unambiguous)
        assert res_unambiguous is not None
        self.assertEqual(res_unambiguous.english_glosses, ["bridge"])

    def test_load_jpdb_candidates(self) -> None:
        """Verify load_jpdb_candidates filters non-tier1, unmatched, and all modes."""
        tsv_path = os.path.join(self.temp_dir.name, "sample_jpdb.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("term\treading\tfrequency\tkana_frequency\n")  # Header row -> skipped
            f.write("まで\tまで\t1\t\n")  # Grammar function word -> skipped
            f.write("猫\tねこ\t2\t\n")    # Tier 1 in test catalog ('猫のイラスト')
            f.write("犬\tいぬ\t3\t\n")    # Unmatched in test catalog
            f.write("傷薬\tきずぐすり\t4\t\n")  # Unmatched in test catalog

        cache_dir = os.path.join(self.temp_dir.name, "jpdb_cache")

        def fake_fetch_entries(term: str, *args: Any, **kwargs: Any) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
            """Simulate Yomitan API returning structured JMdict noun entry for terms."""
            entry = {
                "headwords": [{"term": term}],
                "definitions": [
                    {
                        "dictionary": "JMdict (English)",
                        "tags": [{"name": "n", "category": "partOfSpeech"}],
                        "entries": [f"gloss for {term}"],
                    }
                ],
            }
            return [entry], True

        with patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", side_effect=fake_fetch_entries):
            # non-tier1 mode: skips 'まで' (grammar) and '猫' (Tier 1 match), keeps '犬' and '傷薬'
            cands_non_tier1, _ = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                limit=10,
                mode="non-tier1",
                cache_dir=cache_dir,
            )
            terms_non_tier1 = [c["term"] for c in cands_non_tier1]
            self.assertEqual(terms_non_tier1, ["犬", "傷薬"])
            self.assertEqual(cands_non_tier1[0]["english_glosses"], ["gloss for 犬"])

            # all mode: skips only grammar 'まで', keeps '猫', '犬', and '傷薬'
            cands_all, _ = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                limit=10,
                mode="all",
                cache_dir=cache_dir,
            )
            terms_all = [c["term"] for c in cands_all]
            self.assertEqual(terms_all, ["猫", "犬", "傷薬"])

            # Verify that header was skipped without eating candidate limit slot (inspects top 3: まで, 猫, 犬)
            cands_limit_3, _ = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                limit=3,
                mode="all",
                cache_dir=cache_dir,
            )
            self.assertEqual([c["term"] for c in cands_limit_3], ["猫", "犬"])

    def test_load_jpdb_candidates_csv_tsv_delimiter_detection(self) -> None:
        """Verify load_jpdb_candidates correctly parses comma-separated CSV input."""
        csv_path = os.path.join(self.temp_dir.name, "comma_jpdb.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("term,reading,frequency,sequence\n")
            f.write("猫,ねこ,1,100\n")
            f.write("犬,いぬ,2,101\n")

        def fake_fetch(term: str, **kwargs: Any) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
            """Simulate an empty Yomitan response without network access."""
            return [], True

        with patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", side_effect=fake_fetch):
            candidates, _ = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=csv_path,
                limit=10,
                mode="all",
                cache_dir=self.temp_dir.name,
            )
        terms = [c["term"] for c in candidates]
        self.assertIn("猫", terms)
        self.assertIn("犬", terms)

    def test_load_jpdb_candidates_consecutive_failure_cutoff(self) -> None:
        """Verify load_jpdb_candidates ceases API calls after consecutive network failures."""
        tsv_path = os.path.join(self.temp_dir.name, "fail_jpdb.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            for i in range(12):
                f.write(f"単語{i}\tたんご{i}\t{i+1}\t\n")

        call_count = 0

        def fake_fetch_fail(term: str, **kwargs: Any) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
            """Simulate offline Yomitan API failing every request."""
            nonlocal call_count
            call_count += 1
            return None, False

        with patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", side_effect=fake_fetch_fail):
            candidates, deferred_count = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                limit=12,
                mode="all",
                cache_dir=self.temp_dir.name,
                clear_yomitan_cache=True,
            )
            # Max consecutive failures is 5, so loop should stop calling API after 5 consecutive failures
            self.assertEqual(call_count, 5)
            self.assertEqual(len(candidates), 0)
            self.assertEqual(deferred_count, 12)

    def test_load_jpdb_candidates_skips_term_error_without_deferral(self) -> None:
        """Verify load_jpdb_candidates skips per-term error without caching or incrementing deferred_count."""
        tsv_path = os.path.join(self.temp_dir.name, "single_term.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("問題語\tもんだいご\t1\n良語\tりょうご\t2\n")

        def fake_fetch(term: str, **kwargs: Any) -> Tuple[Optional[List[Dict[str, Any]]], str]:
            """Simulate term error for one term and ok for other."""
            if term == "問題語":
                return None, "term_error"
            return [{"headwords": [{"term": "良語", "reading": "りょうご"}]}], "ok"

        with patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", side_effect=fake_fetch):
            cands, deferred = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                mode="all",
                cache_dir=self.temp_dir.name,
            )
            # Problematic term was skipped without being deferred
            self.assertEqual(deferred, 0)
            self.assertEqual(len(cands), 1)
            self.assertEqual(cands[0]["term"], "良語")

            # Check that cache does NOT contain the problematic term
            cache_path = os.path.join(self.temp_dir.name, "yomitan_api_cache.json")
            with open(cache_path, "r", encoding="utf-8") as f:
                saved_cache = json.load(f)
            self.assertNotIn("http://127.0.0.1:19633\t問題語\tもんだいご", saved_cache)

    def test_load_jpdb_candidates_defers_during_outage(self) -> None:
        """Verify candidates are skipped/deferred during Yomitan API outages."""
        tsv_path = os.path.join(self.temp_dir.name, "sample_outage.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("term\treading\tfrequency\n猫\tねこ\t1\n犬\tいぬ\t2\n")

        cache_dir = os.path.join(self.temp_dir.name, "cache_outage")
        with patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", return_value=([], False)):
            cands, deferred = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                cache_dir=cache_dir,
            )
            # Both terms should be deferred because Yomitan lookups failed
            self.assertEqual(len(cands), 0)
            self.assertEqual(deferred, 2)

    def test_is_english_jmdict_filtering(self) -> None:
        """Verify _parse_yomitan_metadata filters non-English JMdict dictionaries."""
        french_entries = [
            {
                "headwords": [{"term": "林檎", "reading": "りんご"}],
                "definitions": [
                    {
                        "dictionary": "JMdict (French)",
                        "tags": [{"name": "n", "category": "partOfSpeech"}],
                        "entries": ["pomme"],
                    }
                ],
            }
        ]
        meta_fr = _parse_yomitan_metadata(french_entries, term="林檎")
        self.assertIsNone(meta_fr)

        english_entries = [
            {
                "headwords": [{"term": "林檎", "reading": "りんご"}],
                "definitions": [
                    {
                        "dictionary": "JMdict (English)",
                        "tags": [{"name": "n", "category": "partOfSpeech"}],
                        "entries": ["apple"],
                    }
                ],
            }
        ]
        meta_en = _parse_yomitan_metadata(english_entries, term="林檎")
        self.assertIsNotNone(meta_en)
        if meta_en is not None:
            self.assertEqual(meta_en.english_glosses, ["apple"])

    def test_parse_yomitan_metadata_handles_non_dict_entries(self) -> None:
        """Verify _parse_yomitan_metadata tolerates non-dictionary elements gracefully."""
        malformed_entries: List[Any] = [
            None,
            "not a dict",
            {"definitions": ["not a dict", None], "headwords": ["not a dict"]},
            {
                "definitions": [
                    {
                        "dictionary": "JMdict (English)",
                        "tags": [None, "invalid"],
                        "entries": ["feline animal"],
                    }
                ],
                "headwords": [{"term": "猫", "reading": "ねこ"}],
            },
        ]
        meta = _parse_yomitan_metadata(malformed_entries, term="猫", reading="ねこ")
        self.assertIsNotNone(meta)
        if meta:
            self.assertIn("feline animal", meta.english_glosses)

    def test_fetch_yomitan_term_entries_handling(self) -> None:
        """Verify _fetch_yomitan_term_entries handles valid, non-object, and malformed responses."""

        class MockHttpResponse:
            """Mock HTTP response wrapping arbitrary payload."""

            def __init__(self, body: bytes) -> None:
                """Initialize with response body bytes."""
                self._body = body

            def __enter__(self) -> "MockHttpResponse":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self) -> bytes:
                """Return response bytes."""
                return self._body

        # 1. Normal dict return with valid list
        valid_json = json.dumps({"dictionaryEntries": [{"headword": "犬"}]}).encode("utf-8")
        with patch("urllib.request.urlopen", return_value=MockHttpResponse(valid_json)):
            entries, status = _fetch_yomitan_term_entries("犬")
            self.assertEqual(status, "ok")
            self.assertEqual(entries, [{"headword": "犬"}])

        # 2. Non-object JSON return (e.g. JSON list or string)
        list_json = json.dumps(["invalid", "array"]).encode("utf-8")
        with patch("urllib.request.urlopen", return_value=MockHttpResponse(list_json)):
            entries, status = _fetch_yomitan_term_entries("犬")
            self.assertEqual(status, "term_error")
            self.assertIsNone(entries)

        # 3. Malformed dictionaryEntries (non-list, e.g. integer)
        malformed_json = json.dumps({"dictionaryEntries": 123}).encode("utf-8")
        with patch("urllib.request.urlopen", return_value=MockHttpResponse(malformed_json)):
            entries, status = _fetch_yomitan_term_entries("犬")
            self.assertEqual(status, "term_error")
            self.assertIsNone(entries)

        # 4. Connection error (URLError)
        net_err = urllib.error.URLError("Connection reset")
        with patch("urllib.request.urlopen", side_effect=net_err):
            entries, status = _fetch_yomitan_term_entries("犬")
            self.assertEqual(status, "connection_error")
            self.assertIsNone(entries)

        # 5. HTTP error (HTTPError 500)
        http_err = urllib.error.HTTPError(
            url="http://127.0.0.1:19633/termEntries",
            code=500,
            msg="Internal Server Error",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(b"error"),
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            entries, status = _fetch_yomitan_term_entries("犬")
            self.assertEqual(status, "term_error")
            self.assertIsNone(entries)

        # 6. Response missing dictionaryEntries key
        missing_key_json = json.dumps({"otherField": []}).encode("utf-8")
        with patch("urllib.request.urlopen", return_value=MockHttpResponse(missing_key_json)):
            entries, status = _fetch_yomitan_term_entries("犬")
            self.assertEqual(status, "term_error")
            self.assertIsNone(entries)

    def test_save_yomitan_cache_atomic_write(self) -> None:
        """Verify _save_yomitan_cache writes data atomically without leaving temp files."""
        cache_path = os.path.join(self.temp_dir.name, "sub_dir", "test_cache.json")
        sample_cache = {"key1": {"pos_tags": ["n"]}}

        _save_yomitan_cache(sample_cache, cache_path)

        self.assertTrue(os.path.isfile(cache_path))
        self.assertFalse(os.path.isfile(f"{cache_path}.tmp"))
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data, sample_cache)

    def test_yomitan_cache_handles_non_dict_root(self) -> None:
        """Verify load_jpdb_candidates safely ignores non-dictionary cache files."""
        tsv_path = os.path.join(self.temp_dir.name, "sample.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("term\treading\tfrequency\n猫\tねこ\t1\n")

        cache_dir = os.path.join(self.temp_dir.name, "cache_non_dict")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "yomitan_api_cache.json")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(["not", "a", "dict"], f)

        with patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", return_value=([], True)):
            cands, _ = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                cache_dir=cache_dir,
                mode="all",
            )
            self.assertEqual(len(cands), 1)

    def test_yomitan_cache_handles_corrupt_entry(self) -> None:
        """Verify load_jpdb_candidates treats non-dict cached entries as cache misses."""
        tsv_path = os.path.join(self.temp_dir.name, "sample_corrupt.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("term\treading\tfrequency\n猫\tねこ\t1\n")

        cache_dir = os.path.join(self.temp_dir.name, "cache_corrupt_entry")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "yomitan_api_cache.json")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"http://127.0.0.1:19633\t猫\tねこ": ["invalid", "corrupt"]}, f)

        mock_entries = [
            {
                "headwords": [{"term": "猫", "reading": "ねこ"}],
                "definitions": [{"tags": [{"name": "n"}]}],
            }
        ]
        with patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", return_value=(mock_entries, True)) as mock_fetch:
            cands, _ = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                cache_dir=cache_dir,
                mode="all",
            )
            self.assertEqual(len(cands), 1)
            mock_fetch.assert_called_once()

    def test_yomitan_cache_handles_valid_and_malformed_pos_tags(self) -> None:
        """Verify load_jpdb_candidates loads valid cached lists and refetches on malformed tags or non-bool flags."""
        tsv_path = os.path.join(self.temp_dir.name, "cache_tags.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("有効\tゆうこう\t1\n無効\tむこう\t2\n未定\tみてい\t3\n保留\tほりゅう\t4\n")

        cache_path = os.path.join(self.temp_dir.name, "yomitan_api_cache.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                # Valid cached entry with list of strings and bool flag
                "http://127.0.0.1:19633\t有効\tゆうこう": {
                    "pos_tags": ["n"],
                    "english_glosses": ["effective"],
                    "is_uk": False,
                    "is_particle_or_aux": False,
                    "verb_class": None,
                },
                # Malformed entry with non-list pos_tags (e.g. int)
                "http://127.0.0.1:19633\t無効\tむこう": {
                    "pos_tags": 5,
                    "english_glosses": None,
                    "is_uk": None,
                    "is_particle_or_aux": False,
                    "verb_class": None,
                },
                # Malformed entry with list containing unhashable dicts
                "http://127.0.0.1:19633\t未定\tみてい": {
                    "pos_tags": [{"name": "n"}],
                    "english_glosses": None,
                    "is_uk": None,
                    "is_particle_or_aux": False,
                    "verb_class": None,
                },
                # Malformed entry with string is_particle_or_aux instead of bool
                "http://127.0.0.1:19633\t保留\tほりゅう": {
                    "pos_tags": ["n"],
                    "english_glosses": ["pending"],
                    "is_uk": False,
                    "is_particle_or_aux": "false",
                    "verb_class": None,
                },
            }, f)

        mock_entries_map = {
            "無効": [
                {
                    "headwords": [{"term": "無効", "reading": "むこう"}],
                    "definitions": [
                        {
                            "dictionary": "JMdict (English)",
                            "tags": [{"name": "n", "category": "partOfSpeech"}],
                            "entries": ["invalid"],
                        }
                    ],
                }
            ],
            "未定": [
                {
                    "headwords": [{"term": "未定", "reading": "みてい"}],
                    "definitions": [
                        {
                            "dictionary": "JMdict (English)",
                            "tags": [{"name": "n", "category": "partOfSpeech"}],
                            "entries": ["undecided"],
                        }
                    ],
                }
            ],
            "保留": [
                {
                    "headwords": [{"term": "保留", "reading": "ほりゅう"}],
                    "definitions": [
                        {
                            "dictionary": "JMdict (English)",
                            "tags": [{"name": "n", "category": "partOfSpeech"}],
                            "entries": ["pending"],
                        }
                    ],
                }
            ],
        }

        def fake_fetch(t: str, yomitan_url: str = "") -> Tuple[Any, str]:
            """Return mock entries per term."""
            return mock_entries_map.get(t, []), "ok"

        with patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", side_effect=fake_fetch) as mock_fetch:
            cands, deferred = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                mode="all",
                cache_dir=self.temp_dir.name,
            )
            self.assertEqual(deferred, 0)
            self.assertEqual(len(cands), 4)
            # 有効 loaded from cache without API call
            self.assertEqual(cands[0]["term"], "有効")
            self.assertEqual(cands[0]["pos_tags"], ["n"])
            # 無効 (int pos_tags), 未定 (dict in pos_tags), and 保留 (string is_particle_or_aux) are refetched
            self.assertEqual(mock_fetch.call_count, 3)
            self.assertEqual(cands[1]["term"], "無効")
            self.assertEqual(cands[1]["pos_tags"], ["n"])
            self.assertEqual(cands[2]["term"], "未定")
            self.assertEqual(cands[2]["pos_tags"], ["n"])
            self.assertEqual(cands[3]["term"], "保留")
            self.assertEqual(cands[3]["pos_tags"], ["n"])


class TestOllamaAgentCompiler(unittest.TestCase):
    """Test suite for Ollama tool loop, catalog searching, and concept persistence."""

    def setUp(self) -> None:
        """Create an isolated temporary catalog and environment for hermetic testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_catalog.sqlite")

        # Create basic catalog database with one post
        conn = sqlite3.connect(self.db_path)
        img = IrasutoyaImage("可愛い猫", "https://1.bp.blogspot.com/s800/cat.png", "t.png", "猫", post_id="blogger_99999")
        post = ParsedPost("blogger_99999", "猫のイラスト", "desc", "pub", "upd", (), [img])
        write_posts(conn, [post])
        conn.close()

        # Hermetic default: never probe a live Ollama server from unit tests
        ollama_patcher = patch(
            "scripts.synonyms.ollama_agent.check_ollama_availability",
            return_value=(True, None),
        )
        ollama_patcher.start()
        self.addCleanup(ollama_patcher.stop)



    def tearDown(self) -> None:
        """Clean up temporary test workspace."""
        self.temp_dir.cleanup()

    def test_check_ollama_availability_success_and_model_detection(self) -> None:
        """Verify check_ollama_availability detects running Ollama and checks installed models."""
        mock_payload = {
            "models": [
                {"name": "gemma4:31b"},
                {"name": "llama3:latest"},
            ]
        }
        mock_b = json.dumps(mock_payload).encode("utf-8")

        class MockResp:
            """Mock 200 HTTP response."""

            status = 200

            def __enter__(self) -> "MockResp":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self) -> bytes:
                """Return response bytes."""
                return mock_b

        with patch("urllib.request.urlopen", return_value=MockResp()):
            ok, err = check_ollama_availability("http://localhost:11434/api/chat", model="gemma4:31b")
            self.assertTrue(ok)
            self.assertIsNone(err)

    def test_check_ollama_availability_connection_refused(self) -> None:
        """Verify check_ollama_availability handles connection refused gracefully."""
        err = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        with patch("urllib.request.urlopen", side_effect=err):
            ok, msg = check_ollama_availability("http://localhost:11434/api/chat")
            self.assertFalse(ok)
            self.assertIsNotNone(msg)
            self.assertIn("Cannot connect to Ollama", str(msg))

    def test_check_ollama_availability_preserves_path_prefix(self) -> None:
        """Verify check_ollama_availability preserves path prefix when probing tags endpoint."""

        class MockResp:
            """Mock 200 HTTP response."""

            status = 200

            def __enter__(self) -> "MockResp":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self) -> bytes:
                """Return response bytes."""
                return json.dumps({"models": []}).encode("utf-8")

        captured_urls: List[str] = []

        def fake_open(req: Any, *args: Any, **kwargs: Any) -> MockResp:
            """Record request URL."""
            url = req.full_url if hasattr(req, "full_url") else str(req)
            captured_urls.append(url)
            return MockResp()

        with patch("urllib.request.urlopen", side_effect=fake_open):
            ok, err = check_ollama_availability("http://proxy:8080/ollama-service/api/chat")
            self.assertTrue(ok)
            self.assertIn("http://proxy:8080/ollama-service/api/tags", captured_urls)

    def test_check_ollama_availability_root_requires_ollama_body(self) -> None:
        """Verify root fallback probe passes only when body contains 'Ollama is running'."""

        class MockResp:
            """Mock HTTP response."""

            def __init__(self, body: bytes) -> None:
                """Initialize with response body."""
                self._body = body
                self.status = 200

            def __enter__(self) -> "MockResp":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self) -> bytes:
                """Return response bytes."""
                return self._body

        def make_handler(root_body: bytes) -> Callable[..., Any]:
            """Create urlopen handler that fails /api/tags and returns root_body on root."""

            def _open(req: Any, *args: Any, **kwargs: Any) -> MockResp:
                """Open request."""
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if "/api/tags" in url:
                    raise urllib.error.URLError("Not found")
                return MockResp(root_body)

            return _open

        # Generic landing page -> False
        with patch("urllib.request.urlopen", side_effect=make_handler(b"<html>Nginx</html>")):
            ok, err = check_ollama_availability("http://localhost:11434/api/chat")
            self.assertFalse(ok)

        # Genuine Ollama root response -> True
        with patch("urllib.request.urlopen", side_effect=make_handler(b"Ollama is running")):
            ok, err = check_ollama_availability("http://localhost:11434/api/chat")
            self.assertTrue(ok)

    def test_search_catalog_for_agent(self) -> None:
        """Verify search_catalog_for_agent returns matching posts, suggested_target, and runtime top-1."""
        res = search_catalog_for_agent(self.db_path, "猫")
        self.assertEqual(res["query"], "猫")
        self.assertTrue(len(res["posts"]) >= 1)
        post = res["posts"][0]
        self.assertEqual(post["title"], "猫のイラスト")
        self.assertEqual(post["suggested_target"], "猫")

        top1 = res["runtime_top1_if_queried"]
        self.assertIsNotNone(top1)
        assert top1 is not None
        self.assertEqual(top1["matched_image_title"], "可愛い猫")
        self.assertEqual(top1["post_title"], "猫のイラスト")

    def test_search_catalog_for_agent_escapes_like_wildcards(self) -> None:
        """Verify search_catalog_for_agent treats %, _, and backslashes as literal search characters."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO posts (id, blogger_id, title, description, published, updated, image_count) "
            "VALUES (?, ?, ?, ?, '', '', 1)",
            (999, "999", "100%のイラスト", "literal percentage description"),
        )
        cur.execute(
            "INSERT INTO posts (id, blogger_id, title, description, published, updated, image_count) "
            "VALUES (?, ?, ?, ?, '', '', 1)",
            (998, "998", "100人のイラスト", "one hundred people description"),
        )
        conn.commit()
        conn.close()

        res = search_catalog_for_agent(self.db_path, query="100%", limit=5)
        matched_titles = [p["title"] for p in res["posts"]]
        self.assertIn("100%のイラスト", matched_titles)
        self.assertNotIn("100人のイラスト", matched_titles)

    def test_search_catalog_for_agent_empty_or_punctuation_query(self) -> None:
        """Verify search_catalog_for_agent returns empty results for empty/punctuation queries."""
        for empty_q in ("", "   ", "...", "！？"):
            res = search_catalog_for_agent(self.db_path, empty_q)
            self.assertEqual(res["query"], empty_q)
            self.assertEqual(res["posts"], [])
            self.assertIsNone(res["runtime_top1_if_queried"])

    def test_contains_english_word_and_target_filtering(self) -> None:
        """Verify English words are detected while valid Japanese Latin terms are preserved."""
        self.assertTrue(_contains_english_word("cat"))
        self.assertTrue(_contains_english_word("car"))
        self.assertTrue(_contains_english_word("smart car"))
        self.assertTrue(_contains_english_word("猫cat"))
        self.assertTrue(_contains_english_word("Dog"))

        self.assertFalse(_contains_english_word("猫"))
        self.assertFalse(_contains_english_word("可愛い猫"))
        self.assertFalse(_contains_english_word("Tシャツ"))
        self.assertFalse(_contains_english_word("A型"))
        self.assertFalse(_contains_english_word("Wi-Fi"))
        self.assertFalse(_contains_english_word("QRコード"))
        self.assertFalse(_contains_english_word("PC"))
        self.assertFalse(_contains_english_word("DVD"))

    def test_run_ollama_agent_batch_and_compile(self) -> None:
        """Verify run_ollama_agent_batch tool loop and compile writing exclusively to llm_out."""
        candidates = [
            {
                "term": "子猫",
                "reading": "こねこ",
                "frequency": 500,
                "english_glosses": ["kitten"],
                "pos_tags": ["n"],
            }
        ]

        turn_count = 0

        def fake_chat_urlopen(req: Any, *args: Any, **kwargs: Any) -> Any:
            """Simulate Ollama /api/chat tool call turn followed by final answer."""
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                resp_payload = {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_catalog",
                                    "arguments": {"queries": ["猫"]},
                                }
                            }
                        ],
                    }
                }
            else:
                resp_payload = {
                    "message": {
                        "role": "assistant",
                        "content": '{"visualizable": true, "reason": "young feline", "targets": ["猫"]}',
                    }
                }

            mock_b = json.dumps(resp_payload).encode("utf-8")

            class MockChatResponse:
                """Mock chat response."""

                def __enter__(self) -> "MockChatResponse":
                    """Enter context manager."""
                    return self

                def __exit__(self, *args: Any) -> None:
                    """Exit context manager."""
                    pass

                def read(self) -> bytes:
                    """Read response body bytes."""
                    return mock_b

            return MockChatResponse()

        with patch("urllib.request.urlopen", side_effect=fake_chat_urlopen):
            mappings, failed = run_ollama_agent_batch(candidates, catalog_path=self.db_path)
            self.assertEqual(mappings, {"子猫": ["猫"]})
            self.assertEqual(failed, 0)

        llm_out = os.path.join(self.temp_dir.name, "custom_llm_out.json")
        tsv_path = os.path.join(self.temp_dir.name, "compile_jpdb.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("子猫\tこねこ\t500\t\n")

        def fake_load_candidates(*args: Any, **kwargs: Any) -> Tuple[List[Dict[str, Any]], int]:
            """Return static test candidates."""
            return candidates, 0

        turn_count = 0
        with patch("scripts.synonyms.ollama_agent.load_jpdb_candidates", side_effect=fake_load_candidates):
            with patch("urllib.request.urlopen", side_effect=fake_chat_urlopen):
                ok = run_ollama_compile(
                    catalog_path=self.db_path,
                    jpdb_path=tsv_path,
                    llm_out_path=llm_out,
                )
                self.assertTrue(ok)
                self.assertTrue(os.path.isfile(llm_out))
                with open(llm_out, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.assertEqual(saved.get("concepts", {}).get("子猫"), ["猫"])

        # Verify run_ollama_compile skips already compiled candidates
        with patch("scripts.synonyms.ollama_agent.load_jpdb_candidates", side_effect=fake_load_candidates):
            with patch("scripts.synonyms.ollama_agent.run_ollama_agent_batch") as mock_batch:
                ok_skip = run_ollama_compile(
                    catalog_path=self.db_path,
                    jpdb_path=tsv_path,
                    llm_out_path=llm_out,
                )
                self.assertTrue(ok_skip)
                mock_batch.assert_not_called()

        # Verify run_ollama_compile returns False when 0 candidates were compiled or rejected
        with patch("scripts.synonyms.ollama_agent.load_jpdb_candidates", side_effect=fake_load_candidates):
            with patch("scripts.synonyms.ollama_agent.run_ollama_agent_batch", return_value=({}, 0)):
                empty_llm_out = os.path.join(self.temp_dir.name, "empty_llm_out.json")
                ok_empty = run_ollama_compile(
                    catalog_path=self.db_path,
                    jpdb_path=tsv_path,
                    llm_out_path=empty_llm_out,
                )
                self.assertFalse(ok_empty)
                self.assertFalse(os.path.isfile(empty_llm_out))

        # Verify run_ollama_compile persists rejected terms and returns True
        abstract_candidates = [
            {
                "term": "概念",
                "reading": "がいねん",
                "frequency": 2500,
                "english_glosses": ["concept"],
                "pos_tags": ["n"],
            }
        ]

        def fake_load_abstract(*args: Any, **kwargs: Any) -> Tuple[List[Dict[str, Any]], int]:
            """Return abstract candidate."""
            return abstract_candidates, 0

        def fake_batch_reject(
            candidates: Any,
            *args: Any,
            on_reject: Optional[Callable[[str, str], None]] = None,
            **kwargs: Any,
        ) -> Tuple[Dict[str, List[str]], int]:
            """Simulate model rejecting abstract concept."""
            if on_reject:
                on_reject("概念", "purely abstract concept")
            return {}, 0

        rej_llm_out = os.path.join(self.temp_dir.name, "rej_llm_out.json")
        with patch("scripts.synonyms.ollama_agent.load_jpdb_candidates", side_effect=fake_load_abstract):
            with patch("scripts.synonyms.ollama_agent.run_ollama_agent_batch", side_effect=fake_batch_reject):
                ok_rej = run_ollama_compile(
                    catalog_path=self.db_path,
                    jpdb_path=tsv_path,
                    llm_out_path=rej_llm_out,
                )
                self.assertTrue(ok_rej)
                self.assertTrue(os.path.isfile(rej_llm_out))
                with open(rej_llm_out, "r", encoding="utf-8") as f:
                    saved_rej = json.load(f)
                self.assertEqual(saved_rej.get("rejected", {}).get("概念"), "purely abstract concept")

        # Verify run_ollama_compile aborts when existing llm_out is corrupt or invalid
        corrupt_llm_out = os.path.join(self.temp_dir.name, "corrupt_llm_out.json")
        with open(corrupt_llm_out, "w", encoding="utf-8") as f:
            f.write("{\ninvalid json syntax")
        with patch("scripts.synonyms.ollama_agent.load_jpdb_candidates", side_effect=fake_load_candidates):
            ok_corrupt = run_ollama_compile(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                llm_out_path=corrupt_llm_out,
            )
            self.assertFalse(ok_corrupt)

        # Verify run_ollama_compile returns False when catalog index is unavailable
        with patch.object(CatalogStore, "get_index", return_value=None):
            ok_no_idx = run_ollama_compile(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                llm_out_path=llm_out,
            )
            self.assertFalse(ok_no_idx)

    def test_run_ollama_agent_batch_unparsable_response_does_not_reject(self) -> None:
        """Verify unparsable LLM response logs warning and does not invoke on_reject."""
        cands = [
            {
                "term": "テスト語",
                "reading": "てすとご",
                "frequency": 1,
                "english_glosses": ["test term"],
                "pos_tags": ["n"],
            }
        ]

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "message": {
                "role": "assistant",
                "content": "This is plain prose without JSON object formatting.",
            }
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        rejections: List[Tuple[str, str]] = []

        def handle_reject(term: str, reason: str) -> None:
            """Track on_reject invocations."""
            rejections.append((term, reason))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res, failed = run_ollama_agent_batch(
                cands,
                catalog_path=self.db_path,
                on_reject=handle_reject,
            )
            self.assertEqual(res, {})
            self.assertEqual(failed, 0)
            self.assertEqual(rejections, [])

    def test_run_ollama_agent_batch_rejects_latin_targets(self) -> None:
        """Verify run_ollama_agent_batch ignores English/Latin targets returned by model."""
        cands = [
            {
                "term": "子猫",
                "reading": "こねこ",
                "frequency": 1,
                "english_glosses": ["kitten"],
                "pos_tags": ["n"],
            }
        ]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "message": {
                "role": "assistant",
                "content": json.dumps({"visualizable": True, "targets": ["cat", "kitten"]}),
            }
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        rejections: List[Tuple[str, str]] = []

        def handle_reject(term: str, reason: str) -> None:
            """Track on_reject calls."""
            rejections.append((term, reason))

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res, failed = run_ollama_agent_batch(
                cands,
                catalog_path=self.db_path,
                on_reject=handle_reject,
            )
            self.assertEqual(res, {})
            self.assertEqual(failed, 0)
            self.assertEqual(len(rejections), 1)
            self.assertIn("unverified in catalog: proposed targets ['cat', 'kitten']", rejections[0][1])

    def test_run_ollama_agent_batch_turn_exhaustion(self) -> None:
        """Verify candidates that loop tool calls until turn exhaustion are left unmapped without failure."""
        cands = [
            {
                "term": "ループ単語",
                "reading": "るーぷ",
                "frequency": 1,
            }
        ]

        def fake_infinite_tool_call(req: Any, *args: Any, **kwargs: Any) -> Any:
            """Always return tool call without final answer."""
            resp_payload = {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_catalog",
                                "arguments": {"queries": ["猫"]},
                            }
                        }
                    ],
                }
            }
            mock_b = json.dumps(resp_payload).encode("utf-8")

            class MockResp:
                """Mock HTTP response."""

                def __enter__(self) -> "MockResp":
                    """Enter context manager."""
                    return self

                def __exit__(self, *args: Any) -> None:
                    """Exit context manager."""
                    pass

                def read(self) -> bytes:
                    """Read body bytes."""
                    return mock_b

            return MockResp()

        with patch("urllib.request.urlopen", side_effect=fake_infinite_tool_call):
            res, failed = run_ollama_agent_batch(
                cands,
                catalog_path=self.db_path,
                max_turns=2,
            )
            self.assertEqual(res, {})
            self.assertEqual(failed, 0)

    def test_run_ollama_agent_batch_handles_non_list_targets(self) -> None:
        """Verify run_ollama_agent_batch tolerates non-list targets without raising TypeError."""
        cands = [
            {
                "term": "テスト語",
                "reading": "てすとご",
                "frequency": 1,
            }
        ]

        def fake_non_list_targets_resp(req: Any, *args: Any, **kwargs: Any) -> Any:
            """Return response with visualizable=True but targets=null."""
            resp_payload = {
                "message": {
                    "role": "assistant",
                    "content": '{"visualizable": true, "targets": null, "reason": "malformed"}',
                }
            }
            mock_b = json.dumps(resp_payload).encode("utf-8")

            class MockResp:
                """Mock HTTP response."""

                def __enter__(self) -> "MockResp":
                    """Enter context manager."""
                    return self

                def __exit__(self, *args: Any) -> None:
                    """Exit context manager."""
                    pass

                def read(self) -> bytes:
                    """Read body bytes."""
                    return mock_b

            return MockResp()

        rejections: List[Tuple[str, str]] = []

        def handle_reject(term: str, reason: str) -> None:
            """Track rejection callbacks."""
            rejections.append((term, reason))

        with patch("urllib.request.urlopen", side_effect=fake_non_list_targets_resp):
            res, failed = run_ollama_agent_batch(
                cands,
                catalog_path=self.db_path,
                on_reject=handle_reject,
            )
            self.assertEqual(res, {})
            self.assertEqual(failed, 0)
            self.assertEqual(len(rejections), 0)

    def test_run_ollama_compile_persists_stale_on_early_exit(self) -> None:
        """Verify run_ollama_compile persists revalidated stale concepts even when remaining candidates is empty."""
        llm_out = os.path.join(self.temp_dir.name, "early_exit_stale.json")
        with open(llm_out, "w", encoding="utf-8") as f:
            json.dump({
                "concepts": {
                    "猫": ["存在しない架空のイラスト123"],
                },
                "rejected": {},
            }, f)

        tsv_path = os.path.join(self.temp_dir.name, "empty_candidates.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("")

        ok = run_ollama_compile(
            catalog_path=self.db_path,
            llm_out_path=llm_out,
            jpdb_path=tsv_path,
            cache_dir=self.temp_dir.name,
        )
        self.assertTrue(ok)
        with open(llm_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertNotIn("猫", data.get("concepts", {}))
        self.assertEqual(data.get("stale", {}).get("猫"), ["存在しない架空のイラスト123"])

    def test_run_ollama_agent_batch_strict_boolean_visualizable(self) -> None:
        """Verify visualizable string 'false' is not treated as boolean true."""
        cands = [{"term": "偽物", "reading": "にせもの", "frequency": 1}]

        def fake_string_false_resp(req: Any, *args: Any, **kwargs: Any) -> Any:
            """Return string 'false' for visualizable."""
            resp_payload = {
                "message": {
                    "role": "assistant",
                    "content": '{"visualizable": "false", "reason": "not boolean true"}',
                }
            }
            mock_b = json.dumps(resp_payload).encode("utf-8")

            class MockResp:
                """Mock HTTP response."""

                def __enter__(self) -> "MockResp":
                    """Enter context manager."""
                    return self

                def __exit__(self, *args: Any) -> None:
                    """Exit context manager."""
                    pass

                def read(self) -> bytes:
                    """Read body bytes."""
                    return mock_b

            return MockResp()

        rejections: List[Tuple[str, str]] = []

        def handle_reject(term: str, reason: str) -> None:
            """Track rejection callbacks."""
            rejections.append((term, reason))

        with patch("urllib.request.urlopen", side_effect=fake_string_false_resp):
            res, failed = run_ollama_agent_batch(
                cands,
                catalog_path=self.db_path,
                on_reject=handle_reject,
            )
            self.assertEqual(res, {})
            self.assertEqual(failed, 0)
            self.assertEqual(len(rejections), 1)
            self.assertIn("not boolean true", rejections[0][1])

    def test_run_ollama_compile_terms_re_evaluates_prior_decisions(self) -> None:
        """Verify explicit terms argument forces re-evaluation even if term was previously rejected or mapped."""
        llm_out = os.path.join(self.temp_dir.name, "llm_re_eval.json")
        initial_data = {
            "meta": {"source": "test"},
            "concepts": {"猫": ["猫"]},
            "rejected": {"犬": "abstract concept"},
        }
        with open(llm_out, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, ensure_ascii=False)

        tsv_path = os.path.join(self.temp_dir.name, "jpdb_terms_re_eval.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("犬\tいぬ\t1\n")

        with (
            patch("scripts.synonyms.ollama_agent._fetch_yomitan_term_entries", return_value=([], True)),
            patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", return_value=([], True)),
        ):
            with patch(
                "scripts.synonyms.ollama_agent.run_ollama_agent_batch",
                return_value=({"犬": ["猫"]}, 0),
            ) as mock_batch:
                # Without explicit terms, 犬 is skipped as already rejected
                ok = run_ollama_compile(
                    catalog_path=self.db_path,
                    llm_out_path=llm_out,
                    jpdb_path=tsv_path,
                    mode="all",
                    cache_dir=self.temp_dir.name,
                )
                self.assertTrue(ok)
                mock_batch.assert_not_called()

                # With explicit terms=["犬"], 犬 is re-evaluated despite being in rejected
                def fake_batch_match(
                    *args: Any,
                    **kwargs: Any,
                ) -> Tuple[Dict[str, List[str]], int]:
                    """Simulate match callback during re-evaluation."""
                    on_match = kwargs.get("on_match")
                    if on_match:
                        on_match("犬", ["猫"])
                    return {"犬": ["猫"]}, 0

                mock_batch.side_effect = fake_batch_match
                ok = run_ollama_compile(
                    catalog_path=self.db_path,
                    llm_out_path=llm_out,
                    jpdb_path=tsv_path,
                    terms=["犬"],
                    mode="all",
                    cache_dir=self.temp_dir.name,
                )
                self.assertTrue(ok)
                mock_batch.assert_called_once()
                cands_arg = mock_batch.call_args[1].get("candidates")
                self.assertEqual(len(cands_arg), 1)
                self.assertEqual(cands_arg[0]["term"], "犬")

        # Verify disk state was updated: '犬' moved from rejected to concepts
        with open(llm_out, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertIn("犬", saved.get("concepts", {}))
        self.assertNotIn("犬", saved.get("rejected", {}))

        # Test case: explicit re-evaluation that ends in a rejection updates reason and succeeds
        with (
            patch("scripts.synonyms.ollama_agent._fetch_yomitan_term_entries", return_value=([], True)),
            patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", return_value=([], True)),
        ):
            with patch(
                "scripts.synonyms.ollama_agent.run_ollama_agent_batch",
            ) as mock_batch_rej:
                def fake_batch_rej(
                    *args: Any,
                    **kwargs: Any,
                ) -> Tuple[Dict[str, List[str]], int]:
                    """Simulate rejection callback during explicit re-evaluation."""
                    on_reject = kwargs.get("on_reject")
                    if on_reject:
                        on_reject("犬", "still abstract after re-check")
                    return {}, 0

                mock_batch_rej.side_effect = fake_batch_rej
                ok_rej = run_ollama_compile(
                    catalog_path=self.db_path,
                    llm_out_path=llm_out,
                    jpdb_path=tsv_path,
                    terms=["犬"],
                    mode="all",
                    cache_dir=self.temp_dir.name,
                )
                self.assertTrue(ok_rej)

        with open(llm_out, "r", encoding="utf-8") as f:
            saved_rej = json.load(f)
        self.assertNotIn("犬", saved_rej.get("concepts", {}))
        self.assertEqual(saved_rej.get("rejected", {}).get("犬"), "still abstract after re-check")

        # Test case: explicit terms aborts on Yomitan outage
        with (
            patch("scripts.synonyms.ollama_agent._fetch_yomitan_term_entries", return_value=([], False)),
            patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", return_value=([], False)),
        ):
            ok_outage = run_ollama_compile(
                catalog_path=self.db_path,
                llm_out_path=llm_out,
                terms=["犬"],
                cache_dir=self.temp_dir.name,
            )
            self.assertFalse(ok_outage)

        # Test case: explicit terms extracts real reading from Yomitan headwords
        mock_entries = [
            {
                "headwords": [{"term": "犬", "reading": "いぬ"}],
                "definitions": [{"tags": [{"name": "n"}]}],
            }
        ]
        with (
            patch("scripts.synonyms.ollama_agent._fetch_yomitan_term_entries", return_value=(mock_entries, True)),
            patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", return_value=(mock_entries, True)),
            patch("scripts.synonyms.ollama_agent.run_ollama_agent_batch", return_value=({"犬": ["猫"]}, 0)) as mock_b,
        ):
                ok_read = run_ollama_compile(
                    catalog_path=self.db_path,
                    llm_out_path=llm_out,
                    terms=["犬"],
                    cache_dir=self.temp_dir.name,
                )
                self.assertTrue(ok_read)
                cands = mock_b.call_args[1].get("candidates")
                self.assertEqual(cands[0]["reading"], "いぬ")

    def test_run_ollama_compile_aborts_when_ollama_unavailable(self) -> None:
        """Verify run_ollama_compile aborts immediately if Ollama service is unreachable."""
        with patch("scripts.synonyms.ollama_agent.check_ollama_availability", return_value=(False, "Connection refused")):
            ok = run_ollama_compile(
                catalog_path=self.db_path,
                llm_out_path=os.path.join(self.temp_dir.name, "out.json"),
                terms=["犬"],
            )
            self.assertFalse(ok)

    def test_run_ollama_compile_aborts_on_deferred_candidates(self) -> None:
        """Verify run_ollama_compile returns False when candidates are deferred due to Yomitan outage."""
        llm_out = os.path.join(self.temp_dir.name, "outage_llm_out.json")
        tsv_path = os.path.join(self.temp_dir.name, "outage_compile.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("猫\tねこ\t1\n")

        with patch("scripts.synonyms.ollama_agent.load_jpdb_candidates", return_value=([], 1)):
            ok = run_ollama_compile(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                llm_out_path=llm_out,
            )
            self.assertFalse(ok)

    def test_run_ollama_compile_revalidates_saved_concepts(self) -> None:
        """Verify run_ollama_compile re-evaluates terms whose saved targets are no longer in catalog."""
        llm_out = os.path.join(self.temp_dir.name, "revalidate_concepts.json")
        with open(llm_out, "w", encoding="utf-8") as f:
            json.dump({
                "concepts": {
                    "猫": ["存在しない架空のイラスト123"],
                    "未評価語": ["未評価の架空ターゲット999"],
                },
                "rejected": {},
            }, f)

        tsv_path = os.path.join(self.temp_dir.name, "reval.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("term\treading\tfrequency\n猫\tねこ\t1\n")

        with (
            patch("scripts.synonyms.ollama_agent._fetch_yomitan_term_entries", return_value=([], True)),
            patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", return_value=([], True)),
        ):
            with patch("scripts.synonyms.ollama_agent.run_ollama_agent_batch") as mock_batch:
                def fake_batch_reval(*args: Any, **kwargs: Any) -> Tuple[Dict[str, List[str]], int]:
                    """Simulate match callback during revalidation."""
                    on_match = kwargs.get("on_match")
                    if on_match:
                        on_match("猫", ["猫"])
                    return {"猫": ["猫"]}, 0

                mock_batch.side_effect = fake_batch_reval
                ok = run_ollama_compile(
                    catalog_path=self.db_path,
                    llm_out_path=llm_out,
                    jpdb_path=tsv_path,
                    limit=1,
                    mode="all",
                    cache_dir=self.temp_dir.name,
                )
                self.assertTrue(ok)
                mock_batch.assert_called_once()
                cands_arg = mock_batch.call_args[1].get("candidates")
                self.assertEqual(len(cands_arg), 1)
                self.assertEqual(cands_arg[0]["term"], "猫")

        with open(llm_out, "r", encoding="utf-8") as f:
            saved_reval = json.load(f)
        self.assertEqual(saved_reval.get("concepts", {}).get("猫"), ["猫"])
        self.assertNotIn("未評価語", saved_reval.get("concepts", {}))
        self.assertEqual(saved_reval.get("stale", {}).get("未評価語"), ["未評価の架空ターゲット999"])

    def test_run_ollama_compile_deduplicates_candidate_terms(self) -> None:
        """Verify run_ollama_compile evaluates duplicate candidates only once."""
        llm_out = os.path.join(self.temp_dir.name, "dedup_llm_out.json")
        tsv_path = os.path.join(self.temp_dir.name, "dedup.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("何\tなに\t1\n何\tなん\t2\n")

        with (
            patch("scripts.synonyms.ollama_agent._fetch_yomitan_term_entries", return_value=([], True)),
            patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", return_value=([], True)),
        ):
            with patch("scripts.synonyms.ollama_agent.run_ollama_agent_batch", return_value=({"何": ["猫"]}, 0)) as mock_batch:
                ok = run_ollama_compile(
                    catalog_path=self.db_path,
                    llm_out_path=llm_out,
                    jpdb_path=tsv_path,
                    limit=2,
                    mode="all",
                    cache_dir=self.temp_dir.name,
                )
                self.assertTrue(ok)
                mock_batch.assert_called_once()
                cands_arg = mock_batch.call_args[1].get("candidates")
                self.assertEqual(len(cands_arg), 1)
                self.assertEqual(cands_arg[0]["term"], "何")

    def test_run_ollama_agent_batch_consecutive_connection_failure_cutoff(self) -> None:
        """Verify run_ollama_agent_batch aborts after consecutive connection failures."""
        cands = [
            {"term": f"単語_{i}", "reading": f"たんご_{i}", "frequency": i}
            for i in range(1, 10)
        ]
        err = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        with patch("urllib.request.urlopen", side_effect=err):
            res, failed = run_ollama_agent_batch(
                cands,
                catalog_path=self.db_path,
            )
            self.assertEqual(res, {})
            self.assertEqual(failed, 9)

    def test_run_ollama_compile_aborts_when_empty_candidates_and_missing_file(self) -> None:
        """Verify run_ollama_compile returns False when candidates are empty and artifact does not exist."""
        empty_tsv = os.path.join(self.temp_dir.name, "no_candidates.tsv")
        with open(empty_tsv, "w", encoding="utf-8") as f:
            f.write("")

        non_existent_out = os.path.join(self.temp_dir.name, "does_not_exist.json")
        ok = run_ollama_compile(
            catalog_path=self.db_path,
            llm_out_path=non_existent_out,
            jpdb_path=empty_tsv,
            cache_dir=self.temp_dir.name,
        )
        self.assertFalse(ok)

    def test_run_ollama_compile_fails_when_early_exit_flush_raises(self) -> None:
        """Verify run_ollama_compile returns False when early-exit concept flush raises an error."""
        llm_out = os.path.join(self.temp_dir.name, "early_exit_fail.json")
        with open(llm_out, "w", encoding="utf-8") as f:
            json.dump({
                "concepts": {"猫": ["存在しない架空のイラスト123"]},
                "rejected": {},
            }, f)

        empty_tsv = os.path.join(self.temp_dir.name, "empty.tsv")
        with open(empty_tsv, "w", encoding="utf-8") as f:
            f.write("")

        with patch("scripts.synonyms.ollama_agent._flush_llm_concepts", side_effect=OSError("Disk full")):
            ok = run_ollama_compile(
                catalog_path=self.db_path,
                llm_out_path=llm_out,
                jpdb_path=empty_tsv,
                cache_dir=self.temp_dir.name,
            )
            self.assertFalse(ok)

    def test_run_ollama_compile_terms_handles_null_headwords(self) -> None:
        """Verify run_ollama_compile handles Yomitan entries where headwords is null without raising TypeError."""
        llm_out = os.path.join(self.temp_dir.name, "null_hw_out.json")
        mock_entries = [{"headwords": None, "definitions": []}]

        with (
            patch("scripts.synonyms.ollama_agent._fetch_yomitan_term_entries", return_value=(mock_entries, "ok")),
            patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", return_value=(mock_entries, "ok")),
            patch("scripts.synonyms.ollama_agent.check_ollama_availability", return_value=(True, None)),
            patch("scripts.synonyms.ollama_agent.run_ollama_agent_batch", return_value=({"猫": ["猫"]}, 0)),
        ):
            ok = run_ollama_compile(
                catalog_path=self.db_path,
                llm_out_path=llm_out,
                terms=["猫"],
                cache_dir=self.temp_dir.name,
            )
            self.assertTrue(ok)

    def test_main_cli_agent_synonyms(self) -> None:
        """Verify agent_synonyms main CLI entrypoint correctly parses arguments and dispatches compiler."""
        llm_out = os.path.join(self.temp_dir.name, "cli_agent_out.json")
        with patch("scripts.agent_synonyms.run_ollama_compile", return_value=True) as mock_compile:
            code = main([
                "--catalog",
                self.db_path,
                "--out",
                llm_out,
                "--jpdb-limit",
                "150",
                "--jpdb-mode",
                "unmatched",
                "--terms",
                "犬,猫",
            ])
            self.assertEqual(code, 0)
            mock_compile.assert_called_once()
            _, kwargs = mock_compile.call_args
            self.assertEqual(kwargs.get("catalog_path"), self.db_path)
            self.assertEqual(kwargs.get("llm_out_path"), llm_out)
            self.assertEqual(kwargs.get("limit"), 150)
            self.assertEqual(kwargs.get("mode"), "unmatched")
            self.assertEqual(kwargs.get("terms"), "犬,猫")

        # Verify failure return code propagation
        with patch("scripts.agent_synonyms.run_ollama_compile", return_value=False):
            code_fail = main(["--catalog", self.db_path])
            self.assertEqual(code_fail, 1)

        # Verify --help formatting
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with self.assertRaises(SystemExit) as cm:
                main(["--help"])
            self.assertEqual(cm.exception.code, 0)
        output = buf.getvalue()
        self.assertIn("visual concept compiler", output)
        self.assertIn("Irasutoya", output)
        self.assertIn("--jpdb-limit", output)


if __name__ == "__main__":
    unittest.main()
