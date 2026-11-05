"""Unit tests for the local LLM visual concept compiler (scripts/agent_synonyms.py)."""
import array
import io
import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
import urllib.error
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
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
from scripts.pull_models import (
    DEFAULT_MODELS_PATH,
    get_installed_ollama_models,
    load_models_manifest,
    pull_declared_models,
    pull_single_model,
)
from scripts.pull_models import (
    main as pull_models_main,
)
from scripts.synonyms.ollama_agent import (
    _flush_llm_concepts,
    _KeepAliveJSONClient,
    _parse_llm_json_response,
    get_model_options_from_manifest,
)
from scripts.synonyms.ollama_embed import (
    QWEN3_SEARCH_INSTRUCT,
    CatalogVectorIndex,
    OllamaEmbedClient,
    clean_post_for_embedding,
    compute_catalog_posts_sha256,
    format_candidate_context_block,
    format_query_for_embedding,
    resolve_valid_post_target,
    search_hybrid_presearch,
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
            f.write("term\treading\tfreq\tkana_freq\n" + "test content\treading\t1\t1\n" * 100)

        with patch("urllib.request.urlopen") as mock_url:
            path = fetch_jpdb_freq_github(cache_dir=self.temp_dir.name, url=custom_url)
            mock_url.assert_not_called()
        self.assertEqual(os.path.basename(path), "custom_vocab_list.csv")

    def test_fetch_jpdb_freq_github_validates_cache_header(self) -> None:
        """Verify fetch_jpdb_freq_github rejects corrupted cache and re-downloads."""
        cache_dir = os.path.join(self.temp_dir.name, "cache_header_val")
        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(cache_dir, "jpdb_v2.2_freq_list_2024-10-13.csv")

        # 1. Corrupted cached file with HTML error page (> 1000 bytes)
        with open(cached_file, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html><body>Error 404</body></html>\n" * 50)

        mock_csv_bytes = (b"term\treading\tfreq\tkana_freq\n" + b"\xe7\x8c\xab\t\xe3\x81\xad\xe3\x81\x93\t1\t1\n" * 100)

        class MockResp:
            """Mock HTTP response."""

            def __enter__(self) -> "MockResp":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self, n: int = -1) -> bytes:
                """Return response bytes."""
                nonlocal mock_csv_bytes
                chunk = mock_csv_bytes[:n] if n > 0 else mock_csv_bytes
                mock_csv_bytes = mock_csv_bytes[n:] if n > 0 else b""
                return chunk

        with patch("urllib.request.urlopen", return_value=MockResp()) as mock_url:
            path = fetch_jpdb_freq_github(cache_dir=cache_dir)
            mock_url.assert_called_once()
        self.assertEqual(path, os.path.abspath(cached_file))

        # 2. Re-running with now-valid cached header reuses file without re-downloading
        with patch("urllib.request.urlopen") as mock_url_again:
            path_cached = fetch_jpdb_freq_github(cache_dir=cache_dir)
            mock_url_again.assert_not_called()
        self.assertEqual(path_cached, os.path.abspath(cached_file))

    def test_fetch_jpdb_freq_github_accepts_headerless_data_row(self) -> None:
        """Verify fetch_jpdb_freq_github reuses cache for valid headerless frequency files."""
        cache_dir = os.path.join(self.temp_dir.name, "cache_headerless")
        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(cache_dir, "jpdb_v2.2_freq_list_2024-10-13.csv")
        # Headerless row with term, reading, rank (numeric)
        with open(cached_file, "w", encoding="utf-8") as f:
            f.write("猫\tねこ\t1\t1\n" * 100)

        with patch("urllib.request.urlopen") as mock_url:
            path = fetch_jpdb_freq_github(cache_dir=cache_dir)
            mock_url.assert_not_called()
        self.assertEqual(path, os.path.abspath(cached_file))

    def test_fetch_jpdb_freq_github_accepts_quoted_data_row(self) -> None:
        """Verify fetch_jpdb_freq_github reuses cache for valid quoted frequency files."""
        cache_dir = os.path.join(self.temp_dir.name, "cache_quoted")
        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(cache_dir, "jpdb_v2.2_freq_list_2024-10-13.csv")
        # Quoted row with "term", "reading", "rank"
        with open(cached_file, "w", encoding="utf-8") as f:
            f.write('"猫"\t"ねこ"\t"1"\t"1"\n' * 100)

        with patch("urllib.request.urlopen") as mock_url:
            path = fetch_jpdb_freq_github(cache_dir=cache_dir)
            mock_url.assert_not_called()
        self.assertEqual(path, os.path.abspath(cached_file))

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

    def test_parse_yomitan_metadata_homograph_reading_consistency(self) -> None:
        """Verify _parse_yomitan_metadata does not match alternate reading homographs."""
        # 1. 一日 (tsuitachi vs ichinichi)
        tsuitachi_entry = {
            "headwords": [{"term": "一日", "reading": "ついたち"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [{"name": "n", "category": "partOfSpeech"}],
                    "entries": ["first day of the month"],
                }
            ],
        }
        ichinichi_entry = {
            "headwords": [{"term": "一日", "reading": "いちにち"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [{"name": "n", "category": "partOfSpeech"}],
                    "entries": ["one day", "all day long"],
                }
            ],
        }

        # Query for 'いちにち' when both entries are present -> picks ichinichi_entry
        res1 = _parse_yomitan_metadata(
            [tsuitachi_entry, ichinichi_entry],
            term="一日",
            reading="いちにち",
        )
        self.assertIsNotNone(res1)
        assert res1 is not None
        self.assertEqual(res1.english_glosses, ["one day", "all day long"])

        # Query for 'いちにち' when ONLY tsuitachi_entry is present -> must NOT match tsuitachi
        res2 = _parse_yomitan_metadata(
            [tsuitachi_entry],
            term="一日",
            reading="いちにち",
        )
        self.assertIsNone(res2)

        # 2. 方 (kata vs hou)
        kata_entry = {
            "headwords": [{"term": "方", "reading": "かた"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [{"name": "n", "category": "partOfSpeech"}],
                    "entries": ["person", "way of doing"],
                }
            ],
        }
        hou_entry = {
            "headwords": [{"term": "方", "reading": "ほう"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [{"name": "n", "category": "partOfSpeech"}],
                    "entries": ["direction", "side"],
                }
            ],
        }
        res_hou = _parse_yomitan_metadata([kata_entry, hou_entry], term="方", reading="ほう")
        self.assertIsNotNone(res_hou)
        assert res_hou is not None
        self.assertEqual(res_hou.english_glosses, ["direction", "side"])

        # Query for 'ほう' when ONLY kata_entry is present -> must NOT match
        self.assertIsNone(_parse_yomitan_metadata([kata_entry], term="方", reading="ほう"))

        # 3. 角 (kado vs tsuno)
        kado_entry = {
            "headwords": [{"term": "角", "reading": "かど"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [{"name": "n", "category": "partOfSpeech"}],
                    "entries": ["corner", "edge"],
                }
            ],
        }
        tsuno_entry = {
            "headwords": [{"term": "角", "reading": "つの"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [{"name": "n", "category": "partOfSpeech"}],
                    "entries": ["horn", "antler"],
                }
            ],
        }
        res_tsuno = _parse_yomitan_metadata([kado_entry, tsuno_entry], term="角", reading="つの")
        self.assertIsNotNone(res_tsuno)
        assert res_tsuno is not None
        self.assertEqual(res_tsuno.english_glosses, ["horn", "antler"])

        # Query for 'つの' when ONLY kado_entry is present -> must NOT match
        self.assertIsNone(_parse_yomitan_metadata([kado_entry], term="角", reading="つの"))

        # 4. Term matches but headword has no reading -> must NOT match when reading is requested
        no_reading_entry = {
            "headwords": [{"term": "角"}],
            "definitions": [
                {
                    "dictionary": "JMdict (English)",
                    "tags": [{"name": "n", "category": "partOfSpeech"}],
                    "entries": ["angle"],
                }
            ],
        }
        self.assertIsNone(_parse_yomitan_metadata([no_reading_entry], term="角", reading="つの"))
        # But matches when NO reading is requested
        self.assertIsNotNone(_parse_yomitan_metadata([no_reading_entry], term="角", reading=None))

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
            readings = {"まで": "まで", "猫": "ねこ", "犬": "いぬ", "傷薬": "きずぐすり"}
            entry = {
                "headwords": [{"term": term, "reading": readings.get(term, term)}],
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

    def test_load_jpdb_candidates_with_utf8_bom(self) -> None:
        """Verify load_jpdb_candidates strips UTF-8 BOM cleanly from JPDB CSV/TSV input."""
        tsv_path = os.path.join(self.temp_dir.name, "bom_jpdb.tsv")
        with open(tsv_path, "wb") as f:
            f.write(b"\xef\xbb\xbf")  # UTF-8 BOM
            f.write("term\treading\tfrequency\tkana_frequency\n".encode("utf-8"))
            f.write("犬\tいぬ\t1\t1\n".encode("utf-8"))

        def fake_fetch(term: str, **kwargs: Any) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
            """Simulate Yomitan fetch returning empty list."""
            return [], True

        with patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", side_effect=fake_fetch):
            candidates, _ = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                limit=10,
                mode="all",
                cache_dir=self.temp_dir.name,
            )
        terms = [c["term"] for c in candidates]
        self.assertEqual(terms, ["犬"])

    def test_load_jpdb_candidates_unmatched_mode_tier2_filtering(self) -> None:
        """Verify load_jpdb_candidates mode='unmatched' excludes terms matching Tier 2 candidates."""
        # Create a catalog containing a post for 'メンバーのイラスト'
        db_path = os.path.join(self.temp_dir.name, "unmatched_catalog.sqlite")
        conn = sqlite3.connect(db_path)
        img = IrasutoyaImage("メンバー", "https://1.bp.blogspot.com/s800/member.png", "t.png", "メンバー", post_id="b_1")
        post = ParsedPost("b_1", "メンバーのイラスト", "desc", "pub", "upd", (), [img])
        write_posts(conn, [post])
        conn.close()

        tsv_path = os.path.join(self.temp_dir.name, "tier2_test_jpdb.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("term\treading\tfrequency\n")
            f.write("メンバ\tメンバ\t1\n")  # Matches 'メンバー' via Katakana chōonpu expansion (Tier 2)
            f.write("未登録語\tみとうろくご\t2\n")  # Unmatched (neither Tier 1 nor Tier 2)

        def fake_fetch(term: str, **kwargs: Any) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
            """Simulate Yomitan fetch returning empty list."""
            return [], True

        with patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", side_effect=fake_fetch):
            # non-tier1 mode includes 'メンバ' (Tier 2 match is not Tier 1) and '未登録語'
            cands_non_tier1, _ = load_jpdb_candidates(
                catalog_path=db_path,
                jpdb_path=tsv_path,
                limit=10,
                mode="non-tier1",
                cache_dir=self.temp_dir.name,
            )
            terms_non_tier1 = [c["term"] for c in cands_non_tier1]
            self.assertEqual(terms_non_tier1, ["メンバ", "未登録語"])

            # all mode includes both
            cands_all, _ = load_jpdb_candidates(
                catalog_path=db_path,
                jpdb_path=tsv_path,
                limit=10,
                mode="all",
                cache_dir=self.temp_dir.name,
            )
            terms_all = [c["term"] for c in cands_all]
            self.assertEqual(terms_all, ["メンバ", "未登録語"])

            # unmatched mode EXCLUDES 'メンバ' (because it has a Tier 2 match) and keeps only '未登録語'
            cands_unmatched, _ = load_jpdb_candidates(
                catalog_path=db_path,
                jpdb_path=tsv_path,
                limit=10,
                mode="unmatched",
                cache_dir=self.temp_dir.name,
            )
            terms_unmatched = [c["term"] for c in cands_unmatched]
            self.assertEqual(terms_unmatched, ["未登録語"])

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
                        "content": '{"visualizable": true, "tier": 1, "reason": "young feline", "targets": ["猫"]}',
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
                    no_embed=True,
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
                    no_embed=True,
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
                    no_embed=True,
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
                    no_embed=True,
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
        """Verify non-boolean visualizable value is treated as malformed retryable rather than rejection."""
        cands = [{"term": "偽物", "reading": "にせもの", "frequency": 1}]

        def fake_string_false_resp(req: Any, *args: Any, **kwargs: Any) -> Any:
            """Return non-boolean string 'false' for visualizable."""
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

        # 1. Non-boolean string 'false' is skipped without invoking on_reject
        with patch("urllib.request.urlopen", side_effect=fake_string_false_resp):
            res, failed = run_ollama_agent_batch(
                cands,
                catalog_path=self.db_path,
                on_reject=handle_reject,
            )
            self.assertEqual(res, {})
            self.assertEqual(failed, 0)
            self.assertEqual(len(rejections), 0)

        # 2. Genuine boolean False properly invokes on_reject
        def fake_bool_false_resp(req: Any, *args: Any, **kwargs: Any) -> Any:
            """Return genuine boolean False for visualizable."""
            resp_payload = {
                "message": {
                    "role": "assistant",
                    "content": '{"visualizable": false, "reason": "genuine grammatical rejection"}',
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

        with patch("urllib.request.urlopen", side_effect=fake_bool_false_resp):
            res, failed = run_ollama_agent_batch(
                cands,
                catalog_path=self.db_path,
                on_reject=handle_reject,
            )
            self.assertEqual(res, {})
            self.assertEqual(failed, 0)
            self.assertEqual(len(rejections), 1)
            self.assertEqual(rejections[0], ("偽物", "genuine grammatical rejection"))

    def test_run_ollama_agent_batch_callback_arity_dispatch(self) -> None:
        """Verify on_match and on_reject dispatch based on inspect.signature arity without TypeError swallowing."""
        cands = [{"term": "猫", "reading": "ねこ", "frequency": 1}]
        match_payload = {
            "message": {
                "role": "assistant",
                "content": '{"reason": "cat illustration", "visualizable": true, "targets": ["猫"], "tier": 1}',
            }
        }
        mock_b = json.dumps(match_payload).encode("utf-8")

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

        # 1. 2-argument callback receives (term, targets)
        cb2_calls: List[Tuple[str, List[str]]] = []

        def match_cb2(term: str, targets: List[str]) -> None:
            """2-argument match callback."""
            cb2_calls.append((term, targets))

        with patch("urllib.request.urlopen", return_value=MockResp()):
            res, _ = run_ollama_agent_batch(cands, catalog_path=self.db_path, on_match=match_cb2)
            self.assertEqual(len(cb2_calls), 1)
            self.assertEqual(cb2_calls[0], ("猫", ["猫"]))

        # 2. 3-argument callback receives (term, targets, tier)
        cb3_calls: List[Tuple[str, List[str], int]] = []

        def match_cb3(term: str, targets: List[str], tier: int) -> None:
            """3-argument match callback."""
            cb3_calls.append((term, targets, tier))

        with patch("urllib.request.urlopen", return_value=MockResp()):
            res, _ = run_ollama_agent_batch(cands, catalog_path=self.db_path, on_match=match_cb3)
            self.assertEqual(len(cb3_calls), 1)
            self.assertEqual(cb3_calls[0], ("猫", ["猫"], 1))

        # 3. Internal TypeError raised inside callback body is not retried with 2 args
        cb_err_count = 0

        def faulty_cb(term: str, targets: List[str], tier: int) -> None:
            """Callback raising internal TypeError."""
            nonlocal cb_err_count
            cb_err_count += 1
            raise TypeError("internal business logic failure")

        with patch("urllib.request.urlopen", return_value=MockResp()):
            res, failed = run_ollama_agent_batch(cands, catalog_path=self.db_path, on_match=faulty_cb)
            self.assertEqual(cb_err_count, 1)
            self.assertEqual(failed, 1)

    def test_run_ollama_agent_batch_presearch_failure_resilience(self) -> None:
        """Verify run_ollama_agent_batch gracefully survives presearch errors and continues."""
        cands = [{"term": "猫", "reading": "ねこ", "frequency": 1}]
        match_payload = {
            "message": {
                "role": "assistant",
                "content": '{"reason": "cat illustration", "visualizable": true, "targets": ["猫"], "tier": 1}',
            }
        }
        mock_b = json.dumps(match_payload).encode("utf-8")

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

        with (
            patch("scripts.synonyms.ollama_agent.search_hybrid_presearch", side_effect=RuntimeError("embed outage")),
            patch("urllib.request.urlopen", return_value=MockResp()),
        ):
            res, failed = run_ollama_agent_batch(cands, catalog_path=self.db_path)
            self.assertEqual(len(res), 1)
            self.assertEqual(res.get("猫"), ["猫"])
            self.assertEqual(failed, 0)

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

    def test_run_ollama_compile_rejects_malformed_tier2_concepts(self) -> None:
        """Verify run_ollama_compile refuses to overwrite when tier2_concepts is not a dictionary."""
        llm_out = os.path.join(self.temp_dir.name, "malformed_t2.json")
        with open(llm_out, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": 1,
                "concepts": {"猫": ["猫"]},
                "tier2_concepts": ["not_a_dict"],
            }, f)

        with (
            patch("scripts.synonyms.ollama_agent._fetch_yomitan_term_entries", return_value=([], True)),
            patch("scripts.synonyms.yomitan_jpdb._fetch_yomitan_term_entries", return_value=([], True)),
            patch("scripts.synonyms.ollama_agent.check_ollama_availability", return_value=(True, None)),
            patch("scripts.synonyms.ollama_agent.run_ollama_agent_batch") as mock_batch,
        ):
            ok = run_ollama_compile(
                catalog_path=self.db_path,
                llm_out_path=llm_out,
                terms=["犬"],
                cache_dir=self.temp_dir.name,
            )
            self.assertFalse(ok)
            mock_batch.assert_not_called()

        with open(llm_out, "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f).get("tier2_concepts"), ["not_a_dict"])

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
        self.assertIn("--embed-model", output)
        self.assertIn("--no-embed", output)
        self.assertIn("--recheck-rejected", output)
        self.assertIn("--pull-models", output)


class TestPullModelsAndEmbedPresearch(unittest.TestCase):
    """Test suite for declarative models manifest, vector index, and hybrid pre-search."""

    def setUp(self) -> None:
        """Create an isolated temporary environment for hermetic testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_embed_catalog.sqlite")

        # Create basic catalog database with two sample posts
        conn = sqlite3.connect(self.db_path)
        img1 = IrasutoyaImage("可愛い猫", "https://1.bp.blogspot.com/s800/cat.png", "t.png", "猫", post_id="p_cat")
        post1 = ParsedPost("p_cat", "可愛い猫のイラスト", "desc cat", "pub", "upd", ("動物",), [img1])
        img2 = IrasutoyaImage("走る犬", "https://1.bp.blogspot.com/s800/dog.png", "t.png", "犬", post_id="p_dog")
        post2 = ParsedPost("p_dog", "走る犬のイラスト", "desc dog", "pub", "upd", ("動物",), [img2])
        write_posts(conn, [post1, post2])
        conn.close()

    def tearDown(self) -> None:
        """Clean up temporary test workspace."""
        self.temp_dir.cleanup()

    def test_load_models_manifest_valid_and_errors(self) -> None:
        """Verify load_models_manifest parses valid manifests and raises on invalid inputs."""
        # Test loading the repo's real models.yml
        data = load_models_manifest(DEFAULT_MODELS_PATH)
        cfg = data.get("ollama", data)
        self.assertIn("models", cfg)
        self.assertIsInstance(cfg["models"], list)
        self.assertTrue(any(m.get("name") == "gemma4:26b" for m in cfg["models"]))
        self.assertTrue(any(m.get("name") == "qwen3-embedding:4b" for m in cfg["models"]))

        # Test loading the repo's real models.yml without PyYAML via json.loads fallback
        with patch.dict("sys.modules", {"yaml": None}):
            data_no_yaml = load_models_manifest(DEFAULT_MODELS_PATH)
            self.assertIn("models", data_no_yaml.get("ollama", data_no_yaml))

        # Verify JSON manifest saved with .yml or .json extension parses correctly
        json_yml_path = os.path.join(self.temp_dir.name, "json_manifest.json")
        with open(json_yml_path, "w", encoding="utf-8") as f:
            json.dump({"ollama": {"models": [{"name": "test-model"}]}}, f)
        data_json_yml = load_models_manifest(json_yml_path)
        self.assertEqual(data_json_yml["ollama"]["models"][0]["name"], "test-model")

        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            load_models_manifest(os.path.join(self.temp_dir.name, "nonexistent.yml"))

        # Test invalid non-dict structure
        invalid_path = os.path.join(self.temp_dir.name, "invalid.json")
        with open(invalid_path, "w", encoding="utf-8") as f:
            json.dump(["not", "a", "dict"], f)
        with self.assertRaises(ValueError):
            load_models_manifest(invalid_path)

        # Test missing PyYAML on non-JSON YAML content raises clear error
        yaml_path = os.path.join(self.temp_dir.name, "manifest.yml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("ollama:\n  models:\n    - name: test\n")
        with patch.dict("sys.modules", {"yaml": None}):
            with self.assertRaises(ValueError) as ctx:
                load_models_manifest(yaml_path)
            self.assertIn("PyYAML is not installed", str(ctx.exception))

    def test_get_installed_ollama_models_handling(self) -> None:
        """Verify get_installed_ollama_models parses tags and normalizes :latest tag suffix."""
        mock_payload = {
            "models": [
                {"name": "gemma4:26b"},
                {"name": "qwen3-embedding:4b"},
                {"name": "custom-model:latest"},
            ]
        }

        class MockResp:
            """Mock HTTP response."""

            def __enter__(self) -> "MockResp":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self) -> bytes:
                """Read response body."""
                return json.dumps(mock_payload).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=MockResp()) as mock_urlopen:
            installed = get_installed_ollama_models("http://proxy:8080/prefix")
            self.assertIsNotNone(installed)
            assert installed is not None
            self.assertIn("gemma4:26b", installed)
            self.assertIn("qwen3-embedding:4b", installed)
            self.assertIn("custom-model:latest", installed)
            self.assertIn("custom-model", installed)
            self.assertEqual(mock_urlopen.call_args[0][0].full_url, "http://proxy:8080/prefix/api/tags")

        # Verify network error returns None so callers can detect failure
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Refused")):
            installed_err = get_installed_ollama_models("http://localhost:11434")
            self.assertIsNone(installed_err)

    def test_pull_single_model_streaming(self) -> None:
        """Verify pull_single_model parses streaming JSON lines and reports success or error."""
        class MockStreamingResp:
            """Mock streaming HTTP response."""

            def __init__(self, lines: List[bytes]) -> None:
                """Initialize streaming lines."""
                self._lines = lines

            def __enter__(self) -> "MockStreamingResp":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def __iter__(self) -> Any:
                """Iterate lines."""
                return iter(self._lines)

        success_lines = [
            b'{"status": "pulling manifest"}\n',
            b'{"status": "downloading", "completed": 50, "total": 100}\n',
            b'{"status": "success"}\n',
        ]
        with (
            patch("scripts.pull_models.shutil.which", return_value=None),
            patch("urllib.request.urlopen", return_value=MockStreamingResp(success_lines)) as mock_pull_open,
        ):
            ok = pull_single_model("gemma4:26b", ollama_url="http://proxy:8080/prefix")
            self.assertTrue(ok)
            self.assertEqual(mock_pull_open.call_args[0][0].full_url, "http://proxy:8080/prefix/api/pull")

        error_lines = [
            b'{"status": "pulling manifest"}\n',
            b'{"error": "model \'unknown:latest\' not found"}\n',
        ]
        with (
            patch("scripts.pull_models.shutil.which", return_value=None),
            patch("urllib.request.urlopen", return_value=MockStreamingResp(error_lines)),
        ):
            fail = pull_single_model("unknown:latest")
            self.assertFalse(fail)

        # Early stream truncation without 'success' event should report False
        truncated_lines = [
            b'{"status": "downloading", "completed": 50, "total": 100}\n',
        ]
        with (
            patch("scripts.pull_models.shutil.which", return_value=None),
            patch("urllib.request.urlopen", return_value=MockStreamingResp(truncated_lines)),
        ):
            fail_trunc = pull_single_model("truncated:latest")
            self.assertFalse(fail_trunc)

        # CLI subprocess branch honors stream_progress=False by capturing output
        with (
            patch("scripts.pull_models.shutil.which", return_value="/usr/local/bin/ollama"),
            patch("scripts.pull_models.subprocess.run") as mock_cli_run,
        ):
            mock_cli_run.return_value.returncode = 0
            ok_cli_quiet = pull_single_model("gemma4:26b", stream_progress=False)
            self.assertTrue(ok_cli_quiet)
            self.assertEqual(mock_cli_run.call_args[1].get("stdout"), subprocess.PIPE)
            self.assertEqual(mock_cli_run.call_args[1].get("stderr"), subprocess.PIPE)

    def test_pull_declared_models_and_cli(self) -> None:
        """Verify pull_declared_models respects presence, dry-run, and force flags."""
        manifest_path = os.path.join(self.temp_dir.name, "test_models.yml")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "models": [
                    {"name": "model_present:1", "role": "chat", "size_gb": 1.0},
                    {"name": "model_missing:1", "role": "embed", "size_gb": 2.0},
                ]
            }, f)

        # Case 1: dry_run=True should not attempt to pull
        with (
            patch("scripts.pull_models.get_installed_ollama_models", return_value={"model_present:1"}),
            patch("scripts.pull_models.pull_single_model") as mock_pull,
        ):
            ok_dry = pull_declared_models(manifest_path=manifest_path, dry_run=True)
            self.assertTrue(ok_dry)
            mock_pull.assert_not_called()

        # Case 2: dry_run=False pulls missing model
        with (
            patch("scripts.pull_models.get_installed_ollama_models", return_value={"model_present:1"}),
            patch("scripts.pull_models.pull_single_model", return_value=True) as mock_pull,
        ):
            ok_pull = pull_declared_models(manifest_path=manifest_path, dry_run=False)
            self.assertTrue(ok_pull)
            mock_pull.assert_called_once_with("model_missing:1", ollama_url="http://localhost:11434")

        # Case 3: normalizes ollama_url ending in /api/chat
        with (
            patch("scripts.pull_models.get_installed_ollama_models", return_value=set()),
            patch("scripts.pull_models.pull_single_model", return_value=True) as mock_pull_norm,
        ):
            ok_norm = pull_declared_models(
                manifest_path=manifest_path,
                ollama_url="http://localhost:11434/api/chat",
                dry_run=False,
            )
            self.assertTrue(ok_norm)
            self.assertEqual(mock_pull_norm.call_args[1].get("ollama_url"), "http://localhost:11434")

        # Case 4: uses manifest endpoint when ollama_url is None
        manifest_custom_ep = os.path.join(self.temp_dir.name, "test_custom_ep.yml")
        with open(manifest_custom_ep, "w", encoding="utf-8") as f:
            json.dump({
                "ollama": {
                    "endpoint": "http://remote-ollama:11434",
                    "models": [{"name": "remote_model:1"}],
                }
            }, f)
        with (
            patch("scripts.pull_models.get_installed_ollama_models", return_value=set()),
            patch("scripts.pull_models.pull_single_model", return_value=True) as mock_pull_custom,
        ):
            ok_custom = pull_declared_models(manifest_path=manifest_custom_ep, ollama_url=None)
            self.assertTrue(ok_custom)
            self.assertEqual(mock_pull_custom.call_args[1].get("ollama_url"), "http://remote-ollama:11434")

        # Case 5: returns False when get_installed_ollama_models returns None (even in dry_run)
        with patch("scripts.pull_models.get_installed_ollama_models", return_value=None):
            ok_unreach = pull_declared_models(manifest_path=manifest_path, dry_run=True)
            self.assertFalse(ok_unreach)

        # Case 6: returns False when models is not a list
        manifest_bad_models = os.path.join(self.temp_dir.name, "bad_models.json")
        with open(manifest_bad_models, "w", encoding="utf-8") as f:
            json.dump({"ollama": {"models": "not_a_list"}}, f)
        ok_bad_models = pull_declared_models(manifest_path=manifest_bad_models)
        self.assertFalse(ok_bad_models)

        # Case 7: returns False when an entry has no name
        manifest_bad_entry = os.path.join(self.temp_dir.name, "bad_entry.json")
        with open(manifest_bad_entry, "w", encoding="utf-8") as f:
            json.dump({"ollama": {"models": [{"role": "chat"}]}}, f)
        with patch("scripts.pull_models.get_installed_ollama_models", return_value=set()):
            ok_bad_entry = pull_declared_models(manifest_path=manifest_bad_entry)
            self.assertFalse(ok_bad_entry)

        # Test CLI entrypoint
        with patch("scripts.pull_models.pull_declared_models", return_value=True) as mock_decl:
            code = pull_models_main(["--manifest", manifest_path, "--dry-run"])
            self.assertEqual(code, 0)
            mock_decl.assert_called_once()

    def test_pull_single_model_cli_env(self) -> None:
        """Verify pull_single_model sets OLLAMA_HOST environment variable when invoking ollama CLI."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        with (
            patch("scripts.pull_models.shutil.which", return_value="/usr/local/bin/ollama"),
            patch("scripts.pull_models.subprocess.run", return_value=mock_proc) as mock_subproc,
        ):
            ok = pull_single_model("gemma4:26b", ollama_url="http://custom-host:11434")
            self.assertTrue(ok)
            mock_subproc.assert_called_once()
            called_env = mock_subproc.call_args[1].get("env", {})
            self.assertEqual(called_env.get("OLLAMA_HOST"), "http://custom-host:11434")

    def test_keep_alive_json_client(self) -> None:
        """Verify _KeepAliveJSONClient formats requests and routes through patched urlopen in tests."""
        client = _KeepAliveJSONClient(timeout=10.0)
        mock_body = {"response": "test_output"}

        class MockResp:
            """Mock HTTP response."""

            def __enter__(self) -> "MockResp":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self) -> bytes:
                """Read response bytes."""
                return json.dumps(mock_body).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=MockResp()) as mock_url:
            resp = client.post_json("http://localhost:11434/api/chat", {"test": 123})
            self.assertEqual(resp, mock_body)
            mock_url.assert_called_once()

        # Verify unexpected error closes and removes connection
        mock_conn = MagicMock()
        mock_conn.request.side_effect = TimeoutError("read timed out")
        client._connections["localhost:11434"] = mock_conn
        with patch("urllib.request.getproxies", return_value={}):
            with self.assertRaises(TimeoutError):
                client.post_json("http://localhost:11434/api/chat", {"test": 123})
        self.assertNotIn("localhost:11434", client._connections)
        mock_conn.close.assert_called_once()

        client.close()

    def test_ollama_embed_client_batch_and_single(self) -> None:
        """Verify OllamaEmbedClient sends batch payloads and normalizes vector outputs."""
        # Test URL normalization when passed /api/chat or /api/embed
        client_chat_url = OllamaEmbedClient("http://localhost:11434/api/chat")
        self.assertEqual(client_chat_url._embed_url, "http://localhost:11434/api/embed")

        client = OllamaEmbedClient("http://localhost:11434")
        self.assertEqual(client._embed_url, "http://localhost:11434/api/embed")
        mock_resp_data = {
            "embeddings": [
                [3.0, 4.0],  # L2 norm is 5.0 -> normalized: [0.6, 0.8]
                [0.0, 2.0],  # L2 norm is 2.0 -> normalized: [0.0, 1.0]
            ]
        }

        class MockResp:
            """Mock HTTP response."""

            def __enter__(self) -> "MockResp":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self) -> bytes:
                """Read response body."""
                return json.dumps(mock_resp_data).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=MockResp()):
            embeddings = client.embed_batch(["猫", "犬"], batch_size=2)
            self.assertEqual(len(embeddings), 2)
            self.assertAlmostEqual(embeddings[0][0], 0.6)
            self.assertAlmostEqual(embeddings[0][1], 0.8)
            self.assertAlmostEqual(embeddings[1][0], 0.0)
            self.assertAlmostEqual(embeddings[1][1], 1.0)

        class MockSingleResp:
            """Mock single HTTP response."""

            def __enter__(self) -> "MockSingleResp":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self) -> bytes:
                """Read response body."""
                return json.dumps({"embeddings": [[3.0, 4.0]]}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=MockSingleResp()):
            single = client.embed_single("猫")
            self.assertIsNotNone(single)
            if single:
                self.assertAlmostEqual(single[0], 0.6)

        # Empty input handling
        self.assertEqual(client.embed_batch([]), [])
        self.assertIsNone(client.embed_single("   "))

    def test_catalog_vector_index_build_save_load_search(self) -> None:
        """Verify CatalogVectorIndex binary serialization, caching, and cosine similarity search."""
        embed_client = OllamaEmbedClient("http://localhost:11434")
        # Return orthogonal 2D vectors for the 2 posts
        mock_vectors = [[1.0, 0.0], [0.0, 1.0]]

        with patch.object(embed_client, "embed_batch", return_value=mock_vectors):
            idx = CatalogVectorIndex.build_or_load(
                catalog_path=self.db_path,
                client=embed_client,
                cache_dir=self.temp_dir.name,
                batch_size=10,
                dimensions=2,
                build_if_missing=True,
            )
            self.assertIsNotNone(idx)
            if idx is None:
                return

            self.assertEqual(len(idx.post_ids), 2)
            self.assertEqual(idx.dim, 2)

            # Query aligned with vector 0 ([1.0, 0.0]) -> top result should be post 0
            results = idx.search_similar([1.0, 0.0], top_k=2)
            self.assertGreaterEqual(len(results), 1)
            top_post_id, top_score = results[0]
            self.assertEqual(top_post_id, idx.post_ids[0])
            self.assertAlmostEqual(top_score, 1.0, places=4)

            # Verify reload from disk without re-embedding
            with patch.object(embed_client, "embed_batch") as mock_never:
                cached_idx = CatalogVectorIndex.build_or_load(
                    catalog_path=self.db_path,
                    client=embed_client,
                    cache_dir=self.temp_dir.name,
                    dimensions=2,
                )
                mock_never.assert_not_called()
                self.assertIsNotNone(cached_idx)
                if cached_idx:
                    self.assertEqual(len(cached_idx.post_ids), 2)

    def test_clean_post_for_embedding(self) -> None:
        """Verify clean_post_for_embedding strips boilerplate and combines title with description."""
        # 1. Title with のイラスト and description with boilerplate
        t1 = "スクール水着のイラスト（上下セット）"
        d1 = "体育の授業で着る紺色の水着のイラストです。"
        res1 = clean_post_for_embedding(t1, d1)
        self.assertEqual(res1, "スクール水着（上下セット）: 体育の授業で着る紺色の水着")

        # 2. Description ending in を描いたイラストです
        t2 = "猫のイラスト"
        d2 = "可愛い子猫を描いたイラストです。"
        res2 = clean_post_for_embedding(t2, d2)
        self.assertEqual(res2, "猫: 可愛い子猫")

        # 3. Description ending in のキャラクターです
        t3 = "アマビエのイラスト"
        d3 = "疫病退散のアマビエのキャラクターです。"
        res3 = clean_post_for_embedding(t3, d3)
        self.assertEqual(res3, "アマビエ: 疫病退散のアマビエ")

        # 4. Title only
        res4 = clean_post_for_embedding("犬のイラスト", "")
        self.assertEqual(res4, "犬")

        # 5. Empty title fallback
        res5 = clean_post_for_embedding("", "猫のイラストです。")
        self.assertEqual(res5, "猫")

    def test_format_query_for_embedding(self) -> None:
        """Verify format_query_for_embedding formats instruction header and English glosses."""
        # 1. Term only with default instruction
        q1 = format_query_for_embedding("猫")
        self.assertIn(f"Instruct: {QWEN3_SEARCH_INSTRUCT}\nQuery: 猫", q1)

        # 2. Term with English glosses
        q2 = format_query_for_embedding("一喜一憂", english_glosses=["alternating between joy and grief", "mood swings"])
        self.assertIn("Query: 一喜一憂 (alternating between joy and grief, mood swings)", q2)

        # 3. Empty term returns empty string
        self.assertEqual(format_query_for_embedding("   "), "")

        # 4. Custom instruction
        q4 = format_query_for_embedding("犬", instruct="Retrieve dog illustrations")
        self.assertEqual(q4, "Instruct: Retrieve dog illustrations\nQuery: 犬")

    def test_ollama_embed_client_chunk_bisect_and_dimensions(self) -> None:
        """Verify OllamaEmbedClient bisects failing chunks and passes dimensions option."""
        client = OllamaEmbedClient("http://localhost:11434")

        calls: List[int] = []

        def mock_post(url: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
            """Mock post_json recording chunk lengths and bisecting."""
            inputs = payload.get("input", [])
            calls.append(len(inputs))
            if len(inputs) >= 3:
                raise urllib.error.HTTPError(url, 400, "Batch too large", {}, None)
            return {"embeddings": [[1.0, 0.0] for _ in inputs]}

        with patch.object(client._http_client, "post_json", side_effect=mock_post):
            res = client.embed_batch(["1", "2", "3", "4"], batch_size=4, dimensions=512)
            self.assertEqual(len(res), 4)
            self.assertIn(4, calls)
            self.assertEqual(calls.count(2), 2)

        client.close()

    def test_catalog_vector_index_partial_checkpoint_and_resume(self) -> None:
        """Verify CatalogVectorIndex.build_or_load resumes from partial checkpoint files."""
        embed_client = OllamaEmbedClient("http://localhost:11434")
        cache_dir = self.temp_dir.name
        clean_model = "qwen3-embedding_4b"
        partial_base = os.path.join(cache_dir, f"catalog_post_embeddings_{clean_model}.partial")

        # Pre-seed a partial checkpoint with 1 post completed out of 2
        h = compute_catalog_posts_sha256(self.db_path)

        with open(f"{partial_base}.json", "w", encoding="utf-8") as f_meta:
            json.dump({
                "count": 1,
                "dim": 2,
                "catalog_sha256": h,
                "model": "qwen3-embedding:4b",
            }, f_meta)

        with open(f"{partial_base}.bin", "wb") as f_bin:
            arr = array.array("f", [1.0, 0.0])
            arr.tofile(f_bin)

        embed_calls: List[List[str]] = []

        def mock_embed_batch(texts: Sequence[str], **kwargs: Any) -> List[List[float]]:
            """Mock embed_batch tracking passed texts."""
            embed_calls.append(list(texts))
            return [[0.0, 1.0] for _ in texts]

        with patch.object(embed_client, "embed_batch", side_effect=mock_embed_batch):
            idx = CatalogVectorIndex.build_or_load(
                catalog_path=self.db_path,
                client=embed_client,
                cache_dir=cache_dir,
                batch_size=1,
                dimensions=2,
                build_if_missing=True,
            )
            self.assertIsNotNone(idx)
            if idx:
                self.assertEqual(len(idx.post_ids), 2)
                # Verify that only the 2nd post was embedded
                self.assertEqual(len(embed_calls), 1)
                self.assertEqual(len(embed_calls[0]), 1)
                # Verify partial files were cleaned up
                self.assertFalse(os.path.isfile(f"{partial_base}.bin"))
                self.assertFalse(os.path.isfile(f"{partial_base}.json"))

    def test_compute_catalog_posts_sha256_handles_error(self) -> None:
        """Verify compute_catalog_posts_sha256 returns empty string on missing or invalid database."""
        missing_db = os.path.join(self.temp_dir.name, "does_not_exist.sqlite")
        with self.assertLogs("synonyms.ollama_embed", level="WARNING") as cm:
            res = compute_catalog_posts_sha256(missing_db)
            self.assertEqual(res, "")
            self.assertTrue(any("Failed to fingerprint catalog posts" in log for log in cm.output))

    def test_run_ollama_agent_batch_upfront_embedding(self) -> None:
        """Verify run_ollama_agent_batch pre-computes candidate embeddings up front in a single batch."""
        cands = [
            {"term": "猫", "reading": "ねこ", "frequency": 1, "english_glosses": ["cat"]},
            {"term": "犬", "reading": "いぬ", "frequency": 2, "english_glosses": ["dog"]},
        ]
        match_payload = {
            "message": {
                "role": "assistant",
                "content": '{"reason": "test", "visualizable": true, "targets": ["猫"], "tier": 1}',
            }
        }

        class MockResp:
            """Mock HTTP response."""

            def __enter__(self) -> "MockResp":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self) -> bytes:
                """Read response body."""
                return json.dumps(match_payload).encode("utf-8")

        mock_client = MagicMock(spec=OllamaEmbedClient)
        mock_client.embed_batch.return_value = [[1.0, 0.0], [0.0, 1.0]]

        mock_vector_idx = MagicMock(spec=CatalogVectorIndex)
        mock_vector_idx.dim = 2
        mock_vector_idx.search_similar.return_value = [(1, 0.9)]

        with (
            patch("urllib.request.urlopen", return_value=MockResp()),
            patch("scripts.synonyms.ollama_agent.search_hybrid_presearch", return_value=[]) as mock_presearch,
        ):
            res, failed = run_ollama_agent_batch(
                cands,
                catalog_path=self.db_path,
                vector_index=mock_vector_idx,
                embed_client=mock_client,
            )
            # Upfront batch embedding should be called once for all 2 candidates
            mock_client.embed_batch.assert_called_once()
            called_texts = mock_client.embed_batch.call_args[0][0]
            self.assertEqual(len(called_texts), 2)
            self.assertIn("猫 (cat)", called_texts[0])
            self.assertIn("犬 (dog)", called_texts[1])

            # Presearch should receive the pre-computed query vectors
            self.assertEqual(mock_presearch.call_count, 2)
            self.assertEqual(mock_presearch.call_args_list[0][1].get("query_vector"), [1.0, 0.0])
            self.assertEqual(mock_presearch.call_args_list[1][1].get("query_vector"), [0.0, 1.0])

    def test_search_hybrid_presearch_and_formatting(self) -> None:
        """Verify search_hybrid_presearch merges lexical and vector candidates and formats markdown block."""
        embed_client = OllamaEmbedClient("http://localhost:11434")
        mock_vector = [1.0, 0.0]

        with (
            patch.object(embed_client, "embed_single", return_value=mock_vector),
            patch.object(CatalogVectorIndex, "build_or_load") as mock_build_idx,
        ):
            # Create a mock vector index that returns the first post
            mock_idx = MagicMock()
            mock_idx.dim = 2
            mock_idx.search_similar.return_value = [(1, 0.95)]
            mock_build_idx.return_value = mock_idx

            candidates = search_hybrid_presearch(
                term="猫",
                catalog_path=self.db_path,
                embed_client=embed_client,
                vector_index=mock_idx,
                limit=5,
            )
            self.assertIsInstance(candidates, list)
            self.assertGreater(len(candidates), 0)
            self.assertEqual(candidates[0]["suggested_target"], "可愛い猫")

            formatted = format_candidate_context_block(candidates)
            self.assertIn("Target: 「可愛い猫」", formatted)
            self.assertIn("可愛い猫のイラスト", formatted)

    def test_parse_llm_json_response_reason_first_and_tiers(self) -> None:
        """Verify _parse_llm_json_response parses reason-first JSON and extracts confidence tier."""
        # Tier 1 direct mapping
        t1_json = json.dumps({
            "reason": "傷薬 is medicine for cuts, visualizable as 塗り薬",
            "visualizable": True,
            "targets": ["塗り薬"],
            "tier": 1,
        })
        parsed1 = _parse_llm_json_response(t1_json)
        self.assertIsNotNone(parsed1)
        if parsed1:
            self.assertTrue(parsed1.get("visualizable"))
            self.assertEqual(parsed1.get("targets"), ["塗り薬"])
            self.assertEqual(parsed1.get("tier"), 1)
            self.assertIn("medicine for cuts", parsed1.get("reason", ""))

        # Tier 2 visual exemplar / prototypical scene
        t2_json = json.dumps({
            "reason": "広い is an abstract adjective best represented by a wide meadow",
            "visualizable": True,
            "targets": ["広い野原"],
            "tier": 2,
        })
        parsed2 = _parse_llm_json_response(t2_json)
        self.assertIsNotNone(parsed2)
        if parsed2:
            self.assertTrue(parsed2.get("visualizable"))
            self.assertEqual(parsed2.get("targets"), ["広い野原"])
            self.assertEqual(parsed2.get("tier"), 2)
            self.assertIn("wide meadow", parsed2.get("reason", ""))

        # Abstract / non-visual rejection
        t_rej_json = json.dumps({
            "reason": "心の中 is an abstract mental concept with no canonical illustration",
            "visualizable": False,
            "targets": [],
        })
        parsed_rej = _parse_llm_json_response(t_rej_json)
        self.assertIsNotNone(parsed_rej)
        if parsed_rej:
            self.assertFalse(parsed_rej.get("visualizable"))
            self.assertEqual(parsed_rej.get("targets"), [])
            self.assertIn("abstract mental concept", parsed_rej.get("reason", ""))

    def test_flush_llm_concepts_tier1_and_tier2(self) -> None:
        """Verify _flush_llm_concepts writes both Tier 1 concepts and Tier 2 concepts."""
        out_path = os.path.join(self.temp_dir.name, "out_concepts.json")
        _flush_llm_concepts(
            concepts_map={"猫": ["猫"]},
            out_path=out_path,
            model="gemma4:26b",
            tier2_map={"広い": ["広い野原"]},
            rejected_map={"心の中": "abstract mental state"},
        )

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["source"], "llm_concept")
        self.assertEqual(data["model"], "gemma4:26b")
        self.assertEqual(data["concepts"], {"猫": ["猫"]})
        self.assertEqual(data["tier2_concepts"], {"広い": ["広い野原"]})
        self.assertEqual(data["rejected"], {"心の中": "abstract mental state"})

    def test_run_ollama_compile_with_recheck_rejected(self) -> None:
        """Verify run_ollama_compile re-evaluates rejected terms when recheck_rejected is True."""
        llm_out = os.path.join(self.temp_dir.name, "llm_recheck.json")
        with open(llm_out, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": 1,
                "concepts": {},
                "tier2_concepts": {},
                "rejected": {"猫": "previously rejected"},
            }, f)

        mock_candidates = [
            {"term": "猫", "reading": "ねこ", "frequency": 1, "pos_tags": [], "english_glosses": [], "is_uk": False}
        ]
        mock_jpdb_path = os.path.join(self.temp_dir.name, "fake_jpdb.csv")
        with open(mock_jpdb_path, "w", encoding="utf-8") as f:
            f.write("term\treading\tfreq\tkana_freq\n猫\tねこ\t1\t1\n")

        with (
            patch("scripts.synonyms.ollama_agent.load_jpdb_candidates", return_value=(mock_candidates, 0)),
            patch("scripts.synonyms.ollama_agent.check_ollama_availability", return_value=(True, None)),
            patch("scripts.synonyms.ollama_agent.run_ollama_agent_batch", return_value=({"猫": ["猫"]}, 0)) as mock_batch,
        ):
            # Case 1: recheck_rejected=False -> "猫" is skipped because it is already rejected
            ok1 = run_ollama_compile(
                catalog_path=self.db_path,
                jpdb_path=mock_jpdb_path,
                llm_out_path=llm_out,
                cache_dir=self.temp_dir.name,
                recheck_rejected=False,
                no_embed=True,
            )
            self.assertTrue(ok1)
            mock_batch.assert_not_called()

            # Case 2: recheck_rejected=True -> "猫" is included for re-evaluation
            ok2 = run_ollama_compile(
                catalog_path=self.db_path,
                jpdb_path=mock_jpdb_path,
                llm_out_path=llm_out,
                cache_dir=self.temp_dir.name,
                recheck_rejected=True,
                no_embed=True,
            )
            self.assertTrue(ok2)
            mock_batch.assert_called_once()

    def test_main_cli_agent_synonyms_new_flags(self) -> None:
        """Verify agent_synonyms main CLI passes --embed-model, --no-embed, --recheck-rejected, --pull-models."""
        with (
            patch("scripts.agent_synonyms.run_ollama_compile", return_value=True) as mock_compile,
            patch("scripts.pull_models.pull_declared_models", return_value=True) as mock_pull,
        ):
            code = main([
                "--catalog", self.db_path,
                "--embed-model", "custom-embed:latest",
                "--no-embed",
                "--recheck-rejected",
                "--pull-models",
                "--terms", "猫",
            ])
            self.assertEqual(code, 0)
            mock_pull.assert_called_once()
            mock_compile.assert_called_once()
            _, kwargs = mock_compile.call_args
            self.assertEqual(kwargs.get("embed_model"), "custom-embed:latest")
            self.assertTrue(kwargs.get("no_embed"))
            self.assertTrue(kwargs.get("recheck_rejected"))
            self.assertEqual(kwargs.get("ollama_model"), "gemma4:26b")

    def test_get_model_options_from_manifest(self) -> None:
        """Verify get_model_options_from_manifest extracts options dictionary for declared models."""
        opts = get_model_options_from_manifest("gemma4:26b")
        self.assertIsInstance(opts, dict)
        self.assertEqual(opts.get("seed"), 42)
        self.assertEqual(opts.get("num_ctx"), 4096)
        self.assertEqual(opts.get("temperature_turn0"), 0.1)

        # Unknown model returns empty dict
        empty_opts = get_model_options_from_manifest("nonexistent-model")
        self.assertEqual(empty_opts, {})

    def test_run_ollama_agent_batch_invalid_manifest_options_fallback(self) -> None:
        """Verify run_ollama_agent_batch falls back gracefully to defaults when manifest options are malformed or non-finite."""
        invalid_opts = {
            "temperature_turn0": "not-a-float",
            "seed": None,
            "num_ctx": "invalid",
            "temperature_turn1": float("nan"),
            "temperature_turn2": float("inf"),
            "repeat_penalty": float("-inf"),
            "top_k": "1e1000",
        }
        with (
            patch("scripts.synonyms.ollama_agent.get_model_options_from_manifest", return_value=invalid_opts),
            patch("scripts.synonyms.common._KeepAliveJSONClient.post_json", return_value={"message": {"content": '{"reason": "x", "visualizable": false}'}}),
        ):
            mappings, fails = run_ollama_agent_batch(
                candidates=[{"term": "テスト", "reading": "てすと"}],
                catalog_path=self.db_path,
            )
            self.assertEqual(mappings, {})
            self.assertEqual(fails, 0)

    def test_keepalive_client_proxy_routing(self) -> None:
        """Verify _KeepAliveJSONClient routes through urlopen when a proxy is configured."""
        client = _KeepAliveJSONClient()
        mock_resp_body = b'{"status": "ok"}'

        class MockResp:
            """Mock HTTP response."""

            def __enter__(self) -> "MockResp":
                """Enter context manager."""
                return self

            def __exit__(self, *args: Any) -> None:
                """Exit context manager."""
                pass

            def read(self) -> bytes:
                """Read response body."""
                return mock_resp_body

        with (
            patch("urllib.request.getproxies", return_value={"http": "http://corporate-proxy:8080"}),
            patch("urllib.request.proxy_bypass", return_value=False),
            patch("urllib.request.urlopen", return_value=MockResp()) as mock_urlopen,
        ):
            res = client.post_json("http://remote-server/api/embed", {"prompt": "test"})
            self.assertEqual(res, {"status": "ok"})
            self.assertTrue(mock_urlopen.called)

    def test_ollama_agent_accepted_scene_defaults_tier2(self) -> None:
        """Verify that when Ollama omits tier on an accepted scene, it defaults to Tier 2."""
        mock_resp = {
            "message": {
                "content": json.dumps({
                    "reason": "representative scene",
                    "visualizable": True,
                    "targets": ["猫"],
                    # "tier" is omitted
                })
            }
        }

        matched_tiers = []

        def match_cb(term: str, targets: List[str], tier: int) -> None:
            """Callback collecting matched tier."""
            matched_tiers.append(tier)

        with patch("scripts.synonyms.common._KeepAliveJSONClient.post_json", return_value=mock_resp):
            mappings, fail = run_ollama_agent_batch(
                candidates=[{"term": "心の中", "reading": "こころのなか"}],
                catalog_path=self.db_path,
                on_match=match_cb,
            )
            self.assertEqual(len(mappings), 1)
            self.assertEqual(fail, 0)
            self.assertEqual(matched_tiers, [2])

    def test_search_hybrid_presearch_semantic_contribution(self) -> None:
        """Verify semantic hits can contribute even when lexical hits reach the cap."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        for i in range(5):
            cur.execute(
                "INSERT INTO posts (id, blogger_id, title, description, published, updated, image_count) VALUES (?, ?, ?, ?, 'pub', 'upd', 1)",
                (100 + i, f"blog_{100+i}", f"猫_{i}のイラスト", "desc"),
            )
        cur.execute(
            "INSERT INTO posts (id, blogger_id, title, description, published, updated, image_count) VALUES (999, 'blog_999', '特別なイラスト', 'desc', 'pub', 'upd', 1)",
        )
        conn.commit()

        class MockVectorIndex:
            """Mock vector index."""

            dim = 512

            def search_similar(self, vec: Any, top_k: int = 5) -> List[Tuple[int, float]]:
                """Search similar vectors."""
                return [(999, 0.95)]

        hits = search_hybrid_presearch(
            term="猫",
            catalog_path=self.db_path,
            conn=conn,
            vector_index=MockVectorIndex(),  # type: ignore
            query_vector=[0.1] * 512,
            limit=5,
        )
        conn.close()
        sources = [h["source"] for h in hits]
        self.assertIn("semantic", sources)
        self.assertTrue(any(h["post_id"] == 999 for h in hits))

    def test_resolve_valid_post_target(self) -> None:
        """Verify resolve_valid_post_target resolves clean, runtime-valid targets for catalog posts."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        # Post 501: standard suffix '...のイラスト' where candidate matches illustration
        cur.execute(
            "INSERT INTO posts (id, blogger_id, title, description, published, updated, image_count) VALUES (501, 'b_501', '学校の建物のイラスト', '学校です', 'pub', 'upd', 1)",
        )
        cur.execute(
            "INSERT INTO images (id, post_id, key, ord, title, url) VALUES (501, 501, 'k501', 0, '学校の建物のイラスト', 'https://example.com/s0/501.png')",
        )
        # Post 502: lacks standard suffix, clean_search_query is used
        cur.execute(
            "INSERT INTO posts (id, blogger_id, title, description, published, updated, image_count) VALUES (502, 'b_502', '黒い猫', 'desc', 'pub', 'upd', 1)",
        )
        cur.execute(
            "INSERT INTO images (id, post_id, key, ord, title, url) VALUES (502, 502, 'k502', 0, '黒い猫のイラスト', 'https://example.com/s0/502.png')",
        )
        # Post 503: primary title '色々な動物のイラスト' fails validation, but image title 'かわいい犬のイラスト' passes
        cur.execute(
            "INSERT INTO posts (id, blogger_id, title, description, published, updated, image_count) VALUES (503, 'b_503', '色々な動物のイラスト', '動物たちです', 'pub', 'upd', 1)",
        )
        cur.execute(
            "INSERT INTO images (id, post_id, key, ord, title, url) VALUES (503, 503, 'k503', 0, 'かわいい犬のイラスト', 'https://example.com/s0/503.png')",
        )
        conn.commit()

        store = CatalogStore(self.db_path)
        index = store.get_index()

        target_501 = resolve_valid_post_target(501, "学校の建物のイラスト", index, conn)
        self.assertEqual(target_501, "学校の建物")

        target_502 = resolve_valid_post_target(502, "黒い猫", index, conn)
        self.assertEqual(target_502, "黒い猫")

        target_503 = resolve_valid_post_target(503, "色々な動物のイラスト", index, conn)
        self.assertEqual(target_503, "かわいい犬")

        # Without an in-memory index, returns None
        target_no_index = resolve_valid_post_target(501, "学校の建物のイラスト", None, conn)
        self.assertIsNone(target_no_index)

        # Non-matching title with no valid image returns None
        target_unverified = resolve_valid_post_target(999, "完全な架空のタイトル", index, conn)
        self.assertIsNone(target_unverified)
        conn.close()

    def test_search_hybrid_presearch_title_priority_and_seen_blogger_dedup(self) -> None:
        """Verify title lexical hits precede description hits and seen_blogger_ids prevents redundant queries."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        # Post matching only via description
        cur.execute(
            "INSERT INTO posts (id, blogger_id, title, description, published, updated, image_count) VALUES (601, 'b_601', '青空のイラスト', '白い犬が走っています', 'pub', 'upd', 1)",
        )
        cur.execute(
            "INSERT INTO images (id, post_id, key, ord, title, url) VALUES (601, 601, 'k601', 0, '青空のイラスト', 'https://example.com/s0/601.png')",
        )
        # Post matching directly in title
        cur.execute(
            "INSERT INTO posts (id, blogger_id, title, description, published, updated, image_count) VALUES (602, 'b_602', '白い犬のイラスト', 'desc', 'pub', 'upd', 1)",
        )
        cur.execute(
            "INSERT INTO images (id, post_id, key, ord, title, url) VALUES (602, 602, 'k602', 0, '白い犬のイラスト', 'https://example.com/s0/602.png')",
        )
        conn.commit()

        stub_img1 = IrasutoyaImage(
            "白い犬のイラスト",
            "https://example.com/s0/602.png",
            "",
            "白い犬のイラスト",
            post_id="b_602",
        )
        stub_img2 = IrasutoyaImage(
            "白い犬のイラスト2",
            "https://example.com/s0/602.png",
            "",
            "白い犬のイラスト2",
            post_id="b_602",
        )

        class _StubCatalogIndex:
            """Stub index returning multiple records sharing blogger_id 'b_602'."""

            def search(self, query: str) -> List[IrasutoyaImage]:
                """Return multiple illustrations sharing the same post_id/blogger_id."""
                return [stub_img1, stub_img2]

        lookups: List[str] = []

        def _record_lookup(sql: str) -> None:
            """Record statements that query posts by Blogger ID."""
            if "FROM posts WHERE blogger_id" in sql:
                lookups.append(sql)

        conn.set_trace_callback(_record_lookup)

        hits = search_hybrid_presearch(
            term="白い犬",
            catalog_path=self.db_path,
            index=_StubCatalogIndex(),  # type: ignore[arg-type]
            conn=conn,
            limit=5,
        )
        conn.close()
        self.assertGreaterEqual(len(hits), 2)
        # Title hit (602) must come before description-only hit (601)
        self.assertEqual(hits[0]["post_id"], 602)
        self.assertEqual(hits[1]["post_id"], 601)
        # Since b_602 was already in seen_blogger_ids from the title hit, the stub index hit causes 0 extra lookups
        self.assertEqual(len(lookups), 0)

    def test_search_catalog_for_agent_with_semantic_fallback(self) -> None:
        """Verify search_catalog_for_agent appends semantic vector hits when title matches are below limit."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO posts (id, blogger_id, title, description, published, updated, image_count) VALUES (701, 'b_701', '宇宙のイラスト', 'desc', 'pub', 'upd', 1)",
        )
        cur.execute(
            "INSERT INTO images (id, post_id, key, ord, title, url) VALUES (701, 701, 'k701', 0, '宇宙のイラスト', 'https://example.com/s0/701.png')",
        )
        conn.commit()

        class MockVecIdx:
            """Mock vector index."""

            dim = 512

            def search_similar(self, vec: Any, top_k: int = 5) -> List[Tuple[int, float]]:
                """Return mock hit 701."""
                return [(701, 0.88)]

        class MockEmbedClient:
            """Mock embed client."""

            def embed_single(self, text: str, model: str = "qwen3-embedding:4b", dimensions: int = 512) -> List[float]:
                """Return dummy vector."""
                return [0.1] * dimensions

        res = search_catalog_for_agent(
            catalog_path=self.db_path,
            query="銀河",
            limit=5,
            conn=conn,
            vector_index=MockVecIdx(),
            embed_client=MockEmbedClient(),
        )
        conn.close()
        post_ids = [p["post_id"] for p in res["posts"]]
        self.assertIn(701, post_ids)

    def test_num_predict_payload_forwarding(self) -> None:
        """Verify run_ollama_agent_batch passes num_predict in generation options."""
        captured_payloads: List[Dict[str, Any]] = []

        def mock_post_json(url: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
            """Capture payload and return rejection."""
            captured_payloads.append(payload)
            return {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "reason": "非視覚的",
                        "visualizable": False,
                        "tier": 2,
                        "targets": [],
                    }),
                }
            }

        with (
            patch("scripts.synonyms.ollama_agent.get_model_options_from_manifest", return_value={"num_predict": 123}),
            patch("scripts.synonyms.common._KeepAliveJSONClient.post_json", side_effect=mock_post_json),
        ):
            run_ollama_agent_batch(
                candidates=[{"term": "はずがない", "reading": "はずがない"}],
                catalog_path=self.db_path,
                model="gemma4:26b",
            )
        self.assertTrue(captured_payloads)
        opts = captured_payloads[0].get("options", {})
        self.assertIn("num_predict", opts)
        self.assertEqual(opts["num_predict"], 123)

    def test_run_ollama_agent_batch_clean_search_query_canonicalization(self) -> None:
        """Verify candidate targets with gender suffixes are canonicalized via clean_search_query."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO posts (id, blogger_id, title, description, published, updated, image_count) VALUES (801, 'b_801', '喜ぶ人のイラスト', 'desc', 'pub', 'upd', 1)",
        )
        cur.execute(
            "INSERT INTO images (id, post_id, key, ord, title, url) VALUES (801, 801, 'k801', 0, '喜ぶ人のイラスト', 'https://example.com/s0/801.png')",
        )
        conn.commit()
        conn.close()

        # LLM returns target with （女性） suffix
        mock_resp = {
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "reason": "直接一致",
                    "visualizable": True,
                    "tier": 1,
                    "targets": ["喜ぶ人のイラスト（女性）"],
                }),
            }
        }

        matched_targets: List[str] = []

        def match_cb(term: str, targets: List[str], tier: int = 1) -> None:
            """Collect matched targets."""
            matched_targets.extend(targets)

        with patch("scripts.synonyms.common._KeepAliveJSONClient.post_json", return_value=mock_resp):
            mappings, fail = run_ollama_agent_batch(
                candidates=[{"term": "喜ぶ", "reading": "よろこぶ"}],
                catalog_path=self.db_path,
                on_match=match_cb,
            )
            self.assertEqual(len(mappings), 1)
            self.assertEqual(matched_targets, ["喜ぶ人のイラスト"])

    def test_run_ollama_agent_batch_logs_warning_on_done_reason_length(self) -> None:
        """Verify a warning is logged when Ollama response indicates truncation via done_reason='length'."""
        mock_resp = {
            "done_reason": "length",
            "eval_count": 200,
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "reason": "直接一致",
                    "visualizable": False,
                }),
            },
        }

        with (
            patch("scripts.synonyms.common._KeepAliveJSONClient.post_json", return_value=mock_resp),
            self.assertLogs(logger="synonyms.ollama_agent", level="WARNING") as cm,
        ):
            run_ollama_agent_batch(
                candidates=[{"term": "はずがない", "reading": "はずがない"}],
                catalog_path=self.db_path,
            )
        self.assertTrue(any("was truncated (done_reason='length'" in log_msg for log_msg in cm.output))

    def test_ollama_embed_chunk_truncation_to_dimensions(self) -> None:
        """Verify _embed_chunk truncates oversized embedding vectors before L2 normalization."""
        from scripts.synonyms.ollama_embed import OllamaEmbedClient

        client = OllamaEmbedClient(base_url="http://localhost:11434")
        # Oversized 3D vector [3.0, 4.0, 12.0] truncated to 2D is [3.0, 4.0], normalized to [0.6, 0.8]
        with patch.object(client._http_client, "post_json", return_value={"embeddings": [[3.0, 4.0, 12.0]]}):
            result = client._embed_chunk(["test"], model="test-embed", dimensions=2, options={})
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result[0][0], 0.6, places=5)
            self.assertAlmostEqual(result[0][1], 0.8, places=5)

    def test_clean_and_validate_concept_dict_rejects_non_idempotent_targets(self) -> None:
        """Verify _populate_concept_json_mapping rejects targets whose clean_search_query value is non-idempotent."""
        from scripts.synonyms.concepts import _populate_concept_json_mapping

        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as tf:
            json.dump({"concepts": {"ペット": ["猫、犬/鳥"]}}, tf)
            tf_path = tf.name

        try:
            success = _populate_concept_json_mapping(
                catalog_path=self.db_path,
                json_path=tf_path,
                source="test",
                min_matched=1,
                validate_only=True,
            )
            self.assertFalse(success)
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)


if __name__ == "__main__":
    unittest.main()
