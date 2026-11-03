"""Unit tests for the synonym build and validation CLI (scripts/build_synonyms.py)."""
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
import urllib.error
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

from addon.catalog import create_schema, write_posts, write_synonyms
from addon.feed import IrasutoyaImage, ParsedPost
from scripts.build_synonyms import (
    _populate_concept_json_mapping,
    clean_sudachi_headword,
    fetch_jpdb_freq_github,
    fetch_kaishi_github_slice,
    fetch_sudachi_github,
    load_jpdb_candidates,
    main,
    populate_bilingual_glosses,
    populate_kaishi_concepts,
    populate_llm_concepts,
    populate_manual_concepts,
    populate_sudachi_synonyms,
    query_yomitan_term_metadata,
    run_ollama_agent_batch,
    run_ollama_compile,
    search_catalog_for_agent,
    validate_catalog_synonyms,
)


class TestBuildSynonyms(unittest.TestCase):
    """Test suite for manual concept population and catalog synonym validation."""

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

    def tearDown(self) -> None:
        """Clean up temporary test workspace."""
        self.temp_dir.cleanup()

    def test_populate_manual_concepts_unmatched_target_returns_false_without_modifying_synonyms(self) -> None:
        """Verify unmatched manual targets return False and leave existing synonym rows untouched."""
        # Pre-populate an existing synonym row
        conn = sqlite3.connect(self.db_path)
        write_synonyms(conn, source="existing_source", mapping={"既存": ["猫"]})
        conn.close()

        # Create manual concepts file with an unmatched target
        json_path = os.path.join(self.temp_dir.name, "invalid_concepts.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"concepts": {"テスト": ["存在しないターゲット12345"]}}, f)

        success = populate_manual_concepts(self.db_path, json_path)
        self.assertFalse(success)

        # Confirm synonym rows remain unchanged
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target, source FROM synonyms")
        rows = cur.fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], ("既存", "猫", "existing_source"))

    def test_populate_manual_concepts_valid_input_writes_and_refreshes_sha256(self) -> None:
        """Verify valid input writes source='manual' rows and refreshes the sha256 checksum sidecar."""
        json_path = os.path.join(self.temp_dir.name, "valid_concepts.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"concepts": {"ネコ": ["猫"]}}, f)

        success = populate_manual_concepts(self.db_path, json_path)
        self.assertTrue(success)

        # Reopen database and verify row was written
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target, source, ord FROM synonyms WHERE source = 'manual'")
        rows = cur.fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], ("ネコ", "猫", "manual", 0))

        # Verify sha256 sidecar file exists and matches file contents
        sha_path = f"{self.db_path}.sha256"
        self.assertTrue(os.path.isfile(sha_path))
        with open(self.db_path, "rb") as f:
            expected_digest = hashlib.sha256(f.read()).hexdigest()
        with open(sha_path, "r", encoding="utf-8") as f:
            sha_content = f.read()
        self.assertEqual(sha_content, f"{expected_digest}  {os.path.basename(self.db_path)}\n")

    def test_populate_bilingual_glosses_valid_input_writes_and_refreshes_sha256(self) -> None:
        """Verify valid bilingual concepts write source='bilingual_gloss' rows and refresh sha256."""
        json_path = os.path.join(self.temp_dir.name, "valid_bilingual.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"concepts": {"cat": ["猫"], "feline": ["猫"]}}, f)

        success = populate_bilingual_glosses(self.db_path, json_path, min_matched=2)
        self.assertTrue(success)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target, source FROM synonyms WHERE source = 'bilingual_gloss' ORDER BY term")
        rows = cur.fetchall()
        conn.close()
        self.assertEqual(rows, [("cat", "猫", "bilingual_gloss"), ("feline", "猫", "bilingual_gloss")])

        sha_path = f"{self.db_path}.sha256"
        self.assertTrue(os.path.isfile(sha_path))

    def test_populate_bilingual_glosses_unmatched_target_fails_and_rolls_back(self) -> None:
        """Verify unmatched bilingual targets abort and leave database unchanged."""
        json_path = os.path.join(self.temp_dir.name, "invalid_bilingual.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"concepts": {"unknown": ["存在しないイラスト999"]}}, f)

        success = populate_bilingual_glosses(self.db_path, json_path, min_matched=1)
        self.assertFalse(success)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM synonyms WHERE source = 'bilingual_gloss'")
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    def test_populate_bilingual_glosses_min_matched_guard(self) -> None:
        """Verify populate_bilingual_glosses fails when concept count is below min_matched."""
        json_path = os.path.join(self.temp_dir.name, "few_bilingual.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"concepts": {"cat": ["猫"]}}, f)

        success = populate_bilingual_glosses(self.db_path, json_path, min_matched=5)
        self.assertFalse(success)

    def test_populate_bilingual_glosses_min_matched_after_validation(self) -> None:
        """Verify populate_bilingual_glosses fails when valid concepts fall below min_matched."""
        json_path = os.path.join(self.temp_dir.name, "dropped_bilingual.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"concepts": {"cat": ["猫"], "empty": []}}, f)

        # raw count is 2 (passes raw_concepts check), but valid count is 1 < 2
        success = populate_bilingual_glosses(self.db_path, json_path, min_matched=2)
        self.assertFalse(success)

    def test_populate_bilingual_glosses_non_canonical_headword_fails(self) -> None:
        """Verify populate_bilingual_glosses rejects unnormalized headwords at build time."""
        json_path = os.path.join(self.temp_dir.name, "unnormalized_bilingual.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"concepts": {"To Run": ["猫"]}}, f)

        success = populate_bilingual_glosses(self.db_path, json_path, min_matched=1)
        self.assertFalse(success)

    def test_main_cli_bilingual(self) -> None:
        """Verify main() CLI executes --bilingual flag."""
        json_path = os.path.join(self.temp_dir.name, "cli_bilingual.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"concepts": {"cat": ["猫"]}}, f)

        with patch("scripts.build_synonyms.BILINGUAL_MIN_MATCHED", 1):
            ret = main(["--catalog", self.db_path, "--bilingual", json_path])
            self.assertEqual(ret, 0)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target FROM synonyms WHERE source = 'bilingual_gloss'")
        rows = cur.fetchall()
        conn.close()
        self.assertEqual(rows, [("cat", "猫")])

    def test_validate_catalog_synonyms_unmatched_stored_target_fails(self) -> None:
        """Verify validate_catalog_synonyms returns False when a stored target has no title matches."""
        conn = sqlite3.connect(self.db_path)
        write_synonyms(conn, source="manual", mapping={"ネコ": ["存在しないターゲット999"]})
        conn.close()

        valid = validate_catalog_synonyms(self.db_path)
        self.assertFalse(valid)

    def test_validate_catalog_synonyms_missing_or_empty_table_passes(self) -> None:
        """Verify validate_catalog_synonyms returns True when the synonyms table is missing or empty."""
        # 1. Missing table
        fresh_db = os.path.join(self.temp_dir.name, "no_synonyms.sqlite")
        conn = sqlite3.connect(fresh_db)
        post = ParsedPost("1", "猫のイラスト", "desc", "pub", "upd", (), [IrasutoyaImage("猫", "https://cat.png", "t", "猫")])
        write_posts(conn, [post])
        conn.execute("DROP TABLE synonyms")
        conn.commit()
        conn.close()
        self.assertTrue(validate_catalog_synonyms(fresh_db))

        # 2. Empty table
        conn = sqlite3.connect(fresh_db)
        create_schema(conn)
        conn.close()
        self.assertTrue(validate_catalog_synonyms(fresh_db))

    def test_main_cli_kaishi_apkg_and_validate(self) -> None:
        """Verify main() CLI processes --kaishi-apkg and --validate end-to-end."""
        # 1. Create a dummy sqlite db with a note
        raw_db = os.path.join(self.temp_dir.name, "cli_kaishi.sqlite")
        conn = sqlite3.connect(raw_db)
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, flds TEXT NOT NULL)")
        conn.execute("INSERT INTO notes (flds) VALUES (?)", ("キャット\x1fきゃっと\x1f<img src='cat.png'>",))
        conn.commit()
        conn.close()

        # 2. Pack into zip .apkg
        apkg_path = os.path.join(self.temp_dir.name, "cli_test.apkg")
        with open(raw_db, "rb") as f_in:
            data = f_in.read()
        with zipfile.ZipFile(apkg_path, "w") as z:
            z.writestr("collection.anki21", data)

        # 3. Invoke main with --catalog, --kaishi-apkg, --cache-dir, --min-matched 1, --validate
        ret = main([
            "--catalog",
            self.db_path,
            "--kaishi-apkg",
            apkg_path,
            "--cache-dir",
            self.temp_dir.name,
            "--min-matched",
            "1",
            "--validate",
        ])
        self.assertEqual(ret, 0)

        # 4. Confirm data was inserted into catalog
        v_conn = sqlite3.connect(self.db_path)
        cur = v_conn.cursor()
        cur.execute("SELECT keyword, source FROM image_keywords WHERE source = 'donkuri/kaishi'")
        self.assertEqual(cur.fetchall(), [("キャット", "donkuri/kaishi")])
        v_conn.close()

    def test_main_cli_kaishi_github_mocked(self) -> None:
        """Verify main() CLI handles --kaishi-github via mocked collection fetch."""
        raw_db = os.path.join(self.temp_dir.name, "gh_mock.sqlite")
        conn = sqlite3.connect(raw_db)
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, flds TEXT NOT NULL)")
        conn.execute("INSERT INTO notes (flds) VALUES (?)", ("子猫\x1fこねこ\x1f<img src='cat.png'>",))
        conn.commit()
        conn.close()

        with patch("scripts.build_synonyms.fetch_kaishi_github_slice", return_value=raw_db):
            ret = main([
                "--catalog",
                self.db_path,
                "--kaishi-github",
                "--cache-dir",
                self.temp_dir.name,
                "--min-matched",
                "1",
                "--validate",
            ])
            self.assertEqual(ret, 0)

        v_conn = sqlite3.connect(self.db_path)
        cur = v_conn.cursor()
        cur.execute("SELECT keyword, source FROM image_keywords WHERE keyword = '子猫'")
        self.assertEqual(len(cur.fetchall()), 1)
        v_conn.close()

    def test_main_cli_kaishi_apkg_rejects_truncated_collection_by_default(self) -> None:
        """Verify main() CLI rejects an incomplete collection by default when matched count < 1000."""
        raw_db = os.path.join(self.temp_dir.name, "truncated_kaishi.sqlite")
        conn = sqlite3.connect(raw_db)
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, flds TEXT NOT NULL)")
        conn.execute("INSERT INTO notes (flds) VALUES (?)", ("キャット\x1fきゃっと\x1f<img src='cat.png'>",))
        conn.commit()
        conn.close()

        apkg_path = os.path.join(self.temp_dir.name, "truncated_test.apkg")
        with open(raw_db, "rb") as f_in:
            data = f_in.read()
        with zipfile.ZipFile(apkg_path, "w") as z:
            z.writestr("collection.anki21", data)

        # Default min_matched is 1000, so a 1-note deck must return exit code 1
        ret = main([
            "--catalog",
            self.db_path,
            "--kaishi-apkg",
            apkg_path,
            "--cache-dir",
            self.temp_dir.name,
        ])
        self.assertEqual(ret, 1)

    def test_fetch_kaishi_github_slice_validates_zip_header(self) -> None:
        """Verify fetch_kaishi_github_slice raises descriptive ValueError on malformed or truncated headers."""
        import struct

        def make_urlopen_mock(slice_payload: bytes):
            def _fake_open(req, *args, **kwargs):
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if url.startswith("https://api.github.com/"):
                    return io.BytesIO(b'{"tag_name": "test_tag", "assets": []}')
                return io.BytesIO(slice_payload)
            return _fake_open

        # Case 1: Entry not found
        with patch("urllib.request.urlopen", side_effect=make_urlopen_mock(b"random bytes")):
            with self.assertRaises(ValueError) as cm:
                fetch_kaishi_github_slice(cache_dir=self.temp_dir.name)
            self.assertIn("collection.anki21b not found", str(cm.exception))

        # Case 2: Invalid signature
        fake_slice = b"\x00" * 30 + b"collection.anki21b" + b"\x00" * 100
        with patch("urllib.request.urlopen", side_effect=make_urlopen_mock(fake_slice)):
            with self.assertRaises(ValueError) as cm:
                fetch_kaishi_github_slice(cache_dir=self.temp_dir.name)
            self.assertIn("Invalid ZIP local header signature", str(cm.exception))

        # Case 3: Truncated slice data
        hdr = struct.pack("<4sHHHHHIIIHH", b"PK\x03\x04", 20, 0, 0, 0, 0, 0, 1000, 1000, 18, 0)
        truncated_slice = hdr + b"collection.anki21b" + b"short_data"
        with patch("urllib.request.urlopen", side_effect=make_urlopen_mock(truncated_slice)):
            with self.assertRaises(ValueError) as cm:
                fetch_kaishi_github_slice(cache_dir=self.temp_dir.name)
            self.assertIn("is larger than the fetched 512KB slice", str(cm.exception))

    def test_kaishi_concept_dataset_counts_and_validation(self) -> None:
        """Verify synthesized catalog contains deterministic Kaishi dataset counts and passes synonym validation."""
        target_db = os.path.join(self.temp_dir.name, "synth_catalog.sqlite")
        conn = sqlite3.connect(target_db)
        p1 = ParsedPost(
            "1",
            "可愛い猫のイラスト",
            "desc",
            "pub",
            "upd",
            (),
            [IrasutoyaImage("可愛い猫のイラスト", "https://1.bp.blogspot.com/cat.png", "t", "可愛い猫", post_id="1")],
        )
        p2 = ParsedPost(
            "2",
            "走る人のイラスト",
            "desc",
            "pub",
            "upd",
            (),
            [IrasutoyaImage("走る人のイラスト", "https://1.bp.blogspot.com/run.png", "t", "走る人", post_id="2")],
        )
        p3 = ParsedPost(
            "3",
            "先生のイラスト",
            "desc",
            "pub",
            "upd",
            (),
            [IrasutoyaImage("先生のイラスト", "https://1.bp.blogspot.com/teacher.png", "t", "先生", post_id="3")],
        )
        write_posts(conn, [p1, p2, p3])
        conn.close()

        small_kaishi = os.path.join(self.temp_dir.name, "synth_kaishi.sqlite")
        k_conn = sqlite3.connect(small_kaishi)
        k_conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, flds TEXT NOT NULL)")
        notes = [
            ("キャット\x1fきゃっと\x1f<img src='cat.png'>",),
            ("教える\x1fおしえる\x1f<img src='teacher.png'>",),
            ("走る\x1fはしる\x1f<img src='run.png'>",),
            ("未登録\x1fみとうろく\x1f<img src='missing.png'>",),
        ]
        k_conn.executemany("INSERT INTO notes (flds) VALUES (?)", notes)
        k_conn.commit()
        k_conn.close()

        # 4 notes total: 3 matched (75% match ratio >= 70% threshold, min_matched=3), 1 unmatched
        self.assertTrue(populate_kaishi_concepts(target_db, small_kaishi, min_matched=3))

        conn = sqlite3.connect(target_db)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM synonyms WHERE source = 'donkuri/kaishi'")
        synonyms_count = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT term) FROM synonyms WHERE source = 'donkuri/kaishi'")
        unique_terms_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM image_keywords WHERE source = 'donkuri/kaishi'")
        keywords_count = cur.fetchone()[0]
        conn.close()

        # Verify exact deterministic counts:
        # - 3 matched notes produce 3 image_keywords pins ('キャット', '教える', '走る')
        # - '走る' directly matches '走る人のイラスト' so it is omitted from synonyms
        # - 'キャット' and '教える' generate 2 concept synonyms across 2 unique headwords
        self.assertEqual(synonyms_count, 2)
        self.assertEqual(unique_terms_count, 2)
        self.assertEqual(keywords_count, 3)
        self.assertTrue(validate_catalog_synonyms(target_db))

    def test_populate_sudachi_synonyms_compiles_valid_mapping(self) -> None:
        """Verify populate_sudachi_synonyms compiles valid Sudachi CSV records into catalog.sqlite."""
        # Add posts with illustrations for targets '寝る', '買う', 'キューピッド', and '天使'
        conn = sqlite3.connect(self.db_path)
        img1 = IrasutoyaImage("寝る人のイラスト", "https://1.bp.blogspot.com/s800/sleep.png", "t1.png", "寝る", post_id="2")
        post1 = ParsedPost("2", "睡眠のイラスト", "desc", "pub", "upd", (), [img1])
        img2 = IrasutoyaImage("買う人のイラスト", "https://1.bp.blogspot.com/s800/buy.png", "t2.png", "買う", post_id="3")
        post2 = ParsedPost("3", "買う人のイラスト", "desc", "pub", "upd", (), [img2])
        img3 = IrasutoyaImage("キューピッドのイラスト", "https://1.bp.blogspot.com/s800/cupid.png", "t3.png", "キューピッド", post_id="4")
        post3 = ParsedPost("4", "キューピッドのイラスト", "desc", "pub", "upd", (), [img3])
        img4 = IrasutoyaImage("天使のイラスト", "https://1.bp.blogspot.com/s800/angel.png", "t4.png", "天使", post_id="5")
        post4 = ParsedPost("5", "天使のイラスト", "desc", "pub", "upd", (), [img4])
        write_posts(conn, [post1, post2, post3, post4])
        conn.close()

        csv_content = (
            "000016,1,0,1,0,0,0,(),購入,,\n"
            "000016,2,0,3,0,0,0,(),買う,,\n"
            "023242,1,0,1,0,0,0,(),就寝,,\n"
            "023242,2,0,4,0,0,0,(),寝る,,\n"
            "006612,1,0,1,0,0,1,(人名),Cupid,,\n"
            "006612,2,0,3,0,0,0,(),キューピッド,,\n"
            "006613,1,0,1,0,0,1,(人名),CUPID,,\n"
            "006613,2,0,3,0,0,0,(),天使,,\n"
        )
        csv_path = os.path.join(self.temp_dir.name, "test_sudachi.txt")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        success = populate_sudachi_synonyms(self.db_path, csv_path)
        self.assertTrue(success)

        # Verify rows in SQLite
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target, source, ord FROM synonyms WHERE source = 'sudachi' ORDER BY term, ord")
        rows = cur.fetchall()
        cur.execute("SELECT value FROM meta WHERE key = 'synonym_sources'")
        meta_sources = cur.fetchone()[0]
        conn.close()

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0], ("cupid", "キューピッド", "sudachi", 0))
        self.assertEqual(rows[1], ("cupid", "天使", "sudachi", 1))
        self.assertEqual(rows[2], ("就寝", "寝る", "sudachi", 0))
        self.assertEqual(rows[3], ("購入", "買う", "sudachi", 0))
        self.assertIn("sudachi", meta_sources)

        # Validate with catalog validator
        self.assertTrue(validate_catalog_synonyms(self.db_path))

    def test_populate_sudachi_synonyms_filtering_and_guards(self) -> None:
        """Verify compiler precision guards: archaic, misuse, misspelling, derogatory, and predicate partitioning."""
        conn = sqlite3.connect(self.db_path)
        img1 = IrasutoyaImage("走る人のイラスト", "https://1.bp.blogspot.com/s800/run1.png", "t1.png", "走る", post_id="2")
        post1 = ParsedPost("2", "走る人のイラスト", "desc", "pub", "upd", (), [img1])
        img2 = IrasutoyaImage("猫のイラスト2", "https://1.bp.blogspot.com/s800/cat2.png", "t2.png", "猫", post_id="3")
        img3 = IrasutoyaImage("猫のイラスト3", "https://1.bp.blogspot.com/s800/cat3.png", "t3.png", "猫", post_id="4")
        img_run2 = IrasutoyaImage("走る人のイラスト2", "https://1.bp.blogspot.com/s800/run2.png", "t2.png", "走る", post_id="5")
        img_run3 = IrasutoyaImage("走る人のイラスト3", "https://1.bp.blogspot.com/s800/run3.png", "t3.png", "走る", post_id="6")
        post5 = ParsedPost("5", "走る人のイラスト2", "desc", "pub", "upd", (), [img_run2])
        post6 = ParsedPost("6", "走る人のイラスト3", "desc", "pub", "upd", (), [img_run3])
        write_posts(
            conn,
            [
                post1,
                ParsedPost("3", "猫2", "desc", "pub", "upd", (), [img2]),
                ParsedPost("4", "猫3", "desc", "pub", "upd", (), [img3]),
                post5,
                post6,
            ],
        )
        # '猫' and '走る' have >= 3 canonical title matches in catalog -> skipped as sources by delta gate
        # Seed an existing manual concept for '既出' -> should be skipped by delta gate
        write_synonyms(conn, source="manual", mapping={"既出": ["猫"]})
        conn.close()

        csv_content = (
            # Group with archaic (reg=3), misuse (reg=4), misspelling (orth=3), derogatory (不快), disabled (exp=2)
            "000001,1,0,1,0,0,0,(),テスト源,,\n"
            "000001,1,0,2,3,0,0,(),古語標的,,\n"
            "000001,1,0,3,4,0,0,(),誤用標的,,\n"
            "000001,1,0,4,0,0,3,(),誤表記標的,,\n"
            "000001,1,0,5,0,0,0,(不快),不快語,,\n"
            "000001,1,2,6,0,0,0,(),展開不可,,\n"
            "000001,1,0,7,0,0,0,(),走る,,\n"
            # Predicate partitioning: predicate (is_noun=2) must not cross-map to noun (is_noun=1)
            "000002,2,0,1,0,0,0,(),動詞源,,\n"
            "000002,1,0,2,0,0,0,(),走る,,\n"
            # Delta gate: source '猫' has >= 3 canonical matches -> skipped
            "000003,1,0,1,0,0,0,(),猫,,\n"
            "000003,1,0,2,0,0,0,(),走る,,\n"
            # Delta gate: source '既出' already in manual synonyms -> skipped
            "000004,1,0,1,0,0,0,(),既出,,\n"
            "000004,1,0,2,0,0,0,(),走る,,\n"
        )
        csv_path = os.path.join(self.temp_dir.name, "guards_sudachi.txt")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        success = populate_sudachi_synonyms(self.db_path, csv_path)
        self.assertTrue(success)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target FROM synonyms WHERE source = 'sudachi'")
        rows = cur.fetchall()
        conn.close()

        # Only 'テスト源' -> '走る' should be emitted!
        self.assertEqual(rows, [("テスト源", "走る")])

    def test_populate_sudachi_synonyms_missing_files_and_atomic_rollback(self) -> None:
        """Verify populate_sudachi_synonyms handles missing files and rolls back atomically on error."""
        missing_db = os.path.join(self.temp_dir.name, "missing.sqlite")
        missing_csv = os.path.join(self.temp_dir.name, "missing.txt")
        csv_path = os.path.join(self.temp_dir.name, "empty.txt")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("")

        self.assertFalse(populate_sudachi_synonyms(missing_db, csv_path))
        self.assertFalse(populate_sudachi_synonyms(self.db_path, missing_csv))
        self.assertFalse(populate_sudachi_synonyms(self.db_path, csv_path))

        # Test rollback on write failure
        csv_path_valid = os.path.join(self.temp_dir.name, "rollback.txt")
        with open(csv_path_valid, "w", encoding="utf-8") as f:
            f.write("000001,1,0,1,0,0,0,(),犬,,\n000001,1,0,2,0,0,0,(),猫,,\n")

        conn = sqlite3.connect(self.db_path)
        write_synonyms(conn, source="sudachi", mapping={"旧語": ["猫"]})
        conn.close()

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TRIGGER fail_sudachi BEFORE INSERT ON synonyms
            BEGIN
                SELECT RAISE(ABORT, 'Simulated failure during sudachi insert');
            END;
            """
        )
        conn.commit()
        conn.close()

        self.assertFalse(populate_sudachi_synonyms(self.db_path, csv_path_valid))

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target FROM synonyms WHERE source = 'sudachi'")
        self.assertEqual(cur.fetchall(), [("旧語", "猫")])
        conn.close()

    def test_populate_sudachi_synonyms_incomplete_collection_refuses_replacement(self) -> None:
        """Verify populate_sudachi_synonyms refuses to replace existing data if new count is below 80%."""
        # Pre-seed 10 existing sudachi rows
        conn = sqlite3.connect(self.db_path)
        seed_mapping = {f"term_{i}": [f"tgt_{i}"] for i in range(10)}
        write_synonyms(conn, source="sudachi", mapping=seed_mapping)
        conn.close()

        # CSV that compiles to only 1 row (< 80% of 10)
        csv_path = os.path.join(self.temp_dir.name, "small_sudachi.txt")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("000001,1,0,1,0,0,0,(),イヌ,,\n000001,1,0,2,0,0,0,(),猫,,\n")

        self.assertFalse(populate_sudachi_synonyms(self.db_path, csv_path))

        # Confirm original 10 rows are preserved
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM synonyms WHERE source = 'sudachi'")
        self.assertEqual(cur.fetchone()[0], 10)
        conn.close()

    def test_main_cli_sudachi_and_validate(self) -> None:
        """Verify main CLI executes --sudachi and --validate flags."""
        csv_path = os.path.join(self.temp_dir.name, "cli_sudachi.txt")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("000001,1,0,1,0,0,0,(),イヌ,,\n000001,1,0,2,0,0,0,(),猫,,\n")

        ret = main(["--catalog", self.db_path, "--sudachi", csv_path, "--validate"])
        self.assertEqual(ret, 0)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target FROM synonyms WHERE source = 'sudachi'")
        rows = cur.fetchall()
        conn.close()
        self.assertEqual(rows, [("イヌ", "猫")])

    def test_main_cli_sudachi_github_mocked(self) -> None:
        """Verify main() handles --sudachi-github success and fetch failure."""
        csv_path = os.path.join(self.temp_dir.name, "gh_sudachi.txt")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("000001,1,0,1,0,0,0,(),イヌ,,\n000001,1,0,2,0,0,0,(),猫,,\n")

        with patch("scripts.build_synonyms.fetch_sudachi_github", return_value=csv_path):
            self.assertEqual(main(["--catalog", self.db_path, "--sudachi-github"]), 0)

        with patch(
            "scripts.build_synonyms.fetch_sudachi_github",
            side_effect=urllib.error.URLError("Simulated network failure"),
        ):
            self.assertEqual(main(["--catalog", self.db_path, "--sudachi-github"]), 1)

    def test_fetch_sudachi_github_mocked(self) -> None:
        """Verify fetch_sudachi_github downloads, validates minimum size, and caches synonyms.txt."""
        mock_csv = b"000001,1,0,1,0,0,0,(),sample,,\n" * 35000
        mock_resp = io.BytesIO(mock_csv)

        class MockUrlOpen:
            def __enter__(self) -> io.BytesIO:
                return mock_resp

            def __exit__(self, *args: object) -> None:
                pass

        def fake_urlopen(*args: object, **kwargs: object) -> MockUrlOpen:
            return MockUrlOpen()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            cache_dir = os.path.join(self.temp_dir.name, "cache")
            path = fetch_sudachi_github(cache_dir=cache_dir, url="https://example.com/mock_synonyms.txt")
            self.assertTrue(os.path.isfile(path))
            with open(path, "rb") as f:
                self.assertEqual(f.read(), mock_csv)

            # Test cache reuse
            path2 = fetch_sudachi_github(cache_dir=cache_dir, url="https://example.com/mock_synonyms.txt")
            self.assertEqual(path2, path)

        # Test small payload raises ValueError
        small_resp = io.BytesIO(b"too small")

        class SmallUrlOpen:
            def __enter__(self) -> io.BytesIO:
                return small_resp

            def __exit__(self, *args: object) -> None:
                pass

        def fake_small_urlopen(*args: object, **kwargs: object) -> SmallUrlOpen:
            return SmallUrlOpen()

        with patch("urllib.request.urlopen", side_effect=fake_small_urlopen):
            new_cache = os.path.join(self.temp_dir.name, "small_cache")
            with self.assertRaises(ValueError):
                fetch_sudachi_github(cache_dir=new_cache, url="https://example.com/small_synonyms.txt")

    def test_populate_sudachi_synonyms_merges_multiple_groups_for_same_term(self) -> None:
        """Verify populate_sudachi_synonyms merges targets across multiple groups up to 3 targets."""
        conn = sqlite3.connect(self.db_path)
        img1 = IrasutoyaImage("寝る人のイラスト", "https://1.bp.blogspot.com/s800/sleep.png", "t.png", "寝る", post_id="2")
        img2 = IrasutoyaImage("休む人のイラスト", "https://1.bp.blogspot.com/s800/rest.png", "t2.png", "休む", post_id="3")
        write_posts(
            conn,
            [
                ParsedPost("2", "寝る人のイラスト", "desc", "pub", "upd", (), [img1]),
                ParsedPost("3", "休む人のイラスト", "desc", "pub", "upd", (), [img2]),
            ],
        )
        conn.close()

        csv_content = (
            # Group 1: 就寝 -> 寝る
            "000001,1,0,1,0,0,0,(),就寝,,\n"
            "000001,1,0,2,0,0,0,(),寝る,,\n"
            # Group 2: 就寝 -> 休む
            "000002,1,0,1,0,0,0,(),就寝,,\n"
            "000002,1,0,2,0,0,0,(),休む,,\n"
        )
        csv_path = os.path.join(self.temp_dir.name, "multi_group.txt")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        self.assertTrue(populate_sudachi_synonyms(self.db_path, csv_path))

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT target FROM synonyms WHERE source = 'sudachi' AND term = '就寝' ORDER BY ord")
        targets = [r[0] for r in cur.fetchall()]
        conn.close()
        self.assertEqual(targets, ["寝る", "休む"])

    def test_populate_sudachi_synonyms_creates_schema_if_missing(self) -> None:
        """Verify populate_sudachi_synonyms succeeds when synonyms table is missing initially."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DROP TABLE IF EXISTS synonyms")
        conn.commit()
        conn.close()

        csv_path = os.path.join(self.temp_dir.name, "recreate_schema.txt")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("000001,1,0,1,0,0,0,(),イヌ,,\n000001,1,0,2,0,0,0,(),猫,,\n")

        self.assertTrue(populate_sudachi_synonyms(self.db_path, csv_path))

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target FROM synonyms WHERE source = 'sudachi'")
        rows = cur.fetchall()
        conn.close()
        self.assertEqual(rows, [("イヌ", "猫")])

    def test_clean_sudachi_headword_truncation_guards(self) -> None:
        """Verify clean_sudachi_headword rejects truncated terms and accepts intact ones."""
        # Truncated or structural changes rejected
        self.assertEqual(clean_sudachi_headword("ジョン・F・ケネディ"), "")
        self.assertEqual(clean_sudachi_headword("リンゴ / 蜜柑"), "")
        self.assertEqual(clean_sudachi_headword("赤、青"), "")
        self.assertEqual(clean_sudachi_headword(""), "")

        # Intact terms accepted with NFKC normalization
        self.assertEqual(clean_sudachi_headword("メッセージ・カード"), "メッセージ・カード")
        self.assertEqual(clean_sudachi_headword(" ＴＶ "), "TV")
        self.assertEqual(clean_sudachi_headword("就寝"), "就寝")
        self.assertEqual(clean_sudachi_headword("猫"), "猫")

    def test_fetch_sudachi_github_cache_path_includes_url_hash(self) -> None:
        """Verify fetch_sudachi_github incorporates URL hash in cache filename."""
        mock_csv = b"000001,1,0,1,0,0,0,(),sample,,\n" * 35000
        mock_resp = io.BytesIO(mock_csv)

        class MockUrlOpen:
            def __enter__(self) -> io.BytesIO:
                return mock_resp

            def __exit__(self, *args: object) -> None:
                pass

        url1 = "https://example.com/v1/synonyms.txt"
        url2 = "https://example.com/v2/synonyms.txt"
        url1_hash = hashlib.sha256(url1.encode("utf-8")).hexdigest()[:10]
        url2_hash = hashlib.sha256(url2.encode("utf-8")).hexdigest()[:10]

        def fake_urlopen(*args: object, **kwargs: object) -> MockUrlOpen:
            mock_resp.seek(0)
            return MockUrlOpen()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            cache_dir = os.path.join(self.temp_dir.name, "hash_cache")
            path1 = fetch_sudachi_github(cache_dir=cache_dir, url=url1)
            path2 = fetch_sudachi_github(cache_dir=cache_dir, url=url2)

            self.assertIn(f"sudachi_synonyms_{url1_hash}.txt", path1)
            self.assertIn(f"sudachi_synonyms_{url2_hash}.txt", path2)
            self.assertNotEqual(path1, path2)

    def test_fetch_jpdb_freq_github_uses_cache(self) -> None:
        """Verify fetch_jpdb_freq_github reuses cached file when available."""
        cache_dir = os.path.join(self.temp_dir.name, "jpdb_fetch_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(cache_dir, "jpdb_v2.2_freq_list_2024-10-13.csv")
        with open(cached_file, "wb") as f:
            f.write(b"term\treading\tfreq\tkana_freq\n" * 100)

        # Ensure no network request is made
        with patch("urllib.request.urlopen") as mock_url:
            path = fetch_jpdb_freq_github(cache_dir=cache_dir)
            self.assertEqual(path, os.path.abspath(cached_file))
            mock_url.assert_not_called()

    def test_populate_llm_concepts_success_and_sha256(self) -> None:
        """Verify populate_llm_concepts validates targets and writes rows with source 'llm_concept'."""
        json_path = os.path.join(self.temp_dir.name, "llm_concepts.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"concepts": {"テスト語": ["猫"]}}, f)

        ok = populate_llm_concepts(self.db_path, json_path, min_matched=1, strict=True)
        self.assertTrue(ok)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target, source FROM synonyms WHERE source = 'llm_concept'")
        rows = cur.fetchall()
        conn.close()
        self.assertEqual(rows, [("テスト語", "猫", "llm_concept")])

        sha_path = f"{self.db_path}.sha256"
        self.assertTrue(os.path.isfile(sha_path))

    def test_populate_concept_json_mapping_strict_vs_non_strict(self) -> None:
        """Verify _populate_concept_json_mapping handles strict vs CI-resilient 95% threshold."""
        # 19 valid targets ('猫') and 1 invalid target ('存在しないターゲット') = 95.0% valid
        concepts_95: Dict[str, List[str]] = {f"word_{i}": ["猫"] for i in range(19)}
        concepts_95["word_19"] = ["存在しないターゲット123"]

        json_path_95 = os.path.join(self.temp_dir.name, "concepts_95.json")
        with open(json_path_95, "w", encoding="utf-8") as f:
            json.dump({"concepts": concepts_95}, f)

        # strict=True fails on the 1 invalid target
        ok_strict = _populate_concept_json_mapping(
            catalog_path=self.db_path,
            json_path=json_path_95,
            source="test_source",
            min_matched=1,
            strict=True,
        )
        self.assertFalse(ok_strict)

        # strict=False succeeds with warning because valid ratio >= 95%
        ok_non_strict = _populate_concept_json_mapping(
            catalog_path=self.db_path,
            json_path=json_path_95,
            source="test_source",
            min_matched=1,
            strict=False,
        )
        self.assertTrue(ok_non_strict)

        # 18 valid targets ('猫') and 2 invalid targets = 18/20 = 90.0% (< 95%)
        concepts_90: Dict[str, List[str]] = {f"word_{i}": ["猫"] for i in range(18)}
        concepts_90["word_18"] = ["存在しないターゲット1"]
        concepts_90["word_19"] = ["存在しないターゲット2"]

        json_path_90 = os.path.join(self.temp_dir.name, "concepts_90.json")
        with open(json_path_90, "w", encoding="utf-8") as f:
            json.dump({"concepts": concepts_90}, f)

        # strict=False also fails when valid ratio < 95%
        ok_below_95 = _populate_concept_json_mapping(
            catalog_path=self.db_path,
            json_path=json_path_90,
            source="test_source_2",
            min_matched=1,
            strict=False,
        )
        self.assertFalse(ok_below_95)

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
        from scripts.build_synonyms import _parse_yomitan_metadata

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

        with patch("scripts.build_synonyms._fetch_yomitan_term_entries", side_effect=fake_fetch_entries):
            # non-tier1 mode: skips 'まで' (grammar) and '猫' (Tier 1 match), keeps '犬' and '傷薬'
            cands_non_tier1 = load_jpdb_candidates(
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
            cands_all = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                limit=10,
                mode="all",
                cache_dir=cache_dir,
            )
            terms_all = [c["term"] for c in cands_all]
            self.assertEqual(terms_all, ["猫", "犬", "傷薬"])

            # Verify that header was skipped without eating candidate limit slot (inspects top 3: まで, 猫, 犬)
            cands_limit_3 = load_jpdb_candidates(
                catalog_path=self.db_path,
                jpdb_path=tsv_path,
                limit=3,
                mode="all",
                cache_dir=cache_dir,
            )
            self.assertEqual([c["term"] for c in cands_limit_3], ["猫", "犬"])

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
                # Return tool call for search_catalog
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
                # Return final JSON answer
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
                    """Enter context manager and return self."""
                    return self

                def __exit__(self, *args: Any) -> None:
                    """Exit context manager."""
                    pass

                def read(self) -> bytes:
                    """Read response body bytes."""
                    return mock_b

            return MockChatResponse()

        with patch("urllib.request.urlopen", side_effect=fake_chat_urlopen):
            mappings = run_ollama_agent_batch(candidates, catalog_path=self.db_path)
            self.assertEqual(mappings, {"子猫": ["猫"]})

        # Test run_ollama_compile
        llm_out = os.path.join(self.temp_dir.name, "custom_llm_out.json")
        tsv_path = os.path.join(self.temp_dir.name, "compile_jpdb.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("子猫\tこねこ\t500\t\n")

        def fake_load_candidates(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            """Return static test candidates."""
            return candidates

        turn_count = 0
        with patch("scripts.build_synonyms.load_jpdb_candidates", side_effect=fake_load_candidates):
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
        with patch("scripts.build_synonyms.load_jpdb_candidates", side_effect=fake_load_candidates):
            with patch("scripts.build_synonyms.run_ollama_agent_batch") as mock_batch:
                ok_skip = run_ollama_compile(
                    catalog_path=self.db_path,
                    jpdb_path=tsv_path,
                    llm_out_path=llm_out,
                )
                self.assertTrue(ok_skip)
                mock_batch.assert_not_called()

        # Verify run_ollama_compile returns False when 0 candidates were compiled or rejected
        with patch("scripts.build_synonyms.load_jpdb_candidates", side_effect=fake_load_candidates):
            with patch("scripts.build_synonyms.run_ollama_agent_batch", return_value={}):
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

        def fake_load_abstract(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            """Return abstract candidate."""
            return abstract_candidates

        def fake_batch_reject(
            candidates: Any,
            *args: Any,
            on_reject: Optional[Callable[[str, str], None]] = None,
            **kwargs: Any,
        ) -> Dict[str, List[str]]:
            """Simulate model rejecting abstract concept."""
            if on_reject:
                on_reject("概念", "purely abstract concept")
            return {}

        rej_llm_out = os.path.join(self.temp_dir.name, "rej_llm_out.json")
        with patch("scripts.build_synonyms.load_jpdb_candidates", side_effect=fake_load_abstract):
            with patch("scripts.build_synonyms.run_ollama_agent_batch", side_effect=fake_batch_reject):
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

    def test_run_ollama_agent_batch_unparsable_response_does_not_reject(self) -> None:
        """Verify unparsable LLM response logs warning and does not invoke on_reject."""
        from unittest.mock import MagicMock

        from scripts.build_synonyms import run_ollama_agent_batch

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
            res = run_ollama_agent_batch(
                cands,
                catalog_path=self.db_path,
                on_reject=handle_reject,
            )
            self.assertEqual(res, {})
            self.assertEqual(rejections, [])

    def test_main_cli_llm_and_ollama_compile(self) -> None:
        """Verify main CLI dispatches --llm and --ollama-compile correctly."""
        llm_json = os.path.join(self.temp_dir.name, "main_llm.json")
        with open(llm_json, "w", encoding="utf-8") as f:
            json.dump({"concepts": {"テストCLI": ["猫"]}}, f)

        # Test --llm
        code = main(["--catalog", self.db_path, "--llm", llm_json])
        self.assertEqual(code, 0)

        # Test --ollama-compile dispatch
        with patch("scripts.build_synonyms.run_ollama_compile", return_value=True) as mock_compile:
            code_compile = main([
                "--catalog",
                self.db_path,
                "--ollama-compile",
                "--jpdb-limit",
                "100",
                "--jpdb-mode",
                "unmatched",
            ])
            self.assertEqual(code_compile, 0)
            mock_compile.assert_called_once()


if __name__ == "__main__":
    unittest.main()
