"""Unit tests for the synonym build and validation CLI (scripts/build_synonyms.py)."""
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from addon.catalog import create_schema, write_posts, write_synonyms
from addon.feed import IrasutoyaImage, ParsedPost
from scripts.build_synonyms import (
    clean_sudachi_headword,
    fetch_kaishi_github_slice,
    fetch_sudachi_github,
    main,
    populate_kaishi_concepts,
    populate_manual_concepts,
    populate_sudachi_synonyms,
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
        img = IrasutoyaImage("可愛い猫", "https://1.bp.blogspot.com/s800/cat.png", "t.png", "猫", post_id="1")
        post = ParsedPost("1", "猫のイラスト", "desc", "pub", "upd", (), [img])
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
        # Add posts with illustrations for targets '寝る' and '買う'
        conn = sqlite3.connect(self.db_path)
        img1 = IrasutoyaImage("寝る人のイラスト", "https://1.bp.blogspot.com/s800/sleep.png", "t1.png", "寝る", post_id="2")
        post1 = ParsedPost("2", "睡眠のイラスト", "desc", "pub", "upd", (), [img1])
        img2 = IrasutoyaImage("買う人のイラスト", "https://1.bp.blogspot.com/s800/buy.png", "t2.png", "買う", post_id="3")
        post2 = ParsedPost("3", "買う人のイラスト", "desc", "pub", "upd", (), [img2])
        write_posts(conn, [post1, post2])
        conn.close()

        csv_content = (
            "000016,1,0,1,0,0,0,(),購入,,\n"
            "000016,2,0,3,0,0,0,(),買う,,\n"
            "023242,1,0,1,0,0,0,(),就寝,,\n"
            "023242,2,0,4,0,0,0,(),寝る,,\n"
        )
        csv_path = os.path.join(self.temp_dir.name, "test_sudachi.txt")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        success = populate_sudachi_synonyms(self.db_path, csv_path)
        self.assertTrue(success)

        # Verify rows in SQLite
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target, source, ord FROM synonyms WHERE source = 'sudachi' ORDER BY term")
        rows = cur.fetchall()
        cur.execute("SELECT value FROM meta WHERE key = 'synonym_sources'")
        meta_sources = cur.fetchone()[0]
        conn.close()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ("就寝", "寝る", "sudachi", 0))
        self.assertEqual(rows[1], ("購入", "買う", "sudachi", 0))
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
        cur.execute("SELECT count(*) FROM synonyms WHERE source = 'sudachi'")
        self.assertEqual(cur.fetchone()[0], 0)
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


if __name__ == "__main__":
    unittest.main()
