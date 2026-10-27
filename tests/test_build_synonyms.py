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
    fetch_kaishi_github_slice,
    main,
    populate_kaishi_concepts,
    populate_manual_concepts,
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


if __name__ == "__main__":
    unittest.main()
