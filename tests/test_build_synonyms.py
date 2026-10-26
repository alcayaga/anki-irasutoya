"""Unit tests for the synonym build and validation CLI (scripts/build_synonyms.py)."""
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest

from addon.catalog import create_schema, write_posts, write_synonyms
from addon.feed import IrasutoyaImage, ParsedPost
from scripts.build_synonyms import populate_manual_concepts, validate_catalog_synonyms


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
        conn.close()
        self.assertTrue(validate_catalog_synonyms(fresh_db))

        # 2. Empty table
        conn = sqlite3.connect(fresh_db)
        create_schema(conn)
        conn.close()
        self.assertTrue(validate_catalog_synonyms(fresh_db))


if __name__ == "__main__":
    unittest.main()
