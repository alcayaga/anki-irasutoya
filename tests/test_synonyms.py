"""Unit tests for the SQLite-backed visual concept map and synonym loader."""
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from addon.catalog import CatalogStore, create_schema, write_synonyms
from addon.synonyms import (
    lookup_concept_map,
    lookup_synonyms,
)


class TestSynonyms(unittest.TestCase):
    """Test suite for synonym and visual concept map lookups from catalog.sqlite."""

    def setUp(self) -> None:
        """Create an isolated test catalog with manual concepts for hermetic testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_catalog.sqlite")
        conn = sqlite3.connect(self.db_path)
        create_schema(conn)
        conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '1')")
        write_synonyms(
            conn,
            source="manual",
            mapping={
                "亡くなる": ["お葬式", "墓"],
                "手に入れる": ["獲得", "宝箱"],
                "私": ["自己紹介", "女性"],
            },
        )
        conn.commit()
        conn.close()
        self.test_store = CatalogStore(self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary test catalog workspace."""
        self.temp_dir.cleanup()

    def test_lookup_concept_map_from_default_catalog(self) -> None:
        """Verify lookup_concept_map returns manual concept bridges from bundled catalog.sqlite."""
        with patch("addon.synonyms.default_store", return_value=self.test_store):
            targets = lookup_concept_map("亡くなる")
            self.assertEqual(targets, ["お葬式", "墓"])

            targets_space = lookup_concept_map("  私  ")
            self.assertEqual(targets_space, ["自己紹介", "女性"])

            self.assertEqual(lookup_concept_map(""), [])
            self.assertEqual(lookup_concept_map("nonexistent_word_12345"), [])

    def test_lookup_synonyms_with_source(self) -> None:
        """Verify lookup_synonyms filters by source identifier."""
        with patch("addon.synonyms.default_store", return_value=self.test_store):
            targets = lookup_synonyms("亡くなる", source="manual")
            self.assertEqual(targets, ["お葬式", "墓"])

            # Non-matching source returns empty list
            self.assertEqual(lookup_synonyms("亡くなる", source="sudachi"), [])

    def test_lookup_synonyms_custom_store(self) -> None:
        """Verify lookup_synonyms works with an isolated custom CatalogStore."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "custom_catalog.sqlite")
            conn = sqlite3.connect(db_path)
            create_schema(conn)
            conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '1')")
            write_synonyms(
                conn,
                source="manual",
                mapping={"テスト": ["試験", "実験"], "確認": ["チェック"]},
            )
            write_synonyms(
                conn,
                source="sudachi",
                mapping={"テスト": ["トライ"]},
            )
            conn.commit()
            conn.close()

            custom_store = CatalogStore(db_path)

            # Query all sources
            all_targets = lookup_synonyms("テスト", store=custom_store)
            self.assertIn("試験", all_targets)
            self.assertIn("実験", all_targets)
            self.assertIn("トライ", all_targets)

            # Filter by specific source
            manual_targets = lookup_synonyms("テスト", source="manual", store=custom_store)
            self.assertEqual(manual_targets, ["試験", "実験"])

            sudachi_targets = lookup_synonyms("テスト", source="sudachi", store=custom_store)
            self.assertEqual(sudachi_targets, ["トライ"])

            # Nonexistent term
            self.assertEqual(lookup_synonyms("存在しない", store=custom_store), [])

    def test_lookup_synonyms_missing_store(self) -> None:
        """Verify lookup_synonyms handles non-existent database file gracefully."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_db = os.path.join(tmp_dir, "missing.sqlite")
            missing_store = CatalogStore(missing_db)
            self.assertEqual(lookup_synonyms("亡くなる", store=missing_store), [])
