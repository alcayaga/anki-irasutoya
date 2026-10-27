"""Unit tests for the SQLite-backed visual concept map and synonym loader."""
import os
import sqlite3
import tempfile
import unittest
import zipfile
from typing import Any
from unittest.mock import patch

from addon.catalog import (
    CatalogStore,
    create_schema,
    write_image_keywords,
    write_posts,
    write_synonyms,
)
from addon.feed import IrasutoyaImage, ParsedPost
from addon.synonyms import (
    lookup_concept_map,
    lookup_kaishi_concept,
    lookup_synonyms,
)
from scripts.build_synonyms import (
    clean_alt_text,
    extract_collection_from_apkg,
    normalize_stem,
    populate_kaishi_concepts,
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

    def test_lookup_kaishi_concept_source_filtering(self) -> None:
        """Verify lookup_kaishi_concept specifically filters for donkuri/kaishi source mappings."""
        conn = sqlite3.connect(self.db_path)
        write_synonyms(
            conn,
            source="donkuri/kaishi",
            mapping={"私": ["自己紹介"], "今": ["時計"]},
        )
        conn.close()

        with patch("addon.synonyms.default_store", return_value=self.test_store):
            self.assertEqual(lookup_kaishi_concept("私"), ["自己紹介"])
            self.assertEqual(lookup_kaishi_concept("今"), ["時計"])
            self.assertEqual(lookup_kaishi_concept("亡くなる"), [])

    def test_write_image_keywords_and_meta_stamp(self) -> None:
        """Verify write_image_keywords stores image keywords and stamps keyword_sources in meta."""
        conn = sqlite3.connect(self.db_path)
        written = write_image_keywords(
            conn,
            source="donkuri/kaishi",
            keywords=[(1, "猫", 1.0), (1, "ネコ", 0.8), (2, "犬", 1.0)],
        )
        self.assertEqual(written, 3)

        cur = conn.cursor()
        cur.execute("SELECT image_id, keyword, source, weight FROM image_keywords ORDER BY image_id, keyword")
        rows = cur.fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], (1, "ネコ", "donkuri/kaishi", 0.8))
        self.assertEqual(rows[1], (1, "猫", "donkuri/kaishi", 1.0))
        self.assertEqual(rows[2], (2, "犬", "donkuri/kaishi", 1.0))

        cur.execute("SELECT value FROM meta WHERE key = 'keyword_sources'")
        meta_row = cur.fetchone()
        self.assertIsNotNone(meta_row)
        self.assertEqual(meta_row[0], "donkuri/kaishi")
        conn.close()

    def test_normalize_stem_and_clean_alt(self) -> None:
        """Verify normalize_stem and clean_alt_text handle repeated extensions and media hashes."""
        self.assertEqual(
            normalize_stem("jikosyoukai_man-0f017c07b9f1048ff29830827e8503a6984504f6.webp"),
            "jikosyoukai_man",
        )
        self.assertEqual(
            normalize_stem("jitensya_rensyu_boy.png.png"),
            "jitensya_rensyu_boy",
        )
        self.assertEqual(
            normalize_stem("pose_touch_hitosashiyubi_man.webp"),
            "pose_touch_hitosashiyubi_man",
        )
        self.assertEqual(
            clean_alt_text("自己紹介のイラスト（男性） | かわいいフリー素材集 いらすとや"),
            "自己紹介のイラスト（男性）",
        )
        self.assertEqual(
            clean_alt_text("人差し指で押す人のイラスト（男性）"),
            "人差し指で押す人のイラスト（男性）",
        )

    def test_extract_collection_from_apkg_and_populate_kaishi(self) -> None:
        """Hermetically test collection extraction from synthetic .apkg and Kaishi concept population."""
        import zipfile

        # 1. Create a synthetic Kaishi SQLite collection with 2 notes
        kaishi_raw_db = os.path.join(self.temp_dir.name, "raw_kaishi.sqlite")
        k_conn = sqlite3.connect(kaishi_raw_db)
        k_conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, flds TEXT NOT NULL)")
        # Note 1: '自己紹介' word matching cat.png
        # Note 2: '走る' word matching runner.png
        k_conn.execute(
            "INSERT INTO notes (flds) VALUES (?)",
            ("自己紹介\x1fじこしょうかい\x1f<img src='jiko_man-12345678abcdef01.webp'>",),
        )
        k_conn.execute(
            "INSERT INTO notes (flds) VALUES (?)",
            ("全力\x1fぜんりょく\x1f<img alt='走る人のイラスト' src='runner.png'>",),
        )
        k_conn.commit()
        k_conn.close()

        # 2. Package into a synthetic .apkg (ZIP with collection.anki21)
        apkg_path = os.path.join(self.temp_dir.name, "test_deck.apkg")
        with open(kaishi_raw_db, "rb") as f_in:
            db_bytes = f_in.read()
        with zipfile.ZipFile(apkg_path, "w") as z:
            z.writestr("collection.anki21", db_bytes)

        # 3. Create a test catalog with matching illustrations
        catalog_path = os.path.join(self.temp_dir.name, "hermetic_catalog.sqlite")
        cat_conn = sqlite3.connect(catalog_path)
        img1 = IrasutoyaImage("自己紹介のイラスト（男性）", "https://bp.blogspot.com/s0/jiko_man.png", "t", "自己紹介")
        img2 = IrasutoyaImage("走る人のイラスト", "https://bp.blogspot.com/s0/runner.png", "t", "走る人")
        p1 = ParsedPost("101", "自己紹介のイラスト（男性）", "desc", "pub", "upd", (), [img1])
        p2 = ParsedPost("102", "走る人のイラスト", "desc", "pub", "upd", (), [img2])
        write_posts(cat_conn, [p1, p2])
        cat_conn.close()

        # 4. Extract from apkg and populate
        extracted_db = extract_collection_from_apkg(apkg_path, cache_dir=self.temp_dir.name)
        self.assertTrue(os.path.isfile(extracted_db))

        success = populate_kaishi_concepts(catalog_path, extracted_db, min_matched=2)
        self.assertTrue(success)

        # 5. Verify database contents
        verify_conn = sqlite3.connect(catalog_path)
        cur = verify_conn.cursor()
        cur.execute("SELECT term, target, source FROM synonyms WHERE source = 'donkuri/kaishi'")
        syn_rows = cur.fetchall()
        # '自己紹介' directly matches title '自己紹介のイラスト' -> skipped
        # '全力' does not match title '走る人のイラスト' -> target '走る人' populated!
        self.assertEqual(len(syn_rows), 1)
        self.assertEqual(syn_rows[0], ("全力", "走る人", "donkuri/kaishi"))

        cur.execute("SELECT keyword, source FROM image_keywords WHERE source = 'donkuri/kaishi' ORDER BY keyword")
        kw_rows = cur.fetchall()
        self.assertEqual(len(kw_rows), 2)
        self.assertEqual(kw_rows[0], ("全力", "donkuri/kaishi"))
        self.assertEqual(kw_rows[1], ("自己紹介", "donkuri/kaishi"))

        cur.execute("SELECT value FROM meta WHERE key = 'keyword_sources'")
        self.assertEqual(cur.fetchone()[0], "donkuri/kaishi")
        cur.execute("SELECT value FROM meta WHERE key = 'synonym_sources'")
        self.assertEqual(cur.fetchone()[0], "donkuri/kaishi")
        verify_conn.close()

    def test_extract_collection_from_apkg_rejects_oversized_anki21b(self) -> None:
        """Verify extract_collection_from_apkg rejects collection.anki21b exceeding MAX_COLLECTION_SIZE."""
        apkg_path = os.path.join(self.temp_dir.name, "oversized.apkg")
        with zipfile.ZipFile(apkg_path, "w") as z:
            z.writestr("collection.anki21b", b"small_compressed_data")

        class DummyInfo:
            file_size: int = 60 * 1024 * 1024

        with patch.object(zipfile.ZipFile, "getinfo", return_value=DummyInfo()):
            with self.assertRaises(ValueError) as cm:
                extract_collection_from_apkg(apkg_path, cache_dir=self.temp_dir.name)
            self.assertIn("exceeds the 50 MB limit", str(cm.exception))

    def test_populate_kaishi_concepts_zero_matches_refuses_overwrite(self) -> None:
        """Verify populate_kaishi_concepts returns False and leaves catalog untouched when 0 notes match."""
        empty_kaishi_db = os.path.join(self.temp_dir.name, "empty_kaishi.sqlite")
        conn = sqlite3.connect(empty_kaishi_db)
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, flds TEXT NOT NULL)")
        conn.execute("INSERT INTO notes (flds) VALUES (?)", ("言葉\x1fことば\x1f<img src='unmatched_xyz.png'>",))
        conn.commit()
        conn.close()

        catalog_path = os.path.join(self.temp_dir.name, "unmatched_catalog.sqlite")
        cat_conn = sqlite3.connect(catalog_path)
        img = IrasutoyaImage("猫のイラスト", "https://bp.blogspot.com/s0/cat.png", "t", "猫")
        p = ParsedPost("101", "猫のイラスト", "desc", "pub", "upd", (), [img])
        write_posts(cat_conn, [p])
        cat_conn.close()

        success = populate_kaishi_concepts(catalog_path, empty_kaishi_db)
        self.assertFalse(success)

    def test_populate_kaishi_concepts_creates_schema_on_fresh_catalog(self) -> None:
        """Verify populate_kaishi_concepts succeeds on a fresh catalog without pre-existing synonym tables."""
        catalog_path = os.path.join(self.temp_dir.name, "fresh_schema_catalog.sqlite")
        cat_conn = sqlite3.connect(catalog_path)
        img = IrasutoyaImage("猫のイラスト", "https://bp.blogspot.com/s0/cat.png", "t", "猫")
        p = ParsedPost("101", "猫のイラスト", "desc", "pub", "upd", (), [img])
        write_posts(cat_conn, [p])
        cat_conn.execute("DROP TABLE IF EXISTS synonyms")
        cat_conn.execute("DROP TABLE IF EXISTS image_keywords")
        cat_conn.commit()
        cat_conn.close()

        kaishi_db = os.path.join(self.temp_dir.name, "fresh_schema_kaishi.sqlite")
        conn = sqlite3.connect(kaishi_db)
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, flds TEXT NOT NULL)")
        conn.execute("INSERT INTO notes (flds) VALUES (?)", ("キャット\x1fきゃっと\x1f<img src='cat.png'>",))
        conn.commit()
        conn.close()

        success = populate_kaishi_concepts(catalog_path, kaishi_db, min_matched=1)
        self.assertTrue(success)

        v_conn = sqlite3.connect(catalog_path)
        cur = v_conn.cursor()
        cur.execute("SELECT count(*) FROM synonyms WHERE source = 'donkuri/kaishi'")
        self.assertEqual(cur.fetchone()[0], 1)
        cur.execute("SELECT count(*) FROM image_keywords WHERE source = 'donkuri/kaishi'")
        self.assertEqual(cur.fetchone()[0], 1)
        v_conn.close()

    def test_populate_kaishi_concepts_incomplete_collection_refuses_replacement(self) -> None:
        """Verify populate_kaishi_concepts refuses to replace an existing full collection with a partial one."""
        catalog_path = os.path.join(self.temp_dir.name, "partial_replace_catalog.sqlite")
        cat_conn = sqlite3.connect(catalog_path)
        img = IrasutoyaImage("猫のイラスト", "https://bp.blogspot.com/s0/cat.png", "t", "猫")
        p = ParsedPost("101", "猫のイラスト", "desc", "pub", "upd", (), [img])
        write_posts(cat_conn, [p])
        # Seed 10 existing keywords and 10 existing synonyms for donkuri/kaishi
        seed_kws = [(1, f"kw_{i}", 1.0) for i in range(10)]
        write_image_keywords(cat_conn, source="donkuri/kaishi", keywords=seed_kws)
        write_synonyms(cat_conn, source="donkuri/kaishi", mapping={f"term_{i}": [f"tgt_{i}"] for i in range(10)})
        cat_conn.close()

        # Create Kaishi DB with only 1 matching note
        small_kaishi_db = os.path.join(self.temp_dir.name, "small_kaishi.sqlite")
        conn = sqlite3.connect(small_kaishi_db)
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, flds TEXT NOT NULL)")
        conn.execute("INSERT INTO notes (flds) VALUES (?)", ("猫\x1fねこ\x1f<img src='cat.png'>",))
        conn.commit()
        conn.close()

        # Refuses replacement because 1 is significantly fewer than 10 (< 80%) even when min_matched=1
        success = populate_kaishi_concepts(catalog_path, small_kaishi_db, min_matched=1)
        self.assertFalse(success)

        # Confirm original 10 keywords and synonyms are preserved
        v_conn = sqlite3.connect(catalog_path)
        cur = v_conn.cursor()
        cur.execute("SELECT count(*) FROM image_keywords WHERE source = 'donkuri/kaishi'")
        self.assertEqual(cur.fetchone()[0], 10)
        cur.execute("SELECT count(*) FROM synonyms WHERE source = 'donkuri/kaishi'")
        self.assertEqual(cur.fetchone()[0], 10)
        v_conn.close()

    def test_populate_kaishi_concepts_atomic_rollback_on_write_error(self) -> None:
        """Verify transaction rolls back atomically if write_image_keywords fails."""
        catalog_path = os.path.join(self.temp_dir.name, "rollback_catalog.sqlite")
        cat_conn = sqlite3.connect(catalog_path)
        img = IrasutoyaImage("猫のイラスト", "https://bp.blogspot.com/s0/cat.png", "t", "猫")
        p = ParsedPost("101", "猫のイラスト", "desc", "pub", "upd", (), [img])
        write_posts(cat_conn, [p])
        # Add a trigger on image_keywords that aborts insertions after synonyms have succeeded
        cat_conn.execute(
            """
            CREATE TRIGGER fail_keywords BEFORE INSERT ON image_keywords
            BEGIN
                SELECT RAISE(ABORT, 'Simulated failure during image keyword insert');
            END;
            """
        )
        cat_conn.commit()
        cat_conn.close()

        small_kaishi_db = os.path.join(self.temp_dir.name, "rollback_kaishi.sqlite")
        conn = sqlite3.connect(small_kaishi_db)
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, flds TEXT NOT NULL)")
        conn.execute("INSERT INTO notes (flds) VALUES (?)", ("キャット\x1fきゃっと\x1f<img src='cat.png'>",))
        conn.commit()
        conn.close()

        with patch("scripts.build_synonyms.update_catalog_sha256") as mock_sha:
            success = populate_kaishi_concepts(catalog_path, small_kaishi_db, min_matched=1)
            self.assertFalse(success)
            mock_sha.assert_not_called()

        # Confirm synonyms table was rolled back and is empty
        v_conn = sqlite3.connect(catalog_path)
        cur = v_conn.cursor()
        cur.execute("SELECT count(*) FROM synonyms WHERE source = 'donkuri/kaishi'")
        self.assertEqual(cur.fetchone()[0], 0)
        cur.execute("SELECT count(*) FROM image_keywords WHERE source = 'donkuri/kaishi'")
        self.assertEqual(cur.fetchone()[0], 0)
        v_conn.close()

    def test_populate_kaishi_concepts_refreshes_sha_if_commit_succeeds_but_vacuum_fails(self) -> None:
        """Verify checksum is refreshed when Kaishi transaction commits but subsequent vacuum fails."""
        catalog_path = os.path.join(self.temp_dir.name, "vacuum_fail_catalog.sqlite")
        cat_conn = sqlite3.connect(catalog_path)
        img = IrasutoyaImage("猫のイラスト", "https://bp.blogspot.com/s0/cat.png", "t", "猫")
        p = ParsedPost("101", "猫のイラスト", "desc", "pub", "upd", (), [img])
        write_posts(cat_conn, [p])
        cat_conn.commit()
        cat_conn.close()

        kaishi_db = os.path.join(self.temp_dir.name, "vacuum_fail_kaishi.sqlite")
        conn = sqlite3.connect(kaishi_db)
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, flds TEXT NOT NULL)")
        conn.execute("INSERT INTO notes (flds) VALUES (?)", ("キャット\x1fきゃっと\x1f<img src='cat.png'>",))
        conn.commit()
        conn.close()

        real_connect = sqlite3.connect

        class FailingVacuumConnection(sqlite3.Connection):
            def execute(self, sql: str, *parameters: Any) -> sqlite3.Cursor:
                if "VACUUM" in sql.upper():
                    raise sqlite3.OperationalError("Simulated disk full during VACUUM")
                return super().execute(sql, *parameters)

        def fake_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            if args and "vacuum_fail_catalog" in str(args[0]):
                kwargs["factory"] = FailingVacuumConnection
            return real_connect(*args, **kwargs)

        with patch("sqlite3.connect", side_effect=fake_connect):
            with patch("scripts.build_synonyms.update_catalog_sha256") as mock_sha:
                success = populate_kaishi_concepts(catalog_path, kaishi_db, min_matched=1)
                self.assertFalse(success)
                mock_sha.assert_called_once_with(catalog_path)

        # Confirm that rows were committed to disk despite the vacuum failure
        v_conn = sqlite3.connect(catalog_path)
        cur = v_conn.cursor()
        cur.execute("SELECT count(*) FROM synonyms WHERE source = 'donkuri/kaishi'")
        self.assertEqual(cur.fetchone()[0], 1)
        cur.execute("SELECT count(*) FROM image_keywords WHERE source = 'donkuri/kaishi'")
        self.assertEqual(cur.fetchone()[0], 1)
        v_conn.close()


