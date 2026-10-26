"""Unit tests for catalog builder scripts, fixture ingestion, keywords merge, and gates."""
import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import unittest

from addon.catalog import write_image_keywords, write_posts, write_synonyms
from addon.feed import IrasutoyaImage, ParsedPost
from scripts.build_catalog import (
    compare_with_old,
    compile_and_publish_catalog,
    is_sqlite_database,
    load_from_fixtures,
    merge_keywords,
    validate_catalog,
)


class TestBuildCatalog(unittest.TestCase):
    """Test suite for catalog builder logic, validation gates, and keyword merging."""

    def setUp(self):
        """Create temporary workspace for catalog tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fixtures_dir = os.path.join(self.temp_dir.name, "fixtures")
        os.makedirs(self.fixtures_dir, exist_ok=True)
        self.db_path = os.path.join(self.temp_dir.name, "test_catalog.sqlite")

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def _create_sample_fixture(self, fname: str = "feed.json"):
        """Write a sample feed JSON fixture file."""
        data = {
            "feed": {
                "openSearch$totalResults": {"$t": "2"},
                "entry": [
                    {
                        "id": {"$t": "tag:blogger.com,1999:blog-1.post-1111"},
                        "title": {"$t": "走る人のイラスト"},
                        "summary": {"$t": "走っている人のイラストです。"},
                        "published": {"$t": "2026-02-01T10:00:00Z"},
                        "updated": {"$t": "2026-02-01T12:00:00Z"},
                        "category": [{"term": "運動"}],
                        "content": {
                            "$t": '<a href="https://1.bp.blogspot.com/s800/run.png"><img alt="走る人" src="https://1.bp.blogspot.com/s200/run.png" /></a>'
                        },
                    },
                    {
                        "id": {"$t": "tag:blogger.com,1999:blog-1.post-2222"},
                        "title": {"$t": "猫のイラスト"},
                        "summary": {"$t": "可愛い猫のイラストです。"},
                        "published": {"$t": "2026-02-02T10:00:00Z"},
                        "updated": {"$t": "2026-02-02T12:00:00Z"},
                        "category": [{"term": "動物"}],
                        "content": {
                            "$t": '<a href="https://1.bp.blogspot.com/s800/cat.png"><img alt="猫" src="https://1.bp.blogspot.com/s200/cat.png" /></a>'
                        },
                    },
                ],
            }
        }
        path = os.path.join(self.fixtures_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def test_load_from_fixtures(self):
        """Verify load_from_fixtures correctly ingests and deduplicates feed entries."""
        self._create_sample_fixture("feed1.json")
        self._create_sample_fixture("feed2_duplicate.json")

        posts, total = load_from_fixtures(self.fixtures_dir)
        self.assertEqual(len(posts), 2)
        self.assertEqual(total, 2)
        self.assertEqual({p.blogger_id for p in posts}, {"1111", "2222"})

    def test_merge_keywords(self):
        """Verify merge_keywords inserts keyword mappings and reports unknown keys."""
        # Create database with one image
        conn = sqlite3.connect(self.db_path)
        img = IrasutoyaImage("猫", "https://1.bp.blogspot.com/s800/cat.png", "t.png", "猫", post_id="2222")
        post = ParsedPost("2222", "猫", "desc", "pub", "upd", (), [img])
        write_posts(conn, [post])

        # Create keywords directory with known and unknown keys
        kw_dir = os.path.join(self.temp_dir.name, "keywords")
        os.makedirs(kw_dir, exist_ok=True)
        kw_file = os.path.join(kw_dir, "test_kws.jsonl")
        with open(kw_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "2222/cat.png", "keywords": ["ねこ", "にゃんこ"], "source": "test", "weight": 1.5}) + "\n")
            f.write(json.dumps({"key": "unknown_id/unknown.png", "keywords": ["unknown"]}) + "\n")

        merge_keywords(conn, kw_dir)

        cursor = conn.cursor()
        cursor.execute("SELECT keyword, source, weight FROM image_keywords")
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 2)
        kws = {r[0] for r in rows}
        self.assertEqual(kws, {"ねこ", "にゃんこ"})
        self.assertEqual(rows[0][1], "test")
        self.assertEqual(rows[0][2], 1.5)
        conn.close()

    def test_compare_with_old(self):
        """Verify compare_with_old prints correct diff counts between catalog builds."""
        old_db = os.path.join(self.temp_dir.name, "old.sqlite")
        old_conn = sqlite3.connect(old_db)
        post_old = ParsedPost("1", "Title 1", "desc", "pub", "2026-01-01", (), [])
        write_posts(old_conn, [post_old])
        old_conn.close()

        new_conn = sqlite3.connect(self.db_path)
        post_new1 = ParsedPost("1", "Title 1", "desc", "pub", "2026-01-05", (), [])  # updated
        post_new2 = ParsedPost("2", "Title 2", "desc", "pub", "2026-01-05", (), [])  # added
        write_posts(new_conn, [post_new1, post_new2])

        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            compare_with_old(new_conn, old_db)
        out = buf.getvalue()
        self.assertIn("Posts added:   1", out)
        self.assertIn("Posts updated: 1", out)

        # Incompatible old catalog schema is non-fatal
        incompatible_db = os.path.join(self.temp_dir.name, "incompatible.sqlite")
        inc_conn = sqlite3.connect(incompatible_db)
        inc_conn.execute("CREATE TABLE foo (bar TEXT)")
        inc_conn.close()
        # Should not raise exception
        compare_with_old(new_conn, incompatible_db)

        new_conn.close()

    def test_validation_gates_success(self):
        """Verify validate_catalog succeeds when all gates are met."""
        conn = sqlite3.connect(self.db_path)
        img = IrasutoyaImage("猫", "https://1.bp.blogspot.com/s800/cat.png", "t.png", "猫", post_id="1")
        post = ParsedPost("1", "猫", "desc", "pub", "upd", (), [img])
        write_posts(conn, [post])
        conn.close()

        # Should pass with require_probes=False for small test catalog
        validate_catalog(self.db_path, total_results=1, require_probes=False, max_size_mb=40.0)

    def test_validation_gate_fails_on_post_count(self):
        """Verify validate_catalog raises RuntimeError if post count is below 98% threshold."""
        conn = sqlite3.connect(self.db_path)
        post = ParsedPost("1", "猫", "desc", "pub", "upd", (), [IrasutoyaImage("猫", "https://1.bp.blogspot.com/cat.png", "t", "猫")])
        write_posts(conn, [post])
        conn.close()

        with self.assertRaises(RuntimeError) as cm:
            validate_catalog(self.db_path, total_results=1000, require_probes=False)
        self.assertIn("Gate 2 Failed", str(cm.exception))

    def test_validation_gate_fails_on_size_limit(self):
        """Verify validate_catalog raises RuntimeError if file exceeds max size."""
        conn = sqlite3.connect(self.db_path)
        post = ParsedPost("1", "猫", "desc", "pub", "upd", (), [IrasutoyaImage("猫", "https://1.bp.blogspot.com/cat.png", "t", "猫")])
        write_posts(conn, [post])
        conn.close()

        with self.assertRaises(RuntimeError) as cm:
            # Set a tiny max size limit like 0.0001 MB
            validate_catalog(self.db_path, total_results=1, require_probes=False, max_size_mb=0.0001)
        self.assertIn("Gate 4 Failed", str(cm.exception))

    def test_atomic_temp_db_workflow(self):
        """Verify temporary database file is cleaned up if validation fails, leaving existing db intact."""
        # Create an existing good catalog
        with open(self.db_path, "w") as f:
            f.write("ORIGINAL_CONTENT")

        post = ParsedPost("1", "猫", "desc", "pub", "upd", (), [IrasutoyaImage("猫", "https://1.bp.blogspot.com/cat.png", "t", "猫")])

        # Gate 2 failure (expected 1000 posts, only 1 present)
        with self.assertRaises(RuntimeError):
            compile_and_publish_catalog(
                out_path=self.db_path,
                posts=[post],
                total_results=1000,
                require_probes=False,
            )

        # Original db content is untouched, tmp_path is removed
        tmp_files = [f for f in os.listdir(self.temp_dir.name) if f.endswith(".tmp")]
        self.assertEqual(tmp_files, [])
        with open(self.db_path, "r") as f:
            self.assertEqual(f.read(), "ORIGINAL_CONTENT")

    def test_compile_and_publish_catalog_success(self):
        """Verify compile_and_publish_catalog atomically replaces existing catalog and creates valid sha256 sidecar."""
        # Pre-create an old file to ensure atomic replacement
        with open(self.db_path, "w") as f:
            f.write("OLD_CATALOG_CONTENT")

        post = ParsedPost("1", "猫", "desc", "pub", "upd", (), [IrasutoyaImage("猫", "https://1.bp.blogspot.com/cat.png", "t", "猫")])

        published_path = compile_and_publish_catalog(
            out_path=self.db_path,
            posts=[post],
            total_results=1,
            require_probes=False,
        )

        self.assertEqual(published_path, self.db_path)
        self.assertTrue(os.path.exists(self.db_path))
        self.assertEqual(stat.S_IMODE(os.stat(self.db_path).st_mode), 0o644)

        # Ensure temp file was cleaned up / replaced
        tmp_files = [f for f in os.listdir(self.temp_dir.name) if f.endswith(".tmp")]
        self.assertEqual(tmp_files, [])

        # Check published SQLite database integrity and contents
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM posts")
        self.assertEqual(cur.fetchone()[0], 1)
        conn.close()

        # Check sha256 sidecar
        sha_path = f"{self.db_path}.sha256"
        self.assertTrue(os.path.exists(sha_path))

        hasher = hashlib.sha256()
        with open(self.db_path, "rb") as f:
            hasher.update(f.read())
        expected_digest = hasher.hexdigest()

        with open(sha_path, "r", encoding="utf-8") as f:
            sha_content = f.read()

        basename = os.path.basename(self.db_path)
        self.assertEqual(sha_content, f"{expected_digest}  {basename}\n")

    def test_compile_and_publish_catalog_preserves_synonyms(self) -> None:
        """Verify rebuild preserves existing synonyms without sqlite transaction or VACUUM conflicts."""
        post1 = ParsedPost("1", "猫", "desc", "pub", "upd", (), [IrasutoyaImage("猫", "https://1.bp.blogspot.com/cat.png", "t", "猫")])
        compile_and_publish_catalog(
            out_path=self.db_path,
            posts=[post1],
            total_results=1,
            require_probes=False,
        )

        # Inject synonyms into published catalog
        conn = sqlite3.connect(self.db_path)
        write_synonyms(conn, "manual", {"headword": ["target1", "target2"]})
        conn.close()

        # Rebuild catalog on top of existing catalog containing synonyms
        post2 = ParsedPost("2", "犬", "desc", "pub", "upd", (), [IrasutoyaImage("犬", "https://1.bp.blogspot.com/dog.png", "t", "犬")])
        compile_and_publish_catalog(
            out_path=self.db_path,
            posts=[post2],
            total_results=1,
            require_probes=False,
        )

        # Confirm new post was written, synonyms were preserved, and meta was kept
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT blogger_id FROM posts")
        self.assertEqual(cur.fetchone()[0], "2")

        cur.execute("SELECT term, target, source, ord FROM synonyms ORDER BY ord")
        rows = cur.fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ("headword", "target1", "manual", 0))
        self.assertEqual(rows[1], ("headword", "target2", "manual", 1))

        cur.execute("SELECT value FROM meta WHERE key = 'synonym_sources'")
        meta_row = cur.fetchone()
        self.assertIsNotNone(meta_row)
        self.assertEqual(meta_row[0], "manual")
        conn.close()

    def test_compile_and_publish_catalog_preserves_image_keywords(self) -> None:
        """Verify rebuild preserves existing image_keywords mapped by image key without transaction conflicts."""
        post1 = ParsedPost("1", "猫", "desc", "pub", "upd", (), [IrasutoyaImage("猫", "https://1.bp.blogspot.com/cat.png", "t", "猫")])
        compile_and_publish_catalog(
            out_path=self.db_path,
            posts=[post1],
            total_results=1,
            require_probes=False,
        )

        # Inject image keyword into published catalog for image_id 1
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id FROM images WHERE key = '1/cat.png'")
        img_id = cur.fetchone()[0]
        write_image_keywords(conn, "donkuri/kaishi", [(img_id, "ネコ", 1.0)])
        conn.close()

        # Create a fresh keywords directory with an updated weight for the same keyword
        kw_dir = os.path.join(self.temp_dir.name, "fresh_kws")
        os.makedirs(kw_dir, exist_ok=True)
        with open(os.path.join(kw_dir, "test.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "1/cat.png", "keywords": ["ネコ"], "source": "donkuri/kaishi", "weight": 2.5}) + "\n")

        # Rebuild catalog containing post1 and a new post2 with freshly merged keywords
        post2 = ParsedPost("2", "犬", "desc", "pub", "upd", (), [IrasutoyaImage("犬", "https://1.bp.blogspot.com/dog.png", "t", "犬")])
        compile_and_publish_catalog(
            out_path=self.db_path,
            posts=[post1, post2],
            total_results=2,
            keywords_dir=kw_dir,
            require_probes=False,
        )

        # Confirm image keywords were preserved and freshly merged weight (2.5) took priority over old (1.0)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT ik.keyword, ik.source, ik.weight, i.key FROM image_keywords ik JOIN images i ON ik.image_id = i.id")
        rows = cur.fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], ("ネコ", "donkuri/kaishi", 2.5, "1/cat.png"))

        cur.execute("SELECT value FROM meta WHERE key = 'keyword_sources'")
        meta_row = cur.fetchone()
        self.assertIsNotNone(meta_row)
        self.assertEqual(meta_row[0], "donkuri/kaishi")
        conn.close()


    def test_compile_and_publish_catalog_raises_on_synonym_carryover_error(self) -> None:
        """Verify compile_and_publish_catalog raises when synonym carryover encounters an error."""
        post1 = ParsedPost("1", "猫", "desc", "pub", "upd", (), [IrasutoyaImage("猫", "https://1.bp.blogspot.com/cat.png", "t", "猫")])
        compile_and_publish_catalog(
            out_path=self.db_path,
            posts=[post1],
            total_results=1,
            require_probes=False,
        )

        # Break the synonyms table in the previous catalog with an incompatible schema
        conn = sqlite3.connect(self.db_path)
        conn.execute("DROP TABLE IF EXISTS synonyms")
        conn.execute("CREATE TABLE synonyms (incompatible_col INT)")
        conn.commit()
        conn.close()

        with self.assertRaises(sqlite3.OperationalError):
            compile_and_publish_catalog(
                out_path=self.db_path,
                posts=[post1],
                total_results=1,
                require_probes=False,
            )

        # Confirm temporary database was cleaned up
        tmp_files = [f for f in os.listdir(self.temp_dir.name) if f.endswith(".tmp")]
        self.assertEqual(tmp_files, [])

    def test_is_sqlite_database_validation(self) -> None:
        """Verify is_sqlite_database validates intact database and rejects truncated/corrupt header-only files."""
        # Non-existent file
        self.assertFalse(is_sqlite_database(os.path.join(self.temp_dir.name, "missing.sqlite")))

        # Header-only file (valid 16-byte magic header, but corrupt/incomplete SQLite database)
        header_only = os.path.join(self.temp_dir.name, "header_only.sqlite")
        with open(header_only, "wb") as f:
            f.write(b"SQLite format 3\x00")
        self.assertFalse(is_sqlite_database(header_only))

        # Intact database
        valid_db = os.path.join(self.temp_dir.name, "valid.sqlite")
        conn = sqlite3.connect(valid_db)
        conn.execute("CREATE TABLE t (id INT)")
        conn.commit()
        conn.close()
        self.assertTrue(is_sqlite_database(valid_db))

    def test_compile_and_publish_catalog_falls_back_to_compare_with_on_corrupt_out_path(self) -> None:
        """Verify compile_and_publish_catalog falls back to compare_with when out_path has corrupt header-only file."""
        # Create a valid comparison database with synonyms
        compare_db = os.path.join(self.temp_dir.name, "compare.sqlite")
        post1 = ParsedPost("1", "猫", "desc", "pub", "upd", (), [IrasutoyaImage("猫", "https://1.bp.blogspot.com/cat.png", "t", "猫")])
        compile_and_publish_catalog(
            out_path=compare_db,
            posts=[post1],
            total_results=1,
            require_probes=False,
        )
        conn = sqlite3.connect(compare_db)
        write_synonyms(conn, source="manual", mapping={"ネコ": ["猫"]})
        conn.close()

        # Create header-only corrupt file at out_path
        with open(self.db_path, "wb") as f:
            f.write(b"SQLite format 3\x00")

        # Rebuild targeting out_path with compare_with=compare_db
        post2 = ParsedPost("2", "犬", "desc", "pub", "upd", (), [IrasutoyaImage("犬", "https://1.bp.blogspot.com/dog.png", "t", "犬")])
        compile_and_publish_catalog(
            out_path=self.db_path,
            posts=[post2],
            total_results=1,
            compare_with=compare_db,
            require_probes=False,
        )

        # Check synonyms carried over from compare_db
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT term, target, source FROM synonyms")
        rows = cur.fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], ("ネコ", "猫", "manual"))
