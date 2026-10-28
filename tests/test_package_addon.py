"""Unit tests for the add-on packager, allowlist filters, version injection, and integrity checks."""
import json
import os
import sqlite3
import tempfile
import unittest
import zipfile

from addon.catalog import write_posts
from addon.feed import IrasutoyaImage, ParsedPost
from scripts.package_addon import (
    is_allowlisted,
    package_addon,
)


class TestPackageAddon(unittest.TestCase):
    """Test suite for packaging .ankiaddon distribution archives."""

    def setUp(self):
        """Create mock addon directory structure and valid catalog."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_addon = os.path.join(self.temp_dir.name, "addon")
        os.makedirs(os.path.join(self.mock_addon, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.mock_addon, "icons"), exist_ok=True)
        self.out_dir = os.path.join(self.temp_dir.name, "dist")

        # Create mock payload files
        with open(os.path.join(self.mock_addon, "__init__.py"), "w") as f:
            f.write("# init\n")
        os.makedirs(os.path.join(self.mock_addon, "ui"), exist_ok=True)
        with open(os.path.join(self.mock_addon, "ui", "__init__.py"), "w") as f:
            f.write("# ui init\n")
        with open(os.path.join(self.mock_addon, "ui", "dialog.py"), "w") as f:
            f.write("# dialog\n")
        with open(os.path.join(self.mock_addon, "manifest.json"), "w") as f:
            json.dump({"name": "Anki Irasutoya", "package": "anki_irasutoya"}, f)
        with open(os.path.join(self.mock_addon, "config.json"), "w") as f:
            json.dump({"last_source_field": "Expression"}, f)
        with open(os.path.join(self.mock_addon, "config.md"), "w") as f:
            f.write("# Config\n")
        with open(os.path.join(self.mock_addon, "icons", "chevron.svg"), "w") as f:
            f.write("<svg></svg>\n")

        # Create mock catalog
        self.catalog_path = os.path.join(self.mock_addon, "data", "catalog.sqlite")
        conn = sqlite3.connect(self.catalog_path)
        img = IrasutoyaImage("猫", "https://1.bp.blogspot.com/cat.png", "t", "猫", post_id="1")
        post = ParsedPost("1", "猫", "desc", "pub", "upd", (), [img])
        write_posts(conn, [post])
        conn.close()

    def tearDown(self):
        """Clean up workspace."""
        self.temp_dir.cleanup()

    def test_allowlist_matching(self):
        """Verify is_allowlisted permits only valid distribution assets and rejects forbidden files."""
        self.assertTrue(is_allowlisted("__init__.py"))
        self.assertTrue(is_allowlisted("client.py"))
        self.assertTrue(is_allowlisted("ui/__init__.py"))
        self.assertTrue(is_allowlisted("ui/dialog.py"))
        self.assertTrue(is_allowlisted("manifest.json"))
        self.assertTrue(is_allowlisted("config.json"))
        self.assertTrue(is_allowlisted("config.md"))
        self.assertTrue(is_allowlisted("icons/chevron_down.svg"))
        self.assertTrue(is_allowlisted("data/catalog.sqlite"))

        self.assertFalse(is_allowlisted("data/concept_map.json"))
        self.assertFalse(is_allowlisted("meta.json"))
        self.assertFalse(is_allowlisted(".DS_Store"))
        self.assertFalse(is_allowlisted("client.pyc"))
        self.assertFalse(is_allowlisted("__pycache__/client.cpython-313.pyc"))
        self.assertFalse(is_allowlisted("user_files/prefs.db"))
        self.assertFalse(is_allowlisted("random_file.txt"))

    def test_package_addon_success_and_version_stamping(self):
        """Verify package_addon generates clean .ankiaddon archive and injects human_version."""
        out_archive = package_addon(
            version="1.5.0",
            catalog_path=self.catalog_path,
            out_dir=self.out_dir,
            addon_dir=self.mock_addon,
        )

        self.assertTrue(os.path.isfile(out_archive))
        self.assertTrue(out_archive.endswith("anki-irasutoya-1.5.0.ankiaddon"))

        with zipfile.ZipFile(out_archive, "r") as zf:
            names = set(zf.namelist())
            self.assertIn("manifest.json", names)
            self.assertIn("__init__.py", names)
            self.assertIn("ui/__init__.py", names)
            self.assertIn("ui/dialog.py", names)
            self.assertIn("data/catalog.sqlite", names)
            self.assertIn("icons/chevron.svg", names)

            # Check stamped human_version in manifest.json
            manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
            self.assertEqual(manifest_data.get("human_version"), "1.5.0")

    def test_package_addon_refuses_forbidden_meta_json(self):
        """Verify package_addon raises RuntimeError if meta.json or forbidden files are present."""
        forbidden_file = os.path.join(self.mock_addon, "meta.json")
        with open(forbidden_file, "w") as f:
            f.write("{}")

        with self.assertRaises(RuntimeError) as cm:
            package_addon(
                version="1.0.0",
                catalog_path=self.catalog_path,
                out_dir=self.out_dir,
                addon_dir=self.mock_addon,
            )
        self.assertIn("Forbidden file detected", str(cm.exception))

    def test_package_addon_refuses_forbidden_user_files_dir(self):
        """Verify package_addon raises RuntimeError if user_files directory is present."""
        user_files_dir = os.path.join(self.mock_addon, "user_files")
        os.makedirs(user_files_dir, exist_ok=True)

        with self.assertRaises(RuntimeError) as cm:
            package_addon(
                version="1.0.0",
                catalog_path=self.catalog_path,
                out_dir=self.out_dir,
                addon_dir=self.mock_addon,
            )
        self.assertIn("Forbidden directory detected", str(cm.exception))

    def test_package_addon_skips_os_and_bytecode_junk(self):
        """Verify package_addon silently excludes .DS_Store, *.pyc, and __pycache__ from archive."""
        with open(os.path.join(self.mock_addon, ".DS_Store"), "wb") as f:
            f.write(b"junk")
        with open(os.path.join(self.mock_addon, "client.pyc"), "wb") as f:
            f.write(b"bytecode")
        pycache_dir = os.path.join(self.mock_addon, "__pycache__")
        os.makedirs(pycache_dir, exist_ok=True)
        with open(os.path.join(pycache_dir, "test.cpython-313.pyc"), "wb") as f:
            f.write(b"bytecode")

        out_archive = package_addon(
            version="1.5.0",
            catalog_path=self.catalog_path,
            out_dir=self.out_dir,
            addon_dir=self.mock_addon,
        )
        with zipfile.ZipFile(out_archive, "r") as zf:
            names = set(zf.namelist())
            self.assertNotIn(".DS_Store", names)
            self.assertNotIn("client.pyc", names)
            self.assertFalse(any("__pycache__" in n for n in names))

    def test_package_addon_refuses_missing_catalog(self):
        """Verify package_addon raises FileNotFoundError if catalog.sqlite is missing."""
        missing_catalog = os.path.join(self.temp_dir.name, "missing.sqlite")
        with self.assertRaises(FileNotFoundError):
            package_addon(
                version="1.0.0",
                catalog_path=missing_catalog,
                out_dir=self.out_dir,
                addon_dir=self.mock_addon,
            )
