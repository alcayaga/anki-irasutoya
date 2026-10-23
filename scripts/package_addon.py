#!/usr/bin/env python3
"""Package the Anki Irasutoya add-on into a distribution-ready .ankiaddon archive."""
import argparse
import fnmatch
import json
import logging
import os
import sys
import tempfile
import zipfile
from typing import List, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from addon.catalog import CatalogStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("package_addon")

ADDON_DIR = os.path.join(REPO_ROOT, "addon")

# Strict allowlist of file patterns relative to addon/
ALLOWLIST_PATTERNS = [
    "*.py",
    "manifest.json",
    "config.json",
    "config.md",
    "icons/*",
    "icons/**/*",
    "data/catalog.sqlite",
]

# Patterns representing private add-on state that MUST reject packaging
FORBIDDEN_PATTERNS = [
    "meta.json",
    "user_files*",
]

# Patterns representing bytecode and OS artifacts to silently exclude
IGNORED_PATTERNS = [
    "__pycache__*",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "Thumbs.db",
]


def is_allowlisted(rel_path: str) -> bool:
    """Return True if relative path matches any allowlist pattern and no forbidden/ignored pattern."""
    rel_posix = rel_path.replace("\\", "/")
    fname = os.path.basename(rel_posix)
    if any(fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(fname, pat) for pat in FORBIDDEN_PATTERNS + IGNORED_PATTERNS):
        return False
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in ALLOWLIST_PATTERNS)


def collect_payload_files(addon_dir: str, catalog_path: str) -> List[Tuple[str, str]]:
    """Scan addon directory and collect allowlisted files as (abs_path, arcname) tuples."""
    payload: List[Tuple[str, str]] = []

    # Verify catalog exists
    if not os.path.isfile(catalog_path):
        raise FileNotFoundError(
            f"Catalog SQLite file not found at {catalog_path}. Build it first with scripts/build_catalog.py"
        )

    for root, dirs, files in os.walk(addon_dir):
        # Reject forbidden directory structures (like user_files/)
        for d in dirs:
            if any(fnmatch.fnmatch(d, p) for p in FORBIDDEN_PATTERNS):
                rel_d = os.path.relpath(os.path.join(root, d), addon_dir).replace("\\", "/")
                raise RuntimeError(f"Forbidden directory detected in addon directory: {rel_d}")

        # Silently prune bytecode and junk directories from traversal
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in IGNORED_PATTERNS)]

        for fname in files:
            abs_path = os.path.join(root, fname)
            rel_path = os.path.relpath(abs_path, addon_dir).replace("\\", "/")

            # Silently skip junk files (.DS_Store, *.pyc, etc.)
            if any(fnmatch.fnmatch(fname, p) or fnmatch.fnmatch(rel_path, p) for p in IGNORED_PATTERNS):
                continue

            # Reject forbidden private state files
            if any(fnmatch.fnmatch(fname, p) or fnmatch.fnmatch(rel_path, p) for p in FORBIDDEN_PATTERNS):
                raise RuntimeError(f"Forbidden file detected in addon directory: {rel_path}")

            if is_allowlisted(rel_path):
                payload.append((abs_path, rel_path))

    # Ensure explicit catalog_path is packaged as data/catalog.sqlite
    catalog_arcname = "data/catalog.sqlite"
    payload = [(p, a) for p, a in payload if a != catalog_arcname]
    payload.append((os.path.abspath(catalog_path), catalog_arcname))

    return payload


def package_addon(
    version: str,
    catalog_path: str,
    out_dir: str,
    addon_dir: str = ADDON_DIR,
) -> str:
    """Build .ankiaddon zip archive with stamped version and verified allowlist."""
    os.makedirs(out_dir, exist_ok=True)
    archive_name = f"anki-irasutoya-{version}.ankiaddon"
    out_path = os.path.join(out_dir, archive_name)

    logger.info("Collecting payload files from %s...", addon_dir)
    payload_files = collect_payload_files(addon_dir, catalog_path)

    arcnames: Set[str] = {arc for _, arc in payload_files}
    required_roots = {"__init__.py", "manifest.json", "config.json", "config.md", "data/catalog.sqlite"}
    missing = required_roots - arcnames
    if missing:
        raise RuntimeError(f"Required add-on payload files missing: {missing}")

    logger.info("Packaging %d files into %s...", len(payload_files), out_path)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for abs_path, arcname in payload_files:
            if arcname == "manifest.json":
                # Inject human_version into manifest.json
                with open(abs_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                manifest_data["human_version"] = version
                stamped_bytes = json.dumps(manifest_data, indent=2, ensure_ascii=False).encode("utf-8")
                zf.writestr("manifest.json", stamped_bytes)
            else:
                zf.write(abs_path, arcname)

    # Verification self-check
    logger.info("Executing self-check on packaged archive...")
    with zipfile.ZipFile(out_path, "r") as zf:
        namelist = zf.namelist()
        for name in namelist:
            if any(fnmatch.fnmatch(name, pat) for pat in FORBIDDEN_PATTERNS):
                raise RuntimeError(f"Self-check Failed: Forbidden file {name} found in archive")

        manifest_content = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest_content.get("human_version") != version:
            raise RuntimeError(
                f"Self-check Failed: Stamped version mismatch (expected {version}, got {manifest_content.get('human_version')})"
            )

        with tempfile.TemporaryDirectory() as tmp_extract:
            extracted_cat = os.path.join(tmp_extract, "extracted_catalog.sqlite")
            with open(extracted_cat, "wb") as f_cat:
                f_cat.write(zf.read("data/catalog.sqlite"))

            store = CatalogStore(extracted_cat)
            index = store.get_index()
            if index is None:
                raise RuntimeError("Self-check Failed: Packaged catalog could not be read by CatalogStore")
            logger.info("Self-check passed: Catalog loaded successfully (%d images)", len(index))

    logger.info("Successfully packaged add-on: %s (%.2f MB)", out_path, os.path.getsize(out_path) / (1024 * 1024))
    return out_path


def main() -> None:
    """CLI entrypoint for packaging the add-on."""
    parser = argparse.ArgumentParser(description="Package Anki Irasutoya add-on into .ankiaddon file.")
    parser.add_argument("--version", required=True, help="Release version string (e.g. 1.0.0)")
    parser.add_argument("--catalog", default=os.path.join(ADDON_DIR, "data", "catalog.sqlite"), help="Path to catalog.sqlite")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "dist"), help="Output directory for .ankiaddon")
    args = parser.parse_args()

    package_addon(
        version=args.version,
        catalog_path=args.catalog,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
