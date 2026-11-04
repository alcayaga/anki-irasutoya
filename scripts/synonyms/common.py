"""Shared constants and helper functions for synonym building and validation."""
import hashlib
import logging
import os
import sys
import urllib.parse
import urllib.request
from typing import List, Sequence

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from addon.feed import IrasutoyaImage  # noqa: E402
from addon.ranking import is_decorative_asset, is_image_relevant  # noqa: E402

logger = logging.getLogger("synonyms")

DEFAULT_CATALOG_PATH = os.path.join(REPO_ROOT, "addon", "data", "catalog.sqlite")
DEFAULT_BILINGUAL_CONCEPTS_PATH = os.path.join(REPO_ROOT, "scripts", "sources", "bilingual_concepts.json")
DEFAULT_LLM_CONCEPTS_PATH = os.path.join(REPO_ROOT, "scripts", "sources", "llm_concepts.json")


def filter_relevant_illustrations(
    images: Sequence[IrasutoyaImage],
    query: str,
) -> List[IrasutoyaImage]:
    """Filter out decorative assets and images not relevant to query, matching runtime client logic.

    Args:
        images: Sequence of candidate IrasutoyaImage objects.
        query: Search term used to rank and filter candidates.

    Returns:
        List of candidate images that are not decorative and relevant to query.
    """
    return [
        img for img in images
        if not is_decorative_asset(img, query) and is_image_relevant(img, query)
    ]


def _readonly_catalog_uri(catalog_path: str) -> str:
    """Return an encoded read-only immutable SQLite URI for catalog_path.

    Args:
        catalog_path: Filesystem path to the catalog database.

    Returns:
        Properly encoded SQLite URI string with mode=ro&immutable=1.
    """
    uri_path = urllib.request.pathname2url(os.path.abspath(catalog_path))
    return f"file:{uri_path}?mode=ro&immutable=1"


def update_catalog_sha256(catalog_path: str) -> str:
    """Compute and write SHA256 checksum for catalog database.

    Args:
        catalog_path: Path to catalog.sqlite.

    Returns:
        Hex-encoded SHA256 digest string.
    """
    build_mod = sys.modules.get("scripts.build_synonyms")
    if build_mod and hasattr(build_mod, "update_catalog_sha256"):
        build_sha = getattr(build_mod, "update_catalog_sha256")
        if build_sha is not update_catalog_sha256:
            return build_sha(catalog_path)

    hasher = hashlib.sha256()
    with open(catalog_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    digest = hasher.hexdigest()

    sha_path = f"{catalog_path}.sha256"
    basename = os.path.basename(catalog_path)
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {basename}\n")
    logger.info("Updated SHA256 checksum at %s", sha_path)
    return digest
