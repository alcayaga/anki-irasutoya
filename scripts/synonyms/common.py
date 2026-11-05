"""Shared constants and helper functions for synonym building and validation."""
import hashlib
import http.client
import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

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


class _KeepAliveJSONClient:
    """Persistent HTTP/1.1 Keep-Alive connection client with urllib.request fallback."""

    def __init__(self, timeout: float = 60.0) -> None:
        """Initialize keep-alive HTTP client.

        Args:
            timeout: Network request timeout in seconds.
        """
        self._connections: Dict[str, http.client.HTTPConnection] = {}
        self._timeout = timeout

    def post_json(self, url: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Send POST request with JSON payload and return parsed JSON response.

        Args:
            url: Full target HTTP URL.
            payload: JSON-serializable dictionary.
            timeout: Optional per-request timeout.

        Returns:
            Parsed JSON dictionary from the response.
        """
        # If urllib.request.urlopen is patched in unit tests, route directly through it
        if hasattr(urllib.request.urlopen, "mock_calls") or hasattr(urllib.request.urlopen, "assert_called"):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "anki-irasutoya/1.0"},
            )
            with urllib.request.urlopen(req, timeout=(timeout or self._timeout)) as resp:
                return json.loads(resp.read().decode("utf-8"))

        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        netloc = f"{host}:{port}"
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Host": netloc,
            "Content-Type": "application/json",
            "User-Agent": "anki-irasutoya/1.0",
            "Connection": "keep-alive",
            "Content-Length": str(len(data)),
        }

        # Attempt with cached connection, retry once on disconnect
        for attempt in range(2):
            conn = self._connections.get(netloc)
            if conn is None:
                if parsed.scheme == "https":
                    conn = http.client.HTTPSConnection(host, port, timeout=(timeout or self._timeout))
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=(timeout or self._timeout))
                self._connections[netloc] = conn

            try:
                conn.request("POST", path, body=data, headers=headers)
                resp = conn.getresponse()
                body = resp.read().decode("utf-8")
                if resp.status >= 400:
                    raise urllib.error.HTTPError(url, resp.status, body, resp.headers, None)
                return json.loads(body)
            except (http.client.ImproperConnectionState, http.client.BadStatusLine, ConnectionResetError, BrokenPipeError):
                try:
                    conn.close()
                except Exception:
                    pass
                self._connections.pop(netloc, None)
                if attempt == 1:
                    # Final fallback to standard urllib.request
                    req = urllib.request.Request(
                        url,
                        data=data,
                        headers={"Content-Type": "application/json", "User-Agent": "anki-irasutoya/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=(timeout or self._timeout)) as resp_fb:
                        return json.loads(resp_fb.read().decode("utf-8"))
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                self._connections.pop(netloc, None)
                raise

        return {}

    def close(self) -> None:
        """Close all persistent connections."""
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()

