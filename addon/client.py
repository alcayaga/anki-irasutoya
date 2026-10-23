"""Client for searching and downloading illustrations from Irasutoya (いらすとや)."""
import json
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, List, Optional, Sequence, Tuple

from .catalog import CatalogStore
from .feed import IrasutoyaImage, is_allowed_image_host, parse_entries, parse_entry
from .ranking import is_decorative_asset, is_image_relevant, rank_relevant_images

logger = logging.getLogger(__name__)


class InsecureRedirectError(urllib.error.URLError):
    """Raised when an HTTP redirect targets a non-HTTPS or untrusted destination."""


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Enforce HTTPS-only and trusted host redirects to prevent SSRF and protocol downgrades."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Validate redirect target URL is HTTPS and points to an allowlisted host."""
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https":
            raise InsecureRedirectError(f"Refusing non-HTTPS redirect: {newurl}")
        host = (parsed.hostname or "").lower()
        if not is_allowed_image_host(host):
            raise InsecureRedirectError(f"Refusing redirect to untrusted host: {host}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class IrasutoyaClient:
    """Queries Irasutoya's public Blogger JSON feeds to search and retrieve images."""

    BASE_URL = "https://www.irasutoya.com"
    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36 (Anki-Irasutoya/1.0)"
    )

    def __init__(
        self,
        user_agent: Optional[str] = None,
        catalog: Optional[CatalogStore] = None,
        backend: str = "auto",
    ):
        """Initialize the Irasutoya client with custom or default User-Agent and catalog backend."""
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self._opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler)
        self.catalog = catalog
        self.backend = backend

    def uses_catalog(self) -> bool:
        """Return True if client is configured to query the offline metadata catalog."""
        return (
            self.backend != "live"
            and self.catalog is not None
            and self.catalog.get_index() is not None
        )

    def _make_request(self, url: str, timeout: int = 10, max_retries: int = 2) -> bytes:
        """Execute HTTP GET with User-Agent, handling rate-limit backoff."""
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                with self._opener.open(req, timeout=timeout) as resp:
                    return resp.read()
            except InsecureRedirectError:
                raise
            except urllib.error.HTTPError as e:
                last_error = e
                if (e.code == 429 or e.code >= 500) and attempt < max_retries:
                    # Rate limited or transient server error: wait with exponential backoff
                    sleep_time = (attempt + 1) * 2.0
                    logger.warning("Irasutoya HTTP %d error. Backing off for %.1f s...", e.code, sleep_time)
                    time.sleep(sleep_time)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue
                raise

        if last_error:
            raise last_error
        return b""

    def search(
        self,
        query: str,
        max_results: int = 12,
        image_size: str = "s800",
        orderby: Optional[str] = "relevance",
    ) -> List[IrasutoyaImage]:
        """Search Irasutoya for a given query keyword.

        Args:
            query: The keyword to search for (e.g. '猫' or '走る').
            max_results: Maximum posts to fetch from Blogger feed.
            image_size: Resolution replacement, e.g. 's800' or 's0' (original size).
            orderby: Sort order for Blogger feed ('relevance', 'updated', etc.).

        Returns:
            A list of IrasutoyaImage objects.
        """
        clean_q = query.strip()
        if not clean_q:
            return []

        if self.uses_catalog():
            index = self.catalog.get_index()
            if index is not None:
                return index.search(clean_q, image_size=image_size)

        params: dict = {
            "alt": "json",
            "q": clean_q,
            "max-results": max_results,
        }
        if orderby:
            params["orderby"] = orderby
        url = f"{self.BASE_URL}/feeds/posts/default?{urllib.parse.urlencode(params)}"

        raw_data = self._make_request(url)
        data = json.loads(raw_data.decode("utf-8"))

        entries = data.get("feed", {}).get("entry", [])
        return parse_entries(entries, image_size=image_size)

    def search_with_fallbacks(
        self,
        candidates: Sequence[Tuple[str, str, int]],
        max_results: int = 12,
        image_size: str = "s800",
        orderby: Optional[str] = "relevance",
        pace_callback: Optional[Callable[[], None]] = None,
    ) -> Tuple[List[IrasutoyaImage], Optional[Tuple[str, str, int]], Optional[IrasutoyaImage]]:
        """Query candidate terms in sequence until images are found.

        Args:
            candidates: Sequence of (term, reason_label, tier).
            max_results: Maximum posts to fetch per query.
            image_size: Target image resolution code.
            orderby: Blogger ordering parameter.
            pace_callback: Optional callback invoked before each network query to pace requests.

        Returns:
            (ranked_images, matched_candidate, best_image)
        """
        for cand in candidates:
            term = cand[0]
            if pace_callback is not None and not self.uses_catalog():
                pace_callback()
            images = self.search(term, max_results=max_results, image_size=image_size, orderby=orderby)

            # Filter out non-illustration/decorative assets and images not relevant to the query term
            illustration_candidates = [
                img for img in images
                if not is_decorative_asset(img, term) and is_image_relevant(img, term)
            ]
            if illustration_candidates:
                ranked = rank_relevant_images(illustration_candidates, term)
                if max_results:
                    ranked = ranked[:max_results]
                best = ranked[0]
                return ranked, cand, best

        return [], None, None

    def download_image(
        self,
        url: str,
        timeout: int = 15,
        *,
        image: Optional[IrasutoyaImage] = None,
    ) -> bytes:
        """Download raw image binary data from an allowlisted HTTPS URL.

        If HTTP 404 or 410 occurs and image metadata includes a post_id,
        attempts a one-time self-heal by re-fetching the Blogger post entry.
        """
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            raise ValueError(f"Refusing non-HTTPS image URL: {url}")
        host = (parsed.hostname or "").lower()
        if not is_allowed_image_host(host):
            raise ValueError(f"Refusing image download from untrusted host: {host}")

        try:
            return self._make_request(url, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (404, 410) and image is not None and getattr(image, "post_id", None):
                refetched_url = self._self_heal_image_url(image)
                if refetched_url and refetched_url != url:
                    logger.info("Self-healed image URL for %s from %s to %s", image.title, url, refetched_url)
                    return self._make_request(refetched_url, timeout=timeout)
            raise

    def _self_heal_image_url(self, image: IrasutoyaImage) -> Optional[str]:
        """Fetch the single Blogger post entry to locate an updated image URL after a 404/410."""
        post_id = getattr(image, "post_id", "").strip()
        if not post_id:
            return None
        url = f"{self.BASE_URL}/feeds/posts/default/{post_id}?alt=json"
        try:
            raw_data = self._make_request(url, timeout=10)
            data = json.loads(raw_data.decode("utf-8"))
            entry = data.get("entry", data)
            post = parse_entry(entry)
            if not post.images:
                return None
            for img in post.images:
                if img.title == image.title:
                    return img.image_url
            target_base = image.image_url.split("/")[-1]
            for img in post.images:
                if img.image_url.split("/")[-1] == target_base:
                    return img.image_url
            if len(post.images) == 1:
                return post.images[0].image_url
        except Exception as e:
            logger.warning("Failed to self-heal image URL for post %s: %s", post_id, e)
        return None
