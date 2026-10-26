"""Client for searching and downloading illustrations from Irasutoya (いらすとや)."""
import json
import logging
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .catalog import CatalogStore, default_store
from .feed import IrasutoyaImage, ParsedPost, is_allowed_image_host, parse_entry, upgrade_image_url
from .ranking import (
    has_title_illustration_match,
    infer_pos_type,
    is_decorative_asset,
    is_image_relevant,
    rank_relevant_images,
)
from .text_utils import PHONETIC_REASON_PREFIXES

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
    ):
        """Initialize the Irasutoya client with custom or default User-Agent and catalog backend."""
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self._opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler)
        self.catalog = catalog if catalog is not None else default_store()
        self._healed_posts_cache: Dict[Tuple[str, str], Optional[ParsedPost]] = {}
        self._healed_posts_lock = threading.Lock()

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
        max_results: Optional[int] = None,
        image_size: str = "s800",
    ) -> List[IrasutoyaImage]:
        """Search Irasutoya catalog for a given query keyword.

        Args:
            query: The keyword to search for (e.g. '猫' or '走る').
            max_results: Optional maximum number of images to return.
            image_size: Target resolution replacement, e.g. 's800' or 's0' (original size).

        Returns:
            A list of IrasutoyaImage objects.
        """
        clean_q = query.strip()
        if not clean_q:
            return []

        index = self.catalog.get_index() if self.catalog is not None else None
        if index is None:
            logger.warning("Offline catalog index is unavailable.")
            return []

        results = index.search(clean_q, image_size=image_size)
        if max_results is not None and max_results > 0:
            return results[:max_results]
        return results

    def search_with_fallbacks(
        self,
        candidates: Sequence[Tuple[str, str, int]],
        max_results: Optional[int] = 12,
        image_size: str = "s800",
        pos_type: Optional[str] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Tuple[List[IrasutoyaImage], Optional[Tuple[str, str, int]], Optional[IrasutoyaImage]]:
        """Query candidate terms in sequence until images are found.

        Args:
            candidates: Sequence of (term, reason_label, tier).
            max_results: Maximum illustrations to return for the matched candidate (None for uncapped).
            image_size: Target image resolution code.
            pos_type: Optional Part-of-Speech profile ('verb', 'noun', 'adjective') for category ranking.
            is_cancelled: Optional callback returning True if search should abort early.

        Returns:
            (ranked_images, matched_candidate, best_image)
        """
        first_weak_fallback: Optional[Tuple[List[IrasutoyaImage], Tuple[str, str, int], IrasutoyaImage]] = None

        for cand in candidates:
            if is_cancelled and is_cancelled():
                break
            term = cand[0]
            reason = cand[1]
            images = self.search(term, max_results=None, image_size=image_size)

            # Filter out non-illustration/decorative assets and images not relevant to the query term
            illustration_candidates = [
                img for img in images
                if not is_decorative_asset(img, term) and is_image_relevant(img, term)
            ]
            if illustration_candidates:
                wins_now = (
                    not reason.startswith(PHONETIC_REASON_PREFIXES)
                    or first_weak_fallback is None
                ) and has_title_illustration_match(illustration_candidates, term)
                if not wins_now and first_weak_fallback is not None:
                    continue

                cand_pos = pos_type or infer_pos_type(term, cand_reason=reason)
                ranked = rank_relevant_images(illustration_candidates, term, pos_type=cand_pos)
                if max_results and max_results > 0:
                    ranked = ranked[:max_results]

                if wins_now:
                    return ranked, cand, ranked[0]

                first_weak_fallback = (ranked, cand, ranked[0])

        if first_weak_fallback is not None:
            return first_weak_fallback

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
                refetched_url = self._self_heal_image_url(image, original_url=url)
                if refetched_url and refetched_url != url:
                    logger.info("Self-healed image URL for %s from %s to %s", image.title, url, refetched_url)
                    return self._make_request(refetched_url, timeout=timeout)
            raise

    def _self_heal_image_url(self, image: IrasutoyaImage, original_url: Optional[str] = None) -> Optional[str]:
        """Fetch the single Blogger post entry to locate an updated image URL after a 404/410."""
        post_id = getattr(image, "post_id", "").strip()
        if not post_id:
            return None

        size_token = "s800"
        if original_url:
            match = re.search(r"/(s\d+(-c)?|s0)/", original_url)
            if match:
                size_token = match.group(1)

        cache_key = (post_id, size_token)
        with self._healed_posts_lock:
            post = self._healed_posts_cache.get(cache_key)

        if post is None:
            url = f"{self.BASE_URL}/feeds/posts/default/{post_id}?alt=json"
            try:
                raw_data = self._make_request(url, timeout=10)
                data = json.loads(raw_data.decode("utf-8"))
                entry = data.get("entry", data)
                parsed_post = parse_entry(entry, image_size=size_token)
                with self._healed_posts_lock:
                    self._healed_posts_cache[cache_key] = parsed_post
                post = parsed_post
            except Exception as e:
                logger.warning("Failed to self-heal image URL for post %s: %s", post_id, e)
                return None

        if post is None or not post.images:
            return None

        # 1. Match by image filename (most specific across multi-image posts)
        target_base = image.image_url.split("/")[-1]
        for img in post.images:
            if img.image_url.split("/")[-1] == target_base:
                return upgrade_image_url(img.image_url, size_token)

        # 2. Match by title only if unambiguous (exactly 1 match)
        title_matches = [img for img in post.images if img.title == image.title]
        if len(title_matches) == 1:
            return upgrade_image_url(title_matches[0].image_url, size_token)

        # 3. Fallback to single image if post only has 1 image
        if len(post.images) == 1:
            return upgrade_image_url(post.images[0].image_url, size_token)

        return None
