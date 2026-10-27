"""Client for searching and downloading illustrations from Irasutoya (いらすとや)."""
import http.client
import json
import logging
import re
import socket
import ssl
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


class _PooledResponse:
    """Wrapper around http.client.HTTPResponse returning connection to pool on close."""

    def __init__(
        self,
        response: http.client.HTTPResponse,
        conn: http.client.HTTPSConnection,
        return_conn_fn: Callable[[http.client.HTTPSConnection, http.client.HTTPResponse], None],
    ) -> None:
        """Initialize response wrapper."""
        self._resp = response
        self._conn = conn
        self._return_conn_fn = return_conn_fn
        self._returned = False
        self.url = getattr(response, "url", "")
        self.msg = getattr(response, "msg", getattr(response, "reason", ""))
        self.status = response.status
        self.code = response.status
        self.reason = response.reason
        self.headers = response.headers

    def read(self, *args, **kwargs) -> bytes:
        """Read response payload."""
        return self._resp.read(*args, **kwargs)

    def readline(self, *args, **kwargs) -> bytes:
        """Read single response line."""
        return self._resp.readline(*args, **kwargs)

    def close(self) -> None:
        """Close response stream and return active connection to pool."""
        if not self._returned:
            self._returned = True
            self._return_conn_fn(self._conn, self._resp)
        self._resp.close()

    def __enter__(self) -> "_PooledResponse":
        """Enter response context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit response context and trigger connection return."""
        self.close()

    def info(self) -> http.client.HTTPMessage:
        """Return HTTP headers."""
        return self._resp.headers

    def geturl(self) -> str:
        """Return effective request URL."""
        return self.url

    def getcode(self) -> int:
        """Return HTTP status code."""
        return self.status


class _KeepAliveHTTPSHandler(urllib.request.HTTPSHandler):
    """HTTPS handler pooling persistent HTTP/1.1 keep-alive connections per host."""

    MAX_IDLE_PER_HOST = 6

    def __init__(self, context: Optional[ssl.SSLContext] = None) -> None:
        """Initialize keep-alive handler with idle connection pool."""
        super().__init__(context=context)
        self._pool: Dict[Tuple[str, int], List[http.client.HTTPSConnection]] = {}
        self._lock = threading.Lock()

    def _return_connection(
        self,
        host_port: Tuple[str, int],
        conn: http.client.HTTPSConnection,
        resp: http.client.HTTPResponse,
    ) -> None:
        """Return connection to idle pool if response completed cleanly without will_close."""
        with self._lock:
            sock = getattr(conn, "sock", None)
            is_sock_alive = sock is not None and not getattr(sock, "_closed", True)
            if not getattr(resp, "will_close", True) and is_sock_alive:
                conns = self._pool.setdefault(host_port, [])
                if len(conns) < self.MAX_IDLE_PER_HOST:
                    conns.append(conn)
                    return
            try:
                conn.close()
            except Exception:
                pass

    def https_open(self, req: urllib.request.Request) -> _PooledResponse:
        """Open HTTPS request reusing a pooled persistent connection if available."""
        host = req.host
        if not host:
            raise urllib.error.URLError("no host given")

        if ":" in host:
            h_name, port_str = host.split(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = 443
        else:
            h_name = host
            port = 443

        host_port = (h_name.lower(), port)
        conn: Optional[http.client.HTTPSConnection] = None

        with self._lock:
            conns = self._pool.get(host_port, [])
            while conns:
                candidate = conns.pop()
                c_sock = getattr(candidate, "sock", None)
                if c_sock is not None and not getattr(c_sock, "_closed", True):
                    conn = candidate
                    break
                try:
                    candidate.close()
                except Exception:
                    pass

        timeout = getattr(req, "timeout", None) or socket._GLOBAL_DEFAULT_TIMEOUT

        is_reused = conn is not None
        if conn is None:
            conn = http.client.HTTPSConnection(
                h_name,
                port=port,
                timeout=timeout,
                context=self._context,
            )

        headers = dict(req.unredirected_hdrs)
        headers.update(req.headers)
        headers["Connection"] = "keep-alive"
        headers = {name.title(): val for name, val in headers.items()}

        try:
            conn.request(req.get_method(), req.selector, req.data, headers)
            r = conn.getresponse()
        except (
            http.client.RemoteDisconnected,
            http.client.BadStatusLine,
            ConnectionResetError,
            BrokenPipeError,
            ssl.SSLEOFError,
        ):
            try:
                conn.close()
            except Exception:
                pass
            if is_reused:
                # Reused socket was closed by server while idle; retry once with a fresh connection
                conn = http.client.HTTPSConnection(
                    h_name,
                    port=port,
                    timeout=timeout,
                    context=self._context,
                )
                conn.request(req.get_method(), req.selector, req.data, headers)
                r = conn.getresponse()
            else:
                raise
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

        r.url = req.get_full_url()
        r.msg = r.reason
        return _PooledResponse(
            r,
            conn,
            lambda c, resp: self._return_connection(host_port, c, resp),
        )

    def close(self) -> None:
        """Close and discard all idle connections in the pool."""
        with self._lock:
            for conns in self._pool.values():
                for c in conns:
                    try:
                        c.close()
                    except Exception:
                        pass
            self._pool.clear()


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
        self._keep_alive_handler = _KeepAliveHTTPSHandler()
        self._opener = urllib.request.build_opener(
            _HttpsOnlyRedirectHandler, self._keep_alive_handler
        )
        self.catalog = catalog if catalog is not None else default_store()
        self._healed_posts_cache: Dict[Tuple[str, str], Optional[ParsedPost]] = {}
        self._healed_posts_lock = threading.Lock()

    def close(self) -> None:
        """Close any pooled keep-alive HTTP connections."""
        if hasattr(self, "_keep_alive_handler") and self._keep_alive_handler is not None:
            self._keep_alive_handler.close()

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
