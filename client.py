"""Client for searching and downloading illustrations from Irasutoya (いらすとや)."""
from dataclasses import dataclass
import html
from html.parser import HTMLParser
import json
import logging
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class _BloggerContentParser(HTMLParser):
    """HTML parser to reliably extract (href, src, alt) image tuples from Blogger content."""

    def __init__(self) -> None:
        """Initialize parser state and items list."""
        super().__init__(convert_charrefs=True)
        self.items: List[Tuple[str, str, str]] = []
        self._current_href: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        """Track active anchor href and capture nested img src and alt attributes."""
        tag_lower = tag.lower()
        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        if tag_lower == "a":
            self._current_href = attr_dict.get("href")
        elif tag_lower == "img" and self._current_href:
            src = attr_dict.get("src", "")
            alt = attr_dict.get("alt", "")
            self.items.append((self._current_href, src, alt))

    def handle_endtag(self, tag: str) -> None:
        """Clear active anchor when </a> is closed."""
        if tag.lower() == "a":
            self._current_href = None


@dataclass
class IrasutoyaImage:
    """Represents an illustration entry from Irasutoya."""
    title: str
    image_url: str
    thumb_url: str
    post_title: str
    description: str = ""


ALLOWED_IMAGE_HOSTS = (
    "bp.blogspot.com",
    "googleusercontent.com",
    "irasutoya.com",
)


def is_allowed_image_host(host: str) -> bool:
    """Validate if hostname belongs to an allowlisted image domain."""
    host = host.lower()
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_IMAGE_HOSTS)


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

    def __init__(self, user_agent: Optional[str] = None):
        """Initialize the Irasutoya client with custom or default User-Agent."""
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self._opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler)

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
        image_size: str = "s800"
    ) -> List[IrasutoyaImage]:
        """Search Irasutoya for a given query keyword.
        
        Args:
            query: The keyword to search for (e.g. '猫' or '走る').
            max_results: Maximum posts to fetch from Blogger feed.
            image_size: Resolution replacement, e.g. 's800' or 's0' (original size).
        
        Returns:
            A list of IrasutoyaImage objects.
        """
        clean_q = query.strip()
        if not clean_q:
            return []

        params = urllib.parse.urlencode({
            "alt": "json",
            "q": clean_q,
            "max-results": max_results,
        })
        url = f"{self.BASE_URL}/feeds/posts/default?{params}"

        raw_data = self._make_request(url)
        data = json.loads(raw_data.decode("utf-8"))

        entries = data.get("feed", {}).get("entry", [])
        return self.parse_entries(entries, image_size=image_size)

    def parse_entries(self, entries: list, image_size: str = "s800") -> List[IrasutoyaImage]:
        """Parse Blogger feed entries into IrasutoyaImage instances."""
        images: List[IrasutoyaImage] = []
        seen_urls = set()

        for entry in entries:
            post_title = html.unescape(entry.get("title", {}).get("$t", "").strip())
            summary = html.unescape(entry.get("summary", {}).get("$t", "").strip())
            content_html = entry.get("content", {}).get("$t", "")

            found_in_post = False

            parser = _BloggerContentParser()
            try:
                parser.feed(content_html)
            except Exception as e:
                logger.warning("Failed to parse entry HTML: %s", e)

            # 1. Search for <a> links wrapping <img>
            for href, src, alt in parser.items:
                parsed_href = urllib.parse.urlparse(href)
                if parsed_href.scheme != "https":
                    continue
                if not self._is_allowed_image_host(parsed_href.hostname or ""):
                    continue
                if not re.search(r"\.(png|jpe?g)$", parsed_href.path, re.IGNORECASE):
                    continue

                high_res = self._upgrade_image_url(href, image_size)
                if high_res in seen_urls:
                    continue
                seen_urls.add(high_res)

                title = alt.strip() if alt.strip() else post_title
                parsed_src = urllib.parse.urlparse(src) if src else None
                if (
                    parsed_src
                    and parsed_src.scheme == "https"
                    and self._is_allowed_image_host(parsed_src.hostname or "")
                ):
                    thumb = self._to_thumbnail_url(src, "s120-c")
                else:
                    thumb = self._to_thumbnail_url(high_res, "s120-c")

                images.append(
                    IrasutoyaImage(
                        title=title,
                        image_url=high_res,
                        thumb_url=thumb,
                        post_title=post_title,
                        description=summary
                    )
                )
                found_in_post = True

            # 2. Fallback to post thumbnail if no images found in content
            if not found_in_post:
                thumb_raw = entry.get("media$thumbnail", {}).get("url", "")
                if thumb_raw:
                    thumb = html.unescape(thumb_raw).strip()
                    parsed_thumb = urllib.parse.urlparse(thumb)
                    if parsed_thumb.scheme == "https" and self._is_allowed_image_host(parsed_thumb.hostname or ""):
                        high_res = self._upgrade_image_url(thumb, image_size)
                        if high_res not in seen_urls:
                            seen_urls.add(high_res)
                            images.append(
                                IrasutoyaImage(
                                    title=post_title,
                                    image_url=high_res,
                                    thumb_url=self._to_thumbnail_url(thumb, "s120-c"),
                                    post_title=post_title,
                                    description=summary
                                )
                            )

        return images

    @staticmethod
    def _to_thumbnail_url(url: str, size: str = "s120-c") -> str:
        """Scale a Blogger image URL to a tiny thumbnail format."""
        return re.sub(r"/s\d+(-c)?/(?=[^/]+$)", f"/{size}/", url)

    @staticmethod
    def _upgrade_image_url(url: str, image_size: str = "s800") -> str:
        """Replaces Blogger thumbnail dimension codes like /s72-c/ or /s300/ with /s800/ or /s0/."""
        return re.sub(r"/s\d+(-c)?/(?=[^/]+$)", f"/{image_size}/", url)

    @staticmethod
    def _is_allowed_image_host(host: str) -> bool:
        """Validate hostname against permitted image host domains."""
        return is_allowed_image_host(host)

    def download_image(self, url: str, timeout: int = 15) -> bytes:
        """Download raw image binary data from an allowlisted HTTPS URL."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            raise ValueError(f"Refusing non-HTTPS image URL: {url}")
        host = (parsed.hostname or "").lower()
        if not self._is_allowed_image_host(host):
            raise ValueError(f"Refusing image download from untrusted host: {host}")
        return self._make_request(url, timeout=timeout)
