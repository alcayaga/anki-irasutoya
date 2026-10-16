"""Client for searching and downloading illustrations from Irasutoya (いらすとや)."""
from dataclasses import dataclass
import json
import logging
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IrasutoyaImage:
    """Represents an illustration entry from Irasutoya."""
    title: str
    image_url: str
    thumb_url: str
    post_title: str
    description: str = ""


class InsecureRedirectError(urllib.error.URLError):
    """Raised when an HTTP redirect targets a non-HTTPS destination."""


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Enforce HTTPS-only redirects to prevent SSRF protocol downgrade."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Validate redirect target URL is HTTPS."""
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise InsecureRedirectError(f"Refusing non-HTTPS redirect: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


ALLOWED_IMAGE_HOSTS = (
    "bp.blogspot.com",
    "googleusercontent.com",
    "blogger.com",
    "irasutoya.com",
)


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
                if e.code == 429 and attempt < max_retries:
                    # Rate limited: wait with exponential backoff
                    sleep_time = (attempt + 1) * 2.0
                    logger.warning("Irasutoya 429 rate limit hit. Backing off for %.1f s...", sleep_time)
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

        a_pattern = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.IGNORECASE | re.DOTALL)
        img_pattern = re.compile(r"<img\b([^>]*)>", re.IGNORECASE)
        attr_href = re.compile(r"href=[\"\']([^\'\"\s]+)[\"\']", re.IGNORECASE)
        attr_src = re.compile(r"src=[\"\']([^\'\"\s]+)[\"\']", re.IGNORECASE)
        attr_alt = re.compile(r"alt=[\"\']([^\'\"]*)[\"\']", re.IGNORECASE)

        for entry in entries:
            post_title = entry.get("title", {}).get("$t", "").strip()
            summary = entry.get("summary", {}).get("$t", "").strip()
            content_html = entry.get("content", {}).get("$t", "")

            found_in_post = False

            # 1. Search for <a> links wrapping <img>
            for a_match in a_pattern.finditer(content_html):
                a_attrs = a_match.group(1)
                a_inner = a_match.group(2)

                href_m = attr_href.search(a_attrs)
                if not href_m:
                    continue
                href = href_m.group(1)
                if not re.search(r"\.(png|jpe?g)$", urllib.parse.urlparse(href).path, re.IGNORECASE):
                    continue

                img_m = img_pattern.search(a_inner)
                src = ""
                alt = ""
                if img_m:
                    img_attrs = img_m.group(1)
                    src_m = attr_src.search(img_attrs)
                    if src_m:
                        src = src_m.group(1)
                    alt_m = attr_alt.search(img_attrs)
                    if alt_m:
                        alt = alt_m.group(1)

                high_res = self._upgrade_image_url(href, image_size)
                if high_res in seen_urls:
                    continue
                seen_urls.add(high_res)

                title = alt.strip() if alt.strip() else post_title
                thumb = self._to_thumbnail_url(src or high_res, "s120-c")
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
                thumb = entry.get("media$thumbnail", {}).get("url", "")
                if thumb:
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
        return re.sub(r"/s\d+(-c)?/", f"/{size}/", url)

    @staticmethod
    def _upgrade_image_url(url: str, image_size: str = "s800") -> str:
        """Replaces Blogger thumbnail dimension codes like /s72-c/ or /s300/ with /s800/ or /s0/."""
        return re.sub(r"/s\d+(-c)?/", f"/{image_size}/", url)

    @staticmethod
    def _is_allowed_image_host(host: str) -> bool:
        """Validate hostname against permitted image host domains."""
        host = host.lower()
        return any(host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_IMAGE_HOSTS)

    def download_image(self, url: str, timeout: int = 15) -> bytes:
        """Download raw image binary data from an allowlisted HTTPS URL."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            raise ValueError(f"Refusing non-HTTPS image URL: {url}")
        host = (parsed.hostname or "").lower()
        if not self._is_allowed_image_host(host):
            raise ValueError(f"Refusing image download from untrusted host: {host}")
        return self._make_request(url, timeout=timeout)
