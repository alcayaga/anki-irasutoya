"""Blogger feed parsing, URL normalization, and post models for Irasutoya."""
import html
import logging
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_HOSTS = (
    "bp.blogspot.com",
    "googleusercontent.com",
    "irasutoya.com",
)


def is_allowed_image_host(host: str) -> bool:
    """Validate if hostname belongs to an allowlisted image domain."""
    host = host.lower()
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_IMAGE_HOSTS)


def is_image_url(url: str) -> bool:
    """Return True if URL points to an allowlisted HTTPS image endpoint."""
    if not url:
        return False
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and is_allowed_image_host(parsed.hostname or "")
        and bool(
            re.search(r"\.(png|jpe?g|webp)($|\?)", parsed.path, re.IGNORECASE)
            or re.search(r"/s\d+(-c)?/", parsed.path)
        )
    )


def upgrade_image_url(url: str, image_size: str = "s800") -> str:
    """Replace Blogger thumbnail dimension codes like /s72-c/ or /s300/ with /s800/ or /s0/."""
    return re.sub(r"/s\d+(-c)?/(?=[^/]+$)", f"/{image_size}/", url)


def to_thumbnail_url(url: str, size: str = "s120-c") -> str:
    """Scale a Blogger image URL to a compact thumbnail format."""
    return re.sub(r"/s\d+(-c)?/(?=[^/]+$)", f"/{size}/", url)


def is_meaningful_title(text: str) -> bool:
    """Return True if text contains letters or numbers, filtering placeholder symbols like '■' or '・'."""
    if not text:
        return False
    return any(unicodedata.category(ch).startswith(("L", "N")) for ch in text)


@dataclass
class IrasutoyaImage:
    """Represents an illustration entry from Irasutoya."""

    title: str
    image_url: str
    thumb_url: str
    post_title: str
    description: str = ""
    is_single_image_post: bool = True
    post_id: str = ""
    categories: Tuple[str, ...] = ()


@dataclass
class ParsedPost:
    """Represents a fully parsed Blogger post entry."""

    blogger_id: str
    title: str
    description: str
    published: str
    updated: str
    categories: Tuple[str, ...]
    images: List[IrasutoyaImage]


class _BloggerContentParser(HTMLParser):
    """HTML parser to reliably extract (href, src, alt) image tuples and body text from Blogger content."""

    def __init__(self) -> None:
        """Initialize parser state, text chunks, and items list."""
        super().__init__(convert_charrefs=True)
        self.items: List[Tuple[Optional[str], str, str]] = []
        self.text_chunks: List[str] = []
        self._current_href: Optional[str] = None
        self._ignored_depth: int = 0

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        """Track active anchor href and capture nested or standalone img src and alt attributes."""
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript"):
            self._ignored_depth += 1
            return
        if self._ignored_depth > 0:
            return
        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        if tag_lower == "a":
            self._current_href = attr_dict.get("href")
        elif tag_lower == "img":
            src = attr_dict.get("src", "")
            alt = attr_dict.get("alt", "")
            self.items.append((self._current_href, src, alt))

    def handle_endtag(self, tag: str) -> None:
        """Clear active anchor when </a> is closed, and track script/style closing tags."""
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript"):
            if self._ignored_depth > 0:
                self._ignored_depth -= 1
            return
        if self._ignored_depth > 0:
            return
        if tag_lower == "a":
            self._current_href = None

    def handle_data(self, data: str) -> None:
        """Capture visible text content outside of tags and script/style elements."""
        if data and self._ignored_depth == 0:
            self.text_chunks.append(data)


def parse_entry(
    entry: dict,
    image_size: str = "s800",
    seen_urls: Optional[Set[str]] = None,
) -> ParsedPost:
    """Parse a single Blogger feed entry into a ParsedPost object with IrasutoyaImage items."""
    raw_id = entry.get("id", {}).get("$t", "").strip()
    blogger_id = raw_id.split(".post-")[-1] if ".post-" in raw_id else raw_id
    post_title = html.unescape(entry.get("title", {}).get("$t", "").strip())
    summary = html.unescape(entry.get("summary", {}).get("$t", "").strip())
    content_html = entry.get("content", {}).get("$t", "")
    published = entry.get("published", {}).get("$t", "").strip()
    updated = entry.get("updated", {}).get("$t", "").strip()
    categories = tuple(
        html.unescape(cat.get("term", "").strip())
        for cat in entry.get("category", [])
        if cat.get("term")
    )

    parser = _BloggerContentParser()
    try:
        parser.feed(content_html)
    except Exception as e:
        logger.warning("Failed to parse entry HTML: %s", e)

    content_text = re.sub(r"\s+", " ", " ".join(parser.text_chunks)).strip()
    post_desc = content_text or summary

    post_images: List[IrasutoyaImage] = []
    if seen_urls is None:
        seen_urls = set()
    post_urls = set()

    for href, src, alt in parser.items:
        candidate_url = None
        if href and is_image_url(href):
            candidate_url = href
        elif src and is_image_url(src):
            candidate_url = src

        if not candidate_url:
            continue

        high_res = upgrade_image_url(candidate_url, image_size)
        post_urls.add(high_res)
        if high_res in seen_urls:
            continue
        seen_urls.add(high_res)

        clean_alt = alt.strip()
        title = clean_alt if is_meaningful_title(clean_alt) else post_title
        if src and is_image_url(src):
            thumb = to_thumbnail_url(src, "s120-c")
        else:
            thumb = to_thumbnail_url(high_res, "s120-c")

        post_images.append(
            IrasutoyaImage(
                title=title,
                image_url=high_res,
                thumb_url=thumb,
                post_title=post_title,
                description=post_desc,
                post_id=blogger_id,
                categories=categories,
            )
        )

    # Fallback to media$thumbnail if no images found in HTML body
    if not post_images:
        thumb_raw = entry.get("media$thumbnail", {}).get("url", "")
        if thumb_raw:
            thumb = html.unescape(thumb_raw).strip()
            try:
                parsed_thumb = urllib.parse.urlparse(thumb)
            except ValueError:
                parsed_thumb = None
            if parsed_thumb and parsed_thumb.scheme == "https" and is_allowed_image_host(parsed_thumb.hostname or ""):
                high_res = upgrade_image_url(thumb, image_size)
                post_urls.add(high_res)
                if high_res not in seen_urls:
                    seen_urls.add(high_res)
                    post_images.append(
                        IrasutoyaImage(
                            title=post_title,
                            image_url=high_res,
                            thumb_url=to_thumbnail_url(thumb, "s120-c"),
                            post_title=post_title,
                            description=post_desc,
                            post_id=blogger_id,
                            categories=categories,
                        )
                    )

    is_single = (len(post_urls) == 1)
    for img in post_images:
        img.is_single_image_post = is_single

    return ParsedPost(
        blogger_id=blogger_id,
        title=post_title,
        description=post_desc,
        published=published,
        updated=updated,
        categories=categories,
        images=post_images,
    )


def parse_entries(entries: list, image_size: str = "s800") -> List[IrasutoyaImage]:
    """Parse a sequence of Blogger feed entries into a flat list of IrasutoyaImage instances."""
    images: List[IrasutoyaImage] = []
    seen_urls: Set[str] = set()
    for entry in entries:
        entry_seen = set(seen_urls)
        try:
            post = parse_entry(entry, image_size=image_size, seen_urls=entry_seen)
        except Exception as e:
            logger.warning("Failed to parse feed entry: %s", e)
            continue
        seen_urls.update(entry_seen)
        images.extend(post.images)
    return images
