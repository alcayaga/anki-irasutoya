"""Client for searching and downloading illustrations from Irasutoya (いらすとや)."""
from dataclasses import dataclass
import html
from html.parser import HTMLParser
import json
import logging
import re
import socket
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, List, Optional, Sequence, Tuple

try:
    from .text_utils import is_grammar_function_word, conjugate_japanese_verb, SAFE_2MORA_LOANWORDS
except ImportError:
    from text_utils import is_grammar_function_word, conjugate_japanese_verb, SAFE_2MORA_LOANWORDS

logger = logging.getLogger(__name__)

KANJI_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf々〆〇]")
HIRAGANA_CHAR_RE = re.compile(r"[\u3040-\u309F]")
# Exclude ゠ (U+30A0) and ・ (U+30FB); they are word separators, not Katakana letters.
KATAKANA_CHAR_RE = re.compile(r"[\u30A1-\u30FA\u30FC-\u30FF]")
ABSTRACT_SINGLE_KANJI = frozenset({"中", "部", "件", "方", "事", "物", "歳"})




class _BloggerContentParser(HTMLParser):
    """HTML parser to reliably extract (href, src, alt) image tuples and body text from Blogger content."""

    def __init__(self) -> None:
        """Initialize parser state, text chunks, and items list."""
        super().__init__(convert_charrefs=True)
        self.items: List[Tuple[Optional[str], str, str]] = []
        self.text_chunks: List[str] = []
        self._current_href: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        """Track active anchor href and capture nested or standalone img src and alt attributes."""
        tag_lower = tag.lower()
        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        if tag_lower == "a":
            self._current_href = attr_dict.get("href")
        elif tag_lower == "img":
            src = attr_dict.get("src", "")
            alt = attr_dict.get("alt", "")
            self.items.append((self._current_href, src, alt))

    def handle_endtag(self, tag: str) -> None:
        """Clear active anchor when </a> is closed."""
        if tag.lower() == "a":
            self._current_href = None

    def handle_data(self, data: str) -> None:
        """Capture visible text content outside of tags."""
        if data:
            self.text_chunks.append(data)


@dataclass
class IrasutoyaImage:
    """Represents an illustration entry from Irasutoya."""
    title: str
    image_url: str
    thumb_url: str
    post_title: str
    description: str = ""
    is_single_image_post: bool = True


ALLOWED_IMAGE_HOSTS = (
    "bp.blogspot.com",
    "googleusercontent.com",
    "irasutoya.com",
)


def is_allowed_image_host(host: str) -> bool:
    """Validate if hostname belongs to an allowlisted image domain."""
    host = host.lower()
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_IMAGE_HOSTS)


def is_meaningful_title(text: str) -> bool:
    """Return True if text contains letters or numbers, filtering placeholder symbols like '■' or '・'."""
    if not text:
        return False
    return any(unicodedata.category(ch).startswith(("L", "N")) for ch in text)


DECORATIVE_NOISE_KEYWORDS = ("ライン素材", "背景", "フレーム", "枠", "テンプレート", "壁紙")
TEXT_CALLOUT_KEYWORDS = ("イラスト文字",)
SITUATIONAL_NOISE_KEYWORDS = ("事故", "トラブル", "失敗", "注意", "禁止", "マーク")


def classify_match_boundary(text: str, query: str) -> str:
    """Classify the boundary quality of query within text.
    
    Returns:
        'exact_quoted': explicitly quoted 「query」/『query』or exact title match
        'standalone': bounded by particles, punctuation, or script transitions
        'polite_prefix': preceded by お/ご with valid trailing boundary (e.g. お肉のイラスト)
        'inflected': an inflected verb form (e.g. 売ろう, 売り for 売る) appears
        'compound': fused with an adjacent Kanji or Katakana (e.g. 培養肉, カフェオレ)
        'none': not found or invalid mid-word Kana collision
    """
    if not text or not query:
        return "none"

    if (
        f"「{query}」" in text
        or f"『{query}』" in text
        or f"（{query}）" in text
        or f"({query})" in text
        or text in (query, f"{query}のイラスト", f"かわいい{query}のイラスト")
    ):
        return "exact_quoted"

    is_pure_kana = bool(re.fullmatch(r"[\u3040-\u309F\u30A0-\u30FF]+", query))
    is_pure_kanji = bool(re.fullmatch(r"[\u4e00-\u9fff\u3400-\u4dbf々〆〇]+", query))

    start = 0
    best_quality = "none"
    while True:
        idx = text.find(query, start)
        if idx == -1:
            break
        start = idx + 1
        prev_ch = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + len(query)] if idx + len(query) < len(text) else ""

        # Case 1: Pure Kana queries (high collision risk!)
        if is_pure_kana:
            is_katakana_q = bool(re.fullmatch(r"[\u30A0-\u30FF]+", query))
            if is_katakana_q:
                # Must not be embedded mid-Katakana (e.g. オレ in カフェオレ, モン in モンスター)
                prev_is_kata = bool(prev_ch and KATAKANA_CHAR_RE.match(prev_ch))
                next_is_kata = bool(next_ch and KATAKANA_CHAR_RE.match(next_ch))
                if prev_is_kata or next_is_kata:
                    continue
                if is_grammar_function_word(query) or len(query) < 2:
                    continue
                return "standalone"
            else:
                # Hiragana:
                # Grammar stop words and unlisted 2-mora words: only exact quoted matches allowed
                if is_grammar_function_word(query) or (len(query) <= 2 and query not in SAFE_2MORA_LOANWORDS):
                    continue
                particles = ("の", "が", "を", "に", "で", "と", "は", "も", "へ", "や", "な", "か")
                attributive_endings = ("い", "る", "た", "だ")
                # Reject if preceded by Hiragana other than polite お/ご, a particle, or an attributive ending
                prev_is_hira = bool(
                    prev_ch
                    and HIRAGANA_CHAR_RE.match(prev_ch)
                    and prev_ch not in ("お", "ご") + particles + attributive_endings
                )
                # Reject if followed by Hiragana other than a particle
                next_is_hira = bool(
                    next_ch
                    and HIRAGANA_CHAR_RE.match(next_ch)
                    and next_ch not in particles
                )
                if prev_is_hira or next_is_hira:
                    continue
                return "standalone"

        # Case 2: Pure Kanji queries (e.g. 肉, 肌, 本, 二人)
        if is_pure_kanji:
            prev_is_kanji = bool(prev_ch and KANJI_CHAR_RE.match(prev_ch))
            next_is_kanji = bool(next_ch and KANJI_CHAR_RE.match(next_ch))
            if not prev_is_kanji and not next_is_kanji:
                if prev_ch in ("お", "ご"):
                    best_quality = "polite_prefix"
                else:
                    return "standalone"
            else:
                if best_quality == "none":
                    best_quality = "compound"
            continue

        # Case 3: Mixed Kanji + Kana (e.g. 食べる, 1杯, 時に, お金, お茶)
        prev_is_kanji = bool(prev_ch and KANJI_CHAR_RE.match(prev_ch))
        next_is_kanji = bool(next_ch and KANJI_CHAR_RE.match(next_ch))
        # Bare stems (食べ, 売り, 勝ち) fused with a following Kanji form nouns (食べ物, 売り場, 勝ち組)
        ends_in_stem_okurigana = bool(HIRAGANA_CHAR_RE.match(query[-1])) and not query.endswith(
            ("う", "く", "ぐ", "す", "つ", "ぬ", "ぶ", "む", "る", "た", "だ", "い")
        )
        if (prev_is_kanji and KANJI_CHAR_RE.match(query[0])) or (
            next_is_kanji and (KANJI_CHAR_RE.match(query[-1]) or ends_in_stem_okurigana)
        ):
            if best_quality == "none":
                best_quality = "compound"
            continue
        return "standalone"

    # Check verb inflections (e.g. 疲れた人のイラスト for 疲れる, 勝った人のイラスト for 勝つ)
    for inflected in conjugate_japanese_verb(query):
        is_stem = not inflected.endswith(("た", "だ", "う"))
        pos = text.find(inflected)
        while pos != -1:
            p_ch = text[pos - 1] if pos > 0 else ""
            end = pos + len(inflected)
            n_ch = text[end] if end < len(text) else ""
            prev_kanji = bool(p_ch and KANJI_CHAR_RE.match(p_ch))
            next_kanji = bool(n_ch and KANJI_CHAR_RE.match(n_ch))
            # Must not be preceded by Kanji, and bare stems (食べ, 売り) must not form Kanji compounds (食べ物, 売り場)
            if not prev_kanji and not (is_stem and next_kanji):
                return "inflected"
            pos = text.find(inflected, pos + 1)

    return best_quality


def is_image_relevant(img: IrasutoyaImage, query: str) -> bool:
    """Check if the individual image itself is semantically relevant with boundary safety.

    Prevents false-positive matches where a multi-image post matching a query contains
    unrelated illustrations, or where short queries collide mid-word.
    """
    clean_q = query.strip()
    if not clean_q:
        return False
    title = img.title.strip()
    post_title = img.post_title.strip()

    # 1. Title boundary classification
    qual_title = classify_match_boundary(title, clean_q)
    if qual_title in ("exact_quoted", "standalone", "polite_prefix", "inflected"):
        return True
    if qual_title == "compound":
        # Disallow compound matches for 1-Kanji grammatical/abstract words
        if len(clean_q) == 1 and clean_q in ABSTRACT_SINGLE_KANJI:
            return False
        # Bare verb stems fused with Kanji form different nouns (売り場, 勝ち組, 食べ物)
        if HIRAGANA_CHAR_RE.match(clean_q[-1]) and not clean_q.endswith(
            ("う", "く", "ぐ", "す", "つ", "ぬ", "ぶ", "む", "る", "た", "だ", "い")
        ):
            return False
        return True

    # 2. Single-image post caption/description match
    if getattr(img, "is_single_image_post", True) and not is_grammar_function_word(clean_q):
        desc = getattr(img, "description", "").strip()
        if desc:
            qual_desc = classify_match_boundary(desc, clean_q)
            if qual_desc in ("exact_quoted", "standalone", "polite_prefix", "inflected"):
                return True

    # 3. Multi-word queries
    tokens = clean_q.split()
    if len(tokens) > 1 and all(classify_match_boundary(title, t) != "none" for t in tokens):
        return True

    return False


def score_candidate_image(img: IrasutoyaImage, query: str) -> float:
    """Evaluate image semantic match quality and archetypal fit for flashcard learning."""
    score = 0.0
    clean_q = query.strip()
    title = img.title.strip()
    post_title = img.post_title.strip()

    qual_title = classify_match_boundary(title, clean_q)

    # 1. Exact canonical matches
    if is_image_relevant(img, clean_q):
        if title == f"{clean_q}のイラスト":
            score += 100.0
        elif title in (f"お{clean_q}のイラスト", f"ご{clean_q}のイラスト"):
            score += 95.0
        elif title == f"かわいい{clean_q}のイラスト":
            score += 90.0
        elif title == clean_q:
            score += 85.0
        elif title.startswith(f"{clean_q}の"):
            score += 60.0
        elif post_title == f"{clean_q}のイラスト":
            score += 50.0
        elif post_title.startswith(f"{clean_q}の"):
            score += 30.0

    # 2. Boundary, politeness, and focus bonuses
    if qual_title == "exact_quoted" or f"「{clean_q}」" in title or f"（{clean_q}）" in title:
        score += 55.0
    elif qual_title == "polite_prefix":
        score += 45.0
    elif qual_title == "standalone":
        score += 35.0
    elif qual_title == "inflected":
        score += 30.0
    elif qual_title == "compound":
        score -= 35.0
    elif is_image_relevant(img, clean_q):
        # Matched only via description on single-image post
        score += 15.0
    else:
        score -= 50.0  # Heavy penalty: image does not mention the query

    # 3. Penalize length (shorter, direct titles are more archetypal)
    score -= len(title) * 0.8

    # 4. Situational, hazard, and mark noise penalty
    for noise in SITUATIONAL_NOISE_KEYWORDS:
        if noise in title and noise not in clean_q:
            score -= 25.0

    # 5. Heavy penalty for text callouts (イラスト文字 / trailing 文字) so illustrations always win if available
    is_callout = ("イラスト文字" in title and "イラスト文字" not in clean_q) or (
        title.endswith("文字") and "文字" not in clean_q
    )
    if is_callout:
        score -= 65.0

    # 6. Filter out non-illustration decorative assets (lines, borders, wallpaper, templates)
    for noise in DECORATIVE_NOISE_KEYWORDS:
        if noise in title and noise not in clean_q:
            score -= 60.0

    return score


def is_decorative_asset(img: IrasutoyaImage, query: str) -> bool:
    """Return True if image is a decorative asset (frame, border, wallpaper) not matching query."""
    clean_q = query.strip()
    title = img.title.strip()
    return any(noise in title and noise not in clean_q for noise in DECORATIVE_NOISE_KEYWORDS)


def select_best_image(images: Sequence[IrasutoyaImage], query: str) -> Optional[IrasutoyaImage]:
    """Select the highest-ranked archetypal image from candidate images, excluding decorative assets."""
    if not images:
        return None
    illustration_candidates = [
        img for img in images
        if not is_decorative_asset(img, query) and is_image_relevant(img, query)
    ]
    pool = illustration_candidates if illustration_candidates else [
        img for img in images if not is_decorative_asset(img, query)
    ] or images
    return max(pool, key=lambda img: score_candidate_image(img, query))



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
        image_size: str = "s800",
        orderby: Optional[str] = "relevance"
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
            post_images: List[IrasutoyaImage] = []
            post_urls = set()

            parser = _BloggerContentParser()
            try:
                parser.feed(content_html)
            except Exception as e:
                logger.warning("Failed to parse entry HTML: %s", e)

            content_text = re.sub(r"\s+", " ", " ".join(parser.text_chunks)).strip()
            post_desc = content_text or summary

            # 1. Search for <a> links wrapping <img> or standalone <img> tags
            for href, src, alt in parser.items:
                candidate_url = None

                # Check href first if present
                if href:
                    parsed_href = urllib.parse.urlparse(href)
                    if (
                        parsed_href.scheme == "https"
                        and self._is_allowed_image_host(parsed_href.hostname or "")
                        and (
                            re.search(r"\.(png|jpe?g|webp)($|\?)", parsed_href.path, re.IGNORECASE)
                            or re.search(r"/s\d+(-c)?/", parsed_href.path)
                        )
                    ):
                        candidate_url = href

                # If href is not an image URL or missing, check src
                if not candidate_url and src:
                    parsed_src = urllib.parse.urlparse(src)
                    if (
                        parsed_src.scheme == "https"
                        and self._is_allowed_image_host(parsed_src.hostname or "")
                        and (
                            re.search(r"\.(png|jpe?g|webp)($|\?)", parsed_src.path, re.IGNORECASE)
                            or re.search(r"/s\d+(-c)?/", parsed_src.path)
                        )
                    ):
                        candidate_url = src

                if not candidate_url:
                    continue

                high_res = self._upgrade_image_url(candidate_url, image_size)
                post_urls.add(high_res)
                if high_res in seen_urls:
                    continue
                seen_urls.add(high_res)

                clean_alt = alt.strip()
                title = clean_alt if is_meaningful_title(clean_alt) else post_title
                parsed_src = urllib.parse.urlparse(src) if src else None
                if (
                    parsed_src
                    and parsed_src.scheme == "https"
                    and self._is_allowed_image_host(parsed_src.hostname or "")
                ):
                    thumb = self._to_thumbnail_url(src, "s120-c")
                else:
                    thumb = self._to_thumbnail_url(high_res, "s120-c")

                post_images.append(
                    IrasutoyaImage(
                        title=title,
                        image_url=high_res,
                        thumb_url=thumb,
                        post_title=post_title,
                        description=post_desc
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
                        post_urls.add(high_res)
                        if high_res not in seen_urls:
                            seen_urls.add(high_res)
                            post_images.append(
                                IrasutoyaImage(
                                    title=post_title,
                                    image_url=high_res,
                                    thumb_url=self._to_thumbnail_url(thumb, "s120-c"),
                                    post_title=post_title,
                                    description=post_desc
                                )
                            )

            is_single = (len(post_urls) == 1)
            for img in post_images:
                img.is_single_image_post = is_single
                images.append(img)

        return images

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
            if pace_callback is not None:
                pace_callback()
            images = self.search(term, max_results=max_results, image_size=image_size, orderby=orderby)

            # Filter out non-illustration/decorative assets and images not relevant to the query term
            illustration_candidates = [
                img for img in images
                if not is_decorative_asset(img, term) and is_image_relevant(img, term)
            ]
            if illustration_candidates:
                ranked = sorted(illustration_candidates, key=lambda img: score_candidate_image(img, term), reverse=True)
                best = ranked[0]
                return ranked, cand, best

        return [], None, None


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
