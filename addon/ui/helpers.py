"""Headless helper functions for snippet formatting, match classification, and catalog probing."""
import html
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..feed import IrasutoyaImage
from ..ranking import is_decorative_asset, is_image_relevant, is_title_match


def _get_mw() -> Any:
    """Return the active Anki main window instance, respecting mock patches on addon.ui.mw."""
    ui_mod = sys.modules.get("addon.ui")
    if ui_mod is not None and hasattr(ui_mod, "mw"):
        return ui_mod.mw
    try:
        from aqt import mw
        return mw
    except Exception:
        return None


class _StaleSearch(Exception):
    """Internal signal that the current search generation was superseded."""


def make_description_snippet(description: str, query: str, max_chars: int = 70) -> str:
    """Generate an excerpt around the query term with bold highlight for tooltips."""
    if not description or not query:
        return ""
    clean_desc = re.sub(r"\s+", " ", description).strip()
    match = re.search(re.escape(query), clean_desc, re.IGNORECASE)
    if match is None:
        return html.escape(clean_desc[:max_chars])
    m_start, m_end = match.span()
    start = max(0, m_start - 18)
    end = min(len(clean_desc), m_end + 38)
    prefix = ("…" if start > 0 else "") + html.escape(clean_desc[start:m_start])
    body = "<b>" + html.escape(clean_desc[m_start:m_end]) + "</b>"
    suffix = html.escape(clean_desc[m_end:end]) + ("…" if end < len(clean_desc) else "")
    return prefix + body + suffix


def _description_match_term(img: IrasutoyaImage, term: str) -> Optional[str]:
    """Return the query term to highlight if img matches term in description only, else None."""
    clean_t = term.strip()
    if not clean_t or is_title_match(img, clean_t):
        return None
    desc = (img.description or "").casefold()
    tokens = clean_t.split()
    if tokens and all(t.casefold() in desc for t in tokens):
        return tokens[0]
    return None


def _is_desc_only_match(images: Sequence[IrasutoyaImage], term: str) -> bool:
    """Return True if all images match term in description only (not in title or post_title with boundary safety)."""
    clean_t = term.strip()
    if not clean_t or not images:
        return False
    return all(_description_match_term(img, clean_t) is not None for img in images)


def has_existing_media(content: str) -> bool:
    """Return True if note field content already contains an HTML image tag or image filename."""
    if not content:
        return False
    content_lower = content.strip().lower()
    if "<img" in content_lower:
        return True
    return any(
        content_lower.endswith(ext)
        for ext in (".jpg", ".png", ".webp", ".jpeg", ".gif")
    )


def format_fallback_reason(
    query: str,
    probed_counts: Optional[Dict[str, int]] = None,
    probed_desc_matches: Optional[Set[str]] = None,
) -> str:
    """Format explanation reason for why a fallback term is being displayed instead of the direct query."""
    count = probed_counts.get(query) if probed_counts else None
    if count is None:
        return f"No direct title match for '{query}'"
    if count == 0:
        return f"No direct results for '{query}'"
    if probed_desc_matches and query in probed_desc_matches:
        return f"No direct title match for '{query}'"
    if probed_desc_matches is not None and count > 0:
        return f"No direct title illustration for '{query}'"
    return f"No direct title match for '{query}'"


def probe_catalog_term(
    catalog: Any,
    term: str,
    image_size: str,
    count_cache: Dict[str, int],
    desc_cache: Dict[str, bool],
    is_stale: Optional[Callable[[], bool]] = None,
) -> int:
    """Probe catalog result count and description match status for a single term with caching."""
    if is_stale and is_stale():
        raise _StaleSearch()
    if term not in count_cache:
        raw_imgs = catalog.search(term, image_size=image_size)
        rel_imgs = [
            img for img in raw_imgs
            if not is_decorative_asset(img, term) and is_image_relevant(img, term)
        ]
        count_cache[term] = len(rel_imgs)
        desc_cache[term] = _is_desc_only_match(rel_imgs, term)
    return count_cache[term]


def probe_catalog_candidates(
    catalog: Any,
    image_size: str,
    candidate_terms: Sequence[str],
    restore_term: Optional[str],
    count_cache: Dict[str, int],
    desc_cache: Dict[str, bool],
    is_stale: Optional[Callable[[], bool]] = None,
) -> Tuple[Dict[str, int], Set[str]]:
    """Probe catalog result counts and description-only matches for candidates and restore term."""
    terms_to_probe: List[str] = list(candidate_terms)
    if restore_term and restore_term not in terms_to_probe:
        terms_to_probe.append(restore_term)

    counts: Dict[str, int] = {}
    for t in terms_to_probe:
        counts[t] = probe_catalog_term(
            catalog, t, image_size, count_cache, desc_cache, is_stale=is_stale
        )

    desc_matches = {t for t in counts if desc_cache.get(t)}
    return counts, desc_matches
