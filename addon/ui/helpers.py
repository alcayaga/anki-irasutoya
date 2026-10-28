"""Headless helper functions for snippet formatting, match classification, and catalog probing."""
import html
import logging
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..feed import IrasutoyaImage
from ..ranking import (
    is_decorative_asset,
    is_image_relevant,
    is_title_match,
)
from ..text_utils import generate_search_candidates

logger = logging.getLogger(__name__)


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


def resolve_search_plan(
    query: str,
    card_source_term: str,
    card_candidates: Sequence[Tuple[str, str, int]],
    first_note: Optional[Any] = None,
    direct: bool = False,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]], Optional[str]]:
    """Determine display chips, search candidates, and restore term for a single-note query.

    Args:
        query: User query string or term selected from note.
        card_source_term: Raw term extracted from card's source field (if any).
        card_candidates: Precomputed candidate tuples from the note source field.
        first_note: Note instance for POS / furigana fallback extraction.
        direct: If True, execute a direct query without traversing alternative fallbacks.

    Returns:
        Tuple of (display_chips, search_candidates, restore_term).
    """
    card_terms = [c[0] for c in card_candidates]
    is_card_term = bool(card_source_term and (query == card_source_term or query in card_terms))

    if is_card_term:
        display_chips = list(card_candidates)
        restore_term = None
        if direct:
            matching_cand = next((c for c in card_candidates if c[0] == query), (query, "primary", 1))
            search_candidates = [matching_cand]
        elif query == card_source_term:
            search_candidates = list(card_candidates)
        else:
            search_candidates = generate_search_candidates(query, note=first_note, allow_tier2=True)
    else:
        candidates = generate_search_candidates(query, note=None, allow_tier2=True)
        display_chips = candidates
        matching_cand = next((c for c in candidates if c[0] == query), (query, "primary", 1))
        search_candidates = [matching_cand] if direct else candidates
        restore_term = (
            card_source_term
            if (card_source_term and card_source_term != query)
            else None
        )

    return display_chips, search_candidates, restore_term


def execute_search_and_probe(
    client: Any,
    store: Any,
    search_candidates: Sequence[Tuple[str, str, int]],
    display_chips: Sequence[Tuple[str, str, int]],
    image_size: str,
    restore_term: Optional[str],
    count_cache: Dict[str, int],
    desc_cache: Dict[str, bool],
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Tuple[List[IrasutoyaImage], Optional[Tuple[str, str, int]], Optional[Dict[str, int]], Optional[Set[str]]]:
    """Execute candidate search with fallbacks and probe catalog counts.

    Args:
        client: IrasutoyaClient instance.
        store: CatalogStore instance.
        search_candidates: Ordered candidate list to search.
        display_chips: Candidates to display as chips in UI.
        image_size: Target image size identifier (e.g. 's800').
        restore_term: Term to restore note source term if external query.
        count_cache: Per-size term count cache dict.
        desc_cache: Per-size description-only match cache dict.
        is_cancelled: Callable returning True if the search generation is stale.

    Returns:
        Tuple of (images, matched_candidate, probed_counts, probed_desc_matches).
    """
    if is_cancelled and is_cancelled():
        return [], None, None, None

    try:
        images, matched_cand, _best_img = client.search_with_fallbacks(
            search_candidates,
            max_results=None,
            image_size=image_size,
            is_cancelled=is_cancelled,
        )
    except Exception as e:
        logger.exception("Search failed: %s", e)
        images, matched_cand = [], None

    if is_cancelled and is_cancelled():
        return [], None, None, None

    probed_counts: Optional[Dict[str, int]] = None
    probed_desc_matches: Optional[Set[str]] = None
    try:
        catalog = store.get_index() if store is not None else None
        if catalog is not None:
            if matched_cand and images:
                count_cache[matched_cand[0]] = len(images)
                desc_cache[matched_cand[0]] = _is_desc_only_match(images, matched_cand[0])

            probe_terms = [c[0] for c in display_chips]
            if matched_cand and matched_cand[0] not in probe_terms:
                probe_terms.append(matched_cand[0])

            probed_counts, probed_desc_matches = probe_catalog_candidates(
                catalog,
                image_size,
                probe_terms,
                restore_term,
                count_cache,
                desc_cache,
                is_stale=is_cancelled,
            )
    except _StaleSearch:
        return [], None, None, None
    except Exception:
        logger.exception("Catalog count probe failed")

    return images, matched_cand, probed_counts, probed_desc_matches
