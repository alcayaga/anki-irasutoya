"""Synonym dictionaries and visual concept mapping bridges backed by the SQLite catalog."""
import logging
from typing import Any, List, Optional, Set

from .catalog import CatalogStore, default_store

logger = logging.getLogger(__name__)


def lookup_synonyms(
    term: str,
    source: Optional[str] = None,
    store: Optional[CatalogStore] = None,
) -> List[str]:
    """Look up prioritized synonym / visual concept target queries for a given term from the catalog index.

    Args:
        term: Japanese vocabulary headword or phrase to look up.
        source: Optional source identifier to filter by (e.g. 'manual', 'donkuri/kaishi', 'sudachi').
        store: Optional CatalogStore instance (defaults to global default_store()).

    Returns:
        List of catalog-verified search queries, or empty list if no mapping exists.
    """
    if not term:
        return []
    clean = term.strip()
    if not clean:
        return []

    active_store = store or default_store()
    index = active_store.get_index()
    if index is None:
        return []

    return index.get_synonyms(clean, source=source)


def lookup_concept_map(
    term: str,
    store: Optional[CatalogStore] = None,
) -> List[str]:
    """Look up prioritized manual visual concept target queries for a given term.

    Args:
        term: Japanese vocabulary headword or phrase to look up.
        store: Optional CatalogStore instance (defaults to global default_store()).

    Returns:
        List of catalog-verified search queries from the 'manual' concept map source.
    """
    return lookup_synonyms(term, source="manual", store=store)


def lookup_kaishi_concept(
    term: str,
    store: Optional[CatalogStore] = None,
) -> List[str]:
    """Look up prioritized Kaishi 1.5k visual concept target queries for a given term.

    Args:
        term: Japanese vocabulary headword or phrase to look up.
        store: Optional CatalogStore instance (defaults to global default_store()).

    Returns:
        List of catalog-verified search queries from the 'donkuri/kaishi' concept map source.
    """
    return lookup_synonyms(term, source="donkuri/kaishi", store=store)


def lookup_sudachi_synonyms(
    term: str,
    store: Optional[CatalogStore] = None,
) -> List[str]:
    """Look up prioritized Sudachi synonym target queries for a given term.

    Args:
        term: Japanese vocabulary headword or phrase to look up.
        store: Optional CatalogStore instance (defaults to global default_store()).

    Returns:
        List of catalog-verified search queries from the 'sudachi' synonym dictionary source.
    """
    return lookup_synonyms(term, source="sudachi", store=store)


def lookup_bilingual_glosses(
    note: Any,
    yomitan_meta: Optional[Any] = None,
    store: Optional[CatalogStore] = None,
) -> List[str]:
    """Look up visual concept target queries bridged from English definition glosses on a note.

    Args:
        note: Anki note or field dictionary containing vocabulary definitions.
        yomitan_meta: Optional pre-parsed YomitanMetadata.
        store: Optional CatalogStore instance (defaults to global default_store()).

    Returns:
        List of catalog-verified search queries from the 'bilingual_gloss' source (up to 3 distinct targets).
    """
    if not note:
        return []
    from .yomitan_parser import extract_english_glosses_from_note

    glosses = extract_english_glosses_from_note(note, yomitan_meta=yomitan_meta)
    if not glosses:
        return []

    targets: List[str] = []
    seen: Set[str] = set()
    for gloss in glosses:
        for t in lookup_synonyms(gloss, source="bilingual_gloss", store=store):
            if t not in seen:
                seen.add(t)
                targets.append(t)
                if len(targets) >= 3:
                    return targets
    return targets

