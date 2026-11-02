"""Synonym dictionaries and visual concept mapping bridges backed by the SQLite catalog."""
import logging
import re
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


ENGLISH_STOP_WORDS = frozenset({
    "in", "of", "for", "a", "an", "the", "to", "at", "on", "by", "with",
    "and", "or", "from", "as", "about", "into", "over", "after", "one's",
    "sb", "sth", "someone", "something", "etc", "esp", "chiefly", "especially",
})
ENGLISH_PREPOSITIONS = frozenset({"in", "of", "for", "with", "at", "from", "on", "by", "about"})
ENGLISH_NOUN_ONLY_COLLISIONS = frozenset({"train", "weather"})


def _is_verb_note(yomitan_meta: Optional[Any], note: Any = None) -> bool:
    """Determine whether the note represents a verb from Yomitan metadata or note definition fields.

    Args:
        yomitan_meta: Optional pre-parsed YomitanMetadata.
        note: Optional Anki note or field dictionary.

    Returns:
        True if verb POS tags or infinitive 'to <verb>' patterns are detected, False otherwise.
    """
    if yomitan_meta is not None:
        if getattr(yomitan_meta, "verb_class", None):
            return True
        tags = getattr(yomitan_meta, "pos_tags", None) or set()
        if any(t == "v" or t.startswith(("v1", "v5", "vs", "vk", "vz", "vi", "vt")) for t in tags):
            return True
        if "n" in tags:
            return False
    if note and hasattr(note, "__getitem__"):
        from .yomitan_parser import NOTE_DEFINITION_FIELDS, strip_html_excluded_containers

        for f in NOTE_DEFINITION_FIELDS + ("English",):
            if f in note and note[f]:
                val = strip_html_excluded_containers(str(note[f]))
                if re.search(r"(?i)(?:^|[;\n,(])\s*(?:to\s+[a-z]+|\(v(?:erb|[15ktsz])?\))", val):
                    return True
    return False


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
    is_verb = _is_verb_note(yomitan_meta, note)

    # Pass 1: exact matches on full normalized glosses
    for gloss in glosses:
        if is_verb and gloss in ENGLISH_NOUN_ONLY_COLLISIONS:
            continue
        for t in lookup_synonyms(gloss, source="bilingual_gloss", store=store):
            if t not in seen:
                seen.add(t)
                targets.append(t)
                if len(targets) >= 3:
                    return targets

    # Pass 2: headword and constituent words from multi-word glosses
    for gloss in glosses:
        raw_words = gloss.split()
        if len(raw_words) <= 1:
            continue
        words = [w for w in raw_words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
        if not words:
            continue
        if len(words) == 1:
            ordered_words = words
        else:
            # In English prepositional phrases ("doctor in charge", "doctor of medicine"),
            # the head noun is the first word. In compound noun phrases ("family doctor", "sports car"),
            # the head noun is the last word.
            has_prep = any(p in raw_words for p in ENGLISH_PREPOSITIONS)
            if has_prep:
                ordered_words = [words[0], words[-1]] + words[1:-1]
            else:
                ordered_words = [words[-1], words[0]] + words[1:-1]
        for w in dict.fromkeys(ordered_words):
            if is_verb and w in ENGLISH_NOUN_ONLY_COLLISIONS:
                continue
            for t in lookup_synonyms(w, source="bilingual_gloss", store=store):
                if t not in seen:
                    seen.add(t)
                    targets.append(t)
                    if len(targets) >= 3:
                        return targets

    return targets

