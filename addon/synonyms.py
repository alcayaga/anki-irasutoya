"""Synonym dictionaries and visual concept mapping bridges backed by the SQLite catalog."""
import logging
import re
from typing import Any, List, Optional, Set

from .catalog import (
    CatalogStore,
    default_store,
)
from .synonym_constants import (
    BILINGUAL_ACTION_TARGETS,
    ENGLISH_NOUN_ONLY_COLLISIONS,
    ENGLISH_PREPOSITIONS,
    ENGLISH_STOP_WORDS,
    LOCOMOTION_ACTION_TARGETS,
    LOCOMOTION_ACTION_WORDS,
    MAX_BILINGUAL_TARGETS,
    MIN_GLOSS_TOKEN_LENGTH,
    NEGATION_OR_IDIOM_TOKENS,
    SPATIAL_MARGIN_GLOSSES,
    STYLE_SCHOOL_GLOSSES,
)
from .synonym_constants import (
    PASS2_BLOCKED_PREFIXES as PASS2_BLOCKED_PREFIXES,
)

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


def lookup_llm_concept(
    term: str,
    store: Optional[CatalogStore] = None,
) -> List[str]:
    """Look up prioritized LLM-curated visual concept target queries for a given term.

    Args:
        term: Japanese vocabulary headword or phrase to look up.
        store: Optional CatalogStore instance (defaults to global default_store()).

    Returns:
        List of catalog-verified search queries from the 'llm_concept' source.
    """
    return lookup_synonyms(term, source="llm_concept", store=store)


def lookup_llm_concept_tier2(
    term: str,
    store: Optional[CatalogStore] = None,
) -> List[str]:
    """Look up Tier 2 visual exemplar and metaphorical concept target queries.

    Args:
        term: Japanese vocabulary headword or phrase to look up.
        store: Optional CatalogStore instance (defaults to global default_store()).

    Returns:
        List of catalog-verified search queries from the 'llm_concept_tier2' source.
    """
    return lookup_synonyms(term, source="llm_concept_tier2", store=store)


def get_llm_rejection_reason(
    term: str,
    store: Optional[CatalogStore] = None,
) -> Optional[str]:
    """Return the rejection reason string if term was rejected by the LLM as non-visual, else None.

    Args:
        term: Japanese vocabulary headword or phrase to look up.
        store: Optional CatalogStore instance (defaults to global default_store()).

    Returns:
        Rejection reason string, or None if the term was not rejected.
    """
    if not term:
        return None
    active_store = store or default_store()
    index = active_store.get_index()
    if index is None:
        return None
    return index.get_rejection_reason(term)


def is_llm_rejected(
    term: str,
    store: Optional[CatalogStore] = None,
) -> bool:
    """Return True if term was rejected by the LLM as non-visual and not merely unverified.

    Args:
        term: Japanese vocabulary headword or phrase to look up.
        store: Optional CatalogStore instance (defaults to global default_store()).

    Returns:
        True if rejected as non-visual, False otherwise.
    """
    if not term:
        return False
    active_store = store or default_store()
    index = active_store.get_index()
    if index is None:
        return False
    return index.is_rejected(term)


def _is_verbal_gloss_phrase(gloss: str) -> bool:
    """Return True if a gloss phrase represents a verbal or infinitive action.

    Args:
        gloss: Normalized or raw English gloss string.

    Returns:
        True if the gloss phrase indicates an infinitive, locomotion verb, or action collision.
    """
    if not gloss:
        return False
    g_str = gloss.strip().lower()
    if g_str.startswith("to "):
        return True
    parts = g_str.split()
    if len(parts) > 1 and parts[0] in (LOCOMOTION_ACTION_WORDS | ENGLISH_NOUN_ONLY_COLLISIONS):
        return True
    if len(parts) == 1 and parts[0] in LOCOMOTION_ACTION_WORDS:
        return True
    return False


def _is_noun_note(yomitan_meta: Optional[Any], note: Any = None) -> bool:
    """Determine whether the note represents a noun from Yomitan metadata or note definition fields.

    Args:
        yomitan_meta: Optional pre-parsed YomitanMetadata.
        note: Optional Anki note or field dictionary.

    Returns:
        True if noun POS tags or nominal definition structures are detected, False otherwise.
    """
    if yomitan_meta is not None:
        tags = getattr(yomitan_meta, "pos_tags", None) or set()
        if any(t == "n" or t.startswith(("n-", "adj-no")) for t in tags):
            if any(t.startswith("vs") for t in tags) and "vt" in tags:
                glosses = getattr(yomitan_meta, "english_glosses", None) or []
                if glosses and all(_is_verbal_gloss_phrase(g) for g in glosses):
                    return False
            return True
        if tags:
            return False
    return not _is_verb_note(yomitan_meta, note)


def _is_verb_note(yomitan_meta: Optional[Any], note: Any = None) -> bool:
    """Determine whether the note represents a verb from Yomitan metadata or note definition fields.

    Args:
        yomitan_meta: Optional pre-parsed YomitanMetadata.
        note: Optional Anki note or field dictionary.

    Returns:
        True if verb POS tags or infinitive 'to <verb>' patterns are detected, False otherwise.
    """
    if yomitan_meta is not None:
        tags = getattr(yomitan_meta, "pos_tags", None) or set()
        # If the note has noun classification 'n', suru-nouns ('vs') remain noun-eligible,
        # unless transitive ('vt') with purely verbal glosses
        if "n" in tags and any(t.startswith("vs") for t in tags):
            if any(t == "v" or t.startswith(("v1", "v5", "vk", "vz")) for t in tags):
                return True
            if "vt" in tags:
                glosses = getattr(yomitan_meta, "english_glosses", None) or []
                if glosses and all(_is_verbal_gloss_phrase(g) for g in glosses):
                    return True
            return False
        if getattr(yomitan_meta, "verb_class", None):
            return True
        if any(t == "v" or t.startswith(("v1", "v5", "vs", "vk", "vz", "vi", "vt")) for t in tags):
            return True
        if "n" in tags:
            return False
    if note and hasattr(note, "__getitem__"):
        from .yomitan_parser import NOTE_DEFINITION_FIELDS, strip_html_excluded_containers

        for f in NOTE_DEFINITION_FIELDS + ("English", "SelectionText"):
            if f in note and note[f]:
                val = strip_html_excluded_containers(str(note[f]))
                pattern = (
                    r"(?i)(?:^|[;\n,(])\s*(?:to\s+[a-z]+|\(v(?:erb|[15ktsz])?\))"
                    if f in NOTE_DEFINITION_FIELDS
                    else r"(?i)^[\s;]*to\s+[a-z]+"
                )
                if re.search(pattern, val):
                    return True
    return False


def _is_action_target(target: str) -> bool:
    """Return True if target represents an action, verb, or event illustration.

    Args:
        target: Japanese candidate query target.

    Returns:
        True if target is an allowed action target or infers to verb POS, False otherwise.
    """
    if not target:
        return False
    if target in BILINGUAL_ACTION_TARGETS:
        return True
    from .ranking import infer_pos_type

    return infer_pos_type(target) == "verb"


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
    pos_tags = getattr(yomitan_meta, "pos_tags", set()) if yomitan_meta is not None else set()
    is_transitive_only = ("vt" in pos_tags) and ("vi" not in pos_tags)
    is_suru_transitive = bool(
        pos_tags and "n" in pos_tags and any(t.startswith("vs") for t in pos_tags) and "vt" in pos_tags
    )
    has_spatial_margin = any(g in SPATIAL_MARGIN_GLOSSES for g in glosses)
    has_style_school = any(g in STYLE_SCHOOL_GLOSSES for g in glosses)

    # Pass 1: exact matches on full normalized glosses
    for gloss in glosses:
        if has_spatial_margin and gloss == "room":
            continue
        if has_style_school and gloss == "school":
            continue
        for t in lookup_synonyms(gloss, source="bilingual_gloss", store=store):
            if is_verb and not _is_action_target(t):
                continue
            if is_transitive_only and t in LOCOMOTION_ACTION_TARGETS:
                continue
            if t not in seen:
                seen.add(t)
                targets.append(t)
                if len(targets) >= MAX_BILINGUAL_TARGETS:
                    return targets

    # Pass 2: headword and constituent words from multi-word glosses
    # Restricted to noun notes to prevent verb/expression constituent splitting collisions
    if not is_verb and _is_noun_note(yomitan_meta, note):
        for gloss in glosses:
            if (is_suru_transitive or is_transitive_only) and _is_verbal_gloss_phrase(gloss):
                continue
            # Skip glosses containing negation/politeness, coordinate 'or', or partitive/idiomatic prefixes
            if any(tok in gloss.split() for tok in NEGATION_OR_IDIOM_TOKENS):
                continue
            if " or " in gloss:
                continue
            if any(f"{gloss} ".startswith(f"{p} ") for p in PASS2_BLOCKED_PREFIXES):
                continue

            raw_words = gloss.split()
            if len(raw_words) <= 1:
                continue
            words = [w for w in raw_words if w not in ENGLISH_STOP_WORDS and len(w) >= MIN_GLOSS_TOKEN_LENGTH]
            if not words:
                continue
            if len(words) == 1:
                ordered_words = words
            else:
                has_prep = any(p in raw_words for p in ENGLISH_PREPOSITIONS)
                if has_prep:
                    ordered_words = [words[0], words[-1]] + words[1:-1]
                else:
                    ordered_words = [words[-1], words[0]] + words[1:-1]
            for w in dict.fromkeys(ordered_words):
                if has_spatial_margin and w == "room":
                    continue
                if has_style_school and w == "school":
                    continue
                for t in lookup_synonyms(w, source="bilingual_gloss", store=store):
                    if is_transitive_only and t in LOCOMOTION_ACTION_TARGETS:
                        continue
                    if t not in seen:
                        seen.add(t)
                        targets.append(t)
                        if len(targets) >= MAX_BILINGUAL_TARGETS:
                            return targets

    return targets

