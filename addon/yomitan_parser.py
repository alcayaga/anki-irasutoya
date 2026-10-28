"""Parser for extracting structured JMdict metadata from Yomitan note HTML fields."""
import html
import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, List, Optional, Sequence, Set, Tuple

# Common Yomitan definition/glossary field names in popular note models (Lapis, Senren, Jlab, etc.)
NOTE_DEFINITION_FIELDS = (
    "MainDefinition",
    "Glossary",
    "Definition",
    "Definitions",
    "Meanings",
    "Word Meaning",
    "Sentence Meaning",
)

GRAMMAR_POS_TAGS = frozenset({
    "prt", "conj", "aux", "aux-v", "aux-adj"
})

VERB_POS_PATTERN = re.compile(
    r"^(v1(-s)?|v5[a-z]+(-[a-z]+)?|vs|vs-i|vs-s|vs-c|vk)$"
)


@dataclass
class YomitanMetadata:
    """Structured JMdict metadata extracted from Yomitan HTML."""
    pos_tags: Set[str] = field(default_factory=set)
    is_uk: bool = False
    is_particle_or_aux: bool = False
    verb_class: Optional[str] = None
    forms_star: List[str] = field(default_factory=list)
    forms_all: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    sense_kanji: List[str] = field(default_factory=list)
    english_glosses: List[str] = field(default_factory=list)


def normalize_english_gloss(text: str) -> str:
    """Normalize an English definition gloss into a clean 1-4 word search key.

    Performs Unicode NFKC normalization, strips parentheticals/brackets, lowercases,
    strips leading articles / infinitive / copula markers (to, be, a, an, the),
    cleans edge punctuation, and rejects non-Latin phrases, Japanese text, or phrases > 4 words.

    Args:
        text: Raw English gloss string (e.g. "to pass away", "to get angry (at someone)").

    Returns:
        Canonical English gloss string (e.g. "pass away", "get angry"), or empty string if invalid.
    """
    if not text or not isinstance(text, str):
        return ""

    s = unicodedata.normalize("NFKC", text).strip()
    if not s:
        return ""

    # Strip parentheticals and bracketed qualifiers (e.g. "(at someone)", "[archaic]", "(猫)")
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", s)

    # Reject if text contains Japanese characters (Kanji, Kana, punctuation)
    if re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff々〆〇ー]", s):
        return ""

    # Lowercase
    s = s.casefold().strip()

    # Repeatedly strip leading articles, infinitive, and copula markers
    while True:
        prev = s
        s = re.sub(r"^(?:to|be|a|an|the)\s+", "", s)
        if s == prev:
            break

    # Strip leading/trailing non-alphanumeric punctuation (retaining inner hyphens/apostrophes)
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()

    # Must contain at least one cased letter
    if not any(ch.casefold() != ch.upper() for ch in s):
        return ""

    # Reject phrases with > 4 words
    words = s.split()
    if not (1 <= len(words) <= 4):
        return ""

    return s


def _split_and_normalize_glosses(raw_text: str) -> List[str]:
    """Strip parentheticals and split by delimiter, returning canonical glosses.

    Args:
        raw_text: Raw gloss string extracted from HTML markup.

    Returns:
        List of canonical English gloss strings.
    """
    if not raw_text:
        return []
    # Pre-strip parentheticals/brackets to prevent delimiters inside them from splitting phrases
    cleaned = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", raw_text)
    glosses: List[str] = []
    for chunk in re.split(r"[;,/\n]", cleaned):
        norm = normalize_english_gloss(chunk)
        if norm and norm not in glosses:
            glosses.append(norm)
    return glosses


OUTDATED_FORM_MARKERS = frozenset({"⚠", "×", "🅁", "⛬", "㊒"})
OUTDATED_TAG_PATTERN = re.compile(
    r"\b(rk|sk|ok|ik|rare|outdated|irregular|search-only)\b", re.IGNORECASE
)


class YomitanHTMLParser(HTMLParser):
    """Standard-library HTMLParser extracting POS, formsTable, references, and sense notes."""

    def __init__(self) -> None:
        super().__init__()
        self.metadata = YomitanMetadata()

        # POS parsing state: <i>(1, v1, vt, uk, JMdict [2025-11-01])</i>
        self._in_i = False
        self._current_i_text: List[str] = []
        self._sense_tags: List[Set[str]] = []

        # formsTable parsing state: <table data-sc-content="formsTable">
        self._in_forms_table = False
        self._table_headers: List[str] = []
        self._current_row_idx = 0
        self._current_col_idx = 0
        self._in_th = False
        self._in_td = False
        self._current_cell_text: List[str] = []
        self._current_cell_class = ""
        self._forms_cols_valid: List[bool] = []
        self._forms_cols_has_star: List[bool] = []
        self._forms_cols_rare_header: List[bool] = []

        # Simple forms list: <i>(forms...)</i> <ul><li>手袋（★）</li></ul>
        self._in_forms_entry = False
        self._in_forms_li = False
        self._current_forms_li_text: List[str] = []

        # references parsing state: <ul data-sc-content="references"><li>see: ...
        self._in_ref_ul = False
        self._in_ref_li = False
        self._in_ref_gloss = False
        self._current_ref_text: List[str] = []

        # sense notes parsing state: <ul data-sc-content="notes"><li>esp. ...
        self._in_notes_ul = False
        self._in_notes_li = False
        self._current_note_text: List[str] = []

        # Structured glossary parsing state: <ul data-sc-content="glossary"><li>...</li></ul>
        self._in_glossary_ul = False
        self._in_glossary_li = False
        self._current_gloss_text: List[str] = []

        # Legacy span parsing state: <span>...</span>
        self._legacy_span_depth = 0
        self._current_span_text: List[str] = []
        self._legacy_spans: List[str] = []

        # Ancestor container tracking for blocks to exclude (examples, antonyms)
        self._excluded_ancestors: List[str] = []

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        attr_dict = dict(attrs)
        sc_content = attr_dict.get("data-sc-content", "") or ""

        # Track containers explicitly excluded from definition glosses (e.g. examples, antonyms)
        is_excluded_container = sc_content in ("examples", "antonyms")
        if is_excluded_container or self._excluded_ancestors:
            self._excluded_ancestors.append(tag)

        if tag == "i":
            self._in_i = True
            self._current_i_text = []

        # formsTable detection
        if tag == "table" and sc_content == "formsTable":
            self._in_forms_table = True
            self._table_headers = []
            self._current_row_idx = 0
            self._forms_cols_valid = []
            self._forms_cols_has_star = []
            self._forms_cols_rare_header = []

        if self._in_forms_table:
            if tag == "tr":
                self._current_col_idx = 0
            elif tag in ("th", "td"):
                self._in_th = (tag == "th")
                self._in_td = (tag == "td")
                self._current_cell_text = []
                self._current_cell_class = f"{attr_dict.get('class', '')} {attr_dict.get('data-sc-class', '')}"

        # Simple forms list detection
        if self._in_forms_entry and tag == "li":
            self._in_forms_li = True
            self._current_forms_li_text = []

        # references detection
        if tag == "ul" and sc_content == "references":
            self._in_ref_ul = True
        if self._in_ref_ul and tag == "li":
            self._in_ref_li = True
            self._in_ref_gloss = False
            self._current_ref_text = []
        if self._in_ref_li and sc_content == "refGlosses":
            self._in_ref_gloss = True

        # notes detection
        if tag == "ul" and sc_content == "notes":
            self._in_notes_ul = True
        if self._in_notes_ul and tag == "li":
            self._in_notes_li = True
            self._current_note_text = []

        # Structured glossary detection: <ul data-sc-content="glossary">
        if tag == "ul" and sc_content == "glossary":
            self._in_glossary_ul = True
        if self._in_glossary_ul and tag == "li":
            self._in_glossary_li = True
            self._current_gloss_text = []

        # Legacy definition span detection: <span>gloss text</span>
        if (
            tag == "span"
            and not self._in_i
            and not self._in_forms_table
            and not self._in_forms_entry
            and not self._in_ref_ul
            and not self._in_notes_ul
            and not self._in_glossary_ul
            and not self._excluded_ancestors
            and sc_content not in ("references", "refGlosses", "notes", "formsTable", "antonyms", "examples", "glossary")
        ):
            self._legacy_span_depth += 1
            if self._legacy_span_depth == 1:
                self._current_span_text = []
        elif self._legacy_span_depth > 0 and tag == "span" and not self._excluded_ancestors:
            self._legacy_span_depth += 1

    def handle_endtag(self, tag: str) -> None:
        was_excluded = bool(self._excluded_ancestors)
        if self._excluded_ancestors:
            if tag == self._excluded_ancestors[-1]:
                self._excluded_ancestors.pop()
            elif tag in self._excluded_ancestors:
                while self._excluded_ancestors and self._excluded_ancestors[-1] != tag:
                    self._excluded_ancestors.pop()
                if self._excluded_ancestors:
                    self._excluded_ancestors.pop()

        if tag == "i" and self._in_i:
            self._in_i = False
            raw = "".join(self._current_i_text).strip()
            m = re.match(r"^\((.*)\)$", raw)
            if m:
                inner = m.group(1)
                inner = re.sub(r"JMdict(\s*\[.*?\])?", "", inner, flags=re.IGNORECASE)
                parts = [p.strip() for p in inner.split(",") if p.strip()]
                # Check for (forms...) entry indicator
                if "forms" in parts:
                    self._in_forms_entry = True
                sense_tags = {p for p in parts if not p.isdigit()}
                if parts and parts[0].isdigit():
                    is_first_sense = (parts[0] == "1")
                else:
                    is_first_sense = not self._sense_tags
                if sense_tags and "forms" not in sense_tags:
                    self._sense_tags.append(sense_tags)
                for p in dict.fromkeys(x for x in parts if not x.isdigit()):
                    self.metadata.pos_tags.add(p)
                    if p == "uk" and is_first_sense:
                        self.metadata.is_uk = True
                    if not self.metadata.verb_class and VERB_POS_PATTERN.match(p):
                        self.metadata.verb_class = p

        # Handle formsTable
        if self._in_forms_table:
            if tag in ("th", "td"):
                cell_text = "".join(self._current_cell_text).strip()
                if self._current_row_idx == 0:
                    if self._current_col_idx > 0:
                        is_rare_header = (
                            cell_text.startswith("〈")
                            or cell_text.endswith("〉")
                            or any(m in cell_text for m in OUTDATED_FORM_MARKERS)
                            or bool(OUTDATED_TAG_PATTERN.search(cell_text))
                            or bool(OUTDATED_TAG_PATTERN.search(self._current_cell_class))
                        )
                        clean_hdr = re.sub(r"^[〈（(\[]|[\s〉）)\]]+$", "", cell_text).strip()
                        clean_hdr = html.unescape(clean_hdr).strip()
                        self._table_headers.append(clean_hdr)
                        self._forms_cols_rare_header.append(is_rare_header)
                        self._forms_cols_has_star.append(False)
                        self._forms_cols_valid.append(False)
                else:
                    col_offset = self._current_col_idx - 1
                    if 0 <= col_offset < len(self._table_headers):
                        is_rare_cell = (
                            any(m in cell_text for m in OUTDATED_FORM_MARKERS)
                            or bool(OUTDATED_TAG_PATTERN.search(cell_text))
                            or bool(OUTDATED_TAG_PATTERN.search(self._current_cell_class))
                            or cell_text == "×"
                        )
                        is_star = ("★" in cell_text) and not is_rare_cell
                        if is_star:
                            self._forms_cols_has_star[col_offset] = True
                        if not is_rare_cell and (cell_text != "" or is_star):
                            self._forms_cols_valid[col_offset] = True
                self._current_col_idx += 1
                self._in_th = False
                self._in_td = False
            elif tag == "tr":
                self._current_row_idx += 1
            elif tag == "table":
                for col_offset, cand in enumerate(self._table_headers):
                    is_header_rare = (
                        self._forms_cols_rare_header[col_offset]
                        if col_offset < len(self._forms_cols_rare_header)
                        else False
                    )
                    has_star = (
                        self._forms_cols_has_star[col_offset]
                        if col_offset < len(self._forms_cols_has_star)
                        else False
                    )
                    is_valid = (
                        self._forms_cols_valid[col_offset]
                        if col_offset < len(self._forms_cols_valid)
                        else False
                    )

                    if has_star and not is_header_rare:
                        if cand and cand not in self.metadata.forms_star:
                            self.metadata.forms_star.append(cand)

                    if not is_header_rare and (has_star or is_valid):
                        if cand and cand not in self.metadata.forms_all:
                            self.metadata.forms_all.append(cand)

                self._in_forms_table = False

        # Handle simple forms list <li>
        if self._in_forms_entry:
            if tag == "li" and self._in_forms_li:
                self._in_forms_li = False
                li_text = "".join(self._current_forms_li_text).strip()
                is_rare_or_outdated = (
                    li_text.startswith("〈")
                    or any(m in li_text for m in OUTDATED_FORM_MARKERS)
                    or bool(OUTDATED_TAG_PATTERN.search(li_text))
                )
                if not is_rare_or_outdated:
                    if "★" in li_text:
                        clean_term = re.sub(r"[（(]★[)）]|\s*★\s*", "", li_text).strip()
                        clean_term = re.sub(r"^[〈（(\[]|[\s〉）)\]]+$", "", clean_term).strip()
                        clean_term = html.unescape(clean_term).strip()
                        if clean_term and clean_term not in self.metadata.forms_star:
                            self.metadata.forms_star.append(clean_term)

                    clean_form = re.sub(r"[（(]★[)）]|\s*★\s*", "", li_text).strip()
                    clean_form = re.sub(r"[（(\[][^）)\]]*[）)\]]", "", clean_form).strip()
                    clean_form = re.sub(r"^[〈（(\[]|[\s〉）)\]]+$", "", clean_form).strip()
                    clean_form = html.unescape(clean_form).strip()
                    if clean_form and clean_form not in self.metadata.forms_all:
                        self.metadata.forms_all.append(clean_form)
            elif tag in ("ul", "ol", "li"):
                # end of forms block
                self._in_forms_entry = False

        # Handle references
        if self._in_ref_ul:
            if tag == "li" and self._in_ref_li:
                self._in_ref_li = False
                raw_ref = "".join(self._current_ref_text).strip()
                # Remove "see:" prefix and parenthesized readings: e.g. see: 博士（はくし） -> 博士
                clean_ref = re.sub(r"^see(?:\s+also)?:\s*", "", raw_ref, flags=re.IGNORECASE)
                clean_ref = re.sub(r"[（(][^）)]*[）)]", "", clean_ref).strip()
                clean_ref = html.unescape(clean_ref).strip()
                if clean_ref and clean_ref not in self.metadata.references:
                    self.metadata.references.append(clean_ref)
            elif tag == "ul":
                self._in_ref_ul = False

        # Handle sense notes
        if self._in_notes_ul:
            if tag == "li" and self._in_notes_li:
                self._in_notes_li = False
                raw_note = "".join(self._current_note_text).strip()
                m = re.match(r"^esp\.\s*(.*)", raw_note, flags=re.IGNORECASE)
                if m:
                    split_terms = [t.strip() for t in m.group(1).split(",") if t.strip()]
                    for t in split_terms:
                        clean_t = re.sub(r"^[〈（(]|[\s〉）)]+$", "", t).strip()
                        clean_t = html.unescape(clean_t).strip()
                        if (
                            re.fullmatch(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff々〆〇ー]+", clean_t)
                            and clean_t not in self.metadata.sense_kanji
                        ):
                            self.metadata.sense_kanji.append(clean_t)
            elif tag == "ul":
                self._in_notes_ul = False

        # Handle structured glossary
        if self._in_glossary_ul:
            if tag == "li" and self._in_glossary_li:
                self._in_glossary_li = False
                raw_gloss = "".join(self._current_gloss_text).strip()
                for norm in _split_and_normalize_glosses(raw_gloss):
                    if norm not in self.metadata.english_glosses:
                        self.metadata.english_glosses.append(norm)
            elif tag == "ul":
                self._in_glossary_ul = False

        # Handle legacy span
        if self._legacy_span_depth > 0 and tag == "span" and not was_excluded:
            self._legacy_span_depth -= 1
            if self._legacy_span_depth == 0:
                raw_span = "".join(self._current_span_text).strip()
                if raw_span:
                    self._legacy_spans.append(raw_span)

    def handle_data(self, data: str) -> None:
        if self._in_i:
            self._current_i_text.append(data)
        if self._in_forms_table and (self._in_th or self._in_td):
            self._current_cell_text.append(data)
        if self._in_forms_entry and self._in_forms_li:
            self._current_forms_li_text.append(data)
        if self._in_ref_li and not self._in_ref_gloss:
            self._current_ref_text.append(data)
        if self._in_notes_li:
            self._current_note_text.append(data)
        if self._in_glossary_li and not self._excluded_ancestors:
            self._current_gloss_text.append(data)
        elif self._legacy_span_depth > 0 and not self._excluded_ancestors:
            self._current_span_text.append(data)


def parse_yomitan_html(html_text: Optional[str]) -> Optional[YomitanMetadata]:
    """Parse Yomitan HTML string and return extracted YomitanMetadata."""
    if not html_text or not isinstance(html_text, str):
        return None
    # Fast check for Yomitan/JMdict markup
    if not any(marker in html_text for marker in ("JMdict", "data-sc-content", "formsTable", "yomitan")):
        return None
    parser = YomitanHTMLParser()
    try:
        parser.feed(html_text)
        meta = parser.metadata
        meta.is_particle_or_aux = bool(parser._sense_tags) and all(
            bool(tags & GRAMMAR_POS_TAGS) for tags in parser._sense_tags
        )
        # If no structured glossary entries were found, populate from legacy spans
        if not meta.english_glosses and parser._legacy_spans:
            for span_text in parser._legacy_spans:
                for norm in _split_and_normalize_glosses(span_text):
                    if norm not in meta.english_glosses:
                        meta.english_glosses.append(norm)
        return meta
    except Exception:
        return None


def _has_data(meta: Optional[YomitanMetadata]) -> bool:
    """Return True if meta contains any parsed POS tags, forms, references, sense notes, or glosses.

    Args:
        meta: Optional YomitanMetadata instance to inspect.

    Returns:
        True if any metadata fields are populated, False otherwise.
    """
    return bool(
        meta and (
            meta.pos_tags
            or meta.forms_star
            or meta.forms_all
            or meta.references
            or meta.sense_kanji
            or meta.english_glosses
        )
    )


def _has_structured_data(meta: Optional[YomitanMetadata]) -> bool:
    """Return True if meta contains structured grammatical or orthographic JMdict metadata.

    Args:
        meta: Optional YomitanMetadata instance to inspect.

    Returns:
        True if POS tags, forms, references, or sense notes are present.
    """
    return bool(
        meta and (
            meta.pos_tags
            or meta.forms_star
            or meta.forms_all
            or meta.references
            or meta.sense_kanji
        )
    )


def _merge_metadata(target: YomitanMetadata, source: YomitanMetadata) -> None:
    """Merge structured JMdict fields and English glosses from source into target in place.

    Args:
        target: Target YomitanMetadata instance to mutate.
        source: Source YomitanMetadata instance whose attributes should be merged in.
    """
    target.pos_tags.update(source.pos_tags)
    if source.is_uk:
        target.is_uk = True
    if source.is_particle_or_aux:
        target.is_particle_or_aux = True
    if not target.verb_class and source.verb_class:
        target.verb_class = source.verb_class
    for f in source.forms_star:
        if f not in target.forms_star:
            target.forms_star.append(f)
    for f in source.forms_all:
        if f not in target.forms_all:
            target.forms_all.append(f)
    for r in source.references:
        if r not in target.references:
            target.references.append(r)
    for k in source.sense_kanji:
        if k not in target.sense_kanji:
            target.sense_kanji.append(k)
    for g in source.english_glosses:
        if g not in target.english_glosses:
            target.english_glosses.append(g)


def extract_yomitan_metadata_from_note(note: Any) -> Optional[YomitanMetadata]:
    """Inspect definition fields of an Anki note to extract Yomitan metadata.

    Merges metadata across definition fields if an earlier field contains only
    English glosses or partial metadata, ensuring structured JMdict POS tags
    and formsTable data from subsequent fields are not hidden.

    Args:
        note: Anki note or mapping containing flashcard fields.

    Returns:
        Populated YomitanMetadata if valid definition markup was found, or None.
    """
    if not note or not hasattr(note, "__getitem__"):
        return None

    combined_meta: Optional[YomitanMetadata] = None

    # Check prioritized definition fields first
    for f in NOTE_DEFINITION_FIELDS:
        if f in note and note[f]:
            val = str(note[f]).strip()
            meta = parse_yomitan_html(val)
            if _has_data(meta):
                if combined_meta is None:
                    combined_meta = meta
                else:
                    _merge_metadata(combined_meta, meta)
                if _has_structured_data(combined_meta) and combined_meta.english_glosses:
                    return combined_meta

    if combined_meta and _has_structured_data(combined_meta):
        return combined_meta

    # Fallback: scan any note field containing Yomitan markers
    try:
        keys = note.keys() if hasattr(note, "keys") else []
        for k in keys:
            if k in NOTE_DEFINITION_FIELDS:
                continue
            val = str(note[k]).strip()
            if "data-sc-content" in val or "formsTable" in val or "JMdict" in val:
                meta = parse_yomitan_html(val)
                if _has_data(meta):
                    if combined_meta is None:
                        combined_meta = meta
                    else:
                        _merge_metadata(combined_meta, meta)
                    if _has_structured_data(combined_meta) and combined_meta.english_glosses:
                        return combined_meta
    except Exception:
        pass

    return combined_meta if (combined_meta and _has_data(combined_meta)) else None


def extract_english_glosses_from_note(
    note: Any,
    yomitan_meta: Optional[YomitanMetadata] = None,
) -> List[str]:
    """Extract and normalize English gloss candidates from an Anki note.

    Prioritizes structured Yomitan definitions, then falls back to plain-text definition
    and translation fields.

    Args:
        note: Anki note or mapping containing flashcard fields.
        yomitan_meta: Optional pre-parsed YomitanMetadata.

    Returns:
        List of deduplicated canonical English glosses.
    """
    if yomitan_meta and yomitan_meta.english_glosses:
        return list(dict.fromkeys(yomitan_meta.english_glosses))

    if not note or not hasattr(note, "__getitem__"):
        return []

    # Check note using full Yomitan parser
    meta = extract_yomitan_metadata_from_note(note)
    if meta and meta.english_glosses:
        return list(dict.fromkeys(meta.english_glosses))

    # Fallback: scan definition and translation fields for plain-text glosses
    fields_to_check = NOTE_DEFINITION_FIELDS + (
        "SelectionText",
        "Meaning",
        "Meanings",
        "Back",
        "English",
        "Translation",
    )
    glosses: List[str] = []
    seen: Set[str] = set()

    for f in fields_to_check:
        if f in note and note[f]:
            raw_val = str(note[f]).strip()
            if not raw_val:
                continue
            # Replace block boundaries that separate definitions with delimiters before stripping tags
            clean_val = re.sub(r"</(?:li|div|p|tr|h\d)>|<br\s*/?>", "; ", raw_val)
            clean_val = re.sub(r"<[^>]+>", " ", clean_val)
            clean_val = html.unescape(clean_val).strip()
            # Remove parentheticals before delimiter splitting
            clean_val = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", clean_val)
            # Split on standard definition delimiters
            parts = re.split(r"[;,/\n、，]", clean_val)
            for part in parts:
                norm = normalize_english_gloss(part)
                if norm and norm not in seen:
                    seen.add(norm)
                    glosses.append(norm)
            if glosses:
                break

    return glosses
