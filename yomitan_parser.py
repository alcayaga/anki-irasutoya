"""Parser for extracting structured JMdict metadata from Yomitan note HTML fields."""
from dataclasses import dataclass, field
import html
from html.parser import HTMLParser
import re
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
    references: List[str] = field(default_factory=list)
    sense_kanji: List[str] = field(default_factory=list)


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

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        attr_dict = dict(attrs)
        sc_content = attr_dict.get("data-sc-content", "") or ""

        if tag == "i":
            self._in_i = True
            self._current_i_text = []

        # formsTable detection
        if tag == "table" and sc_content == "formsTable":
            self._in_forms_table = True
            self._table_headers = []
            self._current_row_idx = 0

        if self._in_forms_table:
            if tag == "tr":
                self._current_col_idx = 0
            elif tag in ("th", "td"):
                self._in_th = (tag == "th")
                self._in_td = (tag == "td")
                self._current_cell_text = []

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

    def handle_endtag(self, tag: str) -> None:
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
                        clean_hdr = re.sub(r"^[〈（(]|[\s〉）)]+$", "", cell_text).strip()
                        self._table_headers.append(clean_hdr)
                else:
                    col_offset = self._current_col_idx - 1
                    if "★" in cell_text and 0 <= col_offset < len(self._table_headers):
                        cand = self._table_headers[col_offset]
                        if cand and cand not in self.metadata.forms_star:
                            self.metadata.forms_star.append(cand)
                self._current_col_idx += 1
                self._in_th = False
                self._in_td = False
            elif tag == "tr":
                self._current_row_idx += 1
            elif tag == "table":
                self._in_forms_table = False

        # Handle simple forms list <li>
        if self._in_forms_entry:
            if tag == "li" and self._in_forms_li:
                self._in_forms_li = False
                li_text = "".join(self._current_forms_li_text).strip()
                if "★" in li_text:
                    clean_term = re.sub(r"[（(]★[)）]|\s*★\s*", "", li_text).strip()
                    clean_term = re.sub(r"^[〈（(]|[\s〉）)]+$", "", clean_term).strip()
                    if clean_term and clean_term not in self.metadata.forms_star:
                        self.metadata.forms_star.append(clean_term)
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
        return meta
    except Exception:
        return None


def _has_data(meta: Optional[YomitanMetadata]) -> bool:
    """Return True if meta contains any parsed POS tags, forms, references, or sense notes."""
    return bool(
        meta and (meta.pos_tags or meta.forms_star or meta.references or meta.sense_kanji)
    )


def extract_yomitan_metadata_from_note(note: Any) -> Optional[YomitanMetadata]:
    """Inspect definition fields of an Anki note to extract Yomitan metadata."""
    if not note or not hasattr(note, "__getitem__"):
        return None

    # Check prioritized definition fields first
    for f in NOTE_DEFINITION_FIELDS:
        if f in note and note[f]:
            val = str(note[f]).strip()
            meta = parse_yomitan_html(val)
            if _has_data(meta):
                return meta

    # Fallback: scan any note field containing Yomitan markers
    try:
        keys = note.keys() if hasattr(note, "keys") else []
        for k in keys:
            val = str(note[k]).strip()
            if "data-sc-content" in val or "formsTable" in val or "JMdict" in val:
                meta = parse_yomitan_html(val)
                if _has_data(meta):
                    return meta
    except Exception:
        pass

    return None
