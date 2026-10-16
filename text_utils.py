"""Text utilities for sanitizing search queries and generating tiered fallback candidates from Anki card fields."""
import html
import re
from typing import Any, List, Optional, Sequence, Tuple
import unicodedata

# Mapping from Arabic digits to Japanese Kanji numerals
KANJI_NUMS = {
    "1": "一", "2": "二", "3": "三", "4": "四", "5": "五",
    "6": "六", "7": "七", "8": "八", "9": "九", "10": "十"
}
COUNTER_PATTERN = r"杯|人|本|頭|枚|羽|匹|台|個|つ|階|着|足|件|回|歳|才|点|度|組|名"

# Common Japanese reading field names in popular note models (Lapis, Yomitan, Kaishi, Core, etc.)
NOTE_READING_FIELDS = (
    "ExpressionReading",
    "Reading",
    "Word Reading",
    "WordReading",
    "VocabReading",
    "Kana",
    "KanaReading",
)

NOTE_FURIGANA_FIELDS = (
    "ExpressionFurigana",
    "Word Furigana",
    "WordFurigana",
    "Furigana",
    "VocabFurigana",
)


def to_katakana(text: str) -> str:
    """Convert Hiragana characters to Katakana."""
    return "".join(
        chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c
        for c in text
    )


def to_hiragana(text: str) -> str:
    """Convert Katakana characters to Hiragana."""
    return "".join(
        chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c
        for c in text
    )


def extract_furigana_terms(raw_text: Optional[str]) -> Tuple[str, Optional[str]]:
    """Extract surface Kanji text and bracketed Furigana reading from an Anki field value.
    
    Examples:
        '其の儘[そのまま]' -> ('其の儘', 'そのまま')
        '動[うご]かす' -> ('動かす', 'うごかす')
        '林檎[りんご]' -> ('林檎', 'りんご')
        '猫' -> ('猫', None)
    """
    if not raw_text:
        return "", None

    # If no furigana bracket syntax present
    if "[" not in raw_text or "]" not in raw_text:
        return clean_search_query(raw_text), None

    # Surface Kanji: remove bracketed portions
    surface_raw = re.sub(r" ?([^ \]]+)?\[([^\]]+)\]", r"\1", raw_text)
    surface = clean_search_query(surface_raw)

    # Reading: replace "kanji[reading]" with "reading"
    reading_raw = re.sub(r" ?[^ \]]+\[([^\]]+)\]", r"\1", raw_text)
    reading = clean_search_query(reading_raw)

    reading_out = reading if reading and reading != surface else None
    return surface, reading_out


def convert_to_kanji_numerals(text: str) -> Optional[str]:
    """Convert leading Arabic numerals with Japanese counters into Kanji numerals.
    
    Examples:
        '1杯' -> '一杯'
        '2人' -> '二人'
        '1つ' -> '一つ'
    """
    m = re.match(r"^(\d{1,2})(" + COUNTER_PATTERN + r".*)$", text)
    if m:
        num, rest = m.groups()
        if num in KANJI_NUMS:
            return KANJI_NUMS[num] + rest
    return None


def expand_katakana_chouon(text: str) -> List[str]:
    """Expand Katakana words dropping trailing vowel elongation marks per JIS/technical style.
    
    Examples:
        'メンバ' -> ['メンバー']
        'サーバ' -> ['サーバー']
    """
    variants: List[str] = []
    if len(text) >= 2 and re.match(r"^[\u30A0-\u30FF]+$", text) and not text.endswith("ー"):
        variants.append(text + "ー")
    return variants


def deinflect_japanese(text: str) -> List[str]:
    """Extract dictionary base forms and simplify compound headwords using safe heuristics."""
    forms: List[str] = []

    # Geminate verb compound auxiliaries (〜ってくる, 〜っている, 〜っていく -> 〜る, 〜う, 〜つ)
    # e.g. 帰ってくる -> 帰る, 走っている -> 走る
    for aux in ["ってくる", "っている", "っていく"]:
        if text.endswith(aux) and len(text) > len(aux):
            base = text[:-len(aux)]
            forms.append(base + "る")
            forms.append(base + "う")
            forms.append(base + "つ")

    # Regular verb compound auxiliaries (〜てくる, 〜ている, 〜ていく, 〜てみる -> 〜る)
    for aux in ["てくる", "ている", "ていく", "てみる"]:
        if text.endswith(aux) and len(text) > len(aux):
            forms.append(text[:-len(aux)] + "る")

    # Voiced verb compounds (〜でくる, 〜でいる -> 〜ぐ)
    for aux in ["でくる", "でいる"]:
        if text.endswith(aux) and len(text) > len(aux):
            forms.append(text[:-len(aux)] + "ぐ")

    # Excessive suffix (〜すぎる, 〜過ぎる)
    for aux in ["すぎる", "過ぎる"]:
        if text.endswith(aux) and len(text) > len(aux):
            forms.append(text[:-len(aux)])

    # Potential / passive inflections (e.g. 読める -> 読む, 書ける -> 書く, 買える -> 買う)
    potential_map = {
        "える": "う",
        "ける": "く",
        "げる": "ぐ",
        "せる": "す",
        "てる": "つ",
        "べる": "ぶ",
        "める": "む",
        "れる": "る",
    }
    for pot_suffix, base_suffix in potential_map.items():
        if text.endswith(pot_suffix) and len(text) > len(pot_suffix):
            forms.append(text[:-len(pot_suffix)] + base_suffix)

    # Compound determiners (その結果 -> 結果)
    if text.startswith("その") and len(text) > 2:
        forms.append(text[2:])

    # Compound evaluatives (方がいい -> 方)
    if text.endswith("がいい") and len(text) > 3:
        forms.append(text[:-3])

    return forms


def clean_search_query(raw_text: Optional[str]) -> str:
    """Extract a clean keyword from a raw Anki field value.
    
    Removes:
    - HTML tags (e.g. <div>, <b>, <span style=...>)
    - Anki sound tags: [sound:filename.mp3]
    - Anki furigana brackets: 食べる[たべる] -> 食べる
    - Parenthesized readings: 猫（ねこ） -> 猫, 犬(いぬ) -> 犬
    - Non-essential punctuation (preserving valid Katakana prolonged sound mark 'ー')
    - Normalizes Unicode (NFKC)
    - Additional glosses separated by delimiters (takes the primary headword)
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Decode HTML entities and replace non-breaking spaces
    text = html.unescape(text).replace("\xa0", " ")

    # 3. Strip Anki sound tags [sound:...]
    text = re.sub(r"\[sound:[^\]]+\]", " ", text)

    # 4. Strip bracketed furigana: e.g. 漢字[かんじ] -> 漢字
    text = re.sub(r"\[[^\]]*\]", "", text)

    # 5. Strip parenthesized readings: （...） or (...)
    text = re.sub(r"[\(（][^()（）]*[\)）]", "", text)

    # 6. Normalize Unicode (NFKC) for full-width digits, letters, half-width katakana
    text = unicodedata.normalize("NFKC", text)

    # 7. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Separate multi-definition glosses while preserving compound words with middle dots (e.g. メッセージ・カード)
    # Split on middle dot ONLY if it is surrounded by whitespace or not both sides Katakana
    for sep in ["/", ";", "；", "、", ","]:
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if parts:
                text = parts[0]
                break

    if "・" in text:
        # If surrounded by Katakana on both sides, keep middle dot (e.g. メッセージ・カード)
        if not re.search(r"[\u30A0-\u30FF]・[\u30A0-\u30FF]", text):
            parts = [p.strip() for p in text.split("・") if p.strip()]
            if parts:
                text = parts[0]

    # 9. Clean leftover edge punctuation (do NOT strip 'ー' from ends as it is a Katakana vowel letter!)
    text = text.strip(" ~〜-‐―–—:：!！?？. 。")
    text = re.sub(r"^[ー\-]+", "", text)

    return text



def generate_search_candidates(
    raw_source: Optional[str],
    note: Optional[Any] = None,
    allow_tier2: bool = True
) -> List[Tuple[str, str, int]]:
    """Generate prioritized search candidate queries with confidence tiers.
    
    Tiers:
        Tier 1 (Safe): Zero homophone risk. Always automatic in both Batch & UI.
                       (NFKC, same-field furigana, loanword katakana, chōonpu, kanji numerals).
        Tier 2 (Semantic): Reading from separate note fields, deinflection, stem extraction.
                           (Controlled & auditable in Batch mode, interactive in Visual Picker).
    
    Returns:
        List of (candidate_query, reason_label, confidence_tier).
    """
    candidates: List[Tuple[str, str, int]] = []
    seen = set()

    def add(term: Optional[str], reason: str, tier: int) -> None:
        if not term:
            return
        cleaned = clean_search_query(term)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            candidates.append((cleaned, reason, tier))

    # 1. Primary source field & Same-field Furigana
    surface, same_field_reading = extract_furigana_terms(raw_source)
    add(surface, "primary", 1)

    if same_field_reading:
        # High confidence: card author explicitly specified this reading for this term
        add(same_field_reading, "same_field_reading", 1)
        add(to_katakana(same_field_reading), "same_field_reading_katakana", 1)

    # 2. Tier 1 Syntactic Expansions on primary term
    if surface:
        # Kanji numerals (e.g. 1杯 -> 一杯, 2人 -> 二人)
        kanji_num = convert_to_kanji_numerals(surface)
        if kanji_num:
            add(kanji_num, "kanji_numeral", 1)

        # Katakana conversion (crucial for loanwords written in hiragana like とまと -> トマト, ぱん -> パン)
        if re.search(r"[\u3040-\u309F]", surface):
            add(to_katakana(surface), "loanword_katakana", 1)

        # Katakana elongation mark expansion (メンバ -> メンバー, サーバ -> サーバー)
        for chouon_variant in expand_katakana_chouon(surface):
            add(chouon_variant, "katakana_chouon", 1)

    # 3. Tier 2 Semantic Fallbacks (Separate Note Fields & Deinflection)
    if allow_tier2:
        # Check dedicated reading fields on the note
        if note and hasattr(note, "__getitem__"):
            for rf in NOTE_READING_FIELDS:
                if rf in note and note[rf]:
                    val = note[rf].strip()
                    if val:
                        add(val, f"note_reading_{rf}", 2)
                        add(to_katakana(val), f"note_reading_katakana_{rf}", 2)

            # Check furigana fields on the note
            for ff in NOTE_FURIGANA_FIELDS:
                if ff in note and note[ff]:
                    f_val = note[ff].strip()
                    if f_val:
                        _, f_read = extract_furigana_terms(f_val)
                        if f_read:
                            add(f_read, f"note_furigana_{ff}", 2)
                            add(to_katakana(f_read), f"note_furigana_katakana_{ff}", 2)

        # Deinflection / Compound Headwords
        base_candidates = [c[0] for c in candidates if c[2] == 1]
        for base in base_candidates:
            for deinflected in deinflect_japanese(base):
                add(deinflected, "deinflected_base", 2)
                if re.search(r"[\u3040-\u309F]", deinflected):
                    add(to_katakana(deinflected), "deinflected_katakana", 2)

    return candidates

