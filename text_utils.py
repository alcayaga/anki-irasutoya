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

BATCH_MAX_CANDIDATES = 5


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
        'お 前[まえ]' -> ('お前', 'おまえ')
        '動[うご]かす' -> ('動かす', 'うごかす')
        '林檎[りんご]' -> ('林檎', 'りんご')
        '猫' -> ('猫', None)
    """
    if not raw_text:
        return "", None

    # If no furigana bracket syntax present
    if "[" not in raw_text or "]" not in raw_text:
        return clean_search_query(raw_text), None

    text = raw_text.strip()
    # Case 1: Simple single Kanji/counter word with furigana at the end: '林檎[りんご]', '漢字[かんじ]', '１杯[いっぱい]'
    # Must start with Kanji/alphanumeric and contain NO spaces or unbracketed prefix okurigana
    kanji_or_alnum = r"[\u4e00-\u9fff\u3400-\u4dbf々〆〇0-9\uff10-\uff19a-zA-Z\uff21-\uff3a\uff41-\uff5a]"
    m_single = re.match(rf"^({kanji_or_alnum}+)\[([^\]]+)\]$", text)
    if not m_single:
        # Whole-word reading on a mixed-script headword: '其の儘[そのまま]', '食べる[たべる]', 'お茶[おちゃ]'
        m_whole = re.match(r"^([^\s\[\]]+)\[([^\]]+)\]$", text)
        if m_whole and (
            re.search(rf"{kanji_or_alnum}[\u3040-\u30FF]", m_whole.group(1))
            or re.search(rf"[\u3040-\u30FF]{kanji_or_alnum}", m_whole.group(1))
        ):
            m_single = m_whole
    if m_single:
        surface = clean_search_query(m_single.group(1))
        reading = clean_search_query(m_single.group(2))
        reading_out = reading if reading and reading != surface else None
        return surface, reading_out

    # Case 2: Segmented furigana: 'お 前[まえ]', '其[そ]の 儘[まま]', '動[うご]かす'
    surface_raw = re.sub(r" ?([^ \[\]]+)?\[([^\]]+)\]", lambda m: (m.group(1) or ""), text)
    surface = clean_search_query(surface_raw)

    seg_pattern = re.compile(rf"(?: ([^ \[]+)|({kanji_or_alnum}+))\[([^\]]+)\]")
    reading_raw = seg_pattern.sub(r"\3", text)
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
    m = re.match(r"^(\d{1,2})((?:" + COUNTER_PATTERN + r").*)$", text)
    if m:
        num, rest = m.groups()
        if num in KANJI_NUMS:
            return KANJI_NUMS[num] + rest
    return None


CHOUON_OMISSION_ENDINGS = ("バ", "パ", "タ", "ダ", "ザ", "ラ", "マ", "カ", "ガ", "サ")


def expand_katakana_chouon(text: str) -> List[str]:
    """Expand Katakana loanwords dropping trailing vowel elongation marks per JIS/technical style.
    
    Examples:
        'メンバ' -> ['メンバー']
        'サーバ' -> ['サーバー']
    """
    variants: List[str] = []
    if (
        len(text) >= 3
        and re.match(r"^[\u30A0-\u30FF]+$", text)
        and not text.endswith("ー")
        and text.endswith(CHOUON_OMISSION_ENDINGS)
    ):
        variants.append(text + "ー")
    return variants


def deinflect_japanese(text: str) -> List[str]:
    """Extract dictionary base forms and simplify compound headwords using safe heuristics."""
    forms: List[str] = []

    # Geminate verb compound auxiliaries (〜ってくる, 〜っている, 〜っていく -> 〜る, 〜う, 〜つ)
    # e.g. 帰ってくる -> 帰る, 走っている -> 走る
    geminate_matched = False
    for aux in ["ってくる", "っている", "っていく", "ってみる"]:
        if text.endswith(aux) and len(text) > len(aux):
            geminate_matched = True
            base = text[:-len(aux)]
            forms.append(base + "る")
            forms.append(base + "う")
            forms.append(base + "つ")

    # Regular verb compound auxiliaries (〜てくる, 〜ている, 〜ていく, 〜てみる -> 〜る)
    if not geminate_matched:
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
        # Split on middle dots whose adjacent characters are not both Katakana (preserving loanword compounds like メッセージ・カード)
        parts = [p.strip() for p in re.split(r"(?<![\u30A0-\u30FF])・|・(?![\u30A0-\u30FF])", text) if p.strip()]
        if parts:
            text = parts[0]

    # 9. Clean leftover edge punctuation (do NOT strip 'ー' from ends as it is a Katakana vowel letter!)
    text = text.strip(" ~〜-‐―–—:：!！?？. 。")
    text = re.sub(r"^[ー\-]+", "", text)

    return text


GRAMMAR_FUNCTION_WORDS = {
    "よ", "ね", "な", "ぞ", "ぜ", "さ", "か", "わ", "とも", "から", "まで", "より", "ので", "のに",
    "なら", "だが", "けど", "けれど", "ても", "でも", "とか", "たり", "だり", "やら", "なり",
    "こそ", "さえ", "すら", "しか", "ばかり", "だけ", "ほど", "くらい", "ぐらい", "など", "なん",
    "なんて", "とは", "にも", "には", "では", "へと", "かも", "かな", "かしら", "っけ", "もん",
    "もの", "こと", "とき", "ところ", "わけ", "はず", "つもり", "よう", "そう", "らしい", "みたい",
    "べき", "まい", "たい", "ない", "ぬ", "ん", "れる", "られる", "せる", "させる",
    "これ", "それ", "あれ", "どれ", "この", "その", "あの", "どの", "ここ", "そこ", "あそこ", "どこ",
    "こう", "そう", "ああ", "どう", "こちら", "そちら", "あちら", "どちら", "こいつ", "そいつ", "あいつ",
    "する", "なる", "ある", "いる", "おる", "くる", "いく", "ゆく", "やる", "くれる", "あげる", "もらう",
    "いい", "よい", "ない", "もう", "まだ", "また", "すぐ", "よく", "とても", "かなり", "ちょっと",
    "さて", "では", "そして", "しかし", "だが", "けれど", "しかも", "または", "および", "つまり",
    "つつ", "ずつ", "ばかり", "ながら", "がてら", "かたがた", "どころか", "ばかりか", "のみならず",
    "として", "とする", "としても", "としたら", "とすれば", "にしては", "にしても", "について",
    "に関して", "に対して", "にとって", "によって", "により", "による", "につれて", "にしたがって",
    "に伴って", "とともに", "をはじめ", "をめぐって", "をもとに", "にかかわらず", "にもかかわらず",
    "をとわず", "に反して", "に沿って",
}

SAFE_2MORA_LOANWORDS = {"ぱん", "ぴざ", "ばす", "がす", "ぺん", "びる", "べっど", "どあ", "ぺあ", "びあ"}

BIKAGO_PREFIX_WORDS = {
    "肉", "肌", "茶", "酒", "皿", "箸", "風呂", "金", "菓子", "弁当", "祭り", "土産", "年玉",
    "もちゃ", "祝い", "礼", "見舞い", "供え", "握り", "結び", "手洗い", "湯",
}

GODAN_ENDINGS = {
    "う": ("わ", "い", "っ", "え", "お"),
    "く": ("か", "き", "い", "け", "こ"),
    "ぐ": ("が", "ぎ", "い", "げ", "ご"),
    "す": ("さ", "し", "し", "せ", "そ"),
    "つ": ("た", "ち", "っ", "て", "と"),
    "ぬ": ("な", "に", "ん", "ね", "の"),
    "ぶ": ("ば", "び", "ん", "べ", "ぼ"),
    "む": ("ま", "み", "ん", "め", "も"),
    "る": ("ら", "り", "っ", "れ", "ろ"),
}


def is_grammar_function_word(text: str) -> bool:
    """Return True if text is a grammatical particle, conjunction, demonstrative, or single kana."""
    if not text:
        return False
    clean = text.strip()
    if clean in GRAMMAR_FUNCTION_WORDS:
        return True
    if len(clean) == 1 and 0x3040 <= ord(clean[0]) <= 0x30FF:
        return True
    return False


def expand_bikago_prefix(text: str) -> Optional[str]:
    """Expand everyday Japanese nouns canonically titled with polite 'お' on Irasutoya."""
    if text in BIKAGO_PREFIX_WORDS:
        return "お" + text
    return None


SINGLE_KANJI_ICHIDAN = {"見る", "着る", "寝る", "出る", "似る", "煮る", "居る", "得る", "経る", "射る", "干る"}


def conjugate_japanese_verb(text: str) -> List[str]:
    """Generate canonical Irasutoya title inflection forms (past/stative, masu-stem, volitional) for verbs containing Kanji."""
    if len(text) < 2 or not re.search(r"[\u4e00-\u9fff\u3400-\u4dbf々〆〇]", text):
        return []

    # Irregular verbs: する compounds (勉強する -> 勉強した, 勉強し) and 来る
    if text.endswith("する"):
        base = text[:-2]
        return [base + "した", base + "し"] if base else []
    if text.endswith("来る"):
        base = text[:-1]
        return [base + "た", base] if len(base) >= 2 else [base + "た"]

    # Compound auxiliary くる (帰ってくる -> 帰ってきた); other auxiliaries handled by deinflect_japanese
    if text.endswith(("てくる", "でくる")):
        return [text[:-2] + "きた"]
    if text.endswith(("ている", "でいる", "ていく", "てみる")):
        return []

    last_char = text[-1]
    if last_char not in GODAN_ENDINGS:
        return []
    stem = text[:-1]

    # Exclude numeral counters like 一つ, 二つ, 1つ
    if re.fullmatch(r"[0-9\uff10-\uff19一二三四五六七八九十]+", stem):
        return []

    forms: List[str] = []

    # Differentiate Ichidan vs Godan for verbs ending in る
    is_ichidan = last_char == "る" and (
        text in SINGLE_KANJI_ICHIDAN
        or bool(stem and stem[-1] in "いきぎしじちにひびみりえけげせぜてでねへべめれ")
    )
    if is_ichidan:
        # Ichidan: 食べる -> 食べた, 食べ; 疲れる -> 疲れた, 疲れ; 見る -> 見た
        forms.append(stem + "た")
        if len(stem) >= 2:
            forms.append(stem)
    else:
        # Godan: 売る -> 売った, 売り, 売ろう; 勝つ -> 勝った, 勝ち, 勝とう
        a_row, i_row, te_stem, e_row, o_row = GODAN_ENDINGS[last_char]
        past_suffix = "だ" if last_char in ("ぐ", "ぬ", "ぶ", "む") else "た"
        # 行く is irregular in the past tense: 行った (not 行いた)
        te = "っ" if text.endswith("行く") else te_stem
        forms.append(stem + te + past_suffix)
        forms.append(stem + i_row)
        forms.append(stem + o_row + "う")

    seen = set()
    out = []
    for f in forms:
        if f != text and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def generate_search_candidates(
    raw_source: Optional[str],
    note: Optional[Any] = None,
    allow_tier2: bool = True,
    allow_deinflection: bool = True
) -> List[Tuple[str, str, int]]:
    """Generate prioritized search candidate queries with confidence tiers.
    
    Tiers:
        Tier 1 (Safe Lexical & Morphological): Zero homophone risk. Always automatic in both Batch & UI.
                       (NFKC, Kanji numerals, Bikago prefixes, compound verb reductions, verb past/volitional forms, loanword Katakana).
        Tier 2 (Phonetic & Reading Fallbacks): Readings from separate note fields,
                       same-field furigana, verb masu-stems, Katakana chōonpu expansion, heuristic compound deinflection.
    
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

    # 1. Primary source field
    surface, same_field_reading = extract_furigana_terms(raw_source)
    add(surface, "primary", 1)

    # If surface is a pure grammar function word, suppress unanchored phonetic/reading fallbacks
    if not surface or is_grammar_function_word(surface):
        return candidates

    # 2. Tier 1 Safe Lexical & Morphological Expansions on primary term
    # Kanji numerals (e.g. 1杯 -> 一杯, 2人 -> 二人)
    kanji_num = convert_to_kanji_numerals(surface)
    if kanji_num:
        add(kanji_num, "kanji_numeral", 1)

    # Bikago polite prefix (e.g. 肉 -> お肉, 肌 -> お肌, 茶 -> お茶, 風呂 -> お風呂)
    bikago = expand_bikago_prefix(surface)
    if bikago:
        add(bikago, "bikago_prefix", 1)

    # Verb stative/past and volitional inflections (e.g. 疲れる -> 疲れた; 勝つ -> 勝った, 勝とう; 売る -> 売った, 売ろう)
    verb_inflections = conjugate_japanese_verb(surface)
    for inflected in verb_inflections:
        if inflected.endswith(("た", "だ", "う")):
            add(inflected, "verb_inflection", 1)

    # Katakana conversion for loanwords and fauna/flora (e.g. とまと -> トマト, ねこ -> ネコ, ぱん -> パン)
    if (
        re.fullmatch(r"[\u3040-\u309F]+", surface)
        and not is_grammar_function_word(surface)
    ):
        add(to_katakana(surface), "loanword_katakana", 1)

    # 3. Tier 2 Semantic & Phonetic Fallbacks (Separate Note Fields, Furigana Readings, Masu-Stems, Elongation, Heuristic Deinflection)
    if allow_tier2:
        # Check dedicated reading fields on the note first (authoritative dictionary readings)
        if note and hasattr(note, "__getitem__"):
            for rf in NOTE_READING_FIELDS:
                if rf in note and note[rf]:
                    val = str(note[rf]).strip()
                    if val:
                        _, f_read = extract_furigana_terms(val)
                        reading_term = f_read if f_read else val
                        cleaned_read = clean_search_query(reading_term)
                        if cleaned_read and not is_grammar_function_word(cleaned_read):
                            add(cleaned_read, f"note_reading_{rf}", 2)
                            add(to_katakana(cleaned_read), f"note_reading_katakana_{rf}", 2)

            # Check furigana fields on the note
            for ff in NOTE_FURIGANA_FIELDS:
                if ff in note and note[ff]:
                    f_val = str(note[ff]).strip()
                    if f_val:
                        _, f_read = extract_furigana_terms(f_val)
                        if f_read:
                            cleaned_read = clean_search_query(f_read)
                            if cleaned_read and not is_grammar_function_word(cleaned_read):
                                add(cleaned_read, f"note_furigana_{ff}", 2)
                                add(to_katakana(cleaned_read), f"note_furigana_katakana_{ff}", 2)

        # Same-field reading extracted from furigana brackets (e.g. 檸檬[れもん] -> れもん, 本[ほん] -> ほん)
        # Suppress single kana and grammar function words to avoid homophone collisions
        if same_field_reading and not is_grammar_function_word(same_field_reading):
            add(same_field_reading, "same_field_reading", 2)
            add(to_katakana(same_field_reading), "same_field_reading_katakana", 2)

        # Masu-stems (e.g. 疲れ, 勝ち, 売り, 休み) can function as distinct nouns,
        # so classify them as Tier 2 fallbacks with audit tagging.
        for inflected in verb_inflections:
            if not inflected.endswith(("た", "だ", "う")):
                add(inflected, "verb_stem", 2)

        # Katakana elongation mark expansion (メンバ -> メンバー, サーバ -> サーバー)
        for chouon_variant in expand_katakana_chouon(surface):
            add(chouon_variant, "katakana_chouon", 2)

        # Heuristic compound deinflection (e.g. 帰ってくる -> 帰る, 走っている -> 走る)
        if allow_deinflection:
            for deinflected in deinflect_japanese(surface):
                add(deinflected, "deinflected_base", 2)

    return candidates

