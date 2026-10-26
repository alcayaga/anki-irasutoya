"""Text utilities for sanitizing search queries and generating tiered fallback candidates from Anki card fields."""
import html
import re
import unicodedata
from typing import Any, Callable, List, Optional, Tuple

from .synonyms import lookup_concept_map
from .yomitan_parser import (
    extract_yomitan_metadata_from_note,
)

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

# Prefixes of candidate reason labels representing phonetic readings (risk of homophone collision)
PHONETIC_REASON_PREFIXES: Tuple[str, ...] = (
    "note_reading",
    "note_furigana",
    "same_field_reading",
    "yomitan_uk",
)

KANJI_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf々〆〇]")

PRONOUN_KATAKANA_BLOCKLIST = frozenset({
    "わし", "おれ", "ぼく", "わたし", "あたし", "わたくし", "うち",
    "きみ", "おまえ", "あんた", "あなた", "かれ", "かのじょ", "かれら",
    "われ", "われわれ", "じぶん", "てまえ", "こいつ", "そいつ", "あいつ", "どいつ",
    "だれ", "なに", "なん", "ここ", "そこ", "あそこ", "どこ",
    "これ", "それ", "あれ", "どれ", "こちら", "そちら", "あちら", "どちら",
    "こっち", "そっち", "あっち", "どっち",
})


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

    # Strip HTML tags, HTML entities, and Anki sound tags first
    text = re.sub(r"<[^>]+>", " ", raw_text)
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"\[sound:[^\]]+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # If no furigana bracket syntax present
    if "[" not in text or "]" not in text:
        return clean_search_query(text), None
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


GODAN_TSU_BASES = frozenset({"待", "勝", "立", "持", "打", "育", "建", "保", "放", "断", "経"})
GODAN_U_BASES = frozenset({
    "買", "言", "会", "洗", "歌", "使", "払", "笑", "思", "手伝", "吸", "祝",
    "向か", "追", "習", "拾", "失", "通", "願", "争", "奪", "誘", "舞", "嫌",
    "狂", "繕", "雇", "商", "賑わ", "誓", "救", "養", "扱", "結",
})
GODAN_BU_BASES = frozenset({"遊", "呼", "飛", "並", "選", "運", "叫", "結", "喜", "転"})

COMMON_I_ADJECTIVES = frozenset({
    "赤い", "白い", "青い", "黒い", "黄色い", "茶色い", "高い", "安い", "低い", "近い", "遠い",
    "広い", "狭い", "早い", "速い", "遅い", "重い", "軽い", "深い", "浅い", "強い", "弱い",
    "暑い", "熱い", "寒い", "冷たい", "痛い", "辛い", "甘い", "苦い", "若い", "暗い", "明るい",
    "長い", "短い", "丸い", "太い", "細い", "荒い", "粗い", "鋭い", "鈍い", "汚い", "危ない",
    "良い", "善い", "酷い", "凄い", "怖い", "恐い", "偉い", "尊い", "賢い", "臭い", "固い", "硬い", "堅い",
    "多い", "少ない", "美味い", "旨い", "大きい", "小さい", "古い", "薄い", "厚い", "眠い",
    "細かい", "暖かい", "温かい", "柔らかい", "可愛い", "緩い", "痒い", "温い", "ぬるい",
})

ADJECTIVE_CANONICAL_ENDINGS = ("しい", "たい", "かい", "ない", "るい", "おい", "よい", "どい", "とい", "ゆい", "らい", "ぶい")


def is_adjective_ending(text: str) -> bool:
    """Return True if text ends in a standard Japanese i-adjective form."""
    if not text or not text.endswith("い"):
        return False
    return text in COMMON_I_ADJECTIVES or text.endswith(ADJECTIVE_CANONICAL_ENDINGS)


def deinflect_japanese(text: str) -> List[str]:
    """Extract dictionary base forms and simplify compound headwords using safe heuristics."""
    forms: List[str] = []

    # Geminate verb compound auxiliaries (〜ってくる, 〜っている, 〜っていく -> 〜る, 〜う, 〜つ, 行く)
    # e.g. 帰ってくる -> 帰る, 走っている -> 走る, 待っている -> 待つ, 買っている -> 買う, 行ってくる -> 行く
    geminate_matched = False
    for aux in ["ってくる", "っている", "っていく", "ってみる"]:
        if text.endswith(aux) and len(text) > len(aux):
            geminate_matched = True
            base = text[:-len(aux)]
            if base == "行":
                forms.append(base + "く")
            elif base in GODAN_TSU_BASES:
                forms.extend([base + "つ", base + "る", base + "う"])
            elif base in GODAN_U_BASES:
                forms.extend([base + "う", base + "る", base + "つ"])
            else:
                forms.extend([base + "る", base + "う", base + "つ"])

    # Regular verb compound auxiliaries (〜てくる, 〜ている, 〜ていく, 〜てみる -> 〜る, 〜する, 〜す, 〜く)
    # e.g. 勉強している -> 勉強する; 話している -> 話す; 書いている -> 書く; 食べている -> 食べる
    if not geminate_matched:
        for aux in ["てくる", "ている", "ていく", "てみる"]:
            if text.endswith(aux) and len(text) > len(aux):
                base = text[:-len(aux)]
                if base.endswith("し") and len(base) > 1:
                    head = base[:-1]
                    # Multi-Kanji heads are usually suru-nouns (勉強している -> 勉強する); otherwise godan す (話している -> 話す)
                    if len(head) >= 2 and all("\u4e00" <= ch <= "\u9fff" for ch in head):
                        forms.extend([head + "する", head + "す"])
                    else:
                        forms.extend([head + "す", head + "する"])
                elif base.endswith("い") and len(base) > 1:
                    # 書いている -> 書く (godan く); keep ichidan 〜いる as a speculative alternative
                    forms.extend([base[:-1] + "く", base + "る"])
                else:
                    forms.append(base + "る")

    # Voiced verb compounds (〜でくる, 〜でいる, 〜でいく, 〜でみる -> 〜ぐ, 〜む, 〜ぶ, 〜ぬ)
    # e.g. 泳いでいる -> 泳ぐ; 読んでいる -> 読む; 遊んでいる -> 遊ぶ; 飛んでいく -> 飛ぶ
    for aux in ["でくる", "でいる", "でいく", "でみる"]:
        if text.endswith(aux) and len(text) > len(aux):
            base = text[:-len(aux)]
            if base.endswith("い") and len(base) > 1:
                forms.append(base[:-1] + "ぐ")
            elif base.endswith("ん") and len(base) > 1:
                stem = base[:-1]
                if stem.endswith("死"):
                    forms.extend([stem + "ぬ", stem + "む", stem + "ぶ"])
                elif stem in GODAN_BU_BASES:
                    forms.extend([stem + "ぶ", stem + "む", stem + "ぬ"])
                else:
                    forms.extend([stem + "む", stem + "ぶ", stem + "ぬ"])

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


SINGLE_KANJI_ICHIDAN = {
    "見る", "観る", "診る", "看る", "視る",
    "着る", "寝る", "出る", "似る", "煮る", "居る", "得る", "経る", "射る", "干る",
}
NON_VERB_GODAN_LOOKALIKES = frozenset({"全く", "暫く", "漸く", "恐らく", "悉く", "如く", "幾つ"})


def conjugate_japanese_verb(text: str, verb_class: Optional[str] = None) -> List[str]:
    """Generate canonical Irasutoya title inflection forms (past/stative, masu-stem, volitional) for verbs containing Kanji."""
    if len(text) < 2 or not re.search(r"[\u4e00-\u9fff\u3400-\u4dbf々〆〇]", text):
        return []

    # Common non-verb adverbs / interrogative counters ending in godan kana
    if text in NON_VERB_GODAN_LOOKALIKES:
        return []

    # Irregular verbs: する compounds (勉強する -> 勉強した, 勉強しよう, 勉強し) and 来る
    if text.endswith("する"):
        base = text[:-2]
        return [base + "した", base + "しよう", base + "し"] if base else []
    if text.endswith("来る"):
        base = text[:-1]
        return [base + "た", base] if len(base) >= 2 else [base + "た"]

    # Compound auxiliary くる (帰ってくる -> 帰ってきた); other auxiliaries handled by deinflect_japanese
    if text.endswith(("てくる", "でくる")):
        return [text[:-2] + "きた"]
    if text.endswith(("ている", "でいる", "ていく", "でいく", "てみる", "でみる")):
        return []

    last_char = text[-1]
    if last_char not in GODAN_ENDINGS:
        return []
    stem = text[:-1]

    # Adverbial i-adjective forms (早く, 多く, 遅く, 美しく, 大きく, 小さく) are not verbs
    if last_char == "く" and is_adjective_ending(stem + "い"):
        return []

    # Exclude numeral counters like 一つ, 二つ, 1つ
    if re.fullmatch(r"[0-9\uff10-\uff19一二三四五六七八九十]+", stem):
        return []

    forms: List[str] = []

    # Differentiate Ichidan vs Godan for verbs ending in る
    if verb_class and verb_class.startswith("v1") and last_char == "る":
        is_ichidan = True
    elif verb_class and verb_class.startswith("v5"):
        is_ichidan = False
    else:
        is_ichidan = last_char == "る" and (
            text in SINGLE_KANJI_ICHIDAN
            or bool(stem and stem[-1] in "いきぎしじちにひびみりえけげせぜてでねへべめれ")
        )

    if is_ichidan:
        # Ichidan: 食べる -> 食べた, 食べよう, 食べ; 疲れる -> 疲れた, 疲れよう, 疲れ; 見る -> 見た, 見よう
        forms.append(stem + "た")
        forms.append(stem + "よう")
        if len(stem) >= 2:
            forms.append(stem)
    else:
        # Godan: 売る -> 売った, 売り, 売ろう; 勝つ -> 勝った, 勝ち, 勝とう
        a_row, i_row, te_stem, e_row, o_row = GODAN_ENDINGS[last_char]
        past_suffix = "だ" if last_char in ("ぐ", "ぬ", "ぶ", "む") else "た"
        # 行く is irregular in the past tense: 行った (not 行いた)
        te = "っ" if (verb_class == "v5k-s" or text.endswith("行く")) else te_stem
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


def _iter_note_readings(note: Any) -> List[Tuple[str, str, str]]:
    """Extract (field_name, field_type, reading) from note reading and furigana fields."""
    if not note or not hasattr(note, "__getitem__"):
        return []
    results = []
    for rf in NOTE_READING_FIELDS:
        if rf in note and note[rf]:
            val = str(note[rf]).strip()
            if val:
                _, f_read = extract_furigana_terms(val)
                reading_term = f_read if f_read else val
                cleaned_read = clean_search_query(reading_term)
                if cleaned_read and not is_grammar_function_word(cleaned_read):
                    results.append((rf, "reading", cleaned_read))
    for ff in NOTE_FURIGANA_FIELDS:
        if ff in note and note[ff]:
            f_val = str(note[ff]).strip()
            if f_val:
                _, f_read = extract_furigana_terms(f_val)
                if f_read:
                    cleaned_read = clean_search_query(f_read)
                    if cleaned_read and not is_grammar_function_word(cleaned_read):
                        results.append((ff, "furigana", cleaned_read))
    return results


def _find_note_reading(note: Any) -> Optional[str]:
    """Extract first valid authoritative reading from note fields."""
    readings = _iter_note_readings(note)
    return readings[0][2] if readings else None


def _should_suppress_katakana_conversion(
    term: Optional[str],
    yomitan_meta: Optional[Any] = None
) -> bool:
    """Return True if term is a Japanese pronoun or demonstrative that must NOT be converted to Katakana.

    Prevents homophone hijacking (e.g. pronoun わし -> ワシ matching 鷲 'eagle',
    or pronoun あれ -> アレ matching irrelevant terms).
    """
    if not term:
        return False
    clean = clean_search_query(term)
    if clean in PRONOUN_KATAKANA_BLOCKLIST:
        return True
    if yomitan_meta and hasattr(yomitan_meta, "pos_tags"):
        if "pn" in yomitan_meta.pos_tags:
            return True
    return False


def generate_search_candidates(
    raw_source: Optional[str],
    note: Optional[Any] = None,
    allow_tier2: bool = True,
    allow_deinflection: bool = True,
    store: Optional[Any] = None,
    concept_lookup: Optional[Callable[[str], List[str]]] = None,
) -> List[Tuple[str, str, int]]:
    """Generate prioritized search candidate queries with confidence tiers.
    
    Tiers:
        Tier 1 (Safe Lexical & Morphological): Zero homophone risk. Always automatic in both Batch & UI.
                       (NFKC, Kanji numerals, Bikago prefixes, compound verb reductions, verb past/volitional forms, loanword Katakana,
                        Yomitan uk orthography kana readings, formsTable ★ canonical kanji variants).
        Tier 2 (Phonetic & Reading Fallbacks): Readings from separate note fields,
                       same-field furigana, verb masu-stems, Katakana chōonpu expansion, heuristic compound deinflection,
                       Yomitan sense-specific kanji (esp.), Yomitan cross-references (see:), curated visual concept maps.
    
    Args:
        raw_source: Raw text or HTML from the source field.
        note: Optional Anki note object for extracting Yomitan metadata and secondary reading fields.
        allow_tier2: Whether to generate Tier 2 phonetic and semantic fallbacks.
        allow_deinflection: Whether to perform heuristic verb/adjective deinflection.
        store: Optional CatalogStore instance to resolve concept maps from.
        concept_lookup: Optional callable mapping a headword to a list of target concept strings.

    Returns:
        List of (candidate_query, reason_label, confidence_tier).
    """
    candidates: List[Tuple[str, str, int]] = []
    seen = set()

    def add(term: Optional[str], reason: str, tier: int) -> bool:
        if not term:
            return False
        cleaned = clean_search_query(term)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            candidates.append((cleaned, reason, tier))
            return True
        return False

    yomitan_meta = extract_yomitan_metadata_from_note(note) if note is not None else None

    # 1. Primary source field
    surface, same_field_reading = extract_furigana_terms(raw_source)
    add(surface, "primary", 1)

    # If surface is a pure grammar function word, suppress unanchored phonetic/reading fallbacks
    if not surface or is_grammar_function_word(surface) or (yomitan_meta and yomitan_meta.is_particle_or_aux):
        return candidates

    # 2. Tier 1 Safe Lexical & Morphological Expansions on primary term
    # Kanji numerals (e.g. 1杯 -> 一杯, 2人 -> 二人)
    kanji_num = convert_to_kanji_numerals(surface)
    if kanji_num:
        add(kanji_num, "kanji_numeral", 1)

    # Yomitan canonical ★ Kanji variants from formsTable or forms list (e.g. 交代 <-> 交替, 一日 <-> １日)
    if yomitan_meta and yomitan_meta.forms_star:
        for star_form in yomitan_meta.forms_star[:3]:
            add(star_form, "yomitan_star_form", 1)

    # Yomitan 'uk' (usually written using kana alone) orthography promotion
    # Headwords like 其の儘, 懸かる, ご免なさい are canonical in Kana in modern Japanese and on Irasutoya
    if yomitan_meta and yomitan_meta.is_uk:
        uk_reading = same_field_reading or _find_note_reading(note)
        if uk_reading:
            cleaned_uk = clean_search_query(uk_reading)
            if (
                cleaned_uk
                and (len(cleaned_uk) >= 3 or cleaned_uk in SAFE_2MORA_LOANWORDS)
                and not is_grammar_function_word(cleaned_uk)
            ):
                add(cleaned_uk, "yomitan_uk_reading", 1)
                if not _should_suppress_katakana_conversion(cleaned_uk, yomitan_meta):
                    add(to_katakana(cleaned_uk), "yomitan_uk_katakana", 1)

    # Yomitan orthographic forms containing at least one Kanji (e.g. 缶けり -> 缶蹴り)
    if yomitan_meta and yomitan_meta.forms_all:
        added_forms = 0
        for form in yomitan_meta.forms_all:
            if KANJI_RE.search(form):
                if add(form, "yomitan_form", 1):
                    added_forms += 1
                    if added_forms >= 3:
                        break

    # Bikago polite prefix (e.g. 肉 -> お肉, 肌 -> お肌, 茶 -> お茶, 風呂 -> お風呂)
    bikago = expand_bikago_prefix(surface)
    if bikago:
        add(bikago, "bikago_prefix", 1)

    # Verb stative/past and volitional inflections (guided by Yomitan verb_class if present)
    verb_class = yomitan_meta.verb_class if yomitan_meta else None
    verb_inflections = conjugate_japanese_verb(surface, verb_class=verb_class)
    for inflected in verb_inflections:
        if inflected.endswith(("た", "だ", "う")):
            add(inflected, "verb_inflection", 1)

    # Katakana conversion for loanwords and fauna/flora (e.g. とまと -> トマト, ねこ -> ネコ, ぱん -> パン)
    if (
        re.fullmatch(r"[\u3040-\u309F]+", surface)
        and not is_grammar_function_word(surface)
        and not _should_suppress_katakana_conversion(surface, yomitan_meta)
    ):
        add(to_katakana(surface), "loanword_katakana", 1)

    # 3. Tier 2 Semantic & Phonetic Fallbacks (Validated Deinflection, Masu-Stems, Sense Kanji, Cross-References, Note Fields, Readings, Elongation, Speculative Deinflection)
    if allow_tier2:
        deinflections = deinflect_japanese(surface) if allow_deinflection else []
        # Primary validated deinflection from surface (e.g. 帰ってくる -> 帰る, 走っている -> 走る, 待っている -> 待つ)
        # Surface-derived base preserves Kanji stem with zero homophone risk; rank before phonetic readings
        if deinflections:
            add(deinflections[0], "deinflected_base", 2)

        # Masu-stems (e.g. 控え, 疲れ, 勝ち, 売り, 休み) preserve Kanji semantic stem;
        # rank ahead of phonetic kana readings to prevent homophone hijacking (e.g. 控える -> 控え matching 買い控え)
        for inflected in verb_inflections:
            if not inflected.endswith(("た", "だ", "う")):
                add(inflected, "verb_stem", 2)

        # Yomitan sense-specific Kanji splits from notes (e.g. 止める -> esp. 停める, 留める)
        if yomitan_meta and yomitan_meta.sense_kanji:
            added_sense = 0
            for sense_k in yomitan_meta.sense_kanji:
                cleaned_k = clean_search_query(sense_k)
                if (
                    cleaned_k
                    and cleaned_k not in seen
                    and not is_grammar_function_word(cleaned_k)
                ):
                    add(cleaned_k, "yomitan_sense_kanji", 2)
                    added_sense += 1
                    if added_sense >= 3:
                        break

        # Yomitan 'see:' cross-references (e.g. 気づく -> 気がつく, 総理 -> 内閣総理大臣, 軽 -> 軽自動車)
        if yomitan_meta and yomitan_meta.references:
            added_refs = 0
            for ref in yomitan_meta.references:
                cleaned_ref = clean_search_query(ref)
                if (
                    cleaned_ref
                    and cleaned_ref not in seen
                    and not is_grammar_function_word(cleaned_ref)
                ):
                    add(cleaned_ref, "yomitan_see_reference", 2)
                    added_refs += 1
                    if added_refs >= 3:
                        break

        # Curated visual concept map targets (e.g. 亡くなる -> お葬式, 墓; 手に入れる -> 獲得, 宝箱)
        def _resolve_concepts(term: str) -> List[str]:
            """Resolve concept map targets using injected lookup, store, or default catalog."""
            if concept_lookup is not None:
                return concept_lookup(term)
            if store is not None:
                return lookup_concept_map(term, store=store)
            return lookup_concept_map(term)

        concept_targets = _resolve_concepts(surface)
        if not concept_targets and deinflections:
            concept_targets = _resolve_concepts(deinflections[0])
        for c_target in concept_targets:
            add(c_target, "concept_map", 2)

        # Check dedicated reading and furigana fields on the note (authoritative dictionary readings)
        for f_name, f_type, cleaned_read in _iter_note_readings(note):
            prefix = f"note_{f_type}_{f_name}"
            add(cleaned_read, prefix, 2)
            if not _should_suppress_katakana_conversion(cleaned_read, yomitan_meta):
                add(to_katakana(cleaned_read), f"{prefix}_katakana", 2)

        # Same-field reading extracted from furigana brackets (e.g. 檸檬[れもん] -> れもん, 本[ほん] -> ほん)
        # Suppress single kana and grammar function words to avoid homophone collisions
        if same_field_reading and not is_grammar_function_word(same_field_reading):
            add(same_field_reading, "same_field_reading", 2)
            if not _should_suppress_katakana_conversion(same_field_reading, yomitan_meta):
                add(to_katakana(same_field_reading), "same_field_reading_katakana", 2)

        # Katakana elongation mark expansion (メンバ -> メンバー, サーバ -> サーバー)
        for chouon_variant in expand_katakana_chouon(surface):
            add(chouon_variant, "katakana_chouon", 2)

        # Speculative deinflection alternatives (e.g. alternative godan endings)
        if len(deinflections) > 1:
            for alt in deinflections[1:]:
                add(alt, "deinflected_base_speculative", 2)

    return candidates

