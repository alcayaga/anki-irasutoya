"""Boundary classification, relevance scoring, and archetype ranking for Irasutoya illustrations."""
import logging
import re
from typing import Any, List, Optional, Sequence

from .feed import IrasutoyaImage
from .text_utils import (
    NON_VERB_GODAN_LOOKALIKES,
    SAFE_2MORA_LOANWORDS,
    conjugate_japanese_verb,
    is_adjective_ending,
    is_grammar_function_word,
)

logger = logging.getLogger(__name__)

KANJI_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf々〆〇]")
HIRAGANA_CHAR_RE = re.compile(r"[\u3040-\u309F]")
# Exclude ゠ (U+30A0) and ・ (U+30FB); they are word separators, not Katakana letters.
KATAKANA_CHAR_RE = re.compile(r"[\u30A1-\u30FA\u30FC-\u30FF]")

DECORATIVE_NOISE_KEYWORDS = ("ライン素材", "背景", "フレーム", "枠", "テンプレート", "壁紙")
TEXT_CALLOUT_KEYWORDS = ("イラスト文字",)
SITUATIONAL_NOISE_KEYWORDS = ("事故", "トラブル", "失敗", "注意", "禁止", "マーク")
VERB_TERMINAL_ENDINGS = ("う", "く", "ぐ", "す", "つ", "ぬ", "ぶ", "む", "る", "た", "だ")
SINGLE_CHAR_PARTICLES = ("の", "が", "を", "に", "で", "と", "は", "も", "へ", "や", "な", "か")
MULTI_CHAR_PARTICLES = ("から", "より", "ながら", "など", "だけ", "ほど", "ばかり", "まで")
ATTRIBUTIVE_ENDINGS = ("い",) + VERB_TERMINAL_ENDINGS
COLLECTION_PREFIXES = ("いろいろな", "色々な", "様々な", "さまざまな")
COMPOUND_HEADWORD_SYNONYMS = {
    "仮面": ("マスク", "お面", "面"),
}
BYPRODUCT_MODIFIER_SEPARATORS = ("のついた", "がついた", "の付いた", "が付いた", "の入った", "が入った")
NATURAL_DISASTER_QUERIES = frozenset({
    "台風", "地震", "津波", "洪水", "噴火", "竜巻", "落雷", "土砂崩れ", "雪崩",
    "豪雨", "暴風", "暴風雨", "大雨", "大雪", "嵐", "ハリケーン", "サイクロン",
    "山火事", "火災", "火事", "災害", "自然災害", "避難", "防災",
})
SUBTYPE_PREFIXES = frozenset({
    # Colors
    "黒", "白", "赤", "青", "黄", "茶", "灰", "緑", "紫", "金", "銀",
    # Size & age
    "大", "小", "子", "巨", "微", "老", "若", "新", "古",
    # Species & attributes
    "柴", "野", "三",
})
COMMON_OKURIGANA_NOUNS = frozenset({
    "祭り", "祭", "周り", "香り", "祈り", "便り", "望み", "恵み", "答え", "知らせ",
    "怒り", "踊り", "誓い", "願い", "想い", "思い", "集まり", "始まり", "終わり",
    "こたつ", "くつ", "ねずみ", "さる",
    "一つ", "二つ", "三つ", "四つ", "五つ", "六つ", "七つ", "八つ", "九つ",
    "近く", "遠く", "多く",
})

COMMON_KANA_VERBS = frozenset({
    "はしる", "たべる", "あるく", "およぐ", "のむ", "あそぶ",
    "みる", "きく", "ねる", "おきる", "たつ", "すわる",
    "いう", "いく", "くる", "かう", "うる", "かく", "よむ",
    "わらう", "なく", "おこる", "おどる", "うたう", "とぶ",
    "はなす", "まつ", "もつ", "しぬ", "よぶ", "あらう",
})

COMMON_ABSTRACT_NOUNS = frozenset({
    "禁止", "注意", "危険", "安全", "成功", "失敗", "平和", "関係", "状態",
    "方法", "原因", "結果", "自由", "練習", "勉強", "努力", "希望", "夢",
    "愛", "心", "気持ち", "思い出", "時間", "約束", "秘密", "ルール", "マナー",
    "名前",
})

# Anti-archetype categories (callout typography, symbols, frames, decorative lines, templates)
UNIVERSAL_ANTI_ARCHETYPE_CATEGORIES = {
    "文字": -50.0,
    "書体": -50.0,
    "ペンキ書体": -50.0,
    "白抜き書体": -50.0,
    "マーク": -40.0,
    "旗": -35.0,
    "フレーム": -50.0,
    "ライン": -50.0,
    "パターン": -45.0,
    "テンプレート": -40.0,
    "飾り": -40.0,
    "一筆箋": -40.0,
    "伝言メモ": -40.0,
    "メッセージカード": -40.0,
    "POP": -35.0,
}

# Situational hazard noise categories (penalized unless query itself is a hazard/violation term)
SITUATIONAL_HAZARD_CATEGORIES = {
    "事故": -25.0,
    "違反": -25.0,
    "怪我": -25.0,
    "災害": -25.0,
    "トラブル": -25.0,
}

# Action / motion / pose archetype categories (boosted for verbs / actions)
ACTION_ARCHETYPE_CATEGORIES = {
    "ポーズ": 30.0,
    "運動": 25.0,
    "スポーツ": 20.0,
    "表情": 20.0,
    "感情": 20.0,
    "食事": 20.0,
    "掃除": 20.0,
    "洗濯": 20.0,
    "料理": 15.0,
    "仕事": 15.0,
    "職業": 15.0,
    "ビジネス": 10.0,
    "棒人間": 15.0,
}

# Concrete physical object / entity archetype categories (boosted for nouns)
OBJECT_ARCHETYPE_CATEGORIES = {
    "食べ物": 25.0,
    "料理": 20.0,
    "食材": 20.0,
    "フルーツ": 20.0,
    "野菜": 20.0,
    "スイーツ": 20.0,
    "パン": 20.0,
    "動物": 25.0,
    "鳥": 20.0,
    "魚": 20.0,
    "犬": 20.0,
    "猫": 20.0,
    "虫": 20.0,
    "植物": 20.0,
    "花": 20.0,
    "乗り物": 20.0,
    "車": 20.0,
    "電車": 20.0,
    "自転車": 20.0,
    "道具": 20.0,
    "文房具": 20.0,
    "本": 20.0,
    "家具": 15.0,
    "家電": 15.0,
    "おもちゃ": 15.0,
    "楽器": 15.0,
}

# State / attribute / emotion archetype categories (boosted for adjectives)
ADJECTIVE_ARCHETYPE_CATEGORIES = {
    "感情": 30.0,
    "表情": 30.0,
    "ポーズ": 25.0,
    "天気": 20.0,
    "美容": 20.0,
}


def _is_canonical_mark_title(title: Optional[str], query: str) -> bool:
    """Return True if title names the query's own mark illustration (e.g. 禁止マーク)."""
    if not title or not query:
        return False
    t = title.strip()
    q = query.strip()
    # A mark illustration is only canonical for mark-specific queries, abstract concepts, or signage
    if (
        not q.endswith("マーク")
        and q not in COMMON_ABSTRACT_NOUNS
        and q not in ("初心者", "交通安全", "リサイクル", "エコ", "非常口", "優先席")
    ):
        return False
    prefixes = (
        f"{q}マーク",
        f"{q}のマーク",
        f"「{q}」マーク",
        f"『{q}』マーク",
        f"（{q}）マーク",
        f"({q})マーク",
        f"「{q}」のマーク",
        f"『{q}』のマーク",
        f"お{q}マーク",
        f"ご{q}マーク",
        f"お{q}のマーク",
        f"ご{q}のマーク",
    )
    return any(
        t == f"{prefix}{suffix}"
        for prefix in prefixes
        for suffix in ("", "のイラスト")
    )


def _is_byproduct_modifier_title(text: Optional[str], query: str) -> bool:
    """Return True if title describes a byproduct modifier (e.g. 髪の毛のついたブラシ, 髪の毛の入ったご飯)."""
    if not text or not query:
        return False
    q = query.strip()
    return any(
        text.startswith(f"{prefix}{sep}")
        for prefix in (q, f"お{q}", f"ご{q}")
        for sep in BYPRODUCT_MODIFIER_SEPARATORS
    )


def calculate_category_score(
    categories: Sequence[str],
    query: str,
    pos_type: Optional[str] = None,
    title: Optional[str] = None,
    allow_boost: bool = True,
) -> float:
    """Calculate archetype score adjustment from image categories based on query and POS."""
    if not categories:
        return 0.0

    score = 0.0
    clean_q = query.strip()
    cat_set = set(categories)
    is_canonical_mark = _is_canonical_mark_title(title, clean_q)

    # 1. Anti-archetype penalty (capped at -70.0)
    anti_penalty = 0.0
    for cat, penalty in UNIVERSAL_ANTI_ARCHETYPE_CATEGORIES.items():
        if cat in cat_set and cat not in clean_q:
            if cat == "マーク" and is_canonical_mark:
                continue
            # Avoid applying duplicate penalty if title keyword already applied it
            if title and (
                (cat in ("フレーム", "テンプレート") and cat in title)
                or (cat == "ライン" and "ライン素材" in title)
                or (cat == "マーク" and "マーク" in title)
                or (cat == "文字" and ("イラスト文字" in title or title.endswith("文字")))
            ):
                continue
            anti_penalty += penalty
    score += max(anti_penalty, -70.0)

    # 2. Situational hazard penalty (skip if query itself is a hazard term)
    hazard_penalty = 0.0
    has_unrelated_hazard = False
    for cat, penalty in SITUATIONAL_HAZARD_CATEGORIES.items():
        if cat in cat_set and cat not in clean_q:
            if cat == "災害" and clean_q in NATURAL_DISASTER_QUERIES:
                continue
            has_unrelated_hazard = True
            # Avoid applying duplicate penalty if title keyword already applied it
            if title and cat in SITUATIONAL_NOISE_KEYWORDS and cat in title:
                continue
            hazard_penalty += penalty

    # 3. Positive archetype boost (capped at +35.0, gated by allow_boost)
    boost = 0.0
    if allow_boost:
        if clean_q in cat_set:
            boost += 25.0

        effective_pos = pos_type or infer_pos_type(clean_q)
        if effective_pos == "verb":
            for cat, b in ACTION_ARCHETYPE_CATEGORIES.items():
                if cat in cat_set:
                    boost += b
        elif effective_pos == "adjective":
            for cat, b in ADJECTIVE_ARCHETYPE_CATEGORIES.items():
                if cat in cat_set:
                    boost += b
        elif effective_pos == "noun" and clean_q not in COMMON_ABSTRACT_NOUNS:
            for cat, b in OBJECT_ARCHETYPE_CATEGORIES.items():
                if cat in cat_set:
                    boost += b

    # When unrelated hazard categories are present, suppress positive boosts to ensure net penalty
    if has_unrelated_hazard or hazard_penalty < 0.0:
        boost = 0.0
    else:
        boost = min(boost, 35.0)

    score += hazard_penalty + boost
    return score


def infer_note_pos_type(yomitan_meta: Optional[Any]) -> Optional[str]:
    """Infer the part of speech profile ('verb', 'adjective', or 'noun') strictly from note metadata."""
    if yomitan_meta is None:
        return None
    if getattr(yomitan_meta, "verb_class", None):
        return "verb"
    tags = getattr(yomitan_meta, "pos_tags", set())
    if any(t == "v" or t.startswith(("v1", "v5", "vs", "vk", "vz", "vi", "vt")) for t in tags):
        return "verb"
    if any(t.startswith("adj") and t not in ("adj-no", "adj-f", "adj-pn") for t in tags):
        return "adjective"
    if any(t == "n" or t.startswith("n-") for t in tags):
        return "noun"
    return None


def infer_pos_type(
    query: str,
    cand_reason: Optional[str] = None,
    yomitan_meta: Optional[Any] = None,
) -> Optional[str]:
    """Infer the part of speech profile ('verb', 'adjective', 'noun', or None) for category scoring."""
    note_pos = infer_note_pos_type(yomitan_meta)
    if note_pos:
        return note_pos

    if cand_reason:
        if cand_reason.startswith("verb_") or cand_reason in ("primary_verb", "yomitan_masu_stem"):
            return "verb"

    clean_q = query.strip()
    if not clean_q:
        return None

    if is_adjective_ending(clean_q):
        return "adjective"

    # Require Kanji stem + Hiragana terminal okurigana to infer verb from morphology
    if (
        len(clean_q) > 1
        and KANJI_CHAR_RE.search(clean_q)
        and clean_q.endswith(VERB_TERMINAL_ENDINGS)
        and clean_q not in COMMON_OKURIGANA_NOUNS
        and clean_q not in NON_VERB_GODAN_LOOKALIKES
    ):
        return "verb"

    # Recognize common kana-only verbs
    if clean_q in COMMON_KANA_VERBS:
        return "verb"

    # Known nouns ending in okurigana/kana
    if clean_q in COMMON_OKURIGANA_NOUNS:
        return "noun"

    # Words ending in Kanji (e.g. 本, 猫, お肉, 食べ物, 贈り物) are nouns
    if KANJI_CHAR_RE.match(clean_q[-1]):
        return "noun"

    # Pure Katakana words are nouns (loanwords)
    if all(KATAKANA_CHAR_RE.match(c) for c in clean_q):
        return "noun"

    # Pure Hiragana terms that do NOT end in verb terminal endings cannot be verbs
    if all(HIRAGANA_CHAR_RE.match(c) for c in clean_q) and not clean_q.endswith(VERB_TERMINAL_ENDINGS):
        return "noun"

    # Kana terms ending in verb terminal endings without Kanji or explicit metadata are ambiguous;
    # leave them neutral (None) to avoid assigning false noun or verb archetype boosts.
    return None


def _is_bare_verb_stem(text: str) -> bool:
    """Return True if text ends in non-terminal okurigana and is a verb stem (e.g. 売り, 食べ, 買い)."""
    if not text or text in COMMON_OKURIGANA_NOUNS:
        return False
    return bool(
        HIRAGANA_CHAR_RE.match(text[-1])
        and not text.endswith(VERB_TERMINAL_ENDINGS)
        and not is_adjective_ending(text)
    )


def _is_non_particle_hiragana_prefix(text: str, idx: int) -> bool:
    """Return True if character before text[idx] is Hiragana forming a non-particle compound prefix."""
    prev_ch = text[idx - 1] if idx > 0 else ""
    if not prev_ch or not HIRAGANA_CHAR_RE.match(prev_ch):
        return False
    if text[:idx].endswith(MULTI_CHAR_PARTICLES):
        return False
    # Verbal te-form clause connector (〜って, 〜いて, 〜して, etc.) is not a compound prefix
    if prev_ch == "て" and idx >= 2 and HIRAGANA_CHAR_RE.match(text[idx - 2]):
        return False
    if prev_ch in ("お", "ご"):
        return False
    # A single-character particle or attributive ending cannot exist at string-initial idx == 1
    # or immediately following punctuation/bracket/whitespace
    if idx < 2:
        return True
    char_before = text[idx - 2]
    CLOSING_BRACKETS = ("」", "』", "）", ")", "】", "］", "]", "〉", "》")
    if not (
        KANJI_CHAR_RE.match(char_before)
        or KATAKANA_CHAR_RE.match(char_before)
        or HIRAGANA_CHAR_RE.match(char_before)
        or char_before.isalnum()
        or char_before in CLOSING_BRACKETS
    ):
        return True
    return prev_ch not in SINGLE_CHAR_PARTICLES + ATTRIBUTIVE_ENDINGS


def classify_match_boundary(text: str, query: str) -> str:
    """Classify the boundary quality of query within text.

    Returns:
        'exact_quoted': explicitly quoted 「query」/『query』or exact title match
        'standalone': bounded by particles, punctuation, or script transitions
        'polite_prefix': preceded by お/ご with valid trailing boundary (e.g. お肉のイラスト)
        'inflected': an inflected verb form (e.g. 売ろう, 売り for 売る) appears
        'compound': fused with an adjacent Kanji or Katakana (e.g. 培養肉, カフェオレ)
        'none': not found or invalid mid-word Kana collision
    """
    if not text or not query:
        return "none"

    if (
        f"「{query}」" in text
        or f"『{query}』" in text
        or f"（{query}）" in text
        or f"({query})" in text
        or text in (
            query,
            f"{query}のイラスト",
            f"かわいい{query}のイラスト",
        )
        or _is_canonical_mark_title(text, query)
    ):
        return "exact_quoted"

    is_pure_kana = bool(re.fullmatch(r"[\u3040-\u309F\u30A0-\u30FF]+", query))
    is_pure_kanji = bool(re.fullmatch(r"[\u4e00-\u9fff\u3400-\u4dbf々〆〇]+", query))

    start = 0
    best_quality = "none"
    while True:
        idx = text.find(query, start)
        if idx == -1:
            break
        start = idx + 1
        prev_ch = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + len(query)] if idx + len(query) < len(text) else ""

        # Case 1: Pure Kana queries (high collision risk!)
        if is_pure_kana:
            is_katakana_q = bool(re.fullmatch(r"[\u30A0-\u30FF]+", query))
            if is_katakana_q:
                # Must not be embedded mid-Katakana (e.g. オレ in カフェオレ, モン in モンスター)
                prev_is_kata = bool(prev_ch and KATAKANA_CHAR_RE.match(prev_ch))
                next_is_kata = bool(next_ch and KATAKANA_CHAR_RE.match(next_ch))
                if prev_is_kata or next_is_kata:
                    continue
                if is_grammar_function_word(query) or len(query) < 2:
                    continue
                return "standalone"
            else:
                # Hiragana:
                # Grammar stop words and unlisted 2-mora words: only exact quoted matches allowed
                if is_grammar_function_word(query) or (len(query) <= 2 and query not in SAFE_2MORA_LOANWORDS):
                    continue
                # Reject if preceded by Hiragana other than polite お/ご, verbal te-form, particle, or attributive ending
                prev_is_hira = _is_non_particle_hiragana_prefix(text, idx)
                # Reject if followed by Hiragana other than a particle
                next_is_hira = bool(
                    next_ch
                    and HIRAGANA_CHAR_RE.match(next_ch)
                    and next_ch not in SINGLE_CHAR_PARTICLES
                )
                # Reject if followed directly by a Kanji unless query ends in an attributive adjective/verb ending
                next_is_kanji_compound = bool(
                    next_ch
                    and KANJI_CHAR_RE.match(next_ch)
                    and not query.endswith(ATTRIBUTIVE_ENDINGS)
                )
                if prev_is_hira or next_is_hira or next_is_kanji_compound:
                    continue
                return "standalone"

        # Case 2: Pure Kanji queries (e.g. 肉, 肌, 本, 二人, 禁止)
        if is_pure_kanji:
            prev_is_kanji = bool(prev_ch and KANJI_CHAR_RE.match(prev_ch))
            next_is_kanji = bool(next_ch and KANJI_CHAR_RE.match(next_ch))
            prev_is_kata = bool(prev_ch and KATAKANA_CHAR_RE.match(prev_ch))
            next_is_kata = bool(next_ch and KATAKANA_CHAR_RE.match(next_ch))
            prev_is_non_particle_hira = _is_non_particle_hiragana_prefix(text, idx)
            if prev_is_kanji or next_is_kanji or prev_is_kata or next_is_kata or prev_is_non_particle_hira:
                if best_quality == "none":
                    best_quality = "compound"
            else:
                if prev_ch in ("お", "ご"):
                    best_quality = "polite_prefix"
                else:
                    return "standalone"
            continue

        # Case 3: Mixed Kanji + Kana (e.g. 食べる, 1杯, 時に, お金, お茶)
        prev_is_kanji = bool(prev_ch and KANJI_CHAR_RE.match(prev_ch))
        next_is_kanji = bool(next_ch and KANJI_CHAR_RE.match(next_ch))
        prev_is_kata = bool(prev_ch and KATAKANA_CHAR_RE.match(prev_ch))
        # Bare stems (食べ, 売り, 勝ち, 買い) fused with a following Kanji form nouns (食べ物, 売り場, 勝ち組, 買い物)
        # Nouns ending in okurigana (e.g. 祭り) fused with a following Kanji form compound nouns (祭り会場)
        is_noun_compound = query in COMMON_OKURIGANA_NOUNS and next_is_kanji
        ends_in_stem_okurigana = _is_bare_verb_stem(query)
        if prev_is_kata or (prev_is_kanji and KANJI_CHAR_RE.match(query[0])) or (
            next_is_kanji and (KANJI_CHAR_RE.match(query[-1]) or ends_in_stem_okurigana or is_noun_compound)
        ):
            if best_quality == "none":
                best_quality = "compound"
            continue
        return "standalone"

    # Check verb inflections (e.g. 疲れた人のイラスト for 疲れる, 勝った人のイラスト for 勝つ)
    for inflected in conjugate_japanese_verb(query):
        is_stem = not inflected.endswith(("た", "だ", "う"))
        pos = text.find(inflected)
        while pos != -1:
            p_ch = text[pos - 1] if pos > 0 else ""
            end = pos + len(inflected)
            n_ch = text[end] if end < len(text) else ""
            prev_kanji = bool(p_ch and KANJI_CHAR_RE.match(p_ch))
            next_kanji = bool(n_ch and KANJI_CHAR_RE.match(n_ch))
            # Must not be preceded by Kanji, and bare stems (食べ, 売り) must not form Kanji compounds (食べ物, 売り場)
            if not prev_kanji and not (is_stem and next_kanji):
                return "inflected"
            pos = text.find(inflected, pos + 1)

    return best_quality


def _is_relevant_single_kanji_subtype(title: str, query: str, categories: Sequence[str] = ()) -> bool:
    """Return True if title is a genuine subtype compound where query is the head noun (e.g. 黒猫, 柴犬)."""
    if len(query) != 1:
        return True
    idx = title.find(query)
    if idx == -1:
        return False
    next_ch = title[idx + 1] if idx + 1 < len(title) else ""
    next_is_fused = bool(
        next_ch
        and (KANJI_CHAR_RE.match(next_ch) or KATAKANA_CHAR_RE.match(next_ch))
    )
    # A subtype compound must be head-final (e.g. 黒猫, not 猫背 or 猫耳)
    if next_is_fused:
        return False
    # If explicitly categorized by Irasutoya as this entity (e.g. 柴犬 in '犬', 盲導犬 in '犬')
    if query in categories:
        return True
    if idx > 0:
        prev_ch = title[idx - 1]
        char_before = title[idx - 2] if idx >= 2 else ""
        prev_is_word = bool(
            char_before
            and (
                KANJI_CHAR_RE.match(char_before)
                or _is_non_particle_hiragana_prefix(title, idx - 1)
            )
        )
        if prev_ch in SUBTYPE_PREFIXES and not prev_is_word:
            return True
    return False


def is_image_relevant(img: IrasutoyaImage, query: str) -> bool:
    """Check if the individual image itself is semantically relevant with boundary safety.

    Prevents false-positive matches where a multi-image post matching a query contains
    unrelated illustrations, or where short queries collide mid-word.
    """
    clean_q = query.strip()
    if not clean_q:
        return False
    title = img.title.strip()

    # 1. Title boundary classification
    qual_title = classify_match_boundary(title, clean_q)
    if qual_title in ("exact_quoted", "standalone", "polite_prefix", "inflected"):
        return True
    if qual_title == "compound":
        # For single-character queries, only genuine subtype compounds (e.g. 黒猫 for 猫, 柴犬 for 犬)
        # where the query is the head noun are eligible; incidental compounds are rejected.
        if len(clean_q) == 1:
            cats = getattr(img, "categories", ())
            if not _is_relevant_single_kanji_subtype(title, clean_q, cats):
                return False
        # Bare verb stems fused with following Kanji form different nouns (売り場, 勝ち組, 食べ物, 買い物)
        if _is_bare_verb_stem(clean_q) and re.search(
            re.escape(clean_q) + KANJI_CHAR_RE.pattern, title
        ):
            return False
        return True

    # 2. Single-image post caption/description match
    if getattr(img, "is_single_image_post", True) and not is_grammar_function_word(clean_q):
        desc = getattr(img, "description", "").strip()
        if desc:
            qual_desc = classify_match_boundary(desc, clean_q)
            if qual_desc in ("exact_quoted", "standalone", "polite_prefix", "inflected"):
                return True

    # 3. Multi-word queries
    tokens = clean_q.split()
    if len(tokens) > 1 and all(classify_match_boundary(title, t) != "none" for t in tokens):
        return True

    return False


def score_candidate_image(
    img: IrasutoyaImage,
    query: str,
    pos_type: Optional[str] = None,
) -> float:
    """Evaluate image semantic match quality and archetypal fit for flashcard learning."""
    score = 0.0
    clean_q = query.strip()
    title = img.title.strip()
    post_title = img.post_title.strip()

    qual_title = classify_match_boundary(title, clean_q)

    # 1. Exact canonical matches
    if is_image_relevant(img, clean_q):
        if title == f"{clean_q}のイラスト":
            score += 100.0
        elif _is_canonical_mark_title(title, clean_q):
            score += 95.0
        elif title in (f"お{clean_q}のイラスト", f"ご{clean_q}のイラスト"):
            score += 95.0
        elif title == f"かわいい{clean_q}のイラスト":
            score += 90.0
        elif title == clean_q:
            score += 85.0
        elif title.startswith(f"{clean_q}の") and not _is_byproduct_modifier_title(title, clean_q):
            score += 60.0
        elif post_title == f"{clean_q}のイラスト" or _is_canonical_mark_title(post_title, clean_q):
            score += 50.0
        elif post_title.startswith(f"{clean_q}の") and not _is_byproduct_modifier_title(post_title, clean_q):
            score += 30.0

    # Archetype collection boost (e.g. いろいろなネジのイラスト)
    if is_image_relevant(img, clean_q) and any(
        title in (
            f"{prefix}{clean_q}のイラスト",
            f"{prefix}お{clean_q}のイラスト",
            f"{prefix}ご{clean_q}のイラスト",
        )
        for prefix in COLLECTION_PREFIXES
    ):
        score += 15.0

    # 2. Boundary, politeness, and focus bonuses
    if qual_title == "exact_quoted" or f"「{clean_q}」" in title or f"（{clean_q}）" in title:
        score += 55.0
    elif qual_title == "polite_prefix":
        score += 45.0
    elif qual_title == "standalone":
        score += 35.0
    elif qual_title == "inflected":
        score += 30.0
    elif qual_title == "compound":
        has_synonym = any(
            classify_match_boundary(title, syn) in ("exact_quoted", "standalone", "polite_prefix")
            for syn in COMPOUND_HEADWORD_SYNONYMS.get(clean_q, ())
        )
        if has_synonym:
            score += 35.0
        else:
            score -= 35.0
    elif is_image_relevant(img, clean_q):
        # Matched only via description on single-image post
        score += 15.0
    else:
        score -= 50.0  # Heavy penalty: image does not mention the query

    # 3. Penalize length (shorter, direct titles are more archetypal)
    score -= len(title) * 0.8

    # 4. Situational, hazard, and mark noise penalty
    is_canonical_mark = _is_canonical_mark_title(title, clean_q)
    for noise in SITUATIONAL_NOISE_KEYWORDS:
        if noise in title and noise not in clean_q:
            if noise == "マーク" and is_canonical_mark:
                continue
            score -= 25.0

    # Attributive byproduct modifier penalty (e.g. 髪の毛のついたヘアブラシ, 髪の毛の入ったご飯)
    if _is_byproduct_modifier_title(title, clean_q) or _is_byproduct_modifier_title(post_title, clean_q):
        score -= 20.0

    # 5. Heavy penalty for text callouts (イラスト文字 / trailing 文字) so illustrations always win if available
    is_callout = is_callout_asset(img, clean_q)
    if is_callout:
        score -= 65.0

    # 6. Filter out non-illustration decorative assets (lines, borders, wallpaper, templates)
    for noise in DECORATIVE_NOISE_KEYWORDS:
        if noise in title and noise not in clean_q:
            score -= 60.0

    # 7. Category-aware archetype scoring (modulated by POS, gated by title presence)
    cats = getattr(img, "categories", ())
    if cats:
        has_title_presence = bool(
            qual_title == "inflected"
            or clean_q in title
            or clean_q in post_title
            or (post_title and classify_match_boundary(post_title, clean_q) == "inflected")
        )
        score += calculate_category_score(
            cats, clean_q, pos_type=pos_type, title=title, allow_boost=has_title_presence
        )

    return score


def is_callout_asset(img: IrasutoyaImage, query: str) -> bool:
    """Return True if image is an illustrated text callout (イラスト文字 or trailing 文字)."""
    clean_q = query.strip()
    title = img.title.strip()
    return ("イラスト文字" in title and "イラスト文字" not in clean_q) or (
        title.endswith("文字") and "文字" not in clean_q
    )


def is_decorative_asset(img: IrasutoyaImage, query: str) -> bool:
    """Return True if image is a decorative asset (frame, border, wallpaper) not matching query."""
    clean_q = query.strip()
    title = img.title.strip()
    return any(noise in title and noise not in clean_q for noise in DECORATIVE_NOISE_KEYWORDS)


def rank_relevant_images(
    candidates: Sequence[IrasutoyaImage],
    query: str,
    pos_type: Optional[str] = None,
) -> List[IrasutoyaImage]:
    """Rank relevant images by preference: non-callout illustrations first, then text callouts, ordered by score."""
    scored = [
        (img, score_candidate_image(img, query, pos_type=pos_type), is_callout_asset(img, query))
        for img in candidates
    ]
    non_callouts = sorted((s for s in scored if not s[2]), key=lambda s: s[1], reverse=True)
    callouts = sorted((s for s in scored if s[2]), key=lambda s: s[1], reverse=True)
    return [s[0] for s in non_callouts + callouts]


def select_best_image(
    images: Sequence[IrasutoyaImage],
    query: str,
    pos_type: Optional[str] = None,
) -> Optional[IrasutoyaImage]:
    """Select the highest-ranked archetypal image from candidate images, excluding decorative assets."""
    if not images:
        return None
    illustration_candidates = [
        img for img in images
        if not is_decorative_asset(img, query) and is_image_relevant(img, query)
    ]
    if illustration_candidates:
        return rank_relevant_images(illustration_candidates, query, pos_type=pos_type)[0]
    fallback_pool = [img for img in images if not is_decorative_asset(img, query)] or images
    return max(fallback_pool, key=lambda img: score_candidate_image(img, query, pos_type=pos_type))
