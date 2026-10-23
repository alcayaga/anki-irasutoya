"""Boundary classification, relevance scoring, and archetype ranking for Irasutoya illustrations."""
import logging
import re
from typing import List, Optional, Sequence

from .feed import IrasutoyaImage
from .text_utils import (
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
ABSTRACT_SINGLE_KANJI = frozenset({"中", "部", "件", "方", "事", "物", "歳"})

DECORATIVE_NOISE_KEYWORDS = ("ライン素材", "背景", "フレーム", "枠", "テンプレート", "壁紙")
TEXT_CALLOUT_KEYWORDS = ("イラスト文字",)
SITUATIONAL_NOISE_KEYWORDS = ("事故", "トラブル", "失敗", "注意", "禁止", "マーク")
VERB_TERMINAL_ENDINGS = ("う", "く", "ぐ", "す", "つ", "ぬ", "ぶ", "む", "る", "た", "だ")
COMMON_OKURIGANA_NOUNS = frozenset({
    "祭り", "祭", "周り", "香り", "祈り", "便り", "望み", "恵み", "答え", "知らせ",
    "怒り", "踊り", "誓い", "願い", "想い", "思い", "集まり", "始まり", "終わり",
})


def _is_bare_verb_stem(text: str) -> bool:
    """Return True if text ends in non-terminal okurigana and is a verb stem (e.g. 売り, 食べ, 買い)."""
    if not text or text in COMMON_OKURIGANA_NOUNS:
        return False
    return bool(
        HIRAGANA_CHAR_RE.match(text[-1])
        and not text.endswith(VERB_TERMINAL_ENDINGS)
        and not is_adjective_ending(text)
    )


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
        or text in (query, f"{query}のイラスト", f"かわいい{query}のイラスト")
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
                particles = ("の", "が", "を", "に", "で", "と", "は", "も", "へ", "や", "な", "か")
                attributive_endings = ("い", "る", "た", "だ")
                # Reject if preceded by Hiragana other than polite お/ご, a particle, or an attributive ending
                prev_is_hira = bool(
                    prev_ch
                    and HIRAGANA_CHAR_RE.match(prev_ch)
                    and prev_ch not in ("お", "ご") + particles + attributive_endings
                )
                # Reject if followed by Hiragana other than a particle
                next_is_hira = bool(
                    next_ch
                    and HIRAGANA_CHAR_RE.match(next_ch)
                    and next_ch not in particles
                )
                if prev_is_hira or next_is_hira:
                    continue
                return "standalone"

        # Case 2: Pure Kanji queries (e.g. 肉, 肌, 本, 二人)
        if is_pure_kanji:
            prev_is_kanji = bool(prev_ch and KANJI_CHAR_RE.match(prev_ch))
            next_is_kanji = bool(next_ch and KANJI_CHAR_RE.match(next_ch))
            if not prev_is_kanji and not next_is_kanji:
                if prev_ch in ("お", "ご"):
                    best_quality = "polite_prefix"
                else:
                    return "standalone"
            else:
                if best_quality == "none":
                    best_quality = "compound"
            continue

        # Case 3: Mixed Kanji + Kana (e.g. 食べる, 1杯, 時に, お金, お茶)
        prev_is_kanji = bool(prev_ch and KANJI_CHAR_RE.match(prev_ch))
        next_is_kanji = bool(next_ch and KANJI_CHAR_RE.match(next_ch))
        # Bare stems (食べ, 売り, 勝ち, 買い) fused with a following Kanji form nouns (食べ物, 売り場, 勝ち組, 買い物)
        # Nouns ending in okurigana (e.g. 祭り) fused with a following Kanji form compound nouns (祭り会場)
        is_noun_compound = query in COMMON_OKURIGANA_NOUNS and next_is_kanji
        ends_in_stem_okurigana = _is_bare_verb_stem(query)
        if (prev_is_kanji and KANJI_CHAR_RE.match(query[0])) or (
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
        # Disallow compound matches for 1-Kanji grammatical/abstract words
        if len(clean_q) == 1 and clean_q in ABSTRACT_SINGLE_KANJI:
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


def score_candidate_image(img: IrasutoyaImage, query: str) -> float:
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
        elif title in (f"お{clean_q}のイラスト", f"ご{clean_q}のイラスト"):
            score += 95.0
        elif title == f"かわいい{clean_q}のイラスト":
            score += 90.0
        elif title == clean_q:
            score += 85.0
        elif title.startswith(f"{clean_q}の"):
            score += 60.0
        elif post_title == f"{clean_q}のイラスト":
            score += 50.0
        elif post_title.startswith(f"{clean_q}の"):
            score += 30.0

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
        score -= 35.0
    elif is_image_relevant(img, clean_q):
        # Matched only via description on single-image post
        score += 15.0
    else:
        score -= 50.0  # Heavy penalty: image does not mention the query

    # 3. Penalize length (shorter, direct titles are more archetypal)
    score -= len(title) * 0.8

    # 4. Situational, hazard, and mark noise penalty
    for noise in SITUATIONAL_NOISE_KEYWORDS:
        if noise in title and noise not in clean_q:
            score -= 25.0

    # 5. Heavy penalty for text callouts (イラスト文字 / trailing 文字) so illustrations always win if available
    is_callout = is_callout_asset(img, clean_q)
    if is_callout:
        score -= 65.0

    # 6. Filter out non-illustration decorative assets (lines, borders, wallpaper, templates)
    for noise in DECORATIVE_NOISE_KEYWORDS:
        if noise in title and noise not in clean_q:
            score -= 60.0

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


def rank_relevant_images(candidates: Sequence[IrasutoyaImage], query: str) -> List[IrasutoyaImage]:
    """Rank relevant images by preference: non-callout illustrations first, then text callouts, ordered by score."""
    scored = [(img, score_candidate_image(img, query), is_callout_asset(img, query)) for img in candidates]
    non_callouts = sorted((s for s in scored if not s[2]), key=lambda s: s[1], reverse=True)
    callouts = sorted((s for s in scored if s[2]), key=lambda s: s[1], reverse=True)
    return [s[0] for s in non_callouts + callouts]


def select_best_image(images: Sequence[IrasutoyaImage], query: str) -> Optional[IrasutoyaImage]:
    """Select the highest-ranked archetypal image from candidate images, excluding decorative assets."""
    if not images:
        return None
    illustration_candidates = [
        img for img in images
        if not is_decorative_asset(img, query) and is_image_relevant(img, query)
    ]
    if illustration_candidates:
        return rank_relevant_images(illustration_candidates, query)[0]
    fallback_pool = [img for img in images if not is_decorative_asset(img, query)] or images
    return max(fallback_pool, key=lambda img: score_candidate_image(img, query))
