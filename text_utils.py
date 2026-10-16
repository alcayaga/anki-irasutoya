"""Text utilities for sanitizing search queries from Anki card fields."""
import html
import re
from typing import Optional


def clean_search_query(raw_text: Optional[str]) -> str:
    """Extract a clean keyword from a raw Anki field value.
    
    Removes:
    - HTML tags (e.g. <div>, <b>, <span style=...>)
    - Anki sound tags: [sound:filename.mp3]
    - Anki furigana brackets: 食べる[たべる] -> 食べる
    - Parenthesized readings: 猫（ねこ） -> 猫, 犬(いぬ) -> 犬
    - Non-essential punctuation
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

    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. If there are multiple alternatives separated by slashes, commas, middle dots, etc., take the first
    for sep in ["/", "・", ";", "；", "、", ","]:
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if parts:
                text = parts[0]
                break

    # 8. Clean leftover edge punctuation
    text = text.strip(" ~〜-ー:：!！?？. 。")

    return text
