"""Prompt templates, tool definitions, and catalog query helpers for Ollama agent."""
import json
import logging
import re
import sqlite3
from typing import Any, Dict, Optional, Set

from addon.catalog import CatalogIndex, CatalogStore
from addon.ranking import (
    has_title_illustration_match,
    rank_relevant_images,
)
from addon.text_utils import clean_search_query
from scripts.synonyms.common import (
    _readonly_catalog_uri,
    filter_relevant_illustrations,
)

logger = logging.getLogger("synonyms.ollama_prompt")

JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf々〆〇]")

VALID_JAPANESE_LATIN_ACRONYMS: Set[str] = {
    "AI", "AR", "ATM", "BGM", "CD", "CM", "DIY", "DJ", "DVD", "FAQ", "GPS",
    "IT", "LAN", "LED", "MC", "MV", "NG", "OK", "OS", "PC", "PR", "PTA",
    "PV", "QR", "SD", "SDGS", "SIM", "SNS", "TV", "URL", "USB", "VIP", "VR",
    "WEB", "WI", "FI", "WIFI",
}


def _contains_english_word(text: str) -> bool:
    """Check whether a candidate target contains embedded English/Latin words.

    Distinguishes English words (e.g. 'cat', 'car') from valid Japanese concepts
    written with Latin letters or initialisms (e.g. 'Tシャツ', 'Wi-Fi', 'QRコード').

    Args:
        text: Target concept or illustration topic candidate.

    Returns:
        True if text contains an embedded English word, False otherwise.
    """
    latin_tokens = re.findall(r"[A-Za-z]+", text)
    for token in latin_tokens:
        if len(token) == 1:
            continue
        if token.upper() in VALID_JAPANESE_LATIN_ACRONYMS:
            continue
        return True
    return False


SEARCH_CATALOG_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_catalog",
        "description": (
            "Search the Irasutoya catalog by Japanese query keywords. "
            "Returns matching Blogger posts (title, description, suggested_target) "
            "and the runtime top-1 illustration ('runtime_top1_if_queried') that runtime ranking selects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 1-3 Japanese search candidate queries to evaluate in the catalog.",
                }
            },
            "required": ["queries"],
        },
    },
}

OLLAMA_SYSTEM_PROMPT = """You are an expert Japanese visual concept compiler for the Irasutoya (いらすとや) illustration dictionary.
Your task is to evaluate a Japanese vocabulary term and map it to 1-3 concrete, specific Japanese Irasutoya illustration topics, scenes, or visual exemplars.

GUIDELINES & EXEMPLAR STRATEGY:
1. ONLY return Japanese concepts or illustration topics (NEVER English words).
2. Direct Visual Concepts vs. Visual Exemplars:
   - Tier 1 (Direct Match / Synonym): Direct concrete objects, specific nouns, or established visual idioms (e.g. 新札 -> いろいろな新紙幣, 傷薬 -> 軟膏, 頭を下げる -> お辞儀). Set "tier": 1.
   - Tier 2 (Visual Exemplar / Scene / Pronoun): Adjectives, spatial/mental states, pronouns, or general verbs that depict a concrete representative scene or prototypical exemplar on Irasutoya (e.g. 広い -> 広い野原, 心の中 -> 悩む人のイラスト, 彼ら -> 仲良しグループ, あの人 -> 指をさす人, 近づく -> 近づいてくる人). Set "tier": 2.
3. Non-Visual Rejections:
   - Reject purely functional grammar, discourse fillers, temporal markers, or disembodied abstract concepts that cannot be represented by a concrete scene (e.g. そのまま, 確かに, 方針, 概念, 数日, 翌日, ことができる, ということだ, わけではない, はずがない).
   - For rejections, set "visualizable": false, "tier": 2, and "targets": [].
4. Output Schema:
   You MUST evaluate the semantic fit in Japanese and output ONLY a JSON object with "reason" first:
   {
     "reason": "思考プロセスや判断理由の簡潔な説明（日本語）",
     "visualizable": true or false,
     "tier": 1 or 2,
     "targets": ["具体的ターゲット1", "具体的ターゲット2"]
   }
5. You have access to the `search_catalog` tool if the initial candidate illustrations do not provide an appropriate target. Inspect `runtime_top1_if_queried` carefully to pick specific, representative targets. Every chosen target MUST exist in the catalog."""


def search_catalog_for_agent(
    catalog_path: str,
    query: str,
    limit: int = 5,
    index: Optional[CatalogIndex] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """Search catalog posts and runtime top-1 preview for an LLM agent query.

    Args:
        catalog_path: Path to catalog.sqlite.
        query: Search term or concept query.
        limit: Max post results to return.
        index: Optional pre-loaded CatalogIndex.
        conn: Optional SQLite connection.

    Returns:
        Dict with 'query', 'posts', and 'runtime_top1_if_queried'.
    """
    clean_q = clean_search_query(query)
    if not clean_q:
        return {"query": query, "posts": [], "runtime_top1_if_queried": None}

    close_conn = False
    if conn is None:
        conn = sqlite3.connect(_readonly_catalog_uri(catalog_path), uri=True)
        close_conn = True

    try:
        cur = conn.cursor()
        escaped_q = (
            clean_q.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        cur.execute(
            "SELECT id, title, description FROM posts WHERE title LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' LIMIT ?",
            (f"%{escaped_q}%", f"%{escaped_q}%", limit),
        )
        posts = []
        for pid, ptitle, pdesc in cur.fetchall():
            m = re.match(r"^(.+?)のイラスト(?:$|（)", ptitle)
            suggested = m.group(1) if m else ptitle
            posts.append({
                "post_id": pid,
                "title": ptitle,
                "description": pdesc[:120] if pdesc else "",
                "suggested_target": suggested,
            })

        top1 = None
        if clean_q:
            if index is None:
                store = CatalogStore(catalog_path)
                index = store.get_index()
            if index is not None:
                results = filter_relevant_illustrations(index.search(clean_q), clean_q)
                if results and has_title_illustration_match(results, clean_q):
                    best_img = rank_relevant_images(results, clean_q)[0]
                    cur.execute("SELECT title, description FROM posts WHERE blogger_id = ?", (best_img.post_id,))
                    p_row = cur.fetchone()
                    top1 = {
                        "matched_image_title": best_img.title,
                        "post_title": p_row[0] if p_row else "",
                        "post_description": (p_row[1][:120] if p_row and p_row[1] else ""),
                    }
        return {"query": query, "posts": posts, "runtime_top1_if_queried": top1}
    finally:
        if close_conn:
            conn.close()


def _parse_llm_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON object from LLM response text, handling markdown fences and commentary.

    Args:
        text: Raw LLM response string.

    Returns:
        Parsed dictionary if successful, None otherwise.
    """
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None
