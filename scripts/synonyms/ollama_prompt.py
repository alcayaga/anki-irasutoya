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
from scripts.synonyms.ollama_embed import (
    format_query_for_embedding,
    resolve_valid_post_target,
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
   - Tier 1 (Direct Match / Synonym): Direct concrete objects, specific nouns, or established visual idioms (e.g. 新札 -> いろいろな新紙幣, 傷薬 -> 絆創膏, 頭を下げる -> お辞儀). Set "tier": 1.
   - Tier 2 (Visual Exemplar / Scene / Pronoun): Adjectives, spatial/mental states, pronouns, or general verbs that depict a concrete representative scene or prototypical exemplar on Irasutoya (e.g. 広い -> 草原, 心の中 -> 頭を抱えて悩んでいる人, 彼ら -> 集合している人たち, あの人 -> 指差し, 近づく -> どうぞこちらへ). Set "tier": 2.
3. Non-Visual Rejections (No Self-Persuasion):
   - Immediately reject functional grammar, compound particles, discourse fillers, modal adverbs, temporal markers, or disembodied abstract nouns (e.g. そのまま, 確かに, 少しでも, 方針, 概念, 数日, 翌日, ことができる, ということだ, わけではない, はずがない, ようにする, おそらく, に対して).
   - NEVER pivot with "しかし" or "ただし" to rationalize a tangential scene or kanji pun (e.g. never map に対して to 対局 or おそらく to 想像).
   - For rejections, set "visualizable": false, "tier": 2, and "targets": [].
4. Output Schema (Telegraphic Reason Under 40 Characters):
   Output ONLY a JSON object with "reason" first.
   "reason" MUST be a single telegraphic phrase under 40 Japanese characters (e.g. "直接一致: 新紙幣のイラスト", "視覚的模範: 草原の風景", or "複合助詞のため非視覚的"). Never write multi-sentence explanations:
   {
     "reason": "40文字以内の簡潔な1フレーズ理由（『しかし』等の逆接・長文禁止）",
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
    vector_index: Optional[Any] = None,
    embed_client: Optional[Any] = None,
    embed_model: str = "qwen3-embedding:4b",
) -> Dict[str, Any]:
    """Search catalog posts and runtime top-1 preview for an LLM agent query.

    Matches post title first, then post description, and falls back to semantic
    vector search when vector_index is provided and title hits are below limit.

    Args:
        catalog_path: Path to catalog.sqlite.
        query: Search term or concept query.
        limit: Max post results to return.
        index: Optional pre-loaded CatalogIndex.
        conn: Optional SQLite connection.
        vector_index: Optional pre-loaded CatalogVectorIndex for semantic hits.
        embed_client: Optional OllamaEmbedClient instance.
        embed_model: Embedding model tag.

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
        posts = []
        seen_post_ids: Set[int] = set()

        # 1a. Match post title first
        cur.execute(
            "SELECT id, title, description FROM posts WHERE title LIKE ? ESCAPE '\\' LIMIT ?",
            (f"%{escaped_q}%", limit),
        )
        for pid, ptitle, pdesc in cur.fetchall():
            if pid not in seen_post_ids:
                seen_post_ids.add(pid)
                suggested = resolve_valid_post_target(pid, ptitle, index, conn)
                posts.append({
                    "post_id": pid,
                    "title": ptitle,
                    "description": pdesc[:120] if pdesc else "",
                    "suggested_target": suggested,
                })

        # 1b. If fewer than limit, match description
        if len(posts) < limit:
            cur.execute(
                "SELECT id, title, description FROM posts WHERE description LIKE ? ESCAPE '\\' LIMIT ?",
                (f"%{escaped_q}%", limit - len(posts)),
            )
            for pid, ptitle, pdesc in cur.fetchall():
                if pid not in seen_post_ids:
                    seen_post_ids.add(pid)
                    suggested = resolve_valid_post_target(pid, ptitle, index, conn)
                    posts.append({
                        "post_id": pid,
                        "title": ptitle,
                        "description": pdesc[:120] if pdesc else "",
                        "suggested_target": suggested,
                    })

        # 2. Semantic vector fallback if provided and limit not reached
        if len(posts) < limit and vector_index is not None and embed_client is not None:
            formatted_q = format_query_for_embedding(clean_q)
            q_vec = embed_client.embed_single(
                formatted_q,
                model=embed_model,
                dimensions=getattr(vector_index, "dim", 512),
            )
            if q_vec:
                vector_hits = vector_index.search_similar(q_vec, top_k=limit * 2)
                for pid, _score in vector_hits:
                    if pid not in seen_post_ids:
                        cur.execute("SELECT id, title, description FROM posts WHERE id = ?", (pid,))
                        p_row = cur.fetchone()
                        if p_row:
                            seen_post_ids.add(pid)
                            suggested = resolve_valid_post_target(p_row[0], p_row[1], index, conn)
                            posts.append({
                                "post_id": p_row[0],
                                "title": p_row[1],
                                "description": p_row[2][:120] if p_row[2] else "",
                                "suggested_target": suggested,
                            })
                            if len(posts) >= limit:
                                break

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
