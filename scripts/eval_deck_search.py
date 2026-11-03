#!/usr/bin/env python3
"""Evaluation and benchmarking script for Irasutoya search recall on Anki decks.

This script connects via AnkiConnect to sample cards from a specified deck,
runs both Legacy and New search strategies against Irasutoya in a non-destructive
read-only dry-run, and produces a comparative recall report.

Usage:
    python3 scripts/eval_deck_search.py --deck "Mining" --sort "FreqSort" --limit 50
"""

import argparse
import html
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Ensure parent directory is on sys.path so we can import client and text_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from addon.batch import safe_extract_note_pos_type
from addon.catalog import default_store
from addon.client import IrasutoyaClient
from addon.feed import IrasutoyaImage, parse_entries
from addon.ranking import (
    infer_pos_type,
    is_decorative_asset,
    is_image_relevant,
    rank_relevant_images,
)
from addon.synonyms import (
    lookup_concept_map,
    lookup_kaishi_concept,
    lookup_llm_concept,
    lookup_sudachi_synonyms,
)
from addon.text_utils import (
    deinflect_japanese,
    extract_furigana_terms,
    generate_search_candidates,
    strip_honorific_affixes,
)


def legacy_clean_query(raw_text: Optional[str]) -> str:
    """Replicate the legacy search query cleaning logic for accurate benchmarking."""
    if not raw_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_text)
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"\[sound:[^\]]+\]", " ", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"[\(（][^()（）]*[\)）]", "", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    for sep in ["/", "・", ";", "；", "、", ","]:
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if parts:
                text = parts[0]
                break
    text = text.strip(" ~〜-ー:：!！?？. 。")
    return text


class _LiveBloggerSearchClient:
    """Standalone live Blogger feed scraper for developer CLI benchmarking."""

    BASE_FEED_URL = "https://www.irasutoya.com/feeds/posts/default"
    DEFAULT_TIMEOUT = 10
    USER_AGENT = "Mozilla/5.0 (compatible; AnkiIrasutoyaBenchmark/1.0)"

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        image_size: str = "s800",
        orderby: Optional[str] = "relevance"
    ) -> List[IrasutoyaImage]:
        """Search Irasutoya directly via live Blogger JSON feed."""
        if not query or not query.strip():
            return []
        fetch_count = min(max_results or 25, 50)
        encoded_q = urllib.parse.quote(query.strip())
        url = f"{self.BASE_FEED_URL}?alt=json&q={encoded_q}&max-results={fetch_count}"
        if orderby:
            url += f"&orderby={urllib.parse.quote(orderby)}"
        max_retries = 3
        raw_json = None
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.DEFAULT_TIMEOUT) as resp:
                    raw_json = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise
            except urllib.error.URLError:
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise

        entries = raw_json.get("feed", {}).get("entry", []) if isinstance(raw_json, dict) else []
        images = parse_entries(entries, image_size=image_size)
        return images[:max_results] if max_results is not None else images

    def search_with_fallbacks(
        self,
        candidates: Sequence[Tuple[str, str, int]],
        max_results: Optional[int] = 5,
        image_size: str = "s800",
        pace_callback: Optional[Callable[[], None]] = None,
        pos_type: Optional[str] = None,
        orderby: Optional[str] = "relevance"
    ) -> Tuple[List[IrasutoyaImage], Optional[Tuple[str, str, int]], Optional[IrasutoyaImage]]:
        """Perform sequential fallback candidate search against live Blogger feed."""
        for cand in candidates:
            term, _reason, _tier = cand
            if pace_callback:
                pace_callback()
            images = self.search(term, max_results=None, image_size=image_size, orderby=orderby)
            if not images:
                continue
            valid_imgs = [
                img for img in images
                if not is_decorative_asset(img, term) and is_image_relevant(img, term)
            ]
            if not valid_imgs:
                continue
            cand_pos = pos_type or infer_pos_type(term, cand_reason=_reason)
            ranked = rank_relevant_images(valid_imgs, term, pos_type=cand_pos)
            if ranked:
                best = ranked[0]
                return (ranked[:max_results] if max_results else ranked), cand, best
        return [], None, None


def anki_invoke(action: str, anki_url: str = "http://localhost:8765", **params: Any) -> Any:
    """Invoke an AnkiConnect action."""
    payload = json.dumps({"action": action, "version": 6, "params": params}).encode("utf-8")
    req = urllib.request.Request(
        anki_url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("error"):
                raise RuntimeError(f"AnkiConnect error: {data['error']}")
            return data.get("result")
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Could not connect to AnkiConnect at {anki_url}. "
            "Please ensure Anki is running with AnkiConnect installed and enabled."
        ) from e


def fetch_deck_notes(
    deck_name: str,
    sort_field: Optional[str] = "FreqSort",
    limit: int = 50,
    anki_url: str = "http://localhost:8765"
) -> List[Dict[str, Any]]:
    """Fetch and sort notes from the specified deck via AnkiConnect."""
    escaped_deck = deck_name.replace('"', '\\"')
    card_ids = anki_invoke("findCards", anki_url=anki_url, query=f'deck:"{escaped_deck}"')
    if not card_ids:
        print(f"Warning: No cards found in deck '{deck_name}'.")
        return []

    cards_info = anki_invoke("cardsInfo", anki_url=anki_url, cards=card_ids)

    # Sort cards by sort_field (numeric if possible)
    if sort_field:
        def sort_key(card: Dict[str, Any]) -> Any:
            val = card.get("fields", {}).get(sort_field, {}).get("value", "")
            try:
                return (0, int(val))
            except (ValueError, TypeError):
                return (1, val)

        cards_info.sort(key=sort_key)

    # Extract unique note IDs preserving order
    seen_notes = set()
    ordered_note_ids = []
    for c in cards_info:
        nid = c.get("note")
        if nid and nid not in seen_notes:
            seen_notes.add(nid)
            ordered_note_ids.append(nid)
        if limit > 0 and len(ordered_note_ids) >= limit:
            break

    chunk_size = 500
    notes_info: List[Dict[str, Any]] = []
    for i in range(0, len(ordered_note_ids), chunk_size):
        chunk = ordered_note_ids[i:i + chunk_size]
        fetched = anki_invoke("notesInfo", anki_url=anki_url, notes=chunk)
        if fetched and isinstance(fetched, list):
            notes_info.extend(fetched)

    note_map = {n.get("noteId"): n for n in notes_info if n.get("noteId")}
    ordered_notes = [note_map[nid] for nid in ordered_note_ids if nid in note_map]
    return ordered_notes


def detect_source_field(fields_dict: Dict[str, Any], requested_field: Optional[str] = None) -> Optional[str]:
    """Auto-detect the best source vocabulary field if not explicitly requested."""
    if requested_field and requested_field in fields_dict:
        return requested_field
    candidates = ["Expression", "Word", "Term", "Vocabulary", "Kanji", "Front"]
    for cand in candidates:
        if cand in fields_dict:
            return cand
    return list(fields_dict.keys())[0] if fields_dict else None


def run_evaluation(
    deck_name: str = "Mining",
    sort_field: str = "FreqSort",
    source_field_name: Optional[str] = None,
    limit: int = 50,
    delay: float = 0.25,
    allow_tier2: bool = True,
    allow_bilingual: bool = True,
    backend: str = "both",
    verbose: bool = False,
    anki_url: str = "http://localhost:8765",
    track_stages: bool = True,
    show_bilingual_hits: bool = True,
) -> None:
    """Run comparative evaluation between legacy, live, and offline catalog search.

    Args:
        deck_name: Target Anki deck name.
        sort_field: Note field used for sorting.
        source_field_name: Optional explicit source field name.
        limit: Number of notes to evaluate (0 for entire deck).
        delay: Polite network request delay in seconds.
        allow_tier2: Whether to evaluate Tier 2 reading/deinflection fallbacks.
        allow_bilingual: Whether to evaluate bilingual gloss bridge fallbacks.
        backend: Search backend to benchmark ('live', 'catalog', or 'both').
        verbose: Whether to log per-candidate evaluation traces.
        anki_url: Local AnkiConnect HTTP URL.
        track_stages: Whether to calculate stage-by-stage incremental recall deltas.
        show_bilingual_hits: Whether to output the audit table of bilingual gloss hits.
    """
    print("=" * 72)
    print("  IRASUTOYA SEARCH RECALL BENCHMARK (NON-DESTRUCTIVE DRY RUN)")
    print("=" * 72)
    print(f"Deck:               {deck_name}")
    print(f"Sort Field:         {sort_field}")
    print(f"Sample Limit:       {'All Notes' if limit <= 0 else limit}")
    print(f"Polite Delay:       {delay}s")
    print(f"Backend Target:     {backend}")
    print(f"Tier 2 Fallback:    {'Enabled' if allow_tier2 else 'Disabled'}")
    print(f"Bilingual Fallback: {'Enabled' if allow_bilingual else 'Disabled'}")
    print(f"Track Stages:       {'Enabled' if track_stages else 'Disabled'}")
    print("-" * 72)

    try:
        notes = fetch_deck_notes(deck_name, sort_field=sort_field, limit=limit, anki_url=anki_url)
    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)

    if not notes:
        print(f"No notes available to evaluate in deck '{deck_name}'.")
        return

    print(f"Retrieved {len(notes)} notes. Starting search benchmark...\n")

    cat_store = default_store()
    if backend in ("catalog", "both"):
        if cat_store.get_index() is None:
            print("\n[ERROR] Offline catalog is missing or unreadable. Build it with scripts/build_catalog.py.\n")
            sys.exit(1)
    else:
        cat_store.warm_async()
    cat_client = IrasutoyaClient(catalog=cat_store)
    live_client = _LiveBloggerSearchClient()

    legacy_hits = 0
    cat_legacy_hits = 0
    t1_hits = 0
    t2_no_bi_hits = 0
    live_hits = 0
    cat_hits = 0
    failed_requests = 0

    reasons_count: Dict[str, int] = {}
    bilingual_matches: List[Dict[str, Any]] = []
    affix_matches: List[Dict[str, Any]] = []

    results = []
    diff_table = []

    for idx, note in enumerate(notes, 1):
        fields = {k: v.get("value", "") for k, v in note.get("fields", {}).items()}
        src_field = detect_source_field(fields, source_field_name)
        raw_source = fields.get(src_field, "") if src_field else ""

        if not raw_source:
            continue

        sort_val = fields.get(sort_field, "")
        pos_type = safe_extract_note_pos_type(fields)

        # --- 1. LEGACY SEARCH SIMULATION ---
        leg_query = legacy_clean_query(raw_source)
        leg_results: List[IrasutoyaImage] = []
        cat_leg_results: List[IrasutoyaImage] = []
        leg_error = False
        if leg_query:
            if backend in ("live", "both"):
                try:
                    leg_results = live_client.search(leg_query, max_results=5, orderby=None)
                except Exception as e:
                    print(f"      [WARNING] Legacy live search request failed for '{leg_query}': {e}")
                    leg_error = True
                    leg_results = []
                if delay > 0:
                    time.sleep(delay)
            if backend in ("catalog", "both"):
                try:
                    cat_leg_results = cat_client.search(leg_query, max_results=5)
                except Exception as e:
                    print(f"      [WARNING] Legacy catalog search failed for '{leg_query}': {e}")
                    leg_error = True
                    cat_leg_results = []

        # --- 2. EVALUATED SEARCH MECHANISMS ---
        candidates = generate_search_candidates(
            raw_source,
            note=fields,
            allow_tier2=allow_tier2,
            allow_deinflection=True,
            store=cat_store,
            bilingual_lookup=(None if allow_bilingual else (lambda note, yomitan_meta=None: [])),
        )

        # Optional Stage 2 & 3 baseline calculation for catalog backend
        eval_error = False
        note_t1_hit = False
        note_t2_hit = False
        if track_stages and backend in ("catalog", "both"):
            try:
                cands_t1 = generate_search_candidates(
                    raw_source,
                    note=fields,
                    allow_tier2=False,
                    store=cat_store,
                    bilingual_lookup=lambda note, yomitan_meta=None: [],
                )
                t1_res, _, _ = cat_client.search_with_fallbacks(cands_t1, max_results=1, pos_type=pos_type)
                note_t1_hit = bool(t1_res)

                cands_t2 = generate_search_candidates(
                    raw_source,
                    note=fields,
                    allow_tier2=allow_tier2,
                    allow_deinflection=True,
                    store=cat_store,
                    bilingual_lookup=lambda note, yomitan_meta=None: [],
                )
                t2_res, _, _ = cat_client.search_with_fallbacks(cands_t2, max_results=1, pos_type=pos_type)
                note_t2_hit = bool(t2_res)
            except Exception as e:
                print(f"      [WARNING] Stage baseline search failed for '{raw_source}': {e}")
                eval_error = True

        live_res, live_cand, live_best = [], None, None
        cat_res, cat_cand, cat_best = [], None, None

        if backend in ("live", "both"):
            try:
                live_res, live_cand, live_best = live_client.search_with_fallbacks(
                    candidates,
                    max_results=5,
                    pace_callback=lambda: time.sleep(delay) if delay > 0 else None,
                    pos_type=pos_type,
                )
            except Exception as e:
                print(f"      [WARNING] Live search request failed for '{raw_source}': {e}")
                eval_error = True

        if backend in ("catalog", "both"):
            try:
                cat_res, cat_cand, cat_best = cat_client.search_with_fallbacks(
                    candidates,
                    max_results=5,
                    pos_type=pos_type,
                )
            except Exception as e:
                print(f"      [WARNING] Catalog search request failed for '{raw_source}': {e}")
                eval_error = True

        # Choose primary comparison results depending on requested backend
        if backend == "catalog":
            new_results, matched_cand, best_img = cat_res, cat_cand, cat_best
        else:
            new_results, matched_cand, best_img = live_res, live_cand, live_best

        if backend == "both" and not eval_error:
            if bool(live_best) != bool(cat_best) or (live_best and cat_best and live_best.title != cat_best.title):
                diff_table.append(
                    (
                        raw_source,
                        live_best.title if live_best else "None",
                        cat_best.title if cat_best else "None",
                    )
                )

        matched_term = matched_cand[0] if matched_cand else ""
        matched_reason = matched_cand[1] if matched_cand else ""
        matched_tier = matched_cand[2] if matched_cand else 0

        if eval_error or leg_error:
            failed_requests += 1
            status = "REQUEST_ERROR"
            status_marker = " ⚠"
            leg_found = False
            new_found = False
        else:
            chosen_leg = cat_leg_results if backend == "catalog" else leg_results
            valid_leg_results = [
                img for img in chosen_leg
                if not is_decorative_asset(img, leg_query) and is_image_relevant(img, leg_query)
            ]
            valid_cat_leg_results = [
                img for img in cat_leg_results
                if not is_decorative_asset(img, leg_query) and is_image_relevant(img, leg_query)
            ]
            leg_found = len(valid_leg_results) > 0
            if len(valid_cat_leg_results) > 0:
                cat_legacy_hits += 1
            if note_t1_hit:
                t1_hits += 1
            if note_t2_hit:
                t2_no_bi_hits += 1

            new_found = len(new_results) > 0

            if leg_found:
                legacy_hits += 1
            if cat_best:
                cat_hits += 1
            if live_best:
                live_hits += 1

            if matched_reason:
                reasons_count[matched_reason] = reasons_count.get(matched_reason, 0) + 1
            else:
                reasons_count["NO_MATCH"] = reasons_count.get("NO_MATCH", 0) + 1

            if matched_reason == "bilingual_gloss" and best_img:
                bilingual_matches.append({
                    "index": idx,
                    "sort_val": sort_val,
                    "expr": raw_source,
                    "target": matched_term,
                    "title": best_img.title,
                })

            # Determine if winning match originated from an affix stem (directly or via concept/Kaishi/Sudachi/LLM fallback)
            is_affix_hit = (matched_reason == "honorific_affix")
            if not is_affix_hit and matched_reason in ("concept_map", "kaishi_concept", "sudachi_synonym", "llm_concept"):
                clean_surface, _ = extract_furigana_terms(raw_source)
                affix_stems = strip_honorific_affixes(clean_surface)
                if affix_stems:
                    deinfs = deinflect_japanese(clean_surface)
                    first_deinf = deinfs[0] if deinfs else ""
                    direct_targets = set()
                    if matched_reason == "llm_concept" and cat_store:
                        direct_targets = set(lookup_llm_concept(clean_surface, store=cat_store))
                        if first_deinf:
                            direct_targets.update(lookup_llm_concept(first_deinf, store=cat_store))
                    elif matched_reason == "sudachi_synonym" and cat_store:
                        direct_targets = set(lookup_sudachi_synonyms(clean_surface, store=cat_store))
                        if first_deinf:
                            direct_targets.update(lookup_sudachi_synonyms(first_deinf, store=cat_store))
                    elif matched_reason == "kaishi_concept" and cat_store:
                        direct_targets = set(lookup_kaishi_concept(clean_surface, store=cat_store))
                        if first_deinf:
                            direct_targets.update(lookup_kaishi_concept(first_deinf, store=cat_store))
                    elif matched_reason == "concept_map":
                        direct_targets = set(lookup_concept_map(clean_surface))
                        if first_deinf:
                            direct_targets.update(lookup_concept_map(first_deinf))
                    if matched_term not in direct_targets:
                        is_affix_hit = True

            if not leg_found and is_affix_hit and best_img:
                affix_matches.append({
                    "index": idx,
                    "sort_val": sort_val,
                    "expr": raw_source,
                    "target": matched_term,
                    "title": best_img.title,
                })

            if leg_found and new_found:
                status = "BOTH_HIT"
                status_marker = " ✓"
            elif not leg_found and new_found:
                if matched_reason == "primary":
                    status = "NEW_HIT_PRIMARY"
                    status_marker = "★P"
                elif is_affix_hit:
                    status = "NEW_HIT_AFFIX"
                    status_marker = "★A"
                elif matched_reason == "bilingual_gloss":
                    status = "NEW_HIT_BILINGUAL"
                    status_marker = "★B"
                elif matched_tier == 1:
                    status = "NEW_HIT_T1"
                    status_marker = "★1"
                else:
                    status = "NEW_HIT_T2"
                    status_marker = "★2"
            elif leg_found and not new_found:
                status = "LEGACY_ONLY"
                status_marker = "  "
            else:
                status = "NOT_FOUND"
                status_marker = " ✗"

        res_entry = {
            "index": idx,
            "sort_val": sort_val,
            "raw_source": raw_source,
            "leg_query": leg_query,
            "leg_found": leg_found,
            "new_found": new_found,
            "matched_term": matched_term,
            "matched_reason": matched_reason,
            "matched_tier": matched_tier,
            "best_title": best_img.title if best_img else "",
            "status": status,
        }
        results.append(res_entry)

        display_desc = f"New ({backend}): " + (f"YES ({matched_reason[:12]})" if new_found else "NO ")
        print(
            f"[{idx:2d}/{len(notes):2d}] {status_marker} "
            f"[{str(sort_val):>5s}] {raw_source[:12]:<12s} -> "
            f"Legacy: {'ERR' if leg_error else ('YES' if leg_found else 'NO '):3s} | "
            f"{display_desc:22s}"
        )
        if verbose:
            if new_found and matched_cand:
                print(
                    f"      ↳ Matched: '{matched_cand[0]}' ({matched_cand[1]}, tier {matched_cand[2]}) -> "
                    f"'{best_img.title if best_img else ''}'"
                )
            if leg_found:
                print(f"      ↳ Legacy matched: '{leg_query}' -> '{valid_leg_results[0].title}'")
            if not leg_found and not new_found:
                tried_terms = [c[0] for c in candidates]
                print(f"      ↳ Tried candidates: {', '.join(repr(t) for t in tried_terms)}")

    # --- SUMMARY REPORT ---
    total = len(results)
    valid_results = [r for r in results if r["status"] != "REQUEST_ERROR"]
    valid_total = len(valid_results)

    print("\n" + "=" * 72)
    print("  EVALUATION SUMMARY REPORT")
    print("=" * 72)
    print(f"Total Notes Evaluated:    {total}")
    if backend in ("live", "both"):
        leg_pct = (legacy_hits / valid_total * 100) if valid_total else 0.0
        live_pct = (live_hits / valid_total * 100) if valid_total else 0.0
        print(f"Legacy Search Recall:     {legacy_hits:3d} / {valid_total} ({leg_pct:.1f}%)")
        print(f"Live Search Recall:       {live_hits:3d} / {valid_total} ({live_pct:.1f}%)")
    if backend in ("catalog", "both"):
        cat_pct = (cat_hits / valid_total * 100) if valid_total else 0.0
        if backend == "catalog":
            leg_pct = (legacy_hits / valid_total * 100) if valid_total else 0.0
            print(f"Legacy Search Recall:     {legacy_hits:3d} / {valid_total} ({leg_pct:.1f}%)")
        print(f"Catalog Search Recall:    {cat_hits:3d} / {valid_total} ({cat_pct:.1f}%)")
    if allow_bilingual:
        bi_hits = sum(1 for r in valid_results if r.get("matched_reason") == "bilingual_gloss")
        bi_pct = (bi_hits / valid_total * 100) if valid_total else 0.0
        print(f"Bilingual Gloss Hits:     {bi_hits:3d} / {valid_total} ({bi_pct:.1f}%)")

    # --- INCREMENTAL STRATEGY PROGRESSION TABLE ---
    if track_stages and backend in ("catalog", "both") and valid_total:
        cat_leg_baseline = cat_legacy_hits if backend == "both" else legacy_hits
        leg_pct = (cat_leg_baseline / valid_total * 100)
        t1_pct = (t1_hits / valid_total * 100)
        t2_pct = (t2_no_bi_hits / valid_total * 100)
        full_pct = (cat_hits / valid_total * 100)

        delta_t1 = t1_pct - leg_pct
        delta_t2 = t2_pct - t1_pct
        delta_bi = full_pct - t2_pct
        net_delta = full_pct - leg_pct

        print("\n" + "=" * 72)
        print("  INCREMENTAL RECALL & STRATEGY PROGRESSION")
        print("=" * 72)
        print(f"{'Stage':<38s} | {'Hits / Total':<12s} | {'Recall %':<8s} | {'Incremental Gain'}")
        print("-" * 72)
        print(f"{'1. Legacy (Naive Direct Query)':<38s} | {cat_leg_baseline:4d} / {valid_total:<5d} | {leg_pct:6.2f}% | —")
        print(f"{'2. Tier 1 (Syntactic & Script Norm)':<38s} | {t1_hits:4d} / {valid_total:<5d} | {t1_pct:6.2f}% | {delta_t1:+5.2f}% ({t1_hits - cat_leg_baseline:+d} cards)")
        print(f"{'3. Tier 2 (Japanese Readings/Deinf)':<38s} | {t2_no_bi_hits:4d} / {valid_total:<5d} | {t2_pct:6.2f}% | {delta_t2:+5.2f}% ({t2_no_bi_hits - t1_hits:+d} cards)")
        print(f"{'4. Bilingual Gloss (Definition Bridge)':<38s} | {cat_hits:4d} / {valid_total:<5d} | {full_pct:6.2f}% | {delta_bi:+5.2f}% ({cat_hits - t2_no_bi_hits:+d} cards)")
        print("-" * 72)
        print(f"Net Improvement: {cat_hits - cat_leg_baseline:+d} cards ({net_delta:+5.2f}% recall gain over Legacy)")

    # --- BILINGUAL GLOSS MATCH AUDIT TABLE ---
    if show_bilingual_hits and bilingual_matches:
        print("\n" + "-" * 72)
        print(f"  BILINGUAL GLOSS NEW HITS (TOTAL: {len(bilingual_matches)})")
        print("-" * 72)
        print(f"{'#':<4s} | {'Freq':<5s} | {'Expression':<12s} | {'Target':<10s} | {'Matched Image Title'}")
        print("-" * 72)
        for m in bilingual_matches:
            print(
                f"{m['index']:<4d} | {str(m['sort_val']):>5s} | {m['expr'][:12]:<12s} | "
                f"{m['target'][:10]:<10s} | {m['title'][:32]}"
            )

    # --- HONORIFIC AFFIX MATCH AUDIT TABLE ---
    if affix_matches:
        print("\n" + "-" * 72)
        print(f"  HONORIFIC AFFIX NEW HITS (TOTAL: {len(affix_matches)})")
        print("-" * 72)
        print(f"{'#':<4s} | {'Freq':<5s} | {'Expression':<12s} | {'Target':<10s} | {'Matched Image Title'}")
        print("-" * 72)
        for m in affix_matches:
            print(
                f"{m['index']:<4d} | {str(m['sort_val']):>5s} | {m['expr'][:12]:<12s} | "
                f"{m['target'][:10]:<10s} | {m['title'][:32]}"
            )

    # --- MATCH REASON DISTRIBUTION ---
    if reasons_count:
        print("\n" + "-" * 72)
        print("  MATCH REASON DISTRIBUTION")
        print("-" * 72)
        sorted_reasons = sorted(reasons_count.items(), key=lambda kv: kv[1], reverse=True)
        for r_name, count in sorted_reasons:
            pct = (count / valid_total * 100) if valid_total else 0.0
            print(f"  • {r_name:<28s}: {count:4d} cards ({pct:5.2f}%)")

    if backend == "both" and diff_table:
        print("\n" + "-" * 72)
        print("  BACKEND SELECTION DIFFERENCES (LIVE vs CATALOG)")
        print("-" * 72)
        print(f"{'Term':<14s} | {'Live Search Image Title':<26s} | {'Catalog Search Image Title'}")
        print("-" * 72)
        for term, l_title, c_title in diff_table[:15]:
            print(f"{term[:14]:<14s} | {l_title[:26]:<26s} | {c_title[:26]}")
        if len(diff_table) > 15:
            print(f"... and {len(diff_table) - 15} more differences.")
    print("=" * 72)
    if failed_requests:
        print(f"\n[ERROR] {failed_requests} note(s) had failed search requests and were excluded from recall.\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare Irasutoya search recall against an Anki deck via AnkiConnect."
    )
    parser.add_argument("--deck", default="Mining", help="Anki deck name to evaluate (default: 'Mining')")
    parser.add_argument("--sort", default="FreqSort", help="Field to sort cards by (default: 'FreqSort')")
    parser.add_argument("--source-field", default=None, help="Source field to query (default: auto-detect Expression/Word)")
    parser.add_argument("--limit", type=int, default=50, help="Number of notes to evaluate (default: 50, use 0 for all)")
    parser.add_argument("--all", action="store_true", help="Evaluate all notes in the deck (sets limit=0)")
    parser.add_argument("--delay", type=float, default=None, help="Polite delay between HTTP requests in seconds (default: 0.0 for catalog, 0.25 for live)")
    parser.add_argument("--backend", choices=["live", "catalog", "both"], default="both", help="Search backend to benchmark (default: both)")
    parser.add_argument("--disable-tier2", action="store_true", help="Disable Tier 2 reading fallbacks")
    parser.add_argument("--disable-bilingual", action="store_true", help="Disable bilingual definition gloss fallbacks")
    parser.add_argument("--no-stages", action="store_true", help="Disable stage-by-stage incremental progression tracking")
    parser.add_argument("--no-bilingual-hits", action="store_true", help="Omit the audit table of bilingual gloss hits")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed candidate resolution for each note")
    parser.add_argument("--anki-url", default="http://localhost:8765", help="AnkiConnect URL (default: http://localhost:8765)")

    args = parser.parse_args()

    limit = 0 if args.all else args.limit
    delay = args.delay if args.delay is not None else (0.0 if args.backend == "catalog" else 0.25)

    run_evaluation(
        deck_name=args.deck,
        sort_field=args.sort,
        source_field_name=args.source_field,
        limit=limit,
        delay=delay,
        allow_tier2=not args.disable_tier2,
        allow_bilingual=not args.disable_bilingual,
        backend=args.backend,
        verbose=args.verbose,
        anki_url=args.anki_url,
        track_stages=not args.no_stages,
        show_bilingual_hits=not args.no_bilingual_hits,
    )


if __name__ == "__main__":
    main()
