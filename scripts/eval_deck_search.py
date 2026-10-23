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
import urllib.request
from typing import Any, Dict, List, Optional

# Ensure parent directory is on sys.path so we can import client and text_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from addon.catalog import default_store
from addon.client import IrasutoyaClient
from addon.feed import IrasutoyaImage
from addon.ranking import is_decorative_asset, is_image_relevant
from addon.text_utils import (
    BATCH_MAX_CANDIDATES,
    generate_search_candidates,
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


def anki_invoke(action: str, anki_url: str = "http://localhost:8765", **params: Any) -> Any:
    """Invoke an AnkiConnect action."""
    payload = json.dumps({"action": action, "version": 6, "params": params}).encode("utf-8")
    req = urllib.request.Request(
        anki_url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
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
        if len(ordered_note_ids) >= limit:
            break

    notes_info = anki_invoke("notesInfo", anki_url=anki_url, notes=ordered_note_ids)
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
    backend: str = "both",
    verbose: bool = False,
    anki_url: str = "http://localhost:8765"
) -> None:
    """Run comparative evaluation between legacy, live, and offline catalog search."""
    print("=" * 72)
    print("  IRASUTOYA SEARCH RECALL BENCHMARK (NON-DESTRUCTIVE DRY RUN)")
    print("=" * 72)
    print(f"Deck:            {deck_name}")
    print(f"Sort Field:      {sort_field}")
    print(f"Sample Limit:    {limit}")
    print(f"Polite Delay:    {delay}s")
    print(f"Backend Target:  {backend}")
    print(f"Tier 2 Fallback: {'Enabled' if allow_tier2 else 'Disabled'}")
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
    cat_client = IrasutoyaClient(catalog=cat_store, backend="catalog")
    live_client = IrasutoyaClient(backend="live")

    legacy_hits = 0
    live_hits = 0
    cat_hits = 0
    failed_requests = 0

    results = []
    diff_table = []

    for idx, note in enumerate(notes, 1):
        fields = {k: v.get("value", "") for k, v in note.get("fields", {}).items()}
        src_field = detect_source_field(fields, source_field_name)
        raw_source = fields.get(src_field, "") if src_field else ""

        if not raw_source:
            continue

        sort_val = fields.get(sort_field, "")

        # --- 1. LEGACY SEARCH SIMULATION (if live or both) ---
        leg_query = legacy_clean_query(raw_source)
        leg_results: List[IrasutoyaImage] = []
        leg_error = False
        if backend in ("live", "both") and leg_query:
            try:
                leg_results = live_client.search(leg_query, max_results=5, orderby=None)
            except Exception as e:
                print(f"      [WARNING] Legacy search request failed for '{leg_query}': {e}")
                leg_error = True
                leg_results = []
            if delay > 0:
                time.sleep(delay)

        # --- 2. EVALUATED SEARCH MECHANISMS ---
        candidates = generate_search_candidates(
            raw_source, note=fields, allow_tier2=allow_tier2, allow_deinflection=True
        )
        batch_candidates = candidates[:BATCH_MAX_CANDIDATES]

        live_res, live_cand, live_best = [], None, None
        cat_res, cat_cand, cat_best = [], None, None
        eval_error = False

        if backend in ("live", "both"):
            try:
                live_res, live_cand, live_best = live_client.search_with_fallbacks(
                    batch_candidates,
                    max_results=5,
                    orderby="relevance",
                    pace_callback=lambda: time.sleep(delay) if delay > 0 else None,
                )
            except Exception as e:
                print(f"      [WARNING] Live search request failed for '{raw_source}': {e}")
                eval_error = True

        if backend in ("catalog", "both"):
            try:
                cat_res, cat_cand, cat_best = cat_client.search_with_fallbacks(
                    batch_candidates,
                    max_results=5,
                    orderby="relevance",
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
            valid_leg_results = [
                img for img in leg_results
                if not is_decorative_asset(img, leg_query) and is_image_relevant(img, leg_query)
            ]
            leg_found = len(valid_leg_results) > 0
            new_found = len(new_results) > 0

            if leg_found:
                legacy_hits += 1
            if cat_best:
                cat_hits += 1
            if live_best:
                live_hits += 1

            if leg_found and new_found:
                status = "BOTH_HIT"
                status_marker = " ✓"
            elif not leg_found and new_found:
                if matched_reason == "primary":
                    status = "NEW_HIT_PRIMARY"
                    status_marker = "★P"
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
                tried_terms = [c[0] for c in batch_candidates]
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
        print(f"Catalog Search Recall:    {cat_hits:3d} / {valid_total} ({cat_pct:.1f}%)")

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
    parser.add_argument("--limit", type=int, default=50, help="Number of notes to evaluate (default: 50)")
    parser.add_argument("--delay", type=float, default=0.25, help="Polite delay between HTTP requests in seconds (default: 0.25)")
    parser.add_argument("--backend", choices=["live", "catalog", "both"], default="both", help="Search backend to benchmark (default: both)")
    parser.add_argument("--disable-tier2", action="store_true", help="Disable Tier 2 reading fallbacks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed candidate resolution for each note")
    parser.add_argument("--anki-url", default="http://localhost:8765", help="AnkiConnect URL (default: http://localhost:8765)")

    args = parser.parse_args()

    run_evaluation(
        deck_name=args.deck,
        sort_field=args.sort,
        source_field_name=args.source_field,
        limit=args.limit,
        delay=args.delay,
        allow_tier2=not args.disable_tier2,
        backend=args.backend,
        verbose=args.verbose,
        anki_url=args.anki_url,
    )


if __name__ == "__main__":
    main()
