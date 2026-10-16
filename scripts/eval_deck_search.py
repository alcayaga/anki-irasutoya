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
from typing import Any, Dict, List, Optional, Tuple

# Ensure parent directory is on sys.path so we can import client and text_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import IrasutoyaClient, IrasutoyaImage, score_candidate_image, select_best_image
from text_utils import (
    clean_search_query,
    generate_search_candidates,
    NOTE_READING_FIELDS,
    NOTE_FURIGANA_FIELDS,
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
    return notes_info


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
    verbose: bool = False,
    anki_url: str = "http://localhost:8765"
) -> None:
    """Run comparative evaluation between legacy and new search mechanisms."""
    print("=" * 72)
    print("  IRASUTOYA SEARCH RECALL BENCHMARK (NON-DESTRUCTIVE DRY RUN)")
    print("=" * 72)
    print(f"Deck:          {deck_name}")
    print(f"Sort Field:    {sort_field}")
    print(f"Sample Limit:  {limit}")
    print(f"Polite Delay:  {delay}s")
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

    client = IrasutoyaClient()

    legacy_hits = 0
    new_hits = 0
    both_hits = 0
    tier1_improvements = 0
    tier2_improvements = 0
    both_failed = 0

    results = []

    for idx, note in enumerate(notes, 1):
        fields = {k: v.get("value", "") for k, v in note.get("fields", {}).items()}
        src_field = detect_source_field(fields, source_field_name)
        raw_source = fields.get(src_field, "") if src_field else ""

        if not raw_source:
            continue

        sort_val = fields.get(sort_field, "")

        # --- 1. LEGACY SEARCH SIMULATION ---
        leg_query = legacy_clean_query(raw_source)
        leg_results: List[IrasutoyaImage] = []
        if leg_query:
            try:
                # Legacy used default orderby (no orderby="relevance", so orderby=None/updated)
                leg_results = client.search(leg_query, max_results=5, orderby=None)
            except Exception as e:
                leg_results = []
        time.sleep(delay)

        # --- 2. NEW SEARCH MECHANISM ---
        candidates = generate_search_candidates(
            raw_source, note=fields, allow_tier2=allow_tier2, allow_deinflection=False
        )
        new_results: List[IrasutoyaImage] = []
        matched_cand = None
        best_img = None
        try:
            new_results, matched_cand, best_img = client.search_with_fallbacks(
                candidates,
                max_results=5,
                orderby="relevance",
                pace_callback=lambda: time.sleep(delay) if delay > 0 else None,
            )
        except Exception as e:
            new_results, matched_cand, best_img = [], None, None

        leg_found = len(leg_results) > 0
        new_found = len(new_results) > 0

        if leg_found:
            legacy_hits += 1
        if new_found:
            new_hits += 1

        status = ""
        matched_term = matched_cand[0] if matched_cand else ""
        matched_reason = matched_cand[1] if matched_cand else ""
        matched_tier = matched_cand[2] if matched_cand else 0

        if leg_found and new_found:
            both_hits += 1
            status = "BOTH_HIT"
        elif not leg_found and new_found:
            if matched_tier == 1:
                tier1_improvements += 1
                status = "NEW_HIT_T1"
            else:
                tier2_improvements += 1
                status = "NEW_HIT_T2"
        elif leg_found and not new_found:
            status = "LEGACY_ONLY"
        else:
            both_failed += 1
            status = "NOT_FOUND"

        res_entry = {
            "index": idx,
            "sort_val": sort_val,
            "raw_source": raw_source,
            "leg_query": leg_query,
            "leg_found": leg_found,
            "leg_title": leg_results[0].title if leg_results else "",
            "new_found": new_found,
            "matched_term": matched_term,
            "matched_reason": matched_reason,
            "matched_tier": matched_tier,
            "best_title": best_img.title if best_img else "",
            "status": status,
        }
        results.append(res_entry)

        status_marker = "  "
        if status == "NEW_HIT_T1":
            status_marker = "★1"
        elif status == "NEW_HIT_T2":
            status_marker = "★2"
        elif status == "BOTH_HIT":
            status_marker = " ✓"
        elif status == "NOT_FOUND":
            status_marker = " ✗"

        print(
            f"[{idx:2d}/{len(notes):2d}] {status_marker} "
            f"[{sort_val:>5s}] {raw_source[:12]:<12s} -> "
            f"Legacy: {'YES' if leg_found else 'NO ':3s} | "
            f"New: {'YES (' + matched_reason[:15] + ')' if new_found else 'NO':20s}"
        )

        if verbose and new_found and not leg_found:
            print(f"      Legacy query: '{leg_query}' -> 0 results")
            print(f"      Matched cand: '{matched_term}' ({matched_reason}, Tier {matched_tier}) -> '{best_img.title if best_img else ''}'")

    # --- SUMMARY REPORT ---
    total = len(results)
    leg_pct = (legacy_hits / total * 100) if total else 0.0
    new_pct = (new_hits / total * 100) if total else 0.0
    abs_gain = new_pct - leg_pct
    rel_gain = ((new_hits - legacy_hits) / legacy_hits * 100) if legacy_hits else 0.0

    print("\n" + "=" * 72)
    print("  EVALUATION SUMMARY REPORT")
    print("=" * 72)
    print(f"Total Notes Evaluated:    {total}")
    print(f"Legacy Search Recall:     {legacy_hits:3d} / {total} ({leg_pct:.1f}%)")
    print(f"New Search Recall:        {new_hits:3d} / {total} ({new_pct:.1f}%)")
    print(f"Absolute Recall Gain:     {abs_gain:+.1f}%")
    print(f"Relative Improvement:     {rel_gain:+.1f}%")
    print("-" * 72)
    print("Breakdown of New Matches:")
    print(f"  • Both Succeeded (Baseline):         {both_hits:3d}")
    print(f"  • Tier 1 Syntactic Improvements:     {tier1_improvements:3d}")
    print(f"  • Tier 2 Reading / Semantic Fallbacks: {tier2_improvements:3d} (Tagged with 'irasutoya::fallback_reading')")
    print(f"  • Unmatched (No Illustration Found):  {both_failed:3d}")
    print("-" * 72)

    # Exemplars of improvement
    wins = [r for r in results if r["status"] in ("NEW_HIT_T1", "NEW_HIT_T2")]
    if wins:
        print("\nExemplars Recovered by New Search Strategy:")
        print(f"{'Freq':<6s} | {'Term':<12s} | {'Tier':<6s} | {'Strategy Reason':<24s} | {'Matched Image Title'}")
        print("-" * 72)
        for w in wins[:10]:
            print(
                f"{str(w['sort_val']):<6s} | "
                f"{w['raw_source'][:12]:<12s} | "
                f"Tier {w['matched_tier']:<1d} | "
                f"{w['matched_reason']:<24s} | "
                f"{w['best_title'][:30]}"
            )

    print("\nBenchmark completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare Irasutoya search recall against an Anki deck via AnkiConnect."
    )
    parser.add_argument("--deck", default="Mining", help="Anki deck name to evaluate (default: 'Mining')")
    parser.add_argument("--sort", default="FreqSort", help="Field to sort cards by (default: 'FreqSort')")
    parser.add_argument("--source-field", default=None, help="Source field to query (default: auto-detect Expression/Word)")
    parser.add_argument("--limit", type=int, default=50, help="Number of notes to evaluate (default: 50)")
    parser.add_argument("--delay", type=float, default=0.25, help="Polite delay between HTTP requests in seconds (default: 0.25)")
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
        verbose=args.verbose,
        anki_url=args.anki_url,
    )


if __name__ == "__main__":
    main()
