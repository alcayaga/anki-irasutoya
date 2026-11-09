"""Batch processor for adding Irasutoya illustrations to Anki cards in the background."""
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple

from .catalog import default_store
from .client import IrasutoyaClient
from .config import (
    CONFIG_KEY_FULL_DELAY,
    CONFIG_KEY_IMAGE_SIZE,
    get_config,
)
from .ranking import (
    has_title_illustration_match,
    infer_note_pos_type,
    infer_pos_type,
    is_decorative_asset,
    is_image_relevant,
    select_best_image,
)
from .synonyms import is_llm_rejected
from .text_utils import clean_search_query, generate_search_candidates
from .yomitan_parser import extract_yomitan_metadata_from_note

logger = logging.getLogger(__name__)

TAG_NON_VISUAL = "irasutoya::non_visual"
TAG_APPROXIMATE = "irasutoya::approximate"
TAG_REVIEW = "irasutoya::review"
TAG_NO_MATCH = "irasutoya::no_match"
FALLBACK_AUDIT_TAG = TAG_APPROXIMATE
STATUS_TAGS = frozenset({
    TAG_NON_VISUAL,
    TAG_APPROXIMATE,
    TAG_REVIEW,
    TAG_NO_MATCH,
})
ALL_IRASUTOYA_TAGS = STATUS_TAGS | {"irasutoya::fallback_reading"}


class PendingBatchWrite(NamedTuple):
    """Pending batch write record passed to collection commit task."""
    note: Any
    filename: Optional[str]
    img_bytes: Optional[bytes]
    tag: Optional[str]


def apply_status_tag(note: Any, tag: Optional[str]) -> bool:
    """Apply a taxonomy tag to note and remove all other irasutoya status tags.

    Args:
        note: Anki note instance or dict-like mock.
        tag: Status tag to apply, or None to remove all status tags.

    Returns:
        True if tags were modified, False otherwise.
    """
    changed = False
    for stale_tag in ALL_IRASUTOYA_TAGS:
        if stale_tag == tag:
            continue
        if hasattr(note, "remove_tag"):
            if hasattr(note, "has_tag") and not note.has_tag(stale_tag):
                continue
            note.remove_tag(stale_tag)
            changed = True
        elif hasattr(note, "tags") and isinstance(note.tags, list):
            if stale_tag in note.tags:
                note.tags.remove(stale_tag)
                changed = True

    if tag is not None:
        if hasattr(note, "add_tag"):
            if hasattr(note, "has_tag") and note.has_tag(tag):
                pass
            else:
                note.add_tag(tag)
                changed = True
        elif hasattr(note, "tags") and isinstance(note.tags, list):
            if tag not in note.tags:
                note.tags.append(tag)
                changed = True

    return changed


def classify_batch_outcome(
    images: Sequence[Any],
    matched_cand: Optional[Tuple[str, str, int]],
) -> str:
    """Classify the search result into a batch outcome status.

    Returns:
        - 'no_match': no images found.
        - 'review': description-only match or Tier 2 phonetic reading match (held back from auto-fill).
        - 'fill_approximate': Tier 2 match (exemplar, deinflection, compound title).
        - 'fill_direct': Tier 1 direct or curated concept match.
    """
    if not images or not matched_cand:
        return "no_match"

    term, reason, tier = matched_cand

    # Phonetic readings in Tier 2 have high homophone risk (e.g. 顧問 -> こもん -> コモンマーモセット)
    # Tier 1 yomitan_uk* reasons are safe orthographic variants and do NOT produce 'review'.
    if tier == 2 and reason.startswith(("note_reading", "note_furigana", "same_field_reading")):
        return "review"

    # Description-only matches on out-of-index words
    if not has_title_illustration_match(images, term):
        return "review"

    if tier == 2:
        return "fill_approximate"

    return "fill_direct"


def safe_extract_note_pos_type(note: Any) -> Optional[str]:
    """Safely extract Yomitan metadata and infer POS type, returning None on failure."""
    try:
        yomitan_meta = extract_yomitan_metadata_from_note(note)
        return infer_note_pos_type(yomitan_meta)
    except Exception as exc:
        logger.debug("Failed to extract Yomitan metadata/POS from note: %s", exc)
        return None


@dataclass
class BatchStats:
    """Statistics tracking the outcome of a batch fill operation."""
    total: int = 0
    added: int = 0
    direct_matches: int = 0
    approximate: int = 0
    review: int = 0
    non_visual: int = 0
    no_match: int = 0
    skipped: int = 0
    errors: int = 0
    canceled: bool = False
    details: List[str] = field(default_factory=list)

    def __init__(
        self,
        total: int = 0,
        added: int = 0,
        direct_matches: int = 0,
        approximate: int = 0,
        review: int = 0,
        non_visual: int = 0,
        no_match: int = 0,
        skipped: int = 0,
        errors: int = 0,
        canceled: bool = False,
        details: Optional[List[str]] = None,
        *,
        fallback_matches: Optional[int] = None,
        not_found: Optional[int] = None,
    ) -> None:
        """Initialize batch operation statistics with backward-compatible aliases."""
        self.total = total
        self.added = added
        self.direct_matches = direct_matches
        self.approximate = fallback_matches if fallback_matches is not None else approximate
        self.review = review
        self.non_visual = non_visual
        self.no_match = not_found if not_found is not None else no_match
        self.skipped = skipped
        self.errors = errors
        self.canceled = canceled
        self.details = details if details is not None else []

    @property
    def fallback_matches(self) -> int:
        """Backward-compatible alias for approximate matches."""
        return self.approximate

    @fallback_matches.setter
    def fallback_matches(self, val: int) -> None:
        self.approximate = val

    @property
    def not_found(self) -> int:
        """Backward-compatible alias for no_match count."""
        return self.no_match

    @not_found.setter
    def not_found(self, val: int) -> None:
        self.no_match = val

    @property
    def summary_message(self) -> str:
        """Formatted human-readable summary of the batch operation."""
        parts = []
        if self.approximate > 0:
            parts.append(f"Added: {self.added} ({self.direct_matches} direct, {self.approximate} approximate)")
        else:
            parts.append(f"Added: {self.added}")

        if self.review > 0:
            parts.append(f"Review: {self.review}")
        if self.non_visual > 0:
            parts.append(f"Non-visual: {self.non_visual}")
        if self.no_match > 0:
            parts.append(f"No match: {self.no_match}")
        if self.skipped > 0:
            parts.append(f"Skipped: {self.skipped}")
        if self.errors > 0:
            parts.append(f"Errors: {self.errors}")

        msg = ", ".join(parts)
        if self.details:
            details_str = "; ".join(self.details)
            msg += f" ({details_str})"
        if self.canceled:
            if self.added > 0:
                return f"Canceled. Partial results committed: {msg}"
            return f"Canceled. {msg}"
        return f"Finished! {msg}"


def make_safe_media_filename(query: str, title: str) -> str:
    """Generate a sanitized, readable media filename for Anki."""
    combined = f"irasutoya_{query}_{title}".strip()
    safe = re.sub(r'[\\/*?:"<>| \t\n\r]+', '_', combined)
    safe = re.sub(r'_+', '_', safe).strip('_')
    if safe.lower().endswith(".png"):
        safe = safe[:-4]
    return safe[:60] + ".png"


def resolve_surface_term(
    candidates: Sequence[Tuple[str, str, int]],
    raw_source: Optional[str] = None,
) -> str:
    """Return the surface headword: the primary candidate term, or cleaned raw_source.

    Args:
        candidates: Sequence of generated search candidates `(term, reason, tier)`.
        raw_source: Optional uncleaned raw note source string.

    Returns:
        The primary candidate term if present, else cleaned `raw_source` or empty string.
    """
    primary = next((c for c in candidates if c[1] == "primary"), None)
    if primary:
        return primary[0]
    if raw_source:
        return clean_search_query(raw_source)
    return candidates[0][0] if candidates else ""


def process_note_record(
    note: any,
    source_field: str,
    target_field: str,
    overwrite_mode: str,
    client: IrasutoyaClient,
    write_media_func: Callable[[str, bytes], str],
    image_size: str = "s800",
    allow_tier2: bool = True
) -> Tuple[str, Optional[any]]:
    """Process a single note: check condition, search Irasutoya with fallbacks, download and set image.
    
    Returns:
        (status, modified_note_or_none)
        Status can be:
        - "added": image downloaded and assigned
        - "review": description-only or Tier 2 phonetic match held back from auto-fill (tagged irasutoya::review)
        - "non_visual": term rejected by LLM concept compiler (tagged irasutoya::non_visual)
        - "no_match": no matching images on Irasutoya (tagged irasutoya::no_match)
        - "skipped_missing_field": field not in note
        - "skipped_not_empty": target field already has content and overwrite is 'skip'
        - "skipped_empty_query": source field contains no searchable text
        - "error": download or unexpected failure
    """
    if source_field not in note or target_field not in note:
        return "skipped_missing_field", None

    existing_target = note[target_field].strip()
    if existing_target and overwrite_mode == "skip":
        return "skipped_not_empty", None

    raw_source = note[source_field]
    store = getattr(client, "catalog", None)
    candidates = generate_search_candidates(
        raw_source,
        note=note,
        allow_tier2=allow_tier2,
        allow_deinflection=True,
        store=store,
    )
    if not candidates:
        return "skipped_empty_query", None

    surface_term = resolve_surface_term(candidates, raw_source)
    if is_llm_rejected(surface_term, store=store):
        apply_status_tag(note, TAG_NON_VISUAL)
        return "non_visual", note

    try:
        pos_type = safe_extract_note_pos_type(note)
        if hasattr(client, "search_with_fallbacks"):
            images, matched_cand, best_img = client.search_with_fallbacks(
                candidates, max_results=12, image_size=image_size, pos_type=pos_type
            )
        else:
            matched_cand = None
            best_img = None
            images = []
            for cand in candidates:
                imgs = client.search(cand[0], max_results=3, image_size=image_size)
                valid_imgs = [
                    img for img in imgs
                    if not is_decorative_asset(img, cand[0]) and is_image_relevant(img, cand[0])
                ]
                if valid_imgs:
                    matched_cand = cand
                    cand_pos = pos_type or infer_pos_type(cand[0], cand_reason=cand[1])
                    best_img = select_best_image(valid_imgs, cand[0], pos_type=cand_pos) or valid_imgs[0]
                    images = valid_imgs
                    break

        outcome = classify_batch_outcome(images, matched_cand)
        if outcome == "no_match":
            apply_status_tag(note, TAG_NO_MATCH)
            return "no_match", note

        if outcome == "review":
            apply_status_tag(note, TAG_REVIEW)
            return "review", note

        assert best_img is not None
        assert matched_cand is not None
        img_bytes = client.download_image(best_img.image_url, image=best_img)
        if not img_bytes:
            return "error", None

        filename = make_safe_media_filename(matched_cand[0], best_img.title)
        stored_filename = write_media_func(filename, img_bytes)
        encoded_src = urllib.parse.quote(stored_filename)

        note[target_field] = f'<img src="{encoded_src}">'

        target_tag = TAG_APPROXIMATE if outcome == "fill_approximate" else None
        apply_status_tag(note, target_tag)

        return "added", note
    except Exception as e:
        logger.exception("Error processing note with query '%s': %s", surface_term, e)
        return "error", None



def run_batch_fill(
    parent_window: any,
    note_ids: Sequence[int],
    source_field: str,
    target_field: str,
    overwrite_mode: str,
    on_finish: Optional[Callable[[BatchStats], None]] = None
) -> None:
    """Execute batch processing in background without holding collection lock during downloads."""
    import aqt.operations.note
    from aqt import mw
    from aqt.operations import QueryOp
    from aqt.utils import tooltip

    unique_nids = list(dict.fromkeys(note_ids))
    total = len(unique_nids)
    if total == 0:
        return

    config = get_config()
    try:
        delay_ms = float(config.get(CONFIG_KEY_FULL_DELAY, 300))
    except (TypeError, ValueError):
        delay_ms = 300.0
    delay_sec = max(0.05, delay_ms / 1000.0)
    image_size = config.get(CONFIG_KEY_IMAGE_SIZE, "s800")
    store = default_store()
    store.warm_async()
    client = IrasutoyaClient(catalog=store)

    def read_notes_task(col: any) -> Tuple[List[Tuple[any, List[Tuple[str, str, int]], bool]], int, int]:
        """Read note terms in a short collection-backed task and generate candidate search queries."""
        items = []
        load_errors = 0
        missing_field_skips = 0
        for nid in unique_nids:
            try:
                note = col.get_note(nid)
                if target_field not in note or source_field not in note:
                    missing_field_skips += 1
                    continue
                raw_val = note[source_field]
                candidates = generate_search_candidates(
                    raw_val,
                    note=note,
                    allow_tier2=True,
                    allow_deinflection=True,
                    store=store,
                )
                has_target = bool(note[target_field].strip())
                items.append((note, candidates, has_target))
            except Exception as e:
                logger.exception("Failed to load note %s: %s", nid, e)
                load_errors += 1
        return items, load_errors, missing_field_skips

    def fetch_images_worker(
        data: Tuple[List[Tuple[any, List[Tuple[str, str, int]], bool]], int, int]
    ) -> Tuple[BatchStats, List[PendingBatchWrite]]:
        """Download images and pace requests outside the collection lock using fallback search."""
        items, load_errors, missing_field_skips = data
        if store.get_index() is None:
            logger.error("Offline catalog unavailable; aborting batch fill.")
            catalog_errors = 0
            catalog_skips = missing_field_skips
            for _, candidates, has_target in items:
                if (has_target and overwrite_mode == "skip") or not candidates:
                    catalog_skips += 1
                else:
                    catalog_errors += 1
            return BatchStats(
                total=total,
                errors=catalog_errors + load_errors,
                skipped=catalog_skips,
                details=["Offline catalog unavailable or failed to load"],
            ), []

        stats = BatchStats(total=total, errors=load_errors, skipped=missing_field_skips)
        pending_writes: List[PendingBatchWrite] = []
        last_request_time = 0.0

        def pace_request() -> None:
            """Ensure polite delay between consecutive HTTP network requests."""
            nonlocal last_request_time
            if last_request_time > 0:
                elapsed = time.time() - last_request_time
                if elapsed < delay_sec:
                    time.sleep(delay_sec - elapsed)
            last_request_time = time.time()

        for idx, (note, candidates, has_target) in enumerate(items):
            # Dismissing progress window or pressing Esc triggers closeEvent, setting wantCancel
            if hasattr(mw.progress, "want_cancel") and mw.progress.want_cancel():
                stats.canceled = True
                stats.skipped += len(items) - idx
                break

            current_count = idx + 1
            raw_src = note[source_field] if source_field in note else None
            surface_term = resolve_surface_term(candidates, raw_src)
            primary_label = surface_term or "..."
            # Update progress UI on main thread binding loop variables
            mw.taskman.run_on_main(
                lambda q=primary_label, c=current_count: mw.progress.update(
                    label=f"Fetching Irasutoya: {q} ({c}/{total})...",
                    value=c,
                    max=total
                )
            )

            if has_target and overwrite_mode == "skip":
                stats.skipped += 1
                continue

            if not candidates:
                stats.skipped += 1
                continue

            if is_llm_rejected(surface_term, store=store):
                stats.non_visual += 1
                pending_writes.append(
                    PendingBatchWrite(note=note, filename=None, img_bytes=None, tag=TAG_NON_VISUAL)
                )
                continue

            try:
                pos_type = safe_extract_note_pos_type(note)
                images, matched_cand, best_img = client.search_with_fallbacks(
                    candidates,
                    max_results=12,
                    image_size=image_size,
                    pos_type=pos_type,
                )
                outcome = classify_batch_outcome(images, matched_cand)
                if outcome == "no_match":
                    stats.no_match += 1
                    pending_writes.append(
                        PendingBatchWrite(note=note, filename=None, img_bytes=None, tag=TAG_NO_MATCH)
                    )
                    continue

                if outcome == "review":
                    stats.review += 1
                    pending_writes.append(
                        PendingBatchWrite(note=note, filename=None, img_bytes=None, tag=TAG_REVIEW)
                    )
                    continue

                assert best_img is not None
                assert matched_cand is not None
                pace_request()
                img_bytes = client.download_image(best_img.image_url, image=best_img)
                if not img_bytes:
                    stats.errors += 1
                    continue

                target_tag = TAG_APPROXIMATE if outcome == "fill_approximate" else None
                filename = make_safe_media_filename(matched_cand[0], best_img.title)
                pending_writes.append(
                    PendingBatchWrite(note=note, filename=filename, img_bytes=img_bytes, tag=target_tag)
                )
            except Exception as e:
                logger.exception("Error during batch download for note %s: %s", getattr(note, "id", "?"), e)
                stats.errors += 1

        return stats, pending_writes

    def commit_task(
        col: any,
        pending_writes: List[PendingBatchWrite],
        stats: BatchStats,
        processed_tracker: List[int]
    ) -> List[Tuple[any, Optional[str]]]:
        """Write media files (if image present) and apply taxonomy tags within collection op."""
        notes_to_save = []
        saved_records = []
        for pw in pending_writes:
            try:
                fresh_note = col.get_note(pw.note.id)
            except Exception as e:
                logger.exception("Failed to load note %s at commit time: %s", getattr(pw.note, "id", "?"), e)
                stats.errors += 1
                processed_tracker[0] += 1
                continue
            if target_field not in fresh_note:
                stats.skipped += 1
                processed_tracker[0] += 1
                continue
            if overwrite_mode == "skip" and fresh_note[target_field].strip():
                stats.skipped += 1
                processed_tracker[0] += 1
                continue
            try:
                if pw.filename and pw.img_bytes:
                    stored_filename = col.media.write_data(pw.filename, pw.img_bytes)
                    encoded_src = urllib.parse.quote(stored_filename)
                    fresh_note[target_field] = f'<img src="{encoded_src}">'

                apply_status_tag(fresh_note, pw.tag)
                notes_to_save.append(fresh_note)
                saved_records.append((fresh_note, pw.tag))
            except Exception as e:
                logger.exception("Failed to process commit for note %s: %s", getattr(pw.note, "id", "?"), e)
                stats.errors += 1
            processed_tracker[0] += 1
        return saved_records


    def on_fetch_done(result: Tuple[BatchStats, List[PendingBatchWrite]]) -> None:
        """Handle completion of network downloads and trigger media/note commit."""
        stats, pending_writes = result
        if not pending_writes:
            tooltip(stats.summary_message, parent=parent_window)
            if on_finish:
                on_finish(stats)
            return

        processed_tracker = [0]

        def on_saved(saved_records: List[Tuple[any, Optional[str]]]) -> None:
            """Update notes with full undo support and calculate final added count."""
            saved_notes = [r[0] for r in saved_records]
            saved_tags = [r[1] for r in saved_records]
            if not saved_notes:
                show_summary_and_callback(stats)
                return
            op = aqt.operations.note.update_notes(parent=parent_window, notes=saved_notes)

            def _ok(_: any) -> None:
                """Handle successful notes update."""
                stats.added = sum(1 for tag in saved_tags if tag in (None, TAG_APPROXIMATE))
                stats.approximate = sum(1 for tag in saved_tags if tag == TAG_APPROXIMATE)
                stats.direct_matches = stats.added - stats.approximate
                show_summary_and_callback(stats)

            def _fail(err: Exception) -> None:
                """Handle notes update failure."""
                logger.exception("Failed to update notes batch: %s", err)
                stats.errors += len(saved_notes)
                show_summary_and_callback(stats)

            op.success(_ok)
            op.failure(_fail)
            op.run_in_background()

        def on_commit_failure(err: Exception) -> None:
            """Handle failure of commit QueryOp, accounting for unprocessed writes."""
            logger.exception("Batch commit operation failed: %s", err)
            unprocessed = len(pending_writes) - processed_tracker[0]
            if unprocessed > 0:
                stats.errors += unprocessed
            show_summary_and_callback(stats)

        commit_op = QueryOp(
            parent=parent_window,
            op=lambda col: commit_task(col, pending_writes, stats, processed_tracker),
            success=on_saved
        )
        commit_op.failure(on_commit_failure)
        commit_op.run_in_background()

    def show_summary_and_callback(final_stats: BatchStats) -> None:
        """Display completion tooltip and invoke user callback."""
        tooltip(final_stats.summary_message, parent=parent_window)
        if on_finish:
            on_finish(final_stats)

    def on_notes_read(data: Tuple[List[Tuple[any, str, bool]], int, int]) -> None:
        """Launch background network downloads worker without collection lock."""
        def on_fetch_finished(fut: any) -> None:
            """Handle background download completion on the main GUI thread."""
            exc = fut.exception()
            if exc:
                logger.exception("Error during batch download: %s", exc)
                items, load_errors, missing_field_skips = data
                stats = BatchStats(total=total, errors=load_errors + len(items), skipped=missing_field_skips)
                show_summary_and_callback(stats)
                return
            on_fetch_done(fut.result())

        mw.taskman.with_progress(
            task=lambda: fetch_images_worker(data),
            on_done=on_fetch_finished,
            label=f"Connecting to Irasutoya (0/{total})...",
            parent=parent_window,
            uses_collection=False
        )

    # Launch initial note read
    read_op = QueryOp(
        parent=parent_window,
        op=read_notes_task,
        success=on_notes_read
    )
    read_op.run_in_background()
