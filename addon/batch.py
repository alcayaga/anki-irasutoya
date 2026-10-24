"""Batch processor for adding Irasutoya illustrations to Anki cards in the background."""
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

from .catalog import default_store
from .client import IrasutoyaClient
from .config import (
    CONFIG_KEY_FULL_DELAY,
    CONFIG_KEY_IMAGE_SIZE,
    CONFIG_KEY_SEARCH_BACKEND,
    get_config,
)
from .ranking import (
    infer_note_pos_type,
    infer_pos_type,
    is_decorative_asset,
    is_image_relevant,
    select_best_image,
)
from .text_utils import BATCH_MAX_CANDIDATES, generate_search_candidates
from .yomitan_parser import extract_yomitan_metadata_from_note

logger = logging.getLogger(__name__)

FALLBACK_AUDIT_TAG = "irasutoya::fallback_reading"


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
    fallback_matches: int = 0
    skipped: int = 0
    not_found: int = 0
    errors: int = 0
    canceled: bool = False
    details: List[str] = field(default_factory=list)

    @property
    def summary_message(self) -> str:
        """Formatted human-readable summary of the batch operation."""
        if self.fallback_matches > 0:
            msg = f"Added: {self.added} ({self.direct_matches} direct, {self.fallback_matches} via fallback), Skipped: {self.skipped}, Not found: {self.not_found}"
        else:
            msg = f"Added: {self.added}, Skipped: {self.skipped}, Not found: {self.not_found}"

        if self.errors > 0:
            msg += f", Errors: {self.errors}"
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
        - "skipped_missing_field": field not in note
        - "skipped_not_empty": target field already has content and overwrite is 'skip'
        - "skipped_empty_query": source field contains no searchable text
        - "not_found": no matching images on Irasutoya
        - "error": download or unexpected failure
    """
    if source_field not in note or target_field not in note:
        return "skipped_missing_field", None

    existing_target = note[target_field].strip()
    if existing_target and overwrite_mode == "skip":
        return "skipped_not_empty", None

    raw_source = note[source_field]
    candidates = generate_search_candidates(raw_source, note=note, allow_tier2=allow_tier2, allow_deinflection=True)
    if not candidates:
        return "skipped_empty_query", None

    primary_query = candidates[0][0]
    batch_candidates = candidates[:BATCH_MAX_CANDIDATES]

    try:
        pos_type = safe_extract_note_pos_type(note)
        if hasattr(client, "search_with_fallbacks"):
            images, matched_cand, best_img = client.search_with_fallbacks(
                batch_candidates, max_results=12, image_size=image_size, pos_type=pos_type
            )
        else:
            matched_cand = None
            best_img = None
            for cand in batch_candidates:
                imgs = client.search(cand[0], max_results=3, image_size=image_size)
                valid_imgs = [
                    img for img in imgs
                    if not is_decorative_asset(img, cand[0]) and is_image_relevant(img, cand[0])
                ]
                if valid_imgs:
                    matched_cand = cand
                    cand_pos = pos_type or infer_pos_type(cand[0], cand_reason=cand[1])
                    best_img = select_best_image(valid_imgs, cand[0], pos_type=cand_pos) or valid_imgs[0]
                    break

        if not best_img or not matched_cand:
            return "not_found", None


        img_bytes = client.download_image(best_img.image_url, image=best_img)
        if not img_bytes:
            return "not_found", None

        filename = make_safe_media_filename(matched_cand[0], best_img.title)
        stored_filename = write_media_func(filename, img_bytes)
        encoded_src = urllib.parse.quote(stored_filename)

        note[target_field] = f'<img src="{encoded_src}">'

        # If resolved via Tier 2 reading fallback, tag the note for auditability;
        # otherwise remove any stale fallback tag from an earlier batch run if overwritten directly
        if matched_cand[2] == 2:
            if hasattr(note, "add_tag"):
                note.add_tag(FALLBACK_AUDIT_TAG)
            elif hasattr(note, "tags") and isinstance(note.tags, list):
                if FALLBACK_AUDIT_TAG not in note.tags:
                    note.tags.append(FALLBACK_AUDIT_TAG)
        else:
            if hasattr(note, "remove_tag"):
                note.remove_tag(FALLBACK_AUDIT_TAG)
            elif hasattr(note, "tags") and isinstance(note.tags, list):
                if FALLBACK_AUDIT_TAG in note.tags:
                    note.tags.remove(FALLBACK_AUDIT_TAG)

        return "added", note
    except Exception as e:
        logger.exception("Error processing note with query '%s': %s", primary_query, e)
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
    search_backend = config.get(CONFIG_KEY_SEARCH_BACKEND, "auto")
    store = default_store()
    store.warm_async()
    client = IrasutoyaClient(catalog=store, backend=search_backend)

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
                candidates = generate_search_candidates(raw_val, note=note, allow_tier2=True, allow_deinflection=True)
                has_target = bool(note[target_field].strip())
                items.append((note, candidates, has_target))
            except Exception as e:
                logger.exception("Failed to load note %s: %s", nid, e)
                load_errors += 1
        return items, load_errors, missing_field_skips

    def fetch_images_worker(
        data: Tuple[List[Tuple[any, List[Tuple[str, str, int]], bool]], int, int]
    ) -> Tuple[BatchStats, List[Tuple[any, str, bytes, bool]]]:
        """Download images and pace requests outside the collection lock using fallback search."""
        items, load_errors, missing_field_skips = data
        stats = BatchStats(total=total, errors=load_errors, skipped=missing_field_skips)
        pending_writes: List[Tuple[any, str, bytes, bool]] = []
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
            primary_label = candidates[0][0] if candidates else "..."
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

            batch_candidates = candidates[:BATCH_MAX_CANDIDATES]
            try:
                pos_type = safe_extract_note_pos_type(note)
                # pace_request applies before each candidate query to avoid rate limits,
                # incurring polite pacing delays for up to BATCH_MAX_CANDIDATES queries per note.
                images, matched_cand, best_img = client.search_with_fallbacks(
                    batch_candidates,
                    max_results=12,
                    image_size=image_size,
                    pace_callback=pace_request,
                    pos_type=pos_type,
                )
                if not best_img or not matched_cand:
                    stats.not_found += 1
                    continue

                pace_request()
                img_bytes = client.download_image(best_img.image_url, image=best_img)
                if not img_bytes:
                    stats.not_found += 1
                    continue

                is_tier2 = bool(matched_cand[2] == 2)
                filename = make_safe_media_filename(matched_cand[0], best_img.title)
                pending_writes.append((note, filename, img_bytes, is_tier2))
            except Exception as e:
                logger.exception("Error during batch download for note %s: %s", getattr(note, "id", "?"), e)
                stats.errors += 1

        return stats, pending_writes

    def commit_task(
        col: any,
        pending_writes: List[Tuple[any, str, bytes, bool]],
        stats: BatchStats,
        processed_tracker: List[int]
    ) -> Tuple[List[any], List[bool]]:
        """Write media files and assign target field on freshly loaded notes within collection op."""
        notes_to_save = []
        saved_tiers = []
        for old_note, filename, img_bytes, is_tier2 in pending_writes:
            try:
                fresh_note = col.get_note(old_note.id)
            except Exception as e:
                logger.exception("Failed to load note %s at commit time: %s", old_note.id, e)
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
                stored_filename = col.media.write_data(filename, img_bytes)
                encoded_src = urllib.parse.quote(stored_filename)
                fresh_note[target_field] = f'<img src="{encoded_src}">'
                if is_tier2:
                    if hasattr(fresh_note, "add_tag"):
                        fresh_note.add_tag(FALLBACK_AUDIT_TAG)
                    elif hasattr(fresh_note, "tags") and isinstance(fresh_note.tags, list):
                        if FALLBACK_AUDIT_TAG not in fresh_note.tags:
                            fresh_note.tags.append(FALLBACK_AUDIT_TAG)
                else:
                    if hasattr(fresh_note, "remove_tag"):
                        fresh_note.remove_tag(FALLBACK_AUDIT_TAG)
                    elif hasattr(fresh_note, "tags") and isinstance(fresh_note.tags, list):
                        if FALLBACK_AUDIT_TAG in fresh_note.tags:
                            fresh_note.tags.remove(FALLBACK_AUDIT_TAG)
                notes_to_save.append(fresh_note)
                saved_tiers.append(is_tier2)
            except Exception as e:
                logger.exception("Failed to write media for note %s: %s", old_note.id, e)
                stats.errors += 1
            processed_tracker[0] += 1
        return notes_to_save, saved_tiers


    def on_fetch_done(result: Tuple[BatchStats, List[Tuple[any, str, bytes]]]) -> None:
        """Handle completion of network downloads and trigger media/note commit."""
        stats, pending_writes = result
        if not pending_writes:
            tooltip(stats.summary_message, parent=parent_window)
            if on_finish:
                on_finish(stats)
            return

        processed_tracker = [0]

        def on_saved(saved_info: Tuple[List[any], List[bool]]) -> None:
            """Update notes with full undo support and calculate final added count."""
            saved_notes, saved_tiers = saved_info
            if not saved_notes:
                show_summary_and_callback(stats)
                return
            op = aqt.operations.note.update_notes(parent=parent_window, notes=saved_notes)

            def _ok(_: any) -> None:
                """Handle successful notes update."""
                stats.added = len(saved_notes)
                stats.fallback_matches = sum(1 for is_t2 in saved_tiers if is_t2)
                stats.direct_matches = stats.added - stats.fallback_matches
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
