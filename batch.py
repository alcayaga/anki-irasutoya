"""Batch processor for adding Irasutoya illustrations to Anki cards in the background."""
from dataclasses import dataclass, field
import logging
import re
import time
import urllib.parse
from typing import Callable, List, Optional, Sequence, Tuple

try:
    from .client import IrasutoyaClient, IrasutoyaImage
    from .config import get_config, CONFIG_KEY_FULL_DELAY, CONFIG_KEY_IMAGE_SIZE
    from .text_utils import clean_search_query
except ImportError:
    from client import IrasutoyaClient, IrasutoyaImage
    from config import get_config, CONFIG_KEY_FULL_DELAY, CONFIG_KEY_IMAGE_SIZE
    from text_utils import clean_search_query

logger = logging.getLogger(__name__)


@dataclass
class BatchStats:
    """Statistics tracking the outcome of a batch fill operation."""
    total: int = 0
    added: int = 0
    skipped: int = 0
    not_found: int = 0
    errors: int = 0
    canceled: bool = False
    details: List[str] = field(default_factory=list)

    @property
    def summary_message(self) -> str:
        """Formatted human-readable summary of the batch operation."""
        msg = f"Added: {self.added}, Skipped: {self.skipped}, Not found: {self.not_found}"
        if self.errors > 0:
            msg += f", Errors: {self.errors}"
        if self.canceled:
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
    image_size: str = "s800"
) -> Tuple[str, Optional[any]]:
    """Process a single note: check condition, search Irasutoya, download and set image.
    
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
    query = clean_search_query(raw_source)
    if not query:
        return "skipped_empty_query", None

    try:
        images = client.search(query, max_results=3, image_size=image_size)
        if not images:
            return "not_found", None

        best_img = images[0]
        img_bytes = client.download_image(best_img.image_url)
        if not img_bytes:
            return "not_found", None

        filename = make_safe_media_filename(query, best_img.title)
        stored_filename = write_media_func(filename, img_bytes)
        encoded_src = urllib.parse.quote(stored_filename)

        note[target_field] = f'<img src="{encoded_src}">'
        return "added", note
    except Exception as e:
        logger.exception("Error processing note with query '%s': %s", query, e)
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
    from aqt import mw
    from aqt.operations import QueryOp
    import aqt.operations.note
    from aqt.utils import tooltip

    unique_nids = list(dict.fromkeys(note_ids))
    total = len(unique_nids)
    if total == 0:
        return

    config = get_config()
    delay_ms = config.get(CONFIG_KEY_FULL_DELAY, 300)
    delay_sec = max(0.05, delay_ms / 1000.0)
    image_size = config.get(CONFIG_KEY_IMAGE_SIZE, "s800")
    client = IrasutoyaClient()

    def read_notes_task(col: any) -> Tuple[List[Tuple[any, str, bool]], int, int]:
        """Read note terms in a short collection-backed task and track load errors and missing field skips."""
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
                query_hint = clean_search_query(raw_val)
                has_target = bool(note[target_field].strip())
                items.append((note, query_hint, has_target))
            except Exception as e:
                logger.exception("Failed to load note %s: %s", nid, e)
                load_errors += 1
        return items, load_errors, missing_field_skips

    def fetch_images_worker(data: Tuple[List[Tuple[any, str, bool]], int, int]) -> Tuple[BatchStats, List[Tuple[any, str, bytes]]]:
        """Download images and pace requests outside the collection lock."""
        items, load_errors, missing_field_skips = data
        stats = BatchStats(total=total, errors=load_errors, skipped=missing_field_skips)
        pending_writes = []

        for idx, (note, query_hint, has_target) in enumerate(items):
            if mw.progress.want_cancel():
                stats.canceled = True
                break

            current_count = idx + 1
            # Update progress UI on main thread binding loop variables
            mw.taskman.run_on_main(
                lambda q=query_hint, c=current_count: mw.progress.update(
                    label=f"Fetching Irasutoya: {q or '...'} ({c}/{total})...",
                    value=c,
                    max=total
                )
            )

            if has_target and overwrite_mode == "skip":
                stats.skipped += 1
                continue

            if not query_hint:
                stats.skipped += 1
                continue

            if idx > 0:
                time.sleep(delay_sec)

            try:
                images = client.search(query_hint, max_results=3, image_size=image_size)
                if not images:
                    stats.not_found += 1
                    continue

                best_img = images[0]
                img_bytes = client.download_image(best_img.image_url)
                if not img_bytes:
                    stats.not_found += 1
                    continue

                filename = make_safe_media_filename(query_hint, best_img.title)
                pending_writes.append((note, filename, img_bytes))
            except Exception as e:
                logger.exception("Error during batch download for '%s': %s", query_hint, e)
                stats.errors += 1

        return stats, pending_writes

    def commit_task(
        col: any,
        pending_writes: List[Tuple[any, str, bytes]],
        stats: BatchStats,
        processed_tracker: List[int]
    ) -> List[any]:
        """Write media files and assign target field on freshly loaded notes within collection op."""
        notes_to_save = []
        for old_note, filename, img_bytes in pending_writes:
            try:
                fresh_note = col.get_note(old_note.id)
            except Exception:
                stats.skipped += 1
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
                notes_to_save.append(fresh_note)
            except Exception as e:
                logger.exception("Failed to write media for note %s: %s", old_note.id, e)
                stats.errors += 1
            processed_tracker[0] += 1
        return notes_to_save

    def on_fetch_done(result: Tuple[BatchStats, List[Tuple[any, str, bytes]]]) -> None:
        """Handle completion of network downloads and trigger media/note commit."""
        stats, pending_writes = result
        if not pending_writes:
            tooltip(stats.summary_message, parent=parent_window)
            if on_finish:
                on_finish(stats)
            return

        processed_tracker = [0]

        def on_saved(saved_notes: List[any]) -> None:
            """Update notes with full undo support and calculate final added count."""
            if not saved_notes:
                show_summary_and_callback(stats)
                return
            op = aqt.operations.note.update_notes(parent=parent_window, notes=saved_notes)

            def _ok(_: any) -> None:
                """Handle successful notes update."""
                stats.added = len(saved_notes)
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

    def on_notes_read(items: List[Tuple[any, str, bool]]) -> None:
        """Launch background network downloads worker without collection lock."""
        fetch_op = QueryOp(
            parent=parent_window,
            op=lambda _: fetch_images_worker(items),
            success=on_fetch_done
        )
        fetch_op.without_collection().with_progress(
            label=f"Connecting to Irasutoya (0/{total})..."
        ).run_in_background()

    # Launch initial note read
    read_op = QueryOp(
        parent=parent_window,
        op=read_notes_task,
        success=on_notes_read
    )
    read_op.run_in_background()
