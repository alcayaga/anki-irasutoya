"""Batch processor for adding Irasutoya illustrations to Anki cards in the background."""
from dataclasses import dataclass, field
import logging
import re
import time
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
    total: int = 0
    added: int = 0
    skipped: int = 0
    not_found: int = 0
    canceled: bool = False
    details: List[str] = field(default_factory=list)

    @property
    def summary_message(self) -> str:
        if self.canceled:
            return f"Canceled. Added: {self.added}, Skipped: {self.skipped}, Not found: {self.not_found}"
        return f"Finished! Added: {self.added}, Skipped: {self.skipped}, Not found: {self.not_found}"


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

        note[target_field] = f'<img src="{stored_filename}">'
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
    """Execute batch processing in a background worker with progress display and undo."""
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

    stats = BatchStats(total=total)
    modified_notes: List[any] = []

    def background_worker(col: any) -> Tuple[BatchStats, List[any]]:
        for idx, nid in enumerate(unique_nids):
            # Check user cancellation from Anki's progress dialog
            if mw.progress.want_cancel():
                stats.canceled = True
                break

            try:
                note = col.get_note(nid)
            except Exception:
                stats.skipped += 1
                continue

            raw_val = note[source_field] if source_field in note else ""
            query_hint = clean_search_query(raw_val) or "..."
            current_count = idx + 1

            # Update progress UI on main thread
            mw.taskman.run_on_main(
                lambda: mw.progress.update(
                    label=f"Fetching Irasutoya: {query_hint} ({current_count}/{total})...",
                    value=current_count,
                    max=total
                )
            )

            # Polite request pacing (hidden from UI)
            if idx > 0:
                time.sleep(delay_sec)

            status, mod_note = process_note_record(
                note=note,
                source_field=source_field,
                target_field=target_field,
                overwrite_mode=overwrite_mode,
                client=client,
                write_media_func=col.media.write_data,
                image_size=image_size
            )

            if status == "added" and mod_note:
                stats.added += 1
                modified_notes.append(mod_note)
            elif status in ("skipped_missing_field", "skipped_not_empty", "skipped_empty_query"):
                stats.skipped += 1
            elif status == "not_found":
                stats.not_found += 1
            else:
                stats.skipped += 1

        return stats, modified_notes

    def on_complete(result: Tuple[BatchStats, List[any]]) -> None:
        completed_stats, notes_to_save = result

        if notes_to_save:
            # Commit note updates with full native undo support
            op = aqt.operations.note.update_notes(parent=parent_window, notes=notes_to_save)
            op.success(lambda _: show_summary_and_callback(completed_stats))
            op.run_in_background()
        else:
            show_summary_and_callback(completed_stats)

    def show_summary_and_callback(final_stats: BatchStats) -> None:
        tooltip(final_stats.summary_message, parent=parent_window)
        if on_finish:
            on_finish(final_stats)

    op = QueryOp(
        parent=parent_window,
        op=background_worker,
        success=on_complete
    )
    op.with_progress(label=f"Connecting to Irasutoya (0/{total})...").run_in_background()
