"""Thumbnail caching and asynchronous paced downloading pipeline."""
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from aqt.qt import QByteArray, QImage, QPixmap, Qt

from .styles import (
    DIALOG_PIXMAP_CACHE_MAX_ENTRIES,
    THUMB_CACHE_MAX_BYTES,
    THUMB_CACHE_MAX_ENTRIES,
    THUMB_INITIAL_BURST,
    THUMB_WORKER_COUNT,
)


class _ThumbnailByteCache:
    """Thread-safe bounded LRU cache for raw thumbnail image bytes."""

    def __init__(
        self,
        max_entries: int = THUMB_CACHE_MAX_ENTRIES,
        max_bytes: int = THUMB_CACHE_MAX_BYTES,
    ) -> None:
        """Initialize bounded LRU byte cache with entry and size limits."""
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._data: OrderedDict[str, bytes] = OrderedDict()
        self._current_bytes = 0
        self._lock = threading.Lock()

    def get(self, url: str) -> Optional[bytes]:
        """Return cached bytes for url and mark as most-recently used."""
        if not url:
            return None
        with self._lock:
            data = self._data.get(url)
            if data is not None:
                self._data.move_to_end(url)
            return data

    def put(self, url: str, data: bytes) -> None:
        """Insert non-empty thumbnail bytes into the LRU cache."""
        if not url or not data:
            return
        data_len = len(data)
        if data_len > self.max_bytes:
            return
        with self._lock:
            if url in self._data:
                self._current_bytes -= len(self._data[url])
            self._data[url] = data
            self._current_bytes += data_len
            self._data.move_to_end(url)

            while (len(self._data) > self.max_entries or self._current_bytes > self.max_bytes) and self._data:
                _old_url, old_data = self._data.popitem(last=False)
                self._current_bytes -= len(old_data)

    def clear(self) -> None:
        """Remove all cached thumbnail bytes."""
        with self._lock:
            self._data.clear()
            self._current_bytes = 0

    def __len__(self) -> int:
        """Return the number of entries currently cached."""
        with self._lock:
            return len(self._data)


_SESSION_THUMB_CACHE: _ThumbnailByteCache = _ThumbnailByteCache()


def clear_session_thumbnail_cache() -> None:
    """Clear the session-wide thumbnail byte cache (primarily for tests)."""
    _SESSION_THUMB_CACHE.clear()


class ThumbnailLoader:
    """Manager for asynchronous paced thumbnail downloads, rate limiting, and two-tier caching."""

    def __init__(
        self,
        worker_count: int = THUMB_WORKER_COUNT,
        initial_burst: int = THUMB_INITIAL_BURST,
        session_cache: Optional[_ThumbnailByteCache] = None,
    ) -> None:
        """Initialize thumbnail loader with thread pool, rate limiting, and in-flight deduplication."""
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=worker_count)
        self.pending_futures: List[Future] = []
        self._thumb_lock: threading.Lock = threading.Lock()
        self._last_request_time: float = 0.0
        self._next_request_time: float = 0.0
        self._burst_remaining: int = initial_burst
        self.initial_burst: int = initial_burst
        self.byte_cache: _ThumbnailByteCache = (
            session_cache if session_cache is not None else _SESSION_THUMB_CACHE
        )
        self.pixmap_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._inflight: Dict[str, threading.Event] = {}
        self._inflight_lock: threading.Lock = threading.Lock()

    def reset_burst(self) -> None:
        """Reset burst token counter for the first visible row of a search generation."""
        self._burst_remaining = self.initial_burst

    def cancel_pending_futures(self) -> None:
        """Cancel unstarted queued thumbnail download futures."""
        for fut in self.pending_futures:
            if not fut.done():
                fut.cancel()
        self.pending_futures.clear()

    def shutdown(self) -> None:
        """Cancel pending tasks, shut down the worker executor, and clear local pixmap cache."""
        self.cancel_pending_futures()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.pixmap_cache.clear()

    @staticmethod
    def decode_thumbnail(thumb_data: bytes) -> Optional[QImage]:
        """Decode and scale thumbnail bytes into a smooth-transformed QImage, or None if invalid."""
        if not thumb_data:
            return None
        qimg = QImage.fromData(QByteArray(thumb_data))
        if qimg.isNull():
            return None
        return qimg.scaled(
            120,
            96,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def get_cached_pixmap(self, thumb_url: str) -> Optional[QPixmap]:
        """Lookup pre-scaled QPixmap from dialog cache or decode from session byte cache."""
        if not thumb_url:
            return None
        pixmap = self.pixmap_cache.get(thumb_url)
        if pixmap is not None:
            self.pixmap_cache.move_to_end(thumb_url)
            return pixmap

        data = self.byte_cache.get(thumb_url)
        if data:
            scaled_qimg = self.decode_thumbnail(data)
            if scaled_qimg is not None:
                pixmap = QPixmap.fromImage(scaled_qimg)
                self.cache_dialog_pixmap(thumb_url, pixmap)
                return pixmap
        return None

    def cache_dialog_pixmap(self, thumb_url: str, pixmap: QPixmap) -> None:
        """Store a pre-scaled QPixmap in the dialog LRU cache."""
        if not thumb_url or pixmap.isNull():
            return
        self.pixmap_cache[thumb_url] = pixmap
        self.pixmap_cache.move_to_end(thumb_url)
        while len(self.pixmap_cache) > DIALOG_PIXMAP_CACHE_MAX_ENTRIES:
            self.pixmap_cache.popitem(last=False)

    def fetch_thumbnail_task(
        self,
        card: Any,
        gen: int,
        client: Any,
        is_current_gen: Callable[[int], bool],
        run_on_main: Callable[[Callable[[], None]], None],
        delay_ms: int = 50,
    ) -> None:
        """Background thread worker to download a thumbnail with rate limiting and deduplication."""
        if not is_current_gen(gen):
            return
        delay_sec = max(0.0, delay_ms / 1000.0)
        thumb_url = card.image.thumb_url

        # In-flight deduplication: coalesce duplicate requests, but claim slot if previous downloader aborted
        while True:
            if not is_current_gen(gen):
                return

            cached_data = self.byte_cache.get(thumb_url)
            if cached_data:
                scaled_qimg = self.decode_thumbnail(cached_data)
                self._process_and_apply_thumbnail(
                    card, gen, cached_data, scaled_qimg, is_current_gen, run_on_main
                )
                return

            with self._inflight_lock:
                event_to_wait = self._inflight.get(thumb_url)
                if event_to_wait is None:
                    self._inflight[thumb_url] = threading.Event()
                    break

            # Another worker is actively downloading; wait for completion or abort
            event_to_wait.wait(timeout=8.5)

        # Active downloader: rate-limit slot reservation
        try:
            with self._thumb_lock:
                if not is_current_gen(gen):
                    return
                now = time.monotonic()
                if self._burst_remaining > 0:
                    self._burst_remaining -= 1
                    wait_sec = 0.0
                    self._next_request_time = max(now, self._next_request_time)
                else:
                    scheduled = max(now, self._next_request_time)
                    wait_sec = max(0.0, scheduled - now)
                    self._next_request_time = scheduled + delay_sec

            if wait_sec > 0:
                time.sleep(wait_sec)

            if not is_current_gen(gen):
                return

            try:
                thumb_data = client.download_image(thumb_url, timeout=8, image=card.image)
            except Exception:
                thumb_data = b""

            scaled_qimg = self.decode_thumbnail(thumb_data)
            # SAVE-BEFORE-GENERATION-CHECK: Cache only bytes that decode to a valid image
            if scaled_qimg is not None:
                self.byte_cache.put(thumb_url, thumb_data)
            else:
                thumb_data = b""

            self._process_and_apply_thumbnail(
                card, gen, thumb_data, scaled_qimg, is_current_gen, run_on_main
            )
        finally:
            with self._inflight_lock:
                ev = self._inflight.pop(thumb_url, None)
                if ev is not None:
                    ev.set()

    def _process_and_apply_thumbnail(
        self,
        card: Any,
        gen: int,
        thumb_data: bytes,
        scaled_qimg: Optional[QImage],
        is_current_gen: Callable[[int], bool],
        run_on_main: Callable[[Callable[[], None]], None],
    ) -> None:
        """Decode and scale thumbnail on background worker if needed, then marshal to main thread."""
        if scaled_qimg is None and thumb_data:
            scaled_qimg = self.decode_thumbnail(thumb_data)

        run_on_main(
            lambda td=thumb_data, sq=scaled_qimg: self._apply_thumbnail(
                card, gen, td, sq, is_current_gen
            )
        )

    def _apply_thumbnail(
        self,
        card: Any,
        gen: int,
        thumb_data: bytes,
        scaled_qimg: Optional[QImage],
        is_current_gen: Callable[[int], bool],
    ) -> None:
        """Main thread callback to apply thumbnail pixmap and populate dialog cache."""
        thumb_url = card.image.thumb_url
        if thumb_data:
            pixmap = self.pixmap_cache.get(thumb_url)
            if pixmap is None:
                if scaled_qimg is not None and not scaled_qimg.isNull():
                    pixmap = QPixmap.fromImage(scaled_qimg)
                else:
                    raw_pixmap = QPixmap()
                    raw_pixmap.loadFromData(QByteArray(thumb_data))
                    if not raw_pixmap.isNull():
                        pixmap = raw_pixmap.scaled(
                            120,
                            96,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                if pixmap is not None and not pixmap.isNull():
                    self.cache_dialog_pixmap(thumb_url, pixmap)

            if is_current_gen(gen):
                if pixmap is not None and not pixmap.isNull():
                    card.set_scaled_pixmap(pixmap)
                    return
                card.thumb_label.setText("No Preview")
            return

        if is_current_gen(gen):
            card.thumb_label.setText("No Preview")
