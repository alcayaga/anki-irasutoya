"""User interface for the Anki Irasutoya add-on."""
import html
import logging
import os
import re
import threading
import time
import urllib.parse
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Set, Tuple

from aqt import mw
from aqt.qt import (
    QAbstractItemView,
    QButtonGroup,
    QByteArray,
    QColor,
    QComboBox,
    QCursor,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QImage,
    QLabel,
    QLineEdit,
    QPixmap,
    QPushButton,
    QRadioButton,
    QScrollArea,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTimer,
    QVBoxLayout,
    QWidget,
)
from aqt.utils import showWarning, tooltip

from .batch import make_safe_media_filename, run_batch_fill
from .catalog import default_store
from .client import IrasutoyaClient
from .config import (
    CONFIG_KEY_IMAGE_SIZE,
    CONFIG_KEY_OVERWRITE,
    CONFIG_KEY_SOURCE,
    CONFIG_KEY_TARGET,
    CONFIG_KEY_THUMB_DELAY,
    get_config,
    update_user_preferences,
)
from .feed import IrasutoyaImage
from .ranking import is_decorative_asset, is_image_relevant, is_title_match
from .text_utils import clean_search_query, generate_search_candidates

logger = logging.getLogger(__name__)


class _StaleSearch(Exception):
    """Internal signal that the current search generation was superseded."""


ICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
CHEVRON_ICON_PATH = os.path.join(ICONS_DIR, "chevron_down.svg").replace("\\", "/")

DIALOG_STYLESHEET = f"""
QDialog {{
    background-color: palette(window);
}}
QPushButton#primaryBtn {{
    background-color: #0A84FF;
    color: #FFFFFF;
    border: 1px solid #0077ED;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 18px;
    min-height: 20px;
}}
QPushButton#primaryBtn:hover {{
    background-color: #0071E3;
}}
QPushButton#primaryBtn:pressed {{
    background-color: #005BB5;
}}
QPushButton#primaryBtn:disabled {{
    background-color: rgba(120, 120, 128, 0.2);
    color: rgba(255, 255, 255, 0.35);
    border: 1px solid transparent;
}}
QPushButton#secondaryBtn {{
    background-color: rgba(120, 120, 128, 0.16);
    color: palette(text);
    border: 1px solid rgba(120, 120, 128, 0.25);
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 14px;
    min-height: 20px;
}}
QPushButton#secondaryBtn:hover {{
    background-color: rgba(120, 120, 128, 0.24);
}}
QPushButton#secondaryBtn:pressed {{
    background-color: rgba(120, 120, 128, 0.32);
}}
QPushButton#secondaryBtn:disabled {{
    opacity: 0.5;
}}
QLineEdit {{
    border: 1px solid rgba(120, 120, 128, 0.35);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 13px;
    background-color: palette(base);
    color: palette(text);
}}
QLineEdit:focus {{
    border: 1.5px solid #0A84FF;
}}
QComboBox {{
    border: 1px solid rgba(120, 120, 128, 0.35);
    border-radius: 6px;
    padding: 5px 30px 5px 12px;
    font-size: 13px;
    background-color: palette(base);
    color: palette(text);
    min-height: 22px;
}}
QComboBox:focus {{
    border: 1.5px solid #0A84FF;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 26px;
    border-left-width: 0px;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
    background: transparent;
}}
QComboBox::down-arrow {{
    image: url({CHEVRON_ICON_PATH});
    width: 10px;
    height: 6px;
    margin-right: 8px;
}}
QRadioButton {{
    font-size: 12px;
}}
"""


PAGE_SIZE = 24
MAX_SUGGESTION_CHIPS = 8
THUMB_CACHE_MAX_ENTRIES = 512
THUMB_CACHE_MAX_BYTES = 8 * 1024 * 1024  # 8 MB hard cap
DIALOG_PIXMAP_CACHE_MAX_ENTRIES = 128
THUMB_WORKER_COUNT = 6
THUMB_INITIAL_BURST = 4


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


_SESSION_THUMB_CACHE = _ThumbnailByteCache()


def clear_session_thumbnail_cache() -> None:
    """Clear the session-wide thumbnail byte cache (primarily for tests)."""
    _SESSION_THUMB_CACHE.clear()


def make_description_snippet(description: str, query: str, max_chars: int = 70) -> str:
    """Generate an excerpt around the query term with bold highlight for tooltips."""
    if not description or not query:
        return ""
    clean_desc = re.sub(r"\s+", " ", description).strip()
    match = re.search(re.escape(query), clean_desc, re.IGNORECASE)
    if match is None:
        return html.escape(clean_desc[:max_chars])
    m_start, m_end = match.span()
    start = max(0, m_start - 18)
    end = min(len(clean_desc), m_end + 38)
    prefix = ("…" if start > 0 else "") + html.escape(clean_desc[start:m_start])
    body = "<b>" + html.escape(clean_desc[m_start:m_end]) + "</b>"
    suffix = html.escape(clean_desc[m_end:end]) + ("…" if end < len(clean_desc) else "")
    return prefix + body + suffix


def _description_match_term(img: IrasutoyaImage, term: str) -> Optional[str]:
    """Return the query term to highlight if img matches term in description only, else None."""
    clean_t = term.strip()
    if not clean_t or is_title_match(img, clean_t):
        return None
    desc = (img.description or "").casefold()
    tokens = clean_t.split()
    if tokens and all(t.casefold() in desc for t in tokens):
        return tokens[0]
    return None


def _is_desc_only_match(images: Sequence[IrasutoyaImage], term: str) -> bool:
    """Return True if all images match term in description only (not in title or post_title with boundary safety)."""
    clean_t = term.strip()
    if not clean_t or not images:
        return False
    return all(_description_match_term(img, clean_t) is not None for img in images)


class ImageCardWidget(QFrame):
    """Clickable thumbnail card displaying an Irasutoya illustration."""

    def __init__(
        self,
        image: IrasutoyaImage,
        on_select,
        on_double_click,
        parent=None,
        is_description_match: bool = False,
        match_snippet: Optional[str] = None,
    ):
        """Initialize the illustration thumbnail card widget."""
        super().__init__(parent)
        self.image = image
        self.on_select = on_select
        self.on_double_click = on_double_click
        self.is_description_match = is_description_match
        self.match_snippet = match_snippet
        self.is_selected = False

        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedWidth(145)
        self.setFixedHeight(165)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # White canvas container for illustration so transparent PNGs with dark linework/hair stay crisp
        self.thumb_label = QLabel(self)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setFixedSize(129, 105)
        self.thumb_label.setText("Loading...")
        self.thumb_label.setStyleSheet(
            "background-color: #FFFFFF; "
            "color: #8E8E93; "
            "border-radius: 6px; "
            "font-size: 11px;"
        )
        layout.addWidget(self.thumb_label)

        self.title_label = QLabel(image.title, self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-size: 11px; font-weight: 500; color: palette(text);")
        layout.addWidget(self.title_label)

        # Discreet checkmark badge overlay in top-right corner
        self.badge_label = QLabel("✓", self)
        self.badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge_label.setFixedSize(18, 18)
        self.badge_label.setStyleSheet(
            "background-color: #0A84FF; "
            "color: #FFFFFF; "
            "border-radius: 9px; "
            "font-size: 11px; "
            "font-weight: bold; "
            "border: 1.5px solid palette(window);"
        )
        self.badge_label.move(self.width() - 24, 6)
        self.badge_label.setVisible(False)

        # Incidental description match badge overlay in top-left corner
        self.desc_match_badge = QLabel("📝 本文一致", self)
        self.desc_match_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc_match_badge.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0.68); "
            "color: #FFD60A; "
            "border-radius: 4px; "
            "font-size: 10px; "
            "font-weight: 600; "
            "padding: 1px 4px;"
        )
        self.desc_match_badge.adjustSize()
        self.desc_match_badge.move(12, 12)
        if self.is_description_match:
            self.desc_match_badge.setVisible(True)
            if self.match_snippet:
                escaped_title = html.escape(image.title)
                tip = f"<b>{escaped_title}</b><br>本文一致: {self.match_snippet}"
                self.desc_match_badge.setToolTip(tip)
                self.setToolTip(tip)
        else:
            self.desc_match_badge.setVisible(False)

        self.update_style()

    def set_scaled_pixmap(self, pixmap: QPixmap) -> None:
        """Set an already-scaled preview pixmap on the white canvas directly."""
        self.thumb_label.setPixmap(pixmap)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Set preview pixmap on the white canvas, scaling smoothly if needed."""
        if pixmap.width() <= 120 and pixmap.height() <= 96:
            self.thumb_label.setPixmap(pixmap)
            return
        scaled = pixmap.scaled(
            120,
            96,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.thumb_label.setPixmap(scaled)

    def set_selected(self, selected: bool) -> None:
        """Update selected state and toggle checkmark badge."""
        self.is_selected = selected
        self.badge_label.setVisible(selected)
        self.update_style()

    def update_style(self) -> None:
        """Update card border and highlight stylesheet based on selection."""
        if self.is_selected:
            self.setStyleSheet(
                "ImageCardWidget {"
                "   border: 2px solid #0A84FF;"
                "   background-color: rgba(10, 132, 255, 0.12);"
                "   border-radius: 8px;"
                "}"
            )
        else:
            self.setStyleSheet(
                "ImageCardWidget {"
                "   border: 1px solid rgba(120, 120, 128, 0.25);"
                "   background-color: palette(alternate-base);"
                "   border-radius: 8px;"
                "}"
                "ImageCardWidget:hover {"
                "   border: 1.5px solid #0A84FF;"
                "}"
            )

    def mousePressEvent(self, event) -> None:
        """Handle mouse click selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_select(self)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        """Handle double click to select and immediately apply."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_double_click(self)
        super().mouseDoubleClickEvent(event)


class IrasutoyaDialog(QDialog):
    """Main dialog for mapping card fields and executing batch or interactive image insertion."""

    def __init__(self, parent, browser, note_ids: Sequence[int]):
        """Initialize the dialog with selected notes and configuration."""
        super().__init__(parent)
        self.browser = browser
        self.note_ids = list(dict.fromkeys(note_ids))
        self.config = get_config()
        self.store = default_store()
        self.store.warm_async()
        self.client = IrasutoyaClient(catalog=self.store)

        self.selected_card_widget: Optional[ImageCardWidget] = None
        self.image_cards: List[ImageCardWidget] = []
        self.first_note = None
        self._applying = False
        self._note_cache: Dict[int, any] = {}

        if self.note_ids and mw and hasattr(mw, "col") and mw.col:
            try:
                self.first_note = mw.col.get_note(self.note_ids[0])
                self._note_cache[self.note_ids[0]] = self.first_note
            except Exception:
                pass

        self.setWindowTitle("Add Irasutoya Images")
        self.setMinimumWidth(580)
        if len(self.note_ids) == 1:
            self.resize(620, 660)
        else:
            self.resize(620, 520)

        self.setStyleSheet(DIALOG_STYLESHEET)

        self._search_generation = 0
        self._thumb_executor = ThreadPoolExecutor(max_workers=THUMB_WORKER_COUNT)
        self._pending_thumb_futures: List[Future] = []
        self._thumb_lock = threading.Lock()
        self._last_thumb_request_time = 0.0
        self._next_thumb_request_time = 0.0
        self._thumb_burst_remaining = THUMB_INITIAL_BURST
        self._thumb_byte_cache = _SESSION_THUMB_CACHE
        self._thumb_pixmap_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._thumb_inflight: Dict[str, threading.Event] = {}
        self._thumb_inflight_lock = threading.Lock()
        self.backend_lbl: Optional[QLabel] = None

        self._all_images: List[IrasutoyaImage] = []
        self._loaded_count: int = 0
        self._current_query: str = ""
        self.pagination_label: Optional[QLabel] = None
        self.media_notice_label: Optional[QLabel] = None
        self.btn_cancel: Optional[QPushButton] = None
        self._catalog_count_cache: Dict[str, Dict[str, int]] = {}
        self._catalog_desc_cache: Dict[str, Dict[str, bool]] = {}

        self._init_ui()

    def _get_backend_status_text(self) -> str:
        """Format human-readable backend indicator without blocking on background catalog load."""
        if not os.path.isfile(self.store.path):
            return "Offline catalog (missing)"
        if self.store.is_load_failed():
            return "Offline catalog (failed)"
        if not self.store.is_loaded():
            return "Offline catalog (loading...)"
        info = self.store.info()
        if info and info.post_count > 0:
            date_str = info.built_at[:10] if info.built_at else ""
            return f"Offline catalog · {info.post_count:,} posts" + (f" · built {date_str}" if date_str else "")
        return "Offline catalog"

    def _schedule_backend_status_refresh(self) -> None:
        """Poll until the background catalog finishes loading, then update the status label."""
        def tick() -> None:
            if not hasattr(self, "backend_lbl") or self.backend_lbl is None:
                return
            try:
                if not self.isVisible():
                    return
            except Exception:
                return
            self.backend_lbl.setText(self._get_backend_status_text())
            if not self.store.is_loaded() and not self.store.is_load_failed() and os.path.isfile(self.store.path):
                QTimer.singleShot(250, tick)

        if not self.store.is_loaded() and not self.store.is_load_failed() and os.path.isfile(self.store.path):
            QTimer.singleShot(250, tick)

    def _get_cached_note(self, nid: int) -> Optional[any]:
        """Fetch a note with in-dialog memory caching to prevent redundant collection reads."""
        if nid not in self._note_cache:
            try:
                self._note_cache[nid] = mw.col.get_note(nid)
            except Exception:
                self._note_cache[nid] = None
        return self._note_cache[nid]

    def _run_on_main(self, callback) -> None:
        """Execute a callable safely on the Qt main GUI thread."""
        try:
            if mw and hasattr(mw, "taskman") and mw.taskman:
                mw.taskman.run_on_main(callback)
                return
        except Exception:
            pass
        QTimer.singleShot(0, callback)

    def _cancel_pending_thumb_futures(self) -> None:
        """Cancel unstarted queued thumbnail download futures."""
        for fut in self._pending_thumb_futures:
            if not fut.done():
                fut.cancel()
        self._pending_thumb_futures.clear()

    def _get_cached_pixmap(self, thumb_url: str) -> Optional[QPixmap]:
        """Lookup pre-scaled QPixmap from dialog cache or decode from session byte cache."""
        if not thumb_url:
            return None
        pixmap = self._thumb_pixmap_cache.get(thumb_url)
        if pixmap is not None:
            self._thumb_pixmap_cache.move_to_end(thumb_url)
            return pixmap

        data = self._thumb_byte_cache.get(thumb_url)
        if data:
            qimg = QImage.fromData(QByteArray(data))
            if not qimg.isNull():
                scaled_qimg = qimg.scaled(
                    120,
                    96,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                pixmap = QPixmap.fromImage(scaled_qimg)
                self._cache_dialog_pixmap(thumb_url, pixmap)
                return pixmap
        return None

    def _cache_dialog_pixmap(self, thumb_url: str, pixmap: QPixmap) -> None:
        """Store a pre-scaled QPixmap in the dialog LRU cache."""
        if not thumb_url or pixmap.isNull():
            return
        self._thumb_pixmap_cache[thumb_url] = pixmap
        self._thumb_pixmap_cache.move_to_end(thumb_url)
        while len(self._thumb_pixmap_cache) > DIALOG_PIXMAP_CACHE_MAX_ENTRIES:
            self._thumb_pixmap_cache.popitem(last=False)

    def done(self, result: int) -> None:
        """Shut down background thumbnail executor, release caches, and close dialog."""
        self._applying = False
        self._search_generation += 1
        self._cancel_pending_thumb_futures()
        self._thumb_executor.shutdown(wait=False, cancel_futures=True)
        self._thumb_pixmap_cache.clear()
        if hasattr(self, "client") and self.client:
            self.client.close()
        super().done(result)

    def resizeEvent(self, event) -> None:
        """Handle dialog resize and load more items if viewport expands without scrolling."""
        super().resizeEvent(event)
        self._check_viewport_pagination()

    def showEvent(self, event) -> None:
        """Handle dialog show event and ensure viewport is filled."""
        super().showEvent(event)
        self._check_viewport_pagination()

    def keyPressEvent(self, event) -> None:
        """Enable keyboard arrow navigation across the thumbnail gallery and Enter to apply."""
        if hasattr(self, "image_cards") and self.image_cards:
            key = event.key()
            current_idx = -1
            if self.selected_card_widget in self.image_cards:
                current_idx = self.image_cards.index(self.selected_card_widget)

            new_idx = None
            if key == Qt.Key.Key_Right:
                if (current_idx + 1 >= len(self.image_cards)) and (self._loaded_count < len(self._all_images)):
                    self._load_next_page(self._search_generation)
                new_idx = min(current_idx + 1, len(self.image_cards) - 1)
            elif key == Qt.Key.Key_Left:
                new_idx = max(current_idx - 1, 0)
            elif key == Qt.Key.Key_Down:
                if (current_idx + 3 >= len(self.image_cards)) and (self._loaded_count < len(self._all_images)):
                    self._load_next_page(self._search_generation)
                new_idx = min(current_idx + 3, len(self.image_cards) - 1)
            elif key == Qt.Key.Key_Up:
                new_idx = max(current_idx - 3, 0)
            elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                # If query input doesn't have focus, apply selected card
                if not (hasattr(self, "query_input") and self.query_input.hasFocus()):
                    if self.selected_card_widget and hasattr(self, "btn_apply") and self.btn_apply.isEnabled():
                        self._apply_single_selection()
                        return

            if new_idx is not None and new_idx != current_idx:
                target_card = self.image_cards[new_idx]
                self._on_card_selected(target_card)
                if hasattr(self, "scroll_area"):
                    self.scroll_area.ensureWidgetVisible(target_card)
                return

        super().keyPressEvent(event)

    def _get_available_fields(self) -> List[str]:
        """Collect all unique field names across selected notes' models."""
        fields: Set[str] = set()
        if mw and hasattr(mw, "col") and mw.col:
            for nid in self.note_ids[:50]:  # sample notes
                try:
                    note = mw.col.get_note(nid)
                    fields.update(note.keys())
                except Exception:
                    pass

        sorted_fields = sorted(list(fields))
        return sorted_fields or ["Expression", "DefinitionPicture"]

    def _init_ui(self) -> None:
        """Initialize base header, field mapping frame, and mode-specific views."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(12)

        # Header Badge
        header_layout = QHBoxLayout()
        title_label = QLabel("Add Irasutoya Images", self)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: palette(text);")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        note_count = len(self.note_ids)
        count_badge = QLabel(f"<b>{note_count}</b> {'card' if note_count == 1 else 'cards'} selected", self)
        count_badge.setStyleSheet(
            "background-color: rgba(120, 120, 128, 0.18); "
            "color: palette(text); "
            "padding: 4px 10px; "
            "border-radius: 10px; "
            "font-size: 11px;"
        )
        header_layout.addWidget(count_badge)
        main_layout.addLayout(header_layout)

        # Field Mapping Group
        fields_frame = QFrame(self)
        fields_frame.setStyleSheet(
            "QFrame#fieldsFrame {"
            "   background-color: palette(alternate-base);"
            "   border: 1px solid rgba(120, 120, 128, 0.2);"
            "   border-radius: 8px;"
            "}"
        )
        fields_frame.setObjectName("fieldsFrame")
        fields_layout = QVBoxLayout(fields_frame)
        fields_layout.setContentsMargins(14, 12, 14, 12)
        fields_layout.setSpacing(10)

        avail_fields = self._get_available_fields()

        # Source & Target Comboboxes
        combo_layout = QGridLayout()
        combo_layout.setHorizontalSpacing(14)
        combo_layout.setVerticalSpacing(8)

        lbl_src = QLabel("Source Field:", self)
        lbl_src.setStyleSheet("font-weight: 500; font-size: 12px;")
        self.combo_source = QComboBox(self)
        self.combo_source.addItems(avail_fields)

        lbl_tgt = QLabel("Target Field:", self)
        lbl_tgt.setStyleSheet("font-weight: 500; font-size: 12px;")
        self.combo_target = QComboBox(self)
        self.combo_target.addItems(avail_fields)

        # Set default selections based on config or heuristics
        last_src = self.config.get(CONFIG_KEY_SOURCE, "Expression")
        if last_src in avail_fields:
            self.combo_source.setCurrentText(last_src)
        else:
            for cand in ["Expression", "Vocab", "Word", "Front", "Kanji"]:
                if cand in avail_fields:
                    self.combo_source.setCurrentText(cand)
                    break

        last_tgt = self.config.get(CONFIG_KEY_TARGET, "DefinitionPicture")
        if last_tgt in avail_fields:
            self.combo_target.setCurrentText(last_tgt)
        else:
            for cand in ["DefinitionPicture", "Picture", "Image", "Illustration"]:
                if cand in avail_fields:
                    self.combo_target.setCurrentText(cand)
                    break

        combo_layout.addWidget(lbl_src, 0, 0)
        combo_layout.addWidget(self.combo_source, 0, 1)
        combo_layout.addWidget(lbl_tgt, 1, 0)
        combo_layout.addWidget(self.combo_target, 1, 1)
        fields_layout.addLayout(combo_layout)

        # Overwrite policy radio buttons - ONLY IN BATCH MODE
        if len(self.note_ids) > 1:
            policy_layout = QHBoxLayout()
            policy_label = QLabel("Existing Content:", self)
            policy_label.setStyleSheet("font-weight: 500; font-size: 12px;")
            policy_layout.addWidget(policy_label)

            self.radio_skip = QRadioButton("Skip if target field is not empty", self)
            self.radio_overwrite = QRadioButton("Overwrite target field", self)

            self.policy_group = QButtonGroup(self)
            self.policy_group.addButton(self.radio_skip)
            self.policy_group.addButton(self.radio_overwrite)

            last_policy = self.config.get(CONFIG_KEY_OVERWRITE, "skip")
            if last_policy == "overwrite":
                self.radio_overwrite.setChecked(True)
            else:
                self.radio_skip.setChecked(True)

            policy_layout.addWidget(self.radio_skip)
            policy_layout.addWidget(self.radio_overwrite)
            policy_layout.addStretch()
            fields_layout.addLayout(policy_layout)

        main_layout.addWidget(fields_frame)

        # Switch UI based on single note vs multiple notes
        if len(self.note_ids) == 1:
            self._init_single_note_picker(main_layout)
            self.combo_source.currentTextChanged.connect(self._on_source_field_changed)
            self.combo_target.currentTextChanged.connect(self._check_target_media_status)
        else:
            self._init_batch_view(main_layout)

    def _init_batch_view(self, main_layout: QVBoxLayout) -> None:
        """Build UI elements for multi-note batch mode with live summary chips and stable preview table."""
        intro_label = QLabel(
            "Batch Mode: Searches Irasutoya for each card using the Source Field "
            "and inserts the illustration into the Target Field.",
            self
        )
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet("color: #98989D; font-size: 12px; margin-top: 2px;")
        main_layout.addWidget(intro_label)

        # Summary Chips Row
        self.summary_chips_layout = QHBoxLayout()
        self.summary_chips_layout.setSpacing(8)

        self.chip_fill = QLabel("● 0 to Fill", self)
        self.chip_fill.setStyleSheet(
            "background-color: rgba(48, 209, 88, 0.15); "
            "color: #30D158; "
            "font-weight: 600; font-size: 11px; "
            "padding: 3px 10px; border-radius: 10px;"
        )
        self.summary_chips_layout.addWidget(self.chip_fill)

        self.chip_other = QLabel("● 0 to Skip", self)
        self.chip_other.setStyleSheet(
            "background-color: rgba(120, 120, 128, 0.18); "
            "color: #8E8E93; "
            "font-weight: 600; font-size: 11px; "
            "padding: 3px 10px; border-radius: 10px;"
        )
        self.summary_chips_layout.addWidget(self.chip_other)
        self.summary_chips_layout.addStretch()

        main_layout.addLayout(self.summary_chips_layout)

        # Preview Table with Fixed Column Widths (Prevents Layout Jitter)
        self.table_preview = QTableWidget(self)
        self.table_preview.setColumnCount(3)
        self.table_preview.setHorizontalHeaderLabels(["Card Term (Source)", "Target Field Status", "Action"])
        self.table_preview.horizontalHeader().setStretchLastSection(False)
        self.table_preview.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_preview.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table_preview.setColumnWidth(1, 140)
        self.table_preview.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table_preview.setColumnWidth(2, 130)
        self.table_preview.verticalHeader().setVisible(False)
        self.table_preview.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table_preview.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_preview.setStyleSheet(
            "QTableWidget {"
            "   border: 1px solid rgba(120, 120, 128, 0.2);"
            "   border-radius: 6px;"
            "   background-color: palette(base);"
            "   gridline-color: rgba(120, 120, 128, 0.12);"
            "   font-size: 12px;"
            "}"
            "QHeaderView::section {"
            "   background-color: palette(alternate-base);"
            "   color: palette(text);"
            "   padding: 5px 8px;"
            "   border: none;"
            "   border-bottom: 1px solid rgba(120, 120, 128, 0.2);"
            "   font-weight: 600;"
            "   font-size: 11px;"
            "}"
        )
        main_layout.addWidget(self.table_preview)

        self.table_info_label = QLabel(self)
        self.table_info_label.setStyleSheet("color: #98989D; font-size: 11px;")
        main_layout.addWidget(self.table_info_label)

        # Action Buttons
        button_layout = QHBoxLayout()
        self.backend_lbl = QLabel(self._get_backend_status_text(), self)
        self.backend_lbl.setStyleSheet("color: #8E8E93; font-size: 11px;")
        button_layout.addWidget(self.backend_lbl)
        self._schedule_backend_status_refresh()
        button_layout.addStretch()

        btn_cancel = QPushButton("Cancel", self)
        btn_cancel.setObjectName("secondaryBtn")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)

        self.btn_batch = QPushButton(f"Start Batch Fill ({len(self.note_ids)})", self)
        self.btn_batch.setObjectName("primaryBtn")
        self.btn_batch.setDefault(True)
        self.btn_batch.clicked.connect(self._start_batch)
        button_layout.addWidget(self.btn_batch)

        main_layout.addLayout(button_layout)

        # Debounced preview updates
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(50)
        self._preview_timer.timeout.connect(self._do_update_batch_preview_table)

        self._do_update_batch_preview_table()

        # Connect field and policy changes to update table and chips in real-time
        self.combo_source.currentTextChanged.connect(self._schedule_batch_preview_update)
        self.combo_target.currentTextChanged.connect(self._schedule_batch_preview_update)
        if hasattr(self, "radio_skip"):
            self.radio_skip.toggled.connect(self._schedule_batch_preview_update)
        if hasattr(self, "radio_overwrite"):
            self.radio_overwrite.toggled.connect(self._schedule_batch_preview_update)

    def _schedule_batch_preview_update(self, *args) -> None:
        """Schedule a debounced preview table update."""
        if hasattr(self, "_preview_timer"):
            self._preview_timer.start()
        else:
            self._do_update_batch_preview_table()

    def _update_batch_preview_table(self, *args) -> None:
        """Immediately update batch preview table."""
        self._do_update_batch_preview_table()

    def _do_update_batch_preview_table(self) -> None:
        """Populate preview table showing sample notes, planned actions, and live summary chips."""
        if not hasattr(self, "table_preview"):
            return
        if not mw or not hasattr(mw, "col") or not mw.col:
            return

        source_field = self.combo_source.currentText()
        target_field = self.combo_target.currentText()
        is_overwrite = hasattr(self, "radio_overwrite") and self.radio_overwrite.isChecked()

        # Calculate counts across all selected notes using cached notes
        count_fill = 0
        count_overwrite = 0
        count_skip = 0

        for nid in self.note_ids:
            try:
                note = self._get_cached_note(nid)
                if not note or not target_field or target_field not in note or not source_field or source_field not in note:
                    count_skip += 1
                    continue
                raw_src = note[source_field]
                candidates = generate_search_candidates(raw_src, note=note, allow_tier2=True)
                has_target = bool(note[target_field].strip())
                if not candidates:
                    count_skip += 1
                elif not has_target:
                    count_fill += 1
                elif is_overwrite:
                    count_overwrite += 1
                else:
                    count_skip += 1
            except Exception:
                count_skip += 1

        # Update Summary Chips
        if hasattr(self, "chip_fill") and hasattr(self, "chip_other"):
            self.chip_fill.setText(f"● {count_fill} to Fill")
            if is_overwrite:
                self.chip_other.setText(f"● {count_overwrite} to Overwrite")
                self.chip_other.setStyleSheet(
                    "background-color: rgba(10, 132, 255, 0.15); "
                    "color: #0A84FF; "
                    "font-weight: 600; font-size: 11px; "
                    "padding: 3px 10px; border-radius: 10px;"
                )
            else:
                self.chip_other.setText(f"● {count_skip} to Skip")
                self.chip_other.setStyleSheet(
                    "background-color: rgba(120, 120, 128, 0.18); "
                    "color: #8E8E93; "
                    "font-weight: 600; font-size: 11px; "
                    "padding: 3px 10px; border-radius: 10px;"
                )
            self.chip_other.adjustSize()

        # Update Smart CTA button
        active_total = count_fill + (count_overwrite if is_overwrite else 0)
        if hasattr(self, "btn_batch"):
            if source_field and target_field and source_field == target_field:
                self.btn_batch.setText("Source and Target Fields Must Differ")
                self.btn_batch.setEnabled(False)
            elif active_total == 0:
                self.btn_batch.setText("Nothing to Fill (All Skipped)")
                self.btn_batch.setEnabled(False)
            else:
                self.btn_batch.setText(f"Start Batch Fill ({active_total})")
                self.btn_batch.setEnabled(True)

        max_preview = 30
        preview_notes = self.note_ids[:max_preview]

        self.table_preview.setRowCount(len(preview_notes))

        for row_idx, nid in enumerate(preview_notes):
            try:
                note = self._get_cached_note(nid)
                if not note:
                    item_src = QTableWidgetItem(" (Failed to load)")
                    item_tgt = QTableWidgetItem("Unavailable")
                    item_act = QTableWidgetItem("Will Skip")
                    item_src.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                    item_tgt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item_act.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item_act.setForeground(QColor("#8E8E93"))
                    self.table_preview.setItem(row_idx, 0, item_src)
                    self.table_preview.setItem(row_idx, 1, item_tgt)
                    self.table_preview.setItem(row_idx, 2, item_act)
                    continue

                if not target_field or target_field not in note or not source_field or source_field not in note:
                    item_src = QTableWidgetItem(" (Missing Field)")
                    item_tgt = QTableWidgetItem("Missing Field")
                    item_act = QTableWidgetItem("Will Skip")
                    item_src.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                    item_tgt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item_act.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item_act.setForeground(QColor("#8E8E93"))
                    self.table_preview.setItem(row_idx, 0, item_src)
                    self.table_preview.setItem(row_idx, 1, item_tgt)
                    self.table_preview.setItem(row_idx, 2, item_act)
                    continue

                raw_src = note[source_field]
                candidates = generate_search_candidates(raw_src, note=note, allow_tier2=True)
                query = candidates[0][0] if candidates else (clean_search_query(raw_src) or "(Empty)")

                has_target = bool(note[target_field].strip())
                target_status = "Has Content" if has_target else "Empty"

                if not candidates:
                    action_text = "Will Skip"
                    color_hex = "#8E8E93"  # Muted secondary gray
                elif not has_target:
                    action_text = "Will Fill"
                    color_hex = "#30D158"  # Apple HIG vibrant green
                elif is_overwrite:
                    action_text = "Will Overwrite"
                    color_hex = "#0A84FF"  # Apple HIG system blue
                else:
                    action_text = "Will Skip"
                    color_hex = "#8E8E93"  # Muted secondary gray

                item_src = QTableWidgetItem(f" {query}")
                item_tgt = QTableWidgetItem(target_status)
                item_act = QTableWidgetItem(action_text)

                item_src.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                item_tgt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_act.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                item_act.setForeground(QColor(color_hex))

                self.table_preview.setItem(row_idx, 0, item_src)
                self.table_preview.setItem(row_idx, 1, item_tgt)
                self.table_preview.setItem(row_idx, 2, item_act)
            except Exception:
                item_src = QTableWidgetItem(" (Error)")
                item_tgt = QTableWidgetItem("Unavailable")
                item_act = QTableWidgetItem("Will Skip")
                item_src.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                item_tgt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_act.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_act.setForeground(QColor("#8E8E93"))
                self.table_preview.setItem(row_idx, 0, item_src)
                self.table_preview.setItem(row_idx, 1, item_tgt)
                self.table_preview.setItem(row_idx, 2, item_act)

        total = len(self.note_ids)
        if total > max_preview:
            self.table_info_label.setText(f"Showing first {max_preview} of {total} selected cards.")
        else:
            self.table_info_label.setText(f"Showing all {total} selected cards.")

    def _init_single_note_picker(self, main_layout: QVBoxLayout) -> None:
        """Build interactive visual thumbnail picker for a single note."""
        search_box_layout = QHBoxLayout()
        search_box_layout.setSpacing(8)

        search_label = QLabel("Search Query:", self)
        search_label.setStyleSheet("font-weight: 500; font-size: 12px;")
        search_box_layout.addWidget(search_label)
        self._card_source_term: str = ""
        self._card_candidates: List[Tuple[str, str, int]] = []

        initial_query = ""
        source_field = self.combo_source.currentText()
        if self.first_note and source_field in self.first_note:
            raw_val = self.first_note[source_field]
            self._card_source_term = clean_search_query(raw_val)
            initial_query = self._card_source_term
            if raw_val:
                self._card_candidates = generate_search_candidates(
                    raw_val, note=self.first_note, allow_tier2=True
                )

        self.query_input = QLineEdit(initial_query, self)
        self.query_input.setPlaceholderText("Enter word or illustration topic...")
        self.query_input.setClearButtonEnabled(True)
        self.query_input.returnPressed.connect(self._do_search)
        search_box_layout.addWidget(self.query_input)

        self.btn_search = QPushButton("Search", self)
        self.btn_search.setObjectName("secondaryBtn")
        self.btn_search.clicked.connect(self._do_search)
        search_box_layout.addWidget(self.btn_search)

        main_layout.addLayout(search_box_layout)

        # Suggestion Chips Scroll Area & Container
        self.chips_scroll = QScrollArea(self)
        self.chips_scroll.setWidgetResizable(True)
        self.chips_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.chips_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chips_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chips_scroll.setFixedHeight(38)
        self.chips_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:horizontal {"
            "    height: 4px;"
            "    background: transparent;"
            "    margin: 0px;"
            "    border-radius: 2px;"
            "}"
            "QScrollBar::handle:horizontal {"
            "    background: rgba(120, 120, 128, 0.35);"
            "    min-width: 24px;"
            "    border-radius: 2px;"
            "}"
            "QScrollBar::handle:horizontal:hover {"
            "    background: rgba(120, 120, 128, 0.6);"
            "}"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {"
            "    width: 0px;"
            "    height: 0px;"
            "    background: none;"
            "}"
            "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {"
            "    background: none;"
            "}"
        )
        self.chips_scroll.setVisible(False)

        self.chips_widget = QWidget()
        self.chips_widget.setStyleSheet("background: transparent;")
        self.chips_layout = QHBoxLayout(self.chips_widget)
        self.chips_layout.setContentsMargins(0, 0, 0, 4)
        self.chips_layout.setSpacing(6)
        self.chips_scroll.setWidget(self.chips_widget)
        self.chips_widget.setVisible(False)
        main_layout.addWidget(self.chips_scroll)

        # Status Label
        self.status_label = QLabel("Search for an illustration or click 'Search' above.", self)
        self.status_label.setStyleSheet("color: #98989D; font-size: 12px;")
        main_layout.addWidget(self.status_label)

        # Gallery Scroll Area with Right Margin Gutter
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(
            "QScrollArea {"
            "   border: 1px solid rgba(120, 120, 128, 0.2);"
            "   border-radius: 8px;"
            "   background-color: palette(base);"
            "}"
        )

        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_layout.setContentsMargins(10, 10, 16, 10)
        self.gallery_layout.setSpacing(10)
        self.scroll_area.setWidget(self.gallery_widget)
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll)

        main_layout.addWidget(self.scroll_area)

        # Pagination Progress Counter
        self.pagination_label = QLabel(self)
        self.pagination_label.setStyleSheet("color: #8E8E93; font-size: 11px;")
        self.pagination_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.pagination_label)

        # Media status notice if card already contains an image
        self.media_notice_label = QLabel(self)
        self.media_notice_label.setStyleSheet("color: #8E8E93; font-size: 11px;")
        self.media_notice_label.setVisible(False)
        main_layout.addWidget(self.media_notice_label)

        # Single Note Action Buttons
        button_layout = QHBoxLayout()
        self.backend_lbl = QLabel(self._get_backend_status_text(), self)
        self.backend_lbl.setStyleSheet("color: #8E8E93; font-size: 11px;")
        button_layout.addWidget(self.backend_lbl)
        self._schedule_backend_status_refresh()
        button_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setObjectName("secondaryBtn")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)

        self.btn_apply = QPushButton("Apply Selected Image", self)
        self.btn_apply.setObjectName("primaryBtn")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self._apply_single_selection)
        button_layout.addWidget(self.btn_apply)

        main_layout.addLayout(button_layout)

        self._check_target_media_status()

        # Auto-trigger search asynchronously after dialog displays
        if initial_query:
            QTimer.singleShot(50, self._do_search)

    def _check_target_media_status(self, *args) -> None:
        """Inspect note's target field for existing media and adapt cancel button label."""
        if not hasattr(self, "btn_cancel") or not self.btn_cancel:
            return
        target_field = self.combo_target.currentText() if hasattr(self, "combo_target") else ""
        has_media = False
        if self.first_note and target_field and target_field in self.first_note:
            content = self.first_note[target_field].strip()
            content_lower = content.lower()
            if "<img" in content_lower or any(
                content_lower.endswith(ext) for ext in (".jpg", ".png", ".webp", ".jpeg", ".gif")
            ):
                has_media = True

        if has_media:
            self.btn_cancel.setText("Keep Current Image (Esc)")
            self.btn_cancel.setToolTip("Close dialog without changing the existing image on this card")
            if hasattr(self, "media_notice_label") and self.media_notice_label:
                self.media_notice_label.setText(
                    "ℹ️ Card already has an image. It will remain unchanged unless you apply a new one."
                )
                self.media_notice_label.setVisible(True)
        else:
            self.btn_cancel.setText("Cancel")
            self.btn_cancel.setToolTip("")
            if hasattr(self, "media_notice_label") and self.media_notice_label:
                self.media_notice_label.setVisible(False)

    def _check_viewport_pagination(self) -> None:
        """Ensure gallery content overflows the scroll viewport when more results exist."""
        if not hasattr(self, "scroll_area") or not hasattr(self, "gallery_layout") or not self.isVisible():
            return
        viewport_h = self.scroll_area.viewport().height()
        if viewport_h <= 0:
            return
        while (
            self.gallery_layout.sizeHint().height() <= viewport_h
            and self._loaded_count < len(self._all_images)
        ):
            prev_count = self._loaded_count
            self._load_next_page(self._search_generation)
            if self._loaded_count == prev_count:
                break

    def _on_scroll(self, value: int) -> None:
        """Dynamically load more results when user scrolls near the bottom."""
        if not hasattr(self, "scroll_area"):
            return
        bar = self.scroll_area.verticalScrollBar()
        if bar.maximum() > 0 and value >= bar.maximum() - 250:
            if self._loaded_count < len(self._all_images):
                self._load_next_page(self._search_generation)

    def _on_source_field_changed(self, new_field: str) -> None:
        """Update query input when source field selection changes in single note mode."""
        if hasattr(self, "query_input") and self.first_note and new_field in self.first_note:
            raw_val = self.first_note[new_field]
            new_query = clean_search_query(raw_val)
            self._card_source_term = new_query
            self._card_candidates = (
                generate_search_candidates(raw_val, note=self.first_note, allow_tier2=True)
                if raw_val else []
            )
            self.query_input.setText(new_query)
            if new_query:
                self._do_search()
            else:
                self._search_generation += 1
                self.btn_search.setEnabled(True)
                self.btn_search.setText("Search")
                self._clear_gallery()
                self._update_suggestion_chips([], active_term="")
                self.status_label.setStyleSheet("color: #8E8E93; font-size: 12px;")
                self.status_label.setText("Source field is empty.")

    def _do_search(self, direct: bool = False) -> None:
        """Fetch search results from Irasutoya with fallback candidate generation asynchronously.

        Args:
            direct: If True, execute a direct query for the exact search term without
                traversing alternative fallback inflections (used when clicking a specific chip).
        """
        query = self.query_input.text().strip()
        if not query:
            self.status_label.setStyleSheet("color: #FF453A; font-size: 12px;")
            self.status_label.setText("Please enter a query term.")
            return

        self._search_generation += 1
        current_gen = self._search_generation

        self.status_label.setStyleSheet("color: palette(text); font-size: 12px; font-weight: 500;")
        self.status_label.setText(f"Searching Irasutoya for '{query}'...")
        self.btn_search.setEnabled(False)
        self._clear_gallery()

        image_size = self.config.get(CONFIG_KEY_IMAGE_SIZE, "s800")
        card_terms = [c[0] for c in getattr(self, "_card_candidates", [])]
        is_card_term = bool(self._card_source_term and (query == self._card_source_term or query in card_terms))

        if is_card_term:
            display_chips = self._card_candidates
            restore_term = None
            if direct:
                matching_cand = next((c for c in self._card_candidates if c[0] == query), (query, "primary", 1))
                search_candidates = [matching_cand]
            elif query == self._card_source_term:
                search_candidates = self._card_candidates
            else:
                search_candidates = generate_search_candidates(query, note=self.first_note, allow_tier2=True)
        else:
            candidates = generate_search_candidates(query, note=None, allow_tier2=True)
            display_chips = candidates
            matching_cand = next((c for c in candidates if c[0] == query), (query, "primary", 1))
            search_candidates = [matching_cand] if direct else candidates
            restore_term = self._card_source_term if (self._card_source_term and self._card_source_term != query) else None

        def background_search() -> None:
            """Fetch search results on a background thread using fallback queries."""
            probed_counts: Optional[Dict[str, int]] = None
            probed_desc_matches: Optional[Set[str]] = None
            try:
                images, matched_cand, _best_img = self.client.search_with_fallbacks(
                    search_candidates,
                    max_results=None,
                    image_size=image_size,
                    is_cancelled=lambda: current_gen != self._search_generation,
                )
            except Exception as e:
                logger.exception("Search failed: %s", e)
                images, matched_cand, _best_img = [], None, None

            if current_gen != self._search_generation:
                return

            try:
                catalog = self.store.get_index()
                if catalog is not None:
                    count_cache = self._catalog_count_cache.setdefault(image_size, {})
                    desc_cache = self._catalog_desc_cache.setdefault(image_size, {})

                    # Reuse already retrieved and filtered results for the matched candidate
                    if matched_cand and images:
                        count_cache[matched_cand[0]] = len(images)
                        desc_cache[matched_cand[0]] = _is_desc_only_match(images, matched_cand[0])

                    def _count(term: str) -> int:
                        if current_gen != self._search_generation:
                            raise _StaleSearch()
                        if term not in count_cache:
                            raw_imgs = catalog.search(term, image_size=image_size)
                            rel_imgs = [
                                img for img in raw_imgs
                                if not is_decorative_asset(img, term) and is_image_relevant(img, term)
                            ]
                            count_cache[term] = len(rel_imgs)
                            desc_cache[term] = _is_desc_only_match(rel_imgs, term)
                        return count_cache[term]

                    probe_terms = [c[0] for c in display_chips]
                    if matched_cand and matched_cand[0] not in probe_terms:
                        probe_terms.append(matched_cand[0])
                    counts: Dict[str, int] = {t: _count(t) for t in probe_terms}
                    if restore_term:
                        counts[restore_term] = _count(restore_term)
                    probed_counts = counts
                    probed_desc_matches = {t for t in counts if desc_cache.get(t)}
            except _StaleSearch:
                return
            except Exception:
                logger.exception("Catalog count probe failed")

            if current_gen != self._search_generation:
                return

            self._run_on_main(lambda: self._on_search_finished(
                current_gen,
                query,
                display_chips,
                matched_cand,
                images,
                restore_term,
                probed_counts,
                probed_desc_matches,
            ))

        threading.Thread(target=background_search, daemon=True).start()

    def _on_search_finished(
        self,
        gen: int,
        query: str,
        display_chips: List[Tuple[str, str, int]],
        matched_cand: Optional[Tuple[str, str, int]],
        images: List[IrasutoyaImage],
        restore_term: Optional[str] = None,
        probed_counts: Optional[Dict[str, int]] = None,
        probed_desc_matches: Optional[Set[str]] = None,
    ) -> None:
        """Process completed search results on the main thread, display fallbacks, and populate gallery."""
        if gen != self._search_generation:
            return

        self.btn_search.setEnabled(True)
        active_term = matched_cand[0] if (matched_cand and images) else query
        self._current_query = active_term

        if not images:
            self.status_label.setStyleSheet("color: #FF453A; font-size: 12px;")
            self.status_label.setText(f"No illustrations found for '{query}'.")
            self._update_suggestion_chips(
                display_chips,
                active_term=active_term,
                restore_term=restore_term,
                probed_counts=probed_counts,
                probed_desc_matches=probed_desc_matches,
            )
            if hasattr(self, "pagination_label") and self.pagination_label:
                self.pagination_label.setText("")
            return

        if matched_cand and matched_cand[0] != query:
            self.status_label.setStyleSheet("color: #0A84FF; font-size: 12px; font-weight: 500;")
            count = probed_counts.get(query) if probed_counts else None
            if count is None:
                reason_text = f"No direct title match for '{query}'"
            elif count == 0:
                reason_text = f"No direct results for '{query}'"
            elif probed_desc_matches and query in probed_desc_matches:
                reason_text = f"No direct title match for '{query}'"
            elif probed_desc_matches is not None and count > 0:
                reason_text = f"No direct title illustration for '{query}'"
            else:
                reason_text = f"No direct title match for '{query}'"
            self.status_label.setText(
                f"ℹ️ {reason_text}. Showing results for '{matched_cand[0]}' ({matched_cand[1]})."
            )
        else:
            self.status_label.setStyleSheet("color: #98989D; font-size: 12px;")
            if probed_desc_matches is not None:
                all_desc = query in probed_desc_matches
            else:
                all_desc = _is_desc_only_match(images, query)
            count_str = f"{len(images)} illustration" if len(images) == 1 else f"{len(images)} illustrations"
            if all_desc:
                self.status_label.setText(f"Found {count_str} (description match). Click one to select.")
            else:
                self.status_label.setText(f"Found {count_str}. Click one to select.")

        self._update_suggestion_chips(
            display_chips,
            active_term=active_term,
            restore_term=restore_term,
            probed_counts=probed_counts,
            probed_desc_matches=probed_desc_matches,
        )
        self._populate_gallery(images, gen)

    def _update_suggestion_chips(
        self,
        candidates: List[Tuple[str, str, int]],
        active_term: Optional[str] = None,
        restore_term: Optional[str] = None,
        probed_counts: Optional[Dict[str, int]] = None,
        probed_desc_matches: Optional[Set[str]] = None,
    ) -> None:
        """Render one-click suggestion chips for query variations with instant count badges."""
        if not hasattr(self, "chips_layout"):
            return

        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        probed_candidates: List[Tuple[str, str, int, Optional[int]]] = []
        for cand in candidates:
            cand_term, reason, tier = cand[0], cand[1], cand[2]
            if probed_counts is not None:
                cnt = probed_counts.get(cand_term)
                if not cnt:
                    continue  # Zero-result or unprobed suppression
                probed_candidates.append((cand_term, reason, tier, cnt))
            else:
                probed_candidates.append((cand_term, reason, tier, None))

        restore_count = probed_counts.get(restore_term) if (probed_counts and restore_term) else None

        # Only display chips if there are alternative candidates or a restore button
        has_alternatives = len(probed_candidates) > 1 or (
            len(probed_candidates) == 1 and probed_candidates[0][0] != active_term
        )
        if not has_alternatives and not restore_term:
            self.chips_widget.setVisible(False)
            if hasattr(self, "chips_scroll"):
                self.chips_scroll.setVisible(False)
            return

        lbl = QLabel("Try:", self)
        lbl.setStyleSheet("color: #8E8E93; font-size: 11px; font-weight: 500;")
        self.chips_layout.addWidget(lbl)

        # Prepend restore chip if searching an external custom query
        if restore_term:
            is_res_desc = bool(probed_desc_matches and restore_term in probed_desc_matches)
            res_text = (
                f"↩ {restore_term}  {restore_count}"
                if (restore_count is not None and restore_count > 0)
                else f"↩ {restore_term}"
            )
            btn_restore = QPushButton(res_text, self)
            btn_restore.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            if restore_count is not None and restore_count > 0:
                if is_res_desc:
                    match_str = (
                        f"{restore_count} description match / 本文一致"
                        if restore_count == 1
                        else f"{restore_count} description matches / 本文一致"
                    )
                    btn_restore.setToolTip(f"Reset search to card term '{restore_term}' ({match_str})")
                else:
                    cnt_str = (
                        f"{restore_count} illustration"
                        if restore_count == 1
                        else f"{restore_count} illustrations"
                    )
                    btn_restore.setToolTip(f"Reset search to card term '{restore_term}' ({cnt_str})")
            else:
                btn_restore.setToolTip(f"Reset search to card term '{restore_term}'")
            btn_restore.setStyleSheet(
                "background-color: rgba(10, 132, 255, 0.12); "
                "border: 1px dashed #0A84FF; border-radius: 10px; "
                "padding: 2px 8px; font-size: 11px; font-weight: 500; color: #0A84FF;"
            )
            btn_restore.clicked.connect(lambda _, t=restore_term: self._on_restore_clicked(t))
            self.chips_layout.addWidget(btn_restore)

        # Show up to MAX_SUGGESTION_CHIPS candidate chips, ensuring active_term is visible if outside top N
        max_cands = MAX_SUGGESTION_CHIPS - 1 if restore_term else MAX_SUGGESTION_CHIPS
        displayed_candidates: List[Tuple[str, str, int, Optional[int]]] = []
        active_cand = next((c for c in probed_candidates if c[0] == active_term), None)
        if active_cand and active_cand not in probed_candidates[:max_cands]:
            displayed_candidates = probed_candidates[:max_cands - 1] + [active_cand]
        else:
            displayed_candidates = probed_candidates[:max_cands]

        for cand_term, _reason, _tier, count in displayed_candidates:
            is_active = (cand_term == active_term)
            btn_text = f"{cand_term}  {count}" if (count is not None and count > 0) else cand_term
            btn = QPushButton(btn_text, self)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            is_desc = bool(probed_desc_matches and cand_term in probed_desc_matches)
            if count is not None and count > 0:
                if is_desc:
                    match_str = (
                        f"{count} description match / 本文一致"
                        if count == 1
                        else f"{count} description matches / 本文一致"
                    )
                    btn.setToolTip(f"Search for '{cand_term}' ({match_str})")
                else:
                    count_str = f"{count} illustration" if count == 1 else f"{count} illustrations"
                    btn.setToolTip(f"Search for '{cand_term}' ({count_str})")
            else:
                btn.setToolTip(f"Search for '{cand_term}'")
            if is_active:
                btn.setStyleSheet(
                    "background-color: rgba(10, 132, 255, 0.18); "
                    "border: 1px solid #0A84FF; border-radius: 10px; "
                    "padding: 2px 8px; font-size: 11px; font-weight: 600; color: #0A84FF;"
                )
            else:
                btn.setStyleSheet(
                    "background-color: rgba(120, 120, 128, 0.15); "
                    "border: 1px solid rgba(120, 120, 128, 0.2); border-radius: 10px; "
                    "padding: 2px 8px; font-size: 11px; color: palette(text);"
                )
            btn.clicked.connect(lambda _, t=cand_term: self._on_chip_clicked(t))
            self.chips_layout.addWidget(btn)

        self.chips_layout.addStretch()
        self.chips_widget.setVisible(True)
        if hasattr(self, "chips_scroll"):
            self.chips_scroll.setVisible(True)

    def _on_chip_clicked(self, term: str) -> None:
        """Handle clicking a suggestion chip to switch search query directly."""
        if hasattr(self, "query_input"):
            self.query_input.setText(term)
            self._do_search(direct=True)

    def _on_restore_clicked(self, term: str) -> None:
        """Handle clicking the restore chip to reset search to original card query."""
        if hasattr(self, "query_input"):
            self.query_input.setText(term)
            self._do_search(direct=False)

    def _clear_gallery(self) -> None:
        """Remove all cards from the gallery."""
        self._cancel_pending_thumb_futures()
        self.selected_card_widget = None
        self.image_cards.clear()
        self._all_images = []
        self._loaded_count = 0
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if hasattr(self, "btn_apply"):
            self.btn_apply.setEnabled(False)
        if hasattr(self, "pagination_label") and self.pagination_label:
            self.pagination_label.setText("")

    def _populate_gallery(self, images: List[IrasutoyaImage], gen: int) -> None:
        """Initialize gallery with search results and render the first page."""
        self._thumb_burst_remaining = THUMB_INITIAL_BURST
        self._all_images = list(images)
        self._loaded_count = 0
        self._load_next_page(gen)
        self._check_viewport_pagination()
        # Select first card by default
        if self.image_cards:
            self._on_card_selected(self.image_cards[0])

    def _load_next_page(self, gen: int) -> None:
        """Render next chunk of image cards into the gallery grid."""
        if gen != self._search_generation or not self._all_images:
            return
        if self._loaded_count >= len(self._all_images):
            return

        batch_size = PAGE_SIZE
        start_idx = self._loaded_count
        end_idx = min(start_idx + batch_size, len(self._all_images))
        batch = self._all_images[start_idx:end_idx]
        if not batch:
            return

        cols = 3
        try:
            thumb_delay_ms = int(self.config.get(CONFIG_KEY_THUMB_DELAY, 50))
        except (TypeError, ValueError):
            thumb_delay_ms = 50

        query = getattr(self, "_current_query", "").strip()

        for offset, img in enumerate(batch):
            idx = start_idx + offset
            # Classify description-only match
            is_desc_match = False
            snippet = None
            if query:
                match_term = _description_match_term(img, query)
                if match_term:
                    is_desc_match = True
                    snippet = make_description_snippet(img.description or "", match_term)

            card = ImageCardWidget(
                image=img,
                on_select=self._on_card_selected,
                on_double_click=self._on_card_double_clicked,
                parent=self.gallery_widget,
                is_description_match=is_desc_match,
                match_snippet=snippet,
            )
            row = idx // cols
            col = idx % cols
            self.gallery_layout.addWidget(card, row, col)
            self.image_cards.append(card)

            # Check cache first for instant 0ms rendering
            cached_pixmap = self._get_cached_pixmap(img.thumb_url)
            if cached_pixmap is not None:
                card.set_scaled_pixmap(cached_pixmap)
            else:
                fut = self._thumb_executor.submit(
                    self._fetch_thumbnail_task, card, gen, thumb_delay_ms
                )
                self._pending_thumb_futures.append(fut)

        self._loaded_count = end_idx
        self.gallery_layout.activate()
        self._update_pagination_status()

    def _update_pagination_status(self) -> None:
        """Update footer counter showing pagination progress."""
        if not hasattr(self, "pagination_label") or not self.pagination_label:
            return
        total = len(self._all_images)
        if total == 0:
            self.pagination_label.setText("")
            return
        if self._loaded_count < total:
            self.pagination_label.setText(
                f"Showing {self._loaded_count} of {total} illustrations · Scroll for more"
            )
        else:
            self.pagination_label.setText(f"Showing all {total} illustrations")

    def _fetch_thumbnail_task(self, card: ImageCardWidget, gen: int, delay_ms: int = 50) -> None:
        """Background thread worker to download a thumbnail with fast pacing."""
        if gen != self._search_generation:
            return
        delay_sec = max(0.0, delay_ms / 1000.0)
        thumb_url = card.image.thumb_url

        # Check if already cached (e.g. from an in-flight download that just finished)
        cached_data = self._thumb_byte_cache.get(thumb_url)
        if cached_data:
            self._process_and_apply_thumbnail(card, gen, cached_data)
            return

        # In-flight deduplication: coalesce duplicate requests for the same URL
        event_to_wait = None
        is_downloader = False
        with self._thumb_inflight_lock:
            if thumb_url in self._thumb_inflight:
                event_to_wait = self._thumb_inflight[thumb_url]
            else:
                event = threading.Event()
                self._thumb_inflight[thumb_url] = event
                is_downloader = True

        if not is_downloader and event_to_wait is not None:
            # Wait for the active downloader to finish
            event_to_wait.wait(timeout=8.5)
            if gen != self._search_generation:
                return
            data = self._thumb_byte_cache.get(thumb_url) or b""
            self._process_and_apply_thumbnail(card, gen, data)
            return

        # Active downloader: rate-limit slot reservation
        try:
            with self._thumb_lock:
                if gen != self._search_generation:
                    return
                now = time.monotonic()
                if self._thumb_burst_remaining > 0:
                    self._thumb_burst_remaining -= 1
                    wait_sec = 0.0
                    self._next_thumb_request_time = max(now, self._next_thumb_request_time)
                else:
                    scheduled = max(now, self._next_thumb_request_time)
                    wait_sec = max(0.0, scheduled - now)
                    self._next_thumb_request_time = scheduled + delay_sec

            if wait_sec > 0:
                time.sleep(wait_sec)

            if gen != self._search_generation:
                return

            try:
                thumb_data = self.client.download_image(thumb_url, timeout=8, image=card.image)
            except Exception:
                thumb_data = b""

            # SAVE-BEFORE-GENERATION-CHECK: Cache valid bytes immediately
            if thumb_data:
                self._thumb_byte_cache.put(thumb_url, thumb_data)

            self._process_and_apply_thumbnail(card, gen, thumb_data)
        finally:
            with self._thumb_inflight_lock:
                ev = self._thumb_inflight.pop(thumb_url, None)
                if ev is not None:
                    ev.set()

    def _process_and_apply_thumbnail(
        self, card: ImageCardWidget, gen: int, thumb_data: bytes
    ) -> None:
        """Decode and scale thumbnail on background worker, then marshal to main thread."""
        scaled_qimg: Optional[QImage] = None
        if thumb_data:
            qimg = QImage.fromData(QByteArray(thumb_data))
            if not qimg.isNull():
                scaled_qimg = qimg.scaled(
                    120,
                    96,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

        self._run_on_main(lambda: self._apply_thumbnail(card, gen, thumb_data, scaled_qimg))

    def _apply_thumbnail(
        self,
        card: ImageCardWidget,
        gen: int,
        thumb_data: bytes,
        scaled_qimg: Optional[QImage] = None,
    ) -> None:
        """Main thread callback to apply thumbnail pixmap and populate dialog cache."""
        thumb_url = card.image.thumb_url
        if thumb_data:
            pixmap = self._thumb_pixmap_cache.get(thumb_url)
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
                    self._cache_dialog_pixmap(thumb_url, pixmap)

            if gen == self._search_generation:
                if pixmap is not None and not pixmap.isNull():
                    card.set_scaled_pixmap(pixmap)
                    return
                card.thumb_label.setText("No Preview")
            return

        if gen == self._search_generation:
            card.thumb_label.setText("No Preview")

    def _on_card_selected(self, card: ImageCardWidget) -> None:
        """Highlight selected card, display checkmark badge, and enable Apply button."""
        for c in self.image_cards:
            c.set_selected(c == card)
        self.selected_card_widget = card
        if hasattr(self, "btn_apply") and not self._applying:
            source_field = self.combo_source.currentText()
            target_field = self.combo_target.currentText()
            if source_field and target_field and source_field == target_field:
                self.btn_apply.setEnabled(False)
            else:
                self.btn_apply.setEnabled(True)

    def _on_card_double_clicked(self, card: ImageCardWidget) -> None:
        """Double click selects and immediately applies."""
        self._on_card_selected(card)
        self._apply_single_selection()

    def _save_preferences(self) -> None:
        """Persist user's field and overwrite selections."""
        source_field = self.combo_source.currentText()
        target_field = self.combo_target.currentText()
        if hasattr(self, "radio_overwrite"):
            overwrite_mode = "overwrite" if self.radio_overwrite.isChecked() else "skip"
        else:
            overwrite_mode = self.config.get(CONFIG_KEY_OVERWRITE, "skip")
        update_user_preferences(source_field, target_field, overwrite_mode)

    def _apply_single_selection(self) -> None:
        """Download high-res image in background and assign to note without blocking on existing content."""
        if not self.selected_card_widget or not self.first_note or self._applying:
            return

        source_field = self.combo_source.currentText()
        target_field = self.combo_target.currentText()
        if source_field and target_field and source_field == target_field:
            showWarning("Source and target fields cannot be the same.", parent=self)
            return

        if not target_field or target_field not in self.first_note:
            showWarning(f"The selected target field '{target_field}' does not exist on this note.", parent=self)
            return

        self._applying = True
        self._save_preferences()

        best_img = self.selected_card_widget.image
        query = self.query_input.text().strip() or "image"

        self.btn_apply.setEnabled(False)
        self.status_label.setStyleSheet("color: palette(text); font-size: 12px; font-weight: 500;")
        self.status_label.setText(f"Downloading '{best_img.title}'...")

        def background_apply() -> None:
            """Background worker to download the chosen high-res illustration."""
            try:
                img_bytes = self.client.download_image(best_img.image_url, image=best_img)
                if not img_bytes:
                    raise ValueError("Received empty image data from server")

                def on_success() -> None:
                    """Commit media and update note field within a QueryOp on the collection."""
                    if not self.isVisible() or not self._applying:
                        self._applying = False
                        return

                    filename = make_safe_media_filename(query, best_img.title)

                    def commit_single(col: any) -> any:
                        """Write media and assign field to fresh note."""
                        fresh_note = col.get_note(self.first_note.id)
                        if target_field not in fresh_note:
                            raise KeyError(f"Field '{target_field}' no longer exists on this note")
                        stored_filename = col.media.write_data(filename, img_bytes)
                        encoded_src = urllib.parse.quote(stored_filename)
                        fresh_note[target_field] = f'<img src="{encoded_src}">'
                        return fresh_note

                    def on_committed(fresh_note: any) -> None:
                        """Update notes and close dialog upon successful commit."""
                        def on_update_success(_: any) -> None:
                            """Notify success and close dialog."""
                            tooltip(f"Added '{best_img.title}' to {target_field}.", parent=self.browser)
                            if self.isVisible():
                                self.accept()

                        def on_update_failure(err: Exception) -> None:
                            """Restore button state and display warning on update failure."""
                            self._applying = False
                            if self.isVisible():
                                self.btn_apply.setEnabled(True)
                                self.status_label.setStyleSheet("color: #FF453A; font-size: 12px;")
                                self.status_label.setText(f"Error updating note: {err}")
                                showWarning(f"Failed to update note: {err}", parent=self)

                        import aqt.operations.note
                        op = aqt.operations.note.update_notes(parent=self.browser, notes=[fresh_note])
                        op.success(on_update_success)
                        op.failure(on_update_failure)
                        op.run_in_background()

                    def on_commit_error(err: Exception) -> None:
                        """Handle commit failure and restore button state."""
                        self._applying = False
                        if self.isVisible():
                            self.btn_apply.setEnabled(True)
                            self.status_label.setStyleSheet("color: #FF453A; font-size: 12px;")
                            self.status_label.setText(f"Error applying image: {err}")
                            showWarning(f"Failed to write media or update note: {err}", parent=self)

                    from aqt.operations import QueryOp
                    commit_op = QueryOp(
                        parent=self,
                        op=commit_single,
                        success=on_committed
                    )
                    commit_op.failure(on_commit_error)
                    commit_op.run_in_background()

                self._run_on_main(on_success)
            except Exception as e:
                logger.exception("Failed to apply image: %s", e)
                err_msg = str(e)

                def on_error() -> None:
                    """Handle download failure on main GUI thread and restore button state."""
                    self._applying = False
                    if self.isVisible():
                        self.btn_apply.setEnabled(True)
                        self.status_label.setStyleSheet("color: #FF453A; font-size: 12px;")
                        self.status_label.setText(f"Error applying image: {err_msg}")
                        showWarning(f"Failed to download or apply image: {err_msg}", parent=self)

                self._run_on_main(on_error)

        threading.Thread(target=background_apply, daemon=True).start()

    def _start_batch(self) -> None:
        """Save preferences and launch background batch process."""
        source_field = self.combo_source.currentText()
        target_field = self.combo_target.currentText()
        if source_field == target_field:
            showWarning("Source and target fields cannot be the same.", parent=self)
            return

        self._save_preferences()
        overwrite_mode = "overwrite" if hasattr(self, "radio_overwrite") and self.radio_overwrite.isChecked() else "skip"

        self.accept()

        run_batch_fill(
            parent_window=self.browser,
            note_ids=self.note_ids,
            source_field=source_field,
            target_field=target_field,
            overwrite_mode=overwrite_mode
        )
