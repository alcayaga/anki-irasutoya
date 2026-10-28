"""Main dialog shell for mapping card fields and executing batch or interactive image insertion."""
import os
from typing import Any, Dict, List, Optional, Sequence, Set

from aqt.qt import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QImage,
    QLabel,
    QPixmap,
    QPushButton,
    QRadioButton,
    Qt,
    QTimer,
    QVBoxLayout,
)

from ..catalog import default_store
from ..client import IrasutoyaClient
from ..config import (
    CONFIG_KEY_OVERWRITE,
    CONFIG_KEY_SOURCE,
    CONFIG_KEY_TARGET,
    get_config,
    update_user_preferences,
)
from ..feed import IrasutoyaImage
from .batch_view import _BatchModeMixin
from .helpers import _get_mw
from .single_view import _SingleNoteModeMixin
from .styles import (
    DIALOG_STYLESHEET,
    FIELDS_FRAME_STYLE,
    HEADER_COUNT_BADGE_STYLE,
)
from .thumbnails import _SESSION_THUMB_CACHE, ThumbnailLoader
from .widgets import ImageCardWidget


class IrasutoyaDialog(_SingleNoteModeMixin, _BatchModeMixin, QDialog):
    """Main dialog for mapping card fields and executing batch or interactive image insertion."""

    def __init__(self, parent, browser, note_ids: Sequence[int]) -> None:
        """Initialize the dialog with selected notes and configuration."""
        super().__init__(parent)
        self.browser = browser
        self.note_ids: List[int] = list(dict.fromkeys(note_ids))
        self.config: Dict[str, Any] = get_config()
        self.store = default_store()
        self.store.warm_async()
        self.client = IrasutoyaClient(catalog=self.store)

        self.selected_card_widget: Optional[ImageCardWidget] = None
        self.image_cards: List[ImageCardWidget] = []
        self.first_note = None
        self._applying: bool = False
        self._note_cache: Dict[int, Any] = {}

        mw = _get_mw()
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

        self._search_generation: int = 0
        self._thumb_loader: ThumbnailLoader = ThumbnailLoader(session_cache=_SESSION_THUMB_CACHE)
        # Expose direct attributes for backward compatibility with existing tests and callers
        self._thumb_executor = self._thumb_loader.executor
        self._pending_thumb_futures = self._thumb_loader.pending_futures
        self._thumb_byte_cache = self._thumb_loader.byte_cache
        self._thumb_pixmap_cache = self._thumb_loader.pixmap_cache
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

    def _get_cached_note(self, nid: int) -> Optional[Any]:
        """Fetch a note with in-dialog memory caching to prevent redundant collection reads."""
        if nid not in self._note_cache:
            try:
                mw = _get_mw()
                self._note_cache[nid] = mw.col.get_note(nid) if mw and mw.col else None
            except Exception:
                self._note_cache[nid] = None
        return self._note_cache[nid]

    def _run_on_main(self, callback) -> None:
        """Execute a callable safely on the Qt main GUI thread."""
        try:
            mw = _get_mw()
            if mw and hasattr(mw, "taskman") and mw.taskman:
                mw.taskman.run_on_main(callback)
                return
        except Exception:
            pass
        QTimer.singleShot(0, callback)

    def _cancel_pending_thumb_futures(self) -> None:
        """Cancel unstarted queued thumbnail download futures."""
        self._thumb_loader.cancel_pending_futures()

    def _decode_thumbnail(self, thumb_data: bytes) -> Optional[QImage]:
        """Decode and scale thumbnail bytes into a smooth-transformed QImage, or None if invalid."""
        return self._thumb_loader.decode_thumbnail(thumb_data)

    def _get_cached_pixmap(self, thumb_url: str) -> Optional[QPixmap]:
        """Lookup pre-scaled QPixmap from dialog cache or decode from session byte cache."""
        return self._thumb_loader.get_cached_pixmap(thumb_url)

    def _cache_dialog_pixmap(self, thumb_url: str, pixmap: QPixmap) -> None:
        """Store a pre-scaled QPixmap in the dialog LRU cache."""
        self._thumb_loader.cache_dialog_pixmap(thumb_url, pixmap)

    def done(self, result: int) -> None:
        """Shut down background thumbnail executor, release caches, and close dialog."""
        self._applying = False
        self._search_generation += 1
        self._thumb_loader.shutdown()
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
        mw = _get_mw()
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
        count_badge.setStyleSheet(HEADER_COUNT_BADGE_STYLE)
        header_layout.addWidget(count_badge)
        main_layout.addLayout(header_layout)

        # Field Mapping Group
        fields_frame = QFrame(self)
        fields_frame.setStyleSheet(FIELDS_FRAME_STYLE)
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

    def _save_preferences(self) -> None:
        """Persist user's field and overwrite selections."""
        source_field = self.combo_source.currentText()
        target_field = self.combo_target.currentText()
        if hasattr(self, "radio_overwrite"):
            overwrite_mode = "overwrite" if self.radio_overwrite.isChecked() else "skip"
        else:
            overwrite_mode = self.config.get(CONFIG_KEY_OVERWRITE, "skip")
        update_user_preferences(source_field, target_field, overwrite_mode)
