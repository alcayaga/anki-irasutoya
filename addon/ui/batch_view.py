"""Batch mode view mixin and preview table controller."""
import sys
from typing import TYPE_CHECKING

from aqt.qt import (
    QAbstractItemView,
    QColor,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTimer,
    QVBoxLayout,
)
from aqt.utils import showWarning

from ..batch import run_batch_fill
from ..text_utils import clean_search_query, generate_search_candidates
from .styles import (
    BATCH_TABLE_STYLE,
    CHIP_FILL_STYLE,
    CHIP_OVERWRITE_STYLE,
    CHIP_SKIP_STYLE,
)

if TYPE_CHECKING:
    from .dialog import IrasutoyaDialog


def _get_mw() -> any:
    """Return the active Anki main window instance, respecting mock patches on addon.ui.mw."""
    ui_mod = sys.modules.get("addon.ui")
    if ui_mod is not None and hasattr(ui_mod, "mw"):
        return ui_mod.mw
    try:
        from aqt import mw
        return mw
    except Exception:
        return None


class _BatchModeMixin:
    """Mixin providing multi-card batch preview table and execution for IrasutoyaDialog."""

    if TYPE_CHECKING:
        self: "IrasutoyaDialog"

    def _init_batch_view(self, main_layout: QVBoxLayout) -> None:
        """Build UI elements for multi-note batch mode with live summary chips and stable preview table."""
        intro_label = QLabel(
            "Batch Mode: Searches Irasutoya for each card using the Source Field "
            "and inserts the illustration into the Target Field.",
            self,
        )
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet("color: #98989D; font-size: 12px; margin-top: 2px;")
        main_layout.addWidget(intro_label)

        # Summary Chips Row
        self.summary_chips_layout = QHBoxLayout()
        self.summary_chips_layout.setSpacing(8)

        self.chip_fill = QLabel("● 0 to Fill", self)
        self.chip_fill.setStyleSheet(CHIP_FILL_STYLE)
        self.summary_chips_layout.addWidget(self.chip_fill)

        self.chip_other = QLabel("● 0 to Skip", self)
        self.chip_other.setStyleSheet(CHIP_SKIP_STYLE)
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
        self.table_preview.setStyleSheet(BATCH_TABLE_STYLE)
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
        mw = _get_mw()
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
                self.chip_other.setStyleSheet(CHIP_OVERWRITE_STYLE)
            else:
                self.chip_other.setText(f"● {count_skip} to Skip")
                self.chip_other.setStyleSheet(CHIP_SKIP_STYLE)
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

    def _start_batch(self) -> None:
        """Save preferences and launch background batch process."""
        source_field = self.combo_source.currentText()
        target_field = self.combo_target.currentText()
        if source_field == target_field:
            showWarning("Source and target fields cannot be the same.", parent=self)
            return

        self._save_preferences()
        overwrite_mode = (
            "overwrite"
            if hasattr(self, "radio_overwrite") and self.radio_overwrite.isChecked()
            else "skip"
        )

        self.accept()

        run_batch_fill(
            parent_window=self.browser,
            note_ids=self.note_ids,
            source_field=source_field,
            target_field=target_field,
            overwrite_mode=overwrite_mode,
        )
