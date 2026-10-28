"""Custom Qt widgets for illustration thumbnail cards and suggestion chips."""
import html
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from aqt.qt import (
    QCursor,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPixmap,
    QPushButton,
    QScrollArea,
    Qt,
    QVBoxLayout,
    QWidget,
)

from ..feed import IrasutoyaImage
from .styles import (
    CARD_BADGE_STYLE,
    CARD_DEFAULT_STYLE,
    CARD_DESC_MATCH_BADGE_STYLE,
    CARD_SELECTED_STYLE,
    CARD_THUMB_STYLE,
    CARD_TITLE_STYLE,
    CHIP_ACTIVE_STYLE,
    CHIP_INACTIVE_STYLE,
    CHIP_RESTORE_STYLE,
    MAX_SUGGESTION_CHIPS,
)


class ImageCardWidget(QFrame):
    """Clickable thumbnail card displaying an Irasutoya illustration."""

    def __init__(
        self,
        image: IrasutoyaImage,
        on_select: Callable[["ImageCardWidget"], None],
        on_double_click: Callable[["ImageCardWidget"], None],
        parent: Optional[QWidget] = None,
        is_description_match: bool = False,
        match_snippet: Optional[str] = None,
    ) -> None:
        """Initialize the illustration thumbnail card widget."""
        super().__init__(parent)
        self.image: IrasutoyaImage = image
        self.on_select: Callable[["ImageCardWidget"], None] = on_select
        self.on_double_click: Callable[["ImageCardWidget"], None] = on_double_click
        self.is_description_match: bool = is_description_match
        self.match_snippet: Optional[str] = match_snippet
        self.is_selected: bool = False

        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedWidth(145)
        self.setFixedHeight(165)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # White canvas container for illustration so transparent PNGs stay crisp
        self.thumb_label = QLabel(self)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setFixedSize(129, 105)
        self.thumb_label.setText("Loading...")
        self.thumb_label.setStyleSheet(CARD_THUMB_STYLE)
        layout.addWidget(self.thumb_label)

        self.title_label = QLabel(image.title, self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(CARD_TITLE_STYLE)
        layout.addWidget(self.title_label)

        # Discreet checkmark badge overlay in top-right corner
        self.badge_label = QLabel("✓", self)
        self.badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge_label.setFixedSize(18, 18)
        self.badge_label.setStyleSheet(CARD_BADGE_STYLE)
        self.badge_label.move(self.width() - 24, 6)
        self.badge_label.setVisible(False)

        # Incidental description match badge overlay in top-left corner
        self.desc_match_badge = QLabel("📝 本文一致", self)
        self.desc_match_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc_match_badge.setStyleSheet(CARD_DESC_MATCH_BADGE_STYLE)
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
            self.setStyleSheet(CARD_SELECTED_STYLE)
        else:
            self.setStyleSheet(CARD_DEFAULT_STYLE)

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


def populate_suggestion_chips(
    chips_layout: QHBoxLayout,
    chips_widget: QWidget,
    chips_scroll: Optional[QScrollArea],
    candidates: Sequence[Tuple[str, str, int]],
    active_term: Optional[str] = None,
    restore_term: Optional[str] = None,
    probed_counts: Optional[Dict[str, int]] = None,
    probed_desc_matches: Optional[Set[str]] = None,
    on_chip_clicked: Optional[Callable[[str], None]] = None,
    on_restore_clicked: Optional[Callable[[str], None]] = None,
) -> None:
    """Render one-click suggestion chips for query variations with instant count badges."""
    while chips_layout.count():
        item = chips_layout.takeAt(0)
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
        chips_widget.setVisible(False)
        if chips_scroll is not None:
            chips_scroll.setVisible(False)
        return

    lbl = QLabel("Try:", chips_widget)
    lbl.setStyleSheet("color: #8E8E93; font-size: 11px; font-weight: 500;")
    chips_layout.addWidget(lbl)

    # Prepend restore chip if searching an external custom query
    if restore_term:
        is_res_desc = bool(probed_desc_matches and restore_term in probed_desc_matches)
        res_text = (
            f"↩ {restore_term}  {restore_count}"
            if (restore_count is not None and restore_count > 0)
            else f"↩ {restore_term}"
        )
        btn_restore = QPushButton(res_text, chips_widget)
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
        btn_restore.setStyleSheet(CHIP_RESTORE_STYLE)
        if on_restore_clicked:
            btn_restore.clicked.connect(lambda _, t=restore_term: on_restore_clicked(t))
        chips_layout.addWidget(btn_restore)

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
        btn = QPushButton(btn_text, chips_widget)
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
            btn.setStyleSheet(CHIP_ACTIVE_STYLE)
        else:
            btn.setStyleSheet(CHIP_INACTIVE_STYLE)
        if on_chip_clicked:
            btn.clicked.connect(lambda _, t=cand_term: on_chip_clicked(t))
        chips_layout.addWidget(btn)

    chips_layout.addStretch()
    chips_widget.setVisible(True)
    if chips_scroll is not None:
        chips_scroll.setVisible(True)
