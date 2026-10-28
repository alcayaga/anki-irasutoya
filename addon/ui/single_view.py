"""Single-card visual picker view mixin and gallery controller."""
import logging
import threading
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

from aqt.qt import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    Qt,
    QVBoxLayout,
    QWidget,
)
from aqt.utils import showWarning, tooltip

from ..batch import make_safe_media_filename
from ..config import (
    CONFIG_KEY_IMAGE_SIZE,
    CONFIG_KEY_THUMB_DELAY,
)
from ..feed import IrasutoyaImage
from ..text_utils import clean_search_query, generate_search_candidates
from .helpers import (
    _description_match_term,
    _is_desc_only_match,
    _StaleSearch,
    format_fallback_reason,
    has_existing_media,
    make_description_snippet,
    probe_catalog_candidates,
)
from .styles import (
    CHIPS_SCROLL_STYLE,
    GALLERY_SCROLL_STYLE,
    PAGE_SIZE,
    STATUS_ACTIVE_STYLE,
    STATUS_ERROR_STYLE,
    STATUS_INFO_BLUE_STYLE,
    STATUS_MUTED_STYLE,
)
from .widgets import ImageCardWidget, populate_suggestion_chips

if TYPE_CHECKING:
    from .dialog import IrasutoyaDialog

logger = logging.getLogger(__name__)


class _SingleNoteModeMixin:
    """Mixin providing single-card visual thumbnail picker and commit pipeline for IrasutoyaDialog."""

    if TYPE_CHECKING:
        self: "IrasutoyaDialog"

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
        self.chips_scroll.setStyleSheet(CHIPS_SCROLL_STYLE)
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
        self.status_label.setStyleSheet(STATUS_MUTED_STYLE)
        main_layout.addWidget(self.status_label)

        # Gallery Scroll Area with Right Margin Gutter
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(GALLERY_SCROLL_STYLE)

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
            from aqt.qt import QTimer
            QTimer.singleShot(50, self._do_search)

    def _check_target_media_status(self, *args) -> None:
        """Inspect note's target field for existing media and adapt cancel button label."""
        if not hasattr(self, "btn_cancel") or not self.btn_cancel:
            return
        target_field = self.combo_target.currentText() if hasattr(self, "combo_target") else ""
        content = ""
        if self.first_note and target_field and target_field in self.first_note:
            content = self.first_note[target_field]

        has_media = has_existing_media(content)
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
                if raw_val
                else []
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
                self.status_label.setStyleSheet(STATUS_MUTED_STYLE)
                self.status_label.setText("Source field is empty.")

    def _do_search(self, direct: bool = False) -> None:
        """Fetch search results from Irasutoya with fallback candidate generation asynchronously.

        Args:
            direct: If True, execute a direct query for the exact search term without
                traversing alternative fallback inflections (used when clicking a specific chip).
        """
        query = self.query_input.text().strip()
        if not query:
            self.status_label.setStyleSheet(STATUS_ERROR_STYLE)
            self.status_label.setText("Please enter a query term.")
            return

        self._search_generation += 1
        current_gen = self._search_generation

        self.status_label.setStyleSheet(STATUS_ACTIVE_STYLE)
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
            restore_term = (
                self._card_source_term
                if (self._card_source_term and self._card_source_term != query)
                else None
            )

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

                    probe_terms = [c[0] for c in display_chips]
                    if matched_cand and matched_cand[0] not in probe_terms:
                        probe_terms.append(matched_cand[0])

                    probed_counts, probed_desc_matches = probe_catalog_candidates(
                        catalog,
                        image_size,
                        probe_terms,
                        restore_term,
                        count_cache,
                        desc_cache,
                        is_stale=lambda: current_gen != self._search_generation,
                    )
            except _StaleSearch:
                return
            except Exception:
                logger.exception("Catalog count probe failed")

            if current_gen != self._search_generation:
                return

            self._run_on_main(
                lambda: self._on_search_finished(
                    current_gen,
                    query,
                    display_chips,
                    matched_cand,
                    images,
                    restore_term,
                    probed_counts,
                    probed_desc_matches,
                )
            )

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
            self.status_label.setStyleSheet(STATUS_ERROR_STYLE)
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
            self.status_label.setStyleSheet(STATUS_INFO_BLUE_STYLE)
            reason_text = format_fallback_reason(query, probed_counts, probed_desc_matches)
            self.status_label.setText(
                f"ℹ️ {reason_text}. Showing results for '{matched_cand[0]}' ({matched_cand[1]})."
            )
        else:
            self.status_label.setStyleSheet(STATUS_MUTED_STYLE)
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
        candidates: Sequence[Tuple[str, str, int]],
        active_term: Optional[str] = None,
        restore_term: Optional[str] = None,
        probed_counts: Optional[Dict[str, int]] = None,
        probed_desc_matches: Optional[Set[str]] = None,
    ) -> None:
        """Render one-click suggestion chips for query variations with instant count badges."""
        if not hasattr(self, "chips_layout"):
            return
        populate_suggestion_chips(
            chips_layout=self.chips_layout,
            chips_widget=self.chips_widget,
            chips_scroll=getattr(self, "chips_scroll", None),
            candidates=candidates,
            active_term=active_term,
            restore_term=restore_term,
            probed_counts=probed_counts,
            probed_desc_matches=probed_desc_matches,
            on_chip_clicked=self._on_chip_clicked,
            on_restore_clicked=self._on_restore_clicked,
        )

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
        self._thumb_loader.cancel_pending_futures()
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
        self._thumb_loader.reset_burst()
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
            cached_pixmap = self._thumb_loader.get_cached_pixmap(img.thumb_url)
            if cached_pixmap is not None:
                card.set_scaled_pixmap(cached_pixmap)
            else:
                fut = self._thumb_loader.executor.submit(
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
        self._thumb_loader.fetch_thumbnail_task(
            card=card,
            gen=gen,
            client=self.client,
            is_current_gen=lambda g: g == self._search_generation,
            run_on_main=self._run_on_main,
            delay_ms=delay_ms,
        )

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
        self.status_label.setStyleSheet(STATUS_ACTIVE_STYLE)
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

                    def commit_single(col: Any) -> Any:
                        """Write media and assign field to fresh note."""
                        fresh_note = col.get_note(self.first_note.id)
                        if target_field not in fresh_note:
                            raise KeyError(f"Field '{target_field}' no longer exists on this note")
                        stored_filename = col.media.write_data(filename, img_bytes)
                        encoded_src = urllib.parse.quote(stored_filename)
                        fresh_note[target_field] = f'<img src="{encoded_src}">'
                        return fresh_note

                    def on_committed(fresh_note: Any) -> None:
                        """Update notes and close dialog upon successful commit."""
                        def on_update_success(_: Any) -> None:
                            """Notify success and close dialog."""
                            tooltip(f"Added '{best_img.title}' to {target_field}.", parent=self.browser)
                            if self.isVisible():
                                self.accept()

                        def on_update_failure(err: Exception) -> None:
                            """Restore button state and display warning on update failure."""
                            self._applying = False
                            if self.isVisible():
                                self.btn_apply.setEnabled(True)
                                self.status_label.setStyleSheet(STATUS_ERROR_STYLE)
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
                            self.status_label.setStyleSheet(STATUS_ERROR_STYLE)
                            self.status_label.setText(f"Error applying image: {err}")
                            showWarning(f"Failed to write media or update note: {err}", parent=self)

                    from aqt.operations import QueryOp
                    commit_op = QueryOp(
                        parent=self,
                        op=commit_single,
                        success=on_committed,
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
                        self.status_label.setStyleSheet(STATUS_ERROR_STYLE)
                        self.status_label.setText(f"Error applying image: {err_msg}")
                        showWarning(f"Failed to download or apply image: {err_msg}", parent=self)

                self._run_on_main(on_error)

        threading.Thread(target=background_apply, daemon=True).start()
