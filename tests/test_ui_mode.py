import os
import threading
import time
import unittest
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

try:
    from aqt.qt import QApplication, QHeaderView, QKeyEvent, QPushButton, QResizeEvent, QSize, Qt
    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    from addon.feed import IrasutoyaImage
    from addon.ui import (
        CHEVRON_ICON_PATH,
        DIALOG_STYLESHEET,
        MAX_SUGGESTION_CHIPS,
        PAGE_SIZE,
        ImageCardWidget,
        IrasutoyaDialog,
        _ThumbnailByteCache,
        clear_session_thumbnail_cache,
        make_description_snippet,
    )
    HAS_QT = True
except Exception:
    HAS_QT = False


class _SyncThread:
    """Helper thread runner that executes the target synchronously for deterministic tests."""

    def __init__(self, *args, **kwargs):
        self.target = kwargs.get("target")
        self.args = kwargs.get("args", ())

    def start(self):
        if self.target:
            self.target(*self.args)


def _install_catalog_counts(dlg, counts_map, desc_terms=None):
    """Fixture helper to install mocked CatalogIndex counts on a dialog."""
    desc_set = set(desc_terms or [])
    mock_index = MagicMock()

    def _search(q, **kw):
        count = counts_map.get(q, 0)
        if q in desc_set:
            return [
                MagicMock(
                    title="ポイントに悲しむ人のイラスト",
                    post_title="悲しむ人",
                    is_single_image_post=True,
                    description=f"{q}のイラストです。",
                )
            ] * count
        return [
            MagicMock(title=f"{q}のイラスト", post_title=f"{q}のイラスト", is_single_image_post=True, description="")
        ] * count

    mock_index.search.side_effect = _search
    dlg.store.get_index = MagicMock(return_value=mock_index)
    return mock_index


def _chip_texts(dlg):
    """Helper to collect rendered suggestion chip button texts."""
    return [
        dlg.chips_layout.itemAt(i).widget().text()
        for i in range(dlg.chips_layout.count())
        if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
    ]


@unittest.skipUnless(HAS_QT, "Qt/aqt environment not available in default python runner")
class TestUIMode(unittest.TestCase):
    def test_single_mode_omits_overwrite_policy(self):
        """Single mode should not show radio buttons for overwrite/skip."""
        dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
        self.assertFalse(hasattr(dlg, "radio_skip"))
        self.assertFalse(hasattr(dlg, "radio_overwrite"))
        self.assertTrue(hasattr(dlg, "btn_apply"))
        self.assertTrue(hasattr(dlg, "query_input"))
        self.assertTrue(dlg.query_input.isClearButtonEnabled())
        self.assertTrue(hasattr(dlg, "scroll_area"))

    def test_batch_mode_includes_overwrite_and_preview_table(self):
        """Batch mode should provide overwrite/skip options and a preview table with fixed column widths."""
        dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001, 1002, 1003])
        self.assertTrue(hasattr(dlg, "radio_skip"))
        self.assertTrue(hasattr(dlg, "radio_overwrite"))
        self.assertTrue(hasattr(dlg, "table_preview"))
        self.assertEqual(dlg.table_preview.columnCount(), 3)

        # Verify fixed column widths to guarantee zero layout jitter
        header = dlg.table_preview.horizontalHeader()
        self.assertEqual(header.sectionResizeMode(0), QHeaderView.ResizeMode.Stretch)
        self.assertEqual(header.sectionResizeMode(1), QHeaderView.ResizeMode.Fixed)
        self.assertEqual(header.sectionResizeMode(2), QHeaderView.ResizeMode.Fixed)
        self.assertEqual(dlg.table_preview.columnWidth(1), 140)
        self.assertEqual(dlg.table_preview.columnWidth(2), 130)

        # Verify live summary chips exist
        self.assertTrue(hasattr(dlg, "chip_fill"))
        self.assertTrue(hasattr(dlg, "chip_other"))

    def test_image_card_has_white_canvas_and_badge(self):
        """Thumbnail canvas must have white background and selection checkmark badge."""
        img = IrasutoyaImage(
            title="笑顔の子供",
            image_url="https://example.com/kid.png",
            thumb_url="https://example.com/kid_thumb.png",
            post_title="子供のイラスト"
        )
        card = ImageCardWidget(
            image=img,
            on_select=lambda c: None,
            on_double_click=lambda c: None
        )
        self.assertIn("#FFFFFF", card.thumb_label.styleSheet())
        self.assertEqual(card.title_label.text(), "笑顔の子供")

        # Badge starts hidden, becomes visible when selected
        self.assertTrue(card.badge_label.isHidden())
        card.set_selected(True)
        self.assertFalse(card.badge_label.isHidden())
        card.set_selected(False)
        self.assertTrue(card.badge_label.isHidden())

    def test_chevron_icon_and_combobox_stylesheet(self):
        """Verify chevron down SVG exists and is referenced in stylesheet."""
        self.assertTrue(os.path.exists(CHEVRON_ICON_PATH))
        self.assertIn(CHEVRON_ICON_PATH, DIALOG_STYLESHEET)
        self.assertIn("QComboBox::drop-down", DIALOG_STYLESHEET)

    def test_suggestion_chips_preserve_card_candidates_on_click(self):
        """Verify suggestion chips row preserves original card word and candidate cluster when clicking variations."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "本", "Reading": "ほん", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            counts_map = {"本": 1693, "ほん": 3, "ホン": 111, "ねこ": 6, "ネコ": 38}
            _install_catalog_counts(dlg, counts_map)

            mock_img = IrasutoyaImage("本", "https://example.com/hon.png", "https://example.com/hon_t.png", "本")
            dlg.client.search_with_fallbacks = lambda cands, **kw: (
                [mock_img] * counts_map.get(cands[0][0], 1) if cands else [],
                cands[0] if cands else None,
                None,
            )

            # Initial search for "本"
            dlg._do_search()
            self.assertEqual(_chip_texts(dlg), ["本  1693", "ほん  3", "ホン  111"])
            self.assertFalse(dlg.chips_widget.isHidden())

            # Simulate clicking "ほん" chip
            dlg.query_input.setText("ほん")
            dlg._do_search()

            chip_texts_after = _chip_texts(dlg)
            # Original word "本" must NOT disappear
            self.assertIn("本  1693", chip_texts_after)
            self.assertEqual(chip_texts_after, ["本  1693", "ほん  3", "ホン  111"])
            self.assertFalse(dlg.chips_widget.isHidden())

            # Active chip for "ほん" should have active accent styling
            chip_buttons_after = [
                dlg.chips_layout.itemAt(i).widget()
                for i in range(dlg.chips_layout.count())
                if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            active_btn = next(b for b in chip_buttons_after if "ほん" in b.text())
            self.assertIn("font-weight: 600", active_btn.styleSheet())
            inactive_btn = next(b for b in chip_buttons_after if "本" in b.text() and "ほん" not in b.text())
            self.assertNotIn("font-weight: 600", inactive_btn.styleSheet())

    def test_suggestion_chips_restore_chip_on_custom_query(self):
        """Verify an external query displays '↩ {card_term}' and clicking it restores original cluster."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "本", "Reading": "ほん", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            counts_map = {"本": 1693, "ほん": 3, "ホン": 111, "ねこ": 6, "ネコ": 38}
            _install_catalog_counts(dlg, counts_map)
            dlg.client.search_with_fallbacks = MagicMock(return_value=([], None, None))

            # Search custom query "ねこ"
            dlg.query_input.setText("ねこ")
            dlg._do_search()

            self.assertEqual(_chip_texts(dlg), ["↩ 本  1693", "ねこ  6", "ネコ  38"])
            self.assertFalse(dlg.chips_widget.isHidden())

            # Click restore chip
            chip_buttons = [
                dlg.chips_layout.itemAt(i).widget()
                for i in range(dlg.chips_layout.count())
                if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            restore_btn = chip_buttons[0]
            self.assertEqual(restore_btn.text(), "↩ 本  1693")
            restore_btn.click()

            # Dialog should restore search for "本" with original candidates
            self.assertEqual(dlg.query_input.text(), "本")
            self.assertEqual(_chip_texts(dlg), ["本  1693", "ほん  3", "ホン  111"])

    def test_suggestion_chips_stay_visible_on_zero_results(self):
        """Verify suggestion chips remain visible when a search returns zero results."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "本", "Reading": "ほん", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            counts_map = {"本": 1693, "ほん": 3, "ホン": 111}
            _install_catalog_counts(dlg, counts_map)

            # Search returns 0 results
            dlg.client.search_with_fallbacks = MagicMock(return_value=([], None, None))

            dlg.query_input.setText("ほん")
            dlg._do_search()

            # Chips widget must not be hidden
            self.assertFalse(dlg.chips_widget.isHidden())
            self.assertIn("No illustrations found for 'ほん'", dlg.status_label.text())
            self.assertEqual(_chip_texts(dlg), ["本  1693", "ほん  3", "ホン  111"])

    def test_missing_catalog_does_not_suppress_chips(self):
        """When catalog index is unavailable, candidates are rendered without probing or suppression."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "本", "Reading": "ほん", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            dlg.store.get_index = MagicMock(return_value=None)

            mock_img = IrasutoyaImage(
                "本のイラスト", "https://example.com/1.png", "https://example.com/1_t.png", "本のイラスト"
            )
            dlg.client.search_with_fallbacks = MagicMock(
                return_value=([mock_img], ("本", "primary", 0), mock_img)
            )

            dlg.query_input.setText("本")
            dlg._do_search()

            # Neither candidate was suppressed; neither has catalog count appended
            self.assertEqual(_chip_texts(dlg), ["本", "ほん", "ホン"])
            self.assertFalse(dlg.chips_widget.isHidden())

    def test_source_field_change_updates_card_candidates(self):
        """Verify changing the source field dropdown updates card search term and candidates."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "本", "Reading": "ほん", "Vocab": "猫", "EmptyField": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()
            dlg.client.search_with_fallbacks = MagicMock(return_value=([], None, None))
            self.assertEqual(dlg._card_source_term, "本")

            dlg.combo_source.setCurrentText("Vocab")
            self.assertEqual(dlg._card_source_term, "猫")
            terms = [c[0] for c in dlg._card_candidates]
            self.assertIn("猫", terms)

            # Test switching to an empty field clears chips, advances search generation, and re-enables search button
            dlg.btn_search.setEnabled(False)
            dlg.btn_search.setText("Searching...")
            gen_before = dlg._search_generation
            dlg.combo_source.setCurrentText("EmptyField")
            self.assertEqual(dlg._card_source_term, "")
            self.assertEqual(dlg._card_candidates, [])
            self.assertGreater(dlg._search_generation, gen_before)
            self.assertTrue(dlg.btn_search.isEnabled())
            self.assertEqual(dlg.btn_search.text(), "Search")
            self.assertEqual(dlg.status_label.text(), "Source field is empty.")

    def test_make_description_snippet(self):
        """Verify snippet extraction extracts keyword context and adds bold markup."""
        desc = "肩までの高さが147cm以下の小さな馬、ポニーのイラストです。"
        snippet = make_description_snippet(desc, "以下")
        self.assertIn("<b>以下</b>", snippet)
        self.assertIn("147cm", snippet)

        # Test that queries matching HTML entity names do not corrupt markup
        html_desc = "Adult < 100 & quality is high."
        html_snippet = make_description_snippet(html_desc, "lt")
        self.assertIn("Adu<b>lt</b>", html_snippet)
        self.assertIn("&lt;", html_snippet)
        self.assertNotIn("&<b>lt</b>;", html_snippet)

        empty = make_description_snippet("", "猫")
        self.assertEqual(empty, "")

    def test_image_card_description_match_badge_and_tooltip(self):
        """Card widget should display '📝 本文一致' badge and snippet tooltip when matching description."""
        img = IrasutoyaImage(
            title="ポニーのイラスト",
            image_url="https://example.com/pony.png",
            thumb_url="https://example.com/pony_thumb.png",
            post_title="ポニーのイラスト",
            description="肩までの高さが147cm以下の小さな馬、ポニーのイラストです。",
        )
        card = ImageCardWidget(
            image=img,
            on_select=lambda c: None,
            on_double_click=lambda c: None,
            is_description_match=True,
            match_snippet="…147cm<b>以下</b>の小さな馬…",
        )
        self.assertFalse(card.desc_match_badge.isHidden())
        self.assertEqual(card.desc_match_badge.text(), "📝 本文一致")
        self.assertIn("<b>以下</b>", card.desc_match_badge.toolTip())
        self.assertIn("<b>以下</b>", card.toolTip())

    def test_image_card_direct_title_match_omits_description_badge(self):
        """Direct title matches should keep the description match badge hidden."""
        img = IrasutoyaImage(
            title="猫のイラスト",
            image_url="https://example.com/cat.png",
            thumb_url="https://example.com/cat_thumb.png",
            post_title="猫のイラスト",
        )
        card = ImageCardWidget(
            image=img,
            on_select=lambda c: None,
            on_double_click=lambda c: None,
            is_description_match=False,
        )
        self.assertTrue(card.desc_match_badge.isHidden())

    def test_zero_result_chips_omitted(self):
        """Verify suggestion chips with 0 catalog results are omitted from the chips row."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "缶けり", "Reading": "", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            candidates = [("缶けり", "primary", 0), ("缶蹴り", "kanji", 1)]
            dlg._update_suggestion_chips(
                candidates, active_term="缶けり", probed_counts={"缶けり": 0, "缶蹴り": 2}
            )
            chip_buttons = [
                dlg.chips_layout.itemAt(i).widget()
                for i in range(dlg.chips_layout.count())
                if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            chip_texts = [b.text() for b in chip_buttons]
            self.assertEqual(chip_texts, ["缶蹴り  2"])
            self.assertFalse(dlg.chips_widget.isHidden())

    def test_suggestion_row_collapses_when_no_alternatives_have_results(self):
        """Suggestion chips bar should collapse cleanly (hidden) when all candidates yield 0 results."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "徹底的", "Reading": "", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            candidates = [("徹底的", "primary", 0)]
            dlg._update_suggestion_chips(
                candidates, active_term="徹底的", probed_counts={"徹底的": 0}
            )
            self.assertTrue(dlg.chips_widget.isHidden())

    def test_dynamic_pagination_initial_batch_and_scroll(self):
        """Visual picker should render PAGE_SIZE cards initially and load more on scroll."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "北", "Reading": "", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            mock_images = [
                IrasutoyaImage(
                    title=f"北のイラスト {i}",
                    image_url=f"https://example.com/{i}.png",
                    thumb_url=f"https://example.com/{i}_t.png",
                    post_title=f"北のイラスト {i}",
                )
                for i in range(50)
            ]

            dlg._on_search_finished(
                gen=dlg._search_generation,
                query="北",
                display_chips=[("北", "primary", 0)],
                matched_cand=("北", "primary", 0),
                images=mock_images,
            )

            # Initial page should render PAGE_SIZE (24) cards
            self.assertEqual(len(dlg.image_cards), PAGE_SIZE)
            self.assertEqual(dlg._loaded_count, PAGE_SIZE)
            self.assertIn(f"Showing {PAGE_SIZE} of 50 illustrations · Scroll for more", dlg.pagination_label.text())

            # Trigger scroll near bottom (simulate scroll position via Qt vertical scrollbar)
            dlg.scroll_area.verticalScrollBar().setMaximum(1000)
            dlg.scroll_area.verticalScrollBar().setValue(900)  # within 250px of maximum

            # Second page should load 24 more (total 48)
            self.assertEqual(len(dlg.image_cards), 48)
            self.assertEqual(dlg._loaded_count, 48)
            self.assertIn("Showing 48 of 50 illustrations · Scroll for more", dlg.pagination_label.text())

            # Trigger scroll near bottom again to load remaining 2
            dlg.scroll_area.verticalScrollBar().setValue(950)
            self.assertEqual(len(dlg.image_cards), 50)
            self.assertEqual(dlg._loaded_count, 50)
            self.assertEqual(dlg.pagination_label.text(), "Showing all 50 illustrations")

    def test_dynamic_pagination_keyboard_arrow_down(self):
        """Down-arrow keyboard navigation should trigger next page loading near grid bottom."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "北", "Reading": "", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            mock_images = [
                IrasutoyaImage(
                    title=f"北 {i}",
                    image_url=f"https://example.com/{i}.png",
                    thumb_url=f"https://example.com/{i}_t.png",
                    post_title=f"北 {i}",
                )
                for i in range(50)
            ]
            dlg._on_search_finished(
                gen=dlg._search_generation,
                query="北",
                display_chips=[("北", "primary", 0)],
                matched_cand=("北", "primary", 0),
                images=mock_images,
            )

            # Initially 24 cards
            self.assertEqual(len(dlg.image_cards), 24)

            # Select card index 22 (near bottom of the initial 24)
            dlg._on_card_selected(dlg.image_cards[22])

            # Simulate Down Arrow event and verify ensureWidgetVisible is invoked
            with patch.object(dlg.scroll_area, "ensureWidgetVisible") as mock_ensure:
                event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier)
                dlg.keyPressEvent(event)
                mock_ensure.assert_called_once_with(dlg.image_cards[25])

            # Pagination should have triggered, loading next batch
            self.assertEqual(len(dlg.image_cards), 48)
            self.assertEqual(dlg.image_cards.index(dlg.selected_card_widget), 25)

    def test_dynamic_pagination_keyboard_arrow_right_at_boundary(self):
        """Right-arrow keyboard navigation on the last card should trigger next page loading."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "北", "Reading": "", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            mock_images = [
                IrasutoyaImage(
                    title=f"北 {i}",
                    image_url=f"https://example.com/{i}.png",
                    thumb_url=f"https://example.com/{i}_t.png",
                    post_title=f"北 {i}",
                )
                for i in range(50)
            ]
            dlg._on_search_finished(
                gen=dlg._search_generation,
                query="北",
                display_chips=[("北", "primary", 0)],
                matched_cand=("北", "primary", 0),
                images=mock_images,
            )

            # Initially 24 cards
            self.assertEqual(len(dlg.image_cards), 24)

            # Select last card (index 23)
            dlg._on_card_selected(dlg.image_cards[23])

            # Simulate Right Arrow event
            with patch.object(dlg.scroll_area, "ensureWidgetVisible") as mock_ensure:
                event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
                dlg.keyPressEvent(event)
                mock_ensure.assert_called_once_with(dlg.image_cards[24])

            # Pagination should have triggered, loading next batch and advancing to card 24
            self.assertEqual(len(dlg.image_cards), 48)
            self.assertEqual(dlg.image_cards.index(dlg.selected_card_widget), 24)

    def test_keep_current_image_button_and_media_notice(self):
        """When target field contains existing media, cancel button becomes 'Keep Current Image (Esc)'."""
        mock_mw = MagicMock()
        mock_note = {
            "Expression": "猫",
            "Reading": "ねこ",
            "DefinitionPicture": '<img src="anime_shot_123.webp">',
            "EmptyField": "",
        }
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            self.assertEqual(dlg.btn_cancel.text(), "Keep Current Image (Esc)")
            self.assertFalse(dlg.media_notice_label.isHidden())
            self.assertIn("Card already has an image", dlg.media_notice_label.text())

            # Switch to an empty target field: reverts to standard Cancel
            dlg.combo_target.setCurrentText("EmptyField")
            self.assertEqual(dlg.btn_cancel.text(), "Cancel")
            self.assertTrue(dlg.media_notice_label.isHidden())

            # Uppercase tag test
            mock_note["DefinitionPicture"] = '<IMG SRC="UPPERCASE.JPG">'
            dlg.combo_target.setCurrentText("DefinitionPicture")
            self.assertEqual(dlg.btn_cancel.text(), "Keep Current Image (Esc)")
            self.assertFalse(dlg.media_notice_label.isHidden())

    def test_background_search_filters_decorative_and_irrelevant_counts(self):
        """Probed counts in background search should exclude decorative assets and irrelevant images."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "猫", "Reading": "ねこ", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            relevant_img = IrasutoyaImage(
                title="猫のイラスト",
                image_url="https://example.com/cat.png",
                thumb_url="https://example.com/cat_thumb.png",
                post_title="猫のイラスト",
            )
            decorative_img = IrasutoyaImage(
                title="猫のフレーム素材",
                image_url="https://example.com/cat_frame.png",
                thumb_url="https://example.com/cat_frame_thumb.png",
                post_title="猫のフレーム素材",
            )
            neko_hira_img = IrasutoyaImage(
                title="ねこのイラスト",
                image_url="https://example.com/neko.png",
                thumb_url="https://example.com/neko_thumb.png",
                post_title="ねこのイラスト",
            )
            mock_index = MagicMock()
            mock_index.search.side_effect = lambda q, **kw: (
                [relevant_img, decorative_img] if q == "猫" else [neko_hira_img]
            )
            dlg.store.get_index = MagicMock(return_value=mock_index)
            dlg.client.search_with_fallbacks = MagicMock(
                return_value=([relevant_img], ("猫", "primary", 0), relevant_img)
            )

            dlg.query_input.setText("猫")
            dlg._do_search()

            chip_buttons = [
                dlg.chips_layout.itemAt(i).widget()
                for i in range(dlg.chips_layout.count())
                if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            chip_texts = [b.text() for b in chip_buttons]
            # Even though catalog returned 2 images for '猫', 1 was decorative frame, so count badge is 1
            self.assertIn("猫  1", chip_texts)
            self.assertIn("ねこ  1", chip_texts)

    def test_resize_event_loads_more_when_no_scrollbar(self):
        """When dialog is resized so all cards fit without scrolling, additional pages load automatically."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "北", "Reading": "", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            mock_images = [
                IrasutoyaImage(
                    title=f"北 {i}",
                    image_url=f"https://example.com/{i}.png",
                    thumb_url=f"https://example.com/{i}_t.png",
                    post_title=f"北 {i}",
                )
                for i in range(50)
            ]
            dlg._on_search_finished(
                gen=dlg._search_generation,
                query="北",
                display_chips=[("北", "primary", 0)],
                matched_cand=("北", "primary", 0),
                images=mock_images,
            )

            self.assertEqual(len(dlg.image_cards), 24)

            # Dialog is visible with an expanded viewport height
            dlg.setVisible(True)
            # Size hint starts smaller than viewport (400 <= 600), then after next page grows larger (800 > 600)
            heights = [400, 800]
            with patch.object(dlg.scroll_area.viewport(), "height", return_value=600), \
                 patch.object(dlg.gallery_layout, "sizeHint", side_effect=lambda: QSize(800, heights.pop(0) if heights else 800)):
                dlg.resizeEvent(QResizeEvent(QSize(1200, 1000), QSize(800, 600)))

            # Next page should have been loaded (now 48 cards)
            self.assertEqual(len(dlg.image_cards), 48)

    def test_suggestion_chips_cap_expanded_to_eight(self) -> None:
        """Verify suggestion chips row displays up to 8 candidates without truncation."""
        dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
        self.addCleanup(dlg.close)

        # 10 candidates with positive catalog results
        cands: List[Tuple[str, str, int]] = [(f"word_{i}", "candidate", 1) for i in range(10)]
        counts: Dict[str, int] = {f"word_{i}": i + 1 for i in range(10)}

        dlg._update_suggestion_chips(cands, active_term="word_0", probed_counts=counts)
        rendered = _chip_texts(dlg)
        self.assertEqual(len(rendered), MAX_SUGGESTION_CHIPS)
        self.assertEqual(len(rendered), 8)
        self.assertEqual(rendered, [f"word_{i}  {i + 1}" for i in range(8)])

    def test_suggestion_chips_restore_chip_reserves_slot(self) -> None:
        """When restore_term chip is active, candidate chips are capped at 7 to keep total row at 8."""
        dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
        self.addCleanup(dlg.close)

        cands: List[Tuple[str, str, int]] = [(f"cand_{i}", "candidate", 1) for i in range(10)]
        counts: Dict[str, int] = {f"cand_{i}": i + 1 for i in range(10)}
        counts["card_term"] = 99

        dlg._update_suggestion_chips(
            cands,
            active_term="cand_0",
            restore_term="card_term",
            probed_counts=counts,
        )
        rendered = _chip_texts(dlg)
        self.assertEqual(len(rendered), MAX_SUGGESTION_CHIPS)
        self.assertEqual(rendered[0], "↩ card_term  99")
        # 1 restore chip + 7 candidate chips = 8 total chips
        self.assertEqual(len(rendered[1:]), 7)

    def test_suggestion_chips_click_executes_direct_search_without_fallback_loop(self) -> None:
        """Clicking a suggestion chip executes a direct query and breaks the fallback loop."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "手に入れる", "Reading": "", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            counts_map = {
                "手に入れる": 1,
                "手に入れた": 5,
                "手に入れよう": 2,
                "手に入れ": 14,
                "獲得": 9,
                "宝箱": 4,
            }
            _install_catalog_counts(dlg, counts_map, desc_terms={"手に入れる"})
            dlg._card_candidates = [
                ("手に入れる", "primary", 1),
                ("手に入れた", "verb_inflection", 1),
                ("手に入れよう", "verb_inflection", 1),
                ("手に入れ", "verb_stem", 2),
                ("獲得", "concept_map", 2),
                ("宝箱", "concept_map", 2),
            ]

            img_shock = IrasutoyaImage(
                title="ポイントに悲しむ人のイラスト（男性）",
                image_url="https://example.com/shock.png",
                thumb_url="https://example.com/shock_t.png",
                post_title="ポイントに悲しむ人のイラスト（男性）",
                description="ポイントを手に入れることができずに落ち込んで泣いている男性のイラストです。",
            )
            img_ticket = IrasutoyaImage(
                title="チケットを手に入れた人のイラスト（女性）",
                image_url="https://example.com/ticket.png",
                thumb_url="https://example.com/ticket_t.png",
                post_title="チケットを手に入れた人のイラスト（女性）",
                description="チケットを手に入れた女性のイラストです。",
            )

            def mock_search_with_fallbacks(
                candidates: Sequence[Tuple[str, str, int]], **kwargs: Any
            ) -> Tuple[List[IrasutoyaImage], Optional[Tuple[str, str, int]], Optional[IrasutoyaImage]]:
                """Mock search_with_fallbacks behavior distinguishing direct vs fallback queries."""
                cand_terms = [c[0] for c in candidates]
                if "手に入れた" in cand_terms and len(cand_terms) > 1:
                    matched = next(c for c in candidates if c[0] == "手に入れた")
                    return [img_ticket] * 5, matched, img_ticket
                first_c = candidates[0]
                if first_c[0] == "手に入れる":
                    return [img_shock], first_c, img_shock
                return [img_ticket] * 5, first_c, img_ticket

            dlg.client.search_with_fallbacks = MagicMock(side_effect=mock_search_with_fallbacks)

            # Initial card search (runs fallback traversal)
            dlg.query_input.setText("手に入れる")
            dlg._do_search()

            # Active term should be '手に入れた' via fallback
            self.assertEqual(dlg._current_query, "手に入れた")
            self.assertIn("No direct title match for '手に入れる'", dlg.status_label.text())
            self.assertIn("手に入れた", dlg.status_label.text())

            # Candidate chips should include all 6 candidates up to 8
            rendered_chips = _chip_texts(dlg)
            self.assertIn("手に入れる  1", rendered_chips)
            self.assertIn("宝箱  4", rendered_chips)
            self.assertEqual(len(rendered_chips), 6)

            # Click the '手に入れる 1' suggestion chip
            dlg._on_chip_clicked("手に入れる")

            # Must have performed a direct query, breaking the fallback loop!
            self.assertEqual(dlg._current_query, "手に入れる")
            self.assertEqual(dlg.query_input.text(), "手に入れる")
            self.assertIn("description match", dlg.status_label.text())
            self.assertEqual(len(dlg.image_cards), 1)
            self.assertTrue(dlg.image_cards[0].is_description_match)
            self.assertCountEqual(
                _chip_texts(dlg),
                [
                    "手に入れる  1",
                    "手に入れた  5",
                    "手に入れよう  2",
                    "手に入れ  14",
                    "獲得  9",
                    "宝箱  4",
                ],
            )
            chip_tooltips = [
                dlg.chips_layout.itemAt(i).widget().toolTip()
                for i in range(dlg.chips_layout.count())
                if dlg.chips_layout.itemAt(i).widget()
                and isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            self.assertTrue(
                any(
                    "Search for '手に入れる' (1 description match / 本文一致)" in tooltip
                    for tooltip in chip_tooltips
                )
            )

            # Clicking a verb_stem chip preserves its candidate reason in direct search
            dlg._on_chip_clicked("手に入れ")
            last_cands = dlg.client.search_with_fallbacks.call_args[0][0]
            self.assertEqual(last_cands, [("手に入れ", "verb_stem", 2)])

    def test_status_banner_distinguishes_title_match_fallback(self) -> None:
        """Status banner distinguishes between lack of title match and zero results."""
        dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
        self.addCleanup(dlg.close)
        dlg._thumb_executor.submit = MagicMock()

        mock_img = IrasutoyaImage(
            title="結果イラスト",
            image_url="https://example.com/res.png",
            thumb_url="https://example.com/res_t.png",
            post_title="結果イラスト",
        )

        # 1. Fallback when query had non-zero description count
        dlg._on_search_finished(
            gen=dlg._search_generation,
            query="手に入れる",
            display_chips=[("手に入れる", "primary", 1)],
            matched_cand=("手に入れた", "verb_inflection", 1),
            images=[mock_img],
            probed_counts={"手に入れる": 1},
        )
        self.assertIn("No direct title match for '手に入れる'", dlg.status_label.text())

        # 2. Fallback when query had zero catalog results
        dlg._on_search_finished(
            gen=dlg._search_generation,
            query="走っている",
            display_chips=[("走っている", "primary", 1)],
            matched_cand=("走る", "verb_deinflection", 2),
            images=[mock_img],
            probed_counts={"走っている": 0},
        )
        self.assertIn("No direct results for '走っている'", dlg.status_label.text())

        # 3. Fallback when query had positive catalog count but was not a description-only match
        dlg._on_search_finished(
            gen=dlg._search_generation,
            query="文字",
            display_chips=[("文字", "primary", 1)],
            matched_cand=("文字イラスト", "concept_map", 2),
            images=[mock_img],
            probed_counts={"文字": 2},
            probed_desc_matches=set(),
        )
        self.assertIn("No direct title illustration for '文字'", dlg.status_label.text())

        # 4. Direct search where matched candidate term equals query with non-primary reason
        dlg._on_search_finished(
            gen=dlg._search_generation,
            query="手に入れ",
            display_chips=[("手に入れ", "verb_stem", 2)],
            matched_cand=("手に入れ", "verb_stem", 2),
            images=[mock_img],
            probed_counts={"手に入れ": 1},
        )
        self.assertIn("Found 1 illustration", dlg.status_label.text())
        self.assertNotIn("Showing results for", dlg.status_label.text())

        # 5. Fallback when probed_counts is None defaults to neutral title match message
        dlg._on_search_finished(
            gen=dlg._search_generation,
            query="猫",
            display_chips=[("猫", "primary", 1)],
            matched_cand=("ねこ", "reading", 2),
            images=[mock_img],
            probed_counts=None,
        )
        self.assertIn("No direct title match for '猫'", dlg.status_label.text())

    def test_suggestion_chips_horizontal_scroll_area(self) -> None:
        """Suggestion chips row is housed inside a frameless QScrollArea with synced visibility."""
        dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
        self.addCleanup(dlg.close)

        self.assertTrue(hasattr(dlg, "chips_scroll"))
        self.assertEqual(dlg.chips_scroll.widget(), dlg.chips_widget)
        self.assertEqual(
            dlg.chips_scroll.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )
        self.assertEqual(
            dlg.chips_scroll.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertEqual(dlg.chips_scroll.height(), 38)
        self.assertEqual(dlg.chips_layout.contentsMargins().bottom(), 4)
        self.assertIn("QScrollBar:horizontal", dlg.chips_scroll.styleSheet())
        self.assertIn("height: 4px", dlg.chips_scroll.styleSheet())

        # Initially hidden
        self.assertTrue(dlg.chips_scroll.isHidden())
        self.assertTrue(dlg.chips_widget.isHidden())

        # Render 2 candidates -> becomes visible
        dlg._update_suggestion_chips(
            [("猫", "primary", 1), ("ねこ", "reading", 2)],
            active_term="猫",
            probed_counts={"猫": 5, "ねこ": 3},
        )
        self.assertFalse(dlg.chips_scroll.isHidden())
        self.assertFalse(dlg.chips_widget.isHidden())

        # Collapse chips -> hides both
        dlg._update_suggestion_chips([], active_term="")
        self.assertTrue(dlg.chips_scroll.isHidden())
        self.assertTrue(dlg.chips_widget.isHidden())

    def test_suggestion_chips_tooltip_description_match(self) -> None:
        """Suggestion chip tooltip identifies description-only matches with 本文一致 annotation."""
        dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
        self.addCleanup(dlg.close)

        dlg._update_suggestion_chips(
            [("手に入れる", "primary", 1), ("手に入れた", "verb_inflection", 1)],
            active_term="手に入れた",
            restore_term="手に入れる",
            probed_counts={"手に入れる": 1, "手に入れた": 5},
            probed_desc_matches={"手に入れる"},
        )

        tooltips = [
            dlg.chips_layout.itemAt(i).widget().toolTip()
            for i in range(dlg.chips_layout.count())
            if dlg.chips_layout.itemAt(i).widget()
            and isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
        ]

        self.assertTrue(
            any("Reset search to card term '手に入れる' (1 description match / 本文一致)" in t for t in tooltips)
        )
        self.assertTrue(
            any("Search for '手に入れる' (1 description match / 本文一致)" in t for t in tooltips)
        )
        self.assertTrue(
            any("Search for '手に入れた' (5 illustrations)" in t for t in tooltips)
        )

    def test_is_desc_only_match_classification(self) -> None:
        """_is_desc_only_match correctly distinguishes description matches from title/callout matches."""
        from addon.ui import _is_desc_only_match

        img_desc_only = IrasutoyaImage(
            title="ポイントに悲しむ人のイラスト",
            image_url="https://example.com/a.png",
            thumb_url="https://example.com/at.png",
            post_title="悲しい人のイラスト",
            description="手に入れることができなかったポイントのイラストです。",
        )
        img_title_match = IrasutoyaImage(
            title="手に入れた人のイラスト",
            image_url="https://example.com/b.png",
            thumb_url="https://example.com/bt.png",
            post_title="手に入れた人のイラスト",
            description="手に入れた人のイラストです。",
        )
        img_callout_title = IrasutoyaImage(
            title="手に入れるの文字イラスト",
            image_url="https://example.com/c.png",
            thumb_url="https://example.com/ct.png",
            post_title="文字イラスト",
            description="文字のイラストです。",
            categories=("文字",),
        )
        img_boundary_compound = IrasutoyaImage(
            title="犬歯のイラスト",
            image_url="https://example.com/d.png",
            thumb_url="https://example.com/dt.png",
            post_title="犬歯",
            description="犬の歯磨きのイラストです。",
        )
        self.assertTrue(_is_desc_only_match([img_desc_only], "手に入れる"))
        self.assertTrue(_is_desc_only_match([img_boundary_compound], "犬"))
        self.assertFalse(_is_desc_only_match([img_title_match], "手に入れた"))
        self.assertFalse(_is_desc_only_match([img_title_match, img_desc_only], "手に入れる"))
        self.assertFalse(_is_desc_only_match([img_callout_title], "手に入れる"))
        self.assertFalse(_is_desc_only_match([], "手に入れる"))
        self.assertFalse(_is_desc_only_match([img_desc_only], ""))

    def test_thumbnail_byte_cache_lru_and_byte_cap(self) -> None:
        """Verify _ThumbnailByteCache enforces entry limit, byte limit, and LRU eviction."""
        cache = _ThumbnailByteCache(max_entries=3, max_bytes=30)
        self.assertEqual(len(cache), 0)

        # Basic put and get
        cache.put("https://example.com/1.png", b"12345")
        cache.put("https://example.com/2.png", b"12345")
        cache.put("https://example.com/3.png", b"12345")
        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.get("https://example.com/1.png"), b"12345")

        # Adding 4th entry evicts oldest non-MRU (2.png)
        cache.put("https://example.com/4.png", b"12345")
        self.assertEqual(len(cache), 3)
        self.assertIsNone(cache.get("https://example.com/2.png"))
        self.assertIsNotNone(cache.get("https://example.com/1.png"))
        self.assertIsNotNone(cache.get("https://example.com/3.png"))
        self.assertIsNotNone(cache.get("https://example.com/4.png"))

        # Large entry evicts items to stay under max_bytes (30 bytes)
        cache.put("https://example.com/big.png", b"x" * 25)
        self.assertLessEqual(cache._current_bytes, 30)
        self.assertIsNotNone(cache.get("https://example.com/big.png"))
        self.assertEqual(len(cache), 2)

        # Empty data or URL is ignored
        cache.put("", b"data")
        cache.put("https://example.com/empty.png", b"")
        self.assertIsNone(cache.get("https://example.com/empty.png"))

        # Clear
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache._current_bytes, 0)

    def test_thumbnail_cache_reuses_thumbnails_across_suggestion_chips(self) -> None:
        """Verify suggestion chip switch renders cached thumbnails in 0ms without executor submission."""
        valid_png = bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de00000009"
            "7048597300000f6100000f6101a83fa7690000000c49444154089963f8cfc0000003010100"
            "9ce3bf590000000049454e44ae426082"
        )
        mock_mw = MagicMock()
        mock_note = {"Expression": "本", "Reading": "ほん", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        clear_session_thumbnail_cache()
        self.addCleanup(clear_session_thumbnail_cache)

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            img1 = IrasutoyaImage("本", "https://example.com/hon.png", "https://example.com/hon_t.png", "本")
            dlg.client.download_image = MagicMock(return_value=valid_png)

            # Cold render: card requires background download task
            dlg._populate_gallery([img1], gen=dlg._search_generation)
            self.assertEqual(len(dlg.image_cards), 1)

            # Run worker task synchronously to populate cache
            dlg._fetch_thumbnail_task(dlg.image_cards[0], gen=dlg._search_generation, delay_ms=0)
            self.assertEqual(dlg.client.download_image.call_count, 1)
            self.assertFalse(dlg.image_cards[0].thumb_label.pixmap().isNull())

            # Verify bytes were stored in session cache and pixmap in dialog cache
            self.assertIsNotNone(dlg._thumb_byte_cache.get(img1.thumb_url))
            self.assertIsNotNone(dlg._thumb_pixmap_cache.get(img1.thumb_url))

            # Simulate switching to another chip sharing the same thumbnail
            with patch.object(dlg._thumb_executor, "submit") as mock_submit:
                dlg._clear_gallery()

                # Populate with the same image in new search generation
                dlg._search_generation += 1
                dlg._all_images = [img1]
                dlg._load_next_page(gen=dlg._search_generation)

                self.assertEqual(len(dlg.image_cards), 1)
                # Synchronous cache hit: pixmap is set immediately upon creation!
                self.assertFalse(dlg.image_cards[0].thumb_label.pixmap().isNull())
                # No executor task submitted for cached thumbnail!
                mock_submit.assert_not_called()
                # download_image was NOT called again
                self.assertEqual(dlg.client.download_image.call_count, 1)

    def test_inflight_thumbnail_saved_to_cache_when_generation_changes(self) -> None:
        """Verify completed in-flight download populates cache even if search generation advanced."""
        valid_png = bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de00000009"
            "7048597300000f6100000f6101a83fa7690000000c49444154089963f8cfc0000003010100"
            "9ce3bf590000000049454e44ae426082"
        )
        mock_mw = MagicMock()
        mock_note = {"Expression": "猫", "Reading": "ねこ", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        clear_session_thumbnail_cache()
        self.addCleanup(clear_session_thumbnail_cache)

        with patch("addon.ui.mw", mock_mw):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)

            img = IrasutoyaImage("猫", "https://example.com/cat.png", "https://example.com/cat_t.png", "猫")
            card = ImageCardWidget(img, on_select=lambda c: None, on_double_click=lambda c: None)

            def mock_download(url: str, **kwargs: Any) -> bytes:
                """Simulate generation advancing while download is mid-flight."""
                dlg._search_generation += 1
                return valid_png

            dlg.client.download_image = MagicMock(side_effect=mock_download)

            # Start worker task for generation 1
            gen1 = dlg._search_generation
            dlg._fetch_thumbnail_task(card, gen=gen1, delay_ms=0)

            # Verify that even though generation changed during download, bytes were cached!
            cached = dlg._thumb_byte_cache.get(img.thumb_url)
            self.assertEqual(cached, valid_png)

    def test_inflight_thumbnail_deduplication_coalesces_concurrent_requests(self) -> None:
        """Verify multiple cards requesting same thumbnail URL coalesce into single download."""
        valid_png = bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de00000009"
            "7048597300000f6100000f6101a83fa7690000000c49444154089963f8cfc0000003010100"
            "9ce3bf590000000049454e44ae426082"
        )
        mock_mw = MagicMock()
        mock_note = {"Expression": "犬", "Reading": "いぬ", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        clear_session_thumbnail_cache()
        self.addCleanup(clear_session_thumbnail_cache)

        with patch("addon.ui.mw", mock_mw):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)
            applied_callbacks: List[Any] = []
            dlg._run_on_main = applied_callbacks.append

            img1 = IrasutoyaImage("犬 1", "https://example.com/dog.png", "https://example.com/dog_t.png", "犬")
            img2 = IrasutoyaImage("犬 2", "https://example.com/dog.png", "https://example.com/dog_t.png", "犬")
            card1 = ImageCardWidget(img1, on_select=lambda c: None, on_double_click=lambda c: None)
            card2 = ImageCardWidget(img2, on_select=lambda c: None, on_double_click=lambda c: None)

            download_started = threading.Event()
            can_finish_download = threading.Event()

            def mock_slow_download(url: str, **kwargs: Any) -> bytes:
                """Signal download started and wait for permission to finish."""
                download_started.set()
                can_finish_download.wait(timeout=2.0)
                return valid_png

            dlg.client.download_image = MagicMock(side_effect=mock_slow_download)

            original_get = dlg._thumb_byte_cache.get
            get_count = 0
            get_lock = threading.Lock()
            t2_reached_cache = threading.Event()

            def tracking_get(url: str) -> Optional[bytes]:
                """Count cache get calls and signal when Thread 2 reaches cache check."""
                nonlocal get_count
                res = original_get(url)
                with get_lock:
                    get_count += 1
                    if get_count == 2:
                        t2_reached_cache.set()
                return res

            get_patcher = patch.object(dlg._thumb_byte_cache, "get", side_effect=tracking_get)
            get_patcher.start()
            self.addCleanup(get_patcher.stop)

            # Thread 1 starts download
            t1 = threading.Thread(
                target=lambda: dlg._fetch_thumbnail_task(
                    card1, gen=dlg._search_generation, delay_ms=0
                )
            )
            t1.start()

            # Wait until thread 1 has entered download_image
            self.assertTrue(download_started.wait(timeout=1.0))

            # Thread 2 starts for same URL while Thread 1 is downloading
            t2 = threading.Thread(
                target=lambda: dlg._fetch_thumbnail_task(
                    card2, gen=dlg._search_generation, delay_ms=0
                )
            )
            t2.start()

            # Ensure thread 2 has checked cache and entered inflight wait
            self.assertTrue(t2_reached_cache.wait(timeout=1.0))
            time.sleep(0.05)

            # Let thread 1 finish
            can_finish_download.set()

            t1.join(timeout=2.0)
            t2.join(timeout=2.0)

            # download_image must be called only once
            self.assertEqual(dlg.client.download_image.call_count, 1)
            self.assertEqual(dlg._thumb_byte_cache.get(img1.thumb_url), valid_png)
            # Both cards must receive a thumbnail callback
            self.assertEqual(len(applied_callbacks), 2)
            for cb in applied_callbacks:
                cb()
            self.assertFalse(card1.thumb_label.pixmap().isNull())
            self.assertFalse(card2.thumb_label.pixmap().isNull())

    def test_failed_thumbnail_download_is_not_cached(self) -> None:
        """Verify failed (b'') or corrupted thumbnail download is not cached and shows 'No Preview'."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "鳥", "Reading": "とり", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        clear_session_thumbnail_cache()
        self.addCleanup(clear_session_thumbnail_cache)

        with patch("addon.ui.mw", mock_mw):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            self.addCleanup(dlg.close)

            img = IrasutoyaImage("鳥", "https://example.com/bird.png", "https://example.com/bird_t.png", "鳥")
            card = ImageCardWidget(img, on_select=lambda c: None, on_double_click=lambda c: None)

            dlg.client.download_image = MagicMock(side_effect=Exception("Connection timed out"))
            dlg._run_on_main = lambda cb: cb()

            dlg._fetch_thumbnail_task(card, gen=dlg._search_generation, delay_ms=0)

            self.assertEqual(card.thumb_label.text(), "No Preview")
            self.assertIsNone(dlg._thumb_byte_cache.get(img.thumb_url))
            self.assertNotIn(img.thumb_url, dlg._thumb_pixmap_cache)

            # Test corrupted non-image bytes (e.g. captive portal HTML) are not cached
            corrupt_img = IrasutoyaImage("鳥2", "https://example.com/bird2.png", "https://example.com/bird2_t.png", "鳥")
            corrupt_card = ImageCardWidget(corrupt_img, on_select=lambda c: None, on_double_click=lambda c: None)
            dlg.client.download_image = MagicMock(return_value=b"<html>502 Bad Gateway</html>")
            dlg._fetch_thumbnail_task(corrupt_card, gen=dlg._search_generation, delay_ms=0)

            self.assertEqual(corrupt_card.thumb_label.text(), "No Preview")
            self.assertIsNone(dlg._thumb_byte_cache.get(corrupt_img.thumb_url))
            self.assertNotIn(corrupt_img.thumb_url, dlg._thumb_pixmap_cache)

    def test_clear_gallery_cancels_pending_futures_and_done_clears_caches(self) -> None:
        """Verify _clear_gallery cancels pending futures and done cleans up memory and client pool."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "花", "Reading": "はな", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])

            mock_fut = MagicMock()
            mock_fut.done.return_value = False
            dlg._pending_thumb_futures.append(mock_fut)

            # Populate pixmap cache
            dlg._thumb_pixmap_cache["https://example.com/flower_t.png"] = MagicMock()
            self.assertEqual(len(dlg._thumb_pixmap_cache), 1)

            # _clear_gallery cancels pending futures
            dlg._clear_gallery()
            mock_fut.cancel.assert_called_once()
            self.assertEqual(len(dlg._pending_thumb_futures), 0)

            # done() clears pixmap cache and calls client.close()
            with patch.object(dlg.client, "close") as mock_close:
                dlg.done(0)
                self.assertEqual(len(dlg._thumb_pixmap_cache), 0)
                mock_close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
