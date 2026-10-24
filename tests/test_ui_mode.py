import os
import unittest
from unittest.mock import MagicMock, patch

try:
    from aqt.qt import QApplication, QHeaderView, QKeyEvent, QPushButton, Qt
    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    from addon.feed import IrasutoyaImage
    from addon.ui import (
        CHEVRON_ICON_PATH,
        DIALOG_STYLESHEET,
        PAGE_SIZE,
        ImageCardWidget,
        IrasutoyaDialog,
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
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            mock_index = MagicMock()
            counts_map = {"本": 1693, "ほん": 3, "ホン": 111, "ねこ": 6, "ネコ": 38}
            mock_index.search.side_effect = lambda q, **kw: [
                MagicMock(title=f"{q}のイラスト", is_single_image_post=True, description="")
            ] * counts_map.get(q, 0)
            dlg.store.get_index = MagicMock(return_value=mock_index)
            dlg.client.uses_catalog = MagicMock(return_value=True)

            mock_img = IrasutoyaImage("本", "https://example.com/hon.png", "https://example.com/hon_t.png", "本")
            dlg.client.search_with_fallbacks = lambda cands, **kw: ([mock_img], cands[0] if cands else None, None)

            # Initial search for "本"
            dlg._do_search()
            chip_buttons = [
                dlg.chips_layout.itemAt(i).widget()
                for i in range(dlg.chips_layout.count())
                if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            chip_texts = [b.text() for b in chip_buttons]
            self.assertEqual(chip_texts, ["本  1693", "ほん  3", "ホン  111"])
            self.assertFalse(dlg.chips_widget.isHidden())

            # Simulate clicking "ほん" chip
            dlg.query_input.setText("ほん")
            dlg._do_search()

            chip_buttons_after = [
                dlg.chips_layout.itemAt(i).widget()
                for i in range(dlg.chips_layout.count())
                if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            chip_texts_after = [b.text() for b in chip_buttons_after]

            # Original word "本" must NOT disappear
            self.assertIn("本  1693", chip_texts_after)
            self.assertEqual(chip_texts_after, ["本  1693", "ほん  3", "ホン  111"])
            self.assertFalse(dlg.chips_widget.isHidden())

            # Active chip for "ほん" should have active accent styling
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
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            mock_index = MagicMock()
            counts_map = {"本": 1693, "ほん": 3, "ホン": 111, "ねこ": 6, "ネコ": 38}
            mock_index.search.side_effect = lambda q, **kw: [
                MagicMock(title=f"{q}のイラスト", is_single_image_post=True, description="")
            ] * counts_map.get(q, 0)
            dlg.store.get_index = MagicMock(return_value=mock_index)
            dlg.client.uses_catalog = MagicMock(return_value=True)

            dlg.client.search_with_fallbacks = MagicMock(return_value=([], None, None))

            # Search custom query "ねこ"
            dlg.query_input.setText("ねこ")
            dlg._do_search()

            chip_buttons = [
                dlg.chips_layout.itemAt(i).widget()
                for i in range(dlg.chips_layout.count())
                if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            chip_texts = [b.text() for b in chip_buttons]
            self.assertEqual(chip_texts, ["↩ 本  1693", "ねこ  6", "ネコ  38"])
            self.assertFalse(dlg.chips_widget.isHidden())

            # Click restore chip
            restore_btn = chip_buttons[0]
            self.assertEqual(restore_btn.text(), "↩ 本  1693")
            restore_btn.click()

            # Dialog should restore search for "本" with original candidates
            self.assertEqual(dlg.query_input.text(), "本")
            chip_buttons_restored = [
                dlg.chips_layout.itemAt(i).widget()
                for i in range(dlg.chips_layout.count())
                if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            chip_texts_restored = [b.text() for b in chip_buttons_restored]
            self.assertEqual(chip_texts_restored, ["本  1693", "ほん  3", "ホン  111"])

    def test_suggestion_chips_stay_visible_on_zero_results(self):
        """Verify suggestion chips remain visible when a search returns zero results."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "本", "Reading": "ほん", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            mock_index = MagicMock()
            counts_map = {"本": 1693, "ほん": 3, "ホン": 111}
            mock_index.search.side_effect = lambda q, **kw: [
                MagicMock(title=f"{q}のイラスト", is_single_image_post=True, description="")
            ] * counts_map.get(q, 0)
            dlg.store.get_index = MagicMock(return_value=mock_index)
            dlg.client.uses_catalog = MagicMock(return_value=True)

            # Search returns 0 results
            dlg.client.search_with_fallbacks = MagicMock(return_value=([], None, None))

            dlg.query_input.setText("ほん")
            dlg._do_search()

            # Chips widget must not be hidden
            self.assertFalse(dlg.chips_widget.isHidden())
            self.assertIn("No illustrations found for 'ほん'", dlg.status_label.text())
            chip_buttons = [
                dlg.chips_layout.itemAt(i).widget()
                for i in range(dlg.chips_layout.count())
                if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            self.assertEqual([b.text() for b in chip_buttons], ["本  1693", "ほん  3", "ホン  111"])

    def test_live_backend_does_not_suppress_chips_with_catalog(self):
        """When client uses live backend, candidates are not probed or discarded by catalog."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "本", "Reading": "ほん", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()

            # Live backend: uses_catalog returns False
            dlg.client.uses_catalog = MagicMock(return_value=False)
            mock_index = MagicMock()
            mock_index.search.return_value = []  # Catalog reports 0 results
            dlg.store.get_index = MagicMock(return_value=mock_index)

            # Live search returns a hit for "本"
            mock_img = IrasutoyaImage(
                "本のイラスト", "https://example.com/1.png", "https://example.com/1_t.png", "本のイラスト"
            )
            dlg.client.search_with_fallbacks = MagicMock(
                return_value=([mock_img], ("本", "primary", 0), mock_img)
            )

            dlg.query_input.setText("本")
            dlg._do_search()

            # Verify catalog index search was never called because uses_catalog is False
            mock_index.search.assert_not_called()

            chip_buttons = [
                dlg.chips_layout.itemAt(i).widget()
                for i in range(dlg.chips_layout.count())
                if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            chip_texts = [b.text() for b in chip_buttons]
            # Neither candidate was suppressed; neither has catalog count appended
            self.assertEqual(chip_texts, ["本", "ほん", "ホン"])
            self.assertFalse(dlg.chips_widget.isHidden())

    def test_source_field_change_updates_card_candidates(self):
        """Verify changing the source field dropdown updates card search term and candidates."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "本", "Reading": "ほん", "Vocab": "猫", "EmptyField": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
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
            dlg.client.uses_catalog = MagicMock(return_value=True)
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


if __name__ == "__main__":
    unittest.main()
