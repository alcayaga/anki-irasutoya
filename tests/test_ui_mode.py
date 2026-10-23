import os
import unittest
from unittest.mock import MagicMock, patch

try:
    from aqt.qt import QApplication, QHeaderView, QPushButton
    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    from addon.feed import IrasutoyaImage
    from addon.ui import CHEVRON_ICON_PATH, DIALOG_STYLESHEET, ImageCardWidget, IrasutoyaDialog
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
            self.assertEqual(chip_texts, ["本", "ほん", "ホン"])
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
            self.assertIn("本", chip_texts_after)
            self.assertEqual(chip_texts_after, ["本", "ほん", "ホン"])
            self.assertFalse(dlg.chips_widget.isHidden())

            # Active chip for "ほん" should have active accent styling
            active_btn = next(b for b in chip_buttons_after if b.text() == "ほん")
            self.assertIn("font-weight: 600", active_btn.styleSheet())
            inactive_btn = next(b for b in chip_buttons_after if b.text() == "本")
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
            self.assertEqual(chip_texts, ["↩ 本", "ねこ", "ネコ"])
            self.assertFalse(dlg.chips_widget.isHidden())

            # Click restore chip
            restore_btn = chip_buttons[0]
            self.assertEqual(restore_btn.text(), "↩ 本")
            restore_btn.click()

            # Dialog should restore search for "本" with original candidates
            self.assertEqual(dlg.query_input.text(), "本")
            chip_buttons_restored = [
                dlg.chips_layout.itemAt(i).widget()
                for i in range(dlg.chips_layout.count())
                if isinstance(dlg.chips_layout.itemAt(i).widget(), QPushButton)
            ]
            chip_texts_restored = [b.text() for b in chip_buttons_restored]
            self.assertEqual(chip_texts_restored, ["本", "ほん", "ホン"])

    def test_suggestion_chips_stay_visible_on_zero_results(self):
        """Verify suggestion chips remain visible when a search returns zero results."""
        mock_mw = MagicMock()
        mock_note = {"Expression": "本", "Reading": "ほん", "DefinitionPicture": ""}
        mock_mw.col.get_note.return_value = mock_note

        with patch("addon.ui.mw", mock_mw), patch("threading.Thread", _SyncThread):
            dlg = IrasutoyaDialog(parent=None, browser=None, note_ids=[1001])
            dlg._thumb_executor.submit = MagicMock()
            dlg._run_on_main = lambda cb: cb()
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
            self.assertEqual([b.text() for b in chip_buttons], ["本", "ほん", "ホン"])

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


if __name__ == "__main__":
    unittest.main()
