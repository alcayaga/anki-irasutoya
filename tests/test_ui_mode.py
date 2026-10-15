import os
import unittest

try:
    from aqt.qt import QApplication, QHeaderView
    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    from ui import IrasutoyaDialog, ImageCardWidget, CHEVRON_ICON_PATH, DIALOG_STYLESHEET
    from client import IrasutoyaImage
    HAS_QT = True
except Exception:
    HAS_QT = False


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


if __name__ == "__main__":
    unittest.main()
