"""Visual styles, UI sizing constants, and stylesheet templates for Irasutoya dialogs."""
import os

# Sizing and pagination constants
PAGE_SIZE: int = 24
MAX_SUGGESTION_CHIPS: int = 8
THUMB_CACHE_MAX_ENTRIES: int = 512
THUMB_CACHE_MAX_BYTES: int = 8 * 1024 * 1024  # 8 MB hard ceiling
DIALOG_PIXMAP_CACHE_MAX_ENTRIES: int = 128
THUMB_WORKER_COUNT: int = 6
THUMB_INITIAL_BURST: int = 4

# Icon asset paths relative to addon directory
ADDON_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICONS_DIR: str = os.path.join(ADDON_DIR, "icons")
CHEVRON_ICON_PATH: str = os.path.join(ICONS_DIR, "chevron_down.svg").replace("\\", "/")

# Primary dialog stylesheet (dark/light mode compatible via palette tokens)
DIALOG_STYLESHEET: str = f"""
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
    color: rgba(120, 120, 128, 0.6);
    background-color: rgba(120, 120, 128, 0.08);
    border: 1px solid rgba(120, 120, 128, 0.15);
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

# Reusable widget stylesheets
CARD_SELECTED_STYLE: str = (
    "ImageCardWidget {"
    "   border: 2px solid #0A84FF;"
    "   background-color: rgba(10, 132, 255, 0.12);"
    "   border-radius: 8px;"
    "}"
)

CARD_DEFAULT_STYLE: str = (
    "ImageCardWidget {"
    "   border: 1px solid rgba(120, 120, 128, 0.25);"
    "   background-color: palette(alternate-base);"
    "   border-radius: 8px;"
    "}"
    "ImageCardWidget:hover {"
    "   border: 1.5px solid #0A84FF;"
    "}"
)

CARD_THUMB_STYLE: str = (
    "background-color: #FFFFFF; "
    "color: #8E8E93; "
    "border-radius: 6px; "
    "font-size: 11px;"
)

CARD_TITLE_STYLE: str = "font-size: 11px; font-weight: 500; color: palette(text);"

CARD_BADGE_STYLE: str = (
    "background-color: #0A84FF; "
    "color: #FFFFFF; "
    "border-radius: 9px; "
    "font-size: 11px; "
    "font-weight: bold; "
    "border: 1.5px solid palette(window);"
)

CARD_DESC_MATCH_BADGE_STYLE: str = (
    "background-color: rgba(0, 0, 0, 0.68); "
    "color: #FFD60A; "
    "border-radius: 4px; "
    "font-size: 10px; "
    "font-weight: 600; "
    "padding: 1px 4px;"
)

HEADER_COUNT_BADGE_STYLE: str = (
    "background-color: rgba(120, 120, 128, 0.18); "
    "color: palette(text); "
    "padding: 4px 10px; "
    "border-radius: 10px; "
    "font-size: 11px;"
)

FIELDS_FRAME_STYLE: str = (
    "QFrame#fieldsFrame {"
    "   background-color: palette(alternate-base);"
    "   border: 1px solid rgba(120, 120, 128, 0.2);"
    "   border-radius: 8px;"
    "}"
)

CHIP_FILL_STYLE: str = (
    "background-color: rgba(48, 209, 88, 0.15); "
    "color: #30D158; "
    "font-weight: 600; font-size: 11px; "
    "padding: 3px 10px; border-radius: 10px;"
)

CHIP_SKIP_STYLE: str = (
    "background-color: rgba(120, 120, 128, 0.18); "
    "color: #8E8E93; "
    "font-weight: 600; font-size: 11px; "
    "padding: 3px 10px; border-radius: 10px;"
)

CHIP_OVERWRITE_STYLE: str = (
    "background-color: rgba(10, 132, 255, 0.15); "
    "color: #0A84FF; "
    "font-weight: 600; font-size: 11px; "
    "padding: 3px 10px; border-radius: 10px;"
)

BATCH_TABLE_STYLE: str = (
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

CHIPS_SCROLL_STYLE: str = (
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

CHIP_RESTORE_STYLE: str = (
    "background-color: rgba(10, 132, 255, 0.12); "
    "border: 1px dashed #0A84FF; border-radius: 10px; "
    "padding: 2px 8px; font-size: 11px; font-weight: 500; color: #0A84FF;"
)

CHIP_ACTIVE_STYLE: str = (
    "background-color: rgba(10, 132, 255, 0.18); "
    "border: 1px solid #0A84FF; border-radius: 10px; "
    "padding: 2px 8px; font-size: 11px; font-weight: 600; color: #0A84FF;"
)

CHIP_INACTIVE_STYLE: str = (
    "background-color: rgba(120, 120, 128, 0.15); "
    "border: 1px solid rgba(120, 120, 128, 0.2); border-radius: 10px; "
    "padding: 2px 8px; font-size: 11px; color: palette(text);"
)

GALLERY_SCROLL_STYLE: str = (
    "QScrollArea {"
    "   border: 1px solid rgba(120, 120, 128, 0.2);"
    "   border-radius: 8px;"
    "   background-color: palette(base);"
    "}"
)

STATUS_MUTED_STYLE: str = "color: #98989D; font-size: 12px;"
STATUS_ERROR_STYLE: str = "color: #FF453A; font-size: 12px;"
STATUS_ACTIVE_STYLE: str = "color: palette(text); font-size: 12px; font-weight: 500;"
STATUS_INFO_BLUE_STYLE: str = "color: #0A84FF; font-size: 12px; font-weight: 500;"
