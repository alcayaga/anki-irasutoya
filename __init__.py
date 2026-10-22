"""Anki Irasutoya Add-on: Fill cards with Irasutoya illustrations from the Browser screen."""
import logging

logger = logging.getLogger(__name__)

try:
    from aqt import gui_hooks, mw
    from aqt.qt import QAction, QKeySequence, qconnect
    from aqt.utils import showInfo
    HAS_AQT = True
except (ImportError, RuntimeError) as _aqt_exc:
    HAS_AQT = False
    logger.debug("Anki GUI APIs unavailable, hooks not registered: %s", _aqt_exc)


def on_add_irasutoya_images(browser) -> None:
    """Invoked when user clicks 'Add Irasutoya Images...'."""
    try:
        from .ui import IrasutoyaDialog
    except ImportError:
        from ui import IrasutoyaDialog

    selected_nids = browser.selected_notes()
    if not selected_nids:
        showInfo("Please select at least one card in the browser list.", parent=browser)
        return

    dialog = IrasutoyaDialog(parent=browser, browser=browser, note_ids=selected_nids)
    dialog.exec()


def on_browser_menus_did_init(browser) -> None:
    """Hook called when the Browser window builds its top menu bar."""
    action = QAction("Add Irasutoya Images...", browser)
    action.setShortcut(QKeySequence("Ctrl+Alt+I"))
    qconnect(action.triggered, lambda: on_add_irasutoya_images(browser))

    # Add to the Notes menu in Browser
    if hasattr(browser.form, "menu_Notes") and browser.form.menu_Notes:
        browser.form.menu_Notes.addSeparator()
        browser.form.menu_Notes.addAction(action)
    elif hasattr(browser.form, "menuEdit") and browser.form.menuEdit:
        browser.form.menuEdit.addAction(action)


def on_browser_will_show_context_menu(browser, menu) -> None:
    """Hook called when user right-clicks selected cards in Browser table."""
    selected_nids = browser.selected_notes()
    if not selected_nids:
        return

    action = QAction("Add Irasutoya Images...", menu)
    qconnect(action.triggered, lambda: on_add_irasutoya_images(browser))
    menu.addSeparator()
    menu.addAction(action)


def init_addon() -> None:
    """Register all hooks with Anki if GUI environment is available."""
    if not HAS_AQT:
        return
    gui_hooks.browser_menus_did_init.append(on_browser_menus_did_init)
    gui_hooks.browser_will_show_context_menu.append(on_browser_will_show_context_menu)


# Register hooks on load if running inside Anki
if HAS_AQT:
    init_addon()
