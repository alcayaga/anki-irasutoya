"""Unit tests for top-level add-on lifecycle and headless import safety."""
from unittest.mock import MagicMock, patch
import sys


class TestInitAddon:
    """Validate __init__.py import safety and hook registration."""

    def test_has_aqt_boolean_exposed(self):
        """Ensure HAS_AQT is exposed as a boolean flag."""
        import __init__ as addon_init
        assert isinstance(addon_init.HAS_AQT, bool)

    def test_init_addon_noop_when_has_aqt_false(self):
        """Ensure init_addon safely no-ops without exception when HAS_AQT is False."""
        import __init__ as addon_init
        with patch.object(addon_init, "HAS_AQT", False):
            # Calling init_addon should return cleanly without requiring gui_hooks
            addon_init.init_addon()

    def test_init_addon_registers_hooks_when_has_aqt_true(self):
        """Ensure init_addon registers both hooks when HAS_AQT is True."""
        import __init__ as addon_init
        mock_gui_hooks = MagicMock()
        mock_gui_hooks.browser_menus_did_init = []
        mock_gui_hooks.browser_will_show_context_menu = []

        with patch.object(addon_init, "HAS_AQT", True), \
             patch.object(addon_init, "gui_hooks", mock_gui_hooks, create=True):
            addon_init.init_addon()
            assert addon_init.on_browser_menus_did_init in mock_gui_hooks.browser_menus_did_init
            assert addon_init.on_browser_will_show_context_menu in mock_gui_hooks.browser_will_show_context_menu

    def test_on_browser_will_show_context_menu_empty_selection(self):
        """Ensure context menu is untouched when no cards are selected."""
        import __init__ as addon_init
        mock_browser = MagicMock()
        mock_browser.selected_notes.return_value = []
        mock_menu = MagicMock()

        addon_init.on_browser_will_show_context_menu(mock_browser, mock_menu)
        mock_menu.addAction.assert_not_called()
        mock_menu.addSeparator.assert_not_called()
