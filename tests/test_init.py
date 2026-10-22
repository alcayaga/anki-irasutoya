"""Unit tests for top-level add-on lifecycle and headless import safety."""
import builtins
import importlib
import sys
from unittest.mock import MagicMock, patch


class TestInitAddon:
    """Validate __init__.py import safety and hook registration."""

    def test_has_aqt_boolean_exposed(self):
        """Ensure HAS_AQT is exposed as a boolean flag."""
        import __init__ as addon_init
        assert isinstance(addon_init.HAS_AQT, bool)

    def test_headless_fresh_import_sets_has_aqt_false(self):
        """Ensure fresh import without aqt sets HAS_AQT to False and registers no hooks."""
        orig_init = sys.modules.get("__init__")
        orig_aqt = sys.modules.get("aqt")

        try:
            sys.modules.pop("__init__", None)
            with patch.dict(sys.modules, {"aqt": None}):
                addon_init = importlib.import_module("__init__")
                assert addon_init.HAS_AQT is False
        finally:
            if orig_init is not None:
                sys.modules["__init__"] = orig_init
            else:
                sys.modules.pop("__init__", None)
            if orig_aqt is not None:
                sys.modules["aqt"] = orig_aqt
            else:
                sys.modules.pop("aqt", None)

    def test_headless_fresh_import_handles_runtime_error(self):
        """Ensure a RuntimeError while importing aqt sets HAS_AQT to False."""
        orig_init = sys.modules.get("__init__")
        real_import = builtins.__import__

        def import_raising_runtime_error(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "aqt":
                raise RuntimeError("aqt GUI unavailable")
            return real_import(name, globals, locals, fromlist, level)

        try:
            sys.modules.pop("__init__", None)
            with patch.object(
                builtins, "__import__", side_effect=import_raising_runtime_error
            ):
                addon_init = importlib.import_module("__init__")
                assert addon_init.HAS_AQT is False
        finally:
            if orig_init is not None:
                sys.modules["__init__"] = orig_init
            else:
                sys.modules.pop("__init__", None)

    def test_init_addon_noop_when_has_aqt_false(self):
        """Ensure init_addon safely no-ops without exception when HAS_AQT is False."""
        import __init__ as addon_init
        with patch.object(addon_init, "HAS_AQT", new=False):
            # Calling init_addon should return cleanly without requiring gui_hooks
            addon_init.init_addon()

    def test_fresh_import_with_aqt_registers_hooks(self):
        """Ensure fresh import with mocked aqt sets HAS_AQT to True and registers hooks."""
        orig_init = sys.modules.get("__init__")
        orig_aqt = sys.modules.get("aqt")
        orig_aqt_qt = sys.modules.get("aqt.qt")
        orig_aqt_utils = sys.modules.get("aqt.utils")

        mock_aqt = MagicMock()
        mock_aqt.gui_hooks = MagicMock()
        mock_aqt.gui_hooks.browser_menus_did_init = []
        mock_aqt.gui_hooks.browser_will_show_context_menu = []
        mock_qt = MagicMock()
        mock_utils = MagicMock()

        try:
            sys.modules.pop("__init__", None)
            with patch.dict(
                sys.modules,
                {
                    "aqt": mock_aqt,
                    "aqt.qt": mock_qt,
                    "aqt.utils": mock_utils,
                },
            ):
                addon_init = importlib.import_module("__init__")
                assert addon_init.HAS_AQT is True
                assert (
                    addon_init.on_browser_menus_did_init
                    in mock_aqt.gui_hooks.browser_menus_did_init
                )
                assert (
                    addon_init.on_browser_will_show_context_menu
                    in mock_aqt.gui_hooks.browser_will_show_context_menu
                )
        finally:
            if orig_init is not None:
                sys.modules["__init__"] = orig_init
            else:
                sys.modules.pop("__init__", None)
            for mod_name, orig in [
                ("aqt", orig_aqt),
                ("aqt.qt", orig_aqt_qt),
                ("aqt.utils", orig_aqt_utils),
            ]:
                if orig is not None:
                    sys.modules[mod_name] = orig
                else:
                    sys.modules.pop(mod_name, None)

    def test_init_addon_registers_hooks_when_has_aqt_true(self):
        """Ensure init_addon registers both hooks when HAS_AQT is True."""
        import __init__ as addon_init
        mock_gui_hooks = MagicMock()
        mock_gui_hooks.browser_menus_did_init = []
        mock_gui_hooks.browser_will_show_context_menu = []

        with patch.object(addon_init, "HAS_AQT", new=True), \
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
