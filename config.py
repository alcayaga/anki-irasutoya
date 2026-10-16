"""Configuration manager for the Anki Irasutoya add-on."""
from typing import Any, Dict

CONFIG_KEY_SOURCE = "last_source_field"
CONFIG_KEY_TARGET = "last_target_field"
CONFIG_KEY_OVERWRITE = "last_overwrite_mode"
CONFIG_KEY_THUMB_DELAY = "thumbnail_delay_ms"
CONFIG_KEY_FULL_DELAY = "full_image_delay_ms"
CONFIG_KEY_IMAGE_SIZE = "image_size"

DEFAULT_CONFIG: Dict[str, Any] = {
    CONFIG_KEY_SOURCE: "Expression",
    CONFIG_KEY_TARGET: "DefinitionPicture",
    CONFIG_KEY_OVERWRITE: "skip",  # "skip" or "overwrite"
    CONFIG_KEY_THUMB_DELAY: 50,  # 50ms fast delay for gallery thumbnails
    CONFIG_KEY_FULL_DELAY: 300,  # 300ms polite delay for batch full image downloads
    CONFIG_KEY_IMAGE_SIZE: "s800",  # "s800" or "s0"
}

_memory_config: Dict[str, Any] = {}


def get_config() -> Dict[str, Any]:
    """Retrieve current add-on configuration, merged with defaults."""
    cfg: Dict[str, Any] = {}

    try:
        from aqt import mw
        if mw and hasattr(mw, "addonManager") and mw.addonManager:
            addon_pkg = mw.addonManager.addonFromModule(__name__) or __name__.split(".")[0]
            addon_cfg = mw.addonManager.getConfig(addon_pkg)
            if addon_cfg:
                cfg.update(addon_cfg)
        else:
            cfg.update(_memory_config)
    except Exception:
        # Fall back to memory config during tests/headless
        cfg.update(_memory_config)

    # Ensure all defaults are present
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    return merged


def save_config(config: Dict[str, Any]) -> None:
    """Persist updated configuration."""
    _memory_config.update(config)

    try:
        from aqt import mw
        if mw and hasattr(mw, "addonManager") and mw.addonManager:
            addon_pkg = mw.addonManager.addonFromModule(__name__) or __name__.split(".")[0]
            mw.addonManager.writeConfig(addon_pkg, config)
    except Exception:
        pass


def update_user_preferences(
    source_field: str,
    target_field: str,
    overwrite_mode: str
) -> None:
    """Save user choices for field mapping and overwrite policy."""
    cfg = get_config()
    cfg[CONFIG_KEY_SOURCE] = source_field
    cfg[CONFIG_KEY_TARGET] = target_field
    cfg[CONFIG_KEY_OVERWRITE] = overwrite_mode
    save_config(cfg)
