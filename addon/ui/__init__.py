"""User interface package for the Anki Irasutoya add-on."""
from aqt import mw

from .dialog import IrasutoyaDialog
from .helpers import (
    _description_match_term,
    _is_desc_only_match,
    _StaleSearch,
    format_fallback_reason,
    has_existing_media,
    make_description_snippet,
    probe_catalog_candidates,
    probe_catalog_term,
)
from .styles import (
    CHEVRON_ICON_PATH,
    DIALOG_STYLESHEET,
    ICONS_DIR,
    MAX_SUGGESTION_CHIPS,
    PAGE_SIZE,
    THUMB_CACHE_MAX_BYTES,
    THUMB_CACHE_MAX_ENTRIES,
    THUMB_INITIAL_BURST,
    THUMB_WORKER_COUNT,
)
from .thumbnails import (
    _SESSION_THUMB_CACHE,
    ThumbnailLoader,
    _ThumbnailByteCache,
    clear_session_thumbnail_cache,
)
from .widgets import ImageCardWidget, populate_suggestion_chips

__all__ = [
    "CHEVRON_ICON_PATH",
    "DIALOG_STYLESHEET",
    "ICONS_DIR",
    "ImageCardWidget",
    "IrasutoyaDialog",
    "MAX_SUGGESTION_CHIPS",
    "PAGE_SIZE",
    "THUMB_CACHE_MAX_BYTES",
    "THUMB_CACHE_MAX_ENTRIES",
    "THUMB_INITIAL_BURST",
    "THUMB_WORKER_COUNT",
    "ThumbnailLoader",
    "_SESSION_THUMB_CACHE",
    "_StaleSearch",
    "_ThumbnailByteCache",
    "_description_match_term",
    "_is_desc_only_match",
    "clear_session_thumbnail_cache",
    "format_fallback_reason",
    "has_existing_media",
    "make_description_snippet",
    "mw",
    "populate_suggestion_chips",
    "probe_catalog_candidates",
    "probe_catalog_term",
]
