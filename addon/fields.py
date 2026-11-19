"""Headless field-name resolution and explicit metadata selection."""
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple

DEFINITION_FIELDS = (
    "MainDefinition", "Glossary", "Definition", "Definitions", "Meanings", "Meaning", "Word Meaning",
)
READING_FIELDS = ("ExpressionReading", "Reading", "Word Reading", "WordReading", "VocabReading", "Kana", "KanaReading")
FURIGANA_FIELDS = ("ExpressionFurigana", "Word Furigana", "WordFurigana", "Furigana", "VocabFurigana")
GLOSS_FIELDS = ("English", "SelectionText")


class FieldConfigurationError(ValueError):
    """Invalid metadata override that must be corrected before searching."""


@dataclass(frozen=True)
class FieldSelection:
    """Resolved metadata fields; None means automatic and an empty tuple disables a role."""

    definition_fields: Optional[Tuple[str, ...]] = None
    reading_fields: Optional[Tuple[str, ...]] = None
    furigana_fields: Optional[Tuple[str, ...]] = None
    warnings: Tuple[str, ...] = ()


def resolve_fields(available: Iterable[str], aliases: Iterable[str]) -> List[str]:
    """Resolve aliases in priority order, preferring exact then unique casefold matches."""
    names = list(available)
    result: List[str] = []
    for alias in aliases:
        matches = [name for name in names if name.casefold() == alias.casefold()]
        match = alias if alias in names else (matches[0] if len(matches) == 1 else None)
        if match is not None and match not in result:
            result.append(match)
    return result


def metadata_fields(note: Any, role: str, selection: Optional[FieldSelection] = None) -> List[str]:
    """Return explicit fields or automatic vocabulary aliases for one metadata role."""
    explicit = getattr(selection, role, None)
    if explicit is not None:
        return list(explicit)
    aliases = {"definition_fields": DEFINITION_FIELDS, "reading_fields": READING_FIELDS,
               "furigana_fields": FURIGANA_FIELDS}[role]
    return resolve_fields(note.keys(), aliases) if note is not None and hasattr(note, "keys") else []


def gloss_fields(note: Any, selection: Optional[FieldSelection] = None) -> List[str]:
    """Return vocabulary definitions plus automatic auxiliary English gloss fields."""
    fields = metadata_fields(note, "definition_fields", selection)
    if selection is None or selection.definition_fields is None:
        fields += resolve_fields(note.keys(), GLOSS_FIELDS) if hasattr(note, "keys") else []
    return list(dict.fromkeys(fields))


def validate_overrides(overrides: Any) -> None:
    """Reject malformed configuration rather than silently using automatic metadata."""
    prefix = "Fix note_type_field_overrides in the Irasutoya add-on configuration"
    if not isinstance(overrides, dict):
        raise FieldConfigurationError(f"{prefix}: expected an object keyed by note-type name.")
    roles = {"definition_fields", "reading_fields", "furigana_fields"}
    for name, entry in overrides.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(entry, dict):
            raise FieldConfigurationError(f"{prefix}: each note type must have an object of field lists.")
        for role, names in entry.items():
            if role not in roles or not isinstance(names, list) or any(
                not isinstance(field, str) or not field.strip() for field in names
            ):
                raise FieldConfigurationError(f"{prefix}: {name!r}.{role} must be a supported role with a list of names.")


def select_fields(available: Iterable[str], model_name: str, overrides: Mapping[str, Any]) -> FieldSelection:
    """Resolve validated per-model overrides and describe missing or ambiguous names."""
    validate_overrides(overrides)
    names = list(available)
    values = {}
    warnings = []
    for role, aliases in overrides.get(model_name, {}).items():
        values[role] = tuple(resolve_fields(names, aliases))
        for alias in aliases:
            if not resolve_fields(names, [alias]):
                warnings.append(
                    f"{model_name}: {role} field {alias!r} is missing or ambiguous; skipped. "
                    "Update note_type_field_overrides in the Irasutoya add-on configuration."
                )
    return FieldSelection(**values, warnings=tuple(warnings))
