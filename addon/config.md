# Anki Irasutoya Add-on Configuration

This add-on allows you to automatically enrich your cards with illustrations from **いらすとや (Irasutoya)**.

### Settings:
- `last_source_field`: The last used source field (default: `Expression`).
- `last_target_field`: The last used target field to insert the image into (default: `DefinitionPicture`).
- `last_overwrite_mode`: Either `"skip"` (skip if target field already has content) or `"overwrite"`.
- `thumbnail_delay_ms`: Rapid delay in milliseconds between gallery preview thumbnail downloads (default: `50`).
- `full_image_delay_ms`: Polite delay in milliseconds between consecutive full image card downloads in batch mode (default: `300`).
- `image_size`: Resolution of fetched images (`"s800"` for max 800px width/height, or `"s0"` for full original size).

## Note-type compatibility and metadata fields

Lapis and Kiku automatically use `Expression` as the source and `DefinitionPicture`
as the destination. Senren uses `word` and `picture`. Valid saved source and target
choices take precedence. Senren's `picture` field can already contain mining
screenshots; the existing skip/overwrite policy still applies. No note fields or
templates are changed. Select notes with a common source/target layout for batch
operations: source and target selections apply to the whole batch.

Reading, furigana, definition, and selected-text fields are recognized by their
standard aliases, including Senren's lowercase `reading`, `definition`, `glossary`,
and `selectionText`. Each alias prefers an exact match, then a unique
case-insensitive match. Ambiguous matches are never guessed. Sentence translations
and sentence furigana are not vocabulary aliases.

For renamed fields, edit **Tools → Add-ons → Irasutoya → Config** and add optional
`note_type_field_overrides`, keyed by the exact note-type name:

```json
{
  "note_type_field_overrides": {
    "My Custom Senren": {
      "definition_fields": ["MyDefinition", "glossary"],
      "reading_fields": ["MyReading"],
      "furigana_fields": []
    }
  }
}
```

- Omitted roles use automatic detection. Lists replace automatic detection in the
  listed order. Empty lists disable that role.
- `definition_fields` governs structured Yomitan metadata, plain English glosses,
  and bilingual part-of-speech safeguards. When explicitly configured, only those
  fields are inspected: other definitions and selected text cannot re-enter through
  fallback scanning. Automatic mode retains the structured-markup fallback scan.
- Configured field names use the same exact/unique case-insensitive matching.
  Missing or ambiguous fields are skipped with a configuration warning; automatic
  fields do not replace them. Correct the names in Config to remove the warning.
- Invalid objects, unsupported roles, or values other than lists of nonempty field
  names block searches with an error explaining how to fix the configuration.
- Each note uses its own note-type overrides, including in batch operations.
  Rename the configuration key if you rename a note type. Reopen the dialog after
  editing configuration. Existing installations need no migration.

These overrides affect metadata only. Source and destination fields remain
configurable in the image dialog. The standalone benchmark uses automatic metadata
recognition; it does not read a running Anki installation's private add-on settings.
