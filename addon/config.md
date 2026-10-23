# Anki Irasutoya Add-on Configuration

This add-on allows you to automatically enrich your cards with illustrations from **いらすとや (Irasutoya)**.

### Settings:
- `last_source_field`: The last used source field (default: `Expression`).
- `last_target_field`: The last used target field to insert the image into (default: `DefinitionPicture`).
- `last_overwrite_mode`: Either `"skip"` (skip if target field already has content) or `"overwrite"`.
- `thumbnail_delay_ms`: Rapid delay in milliseconds between gallery preview thumbnail downloads (default: `50`).
- `full_image_delay_ms`: Polite delay in milliseconds between consecutive full image card downloads in batch mode (default: `300`).
- `image_size`: Resolution of fetched images (`"s800"` for max 800px width/height, or `"s0"` for full original size).
