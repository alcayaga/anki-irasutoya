# Anki Irasutoya Add-on (いらすとや)

Enrich your Anki flashcards with illustrations from [いらすとや (Irasutoya)](https://www.irasutoya.com/) directly from the Anki Browser.

![Anki](https://img.shields.io/badge/Anki-23.10%2B-blue)
[![Tests](https://github.com/alcayaga/anki-irasutoya/actions/workflows/test.yml/badge.svg)](https://github.com/alcayaga/anki-irasutoya/actions/workflows/test.yml)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)

---

## Features

- **Direct Browse Integration**: Select one or more cards in the Anki Browser, right-click, and choose **Add Irasutoya Images...** (or via the **Notes** menu, or `Ctrl+Alt+I` / `Cmd+Option+I`).
- **Modern Apple HIG & Dark Mode Design**:
  - Polished macOS-native aesthetics with smooth rounded controls and vivid system accent buttons.
  - Dedicated white canvas background for thumbnail previews, ensuring transparent PNG illustrations with dark clothing, hair, or outlines remain crisp in dark mode.
  - Vector down-chevron dropdowns with seamless borders.
- **Smart Search & Candidate Fallback Pipeline**:
  - **Japanese Script Normalization**: Automatically normalizes Unicode NFKC, converts loanwords to Katakana (`とまと` $\rightarrow$ `トマト`), expands shortened Katakana vowel marks (`メンバ` $\rightarrow$ `メンバー`), preserves middle dots in compound loanwords (`メッセージ・カード`), and transforms numeric counters to Kanji (`1杯` $\rightarrow$ `一杯`, `2人` $\rightarrow$ `二人`).
  - **Automatic Note Reading Field Discovery**: In Batch Mode, automatically inspects reading fields on the note (`ExpressionReading`, `Reading`, `Word Reading`, `Kana`, or furigana brackets) if the primary Kanji query yields zero illustrations.
  - **Tiered Confidence Strategy & Audit Tagging**:
    - **Tier 1 (Safe Syntactic Transforms)**: Applied automatically with zero homophone risk.
    - **Tier 2 (Semantic & Reading Fallbacks)**: If an image is matched via a phonetic reading fallback in Batch Mode, the note is automatically tagged with `irasutoya::fallback_reading` for effortless one-click auditability in the Anki Browser.
  - **Archetype & Relevance Scoring Engine**: Uses Blogger `orderby=relevance` and a scoring engine that favors canonical, standalone illustrations over negative situations, multi-figure scenarios, or incidental mentions (e.g. `本` selects a clean book rather than tax paperwork).
- **Dual Mode**:
  - **Single Card Interactive Gallery**:
    - Real-time thumbnail gallery with titles and selection checkmark badge.
    - **Visual Transparency & 1-Click Suggestion Chips**: Subtle Google-style notification banner informs you when fallback results are shown, with clickable chips to toggle variations instantly.
    - Full **keyboard navigation**: use Arrow Keys (`←`, `→`, `↑`, `↓`) to browse illustrations and press `Enter` to apply.
    - Native `(✕)` clear button in search input.
    - Double-click any illustration to apply immediately.
    - Direct assignment without unnecessary overwrite warnings or blocking prompts.
  - **Automated Batch Mode**:
    - **Interactive Preview Table**: preview extracted search queries, target field status (`Empty` vs `Has Content`), and planned actions before running.
    - **Zero-Jitter Fixed Layout**: rock-solid table column widths that do not jump or resize when toggling options.
    - **Glanceable Summary Chips**: live counter badges (`● X to Fill`, `● Y to Overwrite/Skip`).
    - **Smart Dynamic CTA**: primary button dynamically reflects active items (e.g. `Start Batch Fill (3)` or disabled `Nothing to Fill (All Skipped)`).
- **Flexible Field Mapping**:
  - Choose any **Source Field** (e.g. `Expression`, `Vocab`, `Front`, `Kanji`).
  - Choose any **Target Field** (e.g. `DefinitionPicture`, `Picture`, `Image`, `Illustration`).
  - Automatically remembers your field selections across sessions in `config.json`.
- **Smart Query Sanitization**:
  - Automatically strips HTML markup, Anki sound tags (`[sound:...]`), furigana brackets (`食べる[たべる]` -> `食べる`), parenthesized readings (`（ねこ）`), and extracts the primary keyword from multi-definition fields (`/`, `;`, `,`).
- **Decoupled Pacing & Server Politeness**:
  - Rapid parallel thumbnail streaming (50ms delay) for instantaneous visual picking.
  - Steady polite delay (300ms) for batch full-resolution downloads to respect Irasutoya's servers and prevent HTTP 429 throttling.
- **Native Undo Support**:
  - Modifies notes via Anki's native operations framework. Press `Cmd+Z` / `Ctrl+Z` to revert batch or single additions at any time.

---

## Installation

### From Source / Symlink (Recommended for Development)
Link this repository into your Anki `addons21` folder:

```bash
# On macOS:
ln -s /path/to/anki-irasutoya ~/Library/Application\ Support/Anki2/addons21/anki-irasutoya

# On Linux:
ln -s /path/to/anki-irasutoya ~/.local/share/Anki2/addons21/anki-irasutoya

# On Windows (PowerShell Run as Admin):
New-Item -ItemType SymbolicLink -Path "$env:APPDATA\Anki2\addons21\anki-irasutoya" -Target "C:\path\to\anki-irasutoya"
```

Restart Anki after linking.

---

## How to Use

1. Open the **Browse** screen in Anki (`B`).
2. Select one or more cards in the list.
3. Either:
   - Right-click and choose **Add Irasutoya Images...**, or
   - Click the top menu bar: **Notes** -> **Add Irasutoya Images...**, or
   - Press `Ctrl+Alt+I` (`Cmd+Option+I` on Mac).
4. **Single Card**:
   - Browse the illustrations using your mouse or Arrow Keys (`←`, `→`, `↑`, `↓`).
   - Press `Enter` or click **Apply Selected Image** (or double-click) to insert.
5. **Multiple Cards (Batch)**:
   - Choose whether to **Skip** cards that already have images or **Overwrite** them.
   - Check the **Preview Table** and **Summary Chips** to verify what will happen.
   - Click **Start Batch Fill** to automatically fetch and insert illustrations.
6. To undo at any time, press `Cmd+Z` or `Ctrl+Z` in Anki.

---

## Testing & Evaluation

### Hermetic Unit Tests

Run fast, offline unit tests without network or Anki dependencies:

```bash
python3 -m unittest discover tests -v
```

Or run with Anki's bundled packages (includes PyQt6 GUI tests):

```bash
PYTHONPATH="/Applications/Anki.app/Contents/Resources/app_packages:." python3 -m unittest discover tests -v
```

### Live Deck Search Benchmark CLI

Evaluate and measure real search recall improvements against your actual Anki decks via AnkiConnect (read-only, non-destructive):

```bash
python3 scripts/eval_deck_search.py --deck "Mining" --sort "FreqSort" --limit 50 --verbose
```

For detailed developer setup, testing philosophies, and contribution standards, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (GNU AGPLv3).

Illustrations are copyright (c) Takashi Mifune ([いらすとや](https://www.irasutoya.com/)). Please adhere to Irasutoya's terms of service when using illustrations.
