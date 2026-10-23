# Contributing to anki-irasutoya

Thank you for your interest in contributing to **anki-irasutoya**! This guide details development workflows, testing philosophy, evaluation scripts, and architectural standards.

---

## Architecture Overview

The codebase is organized into cleanly decoupled layers inside the `addon/` package:

| Module | Description | Dependencies |
| :--- | :--- | :--- |
| [`addon/text_utils.py`](addon/text_utils.py) | Query sanitization, Japanese script normalization (NFKC, loanword Katakana, chōonpu expansion, Kanji numeral conversion, heuristic deinflection), and Tiered candidate generation. | Standard Library only (`re`, `unicodedata`, `html`). **No Anki/Qt dependency.** |
| [`addon/yomitan_parser.py`](addon/yomitan_parser.py) | Structured JMdict POS, orthography, and furigana table extraction from Yomitan HTML. | Standard Library only (`html.parser`). **No Anki/Qt dependency.** |
| [`addon/feed.py`](addon/feed.py) | Blogger feed HTML parser, high-res URL upgrading, thumbnail helpers, and entry models. | Standard Library only (`urllib`, `html.parser`, `unicodedata`). **No Anki/Qt dependency.** |
| [`addon/ranking.py`](addon/ranking.py) | Morphological boundary classification, noise keyword filtering, and archetype relevance scoring engine. | Standard Library + `text_utils` + `feed`. **No Anki/Qt dependency.** |
| [`addon/client.py`](addon/client.py) | Irasutoya scraper client, HTTPS redirect validation, and candidate fallback search pipeline. | Standard Library + `feed` + `ranking`. **No Anki/Qt dependency.** |
| [`addon/batch.py`](addon/batch.py) | Batch processing worker, preview table data models, rate-limited downloads, and audit tagging (`irasutoya::fallback_reading`). | Anki collection operations & UI abstractions. |
| [`addon/ui.py`](addon/ui.py) | Apple HIG native modal dialogs, visual thumbnail gallery, suggestion chips, transparency banners, and batch configuration views. | `aqt` (PyQt6). |
| [`addon/config.py`](addon/config.py) | User configuration management with schema fallback defaults. | `aqt.mw.addonManager`. |
| [`scripts/eval_deck_search.py`](scripts/eval_deck_search.py) | Standalone CLI tool to evaluate search recall against real Anki decks via AnkiConnect. | Standard Library + `addon.*`. |

---

## Development & Environment Setup

### 1. Linking Add-on for Development

To test live in Anki, symlink the `addon` directory directly into your Anki `addons21` directory:

```bash
# macOS
ln -s "$(pwd)/addon" ~/Library/Application\ Support/Anki2/addons21/anki-irasutoya

# Linux
ln -s "$(pwd)/addon" ~/.local/share/Anki2/addons21/anki-irasutoya

# Windows (PowerShell as Administrator)
New-Item -ItemType SymbolicLink -Path "$env:APPDATA\Anki2\addons21\anki-irasutoya" -Target "$((Get-Location).Path)\addon"
```

Restart Anki to load changes.

### 2. Local Conda Development Environment

We provide a standard [`environment.yml`](environment.yml) matching modern Anki's bundled Python runtime (Python 3.13) with `ruff` and `aqt` pre-configured:

```bash
# Create and activate development environment
conda env create -f environment.yml
conda activate anki-irasutoya
```

Inside this environment, you can lint and run the full test suite (including Qt UI tests) directly:

```bash
# Run linter
ruff check .

# Run all unit tests (including Qt UI dialogs)
python -m unittest discover tests -v
```

---

## Testing Standards & Philosophy

We enforce strict separation between **Hermetic Unit Tests** and **Live Integration/Evaluation Tools**.

### 1. Hermetic Unit Tests (`tests/`)

- **Strictly Offline**: Unit tests in `tests/` must **never** make live network requests to Irasutoya or Google Blogger feeds.
- **Hermetic & Independent**: Tests must not assume Anki is running or that AnkiConnect is accessible.
- **Mocking Strategy**: Use stubbed responses or mock clients (such as `MockIrasutoyaClient`) to verify logic deterministically in milliseconds.
- **Coverage**:
  - `tests/test_text_utils.py`: Text cleaning, middle dot preservation, numeral conversion, katakana loanwords, vowel elongation, heuristic deinflection, candidate tiers.
  - `tests/test_yomitan_parser.py`: Yomitan HTML parsing, JMdict POS tags, uk orthography, and formsTable promotion.
  - `tests/test_feed.py`: Blogger HTML parsing, standalone vs anchor `<img>` tags, HTML unescaping, URL upgrades.
  - `tests/test_ranking.py`: Boundary classification, noise keyword filtering, archetype relevance scoring, regression fixtures.
  - `tests/test_client.py`: HTTP request handling, HTTPS redirect security, and fallback search pipeline.
  - `tests/test_batch_logic.py`: Skip/overwrite logic, safe filename generation, fallback audit tagging (`irasutoya::fallback_reading`).
  - `tests/test_config.py`: Fallback defaults and schema access.
  - `tests/test_ui_mode.py`: Visual picker dialogs, preview tables, and Apple HIG components.
  - `tests/test_init.py`: Headless import safety and Anki browser hook registration.

#### Running Unit Tests

Run all unit tests using pytest (recommended):

```bash
python3 -m pytest tests -v
```

Or using standard `unittest`:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Or run with Anki's bundled Python packages (to execute Qt/UI tests):

```bash
PYTHONPATH="/Applications/Anki.app/Contents/Resources/app_packages:." python3 -m pytest tests -v
```

---

## Live Deck Evaluation Tool (`scripts/eval_deck_search.py`)

To evaluate search recall against real Japanese vocabulary decks without modifying notes, use the non-destructive benchmark CLI:

### Prerequisites

1. Ensure Anki is running with the [AnkiConnect](https://ankiweb.net/shared/info/2055492111) add-on enabled (port 8765).
2. Have a target deck imported (e.g. `Mining`, `Core 2.3k`, etc.).

### Usage

```bash
# Basic benchmark on the first 50 cards of the "Mining" deck sorted by "FreqSort"
python3 scripts/eval_deck_search.py --deck "Mining" --sort "FreqSort" --limit 50

# Verbose output showing per-word fallback chains and matched illustration titles
python3 scripts/eval_deck_search.py --deck "Mining" --sort "FreqSort" --limit 50 --verbose

# Benchmark with Tier 2 semantic/reading fallbacks disabled (pure Tier 1 syntactic recall)
python3 scripts/eval_deck_search.py --deck "Mining" --limit 50 --disable-tier2

# Polite delay between live queries (default: 0.25s)
python3 scripts/eval_deck_search.py --deck "Mining" --limit 100 --delay 0.5
```

### CLI Parameters

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--deck` | `"Mining"` | Anki deck name to sample cards from. |
| `--sort` | `"FreqSort"` | Card field used for ordering notes (e.g., frequency sort index). |
| `--source-field` | Auto-detect | Source vocabulary field (`Expression`, `Word`, `Vocab`, `Front`). |
| `--limit` | `50` | Maximum number of notes to evaluate. |
| `--delay` | `0.25` | Polite delay (in seconds) between live HTTP requests to Irasutoya. |
| `--disable-tier2` | False | Disables Tier 2 reading fallbacks to test only syntactic normalizations. |
| `--verbose`, `-v` | False | Displays detailed fallback candidate chains and image titles. |
| `--anki-url` | `http://localhost:8765` | URL of AnkiConnect API endpoint. |

The evaluation script prints an immediate visual progress log, followed by a summary report displaying **Legacy Search Recall**, **New Search Recall**, **Absolute/Relative Recall Gains**, and an **Exemplars Table** of recovered vocabulary.

---

## Tiered Confidence Strategy

To avoid false positive image matches during automated Batch operations (the **"Homophone Trap"**), searches are categorized into two tiers:

- **Tier 1 (Safe Syntactic Normalization)**:
  - Unicode NFKC normalization (zenkaku/hankaku).
  - Loanword Katakana conversion (`とまと` $\rightarrow$ `トマト`).
  - Kanji numeral conversion (`1杯` $\rightarrow$ `一杯`, `2人` $\rightarrow$ `二人`).
  - *Applied automatically with zero homophone risk.*

- **Tier 2 (Phonetic & Semantic Fallbacks)**:
  - Separate note reading fields (`ExpressionReading`, `Reading`, `Word Reading`, `Kana`).
  - Same-field bracketed furigana readings (`林檎[りんご]` $\rightarrow$ `りんご`, `リンゴ`).
  - Katakana elongation mark expansion (`メンバ` $\rightarrow$ `メンバー`, `サーバ` $\rightarrow$ `サーバー`).
  - Verb and compound heuristic deinflections (`帰ってくる` $\rightarrow$ `帰る`, `走っている` $\rightarrow$ `走る`) (available in Single Card Picker UI; disabled in Batch Mode to avoid semantic drift).
  - In **Single Card Visual Picker UI**: Displays a transparent notification banner with clickable suggestion chips to switch queries in 1 click.
  - In **Batch Mode**: Automatically applies reading fallbacks, and **tags the note with `irasutoya::fallback_reading`** so users can filter and review fallback images at their convenience (heuristic deinflection is disabled in Batch Mode).

---

## Coding Standards

- **Code Quality & Linting**: All code must pass `ruff check .` with zero errors.
- **Python 3.9+ Compatibility**: Maintain syntax and runtime compatibility across modern Anki versions (23.10+).
- **Type Annotations**: Provide explicit type hints for public functions and class members.
- **Preserve Documentation**: Maintain docstrings and inline commentary.
- **Defensive Parsing**: Use `html.unescape()` when extracting image URLs or titles from Blogger feeds to avoid broken `&amp;` attributes.
- **Security**: Strictly validate URLs against trusted domains (`bp.blogspot.com`, `*.googleusercontent.com`, `irasutoya.com`) and enforce HTTPS.
