# Agent Instructions for `anki-irasutoya`

Welcome, AI Agent! This document contains essential context about the `anki-irasutoya` project. Please review this before making changes to the codebase to save time on exploration.

## 🎯 Project Overview
`anki-irasutoya` is an Anki desktop add-on (Anki 23.10+, PyQt6) designed to enrich flashcards with illustrations from [いらすとや (Irasutoya)](https://www.irasutoya.com/) directly from the Anki Browser. It provides both a Single Card interactive thumbnail gallery picker and an automated Batch Mode with a preview table, powered by a bundled offline SQLite metadata catalog (~25,000 posts and ~42,000 illustrations), a live Blogger JSON feed fallback scraper, a tiered Japanese query normalization pipeline, and an archetype relevance scoring engine.

## 🏗️ Architecture & Key Files
The codebase is cleanly split between headless, standard-library-only core modules and Anki/PyQt6 (`aqt`) GUI modules inside `addon/`, alongside standalone CLI scripts in `scripts/`:

- **`addon/__init__.py`**: Add-on entrypoint. Guards `aqt` imports via `HAS_AQT` (for headless test/CLI compatibility) and registers Anki Browser menu and right-click context menu hooks (`Add Irasutoya Images...`).
- **`addon/text_utils.py`**: Query sanitization, Japanese script normalization (Unicode NFKC, loanword Katakana conversion, chōonpu vowel expansion, middle-dot preservation, Arabic-to-Kanji counter conversion, heuristic deinflection), and Tier 1 / Tier 2 candidate generation. *(Stdlib only — zero Anki/Qt dependency).*
- **`addon/yomitan_parser.py`**: Extracts structured JMdict part-of-speech (POS) tags, `uk` (usually-kana) orthography markers, and `formsTable` furigana readings from Yomitan HTML definitions. *(Stdlib only).*
- **`addon/feed.py`**: Blogger JSON feed HTML parser, high-res URL upgrading (`s800`, `s0`), thumbnail URL generation (`s120-c`), domain allowlist validation, and data models (`IrasutoyaImage`, `ParsedPost`). *(Stdlib only).*
- **`addon/ranking.py`**: Morphological boundary classification, decorative/noise asset filtering, POS inference, and the archetype relevance scoring engine (`rank_relevant_images`, `select_best_image`). *(Stdlib only).*
- **`addon/catalog.py`**: Offline SQLite metadata catalog store (`CatalogStore`) and in-memory substring search index (`CatalogIndex`) reading `addon/data/catalog.sqlite`. *(Stdlib only).*
- **`addon/client.py`**: `IrasutoyaClient` handling offline catalog vs. live Blogger feed queries, HTTPS-only and allowlisted-host redirect validation (`_HttpsOnlyRedirectHandler`), HTTP 404/410 image URL self-healing, and sequential candidate fallback search (`search_with_fallbacks`). *(Stdlib only).*
- **`addon/batch.py`**: Background batch processor (`run_batch_fill`, `process_note_record`), decoupled read/download/commit phases via Anki's `QueryOp`, polite network request pacing, and `irasutoya::fallback_reading` audit tagging.
- **`addon/ui.py`**: PyQt6 (`aqt`) modal dialogs following Apple HIG and dark-mode standards (Single Card interactive thumbnail gallery with arrow-key navigation and suggestion chips, and Batch Mode preview table).
- **`addon/config.py`**: User configuration management (`addon/config.json`) via `aqt.mw.addonManager` with schema fallback defaults when running outside Anki.
- **`scripts/build_catalog.py`**, **`scripts/package_addon.py`**, **`scripts/eval_deck_search.py`**: Standalone CLI utilities for building and validating the offline SQLite catalog, packaging the `.ankiaddon` distribution archive, and benchmarking search recall against live decks via AnkiConnect.

## 💾 Offline Catalog Schema (SQLite) & In-Memory Index
The add-on ships with an immutable SQLite database at `addon/data/catalog.sqlite` (`SCHEMA_VERSION = 1`), opened read-only and immutable (`mode=ro&immutable=1`) by `CatalogStore`.
- **`meta` table**: Key-value metadata (`schema_version`, `built_at`, `max_updated`, `post_count`, `image_count`, `source_total_results`, `keyword_sources`).
- **`posts` table**: Stores indexed Blogger posts (`id`, `blogger_id`, `title`, `description`, `published`, `updated`, `image_count`).
- **`images` table**: Stores individual illustrations linked via `post_id` (`id`, `post_id`, `key`, `ord`, `title`, `url` normalized to `/s0/`).
- **`categories` & `post_categories` tables**: Store Blogger category labels per post (`post_categories` uses `WITHOUT ROWID`).
- **`image_keywords` table**: Optional enriched keyword mappings (`image_id`, `keyword`, `source`, `weight`, `WITHOUT ROWID`).
- **In-Memory Substring Index (`CatalogIndex`)**:
  - *Note:* Rather than running SQL `LIKE` or FTS5 queries per search, `CatalogStore.get_index()` lazily loads all rows into memory once under a thread lock (`warm_async()` pre-warms this on a background daemon thread), constructs a casefolded `\x00`-delimited `haystack` string (`title\x00post_title\x00description\x00categories\x00keywords`), and immediately closes the SQLite connection.
  - Substring searches (`clean_q in rec.haystack`) execute in memory in ~2–3 ms and naturally handle 1–2 character Japanese queries without FTS trigram limitations.
- **404/410 Self-Healing**: When downloading an image whose cached catalog URL returns HTTP 404 or 410, `IrasutoyaClient.download_image()` automatically re-fetches the single Blogger post (`post_id`) once via `_self_heal_image_url()` to resolve the updated image URL.

## 🔍 Search, Normalization & Ranking Quirks (Important!)
- **Tiered Confidence Strategy (Avoiding the "Homophone Trap"):**
  - **Tier 1 (Safe Syntactic Transforms):** Unicode NFKC normalization, loanword Hiragana-to-Katakana conversion (`とまと` $\rightarrow$ `トマト`), middle-dot preservation (`メッセージ・カード`), and Arabic-to-Kanji counter conversion (`1杯` $\rightarrow$ `一杯`, `2人` $\rightarrow$ `二人`). Applied automatically with zero homophone risk.
  - **Tier 2 (Phonetic & Semantic Fallbacks):** Separate note reading fields (`ExpressionReading`, `Reading`, `Word Reading`, `Kana`), inline furigana brackets (`林檎[りんご]`), Katakana chōonpu expansion (`メンバ` $\rightarrow$ `メンバー`), and heuristic verb/compound deinflection (`走っている` $\rightarrow$ `走る`).
  - **Audit Tagging in Batch Mode:** When a note matches via a Tier 2 fallback in Batch Mode, `addon/batch.py` automatically tags the note with `irasutoya::fallback_reading`. Conversely, if a note is later updated via a Tier 1 direct match, any stale `irasutoya::fallback_reading` tag must be removed.
- **Catalog vs. Live Feed Parity:** `addon/feed.py` and `addon/catalog.py` must maintain exact parity in HTML parsing, URL normalization (`/s0/` storage vs. target `image_size` upgrading), and metadata extraction (verified by `tests/test_catalog_parity.py`). Always use `html.unescape()` when extracting titles or URLs from Blogger feeds to prevent broken `&amp;` attributes.
- **Security & Domain Allowlisting:** All HTTP requests and redirects in `addon/client.py` enforce `https://` and validate hostnames via `is_allowed_image_host()` (`bp.blogspot.com`, `*.googleusercontent.com`, `irasutoya.com`) using `_HttpsOnlyRedirectHandler`. Never bypass this with raw `urllib.request.urlopen`.

## 🧵 Anki Thread Safety & Collection Operations
- **Strict Main-Thread Mutation Rule:** **NEVER** mutate Anki notes, media files, collection state, or Qt GUI widgets directly from a background worker thread.
- **Decoupled Batch Pipeline (`addon/batch.py`):**
  1. Read notes and generate search candidates inside a short collection-backed `QueryOp`.
  2. Perform network/catalog searches and image downloads outside the collection lock (`mw.taskman.with_progress(..., uses_collection=False)`) with polite pacing between HTTP requests, marshaling progress bar updates to the GUI thread via `mw.taskman.run_on_main(...)`.
  3. Re-fetch fresh notes, write media via `col.media.write_data()`, and commit note updates via `aqt.operations.note.update_notes()` inside `QueryOp` so native Undo (`Cmd+Z` / `Ctrl+Z`) works cleanly.

## 🔌 Anki & AnkiConnect Integration
- **Headless Core Boundary:** Core search, parsing, ranking, and catalog modules (`text_utils`, `yomitan_parser`, `feed`, `ranking`, `catalog`, `client`) must remain 100% free of `aqt` and `PyQt6` imports so they can run headlessly in CLI scripts and multi-version Python CI.
- **Live Deck Benchmark CLI (`scripts/eval_deck_search.py`):** Integrates with a local Anki instance via AnkiConnect (`http://localhost:8765`, read-only) using Python's built-in `urllib.request` to benchmark search recall across real decks (`--backend live|catalog|both`).
- **Hermetic Testing:** Unit tests in `tests/` must **never** make live network requests to Irasutoya/Blogger or connect to a live AnkiConnect server. Always mock network calls (`_make_request`, `_opener.open`, or `MockIrasutoyaClient`) and use stubbed/in-memory fixtures.

## 🧪 Testing Environment
- The project uses `unittest` and `pytest`, with linting via `ruff` (configured in `pyproject.toml`).
- The project relies on a Conda environment named `anki-irasutoya` (Python 3.13, `ruff`, `pytest`, `aqt>=23.10`, defined in `environment.yml`).
- **To run tests & linting:** Run both unit tests (including PyQt6 GUI dialog tests in `tests/test_ui_mode.py`) and `ruff` via the `anki-irasutoya` Conda environment:
  ```bash
  conda run -n anki-irasutoya python -m unittest discover -s tests -p "test_*.py" -v
  # or via pytest:
  conda run -n anki-irasutoya pytest -v
  # Run Ruff linter:
  conda run -n anki-irasutoya ruff check .
  ```
- **UI Tests Execution:** When running inside the `anki-irasutoya` Conda environment, `aqt` is available and all GUI tests in `tests/test_ui_mode.py` must execute (0 skipped tests).
- **Packaging & Catalog Gates:** `scripts/package_addon.py` enforces a strict file allowlist and fails if private state files (`meta.json`, `user_files/`) are present. Any changes to catalog schemas or packaging must pass `tests/test_catalog.py`, `tests/test_catalog_parity.py`, `tests/test_build_catalog.py`, and `tests/test_package_addon.py`.
- **Docstrings, Type Hints & CodeRabbit:** The repository enforces strict code quality and docstring standards via CodeRabbit PR reviews. You **must** provide standard docstrings and explicit type annotations for every new function and class you write, **including** nested helper/mock functions inside your tests.

## 🛠️ General Guidelines
- **Always use the Conda Environment:** Execute all Python, `pytest`/`unittest`, and `ruff` commands strictly via the `anki-irasutoya` Conda environment (e.g., `conda run -n anki-irasutoya pytest -v`). Running them globally may miss `aqt` or `ruff` dependencies.
- **Python 3.9+ Compatibility:** CI executes hermetic unit tests across Python 3.9 through 3.13 (`.github/workflows/test.yml`). Maintain Python 3.9 syntax and typing compatibility across `addon/` (e.g., prefer `Optional`, `Union`, `List`, `Dict`, `Tuple`, `Sequence` from `typing` rather than unguarded Python 3.10+ `X | Y` syntax).
- **Conventional Commits:** It is mandatory to use the Conventional Commits standard (e.g., `feat:`, `fix:`, `docs:`) for all commit messages. This standard must also be applied to branch naming (e.g., `feat/...`, `fix/...`, `docs/...`).

## 🔄 GitHub PR & Review Loop
When committing new features or fixes, you are expected to handle the entire PR lifecycle natively:
1. **Testing is Mandatory:** Existing tests and linting must pass without regression (`conda run -n anki-irasutoya pytest -v` and `conda run -n anki-irasutoya ruff check .`). Furthermore, you **must write new tests** whenever adding new features, fixing bugs, or addressing user feedback. Ensure the full test suite passes before making *any* commit.
2. Commit your changes and push the branch to origin. **Do NOT push multiple commits rapidly**, as this breaks or auto-pauses the CodeRabbit AI review process. Wait for the review to finish before pushing iterative fixes.
3. **Ask for manual user feedback** so the user can test the changes locally in Anki before you create the Pull Request.
4. Once the user approves the local test, create the Pull Request using the GitHub CLI (`gh pr create`). You **MUST** format the PR body using the `.github/pull_request_template.md`. When filling out the pre-merge checklist, you must actually validate each checkbox and not just mark it for the sake of it.
5. Monitor the pre-merge GitHub Actions checks by running `gh pr checks <pr_number> --watch --interval 60`.
6. **Local CodeRabbit Review:** Before wasting your 1-per-hour remote GitHub token, you **MUST** validate your commits locally. You must read and use the `code-review` skill (via `coderabbit review --agent`) to compare your feature branch against the base branch and catch edge-cases. You get 3 local reviews per hour, so use them to guarantee a clean build first.
7. **Trigger & Monitor CodeRabbit Manually:** Only after the local review is completely clean (0 findings) and standard GitHub Actions tests pass, trigger the remote review. After calling `@coderabbitai review`, you **MUST wait for 30 seconds** so the check run registers on GitHub, and then monitor the review using `gh pr checks <pr_number> --watch --interval 60`:
   ```bash
   gh pr comment <pr_number> --body "@coderabbitai review" && sleep 30 && gh pr checks <pr_number> --watch --interval 60
   ```
8. **Inspect the First CodeRabbit Comment & Handle Rate Limits:** A green `CodeRabbit` status check in `gh pr checks` **ONLY** means the analysis finished running; it does **NOT** mean the code was approved. **The very first CodeRabbit comment on the PR is constantly updated** in place and serves as the source of truth for:
   - Whether there is any **pre-merge check issue** or actionable review finding.
   - Whether CodeRabbit is currently **rate-limited**, along with the exact **cooldown time** remaining (if any).
   - **If CodeRabbit is rate-limited:** You **MUST wait until the cooldown limit is completely cleared** before trying again. Do not spam review triggers or attempt to bypass the review while rate-limited.
   - Fetch and inspect the first CodeRabbit comment directly using:
     ```bash
     gh pr view <pr_number> --json comments --jq '[.comments[] | select(.author.login | test("coderabbit"; "i"))] | first'
     ```
   - Also confirm the PR merge status by running:
     ```bash
     gh pr view <pr_number> --json mergeable,mergeStateStatus
     ```
9. **Address Review Comments (NEVER Resolve Manually):**
   - **Do NOT manually resolve review threads or conversations** (via GitHub CLI, GraphQL API, or UI) in an attempt to make the PR mergeable.
   - Instead, commit and push fixes for all actionable review comments, then trigger a new review (respecting local review checks, rate limits, and the 30-second wait). **CodeRabbit will automatically mark its own review threads as resolved on the next review pass** once it verifies the fixes.
10. **Merge:** Once CodeRabbit resolves all threads, pre-merge checks pass, and the PR is cleanly mergeable, merge it via:
    ```bash
    gh pr merge <pr_number> --squash --delete-branch
    ```
