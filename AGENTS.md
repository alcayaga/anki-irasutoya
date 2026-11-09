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
- **`addon/ui/`**: PyQt6 (`aqt`) modal dialog package following Apple HIG and dark-mode standards (`dialog.py` for shell and field mapping, `single_view.py` for interactive thumbnail gallery picker with arrow-key navigation and suggestion chips, `batch_view.py` for multi-card preview table, `widgets.py` for custom thumbnail cards and chips, `thumbnails.py` for paced caching loader, `helpers.py` for headless formatting and probing, and `styles.py` for centralized theme tokens and QSS).
- **`addon/config.py`**: User configuration management (`addon/config.json`) via `aqt.mw.addonManager` with schema fallback defaults when running outside Anki.
- **`scripts/build_catalog.py`**, **`scripts/build_synonyms.py`**, **`scripts/agent_synonyms.py`**, **`scripts/pull_models.py`**, **`scripts/synonyms/`**, **`scripts/package_addon.py`**, **`scripts/eval_deck_search.py`**: Standalone CLI utilities and modular synonym compilation engines for building, validating, and enriching the offline SQLite catalog with Kaishi, Sudachi, and LLM visual concept maps, pulling declarative Ollama models (`models.yml`), packaging the `.ankiaddon` distribution archive, and benchmarking search recall against live decks via AnkiConnect.
- **`models.yml`**: Declarative manifest specifying local Ollama models (`gemma4:26b` for visual concept reasoning and `qwen3-embedding:4b` for vector similarity pre-search), parameters, and VRAM/RAM constraints.
- **`scripts/synonyms/ollama_embed.py`**: Headless vector embedding client, binary-cached catalog vector index (`CatalogVectorIndex`), and hybrid lexical + semantic pre-search engine with zero mandatory third-party dependencies (`array.array('f')` stdlib fallback, accelerated by `numpy` when available).

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
- **Tiered Confidence Strategy & 4-Group Candidate Hierarchy (Avoiding the "Homophone Trap"):**
  - **Group 1 (Tier 1 — Surface Curated Concepts):** Curated concept maps (`concept_map`), Kaishi concept bridges (`kaishi_concept`), and direct verified LLM concepts (`llm_concept`) evaluated directly on the surface headword *before* `primary`. Zero homophone risk, retained even when `allow_tier2=False`.
  - **Group 2 (Tier 1 — Direct Surface & Safe Orthographic Variants):** Primary query with standalone title guard, Arabic-to-Kanji counter conversion (`1杯` $\rightarrow$ `一杯`), Yomitan canonical ★ variants, Yomitan `uk` orthography Kana readings, Yomitan forms with Kanji, Bikago polite prefixes (`お肉`), verb stative/past inflections, and loanword Katakana conversion (`とまと` $\rightarrow$ `トマト`). Compound-only primary title matches are held as deferred fallbacks rather than winning immediately.
  - **Group 3 (Tier 2 — Semantic Exemplars & Morphological Stems):** LLM visual exemplars / prototypical scenes (`llm_concept_tier2`, e.g. representing abstract concepts, states, or pronouns), primary heuristic deinflections (`走っている` $\rightarrow$ `走る`), and honorific / compound affix stripping (`ご挨拶` $\rightarrow$ `挨拶`).
  - **Group 4 (Tier 2 — Late Fallbacks & Phonetic Readings):** Deferred primary compound title matches, verb masu-stems, Yomitan sense-specific Kanji (`esp.`), SudachiDict synonym bridges, bilingual English definition gloss bridges, Yomitan cross-references (`see:`), dedicated note reading/furigana fields, same-field furigana brackets, Katakana chōonpu expansion (`メンバ` $\rightarrow$ `メンバー`), and speculative alternative deinflections.
- **4-Tag UX Workflow & Batch Safety Guards:**
  - **`irasutoya::non_visual`**: Tagged on abstract, non-visual concepts blocked by LLM curation (`llm_rejected`). Batch Mode skips auto-fill; Single Card Mode displays an informational banner while keeping manual search active.
  - **`irasutoya::approximate`**: Tagged on notes auto-filled via Tier 2 semantic exemplars, deinflections, or compound titles.
  - **`irasutoya::review`**: Tagged on notes with description-only or Tier 2 phonetic reading matches (high homophone risk, e.g. `顧問` $\rightarrow$ `こもん` $\rightarrow$ `コモンマーモセット`). Auto-fill is held back for manual confirmation in Single Card Mode.
  - **`irasutoya::no_match`**: Tagged on notes with zero search matches across all candidate groups.
  - Direct Tier 1 auto-fill matches or manual selection in Single Card Mode automatically strip all 4 taxonomy tags.
- **Catalog vs. Live Feed Parity:** `addon/feed.py` and `addon/catalog.py` must maintain exact parity in HTML parsing, URL normalization (`/s0/` storage vs. target `image_size` upgrading), and metadata extraction (verified by `tests/test_catalog_parity.py`). Always use `html.unescape()` when extracting titles or URLs from Blogger feeds to prevent broken `&amp;` attributes.
- **Security & Domain Allowlisting:** All HTTP requests and redirects in `addon/client.py` enforce `https://` and validate hostnames via `is_allowed_image_host()` (`bp.blogspot.com`, `*.googleusercontent.com`, `irasutoya.com`) using `_HttpsOnlyRedirectHandler`. Never bypass this with raw `urllib.request.urlopen`.

## 🧵 Anki Thread Safety & Collection Operations
- **Strict Main-Thread Mutation Rule:** **NEVER** mutate Anki notes, media files, collection state, or Qt GUI widgets directly from a background worker thread.
- **Decoupled Batch Pipeline (`addon/batch.py`):**
  1. Read notes and generate search candidates inside a short collection-backed `QueryOp`.
  2. Perform network/catalog searches and image downloads outside the collection lock (`mw.taskman.with_progress(..., uses_collection=False)`) with polite pacing between HTTP requests, marshaling progress bar updates to the GUI thread via `mw.taskman.run_on_main(...)`.
  3. Re-fetch fresh notes, write media via `col.media.write_data()`, and commit note updates via `aqt.operations.note.update_notes()` inside `QueryOp` so native Undo (`Cmd+Z` / `Ctrl+Z`) works cleanly.

## 🔌 Anki, AnkiConnect & Deck Benchmarking (`scripts/eval_deck_search.py`)
- **Headless Core Boundary:** Core search, parsing, ranking, and catalog modules (`text_utils`, `yomitan_parser`, `feed`, `ranking`, `catalog`, `client`) must remain 100% free of `aqt` and `PyQt6` imports so they can run headlessly in CLI scripts and multi-version Python CI.
- **Hermetic Testing:** Unit tests in `tests/` must **never** make live network requests to Irasutoya/Blogger or connect to a live AnkiConnect server. Always mock network calls (`_make_request`, `_opener.open`, or `MockIrasutoyaClient`) and use stubbed/in-memory fixtures.
- **Live Deck Benchmark CLI (`scripts/eval_deck_search.py`):** Integrates with a local Anki instance via AnkiConnect (`http://localhost:8765`, read-only) using Python's built-in `urllib.request` to benchmark search recall and stage-by-stage progression across real decks.
  ```bash
  conda run --no-capture-output -n anki-irasutoya python scripts/eval_deck_search.py [options]
  ```
  - **Key CLI Options:**
    - `--deck DECK`: Anki deck name to evaluate (default: `'Mining'`).
    - `--query QUERY`: Extra Anki search query (e.g., `'flag:4'`), combined with `--deck` unless it contains an explicit positive `'deck:'` selector.
    - `--sort SORT`: Field to sort cards by (default: `'FreqSort'`).
    - `--source-field SOURCE_FIELD`: Source field to query (default: auto-detect `Expression`/`Word`).
    - `--limit LIMIT` / `--all`: Number of notes to evaluate (default: `50`; `--all` or `--limit 0` evaluates all notes in the deck).
    - `--backend {live,catalog,both}`: Search backend to benchmark (default: `both`; use `catalog` for fast offline evaluation).
    - `--delay DELAY`: Polite delay between HTTP requests in seconds (default: `0.0` for `catalog`, `0.25` for `live`).
    - `--disable-tier2` / `--disable-bilingual`: Disable Tier 2 reading or bilingual definition gloss fallbacks.
    - `--no-stages` / `--no-bilingual-hits`: Omit stage progression tracking or bilingual gloss hit audit table.
    - `--verbose`, `-v`: Print detailed candidate resolution for each note.
    - `--anki-url ANKI_URL`: AnkiConnect URL (default: `http://localhost:8765`).
- **Mandatory Benchmarking Workflow:**
  1. **Baseline Before Changes:** Before modifying any code, run `scripts/eval_deck_search.py` to record baseline search recall and stage progression numbers.
  2. **Pre-Push Regression Check:** Before pushing *any* new commit, re-run `scripts/eval_deck_search.py` with the exact same arguments used for the baseline. Verify there are **no unintended regressions** on true matches, and explicitly record any recall improvements (for `.github/ISSUE_TEMPLATE/enhancement.md` tasks) or intentional false-positive removals (for `.github/ISSUE_TEMPLATE/bug_report.md` precision fixes).
  3. **PR Body Synchronization:** Whenever `scripts/eval_deck_search.py` numbers change across commits, update the benchmark section in the Pull Request description (`gh pr edit`) so the PR body always reflects the latest verified numbers.

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
- **Always use the Conda Environment:** Execute all Python, `pytest`/`unittest`, and `ruff` commands strictly via the `anki-irasutoya` Conda environment (e.g., `conda run -n anki-irasutoya pytest -v`). Running them globally may miss `aqt` or `ruff` dependencies. Pass `--no-capture-output` (e.g., `conda run --no-capture-output -n anki-irasutoya ...`) when running CLI scripts interactively to avoid Conda subprocess output buffering.
- **Python 3.9+ Compatibility:** CI executes hermetic unit tests across Python 3.9 through 3.13 (`.github/workflows/test.yml`). Maintain Python 3.9 syntax and typing compatibility across `addon/` (e.g., prefer `Optional`, `Union`, `List`, `Dict`, `Tuple`, `Sequence` from `typing` rather than unguarded Python 3.10+ `X | Y` syntax).
- **Conventional Commits:** It is mandatory to use the Conventional Commits standard (e.g., `feat:`, `fix:`, `docs:`) for all commit messages. This standard must also be applied to branch naming (e.g., `feat/...`, `fix/...`, `docs/...`).
- **GitHub Issue Templates & Empirical Test Rule:**
  - Whenever creating or editing a GitHub Issue (`gh issue create` / `gh issue edit`), you **MUST** format the body using `.github/ISSUE_TEMPLATE/enhancement.md` (for `feat:` enhancements/architecture) or `.github/ISSUE_TEMPLATE/bug_report.md` (for `fix:` bugs, precision guards, and false-positive removals).
  - When implementing an existing GitHub Issue, you **MUST** codify every concrete headword/input example listed in the issue's *Problem Statement* (both blocked false positives and preserved valid matches) as hermetic unit tests in `tests/`.

## 🐰 CodeRabbit Verification & Triage Protocol (Local & GitHub)
CodeRabbit lacks access to the full offline SQLite catalog (`addon/data/catalog.sqlite`), live Anki user decks, real Yomitan HTML field structures, and Blogger feed quirks. Therefore, **never blindly implement CodeRabbit suggestions.** For **both** local CLI reviews (`coderabbit review --agent`) and remote GitHub PR reviews, follow this mandatory protocol before touching any code:

1. **Summarize Findings & Explain Criticality:**
   - Present a clear summary of every CodeRabbit finding to the user.
   - Explicitly state **why** each finding is critical to the current PR (e.g., correctness bug, thread-safety violation, security issue, test gap).
2. **Verify All Claims Against Real Data First:**
   - Before making changes, empirically verify CodeRabbit's assumptions against the actual codebase and domain data (e.g., query `addon/data/catalog.sqlite`, inspect real Anki note payloads via AnkiConnect, or check Yomitan HTML/JMdict fixtures).
   - If a CodeRabbit claim cannot be verified, is contradicted by real Irasutoya/Anki/Yomitan data, or involves a questionable trade-off, **do not implement it**—notify the user immediately with your findings and ask how to proceed.
3. **Offer to Defer Non-Critical or Unrelated Items:**
   - If a valid finding is non-critical, purely stylistic, or unrelated to the scope of the current PR, do not bloat the PR. Present the finding to the user and offer the option to **defer it to a new GitHub Issue** (`gh issue create`) formatted strictly using `.github/ISSUE_TEMPLATE/enhancement.md` (for refactors/enhancements) or `.github/ISSUE_TEMPLATE/bug_report.md` (for edge-case bugs/precision fixes), including `Deferred from PR: #<pr_number>` in the header.

## 🔄 GitHub PR & Review Loop
When committing new features or fixes, you are expected to handle the entire PR lifecycle natively:

1. **Capture Search Recall Baseline (Before Making Changes):**
   - Before modifying code, run `scripts/eval_deck_search.py` via the `anki-irasutoya` Conda environment and record the baseline metrics:
     ```bash
     conda run --no-capture-output -n anki-irasutoya python scripts/eval_deck_search.py --deck "Mining" --backend catalog --all
     ```
2. **Testing & Pre-Commit Benchmark Gate (Mandatory):**
   - Existing unit tests and linting must pass without regression (`conda run -n anki-irasutoya pytest -v` and `conda run -n anki-irasutoya ruff check .`).
   - You **must write new tests** whenever adding new features, fixing bugs, or addressing user feedback (including all empirical cases listed in the target GitHub Issue). Ensure the full test suite passes before making *any* commit.
   - Re-run `scripts/eval_deck_search.py` and compare the output against your Step 1 baseline to confirm **zero unintended regressions** (and capture any recall improvements or intentional false-positive removals).
3. **Local CodeRabbit Review, Verification & Triage:**
   - Before wasting your 1-per-hour remote GitHub token, validate your changes locally using the `code-review` skill (`coderabbit review --agent`) to compare your feature branch against the base branch (you get 3 local reviews per hour).
   - Apply the **CodeRabbit Verification & Triage Protocol** above: verify all data assumptions (Irasutoya, Anki, Yomitan), summarize findings for the user with their criticality to this PR, notify the user if any claim is unverified/flawed, and offer to defer non-critical/unrelated findings to a new GitHub Issue using `.github/ISSUE_TEMPLATE/enhancement.md` or `.github/ISSUE_TEMPLATE/bug_report.md`.
4. **Ask for Manual User Feedback:**
   - Ask the user to test the changes locally in Anki before pushing and creating the Pull Request.
5. **Commit, Push & Create Pull Request:**
   - Before pushing *any* commit, verify `scripts/eval_deck_search.py` has been run on the exact commit state.
   - Push the branch to `origin`. **Do NOT push multiple commits rapidly**, as this breaks or auto-pauses the CodeRabbit AI review process. Wait for any active review to finish before pushing iterative fixes.
   - Once the user approves the local test, create the Pull Request using `gh pr create`. You **MUST** format the PR body using `.github/pull_request_template.md`, actually validate each checkbox in the pre-merge checklist, and populate the `scripts/eval_deck_search.py` baseline vs. post-change tables.
   - **Keep PR Benchmark Results Synced:** If subsequent commits change the output of `scripts/eval_deck_search.py`, you **MUST** update the benchmark section of the PR body via `gh pr edit <pr_number>`.
6. **Monitor Pre-Merge GitHub Actions Checks:**
   - Monitor CI checks by running `gh pr checks <pr_number> --watch --interval 60`.
7. **Trigger & Monitor Remote CodeRabbit Review:**
   - Only after the local review is clean (all critical items resolved and non-critical items triaged with the user) and standard GitHub Actions tests pass, trigger the remote review.
   - After calling `@coderabbitai review`, you **MUST wait for 30 seconds** so the check run registers on GitHub, and then monitor the review using `gh pr checks <pr_number> --watch --interval 60`:
     ```bash
     gh pr comment <pr_number> --body "@coderabbitai review" && sleep 30 && gh pr checks <pr_number> --watch --interval 60
     ```
8. **Inspect the First CodeRabbit Comment, Verify Claims & Handle Rate Limits:**
   - A green `CodeRabbit` status check in `gh pr checks` **ONLY** means the analysis finished running; it does **NOT** mean the code was approved. **The very first CodeRabbit comment on the PR is constantly updated** in place and serves as the source of truth for:
     - Whether there is any **pre-merge check issue** or actionable review finding.
     - Whether CodeRabbit is currently **rate-limited**, along with the exact **cooldown time** remaining (if any).
   - **If CodeRabbit is rate-limited:** You **MUST wait until the cooldown limit is completely cleared** before trying again. Do not spam review triggers or attempt to bypass the review while rate-limited.
   - Fetch and inspect the first CodeRabbit comment and review threads directly using:
     ```bash
     gh pr view <pr_number> --json comments --jq '[.comments[] | select(.author.login | test("coderabbit"; "i"))] | first'
     ```
   - Confirm the PR merge status by running:
     ```bash
     gh pr view <pr_number> --json mergeable,mergeStateStatus
     ```
   - Apply the **CodeRabbit Verification & Triage Protocol** to all GitHub review findings: summarize the review for the user, explain why each item is critical to this PR, empirically verify claims about Irasutoya/Anki/Yomitan data before making edits, notify the user if a claim is dubious, and offer to defer non-critical or unrelated items to a new GitHub Issue (`.github/ISSUE_TEMPLATE/enhancement.md` or `.github/ISSUE_TEMPLATE/bug_report.md`).
9. **Address Approved Review Comments (NEVER Resolve Manually):**
   - **Do NOT manually resolve review threads or conversations** (via GitHub CLI, GraphQL API, or UI) in an attempt to make the PR mergeable.
   - For verified critical fixes (and any optional items the user chooses to include), implement the changes, run `pytest`, `ruff`, and `scripts/eval_deck_search.py` (updating the PR body if benchmark numbers changed), commit, push once, and trigger a new review (respecting local review checks, rate limits, and the 30-second wait). **CodeRabbit will automatically mark its own review threads as resolved on the next review pass** once it verifies the fixes.
10. **Merge:**
    - Once CodeRabbit resolves all threads, pre-merge checks pass, and the PR is cleanly mergeable, merge it via:
      ```bash
      gh pr merge <pr_number> --squash --delete-branch
      ```
