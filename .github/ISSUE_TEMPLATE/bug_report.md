---
name: Bug Report / Precision & False-Positive Fix
about: Report false-positive matches, parser/ranking defects, regressions, or UI/thread-safety bugs
title: "fix: "
labels: ["bug"]
assignees: ""
---

- **Parent / Incorporates:** #<!-- issue_number (optional, e.g., Parent: #49, incorporates #44) -->
- **Follow-up / Regression from:** #<!-- pr_or_issue_number (e.g., PR #45 / Issue #30, if applicable) -->
- **Deferred from PR:** #<!-- pr_number (if spun out from a CodeRabbit review, otherwise remove) -->
- **Bug Category:** <!-- [ ] False Positive / Precision | [ ] Parser / Normalization | [ ] Ranking / Candidate Priority | [ ] GUI / Thread Safety | [ ] Catalog / CLI Script -->

---

### Problem Statement

<!-- Summarize how the bug or false-positive pattern was uncovered (e.g., full benchmark audit across 3,452 unique notes in Mining, bilingual_gloss audit table, flag:4 inspection, or UI reproduction). -->

**Diagnostic / Reproduction Command:**
```bash
# Exact command used to surface the false matches or bug (adjust flags or remove for UI bugs):
conda run --no-capture-output -n anki-irasutoya python scripts/eval_deck_search.py --deck "Mining" --backend catalog --all -v
```

1. **<!-- Structural Issue 1 Title (`addon/module.py`) -->**:
   - **Root Cause:** <!-- Explain the exact parser, synonym, or ranking flaw -->
   - **False Matches / Broken Cases:**
     - `<!-- headword 1 -->` (<!-- context/sense/gloss -->) → `<!-- wrong target -->`
     - `<!-- headword 2 -->` (<!-- context/sense/gloss -->) → `<!-- wrong target -->`

2. **<!-- Structural Issue 2 Title (`addon/other_module.py`) -->**:
   - **Root Cause:** <!-- Explain the mechanism -->
   - **False Matches / Broken Cases:**
     - `<!-- headword 3 -->` (<!-- context/sense/gloss -->) → `<!-- wrong target -->`

---

### Proposed Solution

1. **<!-- Fix 1 Title (`addon/module.py`) -->**:
   - <!-- Specific guard, cap, regex/AST rule, or POS filter to implement -->
   - <!-- Edge cases or valid patterns that MUST remain untouched -->
2. **<!-- Fix 2 Title (`addon/other_module.py`) -->**:
   - <!-- Specific algorithmic or state handling changes -->

### Architectural Guardrails
- Maintain headless stdlib-only boundary in core modules (`text_utils`, `yomitan_parser`, `feed`, `ranking`, `catalog`, `client`), Python 3.9+ typing compatibility (`Optional`, `List`, `Dict`, `Set`), and main-thread-only Anki mutations.

---

### Acceptance Criteria

- [ ] **False Positives Eliminated / Bug Resolved:** <!-- List the specific broken headwords/inputs (e.g., `いけない`, `誤魔化す`, `仕方ない`) and confirm they no longer match the wrong targets. -->
- [ ] **Valid Matches Preserved (Zero Collateral Damage):** <!-- List valid positive cases in the same code path (e.g., `"after school"` -> `学校`, `"stomach medicine"` -> `薬`) that must continue to match. -->
- [ ] **Hermetic Regression Tests:** Both the blocked false-positive cases and the preserved valid cases are codified as unit tests in `tests/` (with full docstrings and type annotations).
- [ ] **Deck Benchmark Audit (`scripts/eval_deck_search.py`):** Run before and after; confirm zero unintended regressions on true matches (and document the intentional removal of false-positive hits in the PR table).
- [ ] **CI & Quality Gate:** `conda run -n anki-irasutoya pytest -v` (0 skipped UI tests) and `conda run -n anki-irasutoya ruff check .` pass cleanly.
