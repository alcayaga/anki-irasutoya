---
name: Enhancement / Architectural Improvement
about: Propose a data-driven enhancement, search/ranking improvement, or deferred review item
title: "feat: "
labels: ["enhancement"]
assignees: ""
---

- **Parent / Dependencies:** #<!-- issue_number (optional, e.g., Parent: #49, depends on #66, blocks #48) -->
- **Deferred from PR:** #<!-- pr_number (if spun out from a CodeRabbit review or PR triage, otherwise remove) -->
- **Target Layer:** <!-- [ ] Runtime Core (addon/ headless) | [ ] GUI (addon/ui/ PyQt6) | [ ] Offline Catalog / Synonyms (scripts/ & catalog.sqlite) -->

---

### Problem Statement & Empirical Root Causes

<!-- Summarize how the issue was discovered (e.g., flag:4 analysis in Mining, eval_deck_search.py cohort, catalog inspection, or deferred CodeRabbit finding). -->

**Diagnostic / Baseline Command:**
```bash
# Exact command or query used to observe the issue (remove if N/A):
conda run --no-capture-output -n anki-irasutoya python scripts/eval_deck_search.py --deck "Mining" --query "flag:4" --backend catalog -v
```

1. **<!-- Root Cause 1 Title (e.g., Curated Direct Concepts Placed Behind Raw `primary` in `generate_search_candidates()`) -->**:
   - **Mechanism:** <!-- Point to exact module/function, candidate priority index, or score weights -->
   - **Empirical Impact:**
     - `<!-- headword 1 -->` → `<!-- current wrong/suboptimal match -->` instead of `<!-- expected target/tier -->`.
     - `<!-- headword 2 -->` → `<!-- current wrong/suboptimal match -->` instead of `<!-- expected target/tier -->`.

2. **<!-- Root Cause 2 Title (optional) -->**:
   - **Mechanism:** <!-- Code reference & scoring/fallback logic -->
   - **Empirical Impact:** <!-- Concrete examples -->

---

### Implementation Scope

1. **<!-- Subsystem 1 (`addon/module.py`) -->**:
   - <!-- Specific function, priority ordering, or data model changes -->
2. **<!-- Subsystem 2 (`addon/other_module.py` / `scripts/...`) -->**:
   - <!-- Specific scoring weights, SQLite queries, or UI state changes -->

### Non-Goals & Architectural Guardrails
- **Out of Scope:** <!-- e.g., Do not rebuild catalog.sqlite; runtime-only changes in addon/ -->
- **Invariants:** Maintain headless stdlib-only boundary in core modules (`text_utils`, `yomitan_parser`, `feed`, `ranking`, `catalog
