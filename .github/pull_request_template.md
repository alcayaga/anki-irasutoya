## Summary
<!-- Brief 1-2 sentence overview of changes -->

## Details
<!-- Detailed bullet points of the architectural changes, algorithms, and modules -->

## Related Issues
<!-- e.g. Closes #7 -->

## Deck Search Recall Benchmark (`scripts/eval_deck_search.py`)
<!-- Mandatory: Run before making changes (Baseline) and before pushing each commit (This PR). Update this section if subsequent commits change the numbers. -->
Evaluated against the `Mining` deck using:
```bash
conda run --no-capture-output -n anki-irasutoya python scripts/eval_deck_search.py --deck "Mining" --backend catalog --all
```

### 1. Recall Comparison (Baseline vs. This PR)
| Metric | Baseline (`main`) | This PR | Delta |
| :--- | :---: | :---: | :---: |
| **Total Cards** | <!-- e.g. 3,536 --> | <!-- e.g. 3,536 --> | 0 |
| **Matched Cards** | <!-- e.g. 2,179 (61.62%) --> | <!-- e.g. 2,191 (61.96%) --> | <!-- e.g. +12 (+0.34%) --> |
| **Unmatched Cards** | <!-- e.g. 1,357 (38.38%) --> | <!-- e.g. 1,345 (38.04%) --> | <!-- e.g. -12 (-0.34%) --> |
| **Regressions** | — | **0** | **0** |

### 2. Incremental Strategy Progression
| Stage | Baseline Hits (%) | This PR Hits / Total | This PR Recall % | Incremental Gain |
| :--- | :---: | :---: | :---: | :---: |
| **1. Legacy (Naive Direct Query)** | <!-- e.g. 1,518 (42.93%) --> | <!-- e.g. 1,518 / 3,536 --> | <!-- e.g. 42.93% --> | — |
| **2. Tier 1 (Syntactic & Script Norm)** | <!-- e.g. 1,907 (53.93%) --> | <!-- e.g. 1,907 / 3,536 --> | <!-- e.g. 53.93% --> | <!-- e.g. +11.00% (+389 cards) --> |
| **3. Tier 2 (Readings/Deinf/Concepts)** | <!-- e.g. 2,098 (59.33%) --> | <!-- e.g. 2,110 / 3,536 --> | <!-- e.g. 59.67% --> | <!-- e.g. +5.74% (+203 cards) --> |
| **4. Bilingual Gloss (Definition Bridge)** | <!-- e.g. 2,179 (61.62%) --> | <!-- e.g. 2,191 / 3,536 --> | <!-- e.g. 61.96% --> | <!-- e.g. +2.29% (+81 cards) --> |
| **Net Improvement over Legacy** | <!-- e.g. 2,179 (61.62%) --> | <!-- e.g. 2,191 / 3,536 --> | <!-- e.g. 61.96% --> | <!-- e.g. +19.03% (+673 cards) --> |

### 3. Verified New Matches / Notable Changes (If Applicable)
<!-- List newly matched cards (`headword` -> `illustration title`) or write "No recall changes expected (UI/refactoring only)." -->
- 

### 4. Additional Empirical / Feature Benchmarks (Optional)
<!-- Include task-specific tables here when relevant (e.g., 30-Term Golden LLM Benchmark, Concept Target Parity & Idempotency, or Latency profiling). Remove or mark N/A if not applicable. -->

## How to Validate
1. **Automated Unit Tests**:
   ```bash
   # Option A: In conda anki-irasutoya environment
   conda run -n anki-irasutoya python -m unittest discover -s tests -p "test_*.py" -v
   # or via pytest:
   conda run -n anki-irasutoya pytest -v

   # Option B: With Anki's bundled packages (macOS default shown)
   PYTHONPATH="${ANKI_PACKAGES:-/Applications/Anki.app/Contents/Resources/app_packages}:." python -m unittest discover -s tests -p "test_*.py" -v
   ```
2. **Code Quality & Linting**:
   ```bash
   conda run -n anki-irasutoya ruff check .
   coderabbit review --agent --base main
   ```
3. **Deck Search Recall Benchmark**:
   ```bash
   conda run --no-capture-output -n anki-irasutoya python scripts/eval_deck_search.py --deck "Mining" --backend catalog --all
   ```
4. **Manual Validation in Anki**:
   <!-- Clear, actionable reproduction steps to verify in Anki Browser and Visual Picker -->

## Pre-Merge Checklist
- [ ] All unit tests pass in `anki-irasutoya` Conda environment with 0 skipped UI tests
- [ ] Code passes `ruff check .` with 0 errors
- [ ] `scripts/eval_deck_search.py` baseline compared against PR branch with 0 regressions (and PR tables updated to match latest commit)
- [ ] CodeRabbit review completed, all claims verified against real data, and findings summarized/triaged with the user
- [ ] Working tree clean and git commits formatted according to Conventional Commits standard
- [ ] Tested in macOS dark mode and light mode
- [ ] Thread safety confirmed: note and media mutations confined to main GUI thread
