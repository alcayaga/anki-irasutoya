## Summary
<!-- Brief 1-2 sentence overview of changes -->

## Details
<!-- Detailed bullet points of the architectural changes, algorithms, and modules -->

## Related Issues
<!-- e.g. Closes #7 -->

## How to Validate
1. **Automated Unit Tests**:
   ```bash
   # In conda anki-irasutoya environment (or with ANKI_PACKAGES):
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
2. **Code Quality & Linting**:
   ```bash
   ruff check .
   coderabbit review --agent --base main
   ```
3. **Manual Validation in Anki**:
   <!-- Clear, actionable reproduction steps to verify in Anki Browser and Visual Picker -->

## Pre-Merge Checklist
- [ ] All unit tests pass in Anki environment (92/92 tests)
- [ ] Code passes `ruff check .` with 0 errors
- [ ] CodeRabbit review completed with 0 findings
- [ ] Working tree clean and git commits formatted according to Conventional Commits standard
- [ ] Tested in macOS dark mode and light mode
- [ ] Thread safety confirmed: note and media mutations confined to main GUI thread
- [ ] Documentation and README.md updated
