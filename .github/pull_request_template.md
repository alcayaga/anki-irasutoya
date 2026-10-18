## Summary
<!-- Brief 1-2 sentence overview of changes -->

## Details
<!-- Detailed bullet points of the architectural changes, algorithms, and modules -->

## Related Issues
<!-- e.g. Closes #7 -->

## How to Validate
1. **Automated Unit Tests**:
   ```bash
   # Set ANKI_PACKAGES to your Anki app_packages directory (macOS default shown)
   PYTHONPATH="${ANKI_PACKAGES:-/Applications/Anki.app/Contents/Resources/app_packages}:." python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
2. **Code Quality**:
   ```bash
   coderabbit review --agent --base main
   ```
3. **Manual Validation in Anki**:
   <!-- Clear, actionable reproduction steps to verify in Anki Browser and Visual Picker -->

## Pre-Merge Checklist
- [ ] All unit tests pass in Anki environment
- [ ] CodeRabbit review completed with 0 findings
- [ ] Working tree clean and git commits formatted according to Conventional Commits standard
- [ ] Tested in macOS dark mode and light mode
- [ ] Thread safety confirmed: note and media mutations confined to main GUI thread
- [ ] Documentation and README.md updated
