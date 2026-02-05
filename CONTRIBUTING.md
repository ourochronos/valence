# Contributing to Valence

## Workflow

### 1. Issue First
Every code change must be tied to an issue.

```bash
# Create issue first
gh issue create --title "Brief description" --body "Details..."

# Then branch from that issue
git checkout -b fix/123-brief-description  # for bugs
git checkout -b feat/123-brief-description # for features
git checkout -b docs/123-brief-description # for docs
```

**No direct commits to main.** All changes go through PRs.

### 2. Branch Naming
```
<type>/<issue-number>-<brief-description>

Types:
- fix/     Bug fixes
- feat/    New features
- docs/    Documentation
- refactor/ Code restructuring
- test/    Test additions/fixes
- chore/   Maintenance tasks
```

### 3. Commit Messages
Follow conventional commits:
```
<type>: <description>

[optional body]

Closes #123
```

Examples:
```
fix: Handle None in trust computation

The compute_delegated_trust function now respects the
respect_delegation flag when chaining trust edges.

Closes #126
```

### 4. Pull Request Process

1. **Create PR** with descriptive title and body
2. **Link issues** using "Closes #N" in description
3. **Automated checks** run (pre-commit, tests)
4. **Review** required before merge
5. **Squash merge** to main (clean history)

### 5. Review Checklist

Reviewers should check:
- [ ] Tests pass
- [ ] New code has tests
- [ ] No security issues (bandit clean)
- [ ] Types correct (mypy clean on changed files)
- [ ] Documentation updated if needed
- [ ] Commit message follows convention

### 6. Pre-commit Hooks

Installed automatically. Runs on commit:
- Trailing whitespace fix
- End of file fix
- YAML validation
- Ruff (linting + formatting)
- Mypy (type checking)
- Bandit (security)

To skip (emergency only):
```bash
git commit --no-verify -m "message"
```

---

## Code Style

- **Python 3.11+**
- **Ruff** for linting/formatting
- **Type hints** required for public APIs
- **Docstrings** for all public functions/classes

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/valence

# Run specific test
pytest tests/privacy/test_trust.py -v
```

## Security

- Run `bandit -r src/valence` before PRs
- Run `pip-audit --skip-editable` for dependency checks
- No secrets in code
- Parameterized SQL only

---

*Last updated: 2026-02-04*
