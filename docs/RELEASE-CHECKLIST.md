# Release Checklist

This document defines the required steps before any Valence release.

## Pre-Release Audit Requirements

All releases MUST complete the following audits:

### 1. Security Audit
- [ ] Run `bandit -r src/` for Python security issues
- [ ] Check for outdated dependencies with known CVEs
- [ ] Review OAuth/auth flows for vulnerabilities
- [ ] Verify input sanitization (XSS, injection)
- [ ] Check subprocess usage for injection risks

### 2. Privacy Audit
- [ ] Verify differential privacy parameters
- [ ] Check consent chain enforcement
- [ ] Review data minimization practices
- [ ] Validate encryption at rest and in transit
- [ ] Audit logging for PII exposure

### 3. Federation Audit
- [ ] Verify signature validation on all federation endpoints
- [ ] Check replay attack mitigations (nonces, timestamps)
- [ ] Review trust propagation bounds
- [ ] Validate cross-federation consent chains

### 4. Code Quality Audit
- [ ] All tests passing: `pytest`
- [ ] Zero mypy errors: `mypy src/`
- [ ] No critical linting issues: `ruff check src/`
- [ ] Integration tests passing: `pytest tests/integration/ --live-nodes`

## Issue Resolution Gate

**All issues of severity MEDIUM or higher MUST be resolved before release.**

Severity levels:
- **CRITICAL**: Security vulnerabilities, data loss risks → Block release
- **HIGH**: Significant bugs, privacy concerns → Block release
- **MEDIUM**: Quality issues, minor bugs → Block release
- **LOW**: Nice-to-have improvements → May release with open issues

## Deployment Validation

After deploying to production nodes:

### Health Checks
- [ ] All nodes return healthy status
- [ ] Database connections verified
- [ ] Version matches release tag

### Feature Validation
Run the live node integration tests with expanded coverage:

```bash
pytest tests/integration/test_live_nodes.py -v --live-nodes
```

Required test categories:
- [ ] Health endpoints (both nodes)
- [ ] API info and version
- [ ] OAuth registration and token flow
- [ ] Belief CRUD (create, read, update, delete)
- [ ] Semantic search with confidence scoring
- [ ] Federation discovery between nodes
- [ ] Federation trust establishment
- [ ] Federation belief sync
- [ ] MCP protocol endpoints
- [ ] Privacy controls (share levels)
- [ ] Consent chain creation and validation

### Performance Baseline
- [ ] Health endpoint < 100ms
- [ ] Belief query < 500ms
- [ ] Federation sync < 5s

## Release Process

### Phase 1: Preparation
1. **Open release issue**: Create `Release vX.Y.Z` issue listing:
   - All issues/PRs to be included
   - Audit checklist status
   - Blockers (MEDIUM+ issues)
2. **Resolve existing MEDIUM+ issues**: All must be closed before audits

### Phase 2: Audits (Gate)
3. **Run all 4 audits**: Security, Privacy, Federation, Code Quality
4. **For each MEDIUM+ finding**:
   - Create a new issue with severity label
   - Link issue as blocker on the release issue
   - Document in `docs/audits/CATEGORY-AUDIT-YYYY-MM-DD.md`
5. **If any MEDIUM+ findings**: Return to Phase 1, fix issues, re-audit
6. **All audits pass with no MEDIUM+ findings**: Proceed to Phase 3

### Phase 3: Release PR
7. **Create release branch**: `git checkout -b release/vX.Y.Z`
8. **Update CHANGELOG.md**: Document all changes for this version
9. **Update version**: In `pyproject.toml` and `src/valence/__init__.py`
10. **Create release PR**: Reference the release issue
    - Title: `Release vX.Y.Z`
    - Body: Changelog excerpt, link to release issue
11. **Review and merge**: Squash merge to main

### Phase 4: Tag and Deploy
12. **Tag release**: `git tag vX.Y.Z && git push --tags`
13. **Deploy to nodes**: Follow deployment runbook
14. **Run deployment validation**: All checks must pass
15. **Create GitHub release**: With changelog notes, close release issue

## Audit Report Template

Save audit reports to `docs/audits/CATEGORY-AUDIT-YYYY-MM-DD.md`:

```markdown
# [Category] Audit Report - YYYY-MM-DD

## Summary
- **Release**: vX.Y.Z
- **Auditor**: [name/agent]
- **Status**: PASS / FAIL

## Findings

### CRITICAL
(none or list)

### HIGH
(none or list)

### MEDIUM
(none or list)

### LOW
(none or list)

## Recommendations
...
```

## Post-Release

- [ ] Verify nodes are running new version
- [ ] Monitor error rates for 24 hours
- [ ] Update HEARTBEAT.md with new status
- [ ] Announce release (if applicable)
