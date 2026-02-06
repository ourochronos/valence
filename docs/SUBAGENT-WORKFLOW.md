# Sub-Agent Workflow

## Overview

Parallelized work uses a graph-style coordination model where sub-agents work independently until completion.

## Principles

### 1. Work Until Green
Sub-agents continue work until their PRs pass CI:
- Don't just create PR and stop
- Monitor CI results
- Fix failures automatically
- Only report completion when CI is green

### 2. Graph-Style Coordination
- Main agent maintains dependency graph of work items
- Independent work runs in parallel
- Dependent work waits for prerequisites
- Conflicts resolved by rebasing against main

### 3. Heartbeat-Driven Project Assessment
Main agent heartbeat includes:
- Check open PR CI status
- If PR failing → spawn sub-agent to fix
- Assess blocked issues
- Trigger work waves when capacity available

## Model Routing by Complexity

### Issue Complexity Assessment

| Complexity | Indicators | Model |
|------------|------------|-------|
| **High** | Architecture changes, security-critical, cross-cutting concerns, novel problems | Opus |
| **Medium** | Feature implementation, refactoring, test coverage, bug fixes | Sonnet |
| **Low** | Documentation, formatting, simple config, typo fixes | Haiku |

### Routing Heuristics

**Route to Opus when:**
- Issue touches 5+ files
- Security/privacy implications
- API design decisions
- Cross-module refactoring
- Debugging complex failures

**Route to Sonnet when:**
- Standard feature work
- Test writing
- Bug fixes with clear scope
- Documentation with technical depth

**Route to Haiku when:**
- Pure documentation updates
- Config/formatting changes
- Simple one-file fixes
- Changelog updates

## Sub-Agent Task Template

```
Task: [Clear description]
Repo: [path]
Branch: [branch-name]

Requirements:
1. [Specific requirement]
2. [Specific requirement]

Definition of Done:
- [ ] Code complete
- [ ] **Tests updated** — if behavior changes (exceptions, signatures, return types), update affected tests
- [ ] **Docs updated** — README, CHANGELOG, or audit status if applicable
- [ ] Tests pass locally (`./scripts/check`)
- [ ] CI green
- [ ] PR ready for review
- [ ] **Out-of-scope issues filed** — don't fix unrelated problems inline

Model: [opus/sonnet/haiku]
Timeout: [seconds]
```

## Test Expectations

**Code changes require corresponding test updates.** Don't just make code pass tests — ensure tests reflect the new behavior.

### When to Update Tests

| Code Change | Test Update Required |
|-------------|---------------------|
| Change exception type | Update test mocks to raise new type |
| Change function signature | Update test calls with new args |
| Change return type/structure | Update test assertions |
| Add new code path | Add test coverage for it |
| Remove code path | Remove or update affected tests |

### Example: Exception Tightening

**Bad:** Change `except Exception` → `except ValueError` without updating tests.
```python
# Code catches ValueError now
except ValueError as e:
    return None

# Test still mocks generic Exception — WILL FAIL
mock.side_effect = Exception("error")  # ❌
```

**Good:** Update test mock to match:
```python
mock.side_effect = ValueError("error")  # ✅
```

### Pre-Push Checklist

Before pushing any PR:
1. `grep -r "side_effect=Exception" tests/` — check for generic exception mocks
2. Run affected test files directly: `pytest tests/path/to/affected_test.py -v`
3. If tests fail, fix them before pushing

## Out-of-Scope Findings

Sub-agents often discover issues beyond their assigned task. **Don't fix them inline — open issues.**

### When to Open an Issue

| Discovery | Action |
|-----------|--------|
| Bug in unrelated code | Open issue with reproduction steps |
| Missing test coverage elsewhere | Open issue, note affected files |
| Doc inconsistency | Open issue or fix if trivial (<5 lines) |
| Security concern | Open issue with `security` label, flag to main agent |
| Performance problem | Open issue with profiling data if available |
| Architectural smell | Open issue for discussion |

### Issue Template for Sub-Agent Findings

```markdown
## Found During
PR #X / Issue #Y — [brief context]

## Problem
[Clear description]

## Location
`path/to/file.py:123`

## Suggested Fix
[If obvious]

## Severity
[Low/Medium/High]
```

### Why This Matters

- Keeps PRs focused and reviewable
- Creates audit trail of discovered issues
- Allows proper prioritization by main agent
- Prevents scope creep that delays merges

## Documentation Updates

**Code changes often require doc updates.** Check before marking work complete.

### Doc Update Checklist

| Code Change | Doc to Update |
|-------------|---------------|
| New feature | README.md, FEATURE-SPEC.md |
| API change | openapi.yaml, relevant docs/ |
| Config change | README.md (config section) |
| Bug fix | CHANGELOG.md |
| Security fix | CHANGELOG.md, security docs if applicable |
| Behavior change | Affected doc files |

### Where Docs Live

- `README.md` — User-facing overview, quickstart
- `CHANGELOG.md` — All notable changes
- `docs/` — Detailed specs and guides
- `docs/audits/` — Audit reports (update status sections)
- Code docstrings — API documentation

### Audit Doc Updates

When fixing issues found in audits, update the audit doc's status section:
```markdown
| Finding | Status | PR/Issue |

## Audit Sub-Agents

Audit sub-agents analyze the codebase for security, privacy, code quality, or other concerns. They have special requirements to ensure accurate findings.

### Fresh Environment Required

**Always create a fresh venv for audits.** Stale environments can report false positives (e.g., wrong dependency versions).

```bash
# Audit sub-agent setup
cd ~/.openclaw/workspace/repos/valence
git fetch origin && git reset --hard origin/main

# Create fresh venv
python3 -m venv .audit-venv
source .audit-venv/bin/activate
pip install -e ".[dev]" --quiet

# Now run audit checks
pip show PyJWT | grep Version  # Verify actual installed versions
```

### Audit Checklist

Before reporting findings:
- [ ] Fresh venv created and dependencies installed
- [ ] Verified dependency versions match pyproject.toml specs
- [ ] Ran checks against actual installed packages, not cached data
- [ ] Cross-referenced with CI to confirm findings

### Cleanup

Remove audit venv after completion:
```bash
rm -rf .audit-venv
```

### Why This Matters

The PyJWT incident (2026-02-05): Audit reported PyJWT 2.7.0 installed when pyproject.toml specified >=2.11.0. Root cause: stale main venv. Fresh install confirmed 2.11.0 was correct. False positive wasted investigation time.
|---------|--------|----------|
| [finding] | ✅ Fixed | #123 |
```

## Failure Recovery

When sub-agent work fails:
1. Check transcript for root cause
2. If fixable: spawn new sub-agent with context
3. If blocked: escalate to main agent
4. Document failure in daily notes

## PR Lifecycle

```
Issue Created
    ↓
Complexity Assessment → Model Selection
    ↓
Sub-Agent Spawned
    ↓
Work + Local Tests
    ↓
PR Created
    ↓
CI Running ←──────┐
    ↓            │
CI Failed? ──Yes─→ Sub-Agent Fixes
    ↓ No
CI Green
    ↓
Report Complete
    ↓
Main Agent Merges (when ready)
```

## Integration with Heartbeat

Add to HEARTBEAT.md checklist:
```markdown
### PR Health Check
- [ ] List open PRs with CI status
- [ ] For each failing PR: spawn fix sub-agent
- [ ] For PRs waiting >24h: assess blockers
```
