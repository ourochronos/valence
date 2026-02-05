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
- [ ] Tests pass locally
- [ ] CI green
- [ ] PR ready for review

Model: [opus/sonnet/haiku]
Timeout: [seconds]
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
