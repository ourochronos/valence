# Valence v0.2 Code Quality Audit

**Date:** 2026-02-04
**Auditor:** Claude (subagent)
**Scope:** Coverage, type safety, test failures, code smells

---

## Executive Summary

Valence v0.2 has **59% overall line coverage** and **47% branch coverage** — adequate for early development but with critical gaps in federation, auth, and storage modules. **204 type errors** exist across 44 files, concentrated in `trust.py` (30), `protocol.py` (15), and `mcp_server.py` (14).

**7 tests are failing:**
- 3 require a database connection (test isolation issue)
- 2 are regressions in transitive trust computation (returning `None`)
- 1 is a flaky statistical assertion
- 1 is a test isolation issue (passes individually)

**Critical concerns:**
1. **`NodeClient` is a 99-method, 4,217-line god class** — unmaintainable
2. **12 potential SQL injection vectors** flagged by bandit
3. **Auth/OAuth modules have 0% coverage** — dangerous for production
4. **30 type errors in `trust.py`** — core module needs attention

---

## Coverage Gaps (Prioritized by Risk)

### 🔴 Critical: 0% Coverage (Production Risk)

| Module | Lines | Risk |
|--------|-------|------|
| `server/oauth.py` | 0% | **Auth bypass risk** |
| `server/oauth_models.py` | 0% | **Token validation untested** |
| `server/auth.py` | 28.8% | **Incomplete auth coverage** |
| `storage/backend.py` | 0% | **Data integrity risk** |
| `storage/erasure.py` | 0% | **GDPR compliance risk** |
| `storage/integrity.py` | 0% | **Corruption undetected** |
| `server/app.py` | 0% | **Server entrypoint untested** |

### 🟠 High Risk: <30% Coverage (Federation/Sync)

| Module | Coverage | Concern |
|--------|----------|---------|
| `federation/threat_detector.py` | 10.8% | Security monitoring blind |
| `embeddings/federation.py` | 11.2% | Federated search untested |
| `server/federation_endpoints.py` | 27.4% | API surface exposed |
| `federation/identity.py` | 38.4% | DID handling gaps |
| `federation/trust_policy.py` | 42.9% | Policy enforcement unclear |
| `federation/protocol.py` | 59.7% | Wire protocol incomplete |
| `federation/sync.py` | 61.7% | Sync edge cases |

### 🟡 Medium Risk: Core Modules

| Module | Coverage | Concern |
|--------|----------|---------|
| `core/confidence.py` | 19.6% | 6D confidence incomplete |
| `core/db.py` | 24.5% | DB layer gaps |
| `core/corroboration.py` | 26.4% | Source verification weak |
| `core/verification.py` | 34.3% | Signature verification |
| `consensus/anti_gaming.py` | 20.6% | Sybil resistance untested |
| `consensus/selection.py` | 16.3% | Validator selection |

---

## Type Safety Issues

**Total errors:** 204
**Files affected:** 44

### By Category

| Error Type | Count | Severity |
|------------|-------|----------|
| `arg-type` | 47 | High — wrong types passed |
| `attr-defined` | 38 | High — accessing missing attributes |
| `assignment` | 29 | Medium — type mismatches |
| `union-attr` | 25 | Medium — Optional not handled |
| `operator` | 13 | Medium — arithmetic type errors |
| `var-annotated` | 10 | Low — missing annotations |
| `return-value` | 9 | High — wrong return types |
| `name-defined` | 6 | High — undefined names |
| `import-untyped` | 6 | Low — missing stubs |

### Worst Offenders

| File | Errors | Priority |
|------|--------|----------|
| `privacy/trust.py` | 30 | 🔴 Critical |
| `federation/protocol.py` | 15 | 🔴 Critical |
| `substrate/mcp_server.py` | 14 | 🟠 High |
| `network/node.py` | 12 | 🟠 High |
| `network/config.py` | 9 | 🟡 Medium |
| `network/seed.py` | 8 | 🟡 Medium |
| `network/router.py` | 8 | 🟡 Medium |

### Critical Type Bugs (Runtime Risk)

```python
# privacy/trust.py:2370 - Missing method called
"TrustService" has no attribute "get_federation_trust"

# privacy/reports.py:669 - Await missing on async iterators
"Coroutine[...]" has no attribute "__aiter__" (not async iterable)
# Note: Maybe you forgot to use "await"? (6 instances)

# federation/consent.py:693 - Can return None when non-None expected
Incompatible return value type (got "CrossFederationConsentChain | None",
  expected "CrossFederationConsentChain")

# storage/backend.py:228 - List type mismatch
No overload variant of "__setitem__" of "list" matches argument types "int", "StorageShard"
```

---

## Failed Tests Analysis

### Summary: 7 Failed / 4377 Passed / 64 Skipped

| Test | Root Cause | Fix Priority |
|------|------------|--------------|
| `test_full_crud_cycle` | DB required, no skip | Low (mark integration) |
| `test_graph_queries` | DB required, no skip | Low (mark integration) |
| `test_domain_scoped_trust` | DB required, no skip | Low (mark integration) |
| `test_max_hops_respected` | **Regression** — returns None | 🔴 High |
| `test_respect_delegation_flag` | **Regression** — returns None | 🔴 High |
| `test_best_router_selected_for_messages` | Flaky statistical assertion | 🟡 Medium |
| `test_probe_router_no_endpoints` | Test isolation issue | 🟡 Medium |

### Root Cause: Trust Computation Regression

Two tests fail because `compute_transitive_trust()` is returning `None` when it should return a trust value. This likely relates to the 30 type errors in `trust.py`, specifically:

```python
# trust.py:1416 - get_edges_from returns list, assigned to dict
Incompatible types in assignment (expression has type "list[TrustEdge] | Any",
  variable has type "dict[str, TrustEdge]")
```

The method signature changed but callers weren't updated.

### Fix: Database Tests

```python
@pytest.mark.skipif(
    not os.getenv("VALENCE_TEST_DB"),
    reason="Requires database connection"
)
class TestTrustGraphStoreIntegration:
    ...
```

---

## Code Smells

### 🔴 God Classes (Unmaintainable)

| Class | Methods | Lines | Recommendation |
|-------|---------|-------|----------------|
| `NodeClient` | **99** | **4,217** | Split into NodeTransport, NodeCircuit, NodeMessaging |
| `SeedNode` | 32 | 1,331 | Extract registration, discovery handlers |
| `RouterNode` | 36 | 1,236 | Separate routing table, message queue |
| `DiscoveryClient` | 31 | 978 | Split by concern |

**`NodeClient` is a severe design issue.** At 99 methods, no one can reason about it. Suggested decomposition:
- `NodeTransport` — connection management (websocket, circuits)
- `NodeMessaging` — message send/receive/queue
- `NodeDiscovery` — peer discovery, gossip
- `NodeCircuit` — onion routing, cover traffic

### 🟠 Long Functions (>80 lines)

| Function | Lines | File | Issue |
|----------|-------|------|-------|
| `list_tools` | 326 | vkb/mcp_server.py | One giant switch |
| `list_tools` | 313 | substrate/mcp_server.py | Duplicate of above |
| `_extend_trust_service` | 283 | privacy/trust.py | Monkey-patching gone wild |
| `propagate` | 256 | privacy/sharing.py | Needs state machine |
| `cmd_query` | 229 | cli/main.py | CLI should delegate |
| `remove_member` | 228 | federation/groups.py | Complex orchestration |

### 🟡 Duplicate Code Patterns

1. **`list_tools` in MCP servers** — 326 vs 313 lines of nearly identical tool registration. Extract to shared base.

2. **Handler patterns** — 17 `async def _handle_*` methods across network code. Consider a registry pattern:
   ```python
   @message_handler("deliver")
   async def _handle_deliver(self, msg): ...
   ```

3. **Error response building** — Repeated across `server/*.py`. Extract `create_error_response()`.

---

## Security Issues (Bandit)

| Severity | Count |
|----------|-------|
| HIGH | 1 |
| MEDIUM | 18 |
| LOW | 20 |

### High Severity

- **`storage/backend.py:336`** — MD5 used for security
  ```python
  # Use SHA256 or add usedforsecurity=False
  hashlib.md5(data)  # WEAK
  ```

### SQL Injection Vectors (12 locations)

Files flagged for string-based query construction:
- `core/corroboration.py:302`
- `core/db.py:243`
- `federation/discovery.py:506`
- `federation/peer_sync.py` (3 locations)
- `federation/protocol.py` (3 locations)
- `federation/sync.py` (3 locations)

**Recommendation:** Audit each and ensure parameterized queries.

---

## Recommendations

### Immediate (This Sprint)

1. **Fix trust computation regression** — The 2 failing tests indicate broken core functionality. Check `get_edges_from` return type and callers.

2. **Add type: ignore or fix** for the 6 "await missing" errors in `privacy/reports.py` — these are runtime bugs waiting to happen.

3. **Mark DB tests as integration** — Skip when no database available.

### Short Term (Next 2 Sprints)

4. **Split `NodeClient`** — This 4,200-line class is a maintenance nightmare. Create focused sub-classes.

5. **Add OAuth test coverage** — 0% on authentication is unacceptable. Write:
   - Token generation/validation
   - Expired token rejection
   - Invalid signature rejection

6. **Fix `trust.py` type errors** — 30 errors in one file suggests a refactor went wrong. Prioritize fixing attribute/return type mismatches.

### Medium Term

7. **Extract MCP tool registration** — The duplicate 300+ line functions should share infrastructure.

8. **Parameterize all SQL** — Audit the 12 flagged locations. Even if currently safe, string concatenation is a maintenance hazard.

9. **Raise coverage on storage layer** — 0% coverage on the persistence layer is dangerous. Prioritize `backend.py` and `erasure.py`.

---

## Metrics Summary

| Metric | Value | Target |
|--------|-------|--------|
| Line Coverage | 59% | 80% |
| Branch Coverage | 47% | 70% |
| Type Errors | 204 | 0 |
| Failed Tests | 7 | 0 |
| God Classes (>30 methods) | 4 | 0 |
| Long Functions (>80 lines) | 20 | 0 |
| Security Issues (High) | 1 | 0 |
| SQL Injection Flags | 12 | 0 |

---

*Generated by audit-quality-v02 subagent*
