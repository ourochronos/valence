# Code Quality Audit - Valence

**Date:** 2026-02-04  
**Auditor:** OpenClaw Subagent (audit-code-quality)  
**Scope:** Full codebase analysis (71 source files, ~37K lines)

---

## Executive Summary

| Category | Grade | Notes |
|----------|-------|-------|
| **Test Coverage** | B- | 75% overall, but significant gaps in MCP servers |
| **Error Handling** | C+ | Inconsistent patterns, many bare `except Exception` |
| **Code Structure** | B | Clean imports, some large functions need refactoring |
| **Documentation** | A- | Excellent high-level docs, API docstrings could improve |
| **Type Safety** | C | 95 mypy errors, type hints present but incomplete |
| **Performance** | B | No obvious N+1 patterns, could improve query batching |

**Overall: B-** — Solid foundation with clear areas for improvement.

---

## 1. Test Coverage Analysis

### Current State
```
Overall Coverage: 75%
Tests: 2,067 passed, 12 failed, 50 skipped, 9 xfailed
Test Files: 89
Source Files: 71
```

### Module Coverage Breakdown

| Module | Coverage | Risk Level |
|--------|----------|------------|
| `core.models` | 100% ✅ | Low |
| `core.confidence` | 100% ✅ | Low |
| `core.exceptions` | 100% ✅ | Low |
| `core.db` | 77% | Medium |
| `substrate.tools` | 93% ✅ | Low |
| `substrate.mcp_server` | **45%** ⚠️ | **High** |
| `vkb.tools` | 97% ✅ | Low |
| `vkb.mcp_server` | **67%** | Medium |
| `server.app` | 80% | Low |
| `server.auth` | ~60% | Medium |
| `federation.*` | ~50-70% | Medium-High |
| `cli.main` | ~60% | Medium |

### Critical Gaps

1. **`substrate/mcp_server.py`** (45% coverage)
   - Lines 377, 402-512, 634-684, 846-849, 1004-1010, 1100-1425 uncovered
   - MCP server handlers largely untested
   - Risk: Runtime failures in production agent integrations

2. **`vkb/mcp_server.py`** (67% coverage)
   - Lines 845-958, 970-1004 uncovered
   - Session management handlers need tests

3. **`cli/federation.py`** — 12 failing tests
   - Issues with mock setup and attribute errors
   - Needs immediate attention

### Recommendations

- **Priority 1:** Fix the 12 failing federation CLI tests
- **Priority 2:** Add integration tests for MCP server handlers
- **Priority 3:** Add edge case tests for error paths in substrate tools

---

## 2. Error Handling Assessment

### Current Patterns

**Bare `except Exception` blocks:** 30+ instances

Examples:
```python
# src/valence/cli/main.py (multiple instances)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

# src/valence/server/oauth.py
except Exception:  # No logging at all
    pass
```

### Issues Found

| Issue | Count | Severity |
|-------|-------|----------|
| Bare `except Exception` | 30+ | Medium |
| Silent exceptions (no logging) | 5+ | High |
| Missing specific exception types | ~20 | Medium |
| Inconsistent error messages | ~15 | Low |

### Specific Problems

1. **Silent OAuth failures** (`server/oauth.py:94, 303`)
   ```python
   except Exception:  # Silent catch - should log
       pass
   ```

2. **Generic catch-all in federation** (`federation_endpoints.py`)
   - Catches all exceptions, potentially masking bugs

3. **No structured error codes** in CLI
   - Exit codes inconsistent (sometimes 1, sometimes unset)

### Custom Exception Usage

The codebase has good custom exceptions in `core/exceptions.py`:
- `DatabaseException`
- `ValidationException`
- `EmbeddingException`

**Problem:** These are underutilized. Many modules catch generic `Exception` instead.

### Recommendations

```python
# Instead of:
except Exception as e:
    print(f"Error: {e}")

# Use:
except DatabaseException as e:
    logger.error("Database error", exc_info=True)
    raise SystemExit(1)
except Exception as e:
    logger.exception("Unexpected error")
    raise
```

---

## 3. Code Structure Analysis

### Import Organization

**Good:** Uses relative imports consistently
```python
from ..core.db import get_cursor
from ..core.models import Belief, Entity
```

### Dependency Graph

```
core/
├── db.py (foundation)
├── models.py (data structures)
├── confidence.py (confidence calculations)
└── exceptions.py (error types)
    ↓
substrate/, vkb/, embeddings/ (depend on core)
    ↓
federation/ (depends on core + substrate)
    ↓
server/, cli/ (top-level consumers)
```

**No circular dependencies detected.** ✅

### Function Complexity

**Overly long functions identified:**

| File | Function | Lines |
|------|----------|-------|
| `cli/main.py` | `backfill_embeddings` | 200+ |
| `federation/protocol.py` | Various handlers | 100+ each |
| `substrate/mcp_server.py` | Tool handlers | 80+ each |

The awk analysis showed suspicious counts (1000+ lines) but this is likely an artifact of the tool counting to end-of-file rather than function boundaries. Manual inspection shows functions averaging 30-80 lines.

### Dead Code

**TODO markers found:** 5
```
federation/discovery.py:617  # TODO: Implement peer exchange protocol
federation/sync.py:401        # TODO: Add authentication headers
federation/sync.py:538        # TODO: Add authentication
federation/protocol.py:461    # TODO: Store session token (use Redis)
federation/protocol.py:767    # TODO: Implement cursor-based pagination
```

These represent incomplete features, not dead code.

### Recommendations

1. **Extract helper functions** from `backfill_embeddings` in CLI
2. **Split protocol.py** into separate handler files
3. **Create a shared utilities module** for common patterns

---

## 4. Documentation Assessment

### High-Level Documentation

**Excellent coverage:**
- `README.md` — Comprehensive overview ✅
- `docs/VISION.md` — Philosophy and goals ✅
- `docs/SYSTEM.md` — Architecture overview ✅
- `docs/FEDERATION_PROTOCOL.md` — Protocol spec ✅
- `docs/TRUST_MODEL.md` — Trust calculations ✅
- `docs/openapi.yaml` — API specification ✅

### Code Documentation

**Docstring coverage:** ~60%

| Module Type | Docstring Coverage | Quality |
|-------------|-------------------|---------|
| Core models | Good | Clear descriptions |
| CLI commands | Good | Usage examples |
| MCP tools | Partial | Missing param docs |
| Federation handlers | Sparse | Needs improvement |

### Missing Documentation

1. **Public API docstrings** in federation handlers
2. **Complex algorithm explanations** in `consensus/anti_gaming.py`
3. **Configuration reference** (environment variables scattered)

### Recommendations

1. Add Google-style docstrings to all public functions:
   ```python
   def verify_belief_signature(belief: Belief) -> bool:
       """Verify the cryptographic signature of a belief.
       
       Args:
           belief: The belief to verify.
           
       Returns:
           True if signature is valid, False otherwise.
           
       Raises:
           SignatureVerificationError: If signature data is malformed.
       """
   ```

2. Create `docs/CONFIG.md` with all environment variables

---

## 5. Type Safety Analysis

### Mypy Results

```
Found 95 errors in 23 files (checked 71 source files)
```

### Error Categories

| Category | Count | Severity |
|----------|-------|----------|
| `arg-type` mismatches | 25 | Medium |
| `str | None` handling | 20 | Medium |
| Missing stubs (`yaml`) | 5 | Low |
| `attr-defined` errors | 10 | High |
| Incompatible return types | 15 | Medium |
| Index/assignment errors | 10 | Medium |

### Critical Type Issues

1. **Missing attribute access** (`protocol.py:1047-1072`)
   ```python
   # "ProtocolMessage" has no attribute "client_did"
   ```
   Suggests incomplete type hierarchy or wrong type being passed.

2. **Optional handling failures** (`federation_endpoints.py`)
   ```python
   # Argument 1 to "int" has incompatible type "str | None"
   ```
   Missing None checks before conversion.

3. **MCP URI types** (`mcp_server.py`)
   ```python
   # Argument "uri" to "Resource" has incompatible type "str"; expected "AnyUrl"
   ```
   Need to use proper Pydantic URL types.

### Pydantic Usage

Limited to `server/config.py`:
```python
from pydantic_settings import BaseSettings
```

**Opportunity:** Core models use dataclasses but could benefit from Pydantic for validation.

### Recommendations

1. **Install type stubs:** `pip install types-PyYAML`
2. **Fix None handling:** Add explicit None checks
3. **Consider Pydantic for core models** — would provide validation
4. **Run mypy in CI** with `--strict` eventually

---

## 6. Performance Analysis

### Database Patterns

**Query structure:** Generally good
- Uses connection pooling (`get_cursor` context manager)
- Parameterized queries (no SQL injection risks)

**Potential N+1 patterns:** None found in explicit code review.

However, some loops that fetch embeddings could be batched:
```python
# cli/main.py - potential improvement
for belief in beliefs:
    embedding = get_embedding(belief.content)  # Could batch
```

### Embedding Service

**Current:** One OpenAI API call per belief
**Opportunity:** Batch up to 2048 inputs per API call

### Index Coverage

Indexes defined in migrations appear comprehensive. Would need production query analysis to verify.

### Recommendations

1. **Batch embedding calls** in `backfill_embeddings`
2. **Add connection pooling** with pgbouncer for production
3. **Profile slow queries** with `EXPLAIN ANALYZE`

---

## 7. Technical Debt Inventory

### High Priority

| Item | Location | Effort | Impact |
|------|----------|--------|--------|
| Fix 12 failing federation tests | `tests/cli/test_federation.py` | 2-4h | High |
| Fix mypy errors (95) | Various | 4-8h | Medium |
| Add MCP server tests | `tests/mcp/` | 4-8h | High |
| Improve error handling consistency | `cli/main.py`, `server/` | 2-4h | Medium |

### Medium Priority

| Item | Location | Effort | Impact |
|------|----------|--------|--------|
| Complete TODO items (5) | `federation/` | 4-8h | Medium |
| Add missing docstrings | Various | 4-8h | Low |
| Extract large functions | `cli/main.py` | 2-4h | Low |
| Batch embedding calls | `cli/main.py` | 1-2h | Medium |

### Low Priority

| Item | Location | Effort | Impact |
|------|----------|--------|--------|
| Fix ruff warnings (3096) | Various | 2-4h | Low |
| Add Pydantic to core models | `core/models.py` | 4-8h | Low |
| Create CONFIG.md | `docs/` | 1-2h | Low |

---

## 8. Recommendations Summary

### Immediate Actions (This Week)

1. ✅ Fix 12 failing tests in `test_federation.py`
2. ✅ Install `types-PyYAML` for mypy
3. ✅ Add logging to silent exception catches in OAuth

### Short-term (2 Weeks)

1. Increase `substrate/mcp_server.py` coverage to 70%+
2. Fix critical mypy errors (attribute access, None handling)
3. Complete TODO items in federation sync

### Medium-term (1 Month)

1. Achieve 80% overall test coverage
2. Zero mypy errors with `--ignore-missing-imports`
3. Consistent error handling patterns across codebase

### Long-term (3 Months)

1. Run mypy with `--strict`
2. Migrate core models to Pydantic v2
3. Full docstring coverage

---

## Appendix: Commands Used

```bash
# Test coverage
pytest --cov=src/valence --cov-report=term-missing --ignore=tests/integration

# Type checking
mypy src/valence --ignore-missing-imports

# Linting
ruff check src/valence

# Code metrics
wc -l src/valence/**/*.py
find . -name "*.py" -path "./src/*" | wc -l
```

---

*Report generated by OpenClaw audit-code-quality subagent*
