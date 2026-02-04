# Coverage Tracking

**Current Overall: 54%**

Last updated: 2026-02-03

## Targets

| Milestone | Target | Date |
|-----------|--------|------|
| Current | 54% | Now |
| Short-term | 60% | 2026-02-10 |
| Medium-term | 70% | 2026-03-03 |
| Stable | 80% | Ongoing |

## Module Coverage

Coverage varies significantly by module. Priority is given to core modules.

### Core (Target: 80%+)
| Module | Coverage | Priority |
|--------|----------|----------|
| `core.models` | ~75% | ✅ High |
| `core.confidence` | ~70% | ✅ High |
| `core.db` | ~65% | ✅ High |
| `core.exceptions` | ~80% | ✅ High |
| `core.health` | ~60% | Medium |
| `core.logging` | ~50% | Low |
| `core.mcp_base` | ~55% | Medium |
| `core.corroboration` | ~60% | Medium |
| `core.temporal` | ~65% | Medium |

### Substrate (Target: 75%+)
| Module | Coverage | Priority |
|--------|----------|----------|
| `substrate.tools` | ~70% | ✅ High |
| `substrate.mcp_server` | ~55% | Medium |

### Server (Target: 70%+)
| Module | Coverage | Priority |
|--------|----------|----------|
| `server.app` | ~60% | ✅ High |
| `server.auth` | ~50% | ✅ High |
| `server.oauth` | ~45% | ✅ High |
| `server.config` | ~70% | Medium |
| `server.cli` | ~40% | Low |
| `server.*_endpoints` | ~50% | Medium |

### Federation (Target: 65%+)
| Module | Coverage | Priority |
|--------|----------|----------|
| `federation.models` | ~60% | Medium |
| `federation.protocol` | ~55% | Medium |
| `federation.trust` | ~50% | ✅ High |
| `federation.peers` | ~45% | Medium |
| `federation.sync` | ~40% | Medium |
| `federation.discovery` | ~35% | Low |
| `federation.threat_detector` | ~30% | Low |

### Embeddings (Target: 70%+)
| Module | Coverage | Priority |
|--------|----------|----------|
| `embeddings.service` | ~65% | ✅ High |
| `embeddings.registry` | ~70% | Medium |

### CLI (Target: 60%+)
| Module | Coverage | Priority |
|--------|----------|----------|
| `cli.main` | ~45% | Medium |

### Compliance (Target: 75%+)
| Module | Coverage | Priority |
|--------|----------|----------|
| `compliance.pii_scanner` | ~55% | ✅ High |
| `compliance.deletion` | ~60% | ✅ High |

### VKB/Agents (Target: 50%+)
| Module | Coverage | Priority |
|--------|----------|----------|
| `vkb.tools` | ~50% | Medium |
| `vkb.mcp_server` | ~45% | Low |
| `agents.matrix_bot` | ~30% | Low |

## Priority Queue

Modules to focus on next (biggest impact):

1. **`server.auth`** — Security-critical, needs 80%+
2. **`server.oauth`** — Security-critical, needs 80%+
3. **`federation.trust`** — Core trust calculations
4. **`compliance.pii_scanner`** — Privacy-critical
5. **`core.db`** — Foundation layer

## CI Enforcement

- **Project target**: 50% (fail if below)
- **Patch target**: 60% (new code must be well-tested)
- **Threshold**: 2% (allow small regressions)

See `codecov.yml` for configuration.

## How to Improve

```bash
# Run with coverage locally
pytest tests/ -v --cov=src/valence --cov-report=html

# Open coverage report
open htmlcov/index.html

# Check specific module
pytest tests/ -v --cov=src/valence/core --cov-report=term-missing
```

## Notes

- Coverage percentages are estimates based on initial analysis
- Focus on meaningful tests over coverage numbers
- Integration tests count toward coverage but run separately
