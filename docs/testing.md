# Testing Guide

This document describes how to run tests for the Valence project.

## Test Categories

Tests are organized into categories using pytest markers:

- **Unit tests** (`-m "not integration and not slow"`): Fast tests with no external dependencies
- **Integration tests** (`-m integration`): Tests requiring database/server
- **Slow tests** (`-m slow`): Long-running tests (benchmarks, stress tests)

## Quick Start

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run unit tests only (fast, no DB required)
make test

# Run all tests (requires DB)
make test-all
```

## Integration Tests

### Local Development

Integration tests require a PostgreSQL database with pgvector.

#### Using Docker (Recommended)

```bash
# Start test environment (postgres + servers)
make docker-up

# Run integration tests
make test-integration

# Run specific test file
pytest tests/integration/test_belief_lifecycle.py -v

# Stop environment
make docker-down
```

#### Using Local PostgreSQL

If you have PostgreSQL with pgvector installed locally:

```bash
# Set environment variables
export VALENCE_DB_HOST=localhost
export VALENCE_DB_PORT=5432
export VALENCE_DB_NAME=valence_test
export VALENCE_DB_USER=valence
export VALENCE_DB_PASSWORD=yourpassword

# Initialize schema
make db-init

# Run tests
make test-integration
```

### Test Environment Architecture

The Docker test environment (`docker-compose.test.yml`) includes:

```
┌─────────────────┐     ┌─────────────────┐
│  postgres       │     │  postgres-peer  │
│  (port 5433)    │     │  (port 5434)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
┌────────┴────────┐     ┌────────┴────────┐
│ valence-primary │     │  valence-peer   │
│  (port 8080)    │────▶│  (port 8081)    │
└─────────────────┘     └─────────────────┘
         │
┌────────┴────────┐
│  test-runner    │
│  (pytest)       │
└─────────────────┘
```

- **postgres**: Primary test database
- **postgres-peer**: Secondary database for federation tests
- **valence-primary**: Primary Valence server instance
- **valence-peer**: Peer server for federation testing
- **test-runner**: Container that runs pytest

### Integration Test Files

| File | Description |
|------|-------------|
| `test_deployment.py` | Database schema verification |
| `test_belief_lifecycle.py` | Belief CRUD operations |
| `test_federation_sync.py` | Multi-node federation |
| `test_trust_propagation.py` | Trust calculation and propagation |

### Fixtures

Key fixtures available for integration tests:

```python
# Database connections
db_conn              # Auto-rollback after each test
db_conn_committed    # Commits changes (with cleanup)
peer_db_conn         # Connection to peer database

# Server URLs
primary_api          # URL for primary server
peer_api             # URL for peer server

# Test data seeding
seed_beliefs         # Pre-populated test beliefs
seed_entities        # Pre-populated test entities
seed_session         # Test session with exchanges
seed_federation_nodes # Federation node records
seed_node_trust      # Trust relationships

# Factories
make_belief          # Create custom test beliefs
make_session         # Create custom test sessions
```

## CI Integration

Tests run automatically in GitHub Actions:

### Unit Tests (`ci.yml`)
- Run on every PR and push to main
- Fast, no external dependencies
- Required to pass for merge

### Integration Tests (`integration.yml`)
- Run on PR, push to main, and nightly at 2 AM UTC
- Database integration tests always run
- Federation tests run on main branch only
- Can be manually triggered with debug mode

### Coverage

Test coverage is tracked via Codecov with separate flags:
- `unit`: Unit test coverage
- `integration-db`: Database integration coverage
- `integration-federation`: Federation test coverage
- `integration-api`: API test coverage

## Troubleshooting

### Database Connection Failed

```bash
# Check if postgres is running
docker ps | grep postgres

# View postgres logs
make docker-logs

# Reset the database
make db-reset
```

### Server Not Starting

```bash
# Check server health
curl http://localhost:8080/api/v1/health

# View server logs
docker compose -f docker-compose.test.yml logs valence-primary
```

### Tests Timing Out

```bash
# Increase timeout for slow tests
pytest tests/integration/ -v --timeout=300

# Run with debug output
pytest tests/integration/ -v -s --tb=long
```

## Writing Integration Tests

### Best Practices

1. **Use fixtures for database connections** - Don't create connections directly
2. **Use `db_conn` for tests that shouldn't persist data** - Auto-rollback
3. **Use `db_conn_committed` when testing committed state** - With cleanup
4. **Mark all integration tests** - `@pytest.mark.integration`
5. **Handle missing tables gracefully** - Some tables may not exist

### Example Test

```python
import pytest
import psycopg2.extras
from psycopg2.extras import Json

pytestmark = pytest.mark.integration

class TestMyFeature:
    def test_create_something(self, db_conn):
        """Test creating a record."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES (%s, %s, %s)
                RETURNING id
            """, ("Test belief", Json({"overall": 0.8}), ["test"]))
            
            result = cur.fetchone()
            assert result is not None
            assert result["id"] is not None
```
