"""Conftest for integration tests.

Provides database fixtures for tests that require real PostgreSQL.
All tests in this directory are automatically marked as integration tests
and will be skipped if PostgreSQL is not available.
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

# Import psycopg2 only if available
try:
    import psycopg2
    import psycopg2.extras
    from psycopg2.extras import Json

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


# ============================================================================
# Database Connection Configuration
# ============================================================================


def _get_db_params() -> dict:
    """Get database connection parameters from environment."""
    return {
        "host": os.environ.get("VKB_DB_HOST", "localhost"),
        "port": int(os.environ.get("VKB_DB_PORT", "5432")),
        "database": os.environ.get("VKB_DB_NAME", "valence"),
        "user": os.environ.get("VKB_DB_USER", "valence"),
        "password": os.environ.get("VKB_DB_PASSWORD", ""),
    }


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """Mark all tests in integration directory appropriately.

    Note: The main conftest.py handles skipping based on PostgreSQL
    availability. This hook just ensures tests are properly marked.
    """
    for item in items:
        # Tests in integration directory get the integration marker
        if "integration" in str(item.fspath):
            if "integration" not in item.keywords:
                item.add_marker(pytest.mark.integration)


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture
def db_conn() -> Generator:
    """Provide a database connection with automatic rollback.

    This fixture provides a connection that automatically rolls back
    all changes after each test, ensuring test isolation.

    Requires:
        - PostgreSQL database available
        - psycopg2 installed
        - VKB_DB_* environment variables set (or defaults used)

    Yields:
        psycopg2 connection object
    """
    if not PSYCOPG2_AVAILABLE:
        pytest.skip("psycopg2 not installed")

    params = _get_db_params()

    try:
        conn = psycopg2.connect(**params, connect_timeout=5)
    except psycopg2.OperationalError as e:
        pytest.skip(f"PostgreSQL not available: {e}")

    yield conn

    # Always rollback to ensure test isolation
    try:
        conn.rollback()
    except Exception:
        pass
    finally:
        conn.close()


@pytest.fixture
def db_conn_committed() -> Generator:
    """Provide a database connection where changes persist within the test.

    Unlike db_conn, this fixture commits changes so they're visible
    within the same test (e.g., for testing queries against seeded data).
    Changes are still rolled back at the end of the test.

    Requires:
        - PostgreSQL database available
        - psycopg2 installed
        - VKB_DB_* environment variables set (or defaults used)

    Yields:
        psycopg2 connection object with autocommit disabled
    """
    if not PSYCOPG2_AVAILABLE:
        pytest.skip("psycopg2 not installed")

    params = _get_db_params()

    try:
        conn = psycopg2.connect(**params, connect_timeout=5)
    except psycopg2.OperationalError as e:
        pytest.skip(f"PostgreSQL not available: {e}")

    # Create savepoint for cleanup
    cursor = conn.cursor()
    cursor.execute("SAVEPOINT test_savepoint")

    yield conn

    # Rollback to savepoint to clean up test data
    try:
        cursor.execute("ROLLBACK TO SAVEPOINT test_savepoint")
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


@pytest.fixture
def seed_beliefs(db_conn_committed) -> list[dict]:
    """Seed the database with test beliefs.

    Creates a set of test beliefs for integration testing.

    Args:
        db_conn_committed: Database connection fixture

    Returns:
        List of created belief records
    """
    beliefs_data = [
        {
            "content": "Python is a popular programming language",
            "confidence": {"overall": 0.95},
            "domain_path": ["tech", "programming"],
        },
        {
            "content": "Claude is an AI assistant made by Anthropic",
            "confidence": {"overall": 0.90},
            "domain_path": ["tech", "ai"],
        },
        {
            "content": "PostgreSQL is a powerful relational database",
            "confidence": {"overall": 0.85},
            "domain_path": ["tech", "databases"],
        },
        {
            "content": "Test-driven development improves code quality",
            "confidence": {"overall": 0.75},
            "domain_path": ["tech", "practices"],
        },
        {
            "content": "Coffee is essential for developers",
            "confidence": {"overall": 0.60},
            "domain_path": ["humor", "opinions"],
        },
    ]

    created = []
    with db_conn_committed.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for belief in beliefs_data:
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES (%s, %s, %s)
                RETURNING id, content, confidence, domain_path, status, created_at
            """,
                (
                    belief["content"],
                    Json(belief["confidence"]),
                    belief["domain_path"],
                ),
            )
            created.append(cur.fetchone())

        db_conn_committed.commit()

    return created


@pytest.fixture
def seed_entities(db_conn_committed) -> list[dict]:
    """Seed the database with test entities.

    Creates a set of test entities for integration testing.

    Args:
        db_conn_committed: Database connection fixture

    Returns:
        List of created entity records
    """
    entities_data = [
        {
            "name": "Python",
            "entity_type": "programming_language",
            "aliases": ["python3", "py"],
        },
        {
            "name": "PostgreSQL",
            "entity_type": "database",
            "aliases": ["postgres", "pg"],
        },
        {
            "name": "Claude",
            "entity_type": "ai_assistant",
            "aliases": ["claude-3", "anthropic-claude"],
        },
    ]

    created = []
    with db_conn_committed.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for entity in entities_data:
            cur.execute(
                """
                INSERT INTO entities (name, entity_type, aliases)
                VALUES (%s, %s, %s)
                RETURNING id, name, entity_type, aliases, created_at
            """,
                (
                    entity["name"],
                    entity["entity_type"],
                    entity["aliases"],
                ),
            )
            created.append(cur.fetchone())

        db_conn_committed.commit()

    return created
