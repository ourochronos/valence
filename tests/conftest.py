"""Global test fixtures for Valence test suite."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

# ============================================================================
# PostgreSQL Availability Detection
# ============================================================================


def _check_postgres_available() -> tuple[bool, str | None]:
    """Check if PostgreSQL is available for integration tests.

    Returns:
        Tuple of (is_available, error_message)
    """
    try:
        import psycopg2
    except ImportError:
        return False, "psycopg2 not installed"

    # Get connection params from environment
    host = os.environ.get("VKB_DB_HOST", "localhost")
    port = int(os.environ.get("VKB_DB_PORT", "5432"))
    dbname = os.environ.get("VKB_DB_NAME", "valence")
    user = os.environ.get("VKB_DB_USER", "valence")
    password = os.environ.get("VKB_DB_PASSWORD", "")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=dbname,
            user=user,
            password=password,
            connect_timeout=3,
        )
        conn.close()
        return True, None
    except psycopg2.OperationalError as e:
        return False, f"PostgreSQL connection failed: {e}"
    except Exception as e:
        return False, f"Unexpected error connecting to PostgreSQL: {e}"


# Check PostgreSQL availability once at module load
POSTGRES_AVAILABLE, POSTGRES_ERROR = _check_postgres_available()


def pytest_configure(config):
    """Register custom markers and check PostgreSQL availability."""
    config.addinivalue_line(
        "markers", "requires_postgres: mark test as requiring a real PostgreSQL database"
    )

    # Store availability status on config for access in fixtures
    config._postgres_available = POSTGRES_AVAILABLE
    config._postgres_error = POSTGRES_ERROR


def pytest_collection_modifyitems(config, items):
    """Add skip markers to tests that require PostgreSQL when DB is unavailable."""
    if POSTGRES_AVAILABLE:
        # PostgreSQL is available, don't skip anything
        return

    skip_postgres = pytest.mark.skip(reason=f"PostgreSQL not available: {POSTGRES_ERROR}")

    for item in items:
        # Skip tests marked with @pytest.mark.integration
        if "integration" in item.keywords:
            item.add_marker(skip_postgres)
        # Skip tests marked with @pytest.mark.requires_postgres
        elif "requires_postgres" in item.keywords:
            item.add_marker(skip_postgres)


# ============================================================================
# Environment Fixtures
# ============================================================================


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all VKB_ and VALENCE_ environment variables."""
    env_prefixes = ("VKB_", "VALENCE_", "OPENAI_")
    for key in list(os.environ.keys()):
        if any(key.startswith(prefix) for prefix in env_prefixes):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def env_with_db_vars(monkeypatch):
    """Set up database environment variables."""
    monkeypatch.setenv("VKB_DB_HOST", "localhost")
    monkeypatch.setenv("VKB_DB_PORT", "5432")
    monkeypatch.setenv("VKB_DB_NAME", "valence_test")
    monkeypatch.setenv("VKB_DB_USER", "valence")
    monkeypatch.setenv("VKB_DB_PASSWORD", "testpass")


@pytest.fixture
def env_without_db_vars(monkeypatch):
    """Remove database environment variables."""
    for var in ["VKB_DB_HOST", "VKB_DB_NAME", "VKB_DB_USER", "VKB_DB_PASSWORD"]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def env_with_openai_key(monkeypatch):
    """Set up OpenAI API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-12345")


# ============================================================================
# Database Mocking Fixtures (asyncpg)
# ============================================================================


@pytest.fixture
def mock_asyncpg():
    """Mock the asyncpg module for async database operations."""
    with patch("asyncpg.create_pool") as mock_create_pool:
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_transaction = AsyncMock()

        # Configure the mock pool
        mock_create_pool.return_value = mock_pool
        mock_pool.acquire.return_value = mock_connection
        mock_pool.release = AsyncMock()
        mock_pool.close = AsyncMock()

        # Configure the mock connection
        mock_connection.fetch = AsyncMock(return_value=[])
        mock_connection.fetchrow = AsyncMock(return_value=None)
        mock_connection.fetchval = AsyncMock(return_value=None)
        mock_connection.execute = AsyncMock(return_value="")

        # Configure transaction context manager
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction.return_value = mock_transaction

        yield {
            "create_pool": mock_create_pool,
            "pool": mock_pool,
            "connection": mock_connection,
            "transaction": mock_transaction,
        }


@pytest.fixture
def mock_db_connection(mock_asyncpg):
    """Get a mock database connection."""
    return mock_asyncpg["connection"]


@pytest.fixture
def mock_get_cursor():
    """Mock the get_cursor async context manager."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock(return_value="")

    @asynccontextmanager
    async def fake_get_cursor(dict_cursor: bool = True) -> AsyncGenerator:
        yield mock_conn

    with patch("valence.core.db.get_cursor", fake_get_cursor):
        yield mock_conn


# ============================================================================
# psycopg2 Connection Pool fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_db_pool():
    """Reset the connection pool singleton before each test."""
    from valence.core import db

    # Reset the pool singleton
    db.ConnectionPool._instance = None
    db._pool = db.ConnectionPool.get_instance()
    yield
    # Cleanup after test
    try:
        db._pool.close_all()
    except Exception:
        pass
    db.ConnectionPool._instance = None


@pytest.fixture
def mock_psycopg2_pool():
    """Mock the psycopg2 connection pool."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_pool = MagicMock()

    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)

    # Mock pool methods
    mock_pool.getconn.return_value = mock_conn
    mock_pool.putconn = MagicMock()
    mock_pool.closeall = MagicMock()

    with patch("valence.core.db.psycopg2_pool.ThreadedConnectionPool") as mock_pool_class:
        mock_pool_class.return_value = mock_pool
        yield {
            "pool_class": mock_pool_class,
            "pool": mock_pool,
            "connection": mock_conn,
            "cursor": mock_cursor,
        }


@pytest.fixture
def mock_psycopg2():
    """Mock the psycopg2 module (legacy - use mock_psycopg2_pool for pool tests)."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)

    with patch("psycopg2.connect", return_value=mock_conn):
        yield {
            "connect": patch("psycopg2.connect"),
            "connection": mock_conn,
            "cursor": mock_cursor,
        }


# ============================================================================
# Model Factory Fixtures
# ============================================================================


@pytest.fixture
def sample_uuid() -> UUID:
    """Generate a sample UUID."""
    return uuid4()


@pytest.fixture
def sample_datetime() -> datetime:
    """Generate a sample datetime."""
    return datetime(2024, 1, 15, 10, 30, 0)


@pytest.fixture
def belief_row_factory():
    """Factory for creating belief database rows."""

    def factory(
        id: UUID | None = None,
        content: str = "Test belief content",
        confidence: dict | None = None,
        domain_path: list[str] | None = None,
        status: str = "active",
        **kwargs,
    ) -> dict[str, Any]:
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "content": content,
            "confidence": json.dumps(confidence or {"overall": 0.7}),
            "domain_path": domain_path or ["test", "domain"],
            "valid_from": kwargs.get("valid_from"),
            "valid_until": kwargs.get("valid_until"),
            "created_at": kwargs.get("created_at", now),
            "modified_at": kwargs.get("modified_at", now),
            "source_id": kwargs.get("source_id"),
            "extraction_method": kwargs.get("extraction_method"),
            "supersedes_id": kwargs.get("supersedes_id"),
            "superseded_by_id": kwargs.get("superseded_by_id"),
            "status": status,
        }

    return factory


@pytest.fixture
def entity_row_factory():
    """Factory for creating entity database rows."""

    def factory(
        id: UUID | None = None, name: str = "Test Entity", type: str = "concept", **kwargs
    ) -> dict[str, Any]:
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "name": name,
            "type": type,
            "description": kwargs.get("description"),
            "aliases": kwargs.get("aliases", []),
            "canonical_id": kwargs.get("canonical_id"),
            "created_at": kwargs.get("created_at", now),
            "modified_at": kwargs.get("modified_at", now),
        }

    return factory


@pytest.fixture
def session_row_factory():
    """Factory for creating session database rows."""

    def factory(
        id: UUID | None = None, platform: str = "claude-code", status: str = "active", **kwargs
    ) -> dict[str, Any]:
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "platform": platform,
            "project_context": kwargs.get("project_context"),
            "status": status,
            "summary": kwargs.get("summary"),
            "themes": kwargs.get("themes", []),
            "started_at": kwargs.get("started_at", now),
            "ended_at": kwargs.get("ended_at"),
            "claude_session_id": kwargs.get("claude_session_id"),
            "external_room_id": kwargs.get("external_room_id"),
            "metadata": kwargs.get("metadata", {}),
            "exchange_count": kwargs.get("exchange_count"),
            "insight_count": kwargs.get("insight_count"),
        }

    return factory


@pytest.fixture
def exchange_row_factory():
    """Factory for creating exchange database rows."""

    def factory(
        id: UUID | None = None,
        session_id: UUID | None = None,
        sequence: int = 1,
        role: str = "user",
        content: str = "Test message",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "id": id or uuid4(),
            "session_id": session_id or uuid4(),
            "sequence": sequence,
            "role": role,
            "content": content,
            "created_at": kwargs.get("created_at", datetime.now()),
            "tokens_approx": kwargs.get("tokens_approx"),
            "tool_uses": kwargs.get("tool_uses", []),
        }

    return factory


@pytest.fixture
def pattern_row_factory():
    """Factory for creating pattern database rows."""

    def factory(
        id: UUID | None = None,
        type: str = "topic_recurrence",
        description: str = "Test pattern",
        **kwargs,
    ) -> dict[str, Any]:
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "type": type,
            "description": description,
            "evidence": kwargs.get("evidence", []),
            "occurrence_count": kwargs.get("occurrence_count", 1),
            "confidence": kwargs.get("confidence", 0.5),
            "status": kwargs.get("status", "emerging"),
            "first_observed": kwargs.get("first_observed", now),
            "last_observed": kwargs.get("last_observed", now),
        }

    return factory


@pytest.fixture
def tension_row_factory():
    """Factory for creating tension database rows."""

    def factory(
        id: UUID | None = None,
        belief_a_id: UUID | None = None,
        belief_b_id: UUID | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "belief_a_id": belief_a_id or uuid4(),
            "belief_b_id": belief_b_id or uuid4(),
            "type": kwargs.get("type", "contradiction"),
            "description": kwargs.get("description"),
            "severity": kwargs.get("severity", "medium"),
            "status": kwargs.get("status", "detected"),
            "resolution": kwargs.get("resolution"),
            "resolved_at": kwargs.get("resolved_at"),
            "detected_at": kwargs.get("detected_at", now),
        }

    return factory


@pytest.fixture
def source_row_factory():
    """Factory for creating source database rows."""

    def factory(id: UUID | None = None, type: str = "conversation", **kwargs) -> dict[str, Any]:
        return {
            "id": id or uuid4(),
            "type": type,
            "title": kwargs.get("title"),
            "url": kwargs.get("url"),
            "content_hash": kwargs.get("content_hash"),
            "session_id": kwargs.get("session_id"),
            "metadata": kwargs.get("metadata", {}),
            "created_at": kwargs.get("created_at", datetime.now()),
        }

    return factory


# ============================================================================
# External Service Mocks
# ============================================================================


@pytest.fixture
def mock_openai():
    """Mock OpenAI client for embedding generation."""
    with patch("valence.embeddings.service.OpenAI") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client

        # Mock embedding response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536  # Default dimension for text-embedding-3-small
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        yield mock_client


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for Claude CLI testing."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "session_id": "test-session-123",
                "result": "Test response from Claude",
            }
        )
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        yield mock_run


# ============================================================================
# MCP Server Mocks
# ============================================================================


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server components."""
    with patch("mcp.server.Server") as mock_server:
        mock_instance = MagicMock()
        mock_server.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_stdio_server():
    """Mock MCP stdio server."""
    with patch("mcp.server.stdio.stdio_server") as mock_stdio:
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
        yield mock_stdio


# ============================================================================
# Helper Functions
# ============================================================================


def make_uuid() -> UUID:
    """Generate a new UUID for tests."""
    return uuid4()


def make_datetime(days_ago: int = 0) -> datetime:
    """Generate a datetime, optionally in the past."""
    return datetime.now() - timedelta(days=days_ago)


# Export helpers
pytest.make_uuid = make_uuid
pytest.make_datetime = make_datetime
