"""Tests for MCP server entry point and protocol implementation.

Tests cover:
1. list_tools — returns SUBSTRATE_TOOLS
2. call_tool — routes to handlers, error handling
3. list_resources — returns available resources
4. read_resource — reads valence:// URIs
5. get_recent_articles — queries recent articles
6. get_stats — collects database statistics
7. run — server entry point with health checks
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from valence.core.exceptions import DatabaseException, ValidationException
from valence.mcp.server import (
    TOOL_HANDLERS,
    call_tool,
    get_recent_articles,
    get_stats,
    list_resources,
    list_tools,
    read_resource,
    run,
)
from valence.mcp.tools import SUBSTRATE_TOOLS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cursor():
    """Mock psycopg2 cursor with dict-like row results."""
    cur = MagicMock()
    cur.fetchone.return_value = None
    cur.fetchall.return_value = []
    return cur


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Patch get_cursor with a sync context manager."""

    @contextmanager
    def _mock(dict_cursor: bool = True) -> Generator:
        yield mock_cursor

    with patch("valence.mcp.server.get_cursor", _mock):
        yield mock_cursor


# ---------------------------------------------------------------------------
# Tests: list_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools_returns_substrate_tools():
    """list_tools should return the SUBSTRATE_TOOLS definition."""
    result = await list_tools()
    assert result == SUBSTRATE_TOOLS
    assert len(result) > 0
    # Verify it's a list of Tool objects
    assert all(hasattr(tool, "name") for tool in result)


# ---------------------------------------------------------------------------
# Tests: call_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_success():
    """call_tool should route to the handler and return TextContent."""
    mock_handler = Mock(return_value={"success": True, "data": "test_result"})

    with patch.dict(TOOL_HANDLERS, {"test_tool": mock_handler}, clear=False):
        result = await call_tool("test_tool", {"arg1": "value1", "arg2": 42})

    assert len(result) == 1
    assert result[0].type == "text"
    response = json.loads(result[0].text)
    assert response["success"] is True
    assert response["data"] == "test_result"

    mock_handler.assert_called_once_with(arg1="value1", arg2=42)


@pytest.mark.asyncio
async def test_call_tool_unknown_tool():
    """call_tool should return error for unknown tool."""
    result = await call_tool("nonexistent_tool", {})

    assert len(result) == 1
    assert result[0].type == "text"
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "Unknown tool" in response["error"]
    assert "nonexistent_tool" in response["error"]


@pytest.mark.asyncio
async def test_call_tool_validation_error():
    """call_tool should catch ValidationException and return structured error."""
    error = ValidationException("Invalid input", field="test_field")
    mock_handler = Mock(side_effect=error)

    with patch.dict(TOOL_HANDLERS, {"test_tool": mock_handler}, clear=False):
        result = await call_tool("test_tool", {})

    assert len(result) == 1
    assert result[0].type == "text"
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "Validation error" in response["error"]
    assert "Invalid input" in response["error"]
    assert "details" in response


@pytest.mark.asyncio
async def test_call_tool_database_error():
    """call_tool should catch DatabaseException and return structured error."""
    error = DatabaseException("Connection failed")
    mock_handler = Mock(side_effect=error)

    with patch.dict(TOOL_HANDLERS, {"test_tool": mock_handler}, clear=False):
        result = await call_tool("test_tool", {})

    assert len(result) == 1
    assert result[0].type == "text"
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "Database error" in response["error"]
    assert "Connection failed" in response["error"]


@pytest.mark.asyncio
async def test_call_tool_unexpected_error():
    """call_tool should catch unexpected exceptions and return internal error."""
    mock_handler = Mock(side_effect=RuntimeError("Something went wrong"))

    with patch.dict(TOOL_HANDLERS, {"test_tool": mock_handler}, clear=False):
        result = await call_tool("test_tool", {})

    assert len(result) == 1
    assert result[0].type == "text"
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "Internal error" in response["error"]
    assert "Something went wrong" in response["error"]


@pytest.mark.asyncio
async def test_call_tool_none_arguments():
    """Regression: call_tool should not crash when arguments is None (issue #561).

    Some MCP library versions pass None instead of {} when a client sends no args.
    """
    mock_handler = Mock(return_value={"success": True})

    with patch.dict(TOOL_HANDLERS, {"test_tool": mock_handler}, clear=False):
        result = await call_tool("test_tool", None)

    # Should not raise; handler called with no args or returns error
    assert len(result) == 1
    assert result[0].type == "text"
    response = json.loads(result[0].text)
    # The mock handler is called with no kwargs since arguments or {} = {}
    assert response["success"] is True
    mock_handler.assert_called_once_with()


# ---------------------------------------------------------------------------
# Tests: list_resources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_resources_returns_expected():
    """list_resources should return recent articles and stats resources."""
    result = await list_resources()

    assert len(result) == 2

    # Check recent articles resource
    recent = next(r for r in result if "recent" in str(r.uri))
    assert "articles/recent" in str(recent.uri)
    assert recent.name == "Recent Articles"
    assert recent.mimeType == "application/json"

    # Check stats resource
    stats = next(r for r in result if "stats" in str(r.uri))
    assert "stats" in str(stats.uri)
    assert stats.name == "Database Statistics"
    assert stats.mimeType == "application/json"


# ---------------------------------------------------------------------------
# Tests: read_resource
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_resource_recent_articles(mock_get_cursor):
    """read_resource should call get_recent_articles for valence://articles/recent."""
    # Mock article data
    article_id = uuid4()
    mock_get_cursor.fetchall.return_value = [
        {
            "id": article_id,
            "title": "Test Article",
            "content": "Test content",
            "status": "active",
            "version": 1,
            "confidence": 0.8,
            "domain_path": ["test"],
            "modified_at": datetime(2026, 2, 23, 12, 0, 0),
            "created_at": datetime(2026, 2, 23, 10, 0, 0),
            "source_count": 3,
        }
    ]

    result = await read_resource("valence://articles/recent")

    assert len(result) == 1
    assert str(result[0].uri) == "valence://articles/recent"
    assert result[0].mimeType == "application/json"

    data = json.loads(result[0].text)
    assert "articles" in data
    assert len(data["articles"]) == 1
    assert data["articles"][0]["title"] == "Test Article"
    assert data["articles"][0]["id"] == str(article_id)
    assert "as_of" in data


@pytest.mark.asyncio
async def test_read_resource_stats(mock_get_cursor):
    """read_resource should call get_stats for valence://stats."""
    # Mock stats data
    mock_get_cursor.fetchall.side_effect = [
        # domains query
        [{"domain": "test_domain", "count": 42}],
        # confidence distribution query
        [{"confidence_level": "high", "count": 10}],
        # entity types query
        [{"type": "person", "count": 5}],
    ]

    with patch("valence.mcp.server.DatabaseStats") as mock_stats_class:
        mock_stats = Mock()
        mock_stats.to_dict.return_value = {
            "sources": 100,
            "articles": 50,
            "entities": 25,
        }
        mock_stats_class.collect.return_value = mock_stats

        result = await read_resource("valence://stats")

    assert len(result) == 1
    assert str(result[0].uri) == "valence://stats"
    assert result[0].mimeType == "application/json"

    data = json.loads(result[0].text)
    assert "totals" in data
    assert data["totals"]["sources"] == 100
    assert "domains" in data
    assert data["domains"]["test_domain"] == 42
    assert "confidence_distribution" in data
    assert "entity_types" in data
    assert "as_of" in data


@pytest.mark.asyncio
async def test_read_resource_unknown_uri():
    """read_resource should return error for unknown URI."""
    result = await read_resource("valence://unknown")

    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "error" in data
    assert "Unknown resource" in data["error"]


# ---------------------------------------------------------------------------
# Tests: get_recent_articles
# ---------------------------------------------------------------------------


def test_get_recent_articles_returns_articles(mock_get_cursor):
    """get_recent_articles should query and format recent articles."""
    article_id = uuid4()
    mock_get_cursor.fetchall.return_value = [
        {
            "id": article_id,
            "title": "Article 1",
            "content": "Content 1",
            "status": "active",
            "version": 1,
            "confidence": 0.9,
            "domain_path": ["tech", "ai"],
            "modified_at": datetime(2026, 2, 23, 12, 0, 0),
            "created_at": datetime(2026, 2, 23, 10, 0, 0),
            "source_count": 5,
        },
        {
            "id": uuid4(),
            "title": "Article 2",
            "content": "Content 2",
            "status": "active",
            "version": 2,
            "confidence": 0.7,
            "domain_path": ["science"],
            "modified_at": datetime(2026, 2, 23, 11, 0, 0),
            "created_at": datetime(2026, 2, 23, 9, 0, 0),
            "source_count": 2,
        },
    ]

    result = get_recent_articles(limit=20)

    assert result["count"] == 2
    assert len(result["articles"]) == 2
    assert result["articles"][0]["title"] == "Article 1"
    assert result["articles"][0]["id"] == str(article_id)
    assert "2026-02-23T12:00:00" in result["articles"][0]["modified_at"]
    assert "as_of" in result

    # Verify SQL query
    mock_get_cursor.execute.assert_called_once()
    sql_call = mock_get_cursor.execute.call_args[0][0]
    assert "SELECT a.id" in sql_call
    assert "ORDER BY a.modified_at DESC" in sql_call
    assert "LIMIT %s" in sql_call


def test_get_recent_articles_custom_limit(mock_get_cursor):
    """get_recent_articles should respect custom limit parameter."""
    mock_get_cursor.fetchall.return_value = []

    get_recent_articles(limit=10)

    # Check the limit parameter was passed
    call_args = mock_get_cursor.execute.call_args[0]
    assert call_args[1] == (10,)


def test_get_recent_articles_none_timestamps(mock_get_cursor):
    """get_recent_articles should handle None created_at/modified_at."""
    mock_get_cursor.fetchall.return_value = [
        {
            "id": uuid4(),
            "title": "Article",
            "content": "Content",
            "status": "active",
            "version": 1,
            "confidence": 0.8,
            "domain_path": ["test"],
            "modified_at": None,
            "created_at": None,
            "source_count": 0,
        }
    ]

    result = get_recent_articles()

    assert result["count"] == 1
    # Should not raise error, timestamps should be None
    assert result["articles"][0]["created_at"] is None
    assert result["articles"][0]["modified_at"] is None


# ---------------------------------------------------------------------------
# Tests: get_stats
# ---------------------------------------------------------------------------


def test_get_stats_returns_comprehensive_data(mock_get_cursor):
    """get_stats should collect and return comprehensive statistics."""
    # Mock database queries
    mock_get_cursor.fetchall.side_effect = [
        # domains query
        [
            {"domain": "tech", "count": 50},
            {"domain": "science", "count": 30},
        ],
        # confidence distribution query
        [
            {"confidence_level": "high", "count": 40},
            {"confidence_level": "moderate", "count": 30},
        ],
        # entity types query
        [
            {"type": "person", "count": 100},
            {"type": "organization", "count": 50},
        ],
    ]

    with patch("valence.mcp.server.DatabaseStats") as mock_stats_class:
        mock_stats = Mock()
        mock_stats.to_dict.return_value = {
            "sources": 500,
            "articles": 100,
            "entities": 200,
        }
        mock_stats_class.collect.return_value = mock_stats

        result = get_stats()

    assert result["totals"]["sources"] == 500
    assert result["totals"]["articles"] == 100

    assert result["domains"]["tech"] == 50
    assert result["domains"]["science"] == 30

    assert result["confidence_distribution"]["high"] == 40
    assert result["confidence_distribution"]["moderate"] == 30

    assert result["entity_types"]["person"] == 100
    assert result["entity_types"]["organization"] == 50

    assert "as_of" in result


def test_get_stats_handles_query_failures(mock_get_cursor):
    """get_stats should handle query failures gracefully."""
    # Make all queries raise exceptions
    mock_get_cursor.fetchall.side_effect = Exception("Query failed")

    with patch("valence.mcp.server.DatabaseStats") as mock_stats_class:
        mock_stats = Mock()
        mock_stats.to_dict.return_value = {"sources": 0}
        mock_stats_class.collect.return_value = mock_stats

        result = get_stats()

    # Should still have totals from DatabaseStats
    assert "totals" in result

    # Failed queries should return empty dicts
    assert result["domains"] == {}
    assert result["confidence_distribution"] == {}
    assert result["entity_types"] == {}


def test_get_stats_partial_failures(mock_get_cursor):
    """get_stats should handle partial failures (some queries succeed, some fail)."""
    # First query succeeds, rest fail
    mock_get_cursor.fetchall.side_effect = [
        [{"domain": "tech", "count": 10}],
        Exception("Confidence query failed"),
        Exception("Entity query failed"),
    ]

    with patch("valence.mcp.server.DatabaseStats") as mock_stats_class:
        mock_stats = Mock()
        mock_stats.to_dict.return_value = {"sources": 100}
        mock_stats_class.collect.return_value = mock_stats

        result = get_stats()

    assert result["domains"]["tech"] == 10
    assert result["confidence_distribution"] == {}
    assert result["entity_types"] == {}


# ---------------------------------------------------------------------------
# Tests: run (server entry point)
# ---------------------------------------------------------------------------


def test_run_health_check_mode():
    """run should execute health check and exit when --health-check is passed."""
    test_args = ["server.py", "--health-check"]

    with patch("valence.mcp.server.cli_health_check", return_value=0) as mock_health:
        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                run()

    assert exc_info.value.code == 0
    mock_health.assert_called_once()


def test_run_health_check_failure():
    """run should exit with error code when health check fails."""
    test_args = ["server.py", "--health-check"]

    with patch("valence.mcp.server.cli_health_check", return_value=1):
        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                run()

    assert exc_info.value.code == 1


def test_run_startup_checks_enabled():
    """run should execute startup checks by default."""
    test_args = ["server.py"]

    mock_cursor = MagicMock()

    @contextmanager
    def mock_get_cursor_ctx(dict_cursor=True):
        yield mock_cursor

    with patch("valence.mcp.server.startup_checks") as mock_startup:
        with patch("valence.mcp.server.init_schema"):
            with patch("valence.mcp.server.get_cursor", mock_get_cursor_ctx):
                with patch("valence.core.maintenance.check_and_run_maintenance", return_value=None):
                    with patch("valence.mcp.server.asyncio.run"):
                        with patch("sys.argv", test_args):
                            run()

    mock_startup.assert_called_once_with(fail_fast=True)


def test_run_skip_health_check():
    """run should skip startup checks when --skip-health-check is passed."""
    test_args = ["server.py", "--skip-health-check"]

    mock_cursor = MagicMock()

    @contextmanager
    def mock_get_cursor_ctx(dict_cursor=True):
        yield mock_cursor

    with patch("valence.mcp.server.startup_checks") as mock_startup:
        with patch("valence.mcp.server.init_schema"):
            with patch("valence.mcp.server.get_cursor", mock_get_cursor_ctx):
                with patch("valence.core.maintenance.check_and_run_maintenance", return_value=None):
                    with patch("valence.mcp.server.asyncio.run"):
                        with patch("sys.argv", test_args):
                            run()

    mock_startup.assert_not_called()


def test_run_initializes_schema():
    """run should initialize database schema."""
    test_args = ["server.py"]

    mock_cursor = MagicMock()

    @contextmanager
    def mock_get_cursor_ctx(dict_cursor=True):
        yield mock_cursor

    with patch("valence.mcp.server.startup_checks"):
        with patch("valence.mcp.server.init_schema") as mock_init:
            with patch("valence.mcp.server.get_cursor", mock_get_cursor_ctx):
                with patch("valence.core.maintenance.check_and_run_maintenance", return_value=None):
                    with patch("valence.mcp.server.asyncio.run"):
                        with patch("sys.argv", test_args):
                            run()

    mock_init.assert_called_once()
    # Verify schema_dir is the parent of server.py
    schema_dir = mock_init.call_args[0][0]
    assert "valence/mcp" in schema_dir or "valence" in schema_dir


def test_run_schema_init_failure_continues():
    """run should log warning and continue if schema init fails."""
    test_args = ["server.py"]

    mock_cursor = MagicMock()

    @contextmanager
    def mock_get_cursor_ctx(dict_cursor=True):
        yield mock_cursor

    with patch("valence.mcp.server.startup_checks"):
        with patch("valence.mcp.server.init_schema", side_effect=DatabaseException("Already exists")):
            with patch("valence.mcp.server.get_cursor", mock_get_cursor_ctx):
                # Mock maintenance to avoid side effects
                with patch("valence.core.maintenance.check_and_run_maintenance", return_value=None):
                    with patch("valence.mcp.server.asyncio.run"):
                        with patch("valence.mcp.server.logger") as mock_logger:
                            with patch("sys.argv", test_args):
                                run()

    # Should log warning but not raise
    mock_logger.warning.assert_called()
    # Check that at least one warning contains schema initialization
    warnings = [args[0][0] for args in mock_logger.warning.call_args_list]
    assert any("Schema initialization skipped" in w for w in warnings)


def test_run_checks_scheduled_maintenance():
    """run should check and run scheduled maintenance if needed."""
    test_args = ["server.py"]

    mock_cursor = MagicMock()

    @contextmanager
    def mock_get_cursor_ctx(dict_cursor=True):
        yield mock_cursor

    with patch("valence.mcp.server.startup_checks"):
        with patch("valence.mcp.server.init_schema"):
            with patch("valence.mcp.server.get_cursor", mock_get_cursor_ctx):
                with patch("valence.core.maintenance.check_and_run_maintenance") as mock_maint:
                    mock_maint.return_value = {"timestamp": "2026-02-23T12:00:00"}
                    with patch("valence.mcp.server.asyncio.run"):
                        with patch("sys.argv", test_args):
                            run()

    mock_maint.assert_called_once_with(mock_cursor)


def test_run_maintenance_failure_continues():
    """run should log warning and continue if maintenance check fails."""
    test_args = ["server.py"]

    mock_cursor = MagicMock()

    @contextmanager
    def mock_get_cursor_ctx(dict_cursor=True):
        yield mock_cursor

    with patch("valence.mcp.server.startup_checks"):
        with patch("valence.mcp.server.init_schema"):
            with patch("valence.mcp.server.get_cursor", mock_get_cursor_ctx):
                with patch("valence.core.maintenance.check_and_run_maintenance", side_effect=Exception("Failed")):
                    with patch("valence.mcp.server.logger") as mock_logger:
                        with patch("valence.mcp.server.asyncio.run"):
                            with patch("sys.argv", test_args):
                                run()

    # Should log warning but not raise
    mock_logger.warning.assert_called()
    warning_msg = mock_logger.warning.call_args[0][0]
    assert "Scheduled maintenance check failed" in warning_msg


def test_run_starts_asyncio_server():
    """run should start asyncio server with stdio."""
    test_args = ["server.py"]

    mock_cursor = MagicMock()

    @contextmanager
    def mock_get_cursor_ctx(dict_cursor=True):
        yield mock_cursor

    with patch("valence.mcp.server.startup_checks"):
        with patch("valence.mcp.server.init_schema"):
            with patch("valence.mcp.server.get_cursor", mock_get_cursor_ctx):
                with patch("valence.core.maintenance.check_and_run_maintenance", return_value=None):
                    with patch("valence.mcp.server.asyncio.run") as mock_asyncio:
                        with patch("sys.argv", test_args):
                            run()

    # Should call asyncio.run with the main coroutine
    mock_asyncio.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: TOOL_HANDLERS registry
# ---------------------------------------------------------------------------


def test_tool_handlers_registry_complete():
    """TOOL_HANDLERS should include all expected tools."""
    expected_tools = {
        # Source tools
        "source_ingest",
        "source_get",
        "source_search",
        "source_list",
        # Retrieval
        "knowledge_search",
        # Article tools
        "article_create",
        "article_get",
        "article_update",
        "article_search",
        "article_compile",
        "article_split",
        "article_merge",
        # Provenance tools
        "provenance_link",
        "provenance_get",
        "provenance_trace",
        # Entity tools
        # Contention tools
        "contention_detect",
        "contention_list",
        "contention_resolve",
        # Admin tools
        "admin_forget",
        "admin_stats",
        "admin_maintenance",
        # Memory tools
        "memory_store",
        "memory_recall",
        "memory_status",
        "memory_forget",
        # Session tools
        "session_start",
        "session_append",
        "session_flush",
        "session_finalize",
        "session_search",
        "session_list",
        "session_get",
        "session_compile",
        "session_flush_stale",
    }

    assert set(TOOL_HANDLERS.keys()) == expected_tools


def test_tool_handlers_are_callable():
    """All handlers in TOOL_HANDLERS should be callable."""
    for name, handler in TOOL_HANDLERS.items():
        assert callable(handler), f"Handler {name} is not callable"
