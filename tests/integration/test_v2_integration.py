"""Integration tests for WU-11: MCP + REST + CLI integration.

Tests:
    1. MCP tool list returns exactly the expected 16 v2 tools.
    2. Each tool is callable via the handler dispatch (mock DB).
    3. REST endpoints are mountable (Starlette routing smoke test).
    4. CLI modules register without error.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# =============================================================================
# Constants
# =============================================================================

EXPECTED_TOOLS = {
    "source_ingest",
    "source_get",
    "source_search",
    "knowledge_search",
    "article_get",
    "article_create",
    "article_compile",
    "article_update",
    "article_split",
    "article_merge",
    "provenance_trace",
    "contention_list",
    "contention_resolve",
    "admin_forget",
    "admin_stats",
    "admin_maintenance",
}


# =============================================================================
# MCP Tool List Tests
# =============================================================================


class TestMCPToolList:
    """Verify the tool list matches the WU-11 spec exactly."""

    def test_substrate_tools_count(self):
        """SUBSTRATE_TOOLS should have exactly 16 entries."""
        from valence.substrate.tools.definitions import SUBSTRATE_TOOLS

        assert len(SUBSTRATE_TOOLS) == 16, f"Expected 16 tools, got {len(SUBSTRATE_TOOLS)}. Tools: {[t.name for t in SUBSTRATE_TOOLS]}"

    def test_substrate_tools_names_exact(self):
        """SUBSTRATE_TOOLS names must exactly match EXPECTED_TOOLS."""
        from valence.substrate.tools.definitions import SUBSTRATE_TOOLS

        actual = {t.name for t in SUBSTRATE_TOOLS}
        assert actual == EXPECTED_TOOLS, f"Extra tools: {actual - EXPECTED_TOOLS}\nMissing tools: {EXPECTED_TOOLS - actual}"

    def test_all_tools_have_descriptions(self):
        """Every tool must have a non-empty description."""
        from valence.substrate.tools.definitions import SUBSTRATE_TOOLS

        for tool in SUBSTRATE_TOOLS:
            assert tool.description and tool.description.strip(), f"Tool '{tool.name}' has no description"

    def test_all_tools_have_input_schema(self):
        """Every tool must have an inputSchema dict."""
        from valence.substrate.tools.definitions import SUBSTRATE_TOOLS

        for tool in SUBSTRATE_TOOLS:
            assert isinstance(tool.inputSchema, dict), f"Tool '{tool.name}' has no inputSchema"
            assert tool.inputSchema.get("type") == "object", f"Tool '{tool.name}' inputSchema type is not 'object'"

    def test_handlers_registered_for_all_tools(self):
        """Every spec tool should be registered in SUBSTRATE_HANDLERS."""
        from valence.substrate.tools.handlers import SUBSTRATE_HANDLERS

        for tool_name in EXPECTED_TOOLS:
            assert tool_name in SUBSTRATE_HANDLERS, f"Tool '{tool_name}' missing from SUBSTRATE_HANDLERS"


# =============================================================================
# Handler Dispatch Tests (mock DB)
# =============================================================================


class TestHandlerDispatch:
    """Verify each tool can be dispatched without raising import errors.

    These tests use mock DB cursors so they do not require a real database.
    They verify the dispatch plumbing, not business logic.
    """

    def test_unknown_tool_returns_error(self):
        """Dispatching an unknown tool returns an error dict."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        result = handle_substrate_tool("nonexistent_tool", {})
        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_source_ingest_missing_content(self):
        """source_ingest with empty content returns validation error (no DB needed)."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        result = handle_substrate_tool("source_ingest", {"content": "", "source_type": "document"})
        assert result["success"] is False

    def test_source_ingest_invalid_type(self):
        """source_ingest with unknown source_type returns validation error."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        result = handle_substrate_tool(
            "source_ingest",
            {"content": "some content", "source_type": "invalid_type"},
        )
        assert result["success"] is False
        assert "source_type" in result["error"].lower() or "Invalid" in result["error"]

    def test_source_search_empty_query(self):
        """source_search with empty query returns empty results without DB error."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        # Mock the DB cursor so no real connection is needed
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        with patch("valence.substrate.tools.sources.get_cursor", return_value=mock_cursor):
            result = handle_substrate_tool("source_search", {"query": ""})
        # Empty query returns empty results immediately (short-circuits before DB)
        assert result["success"] is True
        assert result["sources"] == []

    def test_knowledge_search_empty_query(self):
        """knowledge_search with empty query returns validation error."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        result = handle_substrate_tool("knowledge_search", {"query": ""})
        assert result["success"] is False

    def test_article_create_missing_content(self):
        """article_create with no content returns error (validated before DB access)."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        # core.articles.create_article validates content before DB access
        result = handle_substrate_tool("article_create", {"content": ""})
        assert result["success"] is False

    def test_article_compile_empty_sources(self):
        """article_compile with empty source_ids returns validation error."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        result = handle_substrate_tool("article_compile", {"source_ids": []})
        assert result["success"] is False

    def test_article_split_missing_id(self):
        """article_split with empty article_id returns validation error."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        result = handle_substrate_tool("article_split", {"article_id": ""})
        assert result["success"] is False

    def test_article_merge_missing_ids(self):
        """article_merge with missing IDs returns validation error."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        result = handle_substrate_tool("article_merge", {"article_id_a": "", "article_id_b": ""})
        assert result["success"] is False

    def test_admin_forget_invalid_type(self):
        """admin_forget with bad target_type returns validation error."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        result = handle_substrate_tool("admin_forget", {"target_type": "belief", "target_id": "some-uuid"})
        assert result["success"] is False
        assert "target_type" in result["error"].lower() or "source" in result["error"]

    def test_admin_forget_missing_id(self):
        """admin_forget with empty target_id returns validation error."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        result = handle_substrate_tool("admin_forget", {"target_type": "source", "target_id": ""})
        assert result["success"] is False

    def test_admin_maintenance_no_operations(self):
        """admin_maintenance with no operations selected returns error."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        result = handle_substrate_tool(
            "admin_maintenance",
            {
                "recompute_scores": False,
                "process_queue": False,
                "evict_if_over_capacity": False,
            },
        )
        assert result["success"] is False

    def test_contention_list_callable(self):
        """contention_list handler exists and is callable."""
        from valence.substrate.tools.handlers import SUBSTRATE_HANDLERS

        assert "contention_list" in SUBSTRATE_HANDLERS
        handler = SUBSTRATE_HANDLERS["contention_list"]
        assert callable(handler)

    def test_contention_resolve_missing_id(self):
        """contention_resolve with empty contention_id returns error."""
        from valence.substrate.tools.handlers import handle_substrate_tool

        result = handle_substrate_tool(
            "contention_resolve",
            {"contention_id": "", "resolution": "dismiss", "rationale": "test"},
        )
        assert result["success"] is False

    def test_provenance_trace_callable(self):
        """provenance_trace handler is registered and callable."""
        from valence.substrate.tools.handlers import SUBSTRATE_HANDLERS

        assert "provenance_trace" in SUBSTRATE_HANDLERS
        assert callable(SUBSTRATE_HANDLERS["provenance_trace"])


# =============================================================================
# REST App Mountability Tests
# =============================================================================


class TestRESTAppMountability:
    """Verify the Starlette app can be created and routes are present."""

    def test_app_creates_without_error(self):
        """create_app() should succeed without raising exceptions."""
        from valence.server.app import create_app

        app = create_app()
        assert app is not None

    def test_v2_sources_routes_present(self):
        """The app should have source routes mounted."""
        from valence.server.app import create_app

        app = create_app()
        route_paths = [str(r.path) for r in app.routes]

        assert any("/sources" in p for p in route_paths), f"No /sources routes found. Paths: {route_paths}"

    def test_v2_articles_routes_present(self):
        """The app should have article routes mounted."""
        from valence.server.app import create_app

        app = create_app()
        route_paths = [str(r.path) for r in app.routes]

        assert any("/articles" in p for p in route_paths), f"No /articles routes found. Paths: {route_paths}"

    def test_v2_provenance_routes_present(self):
        """The app should have provenance routes mounted."""
        from valence.server.app import create_app

        app = create_app()
        route_paths = [str(r.path) for r in app.routes]

        assert any("provenance" in p for p in route_paths), f"No provenance routes found. Paths: {route_paths}"

    def test_mcp_route_present(self):
        """The MCP endpoint route should exist."""
        from valence.server.app import create_app

        app = create_app()
        route_paths = [str(r.path) for r in app.routes]

        assert any("/mcp" in p for p in route_paths), f"No /mcp route found. Paths: {route_paths}"


# =============================================================================
# CLI Registration Tests
# =============================================================================


class TestCLIRegistration:
    """Verify CLI commands register without error."""

    def test_cli_app_builds_without_error(self):
        """app() should build the argparse parser without raising."""
        from valence.cli.main import app

        parser = app()
        assert parser is not None

    def test_sources_command_registered(self):
        """'sources' subcommand should be registered."""
        from valence.cli.main import app

        parser = app()
        # Verify by trying to parse a sources subcommand
        # (Would raise SystemExit if not registered)
        import argparse

        subparsers_action = None
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                subparsers_action = action
                break

        assert subparsers_action is not None
        assert "sources" in subparsers_action.choices, f"'sources' not registered. Available: {list(subparsers_action.choices.keys())}"

    def test_articles_command_registered(self):
        """'articles' subcommand should be registered."""
        import argparse

        from valence.cli.main import app

        parser = app()
        subparsers_action = next(
            (a for a in parser._actions if isinstance(a, argparse._SubParsersAction)),
            None,
        )
        assert subparsers_action is not None
        assert "articles" in subparsers_action.choices

    def test_provenance_command_registered(self):
        """'provenance' subcommand should be registered."""
        import argparse

        from valence.cli.main import app

        parser = app()
        subparsers_action = next(
            (a for a in parser._actions if isinstance(a, argparse._SubParsersAction)),
            None,
        )
        assert subparsers_action is not None
        assert "provenance" in subparsers_action.choices

    def test_sources_list_subcommand(self):
        """'sources list' should parse without error."""
        from valence.cli.main import app

        parser = app()
        args = parser.parse_args(["sources", "list"])
        assert hasattr(args, "func")
        assert callable(args.func)

    def test_articles_search_subcommand(self):
        """'articles search query' should parse without error."""
        from valence.cli.main import app

        parser = app()
        args = parser.parse_args(["articles", "search", "python best practices"])
        assert args.query == "python best practices"
        assert hasattr(args, "func")

    def test_provenance_trace_subcommand(self):
        """'provenance trace <id> <claim>' should parse without error."""
        from valence.cli.main import app

        parser = app()
        args = parser.parse_args(["provenance", "trace", "some-uuid", "Python is fast"])
        assert args.article_id == "some-uuid"
        assert args.claim_text == "Python is fast"
        assert hasattr(args, "func")


# =============================================================================
# Module Import Tests
# =============================================================================


class TestImports:
    """Verify all v2 tool modules import cleanly."""

    def test_definitions_imports(self):
        """definitions.py should import without error."""
        from valence.substrate.tools import definitions  # noqa: F401

    def test_handlers_imports(self):
        """handlers.py should import without error."""
        from valence.substrate.tools import handlers  # noqa: F401

    def test_sources_tool_imports(self):
        """sources tool module should import without error."""
        from valence.substrate.tools import sources  # noqa: F401

    def test_articles_tool_imports(self):
        """articles tool module should import without error."""
        from valence.substrate.tools import articles  # noqa: F401

    def test_retrieval_tool_imports(self):
        """retrieval tool module should import without error."""
        from valence.substrate.tools import retrieval  # noqa: F401

    def test_contention_tool_imports(self):
        """contention tool module should import without error."""
        from valence.substrate.tools import contention  # noqa: F401

    def test_admin_tool_imports(self):
        """admin tool module should import without error."""
        from valence.substrate.tools import admin  # noqa: F401

    def test_mcp_server_imports(self):
        """mcp_server.py should import without error."""
        from valence.substrate import mcp_server  # noqa: F401

    def test_server_app_imports(self):
        """server/app.py should import without error."""
        from valence.server import app  # noqa: F401

    def test_cli_main_imports(self):
        """cli/main.py should import without error."""
        from valence.cli import main  # noqa: F401

    def test_cli_sources_command_imports(self):
        """cli sources command should import without error."""
        from valence.cli.commands import sources  # noqa: F401

    def test_cli_articles_command_imports(self):
        """cli articles command should import without error."""
        from valence.cli.commands import articles  # noqa: F401

    def test_cli_provenance_command_imports(self):
        """cli provenance command should import without error."""
        from valence.cli.commands import provenance  # noqa: F401
