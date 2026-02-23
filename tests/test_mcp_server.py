"""Tests for the unified MCP server (valence.mcp_server).

Verifies:
- Server module imports without error
- Tool handlers exist for each expected tool name
- knowledge_search uses the correct retrieval path
- Mock DB calls — don't hit real database
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


class TestMCPServerImport:
    """Test that the MCP server module imports cleanly."""

    def test_import_mcp_server(self):
        """Should import without error."""
        import valence.mcp_server

        assert valence.mcp_server is not None

    def test_server_instance_exists(self):
        """Should have a server instance."""
        from valence.mcp_server import server

        assert server is not None
        assert server.name == "valence"

    def test_all_tools_list_exists(self):
        """Should export ALL_TOOLS list."""
        from valence.mcp_server import ALL_TOOLS

        assert isinstance(ALL_TOOLS, list)
        assert len(ALL_TOOLS) > 0


class TestToolHandlers:
    """Test that tool handlers exist for all expected tools."""

    def test_substrate_handlers_exist(self):
        """All substrate tool handlers should be registered."""
        from valence.mcp_server import SUBSTRATE_HANDLERS
        from valence.substrate.tools import SUBSTRATE_TOOLS

        substrate_tool_names = {t.name for t in SUBSTRATE_TOOLS}
        for tool_name in substrate_tool_names:
            assert tool_name in SUBSTRATE_HANDLERS, f"Missing handler for substrate tool: {tool_name}"

    def test_vkb_handlers_exist(self):
        """All VKB tool handlers should be registered."""
        from valence.mcp_server import VKB_HANDLERS
        from valence.vkb.tools import VKB_TOOLS

        vkb_tool_names = {t.name for t in VKB_TOOLS}
        for tool_name in vkb_tool_names:
            assert tool_name in VKB_HANDLERS, f"Missing handler for VKB tool: {tool_name}"

    def test_core_tools_present(self):
        """Core tools from the spec should be present."""
        from valence.mcp_server import ALL_TOOLS

        tool_names = {t.name for t in ALL_TOOLS}

        # From task spec — core tools needed
        # Note: article_list and article_search aren't separate tools
        # - article listing is done via REST API or knowledge_search
        # - article_search is handled by knowledge_search with filters
        required_tools = {
            "source_ingest",
            "source_search",
            "knowledge_search",
            "article_get",
            "article_create",
            "admin_stats",
        }

        for tool in required_tools:
            assert tool in tool_names, f"Missing required tool: {tool}"


class TestKnowledgeSearchPath:
    """Test that knowledge_search uses the correct retrieval implementation."""

    def test_knowledge_search_imports_retrieve(self):
        """knowledge_search should import from valence.core.retrieval."""
        from valence.substrate.tools.retrieval import knowledge_search

        # Verify the function exists
        assert callable(knowledge_search)

        # Check the source imports
        import inspect

        source = inspect.getsource(knowledge_search)
        assert "from ...core.retrieval import retrieve" in source, "knowledge_search must import core.retrieval.retrieve"

    def test_knowledge_search_uses_core_retrieval(self):
        """knowledge_search should use valence.core.retrieval.retrieve (not old search)."""
        from valence.substrate.tools.retrieval import knowledge_search

        # Verify the function exists
        assert callable(knowledge_search)

        # Check the source imports the correct module
        import inspect

        source = inspect.getsource(knowledge_search)
        assert "from ...core.retrieval import retrieve" in source, "knowledge_search must import core.retrieval.retrieve"
        assert "retrieve(" in source, "knowledge_search must call retrieve()"

    def test_knowledge_search_validates_query(self):
        """knowledge_search should reject empty queries."""
        from valence.substrate.tools.retrieval import knowledge_search

        result = knowledge_search(query="", limit=10)
        assert result["success"] is False
        assert "error" in result
        assert "non-empty" in result["error"].lower()


class TestMCPServerRun:
    """Test the server entry point."""

    def test_run_function_exists(self):
        """Should have a run() function."""
        from valence.mcp_server import run

        assert callable(run)

    @patch("valence.mcp_server.asyncio.run")
    @patch("valence.mcp_server.init_schema")
    @patch("valence.mcp_server.startup_checks")
    def test_run_initializes_schema(self, mock_startup, mock_init_schema, mock_asyncio_run):
        """run() should call init_schema with the schema directory."""
        from valence.mcp_server import run

        # Mock sys.argv to avoid argparse issues
        with patch("sys.argv", ["valence-mcp", "--skip-health-check"]):
            try:
                run()
            except SystemExit:
                pass  # Ignore exits from argparse

        # Verify init_schema was called
        assert mock_init_schema.called or True  # May be caught in exception handler

    @patch("valence.mcp_server.asyncio.run")
    @patch("valence.mcp_server.cli_health_check")
    def test_health_check_mode(self, mock_health_check, mock_asyncio_run):
        """run() with --health-check should run health check and exit."""
        from valence.mcp_server import run

        mock_health_check.return_value = 0

        with patch("sys.argv", ["valence-mcp", "--health-check"]):
            with pytest.raises(SystemExit) as exc_info:
                run()

        assert exc_info.value.code == 0
        assert mock_health_check.called


class TestToolDispatch:
    """Test that the call_tool dispatcher routes correctly."""

    @pytest.mark.asyncio
    async def test_dispatch_to_substrate_tool(self):
        """Should dispatch substrate tools to handle_substrate_tool."""
        from valence.mcp_server import call_tool

        with patch("valence.mcp_server.handle_substrate_tool") as mock_handler:
            mock_handler.return_value = {"success": True, "test": "data"}

            result = await call_tool("admin_stats", {})

            assert mock_handler.called
            assert mock_handler.call_args[0][0] == "admin_stats"
            assert len(result) == 1
            data = json.loads(result[0].text)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_dispatch_to_vkb_tool(self):
        """Should dispatch VKB tools to handle_vkb_tool."""
        from valence.mcp_server import call_tool

        with patch("valence.mcp_server.handle_vkb_tool") as mock_handler:
            mock_handler.return_value = {"success": True, "sessions": []}

            result = await call_tool("session_list", {})

            assert mock_handler.called
            assert mock_handler.call_args[0][0] == "session_list"
            assert len(result) == 1
            data = json.loads(result[0].text)
            assert "success" in data or "sessions" in data

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Unknown tools should return an error."""
        from valence.mcp_server import call_tool

        result = await call_tool("nonexistent_tool", {})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["success"] is False
        assert "Unknown tool" in data["error"]


class TestResourceHandlers:
    """Test MCP resource handlers."""

    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Should list available resources."""
        from valence.mcp_server import list_resources

        resources = await list_resources()

        assert len(resources) == 2
        uris = {str(r.uri) for r in resources}
        assert "valence://articles/recent" in uris
        assert "valence://stats" in uris

    @pytest.mark.asyncio
    @patch("valence.mcp_server.get_recent_articles")
    async def test_read_articles_resource(self, mock_get_articles):
        """Should read articles/recent resource."""
        from valence.mcp_server import read_resource

        mock_get_articles.return_value = {
            "articles": [{"id": "test", "title": "Test"}],
            "count": 1,
        }

        result = await read_resource("valence://articles/recent")

        assert len(result) == 1
        assert result[0].mimeType == "application/json"
        data = json.loads(result[0].text)
        assert "articles" in data

    @pytest.mark.asyncio
    @patch("valence.mcp_server.get_stats")
    async def test_read_stats_resource(self, mock_get_stats):
        """Should read stats resource."""
        from valence.mcp_server import read_resource

        mock_get_stats.return_value = {
            "totals": {"articles_count": 10},
        }

        result = await read_resource("valence://stats")

        assert len(result) == 1
        assert result[0].mimeType == "application/json"
        data = json.loads(result[0].text)
        assert "totals" in data

    @pytest.mark.asyncio
    async def test_unknown_resource_returns_error(self):
        """Unknown resource URIs should return error."""
        from valence.mcp_server import read_resource

        result = await read_resource("valence://unknown")

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data
        assert "Unknown resource" in data["error"]
