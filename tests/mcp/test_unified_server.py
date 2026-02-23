"""Tests for the unified MCP server — updated for v2 (WU-11).

The unified server combines v2 substrate tools (16) and VKB tools.
Resources are now valence://articles/recent and valence://stats.
"""

from __future__ import annotations

import json

import pytest

from valence.mcp_server import ALL_TOOLS, call_tool, list_tools, server
from valence.substrate.tools import SUBSTRATE_TOOLS
from valence.vkb.tools import VKB_HANDLERS

# ============================================================================
# Expected v2 tool names
# ============================================================================

EXPECTED_SUBSTRATE_TOOLS = {
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


class TestToolList:
    """Tests for tool listing."""

    def test_substrate_tools_count(self):
        """SUBSTRATE_TOOLS should have exactly 16 v2 entries."""
        assert len(SUBSTRATE_TOOLS) == 16, f"Expected 16 substrate tools, got {len(SUBSTRATE_TOOLS)}: {[t.name for t in SUBSTRATE_TOOLS]}"

    def test_substrate_tools_names(self):
        """SUBSTRATE_TOOLS names should match the v2 spec."""
        actual = {t.name for t in SUBSTRATE_TOOLS}
        assert actual == EXPECTED_SUBSTRATE_TOOLS, f"Extra: {actual - EXPECTED_SUBSTRATE_TOOLS}\nMissing: {EXPECTED_SUBSTRATE_TOOLS - actual}"

    def test_all_tools_contains_expected_count(self):
        """ALL_TOOLS should contain substrate (16) + VKB tools."""
        expected = len(SUBSTRATE_TOOLS) + len(VKB_HANDLERS)
        assert len(ALL_TOOLS) == expected, f"Expected {expected} tools, got {len(ALL_TOOLS)}"

    def test_no_duplicate_tool_names(self):
        """Tool names must be unique across both servers."""
        names = [t.name for t in ALL_TOOLS]
        duplicates = [n for n in names if names.count(n) > 1]
        assert not duplicates, f"Duplicate tools: {duplicates}"

    def test_substrate_tools_present(self):
        """All v2 substrate tool names should be in the combined list."""
        tool_names = {t.name for t in ALL_TOOLS}
        for name in EXPECTED_SUBSTRATE_TOOLS:
            assert name in tool_names, f"Missing substrate tool: {name}"

    def test_vkb_tools_present(self):
        """All VKB tool names should be in the combined list."""
        tool_names = {t.name for t in ALL_TOOLS}
        for name in VKB_HANDLERS:
            assert name in tool_names, f"Missing VKB tool: {name}"

    @pytest.mark.asyncio
    async def test_list_tools_returns_all(self):
        """list_tools() handler should return all tools."""
        result = await list_tools()
        assert len(result) == len(ALL_TOOLS)


class TestToolRouting:
    """Tests for tool routing."""

    @pytest.mark.asyncio
    async def test_substrate_tool_routes_correctly(self, mock_get_cursor):
        """Substrate tool names should route to substrate handler."""
        mock_cur = mock_get_cursor.__enter__.return_value
        mock_cur.fetchall.return_value = []
        mock_cur.fetchone.return_value = None

        # knowledge_search validates query before DB access
        result = await call_tool("knowledge_search", {"query": ""})
        assert len(result) == 1
        data = json.loads(result[0].text)
        # Empty query returns an error
        assert "success" in data

    @pytest.mark.asyncio
    async def test_vkb_tool_routes_to_vkb(self, mock_get_cursor):
        """VKB tool names should route to VKB handler."""
        mock_cur = mock_get_cursor.__enter__.return_value
        mock_cur.fetchall.return_value = []

        result = await call_tool("session_list", {})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "sessions" in data or "error" in data or "success" in data

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Unknown tool names should return error response."""
        result = await call_tool("nonexistent_tool", {})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["success"] is False
        assert "Unknown tool" in data["error"]

    @pytest.mark.asyncio
    async def test_server_name(self):
        """Server should be named 'valence'."""
        assert server.name == "valence"


class TestResources:
    """Tests for resource availability — updated for v2 (WU-11)."""

    @pytest.mark.asyncio
    async def test_list_resources_returns_two(self):
        """Should expose articles/recent and stats resources (v2)."""
        from valence.mcp_server import list_resources

        resources = await list_resources()
        assert len(resources) == 2
        uris = {str(r.uri) for r in resources}
        assert "valence://articles/recent" in uris
        assert "valence://stats" in uris

    @pytest.mark.asyncio
    async def test_no_beliefs_resource(self):
        """valence://beliefs/recent should NOT be present in v2."""
        from valence.mcp_server import list_resources

        resources = await list_resources()
        uris = {str(r.uri) for r in resources}
        assert "valence://beliefs/recent" not in uris, "beliefs/recent resource should have been removed in v2"
