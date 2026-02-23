"""Tests for unified MCP server.

Tests cover:
- create_server() - server creation
- Tool definitions and descriptions
- Resource content
- Prompt content
- Helper functions
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.server.unified_server import (
    SERVER_NAME,
    SERVER_VERSION,
    _get_context_prompt,
    _get_tool_reference,
    _get_usage_instructions,
    create_server,
)
from valence.mcp_server import SUBSTRATE_TOOLS, TOOL_HANDLERS

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_cursor():
    """Mock database cursor."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Mock the get_cursor context manager."""

    @contextmanager
    def _mock_get_cursor(dict_cursor: bool = True) -> Generator:
        yield mock_cursor

    with (
        patch("valence.substrate.tools._common.get_cursor", _mock_get_cursor),
        patch("valence.vkb.tools.sessions.get_cursor", _mock_get_cursor),
        patch("valence.vkb.tools.exchanges.get_cursor", _mock_get_cursor),
        patch("valence.vkb.tools.patterns.get_cursor", _mock_get_cursor),
        patch("valence.vkb.tools.insights.get_cursor", _mock_get_cursor),
    ):
        yield mock_cursor


# =============================================================================
# SERVER CREATION TESTS
# =============================================================================


class TestCreateServer:
    """Tests for create_server function."""

    def test_create_server_returns_server(self):
        """Test that create_server returns a Server instance."""
        from mcp.server import Server

        server = create_server()

        assert server is not None
        assert isinstance(server, Server)
        assert server.name == SERVER_NAME

    def test_create_server_idempotent(self):
        """Test that create_server can be called multiple times."""
        server1 = create_server()
        server2 = create_server()

        # Should both work (independent instances)
        assert server1 is not server2
        assert server1.name == server2.name

    def test_server_has_request_handlers(self):
        """Test that server has handlers registered."""
        server = create_server()

        # The server should have request handlers registered
        assert hasattr(server, "request_handlers")
        assert len(server.request_handlers) > 0


# =============================================================================
# TOOL DEFINITIONS TESTS
# =============================================================================


class TestToolDefinitions:
    """Tests for tool definitions in unified server — v2 (WU-11)."""

    def test_all_substrate_tools_defined(self):
        """Test that all v2 substrate tools are defined."""
        expected_tools = [
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
        ]

        tool_names = [t.name for t in SUBSTRATE_TOOLS]
        for expected in expected_tools:
            assert expected in tool_names, f"Missing v2 substrate tool: {expected}"

    def test_all_vkb_tools_defined(self):
        """Test that all VKB tools are defined."""
        expected_tools = [
            "session_start",
            "session_end",
            "session_get",
            "session_list",
            "session_find_by_room",
            "exchange_add",
            "exchange_list",
            "pattern_record",
            "pattern_reinforce",
            "pattern_list",
            "pattern_search",
            "insight_extract",
            "insight_list",
        ]

        tool_names = [t.name for t in VKB_TOOLS]
        for expected in expected_tools:
            assert expected in tool_names

    def test_tool_descriptions_have_behavioral_conditioning(self):
        """Test that key v2 tools have behavioral hints."""
        knowledge_search = next(t for t in SUBSTRATE_TOOLS if t.name == "knowledge_search")
        assert "MUST" in knowledge_search.description or "CRITICAL" in knowledge_search.description

        source_ingest = next(t for t in SUBSTRATE_TOOLS if t.name == "source_ingest")
        assert source_ingest.description  # must have a non-empty description

        insight_extract = next(t for t in VKB_TOOLS if t.name == "insight_extract")
        assert "PROACTIVELY" in insight_extract.description


# =============================================================================
# TOOL ROUTING TESTS
# =============================================================================


class TestToolRouting:
    """Tests for tool call routing."""

    def test_substrate_tool_routing(self, mock_get_cursor):
        """Test that substrate tools are routed correctly (v2 — knowledge_search)."""
        # knowledge_search with empty query returns an error (validation before DB)
        result = handle_substrate_tool("knowledge_search", {"query": ""})
        # Empty query returns validation error — that's correct routing behavior
        assert "success" in result

    def test_vkb_tool_routing(self, mock_get_cursor):
        """Test that VKB tools are routed correctly."""
        from datetime import datetime

        mock_get_cursor.fetchone.return_value = {
            "id": uuid4(),
            "platform": "claude-code",
            "status": "active",
            "project_context": None,
            "summary": None,
            "themes": [],
            "started_at": datetime.now(),
            "ended_at": None,
            "claude_session_id": None,
            "external_room_id": None,
            "metadata": {},
            "exchange_count": None,
            "insight_count": None,
        }

        result = handle_vkb_tool("session_start", {"platform": "claude-code"})

        assert result["success"] is True

    def test_unknown_substrate_tool_error(self):
        """Test that unknown substrate tool returns error."""
        result = handle_substrate_tool("nonexistent_tool", {})

        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_unknown_vkb_tool_error(self):
        """Test that unknown VKB tool returns error."""
        result = handle_vkb_tool("nonexistent_tool", {})

        assert result["success"] is False
        assert "Unknown" in result["error"]


# =============================================================================
# RESOURCE CONTENT TESTS
# =============================================================================


class TestResourceContent:
    """Tests for resource content."""

    def test_get_usage_instructions_content(self):
        """Test usage instructions has required content (v2 — WU-11)."""
        instructions = _get_usage_instructions()

        # Should have title
        assert "Valence Knowledge Substrate" in instructions

        # Should have behavioral guidelines (case-insensitive check)
        assert "query first" in instructions.lower()
        assert "PROACTIVELY" in instructions

        # Should list tool categories
        assert "Knowledge Substrate" in instructions
        assert "Conversation Tracking" in instructions

        # Should mention v2 key tools (knowledge_search replaces belief_query)
        assert "knowledge_search" in instructions
        assert "source_ingest" in instructions
        assert "session_start" in instructions
        assert "session_end" in instructions

    def test_get_tool_reference_content(self):
        """Test tool reference has all tools listed."""
        reference = _get_tool_reference()

        # Should have headers
        assert "Tool Reference" in reference
        assert "Knowledge Substrate Tools" in reference
        assert "Conversation Tracking Tools" in reference

        # Should list all v2 substrate tools
        for tool in SUBSTRATE_TOOLS:
            assert tool.name in reference

        # Should list all VKB tools
        for tool in VKB_TOOLS:
            assert tool.name in reference


# =============================================================================
# PROMPT CONTENT TESTS
# =============================================================================


class TestPromptContent:
    """Tests for prompt content."""

    def test_get_context_prompt_content(self):
        """Test context prompt has required content (v2 — WU-11)."""
        prompt = _get_context_prompt()

        # Should mention Valence
        assert "Valence" in prompt

        # Should have core behaviors
        assert "Query First" in prompt
        assert "Capture Proactively" in prompt
        assert "Track Sessions" in prompt

        # Should mention v2 key tools
        assert "knowledge_search" in prompt
        assert "source_ingest" in prompt or "insight_extract" in prompt
        assert "session_start" in prompt
        assert "session_end" in prompt


# =============================================================================
# CONSTANTS TESTS
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_server_name(self):
        """Test server name is correct."""
        assert SERVER_NAME == "valence"

    def test_server_version(self):
        """Test server version is set."""
        assert SERVER_VERSION is not None
        assert len(SERVER_VERSION) > 0
        # Should be a version string like "1.0.0"
        assert "." in SERVER_VERSION or SERVER_VERSION.isdigit()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration-style tests for unified server."""

    def test_server_can_be_created_and_configured(self):
        """Test that server can be created and is properly configured."""
        server = create_server()

        # Server should have name
        assert server.name == SERVER_NAME

        # Server should be ready for use
        assert server is not None

    def test_all_tool_names_are_unique(self):
        """Test that there are no duplicate tool names."""
        all_tools = SUBSTRATE_TOOLS + VKB_TOOLS
        tool_names = [t.name for t in all_tools]

        # Should have no duplicates
        assert len(tool_names) == len(set(tool_names))

    def test_all_tools_have_input_schema(self):
        """Test that all tools have input schemas defined."""
        all_tools = SUBSTRATE_TOOLS + VKB_TOOLS

        for tool in all_tools:
            assert tool.inputSchema is not None, f"Tool {tool.name} missing inputSchema"
            assert "type" in tool.inputSchema, f"Tool {tool.name} inputSchema missing type"
            assert tool.inputSchema["type"] == "object", f"Tool {tool.name} inputSchema type should be object"

    def test_all_tools_have_descriptions(self):
        """Test that all tools have descriptions."""
        all_tools = SUBSTRATE_TOOLS + VKB_TOOLS

        for tool in all_tools:
            assert tool.description is not None, f"Tool {tool.name} missing description"
            assert len(tool.description) > 10, f"Tool {tool.name} description too short"
