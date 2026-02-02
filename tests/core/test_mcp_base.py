"""Tests for valence.core.mcp_base module."""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from valence.core.exceptions import DatabaseException, ValidationException


# ============================================================================
# success_response Tests
# ============================================================================

class TestSuccessResponse:
    """Tests for success_response helper function."""

    def test_basic_success(self):
        """Should return success=True."""
        from valence.core.mcp_base import success_response

        result = success_response()
        assert result["success"] is True

    def test_with_kwargs(self):
        """Should include additional kwargs."""
        from valence.core.mcp_base import success_response

        result = success_response(data="test", count=42)
        assert result["success"] is True
        assert result["data"] == "test"
        assert result["count"] == 42

    def test_with_nested_data(self):
        """Should handle nested data."""
        from valence.core.mcp_base import success_response

        result = success_response(
            belief={"id": "123", "content": "test"},
            entities=[{"name": "Entity1"}],
        )
        assert result["success"] is True
        assert result["belief"]["id"] == "123"
        assert len(result["entities"]) == 1


# ============================================================================
# error_response Tests
# ============================================================================

class TestErrorResponse:
    """Tests for error_response helper function."""

    def test_basic_error(self):
        """Should return success=False with error message."""
        from valence.core.mcp_base import error_response

        result = error_response("Something went wrong")
        assert result["success"] is False
        assert result["error"] == "Something went wrong"

    def test_with_kwargs(self):
        """Should include additional kwargs."""
        from valence.core.mcp_base import error_response

        result = error_response("Failed", code=404, details={"field": "id"})
        assert result["success"] is False
        assert result["error"] == "Failed"
        assert result["code"] == 404
        assert result["details"]["field"] == "id"


# ============================================================================
# not_found_response Tests
# ============================================================================

class TestNotFoundResponse:
    """Tests for not_found_response helper function."""

    def test_creates_not_found_error(self):
        """Should create appropriate not found message."""
        from valence.core.mcp_base import not_found_response

        result = not_found_response("Belief", "abc-123")
        assert result["success"] is False
        assert "Belief not found" in result["error"]
        assert "abc-123" in result["error"]

    def test_different_resource_types(self):
        """Should work with different resource types."""
        from valence.core.mcp_base import not_found_response

        result1 = not_found_response("Entity", "entity-1")
        assert "Entity not found" in result1["error"]

        result2 = not_found_response("Session", "session-2")
        assert "Session not found" in result2["error"]


# ============================================================================
# ToolRouter Tests
# ============================================================================

class TestToolRouter:
    """Tests for ToolRouter class."""

    def test_register_decorator(self):
        """Should register handlers via decorator."""
        from valence.core.mcp_base import ToolRouter

        router = ToolRouter()

        @router.register("test_tool")
        def handler(arg1: str) -> dict:
            return {"result": arg1}

        assert router.has_tool("test_tool")

    def test_dispatch_to_handler(self):
        """Should dispatch to registered handler."""
        from valence.core.mcp_base import ToolRouter

        router = ToolRouter()

        @router.register("my_tool")
        def handler(value: int) -> dict:
            return {"success": True, "doubled": value * 2}

        result = router.dispatch("my_tool", {"value": 5})
        assert result["success"] is True
        assert result["doubled"] == 10

    def test_dispatch_unknown_tool(self):
        """Should return error for unknown tool."""
        from valence.core.mcp_base import ToolRouter

        router = ToolRouter()
        result = router.dispatch("nonexistent", {})
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_has_tool(self):
        """Should check if tool is registered."""
        from valence.core.mcp_base import ToolRouter

        router = ToolRouter()

        @router.register("exists")
        def handler():
            return {}

        assert router.has_tool("exists") is True
        assert router.has_tool("not_exists") is False

    def test_tool_names(self):
        """Should return list of registered tool names."""
        from valence.core.mcp_base import ToolRouter

        router = ToolRouter()

        @router.register("tool_a")
        def a():
            return {}

        @router.register("tool_b")
        def b():
            return {}

        names = router.tool_names
        assert "tool_a" in names
        assert "tool_b" in names
        assert len(names) == 2

    def test_multiple_handlers(self):
        """Should handle multiple registered tools."""
        from valence.core.mcp_base import ToolRouter

        router = ToolRouter()

        @router.register("add")
        def add_handler(a: int, b: int) -> dict:
            return {"result": a + b}

        @router.register("multiply")
        def mul_handler(a: int, b: int) -> dict:
            return {"result": a * b}

        assert router.dispatch("add", {"a": 3, "b": 4})["result"] == 7
        assert router.dispatch("multiply", {"a": 3, "b": 4})["result"] == 12


# ============================================================================
# MCPServerBase Tests
# ============================================================================

class TestMCPServerBase:
    """Tests for MCPServerBase abstract class."""

    def test_is_abstract(self):
        """Should not be directly instantiable."""
        from valence.core.mcp_base import MCPServerBase

        # Should raise because get_tools and handle_tool are abstract
        with pytest.raises(TypeError, match="abstract"):
            MCPServerBase()

    def test_concrete_implementation(self):
        """Should be implementable."""
        from valence.core.mcp_base import MCPServerBase

        class TestServer(MCPServerBase):
            server_name = "test-server"

            def get_tools(self):
                return []

            def handle_tool(self, name, arguments):
                return {"success": True}

        with patch("mcp.server.Server"):
            server = TestServer()
            assert server.server_name == "test-server"

    def test_parse_args_health_check(self):
        """Should parse --health-check argument."""
        from valence.core.mcp_base import MCPServerBase

        class TestServer(MCPServerBase):
            server_name = "test"

            def get_tools(self):
                return []

            def handle_tool(self, name, arguments):
                return {}

        with patch("mcp.server.Server"):
            server = TestServer()

        with patch("sys.argv", ["test", "--health-check"]):
            args = server.parse_args()
            assert args.health_check is True

    def test_parse_args_skip_health_check(self):
        """Should parse --skip-health-check argument."""
        from valence.core.mcp_base import MCPServerBase

        class TestServer(MCPServerBase):
            server_name = "test"

            def get_tools(self):
                return []

            def handle_tool(self, name, arguments):
                return {}

        with patch("mcp.server.Server"):
            server = TestServer()

        with patch("sys.argv", ["test", "--skip-health-check"]):
            args = server.parse_args()
            assert args.skip_health_check is True


# ============================================================================
# MCPServerBase._handle_tool_call Tests
# ============================================================================

class TestMCPServerBaseHandleToolCall:
    """Tests for MCPServerBase._handle_tool_call method."""

    @pytest.fixture
    def test_server(self):
        """Create a test server implementation."""
        from valence.core.mcp_base import MCPServerBase

        class TestServer(MCPServerBase):
            server_name = "test"

            def get_tools(self):
                return []

            def handle_tool(self, name, arguments):
                if name == "success":
                    return {"success": True, "data": "test"}
                elif name == "validation_error":
                    raise ValidationException("Bad input", field="test")
                elif name == "db_error":
                    raise DatabaseException("DB failed")
                elif name == "generic_error":
                    raise RuntimeError("Unknown error")
                return {"success": False}

        with patch("mcp.server.Server"):
            return TestServer()

    @pytest.mark.asyncio
    async def test_successful_call(self, test_server):
        """Should return success response."""
        result = await test_server._handle_tool_call("success", {})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["success"] is True
        assert data["data"] == "test"

    @pytest.mark.asyncio
    async def test_validation_error(self, test_server):
        """Should handle ValidationException."""
        result = await test_server._handle_tool_call("validation_error", {})
        data = json.loads(result[0].text)
        assert data["success"] is False
        assert "Validation error" in data["error"]

    @pytest.mark.asyncio
    async def test_database_error(self, test_server):
        """Should handle DatabaseException."""
        result = await test_server._handle_tool_call("db_error", {})
        data = json.loads(result[0].text)
        assert data["success"] is False
        assert "Database error" in data["error"]

    @pytest.mark.asyncio
    async def test_generic_error(self, test_server):
        """Should handle generic exceptions."""
        result = await test_server._handle_tool_call("generic_error", {})
        data = json.loads(result[0].text)
        assert data["success"] is False
        assert "Internal error" in data["error"]


# ============================================================================
# MCPServerBase.run Tests
# ============================================================================

class TestMCPServerBaseRun:
    """Tests for MCPServerBase.run method."""

    def test_health_check_mode(self):
        """Should run health check and exit in health check mode."""
        from valence.core.mcp_base import MCPServerBase

        class TestServer(MCPServerBase):
            server_name = "test"

            def get_tools(self):
                return []

            def handle_tool(self, name, arguments):
                return {}

        with patch("mcp.server.Server"):
            server = TestServer()

        with patch("sys.argv", ["test", "--health-check"]):
            with patch("valence.core.mcp_base.cli_health_check", return_value=0):
                with pytest.raises(SystemExit) as exc_info:
                    server.run()
                assert exc_info.value.code == 0

    def test_skip_health_check(self):
        """Should skip startup checks when --skip-health-check is set."""
        from valence.core.mcp_base import MCPServerBase

        class TestServer(MCPServerBase):
            server_name = "test"

            def get_tools(self):
                return []

            def handle_tool(self, name, arguments):
                return {}

        with patch("mcp.server.Server"):
            server = TestServer()

        with patch("sys.argv", ["test", "--skip-health-check"]):
            with patch("valence.core.mcp_base.startup_checks") as mock_startup:
                with patch("valence.core.mcp_base.init_schema"):
                    with patch("asyncio.run"):
                        server.run()
                # startup_checks should not have been called
                mock_startup.assert_not_called()

    def test_normal_run_calls_startup_checks(self):
        """Should call startup checks in normal mode."""
        from valence.core.mcp_base import MCPServerBase

        class TestServer(MCPServerBase):
            server_name = "test"

            def get_tools(self):
                return []

            def handle_tool(self, name, arguments):
                return {}

        with patch("mcp.server.Server"):
            server = TestServer()

        with patch("sys.argv", ["test"]):
            with patch("valence.core.mcp_base.startup_checks") as mock_startup:
                with patch("valence.core.mcp_base.init_schema"):
                    with patch("asyncio.run"):
                        server.run()
                mock_startup.assert_called_once_with(fail_fast=True)
