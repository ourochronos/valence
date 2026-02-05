"""Base class and utilities for MCP servers.

Provides common patterns for tool routing, error handling, and response formatting
to reduce duplication across Valence MCP servers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .db import init_schema
from .exceptions import DatabaseException, ValidationException
from .health import startup_checks, cli_health_check

logger = logging.getLogger(__name__)


class MCPServerBase(ABC):
    """Base class for Valence MCP servers.

    Provides common functionality:
    - Argument parsing with health check support
    - Startup health checks
    - Schema initialization
    - Error handling in tool calls
    - JSON response formatting

    Subclasses should implement:
    - server_name: The MCP server name
    - get_tools(): Return list of Tool definitions
    - handle_tool(): Handle tool calls

    Example:
        class MyServer(MCPServerBase):
            server_name = "my-server"

            def get_tools(self) -> list[Tool]:
                return [Tool(name="my_tool", ...)]

            def handle_tool(self, name: str, arguments: dict) -> dict:
                if name == "my_tool":
                    return {"success": True, "data": "..."}
                return {"success": False, "error": f"Unknown tool: {name}"}

        if __name__ == "__main__":
            MyServer().run()
    """

    server_name: str = "valence-server"
    server_description: str = "Valence MCP Server"

    def __init__(self):
        self.server = Server(self.server_name)
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP server handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return self.get_tools()

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            return await self._handle_tool_call(name, arguments)

    async def _handle_tool_call(
        self,
        name: str,
        arguments: dict[str, Any]
    ) -> list[TextContent]:
        """Handle a tool call with consistent error handling."""
        try:
            result = self.handle_tool(name, arguments)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str)
            )]

        except ValidationException as e:
            logger.warning(f"Validation error in tool {name}: {e}")
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Validation error: {e.message}",
                "details": e.details,
            }))]

        except DatabaseException as e:
            logger.error(f"Database error in tool {name}: {e}")
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Database error: {e.message}",
            }))]

        except Exception as e:  # Intentionally broad: top-level handler for unexpected errors
            logger.exception(f"Unexpected error in tool {name}")
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Internal error: {str(e)}",
            }))]

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return list of tools provided by this server.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def handle_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle a tool call and return result.

        Must be implemented by subclasses.

        Args:
            name: The tool name
            arguments: Tool arguments from the client

        Returns:
            Dict with 'success' key and either result data or 'error' key
        """
        pass

    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description=self.server_description)
        parser.add_argument(
            "--health-check",
            action="store_true",
            help="Run health check and exit"
        )
        parser.add_argument(
            "--skip-health-check",
            action="store_true",
            help="Skip startup health checks"
        )
        return parser.parse_args()

    def run(self) -> None:
        """Run the MCP server."""
        args = self.parse_args()

        # Health check mode
        if args.health_check:
            sys.exit(cli_health_check())

        logger.info(f"{self.server_name} MCP server starting...")

        # Run startup health checks (unless skipped)
        if not args.skip_health_check:
            startup_checks(fail_fast=True)

        # Initialize schema
        try:
            init_schema()
            logger.info("Schema initialized")
        except DatabaseException as e:
            logger.warning(f"Schema initialization failed (may already exist): {e}")

        # Run the server
        async def main():
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )

        asyncio.run(main())


def success_response(**kwargs) -> dict[str, Any]:
    """Create a successful response dict.

    Args:
        **kwargs: Additional data to include in response

    Returns:
        Dict with success=True and all kwargs
    """
    return {"success": True, **kwargs}


def error_response(error: str, **kwargs) -> dict[str, Any]:
    """Create an error response dict.

    Args:
        error: Error message
        **kwargs: Additional data to include in response

    Returns:
        Dict with success=False, error message, and all kwargs
    """
    return {"success": False, "error": error, **kwargs}


def not_found_response(resource_type: str, resource_id: str) -> dict[str, Any]:
    """Create a not found error response.

    Args:
        resource_type: Type of resource (e.g., "Belief", "Entity")
        resource_id: ID of the missing resource

    Returns:
        Dict with success=False and appropriate error message
    """
    return error_response(f"{resource_type} not found: {resource_id}")


class ToolRouter:
    """Simple tool routing helper.

    Allows registering tool handlers and dispatching to them by name.

    Example:
        router = ToolRouter()

        @router.register("my_tool")
        def handle_my_tool(arg1: str, arg2: int = 10) -> dict:
            return {"success": True, "result": arg1 * arg2}

        # In handle_tool:
        result = router.dispatch(name, arguments)
    """

    def __init__(self):
        self._handlers: dict[str, Callable] = {}

    def register(self, name: str):
        """Decorator to register a tool handler.

        Args:
            name: Tool name to register
        """
        def decorator(func: Callable) -> Callable:
            self._handlers[name] = func
            return func
        return decorator

    def dispatch(
        self,
        name: str,
        arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Dispatch to the appropriate handler.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Handler result or error if tool not found
        """
        handler = self._handlers.get(name)
        if handler is None:
            return error_response(f"Unknown tool: {name}")
        return handler(**arguments)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._handlers

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._handlers.keys())
