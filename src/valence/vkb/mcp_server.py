"""MCP Server for Valence VKB (Conversation Tracking).

Provides tools for:
- Session management (start, end, get, list)
- Exchange capture (add, list)
- Pattern tracking (record, reinforce, list)
- Insight extraction (extract to knowledge base)

Tool definitions and implementations live in tools.py. This module
provides the stdio MCP server wrapper that delegates to those shared
implementations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from valence.core.db import DatabaseError as OurDatabaseError
from valence.core.db import init_schema

from ..core.exceptions import DatabaseException, ValidationException
from ..core.health import cli_health_check, startup_checks
from .tools import (
    VKB_TOOLS,
    exchange_add,
    exchange_list,
    handle_vkb_tool,
    insight_extract,
    insight_list,
    pattern_list,
    pattern_record,
    pattern_reinforce,
    pattern_search,
    session_end,
    session_find_by_room,
    session_get,
    session_list,
    session_start,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("valence-vkb")

# Re-export individual tool functions for backward compatibility (tests import these)
__all__ = [
    "run",
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


# ============================================================================
# MCP Handlers
# ============================================================================


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return VKB_TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls by delegating to shared implementations."""
    try:
        result = handle_vkb_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except ValidationException as e:
        logger.warning(f"Validation error in tool {name}: {e}")
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": False,
                        "error": f"Validation error: {e.message}",
                        "details": e.details,
                    }
                ),
            )
        ]
    except DatabaseException as e:
        logger.error(f"Database error in tool {name}: {e}")
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": False,
                        "error": f"Database error: {e.message}",
                    }
                ),
            )
        ]
    except Exception as e:
        logger.exception(f"Unexpected error in tool {name}")
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": False,
                        "error": f"Internal error: {str(e)}",
                    }
                ),
            )
        ]


# ============================================================================
# Server Entry Point
# ============================================================================


def run() -> None:
    """Run the MCP server."""


    parser = argparse.ArgumentParser(description="Valence VKB MCP Server")
    parser.add_argument("--health-check", action="store_true", help="Run health check and exit")
    parser.add_argument("--skip-health-check", action="store_true", help="Skip startup health checks")
    args = parser.parse_args()

    # Health check mode
    if args.health_check:
        sys.exit(cli_health_check())

    logger.info("Valence VKB MCP server starting...")

    # Run startup health checks (unless skipped)
    if not args.skip_health_check:
        startup_checks(fail_fast=True)

    # Initialize schema (schema files live in substrate/, tolerates already-existing)
    try:
        schema_dir = Path(__file__).parent.parent / "substrate"
        init_schema(schema_dir)
        logger.info("Schema initialized")
    except (DatabaseException, OurDatabaseError) as e:
        logger.warning(f"Schema initialization skipped (may already exist): {e}")

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    run()
