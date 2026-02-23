"""Unified MCP Server for Valence v2.

Combines both knowledge substrate and conversation tracking (VKB) servers
into a single MCP endpoint. Clients connect to one server instead of two.

Tools (v2 substrate — 16 tools):
- Sources:     source_ingest, source_get, source_search
- Retrieval:   knowledge_search
- Articles:    article_get, article_create, article_compile, article_update,
               article_split, article_merge
- Provenance:  provenance_trace
- Contention:  contention_list, contention_resolve
- Admin:       admin_forget, admin_stats, admin_maintenance

VKB Tools: session_start, exchange_add, pattern_record, etc.

Resources:
- valence://articles/recent - Recent compiled articles
- valence://stats           - Database statistics

Tool implementations are in substrate/tools/ and vkb/tools/.
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
from mcp.types import Resource, TextContent, TextResourceContents
from pydantic import AnyUrl

from valence.lib.our_db import init_schema
from valence.lib.our_db.exceptions import DatabaseError as OurDatabaseError

from .core.exceptions import DatabaseException, ValidationException
from .core.health import cli_health_check, startup_checks
from .substrate.mcp_server import get_recent_articles, get_stats
from .substrate.tools import SUBSTRATE_HANDLERS, SUBSTRATE_TOOLS, handle_substrate_tool
from .vkb.tools import VKB_HANDLERS, VKB_TOOLS, handle_vkb_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("valence")

# Combined tool list (no duplicates since substrate and VKB tool names don't overlap)
ALL_TOOLS = SUBSTRATE_TOOLS + VKB_TOOLS


# ============================================================================
# Tool Definitions
# ============================================================================


@server.list_tools()
async def list_tools():
    """List all available tools."""
    return ALL_TOOLS


# ============================================================================
# Resource Definitions (from substrate)
# ============================================================================


@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri=AnyUrl("valence://articles/recent"),
            name="Recent Articles",
            description="Most recently created or modified knowledge articles",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("valence://stats"),
            name="Database Statistics",
            description="Current statistics about the Valence knowledge base",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> list[TextResourceContents]:
    """Read a resource by URI."""
    if uri == "valence://articles/recent":
        data = get_recent_articles()
    elif uri == "valence://stats":
        data = get_stats()
    else:
        data = {"error": f"Unknown resource: {uri}"}

    return [
        TextResourceContents(
            uri=AnyUrl(uri),
            mimeType="application/json",
            text=json.dumps(data, indent=2, default=str),
        )
    ]


# ============================================================================
# Tool Router
# ============================================================================


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route tool calls to substrate or VKB handler."""
    try:
        if name in SUBSTRATE_HANDLERS:
            result = handle_substrate_tool(name, arguments)
        elif name in VKB_HANDLERS:
            result = handle_vkb_tool(name, arguments)
        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except ValidationException as e:
        logger.warning(f"Validation error in tool {name}: {e}")
        return [
            TextContent(
                type="text",
                text=json.dumps({"success": False, "error": f"Validation error: {e.message}", "details": e.details}),
            )
        ]
    except DatabaseException as e:
        logger.error(f"Database error in tool {name}: {e}")
        return [
            TextContent(
                type="text",
                text=json.dumps({"success": False, "error": f"Database error: {e.message}"}),
            )
        ]
    except Exception as e:
        logger.exception(f"Unexpected error in tool {name}")
        return [
            TextContent(
                type="text",
                text=json.dumps({"success": False, "error": f"Internal error: {str(e)}"}),
            )
        ]


# ============================================================================
# Server Entry Point
# ============================================================================


def run() -> None:
    """Run the unified MCP server."""
    from .core.config import bridge_db_env

    bridge_db_env()

    parser = argparse.ArgumentParser(description="Valence Unified MCP Server")
    parser.add_argument("--health-check", action="store_true", help="Run health check and exit")
    parser.add_argument("--skip-health-check", action="store_true", help="Skip startup health checks")
    args = parser.parse_args()

    if args.health_check:
        sys.exit(cli_health_check())

    logger.info("Valence unified MCP server starting...")

    if not args.skip_health_check:
        startup_checks(fail_fast=True)

    # Initialize schema (tolerates already-existing tables/constraints)
    try:
        schema_dir = Path(__file__).parent / "substrate"
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
