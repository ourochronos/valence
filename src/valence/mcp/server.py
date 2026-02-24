"""Unified MCP Server for Valence v2.

Provides 24 knowledge substrate tools via MCP protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, TextResourceContents
from pydantic import AnyUrl

from valence.core.db import get_cursor, init_schema
from valence.core.exceptions import DatabaseException, ValidationException
from valence.core.health import DatabaseStats, cli_health_check, startup_checks

from .handlers.admin import admin_forget, admin_maintenance, admin_stats
from .handlers.articles import (
    article_compile,
    article_create,
    article_get,
    article_merge,
    article_search,
    article_split,
    article_update,
    knowledge_search,
)
from .handlers.contention import contention_detect, contention_list, contention_resolve
from .handlers.entities import entity_get, entity_search
from .handlers.provenance import provenance_get, provenance_link, provenance_trace
from .handlers.sources import source_get, source_ingest, source_list, source_search
from .tools import SUBSTRATE_TOOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("valence")


# ============================================================================
# Tool Handler Registry
# ============================================================================

TOOL_HANDLERS: dict[str, Any] = {
    # Source tools
    "source_ingest": source_ingest,
    "source_get": source_get,
    "source_search": source_search,
    "source_list": source_list,
    # Retrieval
    "knowledge_search": knowledge_search,
    # Article tools
    "article_create": article_create,
    "article_get": article_get,
    "article_update": article_update,
    "article_search": article_search,
    "article_compile": article_compile,
    "article_split": article_split,
    "article_merge": article_merge,
    # Provenance tools
    "provenance_link": provenance_link,
    "provenance_get": provenance_get,
    "provenance_trace": provenance_trace,
    # Entity tools
    "entity_get": entity_get,
    "entity_search": entity_search,
    # Contention tools
    "contention_detect": contention_detect,
    "contention_list": contention_list,
    "contention_resolve": contention_resolve,
    # Admin tools
    "admin_forget": admin_forget,
    "admin_stats": admin_stats,
    "admin_maintenance": admin_maintenance,
}


# ============================================================================
# MCP Server Protocol Implementation
# ============================================================================


@server.list_tools()
async def list_tools():
    """List all available tools."""
    return SUBSTRATE_TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route tool calls to the appropriate handler."""
    try:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            result = {"success": False, "error": f"Unknown tool: {name}"}
        else:
            result = handler(**arguments)

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
# Resource Definitions
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


def get_recent_articles(limit: int = 20) -> dict[str, Any]:
    """Get recent articles for the resource."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT a.id, a.title, a.content, a.status, a.version,
                   a.confidence, a.domain_path,
                   a.modified_at, a.created_at,
                   COUNT(DISTINCT asrc.source_id) AS source_count
            FROM articles a
            LEFT JOIN article_sources asrc ON a.id = asrc.article_id
            WHERE a.status = 'active'
            GROUP BY a.id
            ORDER BY a.modified_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()

        articles = []
        for row in rows:
            d = dict(row)
            d["id"] = str(d["id"])
            if d.get("created_at"):
                d["created_at"] = d["created_at"].isoformat()
            if d.get("modified_at"):
                d["modified_at"] = d["modified_at"].isoformat()
            articles.append(d)

        return {
            "articles": articles,
            "count": len(articles),
            "as_of": datetime.now().isoformat(),
        }


def get_stats() -> dict[str, Any]:
    """Get database statistics for the resource."""
    stats = DatabaseStats.collect()

    with get_cursor() as cur:
        try:
            cur.execute(
                """
                SELECT domain_path[1] as domain, COUNT(*) as count
                FROM articles
                WHERE status = 'active' AND array_length(domain_path, 1) > 0
                GROUP BY domain_path[1]
                ORDER BY count DESC
                LIMIT 10
                """
            )
            domains = {row["domain"]: row["count"] for row in cur.fetchall()}
        except Exception:
            domains = {}

        try:
            cur.execute(
                """
                SELECT
                    CASE
                        WHEN (confidence->>'overall')::numeric >= 0.9 THEN 'very_high'
                        WHEN (confidence->>'overall')::numeric >= 0.75 THEN 'high'
                        WHEN (confidence->>'overall')::numeric >= 0.5 THEN 'moderate'
                        WHEN (confidence->>'overall')::numeric >= 0.25 THEN 'low'
                        ELSE 'very_low'
                    END as confidence_level,
                    COUNT(*) as count
                FROM articles
                WHERE status = 'active'
                GROUP BY confidence_level
                ORDER BY count DESC
                """
            )
            confidence_dist = {row["confidence_level"]: row["count"] for row in cur.fetchall()}
        except Exception:
            confidence_dist = {}

        try:
            cur.execute(
                """
                SELECT type, COUNT(*) as count
                FROM entities
                WHERE canonical_id IS NULL
                GROUP BY type
                ORDER BY count DESC
                """
            )
            entity_types = {row["type"]: row["count"] for row in cur.fetchall()}
        except Exception:
            entity_types = {}

    return {
        "totals": stats.to_dict(),
        "domains": domains,
        "confidence_distribution": confidence_dist,
        "entity_types": entity_types,
        "as_of": datetime.now().isoformat(),
    }


# ============================================================================
# Server Entry Point
# ============================================================================


def run() -> None:
    """Run the unified MCP server."""
    parser = argparse.ArgumentParser(description="Valence Unified MCP Server")
    parser.add_argument("--health-check", action="store_true", help="Run health check and exit")
    parser.add_argument("--skip-health-check", action="store_true", help="Skip startup health checks")
    args = parser.parse_args()

    if args.health_check:
        sys.exit(cli_health_check())

    logger.info("Valence unified MCP server starting...")

    if not args.skip_health_check:
        startup_checks(fail_fast=True)

    try:
        schema_dir = str(Path(__file__).parent.parent)
        init_schema(schema_dir)
        logger.info("Schema initialized")
    except (DatabaseException, Exception) as e:
        logger.warning(f"Schema initialization skipped (may already exist): {e}")

    # Check and run scheduled maintenance if needed
    try:
        from valence.core.maintenance import check_and_run_maintenance

        with get_cursor() as cur:
            result = check_and_run_maintenance(cur)
            if result:
                logger.info(f"Scheduled maintenance completed: {result['timestamp']}")
            else:
                logger.debug("Scheduled maintenance check: no action needed")
    except Exception as e:
        logger.warning(f"Scheduled maintenance check failed: {e}")

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    run()
