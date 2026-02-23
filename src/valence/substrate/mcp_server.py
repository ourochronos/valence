"""MCP Server for Valence Knowledge Substrate — v2.

Provides 16 tools for the v2 knowledge system:
  Sources:    source_ingest, source_get, source_search
  Retrieval:  knowledge_search
  Articles:   article_get, article_create, article_compile, article_update,
              article_split, article_merge
  Provenance: provenance_trace
  Contention: contention_list, contention_resolve
  Admin:      admin_forget, admin_stats, admin_maintenance

Resources:
  valence://articles/recent — Recent articles
  valence://stats           — Database statistics

Tool implementations are in substrate/tools/ (delegated from handlers.py).
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

from valence.core.db import DatabaseError as OurDatabaseError
from valence.core.db import get_cursor, init_schema

from ..core.exceptions import DatabaseException, ValidationException
from ..core.health import DatabaseStats, cli_health_check, startup_checks
from .tools import SUBSTRATE_TOOLS, handle_substrate_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("valence-substrate")


# ============================================================================
# Tool Definitions (delegated to tools/definitions.py via tools/__init__.py)
# ============================================================================


@server.list_tools()
async def list_tools():
    """List available tools."""
    return SUBSTRATE_TOOLS


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
        # Get domain distribution from articles
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

        # Get confidence distribution from articles
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

        # Get entity type distribution
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
# Tool Router (delegates to tools/handlers.py)
# ============================================================================


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls by delegating to the shared tool implementations."""
    try:
        result = handle_substrate_tool(name, arguments)
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


    parser = argparse.ArgumentParser(description="Valence Substrate MCP Server")
    parser.add_argument("--health-check", action="store_true", help="Run health check and exit")
    parser.add_argument("--skip-health-check", action="store_true", help="Skip startup health checks")
    args = parser.parse_args()

    # Health check mode
    if args.health_check:
        sys.exit(cli_health_check())

    logger.info("Valence Substrate MCP server starting...")

    # Run startup health checks (unless skipped)
    if not args.skip_health_check:
        startup_checks(fail_fast=True)

    # Initialize schema (tolerates already-existing tables/constraints)
    try:
        schema_dir = Path(__file__).parent
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
