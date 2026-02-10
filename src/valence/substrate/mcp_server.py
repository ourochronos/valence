"""MCP Server for Valence Knowledge Substrate.

Provides tools for:
- Belief management (query, create, supersede, get)
- Semantic search (belief_search with embeddings)
- Entity operations (get, search, merge)
- Tension handling (list, resolve)
- Trust checking (who do I trust on topics)
- Confidence explanation (why this confidence score)

Resources:
- valence://beliefs/recent - Recent beliefs
- valence://trust/graph - Trust relationships
- valence://stats - Database statistics

Tool implementations are in tools.py (shared with the HTTP server).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, TextResourceContents
from oro_db import get_cursor, init_schema
from oro_models import Belief
from pydantic import AnyUrl

from ..core.exceptions import DatabaseException, ValidationException
from ..core.health import DatabaseStats, cli_health_check, startup_checks
from .tools import SUBSTRATE_TOOLS, handle_substrate_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("valence-substrate")


# ============================================================================
# Tool Definitions (delegated to tools.py)
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
            uri=AnyUrl("valence://beliefs/recent"),
            name="Recent Beliefs",
            description="Most recently created or modified beliefs",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("valence://trust/graph"),
            name="Trust Graph",
            description="Trust relationships between entities and federation nodes",
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
    if uri == "valence://beliefs/recent":
        data = get_recent_beliefs()
    elif uri == "valence://trust/graph":
        data = get_trust_graph()
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


def get_recent_beliefs(limit: int = 20) -> dict[str, Any]:
    """Get recent beliefs for the resource."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT b.*, array_agg(DISTINCT e.name) FILTER (WHERE e.name IS NOT NULL) as entity_names
            FROM beliefs b
            LEFT JOIN belief_entities be ON b.id = be.belief_id
            LEFT JOIN entities e ON be.entity_id = e.id
            WHERE b.status = 'active'
            GROUP BY b.id
            ORDER BY b.modified_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()

        beliefs = []
        for row in rows:
            belief = Belief.from_row(dict(row))
            belief_dict = belief.to_dict()
            belief_dict["entity_names"] = row.get("entity_names") or []
            beliefs.append(belief_dict)

        return {
            "beliefs": beliefs,
            "count": len(beliefs),
            "as_of": datetime.now().isoformat(),
        }


def get_trust_graph() -> dict[str, Any]:
    """Get trust relationships for the resource."""
    result: dict[str, Any] = {
        "entities": [],
        "federation_nodes": [],
        "as_of": datetime.now().isoformat(),
    }

    with get_cursor() as cur:
        # Get entities with belief counts (as proxy for trust/authority)
        cur.execute(
            """
            SELECT e.id, e.name, e.type, COUNT(be.belief_id) as belief_count,
                   AVG((b.confidence->>'overall')::numeric) as avg_confidence
            FROM entities e
            LEFT JOIN belief_entities be ON e.id = be.entity_id
            LEFT JOIN beliefs b ON be.belief_id = b.id AND b.status = 'active'
            WHERE e.canonical_id IS NULL
            GROUP BY e.id
            HAVING COUNT(be.belief_id) > 0
            ORDER BY belief_count DESC
            LIMIT 50
            """
        )
        for row in cur.fetchall():
            result["entities"].append(
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "type": row["type"],
                    "belief_count": row["belief_count"],
                    "avg_confidence": (float(row["avg_confidence"]) if row["avg_confidence"] else None),
                }
            )

        # Get federation nodes with trust scores
        try:
            cur.execute(
                """
                SELECT fn.id, fn.name, fn.instance_url, fn.status,
                       nt.trust, nt.beliefs_received, nt.beliefs_corroborated, nt.beliefs_disputed
                FROM federation_nodes fn
                LEFT JOIN node_trust nt ON fn.id = nt.node_id
                WHERE fn.status != 'blocked'
                ORDER BY (nt.trust->>'overall')::numeric DESC NULLS LAST
                LIMIT 20
                """
            )
            for row in cur.fetchall():
                node_data = {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "instance_url": row["instance_url"],
                    "status": row["status"],
                }
                if row["trust"]:
                    node_data["trust"] = row["trust"]
                    node_data["beliefs_received"] = row["beliefs_received"]
                    node_data["beliefs_corroborated"] = row["beliefs_corroborated"]
                    node_data["beliefs_disputed"] = row["beliefs_disputed"]
                result["federation_nodes"].append(node_data)
        except Exception as e:
            logger.debug(f"Federation tables may not exist: {e}")

    return result


def get_stats() -> dict[str, Any]:
    """Get database statistics for the resource."""
    stats = DatabaseStats.collect()

    with get_cursor() as cur:
        # Get domain distribution
        cur.execute(
            """
            SELECT domain_path[1] as domain, COUNT(*) as count
            FROM beliefs
            WHERE status = 'active' AND array_length(domain_path, 1) > 0
            GROUP BY domain_path[1]
            ORDER BY count DESC
            LIMIT 10
            """
        )
        domains = {row["domain"]: row["count"] for row in cur.fetchall()}

        # Get confidence distribution
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
            FROM beliefs
            WHERE status = 'active'
            GROUP BY confidence_level
            ORDER BY count DESC
            """
        )
        confidence_dist = {row["confidence_level"]: row["count"] for row in cur.fetchall()}

        # Get entity type distribution
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

    return {
        "totals": stats.to_dict(),
        "domains": domains,
        "confidence_distribution": confidence_dist,
        "entity_types": entity_types,
        "as_of": datetime.now().isoformat(),
    }


# ============================================================================
# Tool Router (delegates to tools.py)
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

    # Initialize schema
    try:
        init_schema()
        logger.info("Schema initialized")
    except DatabaseException as e:
        logger.warning(f"Schema initialization failed (may already exist): {e}")

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    run()
