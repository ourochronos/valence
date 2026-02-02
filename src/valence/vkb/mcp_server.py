"""MCP Server for Valence VKB (Conversation Tracking).

Provides tools for:
- Session management (start, end, get, list)
- Exchange capture (add, list)
- Pattern tracking (record, reinforce, list)
- Insight extraction (extract to knowledge base)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any
from uuid import UUID

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from ..core.db import get_cursor, init_schema
from ..core.models import Session, Exchange, Pattern
from ..core.confidence import DimensionalConfidence
from ..core.health import startup_checks, cli_health_check
from ..core.exceptions import DatabaseException, ValidationException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("valence-vkb")


# ============================================================================
# Tool Definitions
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        # Session tools
        Tool(
            name="session_start",
            description="Begin a new conversation session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "enum": ["claude-code", "matrix", "api", "slack"],
                        "description": "Platform this session is on"
                    },
                    "project_context": {
                        "type": "string",
                        "description": "Project or topic context"
                    },
                    "external_room_id": {
                        "type": "string",
                        "description": "Room/channel ID for chat platforms"
                    },
                    "claude_session_id": {
                        "type": "string",
                        "description": "Claude Code session ID for resume"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional session metadata"
                    },
                },
                "required": ["platform"],
            },
        ),
        Tool(
            name="session_end",
            description="Close a session with summary and themes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "UUID of the session"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Session summary"
                    },
                    "themes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key themes from session"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["completed", "abandoned"],
                        "default": "completed"
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="session_get",
            description="Get session details.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "UUID of the session"
                    },
                    "include_exchanges": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include recent exchanges"
                    },
                    "exchange_limit": {
                        "type": "integer",
                        "default": 10
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="session_list",
            description="List sessions with filters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "description": "Filter by platform"
                    },
                    "project_context": {
                        "type": "string",
                        "description": "Filter by project"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "completed", "abandoned"],
                        "description": "Filter by status"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20
                    },
                },
            },
        ),
        Tool(
            name="session_find_by_room",
            description="Find active session by external room ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "external_room_id": {
                        "type": "string",
                        "description": "Room/channel ID"
                    },
                },
                "required": ["external_room_id"],
            },
        ),
        # Exchange tools
        Tool(
            name="exchange_add",
            description="Record a conversation turn.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "UUID of the session"
                    },
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant", "system"]
                    },
                    "content": {
                        "type": "string",
                        "description": "Message content"
                    },
                    "tokens_approx": {
                        "type": "integer",
                        "description": "Approximate token count"
                    },
                    "tool_uses": {
                        "type": "array",
                        "description": "Tools used in this turn"
                    },
                },
                "required": ["session_id", "role", "content"],
            },
        ),
        Tool(
            name="exchange_list",
            description="Get exchanges from a session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "UUID of the session"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max exchanges to return"
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0
                    },
                },
                "required": ["session_id"],
            },
        ),
        # Pattern tools
        Tool(
            name="pattern_record",
            description="Record a new behavioral pattern.",
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Pattern type (topic_recurrence, preference, working_style, etc.)"
                    },
                    "description": {
                        "type": "string",
                        "description": "What the pattern is"
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Session IDs as evidence"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5
                    },
                },
                "required": ["type", "description"],
            },
        ),
        Tool(
            name="pattern_reinforce",
            description="Strengthen an existing pattern.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern_id": {
                        "type": "string",
                        "description": "UUID of the pattern"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session that supports this pattern"
                    },
                },
                "required": ["pattern_id"],
            },
        ),
        Tool(
            name="pattern_list",
            description="List patterns with filters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Filter by pattern type"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["emerging", "established", "fading", "archived"],
                        "description": "Filter by status"
                    },
                    "min_confidence": {
                        "type": "number",
                        "default": 0
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20
                    },
                },
            },
        ),
        Tool(
            name="pattern_search",
            description="Search patterns by description.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10
                    },
                },
                "required": ["query"],
            },
        ),
        # Insight tools
        Tool(
            name="insight_extract",
            description="Extract an insight from a session and create a belief in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Source session"
                    },
                    "content": {
                        "type": "string",
                        "description": "The insight/belief content"
                    },
                    "domain_path": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Domain classification"
                    },
                    "confidence": {
                        "type": "object",
                        "description": "Confidence dimensions",
                        "default": {"overall": 0.8}
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "role": {"type": "string"}
                            }
                        },
                        "description": "Entities to link"
                    },
                },
                "required": ["session_id", "content"],
            },
        ),
        Tool(
            name="insight_list",
            description="List insights extracted from a session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session to get insights from"
                    },
                },
                "required": ["session_id"],
            },
        ),
    ]


# ============================================================================
# Tool Implementations
# ============================================================================

def session_start(
    platform: str,
    project_context: str | None = None,
    external_room_id: str | None = None,
    claude_session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a new session."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO vkb_sessions (platform, project_context, external_room_id, claude_session_id, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
            """,
            (platform, project_context, external_room_id, claude_session_id, json.dumps(metadata or {}))
        )
        row = cur.fetchone()

        # Create source record for this session
        cur.execute(
            """
            INSERT INTO sources (type, session_id, title)
            VALUES ('conversation', %s, %s)
            """,
            (row["id"], f"Session: {row['id']}")
        )

        session = Session.from_row(dict(row))
        return {
            "success": True,
            "session": session.to_dict(),
        }


def session_end(
    session_id: str,
    summary: str | None = None,
    themes: list[str] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    """End a session."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE vkb_sessions
            SET status = %s, summary = COALESCE(%s, summary), themes = COALESCE(%s, themes), ended_at = NOW()
            WHERE id = %s
            RETURNING *
            """,
            (status, summary, themes, session_id)
        )
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Session not found: {session_id}"}

        session = Session.from_row(dict(row))
        return {
            "success": True,
            "session": session.to_dict(),
        }


def session_get(
    session_id: str,
    include_exchanges: bool = False,
    exchange_limit: int = 10,
) -> dict[str, Any]:
    """Get a session by ID."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM vkb_sessions_overview WHERE id = %s", (session_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Session not found: {session_id}"}

        session = Session.from_row(dict(row))
        result = {
            "success": True,
            "session": session.to_dict(),
        }

        if include_exchanges:
            cur.execute(
                """
                SELECT * FROM vkb_exchanges
                WHERE session_id = %s
                ORDER BY sequence DESC
                LIMIT %s
                """,
                (session_id, exchange_limit)
            )
            exchange_rows = cur.fetchall()
            result["exchanges"] = [
                Exchange.from_row(dict(r)).to_dict()
                for r in reversed(exchange_rows)  # Return in chronological order
            ]

        return result


def session_list(
    platform: str | None = None,
    project_context: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """List sessions."""
    with get_cursor() as cur:
        sql = "SELECT * FROM vkb_sessions_overview WHERE 1=1"
        params: list[Any] = []

        if platform:
            sql += " AND platform = %s"
            params.append(platform)

        if project_context:
            sql += " AND project_context = %s"
            params.append(project_context)

        if status:
            sql += " AND status = %s"
            params.append(status)

        sql += " ORDER BY started_at DESC LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        return {
            "success": True,
            "sessions": [Session.from_row(dict(r)).to_dict() for r in rows],
            "total_count": len(rows),
        }


def session_find_by_room(external_room_id: str) -> dict[str, Any]:
    """Find active session by external room ID."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM vkb_sessions
            WHERE external_room_id = %s AND status = 'active'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (external_room_id,)
        )
        row = cur.fetchone()

        if row:
            session = Session.from_row(dict(row))
            return {
                "success": True,
                "found": True,
                "session": session.to_dict(),
            }
        else:
            return {
                "success": True,
                "found": False,
                "session": None,
            }


def exchange_add(
    session_id: str,
    role: str,
    content: str,
    tokens_approx: int | None = None,
    tool_uses: list[dict] | None = None,
) -> dict[str, Any]:
    """Add an exchange to a session."""
    with get_cursor() as cur:
        # Get next sequence number
        cur.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 as next_seq FROM vkb_exchanges WHERE session_id = %s",
            (session_id,)
        )
        sequence = cur.fetchone()["next_seq"]

        cur.execute(
            """
            INSERT INTO vkb_exchanges (session_id, sequence, role, content, tokens_approx, tool_uses)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (session_id, sequence, role, content, tokens_approx, json.dumps(tool_uses or []))
        )
        row = cur.fetchone()

        exchange = Exchange.from_row(dict(row))
        return {
            "success": True,
            "exchange": exchange.to_dict(),
        }


def exchange_list(
    session_id: str,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Get exchanges from a session."""
    with get_cursor() as cur:
        sql = "SELECT * FROM vkb_exchanges WHERE session_id = %s ORDER BY sequence"
        params: list[Any] = [session_id]

        if limit:
            sql += " LIMIT %s OFFSET %s"
            params.extend([limit, offset])

        cur.execute(sql, params)
        rows = cur.fetchall()

        return {
            "success": True,
            "exchanges": [Exchange.from_row(dict(r)).to_dict() for r in rows],
            "total_count": len(rows),
        }


def pattern_record(
    type: str,
    description: str,
    evidence: list[str] | None = None,
    confidence: float = 0.5,
) -> dict[str, Any]:
    """Record a new pattern."""
    with get_cursor() as cur:
        # Convert evidence strings to UUIDs if needed
        evidence_uuids = [UUID(e) if isinstance(e, str) else e for e in (evidence or [])]

        cur.execute(
            """
            INSERT INTO vkb_patterns (type, description, evidence, confidence)
            VALUES (%s, %s, %s, %s)
            RETURNING *
            """,
            (type, description, evidence_uuids, confidence)
        )
        row = cur.fetchone()

        pattern = Pattern.from_row(dict(row))
        return {
            "success": True,
            "pattern": pattern.to_dict(),
        }


def pattern_reinforce(
    pattern_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Reinforce an existing pattern."""
    with get_cursor() as cur:
        # Get current pattern
        cur.execute("SELECT * FROM vkb_patterns WHERE id = %s", (pattern_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Pattern not found: {pattern_id}"}

        pattern = Pattern.from_row(dict(row))
        evidence = pattern.evidence

        # Add session to evidence if not already present
        if session_id:
            session_uuid = UUID(session_id)
            if session_uuid not in evidence:
                evidence.append(session_uuid)

        # Increase confidence (asymptotic to 1.0)
        new_confidence = min(0.99, pattern.confidence + (1 - pattern.confidence) * 0.1)

        # Update status if appropriate
        new_status = pattern.status.value
        if pattern.occurrence_count >= 4 and pattern.status.value == "emerging":
            new_status = "established"

        cur.execute(
            """
            UPDATE vkb_patterns
            SET evidence = %s, occurrence_count = occurrence_count + 1,
                confidence = %s, last_observed = NOW(), status = %s
            WHERE id = %s
            RETURNING *
            """,
            (evidence, new_confidence, new_status, pattern_id)
        )
        row = cur.fetchone()

        pattern = Pattern.from_row(dict(row))
        return {
            "success": True,
            "pattern": pattern.to_dict(),
        }


def pattern_list(
    type: str | None = None,
    status: str | None = None,
    min_confidence: float = 0,
    limit: int = 20,
) -> dict[str, Any]:
    """List patterns."""
    with get_cursor() as cur:
        sql = "SELECT * FROM vkb_patterns WHERE confidence >= %s"
        params: list[Any] = [min_confidence]

        if type:
            sql += " AND type = %s"
            params.append(type)

        if status:
            sql += " AND status = %s"
            params.append(status)

        sql += " ORDER BY occurrence_count DESC, confidence DESC LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        return {
            "success": True,
            "patterns": [Pattern.from_row(dict(r)).to_dict() for r in rows],
            "total_count": len(rows),
        }


def pattern_search(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Search patterns by description."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM vkb_patterns
            WHERE description ILIKE %s
            ORDER BY confidence DESC
            LIMIT %s
            """,
            (f"%{query}%", limit)
        )
        rows = cur.fetchall()

        return {
            "success": True,
            "patterns": [Pattern.from_row(dict(r)).to_dict() for r in rows],
            "total_count": len(rows),
        }


def insight_extract(
    session_id: str,
    content: str,
    domain_path: list[str] | None = None,
    confidence: dict[str, Any] | None = None,
    entities: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Extract an insight from a session and create a belief."""
    confidence_obj = DimensionalConfidence.from_dict(confidence or {"overall": 0.8})

    with get_cursor() as cur:
        # Get source for this session
        cur.execute("SELECT id FROM sources WHERE session_id = %s LIMIT 1", (session_id,))
        source_row = cur.fetchone()
        source_id = source_row["id"] if source_row else None

        # Create belief
        cur.execute(
            """
            INSERT INTO beliefs (content, confidence, domain_path, source_id, extraction_method)
            VALUES (%s, %s, %s, %s, 'conversation_extraction')
            RETURNING *
            """,
            (content, json.dumps(confidence_obj.to_dict()), domain_path or [], source_id)
        )
        belief_row = cur.fetchone()
        belief_id = belief_row["id"]

        # Link entities
        if entities:
            for entity in entities:
                cur.execute(
                    """
                    INSERT INTO entities (name, type)
                    VALUES (%s, %s)
                    ON CONFLICT (name, type) WHERE canonical_id IS NULL
                    DO UPDATE SET modified_at = NOW()
                    RETURNING id
                    """,
                    (entity["name"], entity.get("type", "concept"))
                )
                entity_id = cur.fetchone()["id"]

                cur.execute(
                    """
                    INSERT INTO belief_entities (belief_id, entity_id, role)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (belief_id, entity_id, entity.get("role", "subject"))
                )

        # Link to session
        cur.execute(
            """
            INSERT INTO vkb_session_insights (session_id, belief_id, extraction_method)
            VALUES (%s, %s, 'manual')
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            (session_id, belief_id)
        )
        insight_row = cur.fetchone()

        return {
            "success": True,
            "insight_id": str(insight_row["id"]) if insight_row else None,
            "belief_id": str(belief_id),
            "session_id": session_id,
        }


def insight_list(session_id: str) -> dict[str, Any]:
    """List insights from a session."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT si.*, b.content, b.confidence, b.domain_path, b.created_at as belief_created_at
            FROM vkb_session_insights si
            JOIN beliefs b ON si.belief_id = b.id
            WHERE si.session_id = %s
            ORDER BY si.extracted_at
            """,
            (session_id,)
        )
        rows = cur.fetchall()

        insights = []
        for row in rows:
            insights.append({
                "id": str(row["id"]),
                "session_id": str(row["session_id"]),
                "belief_id": str(row["belief_id"]),
                "extraction_method": row["extraction_method"],
                "extracted_at": row["extracted_at"].isoformat(),
                "belief": {
                    "content": row["content"],
                    "confidence": row["confidence"],
                    "domain_path": row["domain_path"],
                    "created_at": row["belief_created_at"].isoformat(),
                },
            })

        return {
            "success": True,
            "insights": insights,
            "total_count": len(insights),
        }


# ============================================================================
# Tool Router
# ============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        result: dict[str, Any]

        # Session tools
        if name == "session_start":
            result = session_start(
                platform=arguments["platform"],
                project_context=arguments.get("project_context"),
                external_room_id=arguments.get("external_room_id"),
                claude_session_id=arguments.get("claude_session_id"),
                metadata=arguments.get("metadata"),
            )
        elif name == "session_end":
            result = session_end(
                session_id=arguments["session_id"],
                summary=arguments.get("summary"),
                themes=arguments.get("themes"),
                status=arguments.get("status", "completed"),
            )
        elif name == "session_get":
            result = session_get(
                session_id=arguments["session_id"],
                include_exchanges=arguments.get("include_exchanges", False),
                exchange_limit=arguments.get("exchange_limit", 10),
            )
        elif name == "session_list":
            result = session_list(
                platform=arguments.get("platform"),
                project_context=arguments.get("project_context"),
                status=arguments.get("status"),
                limit=arguments.get("limit", 20),
            )
        elif name == "session_find_by_room":
            result = session_find_by_room(
                external_room_id=arguments["external_room_id"],
            )

        # Exchange tools
        elif name == "exchange_add":
            result = exchange_add(
                session_id=arguments["session_id"],
                role=arguments["role"],
                content=arguments["content"],
                tokens_approx=arguments.get("tokens_approx"),
                tool_uses=arguments.get("tool_uses"),
            )
        elif name == "exchange_list":
            result = exchange_list(
                session_id=arguments["session_id"],
                limit=arguments.get("limit"),
                offset=arguments.get("offset", 0),
            )

        # Pattern tools
        elif name == "pattern_record":
            result = pattern_record(
                type=arguments["type"],
                description=arguments["description"],
                evidence=arguments.get("evidence"),
                confidence=arguments.get("confidence", 0.5),
            )
        elif name == "pattern_reinforce":
            result = pattern_reinforce(
                pattern_id=arguments["pattern_id"],
                session_id=arguments.get("session_id"),
            )
        elif name == "pattern_list":
            result = pattern_list(
                type=arguments.get("type"),
                status=arguments.get("status"),
                min_confidence=arguments.get("min_confidence", 0),
                limit=arguments.get("limit", 20),
            )
        elif name == "pattern_search":
            result = pattern_search(
                query=arguments["query"],
                limit=arguments.get("limit", 10),
            )

        # Insight tools
        elif name == "insight_extract":
            result = insight_extract(
                session_id=arguments["session_id"],
                content=arguments["content"],
                domain_path=arguments.get("domain_path"),
                confidence=arguments.get("confidence"),
                entities=arguments.get("entities"),
            )
        elif name == "insight_list":
            result = insight_list(
                session_id=arguments["session_id"],
            )

        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

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
    except Exception as e:
        logger.exception(f"Unexpected error in tool {name}")
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": f"Internal error: {str(e)}",
        }))]


# ============================================================================
# Server Entry Point
# ============================================================================

def run() -> None:
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="Valence VKB MCP Server")
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
    args = parser.parse_args()

    # Health check mode
    if args.health_check:
        sys.exit(cli_health_check())

    logger.info("Valence VKB MCP server starting...")

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
