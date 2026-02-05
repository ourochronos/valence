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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource, ResourceContents, TextResourceContents
from pydantic import AnyUrl

from ..core.db import get_cursor, init_schema, DatabaseStats
from ..core.models import Belief, Entity, Tension
from ..core.confidence import (
    DimensionalConfidence, 
    ConfidenceDimension,
    DEFAULT_WEIGHTS,
    confidence_label,
)
from ..core.health import startup_checks, cli_health_check
from ..core.exceptions import DatabaseException, ValidationException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("valence-substrate")


# ============================================================================
# Tool Definitions
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        # Belief tools
        Tool(
            name="belief_query",
            description="Search beliefs by content, domain, or entity. Uses hybrid search (keyword + semantic).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "domain_filter": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by domain path (e.g., ['tech', 'architecture'])"
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "Filter by related entity UUID"
                    },
                    "include_superseded": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include superseded beliefs"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum results"
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="belief_search",
            description="Semantic search for beliefs using vector embeddings. Best for finding conceptually related beliefs even with different wording.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to find semantically similar beliefs"
                    },
                    "min_similarity": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Minimum similarity threshold (0-1)"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Filter by minimum overall confidence"
                    },
                    "domain_filter": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by domain path"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum results"
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="belief_create",
            description="Create a new belief with optional entity links.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The belief content"
                    },
                    "confidence": {
                        "type": "object",
                        "description": "Confidence dimensions (or single 'overall' value)",
                        "default": {"overall": 0.7}
                    },
                    "domain_path": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Domain classification"
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["document", "conversation", "inference", "observation", "user_input"],
                        "description": "Type of source"
                    },
                    "source_ref": {
                        "type": "string",
                        "description": "Reference to source (URL, session_id, etc.)"
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "role": {"type": "string", "enum": ["subject", "object", "context"]}
                            },
                            "required": ["name"]
                        },
                        "description": "Entities to link (will be created if not exist)"
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="belief_supersede",
            description="Replace an old belief with a new one, maintaining history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "old_belief_id": {
                        "type": "string",
                        "description": "UUID of belief to supersede"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "Updated belief content"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this belief is being superseded"
                    },
                    "confidence": {
                        "type": "object",
                        "description": "Confidence for new belief"
                    },
                },
                "required": ["old_belief_id", "new_content", "reason"],
            },
        ),
        Tool(
            name="belief_get",
            description="Get a single belief by ID with full details.",
            inputSchema={
                "type": "object",
                "properties": {
                    "belief_id": {
                        "type": "string",
                        "description": "UUID of the belief"
                    },
                    "include_history": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include supersession chain"
                    },
                    "include_tensions": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include related tensions"
                    },
                },
                "required": ["belief_id"],
            },
        ),
        # Entity tools
        Tool(
            name="entity_get",
            description="Get entity details with optional beliefs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "UUID of the entity"
                    },
                    "include_beliefs": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include related beliefs"
                    },
                    "belief_limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max beliefs to include"
                    },
                },
                "required": ["entity_id"],
            },
        ),
        Tool(
            name="entity_search",
            description="Find entities by name or type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (matches name and aliases)"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["person", "organization", "tool", "concept", "project", "location", "service"],
                        "description": "Filter by entity type"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20
                    },
                },
                "required": ["query"],
            },
        ),
        # Tension tools
        Tool(
            name="tension_list",
            description="List contradictions/tensions between beliefs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["detected", "investigating", "resolved", "accepted"],
                        "description": "Filter by status"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Minimum severity"
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "Tensions involving this entity"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20
                    },
                },
            },
        ),
        Tool(
            name="tension_resolve",
            description="Mark a tension as resolved with explanation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tension_id": {
                        "type": "string",
                        "description": "UUID of the tension"
                    },
                    "resolution": {
                        "type": "string",
                        "description": "How the tension was resolved"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["supersede_a", "supersede_b", "keep_both", "archive_both"],
                        "description": "What to do with the beliefs"
                    },
                },
                "required": ["tension_id", "resolution", "action"],
            },
        ),
        # Trust tools
        Tool(
            name="trust_check",
            description="Check trust levels for entities or federation nodes on a specific topic/domain.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or domain to check trust for"
                    },
                    "entity_name": {
                        "type": "string",
                        "description": "Specific entity to check trust for"
                    },
                    "include_federated": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include federated node trust"
                    },
                    "min_trust": {
                        "type": "number",
                        "default": 0.3,
                        "description": "Minimum trust threshold"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10
                    },
                },
                "required": ["topic"],
            },
        ),
        # Confidence explanation tool
        Tool(
            name="confidence_explain",
            description="Explain why a belief has a particular confidence score, showing all contributing dimensions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "belief_id": {
                        "type": "string",
                        "description": "UUID of the belief to explain"
                    },
                },
                "required": ["belief_id"],
            },
        ),
    ]


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

    return [TextResourceContents(
        uri=AnyUrl(uri),
        mimeType="application/json",
        text=json.dumps(data, indent=2, default=str),
    )]


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
            (limit,)
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
            result["entities"].append({
                "id": str(row["id"]),
                "name": row["name"],
                "type": row["type"],
                "belief_count": row["belief_count"],
                "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] else None,
            })
        
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
# Tool Implementations
# ============================================================================

def belief_query(
    query: str,
    domain_filter: list[str] | None = None,
    entity_id: str | None = None,
    include_superseded: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    """Search beliefs."""
    with get_cursor() as cur:
        # Build query
        sql = """
            SELECT b.*, ts_rank(b.content_tsv, websearch_to_tsquery('english', %s)) as relevance
            FROM beliefs b
            WHERE b.content_tsv @@ websearch_to_tsquery('english', %s)
        """
        params: list[Any] = [query, query]

        if not include_superseded:
            sql += " AND b.status = 'active' AND b.superseded_by_id IS NULL"

        if domain_filter:
            sql += " AND b.domain_path && %s"
            params.append(domain_filter)

        if entity_id:
            sql += " AND EXISTS (SELECT 1 FROM belief_entities be WHERE be.belief_id = b.id AND be.entity_id = %s)"
            params.append(entity_id)

        sql += " ORDER BY relevance DESC LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        beliefs = []
        for row in rows:
            belief = Belief.from_row(dict(row))
            belief_dict = belief.to_dict()
            belief_dict["relevance_score"] = float(row.get("relevance", 0))
            beliefs.append(belief_dict)

        return {
            "success": True,
            "beliefs": beliefs,
            "total_count": len(beliefs),
        }


def belief_search(
    query: str,
    min_similarity: float = 0.5,
    min_confidence: float | None = None,
    domain_filter: list[str] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Semantic search for beliefs using embeddings."""
    try:
        from ..embeddings.service import generate_embedding, vector_to_pgvector
    except ImportError as e:
        return {
            "success": False,
            "error": "Embeddings service not available. Install openai package.",
        }
    
    try:
        # Generate query embedding
        query_vector = generate_embedding(query)
        query_str = vector_to_pgvector(query_vector)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate embedding: {str(e)}",
        }
    
    with get_cursor() as cur:
        # Build query with similarity filter
        sql = """
            SELECT b.*, 1 - (b.embedding <=> %s::vector) as similarity
            FROM beliefs b
            WHERE b.embedding IS NOT NULL
            AND b.status = 'active'
            AND 1 - (b.embedding <=> %s::vector) >= %s
        """
        params: list[Any] = [query_str, query_str, min_similarity]
        
        if min_confidence is not None:
            sql += " AND (b.confidence->>'overall')::numeric >= %s"
            params.append(min_confidence)
        
        if domain_filter:
            sql += " AND b.domain_path && %s"
            params.append(domain_filter)
        
        sql += " ORDER BY b.embedding <=> %s::vector LIMIT %s"
        params.extend([query_str, limit])
        
        cur.execute(sql, params)
        rows = cur.fetchall()
        
        beliefs = []
        for row in rows:
            belief = Belief.from_row(dict(row))
            belief_dict = belief.to_dict()
            belief_dict["similarity"] = float(row["similarity"])
            beliefs.append(belief_dict)
        
        return {
            "success": True,
            "beliefs": beliefs,
            "total_count": len(beliefs),
            "query_embedded": True,
        }


def belief_create(
    content: str,
    confidence: dict[str, Any] | None = None,
    domain_path: list[str] | None = None,
    source_type: str | None = None,
    source_ref: str | None = None,
    entities: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Create a new belief."""
    confidence_obj = DimensionalConfidence.from_dict(confidence or {"overall": 0.7})

    with get_cursor() as cur:
        # Create source if provided
        source_id = None
        if source_type:
            cur.execute(
                "INSERT INTO sources (type, url) VALUES (%s, %s) RETURNING id",
                (source_type, source_ref)
            )
            source_id = cur.fetchone()["id"]

        # Create belief
        cur.execute(
            """
            INSERT INTO beliefs (content, confidence, domain_path, source_id)
            VALUES (%s, %s, %s, %s)
            RETURNING *
            """,
            (content, json.dumps(confidence_obj.to_dict()), domain_path or [], source_id)
        )
        belief_row = cur.fetchone()
        belief_id = belief_row["id"]

        # Link entities
        if entities:
            for entity in entities:
                # Find or create entity
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

                # Link to belief
                cur.execute(
                    """
                    INSERT INTO belief_entities (belief_id, entity_id, role)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (belief_id, entity_id, entity.get("role", "subject"))
                )

        belief = Belief.from_row(dict(belief_row))
        return {
            "success": True,
            "belief": belief.to_dict(),
        }


def belief_supersede(
    old_belief_id: str,
    new_content: str,
    reason: str,
    confidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Supersede an existing belief."""
    with get_cursor() as cur:
        # Get old belief
        cur.execute("SELECT * FROM beliefs WHERE id = %s", (old_belief_id,))
        old_row = cur.fetchone()
        if not old_row:
            return {"success": False, "error": f"Belief not found: {old_belief_id}"}

        old_belief = Belief.from_row(dict(old_row))

        # Determine new confidence
        new_confidence = DimensionalConfidence.from_dict(
            confidence or old_belief.confidence.to_dict()
        )

        # Create new belief
        cur.execute(
            """
            INSERT INTO beliefs (content, confidence, domain_path, source_id, extraction_method, supersedes_id, valid_from)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            RETURNING *
            """,
            (
                new_content,
                json.dumps(new_confidence.to_dict()),
                old_belief.domain_path,
                str(old_belief.source_id) if old_belief.source_id else None,
                f"supersession: {reason}",
                old_belief_id,
            )
        )
        new_row = cur.fetchone()
        new_belief_id = new_row["id"]

        # Update old belief
        cur.execute(
            """
            UPDATE beliefs
            SET status = 'superseded', superseded_by_id = %s, valid_until = NOW(), modified_at = NOW()
            WHERE id = %s
            """,
            (new_belief_id, old_belief_id)
        )

        # Copy entity links
        cur.execute(
            """
            INSERT INTO belief_entities (belief_id, entity_id, role)
            SELECT %s, entity_id, role FROM belief_entities WHERE belief_id = %s
            """,
            (new_belief_id, old_belief_id)
        )

        new_belief = Belief.from_row(dict(new_row))
        return {
            "success": True,
            "old_belief_id": old_belief_id,
            "new_belief": new_belief.to_dict(),
            "reason": reason,
        }


def belief_get(
    belief_id: str,
    include_history: bool = False,
    include_tensions: bool = False,
) -> dict[str, Any]:
    """Get a belief by ID."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM beliefs WHERE id = %s", (belief_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Belief not found: {belief_id}"}

        belief = Belief.from_row(dict(row))
        belief_dict = belief.to_dict()
        result: dict[str, Any] = {
            "success": True,
            "belief": belief_dict,
        }

        # Load source
        if belief.source_id:
            cur.execute("SELECT * FROM sources WHERE id = %s", (str(belief.source_id),))
            source_row = cur.fetchone()
            if source_row:
                belief_dict["source"] = dict(source_row)

        # Load entities
        cur.execute(
            """
            SELECT e.*, be.role
            FROM entities e
            JOIN belief_entities be ON e.id = be.entity_id
            WHERE be.belief_id = %s
            """,
            (belief_id,)
        )
        entity_rows = cur.fetchall()
        belief_dict["entities"] = [
            {"entity": Entity.from_row(dict(r)).to_dict(), "role": r["role"]}
            for r in entity_rows
        ]

        # Load history if requested
        if include_history:
            history: list[dict[str, Any]] = []
            current_id: str | None = belief_id

            # Walk backwards through supersession chain
            while current_id:
                cur.execute(
                    "SELECT id, supersedes_id, created_at, extraction_method FROM beliefs WHERE id = %s",
                    (current_id,)
                )
                hist_row = cur.fetchone()
                if hist_row:
                    history.append({
                        "id": str(hist_row["id"]),
                        "created_at": hist_row["created_at"].isoformat(),
                        "reason": hist_row.get("extraction_method"),
                    })
                    current_id = str(hist_row["supersedes_id"]) if hist_row["supersedes_id"] else None
                else:
                    break

            result["history"] = list(reversed(history))

        # Load tensions if requested
        if include_tensions:
            cur.execute(
                """
                SELECT * FROM tensions
                WHERE (belief_a_id = %s OR belief_b_id = %s)
                AND status != 'resolved'
                """,
                (belief_id, belief_id)
            )
            tension_rows = cur.fetchall()
            result["tensions"] = [
                Tension.from_row(dict(r)).to_dict()
                for r in tension_rows
            ]

        return result


def entity_get(
    entity_id: str,
    include_beliefs: bool = False,
    belief_limit: int = 10,
) -> dict[str, Any]:
    """Get an entity by ID."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM entities WHERE id = %s", (entity_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Entity not found: {entity_id}"}

        entity = Entity.from_row(dict(row))
        result = {
            "success": True,
            "entity": entity.to_dict(),
        }

        if include_beliefs:
            cur.execute(
                """
                SELECT b.*, be.role
                FROM beliefs b
                JOIN belief_entities be ON b.id = be.belief_id
                WHERE be.entity_id = %s
                AND b.status = 'active'
                AND b.superseded_by_id IS NULL
                ORDER BY b.created_at DESC
                LIMIT %s
                """,
                (entity_id, belief_limit)
            )
            belief_rows = cur.fetchall()
            result["beliefs"] = [
                {**Belief.from_row(dict(r)).to_dict(), "role": r["role"]}
                for r in belief_rows
            ]

        return result


def entity_search(
    query: str,
    type: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Search for entities."""
    with get_cursor() as cur:
        sql = """
            SELECT * FROM entities
            WHERE (name ILIKE %s OR %s = ANY(aliases))
            AND canonical_id IS NULL
        """
        params: list[Any] = [f"%{query}%", query]

        if type:
            sql += " AND type = %s"
            params.append(type)

        sql += " ORDER BY name LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        return {
            "success": True,
            "entities": [Entity.from_row(dict(r)).to_dict() for r in rows],
            "total_count": len(rows),
        }


def tension_list(
    status: str | None = None,
    severity: str | None = None,
    entity_id: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """List tensions."""
    with get_cursor() as cur:
        sql = "SELECT * FROM tensions WHERE 1=1"
        params: list[Any] = []

        if status:
            sql += " AND status = %s"
            params.append(status)

        if severity:
            severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            min_severity = severity_order.get(severity, 2)
            sql += " AND CASE severity WHEN 'low' THEN 1 WHEN 'medium' THEN 2 WHEN 'high' THEN 3 WHEN 'critical' THEN 4 END >= %s"
            params.append(min_severity)

        if entity_id:
            sql += """
                AND (
                    EXISTS (SELECT 1 FROM belief_entities WHERE belief_id = belief_a_id AND entity_id = %s)
                    OR EXISTS (SELECT 1 FROM belief_entities WHERE belief_id = belief_b_id AND entity_id = %s)
                )
            """
            params.extend([entity_id, entity_id])

        sql += " ORDER BY detected_at DESC LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        tensions = []
        for row in rows:
            tension = Tension.from_row(dict(row))
            tension_dict = tension.to_dict()

            # Load belief content for context
            cur.execute("SELECT id, content FROM beliefs WHERE id IN (%s, %s)", (row["belief_a_id"], row["belief_b_id"]))
            belief_rows = cur.fetchall()
            belief_map = {str(r["id"]): r["content"] for r in belief_rows}
            tension_dict["belief_a_content"] = belief_map.get(str(row["belief_a_id"]))
            tension_dict["belief_b_content"] = belief_map.get(str(row["belief_b_id"]))

            tensions.append(tension_dict)

        return {
            "success": True,
            "tensions": tensions,
            "total_count": len(tensions),
        }


def tension_resolve(
    tension_id: str,
    resolution: str,
    action: str,
) -> dict[str, Any]:
    """Resolve a tension."""
    with get_cursor() as cur:
        # Get tension
        cur.execute("SELECT * FROM tensions WHERE id = %s", (tension_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Tension not found: {tension_id}"}

        belief_a_id = row["belief_a_id"]
        belief_b_id = row["belief_b_id"]

        # Perform action
        if action == "supersede_a":
            # Get belief B content and supersede A with it
            cur.execute("SELECT content FROM beliefs WHERE id = %s", (belief_b_id,))
            b_content = cur.fetchone()["content"]
            belief_supersede(str(belief_a_id), b_content, f"Tension resolution: {resolution}")

        elif action == "supersede_b":
            # Get belief A content and supersede B with it
            cur.execute("SELECT content FROM beliefs WHERE id = %s", (belief_a_id,))
            a_content = cur.fetchone()["content"]
            belief_supersede(str(belief_b_id), a_content, f"Tension resolution: {resolution}")

        elif action == "archive_both":
            cur.execute(
                "UPDATE beliefs SET status = 'archived', modified_at = NOW() WHERE id IN (%s, %s)",
                (belief_a_id, belief_b_id)
            )

        # Mark tension as resolved
        cur.execute(
            """
            UPDATE tensions
            SET status = %s, resolution = %s, resolved_at = NOW()
            WHERE id = %s
            """,
            ("resolved" if action != "keep_both" else "accepted", resolution, tension_id)
        )

        return {
            "success": True,
            "tension_id": tension_id,
            "action": action,
            "resolution": resolution,
        }


def trust_check(
    topic: str,
    entity_name: str | None = None,
    include_federated: bool = True,
    min_trust: float = 0.3,
    limit: int = 10,
) -> dict[str, Any]:
    """Check trust levels for a topic/domain."""
    result: dict[str, Any] = {
        "success": True,
        "topic": topic,
        "trusted_entities": [],
        "trusted_nodes": [],
    }
    
    with get_cursor() as cur:
        # Find entities that have high-confidence beliefs in this domain
        entity_sql = """
            SELECT e.id, e.name, e.type,
                   COUNT(b.id) as belief_count,
                   AVG((b.confidence->>'overall')::numeric) as avg_confidence,
                   MAX((b.confidence->>'overall')::numeric) as max_confidence
            FROM entities e
            JOIN belief_entities be ON e.id = be.entity_id
            JOIN beliefs b ON be.belief_id = b.id
            WHERE b.status = 'active'
            AND (
                b.domain_path && ARRAY[%s]
                OR b.content ILIKE %s
            )
        """
        params: list[Any] = [topic, f"%{topic}%"]
        
        if entity_name:
            entity_sql += " AND e.name ILIKE %s"
            params.append(f"%{entity_name}%")
        
        entity_sql += """
            GROUP BY e.id
            HAVING AVG((b.confidence->>'overall')::numeric) >= %s
            ORDER BY avg_confidence DESC, belief_count DESC
            LIMIT %s
        """
        params.extend([min_trust, limit])
        
        cur.execute(entity_sql, params)
        for row in cur.fetchall():
            result["trusted_entities"].append({
                "id": str(row["id"]),
                "name": row["name"],
                "type": row["type"],
                "belief_count": row["belief_count"],
                "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] else None,
                "max_confidence": float(row["max_confidence"]) if row["max_confidence"] else None,
                "trust_reason": f"Has {row['belief_count']} beliefs about {topic} with avg confidence {float(row['avg_confidence']):.2f}",
            })
        
        # Check federated node trust if enabled
        if include_federated:
            try:
                cur.execute(
                    """
                    SELECT fn.id, fn.name, fn.instance_url,
                           nt.trust, nt.beliefs_corroborated, nt.beliefs_disputed
                    FROM federation_nodes fn
                    JOIN node_trust nt ON fn.id = nt.node_id
                    WHERE fn.status = 'active'
                    AND (nt.trust->>'overall')::numeric >= %s
                    ORDER BY (nt.trust->>'overall')::numeric DESC
                    LIMIT %s
                    """,
                    (min_trust, limit)
                )
                for row in cur.fetchall():
                    trust_score = row["trust"].get("overall", 0) if row["trust"] else 0
                    result["trusted_nodes"].append({
                        "id": str(row["id"]),
                        "name": row["name"],
                        "instance_url": row["instance_url"],
                        "trust_score": trust_score,
                        "beliefs_corroborated": row["beliefs_corroborated"],
                        "beliefs_disputed": row["beliefs_disputed"],
                        "trust_reason": f"Corroborated {row['beliefs_corroborated']} beliefs, disputed {row['beliefs_disputed']}",
                    })
            except Exception as e:
                logger.debug(f"Federation tables may not exist: {e}")
    
    return result


def confidence_explain(belief_id: str) -> dict[str, Any]:
    """Explain confidence score for a belief."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM beliefs WHERE id = %s", (belief_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Belief not found: {belief_id}"}
        
        belief = Belief.from_row(dict(row))
        conf = belief.confidence
        
        # Build explanation
        dimensions_dict: dict[str, Any] = {}
        weights_used: dict[str, float] = {}
        explanation: dict[str, Any] = {
            "success": True,
            "belief_id": belief_id,
            "content_preview": belief.content[:100] + "..." if len(belief.content) > 100 else belief.content,
            "overall_confidence": conf.overall,
            "overall_label": confidence_label(conf.overall),
            "dimensions": dimensions_dict,
            "computation_method": "weighted_geometric_mean",
            "weights_used": weights_used,
        }
        
        # Document each dimension
        dimension_explanations = {
            "source_reliability": "How trustworthy is the information source? Higher for verified sources, lower for hearsay.",
            "method_quality": "How rigorous was the method of acquiring this knowledge? Higher for systematic analysis, lower for casual observation.",
            "internal_consistency": "Does this belief align with other beliefs? Higher if consistent, lower if it contradicts known facts.",
            "temporal_freshness": "How recent is this information? Higher for fresh data, decays over time.",
            "corroboration": "Is this supported by multiple independent sources? Higher with more confirmation.",
            "domain_applicability": "How relevant is this to the current context/domain? Higher if directly applicable.",
        }
        
        for dim in ConfidenceDimension:
            if dim == ConfidenceDimension.OVERALL:
                continue
            
            value = getattr(conf, dim.value, None)
            if value is not None:
                weight = DEFAULT_WEIGHTS.get(dim, 0)
                dimensions_dict[dim.value] = {
                    "value": value,
                    "label": confidence_label(value),
                    "weight": weight,
                    "explanation": dimension_explanations.get(dim.value, ""),
                }
                weights_used[dim.value] = weight
        
        # Add recommendations
        recommendations = []
        if conf.source_reliability is not None and conf.source_reliability < 0.5:
            recommendations.append("Consider verifying the source or finding corroborating evidence")
        if conf.corroboration is not None and conf.corroboration < 0.3:
            recommendations.append("This belief has low corroboration - seek additional sources")
        if conf.temporal_freshness is not None and conf.temporal_freshness < 0.5:
            recommendations.append("This information may be outdated - consider refreshing")
        if conf.internal_consistency is not None and conf.internal_consistency < 0.5:
            recommendations.append("This belief may conflict with other knowledge - review tensions")
        
        if recommendations:
            explanation["recommendations"] = recommendations
        else:
            explanation["recommendations"] = ["Confidence dimensions are balanced - no immediate concerns"]
        
        # Check for trust annotations
        try:
            cur.execute(
                """
                SELECT type, confidence_delta, created_at
                FROM belief_trust_annotations
                WHERE belief_id = %s
                AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY created_at DESC
                """,
                (belief_id,)
            )
            annotations = cur.fetchall()
            if annotations:
                explanation["trust_annotations"] = [
                    {
                        "type": a["type"],
                        "confidence_delta": float(a["confidence_delta"]),
                        "created_at": a["created_at"].isoformat(),
                    }
                    for a in annotations
                ]
        except Exception as e:
            logger.debug(f"Trust annotations table may not exist: {e}")
        
        return explanation


# ============================================================================
# Tool Router
# ============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        result: dict[str, Any]

        if name == "belief_query":
            result = belief_query(
                query=arguments["query"],
                domain_filter=arguments.get("domain_filter"),
                entity_id=arguments.get("entity_id"),
                include_superseded=arguments.get("include_superseded", False),
                limit=arguments.get("limit", 20),
            )
        elif name == "belief_search":
            result = belief_search(
                query=arguments["query"],
                min_similarity=arguments.get("min_similarity", 0.5),
                min_confidence=arguments.get("min_confidence"),
                domain_filter=arguments.get("domain_filter"),
                limit=arguments.get("limit", 10),
            )
        elif name == "belief_create":
            result = belief_create(
                content=arguments["content"],
                confidence=arguments.get("confidence"),
                domain_path=arguments.get("domain_path"),
                source_type=arguments.get("source_type"),
                source_ref=arguments.get("source_ref"),
                entities=arguments.get("entities"),
            )
        elif name == "belief_supersede":
            result = belief_supersede(
                old_belief_id=arguments["old_belief_id"],
                new_content=arguments["new_content"],
                reason=arguments["reason"],
                confidence=arguments.get("confidence"),
            )
        elif name == "belief_get":
            result = belief_get(
                belief_id=arguments["belief_id"],
                include_history=arguments.get("include_history", False),
                include_tensions=arguments.get("include_tensions", False),
            )
        elif name == "entity_get":
            result = entity_get(
                entity_id=arguments["entity_id"],
                include_beliefs=arguments.get("include_beliefs", False),
                belief_limit=arguments.get("belief_limit", 10),
            )
        elif name == "entity_search":
            result = entity_search(
                query=arguments["query"],
                type=arguments.get("type"),
                limit=arguments.get("limit", 20),
            )
        elif name == "tension_list":
            result = tension_list(
                status=arguments.get("status"),
                severity=arguments.get("severity"),
                entity_id=arguments.get("entity_id"),
                limit=arguments.get("limit", 20),
            )
        elif name == "tension_resolve":
            result = tension_resolve(
                tension_id=arguments["tension_id"],
                resolution=arguments["resolution"],
                action=arguments["action"],
            )
        elif name == "trust_check":
            result = trust_check(
                topic=arguments["topic"],
                entity_name=arguments.get("entity_name"),
                include_federated=arguments.get("include_federated", True),
                min_trust=arguments.get("min_trust", 0.3),
                limit=arguments.get("limit", 10),
            )
        elif name == "confidence_explain":
            result = confidence_explain(
                belief_id=arguments["belief_id"],
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
    parser = argparse.ArgumentParser(description="Valence Substrate MCP Server")
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
