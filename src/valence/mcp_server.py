"""Unified MCP Server for Valence v2.

Provides 16 knowledge substrate tools directly without intermediary layers.

Tools (v2 substrate):
- Sources:     source_ingest, source_get, source_search, source_list
- Retrieval:   knowledge_search
- Articles:    article_get, article_create, article_compile, article_update,
               article_search, article_split, article_merge
- Provenance:  provenance_link, provenance_get, provenance_trace
- Entities:    entity_get, entity_search
- Contention:  contention_detect, contention_list, contention_resolve
- Admin:       admin_forget, admin_stats, admin_maintenance

Resources:
- valence://articles/recent - Recent compiled articles
- valence://stats           - Database statistics
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
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

from .core.db import DatabaseError as OurDatabaseError
from .core.db import get_cursor, init_schema
from .core.exceptions import DatabaseException, ValidationException
from .core.health import DatabaseStats, cli_health_check, startup_checks
from .core.response import ValenceResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("valence")


# ============================================================================
# Helper Functions
# ============================================================================


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context (event-loop aware).

    Handles:
    - No event loop (create new)
    - Existing but not running loop (use it)
    - Running loop (execute in thread pool to avoid blocking)
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in event loop - run in thread to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=60)
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists
        return asyncio.run(coro)


def _validate_enum(value: str, valid_values: list[str], param_name: str) -> dict[str, Any] | None:
    """Return error dict if value is not in valid_values, else None."""
    if value not in valid_values:
        return {
            "success": False,
            "error": f"Invalid {param_name} '{value}'. Must be one of: {', '.join(valid_values)}"
        }
    return None


# ============================================================================
# Source Tools (from core.sources)
# ============================================================================

from .core.sources import (  # noqa: E402
    RELIABILITY_DEFAULTS,
    VALID_SOURCE_TYPES,
    _compute_fingerprint,
    _row_to_dict,
)


def source_ingest(
    content: str,
    source_type: str,
    title: str | None = None,
    url: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Ingest a new source into the knowledge substrate."""
    if not content or not content.strip():
        return {"success": False, "error": "content must be non-empty"}

    if source_type not in VALID_SOURCE_TYPES:
        return {
            "success": False,
            "error": f"Invalid source_type '{source_type}'. Must be one of: {', '.join(sorted(VALID_SOURCE_TYPES))}",
        }

    fingerprint = _compute_fingerprint(content)
    reliability = RELIABILITY_DEFAULTS.get(source_type, 0.5)
    metadata_json = json.dumps(metadata or {})

    with get_cursor() as cur:
        cur.execute(
            "SELECT id FROM sources WHERE fingerprint = %s LIMIT 1",
            (fingerprint,),
        )
        existing = cur.fetchone()
        if existing:
            return {
                "success": False,
                "duplicate": True,
                "existing_id": str(existing["id"]),
                "error": "Duplicate source: fingerprint already exists",
            }

        cur.execute(
            """
            INSERT INTO sources (type, title, url, content, fingerprint, reliability,
                                 content_hash, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            RETURNING id, type, title, url, content, fingerprint, reliability,
                      content_hash, metadata, created_at
            """,
            (
                source_type,
                title,
                url,
                content,
                fingerprint,
                reliability,
                fingerprint,
                metadata_json,
            ),
        )
        row = cur.fetchone()

    source = _row_to_dict(row)
    return {"success": True, "source": source}


def source_get(source_id: str) -> dict[str, Any]:
    """Retrieve a source by UUID."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, type, title, url, content, fingerprint, reliability,
                   content_hash, metadata, created_at
            FROM sources
            WHERE id = %s
            """,
            (source_id,),
        )
        row = cur.fetchone()

    if not row:
        return {"success": False, "error": f"Source not found: {source_id}"}

    source = _row_to_dict(row)
    return {"success": True, "source": source}


def source_search(query: str, limit: int = 20) -> dict[str, Any]:
    """Full-text search over source content."""
    if not query or not query.strip():
        return {"success": False, "error": "query must be non-empty"}

    limit = max(1, min(int(limit), 200))

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, type, title, url, content, fingerprint, reliability,
                   content_hash, metadata, created_at,
                   ts_rank_cd(content_tsv, websearch_to_tsquery('english', %s)) AS rank
            FROM sources
            WHERE content_tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query, query, limit),
        )
        rows = cur.fetchall()

    sources = [_row_to_dict(row) for row in rows]
    return {"success": True, "sources": sources, "total_count": len(sources)}


def source_list(limit: int = 50) -> dict[str, Any]:
    """List recent sources."""
    limit = max(1, min(int(limit), 200))

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, type, title, url, fingerprint, reliability, created_at, metadata
            FROM sources
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()

    sources = []
    for row in rows:
        d = dict(row)
        d["id"] = str(d["id"])
        if d.get("created_at"):
            d["created_at"] = d["created_at"].isoformat()
        sources.append(d)

    return {"success": True, "sources": sources, "total_count": len(sources)}


# ============================================================================
# Retrieval Tool (from core.retrieval)
# ============================================================================


def knowledge_search(
    query: str,
    limit: int = 10,
    include_sources: bool = False,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Unified knowledge retrieval."""
    if not query or not query.strip():
        return {"success": False, "error": "query must be non-empty"}

    limit = max(1, min(int(limit), 200))

    try:
        from .core.retrieval import retrieve

        results = _run_async(
            retrieve(query, limit=limit, include_sources=include_sources, session_id=session_id)
        )

    except Exception as exc:
        logger.exception("knowledge_search failed for query %r: %s", query, exc)
        return {"success": False, "error": str(exc)}

    # retrieve() returns ValenceResponse — unwrap to MCP dict format
    if isinstance(results, ValenceResponse):
        if not results.success:
            return {"success": False, "error": results.error}
        result_list = results.data or []
    else:
        result_list = results  # backward compat

    return {
        "success": True,
        "results": result_list,
        "total_count": len(result_list),
        "query": query,
    }


# ============================================================================
# Article Tools (from core.articles)
# ============================================================================


def article_create(
    content: str,
    title: str | None = None,
    source_ids: list[str] | None = None,
    author_type: str = "system",
    domain_path: list[str] | None = None,
) -> dict[str, Any]:
    """Create a new article."""
    from .core.articles import create_article

    result = _run_async(
        create_article(
            content=content,
            title=title,
            source_ids=source_ids,
            author_type=author_type,
            domain_path=domain_path,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "article": result.data}


def article_get(
    article_id: str,
    include_provenance: bool = False,
) -> dict[str, Any]:
    """Get an article by ID."""
    from .core.articles import get_article

    result = _run_async(get_article(article_id=article_id, include_provenance=include_provenance))
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "article": result.data}


def article_update(
    article_id: str,
    content: str,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Update an article's content."""
    from .core.articles import update_article

    result = _run_async(
        update_article(
            article_id=article_id,
            content=content,
            source_id=source_id,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "article": result.data}


def article_search(
    query: str,
    limit: int = 20,
    domain_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Search articles by content."""
    from .core.articles import search_articles

    result = _run_async(
        search_articles(
            query=query,
            limit=limit,
            domain_filter=domain_filter,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    articles = result.data or []
    return {
        "success": True,
        "articles": articles,
        "count": len(articles),
    }


def article_compile(
    source_ids: list[str],
    title_hint: str | None = None,
) -> dict[str, Any]:
    """Compile one or more sources into a new knowledge article."""
    if not source_ids:
        return {"success": False, "error": "source_ids must be a non-empty list"}

    try:
        from .core.compilation import compile_article

        result = _run_async(compile_article(source_ids=source_ids, title_hint=title_hint))
        if not result.success:
            return {"success": False, "error": result.error}
        out: dict[str, Any] = {"success": True, "article": result.data}
        if result.degraded:
            out["degraded"] = True
        return out
    except Exception as exc:
        logger.exception("article_compile failed")
        return {"success": False, "error": str(exc)}


def article_split(article_id: str) -> dict[str, Any]:
    """Split an oversized article into two smaller articles."""
    if not article_id:
        return {"success": False, "error": "article_id is required"}

    try:
        from .core.articles import split_article

        result = _run_async(split_article(article_id=article_id))
        if not result.success:
            return {"success": False, "error": result.error}
        return {"success": True, **result.data}
    except Exception as exc:
        logger.exception("article_split failed")
        return {"success": False, "error": str(exc)}


def article_merge(article_id_a: str, article_id_b: str) -> dict[str, Any]:
    """Merge two related articles into one."""
    if not article_id_a or not article_id_b:
        return {"success": False, "error": "Both article_id_a and article_id_b are required"}

    try:
        from .core.articles import merge_articles

        result = _run_async(merge_articles(article_id_a=article_id_a, article_id_b=article_id_b))
        if not result.success:
            return {"success": False, "error": result.error}
        return {"success": True, "merged_article": result.data}
    except Exception as exc:
        logger.exception("article_merge failed")
        return {"success": False, "error": str(exc)}


# ============================================================================
# Provenance Tools (from core.provenance)
# ============================================================================


def provenance_link(
    article_id: str,
    source_id: str,
    relationship: str = "confirms",
) -> dict[str, Any]:
    """Link a source to an article with a provenance relationship."""
    from .core.provenance import link_source

    result = _run_async(
        link_source(
            article_id=article_id,
            source_id=source_id,
            relationship=relationship,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "link": result.data}


def provenance_get(article_id: str) -> dict[str, Any]:
    """Get all provenance sources for an article."""
    from .core.provenance import get_provenance

    result = _run_async(get_provenance(article_id=article_id))
    if not result.success:
        return {"success": False, "error": result.error}
    provenance = result.data or []
    return {
        "success": True,
        "provenance": provenance,
        "count": len(provenance),
    }


def provenance_trace(article_id: str, claim_text: str) -> dict[str, Any]:
    """Trace which sources likely contributed a specific claim."""
    from .core.provenance import trace_claim

    result = _run_async(trace_claim(article_id=article_id, claim_text=claim_text))
    if not result.success:
        return {"success": False, "error": result.error}
    sources = result.data or []
    return {
        "success": True,
        "sources": sources,
        "count": len(sources),
    }


# ============================================================================
# Entity Tools (if core.entities exists)
# ============================================================================


def entity_get(entity_id: str) -> dict[str, Any]:
    """Get an entity by ID."""
    try:
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT id, type, name, canonical_id, aliases, metadata, created_at
                FROM entities
                WHERE id = %s
                """,
                (entity_id,),
            )
            row = cur.fetchone()

        if not row:
            return {"success": False, "error": f"Entity not found: {entity_id}"}

        entity = dict(row)
        entity["id"] = str(entity["id"])
        if entity.get("canonical_id"):
            entity["canonical_id"] = str(entity["canonical_id"])
        if entity.get("created_at"):
            entity["created_at"] = entity["created_at"].isoformat()

        return {"success": True, "entity": entity}
    except Exception as exc:
        logger.exception("entity_get failed")
        return {"success": False, "error": str(exc)}


def entity_search(query: str, entity_type: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Search entities by name."""
    if not query or not query.strip():
        return {"success": False, "error": "query must be non-empty"}

    limit = max(1, min(int(limit), 200))

    try:
        with get_cursor() as cur:
            if entity_type:
                cur.execute(
                    """
                    SELECT id, type, name, canonical_id, aliases, metadata, created_at
                    FROM entities
                    WHERE name ILIKE %s AND type = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (f"%{query}%", entity_type, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, type, name, canonical_id, aliases, metadata, created_at
                    FROM entities
                    WHERE name ILIKE %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (f"%{query}%", limit),
                )
            rows = cur.fetchall()

        entities = []
        for row in rows:
            e = dict(row)
            e["id"] = str(e["id"])
            if e.get("canonical_id"):
                e["canonical_id"] = str(e["canonical_id"])
            if e.get("created_at"):
                e["created_at"] = e["created_at"].isoformat()
            entities.append(e)

        return {"success": True, "entities": entities, "total_count": len(entities)}
    except Exception as exc:
        logger.exception("entity_search failed")
        return {"success": False, "error": str(exc)}


# ============================================================================
# Contention Tools (from core.contention)
# ============================================================================


def contention_detect(
    article_id: str,
    source_id: str,
) -> dict[str, Any]:
    """Detect a contention between an article and a source."""
    from .core.contention import detect_contention

    result = _run_async(
        detect_contention(
            article_id=article_id,
            source_id=source_id,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "contention": result.data}


def contention_list(
    article_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """List active contentions."""
    from .core.contention import list_contentions

    result = _run_async(
        list_contentions(
            article_id=article_id,
            status=status,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    contentions = result.data or []
    return {
        "success": True,
        "contentions": contentions,
        "total_count": len(contentions),
    }


def contention_resolve(
    contention_id: str,
    resolution: str,
    rationale: str,
) -> dict[str, Any]:
    """Resolve a contention."""
    from .core.contention import resolve_contention

    result = _run_async(
        resolve_contention(
            contention_id=contention_id,
            resolution=resolution,
            rationale=rationale,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "contention": result.data}


# ============================================================================
# Admin Tools (from core.forgetting, core.usage, core.compilation)
# ============================================================================


def admin_forget(target_type: str, target_id: str) -> dict[str, Any]:
    """Permanently remove a source or article."""
    if target_type not in ["source", "article"]:
        return {"success": False, "error": "target_type must be 'source' or 'article'"}

    try:
        if target_type == "source":
            from .core.forgetting import remove_source
            result = _run_async(remove_source(target_id))
        else:
            from .core.forgetting import remove_article
            result = _run_async(remove_article(target_id))

        if not result.success:
            return {"success": False, "error": result.error}
        return {"success": True, **result.data}
    except Exception as exc:
        logger.exception("admin_forget failed")
        return {"success": False, "error": str(exc)}


def admin_stats() -> dict[str, Any]:
    """Return health and capacity statistics."""
    try:
        stats = DatabaseStats.collect()
        return {"success": True, "stats": stats.to_dict()}
    except Exception as exc:
        logger.exception("admin_stats failed")
        return {"success": False, "error": str(exc)}


def admin_maintenance(
    recompute_scores: bool = False,
    process_queue: bool = False,
    evict_if_over_capacity: bool = False,
    evict_count: int = 10,
) -> dict[str, Any]:
    """Trigger maintenance operations."""
    results = {}

    try:
        if recompute_scores:
            from .core.usage import compute_usage_scores
            result = _run_async(compute_usage_scores())
            results["recompute_scores"] = result.success
            if not result.success:
                results["recompute_scores_error"] = result.error

        if process_queue:
            from .core.compilation import process_mutation_queue
            result = _run_async(process_mutation_queue())
            results["process_queue"] = result.success
            if not result.success:
                results["process_queue_error"] = result.error

        if evict_if_over_capacity:
            from .core.forgetting import evict_lowest
            result = _run_async(evict_lowest(count=evict_count))
            results["evict_if_over_capacity"] = result.success
            if not result.success:
                results["evict_error"] = result.error

        return {"success": True, "maintenance_results": results}
    except Exception as exc:
        logger.exception("admin_maintenance failed")
        return {"success": False, "error": str(exc)}


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
# Tool Definitions (MCP schema)
# ============================================================================

# Import tool definitions from the consolidated definitions file
# Must be after TOOL_HANDLERS is defined
from .mcp_tool_definitions import SUBSTRATE_TOOLS  # noqa: E402

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
                text=json.dumps({
                    "success": False,
                    "error": f"Validation error: {e.message}",
                    "details": e.details
                }),
            )
        ]
    except DatabaseException as e:
        logger.error(f"Database error in tool {name}: {e}")
        return [
            TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Database error: {e.message}"
                }),
            )
        ]
    except Exception as e:
        logger.exception(f"Unexpected error in tool {name}")
        return [
            TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Internal error: {str(e)}"
                }),
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

    # Initialize schema (tolerates already-existing tables/constraints)
    try:
        # Schema is now in root, not in substrate/
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
