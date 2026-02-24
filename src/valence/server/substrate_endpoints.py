# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""REST API endpoints for the Valence knowledge substrate (v2).

Provides RESTful access to articles, sources, entities, contentions, and stats.
Legacy /beliefs and /tensions routes delegate to v2 tool functions for backward
compatibility. New clients should use /articles and /sources endpoints instead.
"""

from __future__ import annotations

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .auth_helpers import authenticate, require_scope
from .endpoint_utils import _parse_bool, _parse_float, _parse_int, format_response, parse_output_format
from .errors import internal_error, invalid_json_error, missing_field_error, validation_error
from .formatters import format_conflicts_text, format_stats_text

logger = logging.getLogger(__name__)


# =============================================================================
# BELIEFS
# =============================================================================


async def beliefs_list_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/beliefs — Search beliefs by keyword/domain/entity."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    query = request.query_params.get("query")
    if not query:
        return missing_field_error("query")

    try:
        from ..mcp.handlers.articles import article_search

        domain_filter = None
        df_raw = request.query_params.get("domain_filter")
        if df_raw:
            domain_filter = [s.strip() for s in df_raw.split(",")]

        ranking = None
        ranking_raw = request.query_params.get("ranking")
        if ranking_raw:
            try:
                ranking = json.loads(ranking_raw)
                if not isinstance(ranking, dict):
                    return validation_error("ranking must be a JSON object")
            except json.JSONDecodeError:
                return validation_error("ranking must be valid JSON")

        result = article_search(
            query=query,
            domain_filter=domain_filter,
            limit=_parse_int(request.query_params.get("limit"), 20, 100),
        )
        return JSONResponse(result)
    except Exception:
        logger.exception("Error querying beliefs")
        return internal_error()


async def beliefs_create_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/beliefs — Create a new belief."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:write"):
        return err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    content = body.get("content")
    if not content:
        return missing_field_error("content")

    try:
        from ..mcp.handlers.articles import article_create

        result = article_create(
            content=content,
            title=body.get("title"),
            source_ids=body.get("source_ids"),
            domain_path=body.get("domain_path"),
        )
        status_code = 201 if result.get("success") else 400
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception("Error creating belief")
        return internal_error()


async def beliefs_search_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/beliefs/search — Semantic search via embeddings."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    query = request.query_params.get("query")
    if not query:
        return missing_field_error("query")

    try:
        from ..mcp.handlers.articles import article_search

        domain_filter = None
        df_raw = request.query_params.get("domain_filter")
        if df_raw:
            domain_filter = [s.strip() for s in df_raw.split(",")]

        ranking = None
        ranking_raw = request.query_params.get("ranking")
        if ranking_raw:
            try:
                ranking = json.loads(ranking_raw)
                if not isinstance(ranking, dict):
                    return validation_error("ranking must be a JSON object")
            except json.JSONDecodeError:
                return validation_error("ranking must be valid JSON")

        result = article_search(
            query=query,
            domain_filter=domain_filter,
            limit=_parse_int(request.query_params.get("limit"), 10, 100),
        )
        return JSONResponse(result)
    except Exception:
        logger.exception("Error searching beliefs")
        return internal_error()


async def beliefs_get_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/beliefs/{belief_id} — Get a belief by ID."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    belief_id = request.path_params.get("belief_id")
    if not belief_id:
        return missing_field_error("belief_id")

    try:
        from ..mcp.handlers.articles import article_get

        result = article_get(
            article_id=belief_id,
            include_provenance=_parse_bool(request.query_params.get("include_provenance")),
        )
        status_code = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error getting belief {belief_id}")
        return internal_error()


async def beliefs_supersede_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/beliefs/{belief_id}/supersede — Replace a belief."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:write"):
        return err

    belief_id = request.path_params.get("belief_id")
    if not belief_id:
        return missing_field_error("belief_id")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    new_content = body.get("new_content")
    if not new_content:
        return missing_field_error("new_content")

    reason = body.get("reason")
    if not reason:
        return missing_field_error("reason")

    try:
        from ..mcp.handlers.articles import article_update

        result = article_update(
            article_id=belief_id,
            content=new_content,
        )
        status_code = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error superseding belief {belief_id}")
        return internal_error()


# =============================================================================
# ENTITIES
# =============================================================================


async def entities_list_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/entities — Search entities by name or type."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    query = request.query_params.get("query")
    if not query:
        return missing_field_error("query")

    try:
        from ..mcp.handlers.entities import entity_search

        result = entity_search(
            query=query,
            entity_type=request.query_params.get("type"),
            limit=_parse_int(request.query_params.get("limit"), 20, 100),
        )
        return JSONResponse(result)
    except Exception:
        logger.exception("Error searching entities")
        return internal_error()


async def entities_get_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/entities/{id} — Get entity by ID."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    entity_id = request.path_params.get("id")
    if not entity_id:
        return missing_field_error("id")

    try:
        from ..mcp.handlers.entities import entity_get

        result = entity_get(
            entity_id=entity_id,
            include_beliefs=_parse_bool(request.query_params.get("include_beliefs")),
            belief_limit=_parse_int(request.query_params.get("belief_limit"), 10, 100),
        )
        status_code = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error getting entity {entity_id}")
        return internal_error()


# =============================================================================
# TENSIONS
# =============================================================================


async def tensions_list_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/tensions — List tensions/contradictions."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    try:
        from ..mcp.handlers.contention import contention_list

        result = contention_list(
            article_id=request.query_params.get("article_id"),
            status=request.query_params.get("status", "detected"),
        )
        return JSONResponse(result)
    except Exception:
        logger.exception("Error listing tensions")
        return internal_error()


async def tensions_resolve_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/tensions/{id}/resolve — Resolve a tension."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:write"):
        return err

    tension_id = request.path_params.get("id")
    if not tension_id:
        return missing_field_error("id")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    resolution = body.get("resolution")
    if not resolution:
        return missing_field_error("resolution")

    action = body.get("action")
    if not action:
        return missing_field_error("action")

    try:
        from ..mcp.handlers.contention import contention_resolve

        result = contention_resolve(
            contention_id=tension_id,
            resolution=action,
            rationale=resolution,
        )
        status_code = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error resolving tension {tension_id}")
        return internal_error()


# =============================================================================
# STATS
# =============================================================================


async def stats_endpoint(request: Request) -> Response:
    """GET /api/v1/stats — Aggregate database statistics."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    output_format = parse_output_format(request)

    try:
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM articles")
            total = cur.fetchone()["total"]

            cur.execute("SELECT COUNT(*) as active FROM articles WHERE status = 'active' AND superseded_by_id IS NULL")
            active = cur.fetchone()["active"]

            cur.execute("SELECT COUNT(*) as with_emb FROM articles WHERE embedding IS NOT NULL")
            with_embedding = cur.fetchone()["with_emb"]

            cur.execute("SELECT COUNT(*) as cnt FROM contentions WHERE status = 'detected'")
            unresolved_contentions = cur.fetchone()["cnt"]

            try:
                cur.execute("SELECT COUNT(DISTINCT d) as count FROM articles, LATERAL unnest(domain_path) as d")
                domains = cur.fetchone()["count"]
            except Exception:
                domains = 0

            cur.execute("SELECT COUNT(*) as cnt FROM sources")
            source_count = cur.fetchone()["cnt"]

        result = {
            "success": True,
            "stats": {
                "total_articles": total,
                "active_articles": active,
                "total_sources": source_count,
                "with_embeddings": with_embedding,
                "unique_domains": domains,
                "unresolved_contentions": unresolved_contentions,
            },
        }
        return format_response(result, output_format, text_formatter=format_stats_text)
    except Exception:
        logger.exception("Error getting stats")
        return internal_error()


# =============================================================================
# CONFLICT DETECTION
# =============================================================================


async def conflicts_endpoint(request: Request) -> Response:
    """GET /api/v1/beliefs/conflicts — Detect potential contradictions."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    threshold = _parse_float(request.query_params.get("threshold"), 0.85) or 0.85
    auto_record = _parse_bool(request.query_params.get("auto_record"))
    output_format = parse_output_format(request)

    try:
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute(
                """
                WITH article_pairs AS (
                    SELECT
                        a1.id as id_a, a1.content as content_a, a1.confidence as confidence_a,
                        a2.id as id_b, a2.content as content_b, a2.confidence as confidence_b,
                        1 - (a1.embedding <=> a2.embedding) as similarity
                    FROM articles a1
                    CROSS JOIN articles a2
                    WHERE a1.id < a2.id
                      AND a1.embedding IS NOT NULL AND a2.embedding IS NOT NULL
                      AND a1.status = 'active' AND a2.status = 'active'
                      AND a1.superseded_by_id IS NULL AND a2.superseded_by_id IS NULL
                      AND 1 - (a1.embedding <=> a2.embedding) > %s
                    ORDER BY similarity DESC
                    LIMIT 50
                )
                SELECT * FROM article_pairs
                WHERE NOT EXISTS (
                    SELECT 1 FROM contentions t
                    WHERE (t.article_id = article_pairs.id_a AND t.related_article_id = article_pairs.id_b)
                       OR (t.article_id = article_pairs.id_b AND t.related_article_id = article_pairs.id_a)
                )
                """,
                (threshold,),
            )
            pairs = cur.fetchall()

            negation_words = {
                "not",
                "never",
                "no",
                "n't",
                "cannot",
                "without",
                "neither",
                "none",
                "nobody",
                "nothing",
                "nowhere",
                "false",
                "incorrect",
                "wrong",
                "fail",
                "reject",
                "deny",
                "refuse",
                "avoid",
            }
            opposites = [
                ("good", "bad"),
                ("right", "wrong"),
                ("true", "false"),
                ("should", "should not"),
                ("always", "never"),
                ("prefer", "avoid"),
                ("like", "dislike"),
                ("works", "fails"),
                ("correct", "incorrect"),
            ]

            conflicts = []
            for pair in pairs:
                content_a = pair["content_a"].lower()
                content_b = pair["content_b"].lower()
                words_a = set(content_a.split())
                words_b = set(content_b.split())

                neg_a = bool(words_a & negation_words)
                neg_b = bool(words_b & negation_words)

                conflict_signal = 0.0
                reason = []

                if neg_a != neg_b:
                    conflict_signal += 0.4
                    reason.append("negation asymmetry")

                for pos, neg in opposites:
                    if (pos in content_a and neg in content_b) or (neg in content_a and pos in content_b):
                        conflict_signal += 0.3
                        reason.append(f"opposite: {pos}/{neg}")
                        break

                if conflict_signal > 0.2 or pair["similarity"] > 0.92:
                    conflicts.append(
                        {
                            "id_a": str(pair["id_a"]),
                            "content_a": pair["content_a"],
                            "id_b": str(pair["id_b"]),
                            "content_b": pair["content_b"],
                            "similarity": float(pair["similarity"]),
                            "conflict_score": conflict_signal + (float(pair["similarity"]) - threshold) * 0.5,
                            "reason": ", ".join(reason) if reason else "high similarity",
                        }
                    )

            conflicts.sort(key=lambda x: x["conflict_score"], reverse=True)

            recorded = []
            if auto_record and conflicts:
                for c in conflicts:
                    cur.execute(
                        """
                        INSERT INTO contentions (article_id, related_article_id, type, description, severity)
                        VALUES (%s, %s, 'contradiction', %s, %s)
                        ON CONFLICT DO NOTHING
                        RETURNING id
                        """,
                        (
                            c["id_a"],
                            c["id_b"],
                            f"Auto-detected: {c['reason']}",
                            "high" if c["conflict_score"] > 0.5 else "medium",
                        ),
                    )
                    row = cur.fetchone()
                    if row:
                        recorded.append(str(row["id"]))

        result = {
            "success": True,
            "conflicts": conflicts,
            "count": len(conflicts),
            "threshold": threshold,
            "recorded_contentions": recorded,
        }
        return format_response(result, output_format, text_formatter=format_conflicts_text)
    except Exception:
        logger.exception("Error detecting conflicts")
        return internal_error()
