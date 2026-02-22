"""REST API endpoints for the Valence knowledge substrate.

Provides RESTful access to beliefs, entities, tensions, confidence, and trust.
All endpoints require authentication and scope-based authorization.
"""

from __future__ import annotations

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

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
        from ..substrate.tools.articles import article_search

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
        from ..substrate.tools.articles import article_create

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
        from ..substrate.tools.articles import article_search

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
        from ..substrate.tools.articles import article_get

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
        from ..substrate.tools.articles import article_create
        from ..substrate.tools.articles import provenance_link

        # Create new article
        create_result = article_create(
            content=new_content,
            title=body.get("title"),
        )
        
        if not create_result.get("success"):
            return JSONResponse(create_result, status_code=400)
        
        new_article_id = create_result["article"]["id"]
        
        # Link with supersedes relationship
        link_result = provenance_link(
            article_id=new_article_id,
            source_id=belief_id,
            relationship="supersedes",
        )
        
        result = {
            "success": True,
            "old_belief_id": belief_id,
            "new_article": create_result["article"],
            "reason": reason,
        }
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
        from ..substrate.tools.entities import entity_search

        result = entity_search(
            query=query,
            type=request.query_params.get("type"),
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
        from ..substrate.tools.entities import entity_get

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
        from ..substrate.tools.contention import contention_list

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
        from ..substrate.tools.contention import contention_resolve

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


async def stats_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/stats — Aggregate database statistics."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    output_format = parse_output_format(request)

    try:
        from our_db import get_cursor

        with get_cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM beliefs")
            total = cur.fetchone()["total"]

            cur.execute("SELECT COUNT(*) as active FROM beliefs WHERE status = 'active' AND superseded_by_id IS NULL")
            active = cur.fetchone()["active"]

            cur.execute("SELECT COUNT(*) as with_emb FROM beliefs WHERE embedding IS NOT NULL")
            with_embedding = cur.fetchone()["with_emb"]

            cur.execute("SELECT COUNT(*) as tensions FROM tensions WHERE status = 'detected'")
            tensions = cur.fetchone()["tensions"]

            try:
                cur.execute("SELECT COUNT(DISTINCT d) as count FROM beliefs, LATERAL unnest(domain_path) as d")
                domains = cur.fetchone()["count"]
            except Exception:
                domains = 0

            try:
                cur.execute("SELECT COUNT(*) as federated FROM beliefs WHERE is_local = FALSE")
                federated = cur.fetchone()["federated"]
            except Exception:
                federated = 0

        result = {
            "success": True,
            "stats": {
                "total_beliefs": total,
                "active_beliefs": active,
                "local_beliefs": active - federated,
                "federated_beliefs": federated,
                "with_embeddings": with_embedding,
                "unique_domains": domains,
                "unresolved_tensions": tensions,
            },
        }
        return format_response(result, output_format, text_formatter=format_stats_text)
    except Exception:
        logger.exception("Error getting stats")
        return internal_error()


# =============================================================================
# CONFLICT DETECTION
# =============================================================================


async def conflicts_endpoint(request: Request) -> JSONResponse:
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
        from our_db import get_cursor

        with get_cursor() as cur:
            cur.execute(
                """
                WITH belief_pairs AS (
                    SELECT
                        b1.id as id_a, b1.content as content_a, b1.confidence as confidence_a,
                        b2.id as id_b, b2.content as content_b, b2.confidence as confidence_b,
                        1 - (b1.embedding <=> b2.embedding) as similarity
                    FROM beliefs b1
                    CROSS JOIN beliefs b2
                    WHERE b1.id < b2.id
                      AND b1.embedding IS NOT NULL AND b2.embedding IS NOT NULL
                      AND b1.status = 'active' AND b2.status = 'active'
                      AND b1.superseded_by_id IS NULL AND b2.superseded_by_id IS NULL
                      AND 1 - (b1.embedding <=> b2.embedding) > %s
                    ORDER BY similarity DESC
                    LIMIT 50
                )
                SELECT * FROM belief_pairs
                WHERE NOT EXISTS (
                    SELECT 1 FROM contentions t
                    WHERE (t.article_id = belief_pairs.id_a AND t.related_article_id = belief_pairs.id_b)
                       OR (t.article_id = belief_pairs.id_b AND t.related_article_id = belief_pairs.id_a)
                )
                """,
                (threshold,),
            )
            pairs = cur.fetchall()

            negation_words = {
                "not", "never", "no", "n't", "cannot", "without", "neither", "none",
                "nobody", "nothing", "nowhere", "false", "incorrect", "wrong", "fail",
                "reject", "deny", "refuse", "avoid",
            }
            opposites = [
                ("good", "bad"), ("right", "wrong"), ("true", "false"),
                ("should", "should not"), ("always", "never"), ("prefer", "avoid"),
                ("like", "dislike"), ("works", "fails"), ("correct", "incorrect"),
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
                    conflicts.append({
                        "id_a": str(pair["id_a"]),
                        "content_a": pair["content_a"],
                        "id_b": str(pair["id_b"]),
                        "content_b": pair["content_b"],
                        "similarity": float(pair["similarity"]),
                        "conflict_score": conflict_signal + (float(pair["similarity"]) - threshold) * 0.5,
                        "reason": ", ".join(reason) if reason else "high similarity",
                    })

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
            "recorded_tensions": recorded,
        }
        return format_response(result, output_format, text_formatter=format_conflicts_text)
    except Exception:
        logger.exception("Error detecting conflicts")
        return internal_error()
