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
from .errors import internal_error, invalid_json_error, missing_field_error

logger = logging.getLogger(__name__)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes")


def _parse_int(value: str | None, default: int, maximum: int = 1000) -> int:
    if value is None:
        return default
    try:
        return min(int(value), maximum)
    except ValueError:
        return default


def _parse_float(value: str | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


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
        from ..substrate.tools.beliefs import belief_query

        domain_filter = None
        df_raw = request.query_params.get("domain_filter")
        if df_raw:
            domain_filter = [s.strip() for s in df_raw.split(",")]

        ranking = None
        ranking_raw = request.query_params.get("ranking")
        if ranking_raw:
            ranking = json.loads(ranking_raw)

        result = belief_query(
            query=query,
            domain_filter=domain_filter,
            entity_id=request.query_params.get("entity_id"),
            include_superseded=_parse_bool(request.query_params.get("include_superseded")),
            include_revoked=_parse_bool(request.query_params.get("include_revoked")),
            include_archived=_parse_bool(request.query_params.get("include_archived")),
            limit=_parse_int(request.query_params.get("limit"), 20, 100),
            ranking=ranking,
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
        from ..substrate.tools.beliefs import belief_create

        result = belief_create(
            content=content,
            confidence=body.get("confidence"),
            domain_path=body.get("domain_path"),
            source_type=body.get("source_type"),
            source_ref=body.get("source_ref"),
            opt_out_federation=body.get("opt_out_federation", False),
            entities=body.get("entities"),
            visibility=body.get("visibility", "private"),
            sharing_intent=body.get("sharing_intent"),
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
        from ..substrate.tools.beliefs import belief_search

        domain_filter = None
        df_raw = request.query_params.get("domain_filter")
        if df_raw:
            domain_filter = [s.strip() for s in df_raw.split(",")]

        ranking = None
        ranking_raw = request.query_params.get("ranking")
        if ranking_raw:
            ranking = json.loads(ranking_raw)

        result = belief_search(
            query=query,
            min_similarity=_parse_float(request.query_params.get("min_similarity"), 0.5) or 0.5,
            min_confidence=_parse_float(request.query_params.get("min_confidence")),
            domain_filter=domain_filter,
            include_archived=_parse_bool(request.query_params.get("include_archived")),
            limit=_parse_int(request.query_params.get("limit"), 10, 100),
            ranking=ranking,
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
        from ..substrate.tools.beliefs import belief_get

        result = belief_get(
            belief_id=belief_id,
            include_history=_parse_bool(request.query_params.get("include_history")),
            include_tensions=_parse_bool(request.query_params.get("include_tensions")),
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
        from ..substrate.tools.beliefs import belief_supersede

        result = belief_supersede(
            old_belief_id=belief_id,
            new_content=new_content,
            reason=reason,
            confidence=body.get("confidence"),
        )
        status_code = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error superseding belief {belief_id}")
        return internal_error()


async def beliefs_confidence_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/beliefs/{belief_id}/confidence — Explain confidence score."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    belief_id = request.path_params.get("belief_id")
    if not belief_id:
        return missing_field_error("belief_id")

    try:
        from ..substrate.tools.confidence import confidence_explain

        result = confidence_explain(belief_id=belief_id)
        status_code = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error explaining confidence for {belief_id}")
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
        from ..substrate.tools.tensions import tension_list

        result = tension_list(
            status=request.query_params.get("status"),
            severity=request.query_params.get("severity"),
            entity_id=request.query_params.get("entity_id"),
            limit=_parse_int(request.query_params.get("limit"), 20, 100),
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
        from ..substrate.tools.tensions import tension_resolve

        result = tension_resolve(
            tension_id=tension_id,
            resolution=resolution,
            action=action,
        )
        status_code = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error resolving tension {tension_id}")
        return internal_error()


# =============================================================================
# TRUST
# =============================================================================


async def trust_check_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/trust — Check trust levels for a topic."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    topic = request.query_params.get("topic")
    if not topic:
        return missing_field_error("topic")

    try:
        from ..substrate.tools.trust import trust_check

        result = trust_check(
            topic=topic,
            entity_name=request.query_params.get("entity_name"),
            include_federated=_parse_bool(request.query_params.get("include_federated"), True),
            min_trust=_parse_float(request.query_params.get("min_trust"), 0.3) or 0.3,
            limit=_parse_int(request.query_params.get("limit"), 10, 100),
            domain=request.query_params.get("domain"),
        )
        return JSONResponse(result)
    except Exception:
        logger.exception("Error checking trust")
        return internal_error()
