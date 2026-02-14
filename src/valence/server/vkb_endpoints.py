"""REST API endpoints for Valence VKB (conversation tracking).

Provides RESTful access to sessions, exchanges, patterns, and insights.
All endpoints require authentication and scope-based authorization.
"""

from __future__ import annotations

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

from .auth_helpers import authenticate, require_scope
from .endpoint_utils import _parse_bool, _parse_float, _parse_int
from .errors import internal_error, invalid_json_error, missing_field_error

logger = logging.getLogger(__name__)


# =============================================================================
# SESSIONS
# =============================================================================


async def sessions_create_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sessions — Start a new session."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:write"):
        return err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    platform = body.get("platform")
    if not platform:
        return missing_field_error("platform")

    try:
        from ..vkb.tools.sessions import session_start

        result = session_start(
            platform=platform,
            project_context=body.get("project_context"),
            external_room_id=body.get("external_room_id"),
            claude_session_id=body.get("claude_session_id"),
            metadata=body.get("metadata"),
        )
        status_code = 201 if result.get("success") else 400
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception("Error creating session")
        return internal_error()


async def sessions_list_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sessions — List sessions with optional filters."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:read"):
        return err

    try:
        from ..vkb.tools.sessions import session_list

        result = session_list(
            platform=request.query_params.get("platform"),
            project_context=request.query_params.get("project_context"),
            status=request.query_params.get("status"),
            limit=_parse_int(request.query_params.get("limit"), 20, 100),
        )
        return JSONResponse(result)
    except Exception:
        logger.exception("Error listing sessions")
        return internal_error()


async def sessions_by_room_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sessions/by-room/{room_id} — Find active session by room ID."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:read"):
        return err

    room_id = request.path_params.get("room_id")
    if not room_id:
        return missing_field_error("room_id")

    try:
        from ..vkb.tools.sessions import session_find_by_room

        result = session_find_by_room(external_room_id=room_id)
        status_code = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error finding session for room {room_id}")
        return internal_error()


async def sessions_get_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sessions/{id} — Get session by ID."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:read"):
        return err

    session_id = request.path_params.get("id")
    if not session_id:
        return missing_field_error("id")

    try:
        from ..vkb.tools.sessions import session_get

        result = session_get(
            session_id=session_id,
            include_exchanges=_parse_bool(request.query_params.get("include_exchanges")),
            exchange_limit=_parse_int(request.query_params.get("exchange_limit"), 10, 100),
        )
        status_code = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error getting session {session_id}")
        return internal_error()


async def sessions_end_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sessions/{id}/end — End a session."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:write"):
        return err

    session_id = request.path_params.get("id")
    if not session_id:
        return missing_field_error("id")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    try:
        from ..vkb.tools.sessions import session_end

        result = session_end(
            session_id=session_id,
            summary=body.get("summary"),
            themes=body.get("themes"),
            status=body.get("status", "completed"),
        )
        status_code = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error ending session {session_id}")
        return internal_error()


# =============================================================================
# EXCHANGES
# =============================================================================


async def exchanges_add_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sessions/{id}/exchanges — Add an exchange to a session."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:write"):
        return err

    session_id = request.path_params.get("id")
    if not session_id:
        return missing_field_error("id")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    role = body.get("role")
    if not role:
        return missing_field_error("role")

    content = body.get("content")
    if not content:
        return missing_field_error("content")

    try:
        from ..vkb.tools.exchanges import exchange_add

        result = exchange_add(
            session_id=session_id,
            role=role,
            content=content,
            tokens_approx=body.get("tokens_approx"),
            tool_uses=body.get("tool_uses"),
        )
        status_code = 201 if result.get("success") else 400
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error adding exchange to session {session_id}")
        return internal_error()


async def exchanges_list_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sessions/{id}/exchanges — List exchanges for a session."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:read"):
        return err

    session_id = request.path_params.get("id")
    if not session_id:
        return missing_field_error("id")

    try:
        from ..vkb.tools.exchanges import exchange_list

        limit_raw = request.query_params.get("limit")
        limit = _parse_int(limit_raw, 0, 1000) if limit_raw else None

        result = exchange_list(
            session_id=session_id,
            limit=limit,
            offset=_parse_int(request.query_params.get("offset"), 0, 10000),
        )
        return JSONResponse(result)
    except Exception:
        logger.exception(f"Error listing exchanges for session {session_id}")
        return internal_error()


# =============================================================================
# INSIGHTS
# =============================================================================


async def insights_extract_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sessions/{id}/insights — Extract an insight from a session."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:write"):
        return err

    session_id = request.path_params.get("id")
    if not session_id:
        return missing_field_error("id")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    content = body.get("content")
    if not content:
        return missing_field_error("content")

    try:
        from ..vkb.tools.insights import insight_extract

        result = insight_extract(
            session_id=session_id,
            content=content,
            domain_path=body.get("domain_path"),
            confidence=body.get("confidence"),
            entities=body.get("entities"),
        )
        status_code = 201 if result.get("success") else 400
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error extracting insight from session {session_id}")
        return internal_error()


async def insights_list_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sessions/{id}/insights — List insights from a session."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:read"):
        return err

    session_id = request.path_params.get("id")
    if not session_id:
        return missing_field_error("id")

    try:
        from ..vkb.tools.insights import insight_list

        result = insight_list(session_id=session_id)
        return JSONResponse(result)
    except Exception:
        logger.exception(f"Error listing insights for session {session_id}")
        return internal_error()


# =============================================================================
# PATTERNS
# =============================================================================


async def patterns_create_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/patterns — Record a new behavioral pattern."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:write"):
        return err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    pattern_type = body.get("type")
    if not pattern_type:
        return missing_field_error("type")

    description = body.get("description")
    if not description:
        return missing_field_error("description")

    try:
        from ..vkb.tools.patterns import pattern_record

        result = pattern_record(
            type=pattern_type,
            description=description,
            evidence=body.get("evidence"),
            confidence=body.get("confidence", 0.5),
        )
        status_code = 201 if result.get("success") else 400
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception("Error recording pattern")
        return internal_error()


async def patterns_list_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/patterns — List patterns with optional filters."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:read"):
        return err

    try:
        from ..vkb.tools.patterns import pattern_list

        result = pattern_list(
            type=request.query_params.get("type"),
            status=request.query_params.get("status"),
            min_confidence=_parse_float(request.query_params.get("min_confidence"), 0.0),
            limit=_parse_int(request.query_params.get("limit"), 20, 100),
        )
        return JSONResponse(result)
    except Exception:
        logger.exception("Error listing patterns")
        return internal_error()


async def patterns_search_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/patterns/search — Search patterns by description."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:read"):
        return err

    query = request.query_params.get("query")
    if not query:
        return missing_field_error("query")

    try:
        from ..vkb.tools.patterns import pattern_search

        result = pattern_search(
            query=query,
            limit=_parse_int(request.query_params.get("limit"), 10, 100),
        )
        return JSONResponse(result)
    except Exception:
        logger.exception("Error searching patterns")
        return internal_error()


async def patterns_reinforce_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/patterns/{id}/reinforce — Reinforce an existing pattern."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "vkb:write"):
        return err

    pattern_id = request.path_params.get("id")
    if not pattern_id:
        return missing_field_error("id")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    try:
        from ..vkb.tools.patterns import pattern_reinforce

        result = pattern_reinforce(
            pattern_id=pattern_id,
            session_id=body.get("session_id"),
        )
        status_code = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status_code)
    except Exception:
        logger.exception(f"Error reinforcing pattern {pattern_id}")
        return internal_error()
