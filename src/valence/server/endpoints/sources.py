# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""REST endpoints for source ingestion and retrieval (C1).

Routes (mounted under /api/v1 in app.py):
    POST   /sources         — Ingest a new source (201)
    GET    /sources         — List sources
    GET    /sources/{id}    — Get source by ID
    POST   /sources/search  — Full-text search

All endpoints require authentication (substrate:read or substrate:write scope).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..auth_helpers import authenticate, require_scope
from ..endpoint_utils import _parse_int
from ..errors import conflict_error, internal_error, invalid_json_error, missing_field_error, not_found_error

logger = logging.getLogger(__name__)


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, UUID):
            return str(o)
        return super().default(o)


def _json_response(data, **kw):
    body = json.dumps(data, cls=_Encoder)
    return JSONResponse(content=json.loads(body), **kw)


# ---------------------------------------------------------------------------
# POST /api/v1/sources — ingest
# ---------------------------------------------------------------------------


async def sources_create_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sources — Ingest a new source.

    Request body (JSON):
        content      (required) — raw text content
        source_type  (required) — document | conversation | web | code | observation |
                                   tool_output | user_input
        title        (optional) — human-readable title
        url          (optional) — canonical URL
        metadata     (optional) — arbitrary JSON object

    Returns 201 on success, 409 on duplicate fingerprint.
    """
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

    source_type = body.get("source_type")
    if not source_type:
        return missing_field_error("source_type")

    try:
        from ...core.exceptions import ConflictError, ValidationException
        from ...core.sources import ingest_source

        result = await ingest_source(
            content=content,
            source_type=source_type,
            title=body.get("title"),
            url=body.get("url"),
            metadata=body.get("metadata"),
        )
        if not result.success:
            return _json_response({"success": False, "error": result.error}, status_code=400)
        from ...server.app import record_write

        record_write()
        return _json_response({"success": True, "source": result.data}, status_code=201)

    except ConflictError as e:
        return conflict_error(e.message)
    except ValidationException as e:
        from ..errors import validation_error

        return validation_error(e.message)
    except Exception:
        logger.exception("Error ingesting source")
        return internal_error()


# ---------------------------------------------------------------------------
# GET /api/v1/sources/{source_id} — get by ID
# ---------------------------------------------------------------------------


async def sources_get_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sources/{source_id} — Retrieve a source by ID."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    source_id = request.path_params.get("source_id")
    if not source_id:
        return missing_field_error("source_id")

    try:
        from ...core.exceptions import NotFoundError
        from ...core.sources import get_source

        result = await get_source(source_id)
        return _json_response({"success": True, "source": result.data})

    except NotFoundError:
        return not_found_error(f"source {source_id}")
    except Exception:
        logger.exception("Error getting source %s", source_id)
        return internal_error()


# ---------------------------------------------------------------------------
# GET /api/v1/sources — list
# ---------------------------------------------------------------------------


async def sources_list_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sources — List sources.

    Query params:
        source_type  — optional filter
        limit        — page size (default 50)
        offset       — pagination offset (default 0)
    """
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    source_type = request.query_params.get("source_type")
    limit = _parse_int(request.query_params.get("limit"), default=50, maximum=200)
    offset = _parse_int(request.query_params.get("offset"), default=0, maximum=100_000)

    try:
        from ...core.sources import list_sources

        result = await list_sources(
            source_type=source_type,
            limit=limit,
            offset=offset,
        )
        sources = result.data if result.success else []
        return _json_response({"success": True, "sources": sources, "total_count": len(sources)})

    except Exception:
        logger.exception("Error listing sources")
        return internal_error()


# ---------------------------------------------------------------------------
# POST /api/v1/sources/search — full-text search
# ---------------------------------------------------------------------------


async def sources_search_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sources/search — Full-text search over source content.

    Request body (JSON):
        query  (required) — search terms
        limit  (optional) — max results (default 20)
    """
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    query = body.get("query")
    if not query:
        return missing_field_error("query")

    limit = int(body.get("limit", 20))

    try:
        from ...core.sources import search_sources

        result = await search_sources(query=query, limit=limit)
        sources = result.data if result.success else []
        return _json_response({"success": True, "sources": sources, "total_count": len(sources)})

    except Exception:
        logger.exception("Error searching sources")
        return internal_error()


# ---------------------------------------------------------------------------
# DELETE /api/v1/sources/{source_id} — delete (admin_forget)
# ---------------------------------------------------------------------------


async def sources_delete_endpoint(request: Request) -> JSONResponse:
    """DELETE /api/v1/sources/{source_id} — Permanently remove a source."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:write"):
        return err

    source_id = request.path_params.get("source_id")
    if not source_id:
        return missing_field_error("source_id")

    try:
        from ...core.forgetting import remove_source

        result = await remove_source(source_id)
        status = 200 if result.success else 404
        return JSONResponse(result.to_dict(), status_code=status)

    except Exception:
        logger.exception("Error deleting source %s", source_id)
        return internal_error()


async def sources_index_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sources/{source_id}/index — Build tree index for a source."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:write"):
        return err

    source_id = request.path_params.get("source_id")
    if not source_id:
        return missing_field_error("source_id")

    try:
        body = await request.json()
    except Exception:
        body = {}

    force = body.get("force", False)
    window_tokens = body.get("window_tokens")

    try:
        from ...core.tree_index import build_tree_index

        kwargs: dict = {"source_id": source_id, "force": force}
        if window_tokens is not None:
            kwargs["window_tokens"] = int(window_tokens)

        result = await build_tree_index(**kwargs)
        status = 200 if result.success else 400
        return JSONResponse(result.to_dict(), status_code=status)

    except Exception:
        logger.exception("Error building tree index for source %s", source_id)
        return internal_error()


async def sources_tree_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sources/{source_id}/tree — Get tree index for a source."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client

    source_id = request.path_params.get("source_id")
    if not source_id:
        return missing_field_error("source_id")

    try:
        from ...core.tree_index import get_tree_index

        result = await get_tree_index(source_id)
        status = 200 if result.success else 404
        return JSONResponse(result.to_dict(), status_code=status)

    except Exception:
        logger.exception("Error getting tree index for source %s", source_id)
        return internal_error()


async def sources_region_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sources/{source_id}/region — Get text region by char offsets."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client

    source_id = request.path_params.get("source_id")
    if not source_id:
        return missing_field_error("source_id")

    try:
        start_char = int(request.query_params.get("start", 0))
        end_char = int(request.query_params.get("end", 0))
    except (TypeError, ValueError):
        return missing_field_error("start and end must be integers")

    try:
        from ...core.tree_index import get_tree_region

        result = await get_tree_region(source_id, start_char, end_char)
        status = 200 if result.success else 400
        return JSONResponse(result.to_dict(), status_code=status)

    except Exception:
        logger.exception("Error getting region for source %s", source_id)
        return internal_error()
