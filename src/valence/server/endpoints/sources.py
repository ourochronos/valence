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
from uuid import UUID

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..auth_helpers import authenticate, require_scope


class _Encoder(json.JSONEncoder):
    def default(self, o):
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
from ..endpoint_utils import _parse_int
from ..errors import conflict_error, internal_error, invalid_json_error, missing_field_error, not_found_error

logger = logging.getLogger(__name__)


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

        source = await ingest_source(
            content=content,
            source_type=source_type,
            title=body.get("title"),
            url=body.get("url"),
            metadata=body.get("metadata"),
        )
        return _json_response({"success": True, "source": source}, status_code=201)

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

        source = await get_source(source_id)
        return _json_response({"success": True, "source": source})

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
    limit = _parse_int(request.query_params.get("limit"), default=50, max_val=200)
    offset = _parse_int(request.query_params.get("offset"), default=0, max_val=100_000)

    try:
        from ...core.sources import list_sources

        sources = await list_sources(
            source_type=source_type,
            limit=limit,
            offset=offset,
        )
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
