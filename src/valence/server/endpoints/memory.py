# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""REST endpoints for memory operations.

Routes:
    POST   /api/v1/memory             — store a memory
    GET    /api/v1/memory/search       — recall memories
    GET    /api/v1/memory/status       — memory system stats
    POST   /api/v1/memory/{id}/forget  — soft-delete a memory
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..auth_helpers import authenticate, require_scope
from ..errors import internal_error, invalid_json_error, missing_field_error

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


def _json_response(data: Any, **kw: Any) -> JSONResponse:
    body = json.dumps(data, cls=_Encoder)
    return JSONResponse(content=json.loads(body), **kw)


def _require_auth_write(request: Request):
    """Authenticate and require substrate:write scope."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client, None
    if err := require_scope(client, "substrate:write"):
        return err, None
    return None, client


def _require_auth_read(request: Request):
    """Authenticate and require substrate:read scope."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client, None
    if err := require_scope(client, "substrate:read"):
        return err, None
    return None, client


# ---------------------------------------------------------------------------
# Memory endpoints
# ---------------------------------------------------------------------------


async def memory_store_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/memory — Store a memory."""
    err, _ = _require_auth_write(request)
    if err:
        return err

    try:
        body = await request.json()
    except Exception:
        return invalid_json_error()

    content = body.get("content")
    if not content:
        return missing_field_error("content")

    try:
        from ...mcp.handlers.memory import memory_store

        result = await asyncio.to_thread(
            memory_store,
            content=content,
            context=body.get("context"),
            importance=body.get("importance", 0.5),
            tags=body.get("tags"),
            supersedes_id=body.get("supersedes_id"),
        )

        status_code = 201 if result.get("success") else 400
        return _json_response(result, status_code=status_code)
    except Exception:
        logger.exception("Error storing memory")
        return internal_error()


async def memory_search_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/memory/search — Recall memories."""
    err, _ = _require_auth_read(request)
    if err:
        return err

    query = request.query_params.get("query")
    if not query:
        return missing_field_error("query")

    limit = int(request.query_params.get("limit", "5"))
    min_confidence_str = request.query_params.get("min_confidence")
    min_confidence = float(min_confidence_str) if min_confidence_str else None
    tags_str = request.query_params.get("tags")
    tags = tags_str.split(",") if tags_str else None

    try:
        from ...mcp.handlers.memory import memory_recall

        result = await asyncio.to_thread(
            memory_recall,
            query=query,
            limit=limit,
            min_confidence=min_confidence,
            tags=tags,
        )

        return _json_response(result)
    except Exception:
        logger.exception("Error searching memories")
        return internal_error()


async def memory_status_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/memory/status — Memory system statistics."""
    err, _ = _require_auth_read(request)
    if err:
        return err

    try:
        from ...mcp.handlers.memory import memory_status

        result = await asyncio.to_thread(memory_status)
        return _json_response(result)
    except Exception:
        logger.exception("Error getting memory status")
        return internal_error()


async def memory_forget_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/memory/{memory_id}/forget — Soft-delete a memory."""
    err, _ = _require_auth_write(request)
    if err:
        return err

    memory_id = request.path_params.get("memory_id")
    if not memory_id:
        return missing_field_error("memory_id")

    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        from ...mcp.handlers.memory import memory_forget

        result = await asyncio.to_thread(
            memory_forget,
            memory_id=memory_id,
            reason=body.get("reason"),
        )

        status_code = 200 if result.get("success") else 400
        return _json_response(result, status_code=status_code)
    except Exception:
        logger.exception("Error forgetting memory %s", memory_id)
        return internal_error()


async def memory_list_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/memory — List recent memories."""
    err, _ = _require_auth_read(request)
    if err:
        return err

    limit = int(request.query_params.get("limit", "20"))
    tags_str = request.query_params.get("tags")
    tags = tags_str.split(",") if tags_str else None

    try:
        from ...mcp.handlers.memory import memory_list

        result = await asyncio.to_thread(memory_list, limit=limit, tags=tags)
        return _json_response(result)
    except Exception:
        logger.exception("Error listing memories")
        return internal_error()
