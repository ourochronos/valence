# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""REST endpoints for knowledge search, article compilation, and admin forget.

Routes:
    GET    /api/v1/search                          — knowledge_search_endpoint
    POST   /api/v1/articles/compile                — compile_article_endpoint
    DELETE /api/v1/admin/forget/:target_type/:id   — admin_forget_endpoint
"""

from __future__ import annotations

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..auth_helpers import authenticate, require_scope
from ..endpoint_utils import _parse_bool, _parse_int
from ..errors import internal_error, missing_field_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge Search
# ---------------------------------------------------------------------------


async def knowledge_search_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/search — Unified knowledge retrieval."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    query = request.query_params.get("query", "").strip()
    if not query:
        return missing_field_error("query")

    limit = _parse_int(request.query_params.get("limit"), 10)
    include_sources = _parse_bool(request.query_params.get("include_sources"))
    session_id = request.query_params.get("session_id")
    temporal_mode = request.query_params.get("temporal_mode", "default")

    try:
        from valence.mcp.handlers.articles import knowledge_search

        result = knowledge_search(
            query=query,
            limit=limit,
            include_sources=include_sources,
            session_id=session_id,
            temporal_mode=temporal_mode,
        )
        status = 200 if result.get("success") else 400
        return JSONResponse(result, status_code=status)
    except Exception:
        logger.exception("knowledge_search_endpoint failed")
        return internal_error()


# ---------------------------------------------------------------------------
# Article Compile
# ---------------------------------------------------------------------------


async def compile_article_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/articles/compile — Compile sources into an article via LLM."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:write"):
        return err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"success": False, "error": {"code": "invalid_json", "message": "Invalid JSON body"}},
            status_code=400,
        )

    source_ids = body.get("source_ids")
    if not source_ids or not isinstance(source_ids, list):
        return missing_field_error("source_ids")

    title_hint = body.get("title_hint")

    try:
        from valence.mcp.handlers.articles import article_compile

        result = article_compile(source_ids=source_ids, title_hint=title_hint)
        # Ensure all nested dicts are JSON-safe (Decimal, UUID, datetime)
        import json as _json

        safe_result = _json.loads(_json.dumps(result, default=str))
        status = 200 if safe_result.get("success") else 400
        return JSONResponse(safe_result, status_code=status)
    except Exception:
        logger.exception("compile_article_endpoint failed")
        return internal_error()


# ---------------------------------------------------------------------------
# Admin Forget
# ---------------------------------------------------------------------------


async def admin_forget_endpoint(request: Request) -> JSONResponse:
    """DELETE /api/v1/admin/forget/:target_type/:target_id — Permanently remove a source or article."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:write"):
        return err

    target_type = request.path_params.get("target_type", "")
    target_id = request.path_params.get("target_id", "")

    if target_type not in ("source", "article"):
        return JSONResponse(
            {"success": False, "error": {"code": "validation_error", "message": "target_type must be 'source' or 'article'"}},
            status_code=400,
        )

    if not target_id:
        return missing_field_error("target_id")

    try:
        from valence.mcp.handlers.admin import admin_forget

        result = admin_forget(target_type=target_type, target_id=target_id)
        status = 200 if result.get("success") else 400
        return JSONResponse(result, status_code=status)
    except Exception:
        logger.exception("admin_forget_endpoint failed")
        return internal_error()
