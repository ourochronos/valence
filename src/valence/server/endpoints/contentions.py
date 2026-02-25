# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""REST endpoints for contention management.

Routes:
    GET    /api/v1/contentions            — list_contentions_endpoint
    POST   /api/v1/contentions/:id/resolve — resolve_contention_endpoint
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


def _json_response(data, **kw):
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
# Contention endpoints
# ---------------------------------------------------------------------------


async def list_contentions_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/contentions — List active contentions."""
    err, _ = _require_auth_read(request)
    if err:
        return err

    article_id = request.query_params.get("article_id")
    status = request.query_params.get("status")

    try:
        from ...core.contention import list_contentions

        kwargs = {}
        if article_id:
            kwargs["article_id"] = article_id
        if status:
            kwargs["status"] = status

        result = await list_contentions(**kwargs)
        contentions = result.data if result.success else []
        return _json_response(
            {
                "success": True,
                "contentions": contentions,
                "total_count": len(contentions),
            }
        )
    except Exception:
        logger.exception("Error listing contentions")
        return internal_error()


async def resolve_contention_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/contentions/{contention_id}/resolve — Resolve a contention."""
    err, _ = _require_auth_write(request)
    if err:
        return err

    contention_id = request.path_params.get("contention_id")
    if not contention_id:
        return missing_field_error("contention_id")

    try:
        body = await request.json()
    except Exception:
        return invalid_json_error()

    resolution = body.get("resolution")
    rationale = body.get("rationale")

    if not resolution:
        return missing_field_error("resolution")
    if not rationale:
        return missing_field_error("rationale")

    try:
        from ...core.contention import resolve_contention

        result = await resolve_contention(
            contention_id=contention_id,
            resolution=resolution,
            rationale=rationale,
        )
        status = 200 if result.success else 400
        return JSONResponse(result.to_dict(), status_code=status)
    except Exception:
        logger.exception("Error resolving contention %s", contention_id)
        return internal_error()
