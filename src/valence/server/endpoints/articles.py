"""REST endpoints for article CRUD and provenance operations (WU-04).

Routes:
    POST   /api/v1/articles            — create_article_endpoint
    GET    /api/v1/articles/:id        — get_article_endpoint
    PUT    /api/v1/articles/:id        — update_article_endpoint
    POST   /api/v1/articles/search     — search_articles_endpoint
    POST   /api/v1/articles/:id/provenance/link  — link_provenance_endpoint
    GET    /api/v1/articles/:id/provenance        — get_provenance_endpoint
    POST   /api/v1/articles/:id/provenance/trace  — trace_claim_endpoint
"""

from __future__ import annotations

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..auth_helpers import authenticate, require_scope
from ..endpoint_utils import _parse_bool, _parse_int
from ..errors import internal_error, invalid_json_error, missing_field_error, validation_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# Article endpoints
# ---------------------------------------------------------------------------


async def create_article_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/articles — Create a new article."""
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
        from ...core.articles import create_article

        result = create_article(
            content=content,
            title=body.get("title"),
            source_ids=body.get("source_ids"),
            author_type=body.get("author_type", "system"),
            domain_path=body.get("domain_path"),
        )
        status = 201 if result.get("success") else 400
        return JSONResponse(result, status_code=status)
    except Exception:
        logger.exception("Error creating article")
        return internal_error()


async def get_article_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/articles/{article_id} — Get an article by ID."""
    err, _ = _require_auth_read(request)
    if err:
        return err

    article_id = request.path_params.get("article_id")
    if not article_id:
        return missing_field_error("article_id")

    include_provenance = _parse_bool(request.query_params.get("include_provenance"))

    try:
        from ...core.articles import get_article

        result = get_article(article_id=article_id, include_provenance=include_provenance)
        status = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status)
    except Exception:
        logger.exception("Error fetching article %s", article_id)
        return internal_error()


async def update_article_endpoint(request: Request) -> JSONResponse:
    """PUT /api/v1/articles/{article_id} — Update article content."""
    err, _ = _require_auth_write(request)
    if err:
        return err

    article_id = request.path_params.get("article_id")
    if not article_id:
        return missing_field_error("article_id")

    try:
        body = await request.json()
    except Exception:
        return invalid_json_error()

    content = body.get("content")
    if not content:
        return missing_field_error("content")

    try:
        from ...core.articles import update_article

        result = update_article(
            article_id=article_id,
            content=content,
            source_id=body.get("source_id"),
        )
        status = 200 if result.get("success") else 404
        return JSONResponse(result, status_code=status)
    except Exception:
        logger.exception("Error updating article %s", article_id)
        return internal_error()


async def search_articles_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/articles/search — Search articles."""
    err, _ = _require_auth_read(request)
    if err:
        return err

    try:
        body = await request.json()
    except Exception:
        return invalid_json_error()

    query = body.get("query")
    if not query:
        return missing_field_error("query")

    limit = min(int(body.get("limit", 10)), 50)

    try:
        from ...core.articles import search_articles

        result = await search_articles(
            query=query,
            limit=limit,
            domain_filter=body.get("domain_filter"),
        )
        articles = result.data if result.success else []
        return JSONResponse({
            "success": True,
            "articles": articles,
            "total_count": len(articles),
        })
    except Exception:
        logger.exception("Error searching articles")
        return internal_error()


# ---------------------------------------------------------------------------
# Provenance endpoints
# ---------------------------------------------------------------------------


async def link_provenance_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/articles/{article_id}/provenance/link — Link a source."""
    err, _ = _require_auth_write(request)
    if err:
        return err

    article_id = request.path_params.get("article_id")
    if not article_id:
        return missing_field_error("article_id")

    try:
        body = await request.json()
    except Exception:
        return invalid_json_error()

    source_id = body.get("source_id")
    relationship = body.get("relationship")
    if not source_id:
        return missing_field_error("source_id")
    if not relationship:
        return missing_field_error("relationship")

    try:
        from ...core.provenance import link_source

        result = link_source(
            article_id=article_id,
            source_id=source_id,
            relationship=relationship,
            notes=body.get("notes"),
        )
        status = 201 if result.get("success") else 400
        return JSONResponse(result, status_code=status)
    except Exception:
        logger.exception("Error linking provenance for article %s", article_id)
        return internal_error()


async def get_provenance_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/articles/{article_id}/provenance — Get provenance list."""
    err, _ = _require_auth_read(request)
    if err:
        return err

    article_id = request.path_params.get("article_id")
    if not article_id:
        return missing_field_error("article_id")

    try:
        from ...core.provenance import get_provenance

        provenance = get_provenance(article_id=article_id)
        return JSONResponse({
            "success": True,
            "provenance": provenance,
            "count": len(provenance),
        })
    except Exception:
        logger.exception("Error fetching provenance for article %s", article_id)
        return internal_error()


async def trace_claim_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/articles/{article_id}/provenance/trace — Trace a claim."""
    err, _ = _require_auth_read(request)
    if err:
        return err

    article_id = request.path_params.get("article_id")
    if not article_id:
        return missing_field_error("article_id")

    try:
        body = await request.json()
    except Exception:
        return invalid_json_error()

    claim_text = body.get("claim_text")
    if not claim_text:
        return missing_field_error("claim_text")

    try:
        from ...core.provenance import trace_claim

        sources = trace_claim(article_id=article_id, claim_text=claim_text)
        return JSONResponse({
            "success": True,
            "sources": sources,
            "count": len(sources),
        })
    except Exception:
        logger.exception("Error tracing claim for article %s", article_id)
        return internal_error()
