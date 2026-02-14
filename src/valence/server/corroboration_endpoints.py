"""Corroboration API endpoints for Valence.

Provides REST endpoints for querying belief corroboration:
- GET /beliefs/{id}/corroboration - Get corroboration details for a belief
"""

from __future__ import annotations

import logging
from uuid import UUID

from starlette.requests import Request
from starlette.responses import JSONResponse

from .auth_helpers import authenticate, require_scope
from .errors import (
    NOT_FOUND_BELIEF,
    internal_error,
    invalid_format_error,
    missing_field_error,
    not_found_error,
)

logger = logging.getLogger(__name__)


async def belief_corroboration_endpoint(request: Request) -> JSONResponse:
    """Get corroboration details for a belief.

    Returns information about how many independent sources
    have corroborated this belief.

    Endpoint: GET /beliefs/{belief_id}/corroboration

    Response:
    {
        "success": true,
        "belief_id": "uuid",
        "corroboration_count": 3,
        "confidence_corroboration": 0.47,
        "corroborating_sources": [
            {
                "source_did": "did:vkb:web:example.com",
                "similarity": 0.95,
                "corroborated_at": "2025-02-03T12:00:00Z"
            }
        ],
        "confidence_label": "moderately corroborated"
    }
    """
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    belief_id_str = request.path_params.get("belief_id")

    if not belief_id_str:
        return missing_field_error("belief_id")

    try:
        belief_id = UUID(belief_id_str)
    except ValueError:
        return invalid_format_error("belief_id", "must be valid UUID")

    try:
        from our_federation.corroboration import get_corroboration

        corroboration = get_corroboration(belief_id)

        if not corroboration:
            return not_found_error("Belief", code=NOT_FOUND_BELIEF)

        # Generate confidence label
        count = corroboration.corroboration_count
        if count == 0:
            label = "uncorroborated"
        elif count == 1:
            label = "single corroboration"
        elif count <= 3:
            label = "moderately corroborated"
        elif count <= 6:
            label = "well corroborated"
        else:
            label = "highly corroborated"

        return JSONResponse(
            {
                "success": True,
                "belief_id": str(corroboration.belief_id),
                "corroboration_count": corroboration.corroboration_count,
                "confidence_corroboration": corroboration.confidence_corroboration,
                "corroborating_sources": corroboration.sources,
                "confidence_label": label,
            }
        )

    except Exception:
        logger.exception(f"Error getting corroboration for {belief_id_str}")
        return internal_error("Internal server error")


async def most_corroborated_beliefs_endpoint(request: Request) -> JSONResponse:
    """Get the most corroborated beliefs.

    Query params:
    - limit: Max results (default 10)
    - min_count: Minimum corroboration count (default 1)
    - domain: Filter by domain

    Endpoint: GET /beliefs/most-corroborated

    Response:
    {
        "success": true,
        "beliefs": [
            {
                "id": "uuid",
                "content": "...",
                "corroboration_count": 5,
                "confidence_corroboration": 0.60,
                "sources": [...],
                "domain_path": ["tech", "python"]
            }
        ]
    }
    """
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    try:
        limit = int(request.query_params.get("limit", 10))
        min_count = int(request.query_params.get("min_count", 1))
        domain = request.query_params.get("domain")

        domain_filter = [domain] if domain else None

        from our_federation.corroboration import get_most_corroborated_beliefs

        beliefs = get_most_corroborated_beliefs(
            limit=limit,
            min_count=min_count,
            domain_filter=domain_filter,
        )

        return JSONResponse(
            {
                "success": True,
                "beliefs": beliefs,
                "total_count": len(beliefs),
            }
        )

    except Exception:
        logger.exception("Error getting most corroborated beliefs")
        return internal_error("Internal server error")
