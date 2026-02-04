"""Compliance API endpoints.

Implements:
- DELETE /api/v1/users/{id}/data - GDPR Article 17 deletion (Issue #25)
- Deletion verification endpoint
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..compliance.deletion import (
    delete_user_data,
    get_deletion_verification,
    DeletionReason,
)

logger = logging.getLogger(__name__)


async def delete_user_data_endpoint(request: Request) -> JSONResponse:
    """DELETE /api/v1/users/{id}/data - Delete all user data (GDPR Article 17).
    
    Implements cryptographic erasure per COMPLIANCE.md §3:
    - Creates tombstone records for federation propagation
    - Performs key revocation (cryptographic erasure)
    - Maintains audit trail
    
    Path Parameters:
        id: User identifier
        
    Query Parameters:
        reason: Deletion reason (optional, defaults to 'user_request')
        legal_basis: Additional legal basis description (optional)
        
    Returns:
        200: Deletion result with tombstone ID and counts
        400: Invalid request
        500: Server error
    """
    user_id = request.path_params.get("id")
    
    if not user_id:
        return JSONResponse(
            {"error": "User ID is required"},
            status_code=400,
        )
    
    # Parse reason from query params
    reason_str = request.query_params.get("reason", "user_request")
    try:
        reason = DeletionReason(reason_str)
    except ValueError:
        return JSONResponse(
            {
                "error": f"Invalid reason: {reason_str}",
                "valid_reasons": [r.value for r in DeletionReason],
            },
            status_code=400,
        )
    
    legal_basis = request.query_params.get("legal_basis")
    
    try:
        result = delete_user_data(
            user_id=user_id,
            reason=reason,
            legal_basis=legal_basis,
        )
        
        if result.success:
            return JSONResponse(result.to_dict(), status_code=200)
        else:
            return JSONResponse(result.to_dict(), status_code=500)
            
    except Exception as e:
        logger.exception(f"Error deleting user data: {e}")
        return JSONResponse(
            {"error": str(e), "success": False},
            status_code=500,
        )


async def get_deletion_verification_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/tombstones/{id}/verification - Get deletion verification report.
    
    Returns a verification report for compliance auditing per COMPLIANCE.md §3.
    
    Path Parameters:
        id: Tombstone UUID
        
    Returns:
        200: Verification report
        404: Tombstone not found
    """
    tombstone_id_str = request.path_params.get("id")
    
    try:
        tombstone_id = UUID(tombstone_id_str)
    except (TypeError, ValueError):
        return JSONResponse(
            {"error": "Invalid tombstone ID"},
            status_code=400,
        )
    
    report = get_deletion_verification(tombstone_id)
    
    if report is None:
        return JSONResponse(
            {"error": "Tombstone not found"},
            status_code=404,
        )
    
    return JSONResponse(report, status_code=200)
