"""Compliance API endpoints.

Implements:
- DELETE /api/v1/users/{id}/data - GDPR Article 17 deletion (Issue #25)
- GET /api/v1/compliance/access - GDPR Article 15 data access
- GET /api/v1/compliance/export - GDPR Article 20 data portability
- POST /api/v1/compliance/import - GDPR Article 20 data import
- Deletion verification endpoint
"""

from __future__ import annotations

import json
import logging
from uuid import UUID

from our_compliance.deletion import (
    DeletionReason,
    delete_user_data,
    get_deletion_verification,
)
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..compliance.data_access import export_holder_data, get_holder_data, import_holder_data
from .auth_helpers import authenticate, require_scope
from .errors import (
    NOT_FOUND_TOMBSTONE,
    VALIDATION_INVALID_VALUE,
    internal_error,
    invalid_format_error,
    invalid_json_error,
    missing_field_error,
    not_found_error,
    validation_error,
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
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:write"):
        return err

    user_id = request.path_params.get("id")

    if not user_id:
        return missing_field_error("User ID")

    # Parse reason from query params
    reason_str = request.query_params.get("reason", "user_request")
    try:
        reason = DeletionReason(reason_str)
    except ValueError:
        valid = ", ".join(r.value for r in DeletionReason)
        return validation_error(
            f"Invalid reason: {reason_str}. Valid: {valid}",
            code=VALIDATION_INVALID_VALUE,
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
        return internal_error("Internal server error")


async def data_access_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/compliance/access - GDPR Article 15 data access.

    Returns all data held about a data subject.

    Query Parameters:
        holder_did: DID of the data subject (required)

    Returns:
        200: All holder data organized by category
        400: Missing holder_did parameter
        500: Server error
    """
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    holder_did = request.query_params.get("holder_did")

    if not holder_did:
        return missing_field_error("holder_did")

    try:
        data = get_holder_data(holder_did)
        return JSONResponse({"success": True, **data}, status_code=200)
    except Exception as e:
        logger.exception(f"Error accessing holder data: {e}")
        return internal_error("Internal server error")


async def data_export_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/compliance/export - GDPR Article 20 data portability.

    Returns holder data in a portable, machine-readable format.

    Query Parameters:
        holder_did: DID of the data subject (required)

    Returns:
        200: Portable export with format version and all data
        400: Missing holder_did parameter
        500: Server error
    """
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    holder_did = request.query_params.get("holder_did")

    if not holder_did:
        return missing_field_error("holder_did")

    try:
        export = export_holder_data(holder_did)
        return JSONResponse({"success": True, **export}, status_code=200)
    except Exception as e:
        logger.exception(f"Error exporting holder data: {e}")
        return internal_error("Internal server error")


async def data_import_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/compliance/import - GDPR Article 20 data import.

    Imports data from a portable export.

    Body:
        JSON export data (from data_export_endpoint)

    Returns:
        200: Import results with counts
        400: Invalid JSON or format
        500: Server error
    """
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:write"):
        return err

    try:
        body = await request.body()
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return invalid_json_error()

    try:
        result = import_holder_data(data)
        status_code = 200 if result.get("success") else 400
        return JSONResponse(result, status_code=status_code)
    except Exception as e:
        logger.exception(f"Error importing data: {e}")
        return internal_error("Internal server error")


async def get_deletion_verification_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/tombstones/{id}/verification - Get deletion verification report.

    Returns a verification report for compliance auditing per COMPLIANCE.md §3.

    Path Parameters:
        id: Tombstone UUID

    Returns:
        200: Verification report
        404: Tombstone not found
    """
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    tombstone_id_str = request.path_params.get("id")

    try:
        tombstone_id = UUID(tombstone_id_str)
    except (TypeError, ValueError):
        return invalid_format_error("tombstone ID", "must be valid UUID")

    report = get_deletion_verification(tombstone_id)

    if report is None:
        return not_found_error("Tombstone", code=NOT_FOUND_TOMBSTONE)

    return JSONResponse(report, status_code=200)
