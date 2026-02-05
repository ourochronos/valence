"""Standardized REST error responses for Valence API.

All REST endpoints should use these helpers for consistent error format:
{
    "success": false,
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable message"
    }
}

Error codes follow the pattern: DOMAIN_SPECIFIC_ERROR
Examples: VALIDATION_MISSING_FIELD, AUTH_INVALID_TOKEN, NOT_FOUND_BELIEF
"""

from __future__ import annotations

from starlette.responses import JSONResponse


# =============================================================================
# STANDARD ERROR CODES
# =============================================================================

# Validation errors (400)
VALIDATION_MISSING_FIELD = "VALIDATION_MISSING_FIELD"
VALIDATION_INVALID_FORMAT = "VALIDATION_INVALID_FORMAT"
VALIDATION_INVALID_VALUE = "VALIDATION_INVALID_VALUE"
VALIDATION_INVALID_JSON = "VALIDATION_INVALID_JSON"

# Authentication errors (401)
AUTH_INVALID_TOKEN = "AUTH_INVALID_TOKEN"
AUTH_MISSING_TOKEN = "AUTH_MISSING_TOKEN"
AUTH_SIGNATURE_FAILED = "AUTH_SIGNATURE_FAILED"

# Authorization errors (403)
FORBIDDEN_NOT_OWNER = "FORBIDDEN_NOT_OWNER"
FORBIDDEN_INSUFFICIENT_PERMISSION = "FORBIDDEN_INSUFFICIENT_PERMISSION"

# Not found errors (404)
NOT_FOUND_RESOURCE = "NOT_FOUND_RESOURCE"
NOT_FOUND_BELIEF = "NOT_FOUND_BELIEF"
NOT_FOUND_USER = "NOT_FOUND_USER"
NOT_FOUND_SHARE = "NOT_FOUND_SHARE"
NOT_FOUND_NOTIFICATION = "NOT_FOUND_NOTIFICATION"
NOT_FOUND_TOMBSTONE = "NOT_FOUND_TOMBSTONE"
NOT_FOUND_NODE = "NOT_FOUND_NODE"
FEATURE_NOT_ENABLED = "FEATURE_NOT_ENABLED"

# Conflict errors (409)
CONFLICT_ALREADY_EXISTS = "CONFLICT_ALREADY_EXISTS"
CONFLICT_ALREADY_REVOKED = "CONFLICT_ALREADY_REVOKED"

# Rate limiting (429)
RATE_LIMITED = "RATE_LIMITED"

# Server errors (500)
INTERNAL_ERROR = "INTERNAL_ERROR"

# Service unavailable (503)
SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


# =============================================================================
# ERROR RESPONSE HELPERS
# =============================================================================


def error_response(
    code: str,
    message: str,
    status_code: int = 400,
) -> JSONResponse:
    """Create a standardized error response.
    
    Args:
        code: Error code (e.g., VALIDATION_MISSING_FIELD)
        message: Human-readable error message
        status_code: HTTP status code (default 400)
    
    Returns:
        JSONResponse with standardized error format
    """
    return JSONResponse(
        {
            "success": False,
            "error": {
                "code": code,
                "message": message,
            },
        },
        status_code=status_code,
    )


def validation_error(message: str, code: str = VALIDATION_INVALID_VALUE) -> JSONResponse:
    """Create a 400 validation error response."""
    return error_response(code, message, status_code=400)


def missing_field_error(field_name: str) -> JSONResponse:
    """Create a 400 error for missing required field."""
    return error_response(
        VALIDATION_MISSING_FIELD,
        f"{field_name} is required",
        status_code=400,
    )


def invalid_format_error(field_name: str, details: str = "") -> JSONResponse:
    """Create a 400 error for invalid field format."""
    message = f"Invalid {field_name} format"
    if details:
        message = f"{message}: {details}"
    return error_response(VALIDATION_INVALID_FORMAT, message, status_code=400)


def invalid_json_error() -> JSONResponse:
    """Create a 400 error for invalid JSON body."""
    return error_response(VALIDATION_INVALID_JSON, "Invalid JSON body", status_code=400)


def auth_error(message: str = "Authentication failed", code: str = AUTH_INVALID_TOKEN) -> JSONResponse:
    """Create a 401 authentication error response."""
    return error_response(code, message, status_code=401)


def forbidden_error(message: str = "Permission denied", code: str = FORBIDDEN_INSUFFICIENT_PERMISSION) -> JSONResponse:
    """Create a 403 forbidden error response."""
    return error_response(code, message, status_code=403)


def not_found_error(resource: str, code: str = NOT_FOUND_RESOURCE) -> JSONResponse:
    """Create a 404 not found error response."""
    return error_response(code, f"{resource} not found", status_code=404)


def feature_not_enabled_error(feature: str) -> JSONResponse:
    """Create a 404 error for disabled features."""
    return error_response(FEATURE_NOT_ENABLED, f"{feature} not enabled", status_code=404)


def conflict_error(message: str, code: str = CONFLICT_ALREADY_EXISTS) -> JSONResponse:
    """Create a 409 conflict error response."""
    return error_response(code, message, status_code=409)


def internal_error(message: str = "Internal server error") -> JSONResponse:
    """Create a 500 internal error response."""
    return error_response(INTERNAL_ERROR, message, status_code=500)


def service_unavailable_error(service: str) -> JSONResponse:
    """Create a 503 service unavailable error response."""
    return error_response(SERVICE_UNAVAILABLE, f"{service} not initialized", status_code=503)
