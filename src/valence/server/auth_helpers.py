# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Authentication and authorization helpers for REST endpoints.

Provides authenticate() and require_scope() for use by endpoint handlers.
Supports both legacy Bearer tokens and OAuth 2.1 JWT tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import JSONResponse

from .auth import verify_token
from .errors import AUTH_MISSING_TOKEN, FORBIDDEN_INSUFFICIENT_PERMISSION, auth_error, forbidden_error

logger = logging.getLogger(__name__)


@dataclass
class AuthenticatedClient:
    """Represents an authenticated client from either auth method."""

    client_id: str
    user_id: str | None = None
    scope: str | None = None
    auth_method: str = "bearer"  # "bearer" or "oauth"


def authenticate(request: Request) -> AuthenticatedClient | JSONResponse:
    """Authenticate a request. Returns client on success, error JSONResponse on failure.

    Tries legacy Bearer token first, then OAuth JWT.

    Usage in endpoints::

        client = authenticate(request)
        if isinstance(client, JSONResponse):
            return client
        # client is AuthenticatedClient
    """
    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return auth_error("Missing or invalid authentication token", code=AUTH_MISSING_TOKEN)

    # Try legacy Bearer token
    legacy_token = verify_token(auth_header)
    if legacy_token is not None:
        return AuthenticatedClient(
            client_id=legacy_token.client_id,
            user_id=None,
            scope=None,
            auth_method="bearer",
        )

    return auth_error("Invalid authentication token", code=AUTH_MISSING_TOKEN)


def require_scope(client: AuthenticatedClient, scope: str) -> JSONResponse | None:
    """Check if client has the required scope. Returns 403 response or None.

    For backwards compatibility, bearer tokens (which always have mcp:access)
    are granted all REST access. OAuth tokens must have the specific scope
    or the mcp:access catch-all.
    """
    # Bearer tokens always get full access (they have mcp:access by default)
    if client.auth_method == "bearer":
        return None

    # OAuth tokens: check scope string
    if client.scope:
        granted_scopes = client.scope.split()
        if scope in granted_scopes or "mcp:access" in granted_scopes:
            return None

    return forbidden_error(
        f"Insufficient scope. Required: {scope}",
        code=FORBIDDEN_INSUFFICIENT_PERMISSION,
    )
