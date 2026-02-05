"""OAuth 2.1 endpoints for the Valence MCP server.

Implements:
- Protected Resource Metadata (RFC 9728)
- Authorization Server Metadata (RFC 8414)
- Authorization Endpoint with PKCE
- Token Endpoint
- Dynamic Client Registration (RFC 7591)
"""

from __future__ import annotations

import hashlib
import html
import logging
import secrets
import urllib.parse
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from .config import get_settings
from .oauth_models import (
    create_access_token,
    get_client_store,
    get_code_store,
    get_refresh_store,
    verify_pkce,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Metadata Endpoints
# ============================================================================


async def protected_resource_metadata(request: Request) -> JSONResponse:
    """RFC 9728 Protected Resource Metadata.

    This tells clients where to find the authorization server and what scopes are supported.
    """
    settings = get_settings()

    metadata = {
        "resource": settings.mcp_resource_url,
        "authorization_servers": [settings.issuer_url],
        "scopes_supported": ["mcp:tools", "mcp:resources", "mcp:prompts"],
        "bearer_methods_supported": ["header"],
        "resource_documentation": f"{settings.base_url}/docs",
    }

    return JSONResponse(metadata)


async def authorization_server_metadata(request: Request) -> JSONResponse:
    """RFC 8414 Authorization Server Metadata.

    This tells clients how to authenticate.
    """
    settings = get_settings()
    base = settings.base_url

    metadata = {
        "issuer": settings.issuer_url,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "registration_endpoint": f"{base}/oauth/register",
        "scopes_supported": ["mcp:tools", "mcp:resources", "mcp:prompts"],
        "response_types_supported": ["code"],
        "response_modes_supported": ["query"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["none"],  # Public clients
        "code_challenge_methods_supported": ["S256"],
        "service_documentation": f"{base}/docs",
    }

    return JSONResponse(metadata)


# ============================================================================
# Dynamic Client Registration (RFC 7591)
# ============================================================================


async def register_client(request: Request) -> JSONResponse:
    """Dynamic Client Registration endpoint.

    Allows clients like Claude to register themselves.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Invalid JSON body"},
            status_code=400,
        )

    # Required fields
    redirect_uris = body.get("redirect_uris")
    if not redirect_uris or not isinstance(redirect_uris, list):
        return JSONResponse(
            {"error": "invalid_request", "error_description": "redirect_uris required"},
            status_code=400,
        )

    # Validate redirect URIs
    for uri in redirect_uris:
        parsed = urllib.parse.urlparse(uri)
        if not parsed.scheme or not parsed.netloc:
            return JSONResponse(
                {"error": "invalid_redirect_uri", "error_description": f"Invalid URI: {uri}"},
                status_code=400,
            )

    # Optional fields
    client_name = body.get("client_name", "Unknown Client")
    grant_types = body.get("grant_types", ["authorization_code", "refresh_token"])
    response_types = body.get("response_types", ["code"])
    scope = body.get("scope", "mcp:tools mcp:resources")

    # Validate grant types
    valid_grant_types = {"authorization_code", "refresh_token"}
    if not set(grant_types).issubset(valid_grant_types):
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Invalid grant_types"},
            status_code=400,
        )

    # Register the client
    store = get_client_store()
    client = store.register(
        client_name=client_name,
        redirect_uris=redirect_uris,
        grant_types=grant_types,
        response_types=response_types,
        scope=scope,
    )

    # Return client information (RFC 7591 response)
    return JSONResponse(
        {
            "client_id": client.client_id,
            "client_name": client.client_name,
            "redirect_uris": client.redirect_uris,
            "grant_types": client.grant_types,
            "response_types": client.response_types,
            "scope": client.scope,
            "client_id_issued_at": int(client.created_at),
        },
        status_code=201,
    )


# ============================================================================
# Authorization Endpoint
# ============================================================================


async def authorize(request: Request) -> Response:
    """OAuth 2.1 Authorization Endpoint.

    Handles the authorization code flow with PKCE.
    """
    settings = get_settings()

    # Get query parameters
    params = dict(request.query_params)

    response_type = params.get("response_type")
    client_id = params.get("client_id")
    redirect_uri = params.get("redirect_uri")
    scope = params.get("scope", "mcp:tools")
    state = params.get("state")
    code_challenge = params.get("code_challenge")
    code_challenge_method = params.get("code_challenge_method", "S256")

    # Validate required parameters
    if response_type != "code":
        return _auth_error_response(
            redirect_uri, state, "unsupported_response_type", "Only 'code' is supported"
        )

    if not client_id:
        return HTMLResponse(_error_page("Missing client_id"), status_code=400)

    if not redirect_uri:
        return HTMLResponse(_error_page("Missing redirect_uri"), status_code=400)

    if not code_challenge:
        return _auth_error_response(
            redirect_uri, state, "invalid_request", "PKCE code_challenge required"
        )

    if code_challenge_method != "S256":
        return _auth_error_response(
            redirect_uri, state, "invalid_request", "Only S256 code_challenge_method supported"
        )

    # Validate client
    store = get_client_store()
    client = store.get(client_id)
    if not client:
        return HTMLResponse(_error_page("Unknown client_id"), status_code=400)

    if not store.validate_redirect_uri(client_id, redirect_uri):
        return HTMLResponse(_error_page("Invalid redirect_uri for this client"), status_code=400)

    # Check if this is a POST (form submission) or GET (show login)
    if request.method == "POST":
        return await _handle_authorize_post(
            request, client_id, redirect_uri, scope, state, code_challenge, code_challenge_method
        )

    # Show login form
    return HTMLResponse(_login_page(params, client.client_name))


async def _handle_authorize_post(
    request: Request,
    client_id: str,
    redirect_uri: str,
    scope: str,
    state: str | None,
    code_challenge: str,
    code_challenge_method: str,
) -> Response:
    """Handle authorization form submission."""
    settings = get_settings()

    # Parse form data
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")

    # Validate credentials using constant-time comparison to prevent timing attacks
    if not (secrets.compare_digest(username, settings.oauth_username or "") and
            secrets.compare_digest(password, settings.oauth_password or "")):
        # Re-show login with error
        params = dict(request.query_params)
        return HTMLResponse(_login_page(params, "Valence", error="Invalid username or password"))

    # Check if OAuth password is configured
    if not settings.oauth_password:
        return HTMLResponse(
            _error_page("OAuth password not configured. Set VALENCE_OAUTH_PASSWORD."),
            status_code=500,
        )

    # Create authorization code
    code_store = get_code_store()
    code = code_store.create(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        user_id=username,
    )

    # Redirect back to client with code
    redirect_params = {"code": code}
    if state:
        redirect_params["state"] = state

    redirect_url = f"{redirect_uri}?{urllib.parse.urlencode(redirect_params)}"
    return RedirectResponse(redirect_url, status_code=302)


def _auth_error_response(
    redirect_uri: str | None,
    state: str | None,
    error: str,
    description: str,
) -> Response:
    """Create an OAuth error response."""
    if not redirect_uri:
        return HTMLResponse(_error_page(f"{error}: {description}"), status_code=400)

    params = {"error": error, "error_description": description}
    if state:
        params["state"] = state

    redirect_url = f"{redirect_uri}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(redirect_url, status_code=302)


# ============================================================================
# Token Endpoint
# ============================================================================


async def token(request: Request) -> JSONResponse:
    """OAuth 2.1 Token Endpoint.

    Handles authorization_code and refresh_token grants.
    """
    settings = get_settings()

    # Parse form data
    try:
        form = await request.form()
    except Exception:
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Invalid form data"},
            status_code=400,
        )

    grant_type = form.get("grant_type")

    if grant_type == "authorization_code":
        return await _handle_authorization_code_grant(form)
    elif grant_type == "refresh_token":
        return await _handle_refresh_token_grant(form)
    else:
        return JSONResponse(
            {"error": "unsupported_grant_type", "error_description": f"Unknown grant_type: {grant_type}"},
            status_code=400,
        )


async def _handle_authorization_code_grant(form: Any) -> JSONResponse:
    """Handle authorization_code grant."""
    settings = get_settings()

    code = form.get("code")
    redirect_uri = form.get("redirect_uri")
    client_id = form.get("client_id")
    code_verifier = form.get("code_verifier")

    if not all([code, redirect_uri, client_id, code_verifier]):
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Missing required parameters"},
            status_code=400,
        )

    # Consume the authorization code
    code_store = get_code_store()
    auth_code = code_store.consume(code)

    if not auth_code:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Invalid or expired authorization code"},
            status_code=400,
        )

    # Validate client_id and redirect_uri match
    if auth_code.client_id != client_id:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "client_id mismatch"},
            status_code=400,
        )

    if auth_code.redirect_uri != redirect_uri:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "redirect_uri mismatch"},
            status_code=400,
        )

    # Verify PKCE
    if not verify_pkce(code_verifier, auth_code.code_challenge, auth_code.code_challenge_method):
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "PKCE verification failed"},
            status_code=400,
        )

    # Create tokens
    access_token = create_access_token(
        client_id=client_id,
        user_id=auth_code.user_id,
        scope=auth_code.scope,
        audience=settings.mcp_resource_url,
    )

    refresh_store = get_refresh_store()
    refresh_token = refresh_store.create(
        client_id=client_id,
        user_id=auth_code.user_id,
        scope=auth_code.scope,
    )

    return JSONResponse({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": settings.oauth_access_token_expiry,
        "refresh_token": refresh_token,
        "scope": auth_code.scope,
    })


async def _handle_refresh_token_grant(form: Any) -> JSONResponse:
    """Handle refresh_token grant."""
    settings = get_settings()

    refresh_token_value = form.get("refresh_token")
    client_id = form.get("client_id")

    if not refresh_token_value:
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Missing refresh_token"},
            status_code=400,
        )

    # Validate refresh token
    refresh_store = get_refresh_store()
    token_data = refresh_store.validate(refresh_token_value)

    if not token_data:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Invalid or expired refresh token"},
            status_code=400,
        )

    # Validate client_id if provided
    if client_id and token_data.client_id != client_id:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "client_id mismatch"},
            status_code=400,
        )

    # Create new access token
    access_token = create_access_token(
        client_id=token_data.client_id,
        user_id=token_data.user_id,
        scope=token_data.scope,
        audience=settings.mcp_resource_url,
    )

    return JSONResponse({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": settings.oauth_access_token_expiry,
        "scope": token_data.scope,
    })


# ============================================================================
# HTML Templates
# ============================================================================


def _login_page(params: dict[str, Any], client_name: str, error: str | None = None) -> str:
    """Generate the login page HTML."""
    # Preserve query params for form submission
    query_string = urllib.parse.urlencode(params)

    # Escape user-controlled values to prevent XSS
    safe_client_name = html.escape(client_name)

    error_html = ""
    if error:
        safe_error = html.escape(error)
        error_html = f'<div class="error">{safe_error}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Valence - Sign In</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            padding: 40px;
            max-width: 400px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            color: #1a1a2e;
            font-size: 28px;
            margin-bottom: 8px;
            text-align: center;
        }}
        .subtitle {{
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .client-name {{
            background: #f0f0f0;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 24px;
            text-align: center;
        }}
        .client-name span {{
            color: #333;
            font-weight: 500;
        }}
        .error {{
            background: #fee;
            border: 1px solid #fcc;
            color: #c00;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }}
        label {{
            display: block;
            color: #333;
            font-weight: 500;
            margin-bottom: 8px;
        }}
        input[type="text"], input[type="password"] {{
            width: 100%;
            padding: 14px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            margin-bottom: 20px;
            transition: border-color 0.2s;
        }}
        input:focus {{
            outline: none;
            border-color: #4a6cf7;
        }}
        button {{
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #4a6cf7 0%, #6366f1 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(74, 108, 247, 0.4);
        }}
        .footer {{
            margin-top: 24px;
            text-align: center;
            color: #888;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Valence</h1>
        <p class="subtitle">Personal Knowledge Substrate</p>
        <div class="client-name">
            Authorize <span>{safe_client_name}</span>
        </div>
        {error_html}
        <form method="POST" action="/oauth/authorize?{query_string}">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" required autocomplete="username">

            <label for="password">Password</label>
            <input type="password" id="password" name="password" required autocomplete="current-password">

            <button type="submit">Sign In</button>
        </form>
        <p class="footer">Your data stays on your server.</p>
    </div>
</body>
</html>"""


def _error_page(message: str) -> str:
    """Generate an error page HTML."""
    # Escape user-controlled values to prevent XSS
    safe_message = html.escape(message)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Valence</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #fff;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 40px;
            max-width: 500px;
            text-align: center;
        }}
        h1 {{ color: #ff6b6b; margin-bottom: 16px; }}
        p {{ color: #ccc; line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Error</h1>
        <p>{safe_message}</p>
    </div>
</body>
</html>"""
