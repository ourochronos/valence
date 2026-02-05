"""Starlette ASGI application for the Valence HTTP MCP server.

Provides HTTP transport for the unified MCP server with authentication,
health checks, and rate limiting.

Supports both:
- Legacy Bearer token authentication (for Claude Code)
- OAuth 2.1 with PKCE (for Claude mobile/web)
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from .auth import get_token_store, verify_token
from .compliance_endpoints import (
    delete_user_data_endpoint,
    get_deletion_verification_endpoint,
)
from .config import get_settings
from .corroboration_endpoints import (
    belief_corroboration_endpoint,
    most_corroborated_beliefs_endpoint,
)
from .federation_endpoints import (
    federation_status,
    vfp_node_metadata,
    vfp_trust_anchors,
)
from .metrics import MetricsMiddleware, metrics_endpoint
from .notification_endpoints import (
    acknowledge_notification_endpoint,
    list_notifications_endpoint,
)
from .oauth import (
    authorization_server_metadata,
    authorize,
    protected_resource_metadata,
    register_client,
    token,
)
from .oauth_models import get_client_store, get_code_store, get_refresh_store, verify_access_token
from .sharing_endpoints import (
    get_share_endpoint,
    list_shares_endpoint,
    revoke_share_endpoint,
    share_belief_endpoint,
)

logger = logging.getLogger(__name__)

# Rate limiting state (in-memory, per-instance)
_rate_limits: dict[str, list[float]] = defaultdict(list)


@dataclass
class AuthenticatedClient:
    """Represents an authenticated client from either auth method."""

    client_id: str
    user_id: str | None = None
    scope: str | None = None
    auth_method: str = "bearer"  # "bearer" or "oauth"


def _check_rate_limit(client_id: str, rpm_limit: int) -> bool:
    """Check if client is within rate limit.

    Args:
        client_id: Client identifier
        rpm_limit: Requests per minute limit

    Returns:
        True if request is allowed, False if rate limited
    """
    now = time.time()
    window_start = now - 60

    # Clean old entries
    _rate_limits[client_id] = [t for t in _rate_limits[client_id] if t > window_start]

    # Check limit
    if len(_rate_limits[client_id]) >= rpm_limit:
        return False

    # Record request
    _rate_limits[client_id].append(now)
    return True


async def health_endpoint(request: Request) -> JSONResponse:
    """Health check endpoint."""
    settings = get_settings()

    # Basic health check - just verify we can respond
    health_data: dict[str, Any] = {
        "status": "healthy",
        "server": settings.server_name,
        "version": settings.server_version,
    }

    # Optionally check database connectivity
    try:
        from ..core.db import get_cursor

        with get_cursor() as cur:
            cur.execute("SELECT 1")
        health_data["database"] = "connected"
    except Exception as e:  # Intentionally broad: health check should report all errors
        health_data["database"] = f"error: {str(e)}"
        health_data["status"] = "degraded"

    status_code = 200 if health_data["status"] == "healthy" else 503
    return JSONResponse(health_data, status_code=status_code)


def _authenticate_request(request: Request) -> AuthenticatedClient | None:
    """Authenticate a request using either Bearer token or OAuth JWT.

    Tries legacy Bearer token first, then OAuth JWT.

    Returns:
        AuthenticatedClient if valid, None otherwise
    """
    settings = get_settings()
    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return None

    token_value = auth_header[7:]  # Strip "Bearer "

    # Try legacy Bearer token first
    legacy_token = verify_token(auth_header)
    if legacy_token is not None:
        return AuthenticatedClient(
            client_id=legacy_token.client_id,
            user_id=None,
            scope=None,
            auth_method="bearer",
        )

    # Try OAuth JWT token
    if settings.oauth_enabled:
        payload = verify_access_token(token_value, settings.mcp_resource_url)
        if payload is not None:
            return AuthenticatedClient(
                client_id=payload.get("client_id", "unknown"),
                user_id=payload.get("sub"),
                scope=payload.get("scope"),
                auth_method="oauth",
            )

    return None


async def mcp_endpoint(request: Request) -> Response:
    """MCP endpoint for tool calls.

    Handles MCP JSON-RPC requests over HTTP with authentication.
    Supports both legacy Bearer tokens and OAuth 2.1 JWT tokens.
    """
    settings = get_settings()

    # Authenticate using either method
    client = _authenticate_request(request)

    if client is None:
        # Return 401 with WWW-Authenticate header per RFC 9728
        resource_metadata_url = f"{settings.base_url}/.well-known/oauth-protected-resource"
        www_auth = f'Bearer realm="mcp", resource_metadata="{resource_metadata_url}"'

        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32001,
                    "message": "Unauthorized: Invalid or missing authentication token",
                },
                "id": None,
            },
            status_code=401,
            headers={"WWW-Authenticate": www_auth},
        )

    # Rate limit check
    if not _check_rate_limit(client.client_id, settings.rate_limit_rpm):
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "error": {"code": -32002, "message": "Rate limited: Too many requests"},
                "id": None,
            },
            status_code=429,
        )

    try:
        # Parse JSON-RPC request
        body = await request.body()
        rpc_request = json.loads(body)

        # Handle batch requests
        if isinstance(rpc_request, list):
            responses = []
            for req in rpc_request:
                response = await _handle_rpc_request(req)
                if response is not None:  # Notifications don't return responses
                    responses.append(response)
            return JSONResponse(responses)

        # Handle single request
        response = await _handle_rpc_request(rpc_request)
        if response is None:
            # Notification - return empty 204
            return Response(status_code=204)
        return JSONResponse(response)

    except json.JSONDecodeError as e:
        logger.warning(f"MCP parse error: {e}")
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None,
            },
            status_code=400,
        )
    except Exception as e:  # Intentionally broad: top-level MCP endpoint handler
        logger.exception("Error handling MCP request")
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": "Internal error"},
                "id": None,
            },
            status_code=500,
        )


async def _handle_rpc_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle a single JSON-RPC request.

    Args:
        request: The JSON-RPC request object

    Returns:
        JSON-RPC response object, or None for notifications
    """
    # Validate JSON-RPC structure
    if request.get("jsonrpc") != "2.0":
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32600,
                "message": "Invalid request: missing or wrong jsonrpc version",
            },
            "id": request.get("id"),
        }

    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")

    # Notifications have no id
    is_notification = request_id is None

    if not method or not isinstance(method, str):
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": "Invalid Request: missing or invalid method"},
            "id": request_id,
        }

    try:
        result = await _dispatch_method(method, params)

        if is_notification:
            return None

        return {"jsonrpc": "2.0", "result": result, "id": request_id}

    except MethodNotFoundError as e:
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": str(e)},
            "id": request_id,
        }
    except InvalidParamsError as e:
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": str(e)},
            "id": request_id,
        }
    except Exception as e:  # Intentionally broad: top-level method handler
        logger.exception(f"Error in method {method}")
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": "Internal error"},
            "id": request_id,
        }


class MethodNotFoundError(Exception):
    pass


class InvalidParamsError(Exception):
    pass


async def _dispatch_method(method: str, params: dict[str, Any]) -> Any:
    """Dispatch a method call to the appropriate handler.

    Args:
        method: The JSON-RPC method name
        params: Method parameters

    Returns:
        Method result

    Raises:
        MethodNotFoundError: If method is not found
        InvalidParamsError: If parameters are invalid
    """
    from ..substrate.tools import SUBSTRATE_TOOLS, handle_substrate_tool
    from ..vkb.tools import VKB_TOOLS, handle_vkb_tool

    settings = get_settings()

    # MCP standard methods
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {
                "name": settings.server_name,
                "version": settings.server_version,
            },
            "instructions": _get_server_instructions(),
        }

    elif method == "initialized":
        # Client acknowledging initialization - nothing to return
        return {}

    elif method == "ping":
        return {}

    elif method == "tools/list":
        tools = []
        for tool in SUBSTRATE_TOOLS + VKB_TOOLS:
            # Add version to inputSchema for tool versioning
            schema_with_version = {
                **tool.inputSchema,
                "x-version": "1.0",  # Tool schema version for API versioning
            }
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": schema_with_version,
                }
            )
        return {"tools": tools}

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            raise InvalidParamsError("Missing tool name")

        # Route to appropriate handler
        substrate_tool_names = [t.name for t in SUBSTRATE_TOOLS]
        vkb_tool_names = [t.name for t in VKB_TOOLS]

        if tool_name in substrate_tool_names:
            result = handle_substrate_tool(tool_name, arguments)
        elif tool_name in vkb_tool_names:
            result = handle_vkb_tool(tool_name, arguments)
        else:
            raise MethodNotFoundError(f"Unknown tool: {tool_name}")

        # Return in MCP format
        return {
            "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}],
            "isError": not result.get("success", True),
        }

    elif method == "resources/list":
        return {
            "resources": [
                {
                    "uri": "valence://instructions",
                    "name": "Valence Usage Instructions",
                    "description": "Guidelines for using Valence knowledge tools effectively",
                    "mimeType": "text/markdown",
                },
                {
                    "uri": "valence://tools",
                    "name": "Valence Tool Reference",
                    "description": "Quick reference for all available tools",
                    "mimeType": "text/markdown",
                },
            ]
        }

    elif method == "resources/read":
        uri = params.get("uri")
        if uri == "valence://instructions":
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/markdown",
                        "text": _get_usage_instructions(),
                    }
                ]
            }
        elif uri == "valence://tools":
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/markdown",
                        "text": _get_tool_reference(),
                    }
                ]
            }
        else:
            raise InvalidParamsError(f"Unknown resource: {uri}")

    elif method == "prompts/list":
        return {
            "prompts": [
                {
                    "name": "valence-context",
                    "description": "Inject Valence context and behavioral guidelines",
                    "arguments": [],
                },
                {
                    "name": "recall-context",
                    "description": "Query knowledge base for relevant context on a topic",
                    "arguments": [
                        {
                            "name": "topic",
                            "description": "Topic to recall context about",
                            "required": True,
                        }
                    ],
                },
            ]
        }

    elif method == "prompts/get":
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name == "valence-context":
            return {
                "description": "Valence context and behavioral guidelines",
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": _get_context_prompt()},
                    }
                ],
            }
        elif name == "recall-context":
            topic = arguments.get("topic", "")
            return {
                "description": f"Recall context about: {topic}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"Before responding, use belief_query to search for relevant knowledge about: {topic}",
                        },
                    }
                ],
            }
        else:
            raise InvalidParamsError(f"Unknown prompt: {name}")

    else:
        raise MethodNotFoundError(f"Method not found: {method}")


def _get_server_instructions() -> str:
    """Get server instructions for behavioral conditioning.

    These instructions are returned in the initialize response and guide
    Claude's behavior when using Valence tools.
    """
    return """You have access to Valence, a personal knowledge substrate that stores beliefs, tracks conversations, and learns patterns over time.

## Required Behaviors

### 1. Query Before Answering
BEFORE responding to questions about past context, decisions, preferences, or any topic that may have been discussed previously, you MUST call `belief_query` to search for relevant knowledge. This ensures your responses are grounded in accumulated knowledge rather than assumptions.

### 2. Capture Knowledge Proactively
When valuable information emerges during conversation, you SHOULD capture it without being asked:
- Use `belief_create` for decisions, preferences, facts, or insights
- Use `insight_extract` when in an active session to link the belief to conversation context
- Include relevant entities (people, tools, concepts) when creating beliefs

### 3. Track Sessions
- Call `session_start` at the beginning of substantive conversations
- Call `session_end` with a summary and themes when concluding
- This enables context continuity across conversations

### 4. Record Patterns
When you observe recurring themes, preferences, or behaviors across conversations, use `pattern_record` to capture them for future reference.

## Tool Usage Priority
1. `belief_query` - Always check first for relevant context
2. `belief_create` / `insight_extract` - Capture new knowledge as it emerges
3. `session_start` / `session_end` - Maintain conversation continuity
4. `pattern_record` - Track recurring behaviors and preferences

## Key Principle
Query first, respond with grounded information, capture insights for the future."""


def _get_usage_instructions() -> str:
    """Get the usage instructions resource content."""
    return """# Valence Knowledge Substrate

## Core Principle
Query first, respond with grounded information, capture insights for the future.

## Required Behaviors

### 1. ALWAYS Query Before Answering
When the user asks about:
- Past decisions or discussions
- User preferences or values
- Technical approaches previously explored
- Any topic that may have been discussed before

**FIRST** use `belief_query` to search for relevant knowledge, **THEN** respond with grounded information.

### 2. PROACTIVELY Capture Knowledge
Use `belief_create` or `insight_extract` when:
- A decision is made with clear rationale
- User expresses a preference or value
- A problem is solved with a novel approach
- Important factual information is shared
- Architectural or design choices are finalized

### 3. Track Sessions
- Call `session_start` at the beginning of conversations
- Call `session_end` with summary and themes when concluding
- This enables session-based context for future conversations

### 4. Record Patterns
When you notice recurring themes, preferences, or behaviors across multiple conversations,
use `pattern_record` to capture them.

## Tool Categories

### Knowledge Substrate (Beliefs)
- `belief_query` - Search for existing knowledge (USE FIRST)
- `belief_create` - Store new knowledge
- `belief_supersede` - Update existing knowledge with history
- `belief_get` - Get detailed belief information
- `entity_get`, `entity_search` - Work with entities (people, tools, concepts)
- `tension_list`, `tension_resolve` - Handle contradictions

### Conversation Tracking (VKB)
- `session_start`, `session_end`, `session_get`, `session_list` - Session management
- `exchange_add`, `exchange_list` - Conversation turn tracking
- `pattern_record`, `pattern_reinforce`, `pattern_list` - Behavioral patterns
- `insight_extract`, `insight_list` - Extract knowledge from conversations

## Best Practices

1. **Be proactive** - Don't wait to be asked to query or capture knowledge
2. **Be specific** - Use detailed queries and create focused beliefs
3. **Link entities** - Connect beliefs to relevant people, tools, and concepts
4. **Maintain context** - Use sessions to group related conversations
5. **Resolve tensions** - When you find conflicting information, work to resolve it
"""


def _get_tool_reference() -> str:
    """Get the tool reference resource content."""
    from ..substrate.tools import SUBSTRATE_TOOLS
    from ..vkb.tools import VKB_TOOLS

    lines = ["# Valence Tool Reference\n"]

    lines.append("## Knowledge Substrate Tools\n")
    for tool in SUBSTRATE_TOOLS:
        # Just first line of description
        desc = tool.description.split("\n")[0] if tool.description else ""
        lines.append(f"- **{tool.name}**: {desc}")

    lines.append("\n## Conversation Tracking Tools\n")
    for tool in VKB_TOOLS:
        desc = tool.description.split("\n")[0] if tool.description else ""
        lines.append(f"- **{tool.name}**: {desc}")

    return "\n".join(lines)


def _get_context_prompt() -> str:
    """Get the context injection prompt."""
    return """You have access to Valence, a personal knowledge substrate.

**Core Behaviors:**
1. **Query First**: Before answering questions about past context, decisions, or preferences,
   use `belief_query` to search the knowledge base.
2. **Capture Proactively**: When valuable information emerges (decisions, preferences, insights),
   use `belief_create` or `insight_extract` to store it for future reference.
3. **Track Sessions**: Use `session_start` and `session_end` to maintain conversation context.

This ensures your responses are grounded in accumulated knowledge and that new insights
are preserved for future conversations.
"""


# ============================================================================
# OpenAPI Endpoints
# ============================================================================

# Cache for the OpenAPI spec
_openapi_spec_cache: dict | None = None


def _load_openapi_spec() -> dict:
    """Load and cache the OpenAPI specification."""
    global _openapi_spec_cache
    if _openapi_spec_cache is None:
        spec_path = Path(__file__).parent.parent.parent.parent / "docs" / "openapi.yaml"
        if spec_path.exists():
            with open(spec_path) as f:
                _openapi_spec_cache = yaml.safe_load(f)
        else:
            # Fallback minimal spec if file not found
            _openapi_spec_cache = {
                "openapi": "3.0.3",
                "info": {
                    "title": "Valence API",
                    "version": "1.0.0",
                },
                "paths": {},
            }
    return _openapi_spec_cache


async def openapi_spec_endpoint(request: Request) -> JSONResponse:
    """Serve the OpenAPI specification as JSON."""
    spec = _load_openapi_spec()
    return JSONResponse(spec)


async def swagger_ui_endpoint(request: Request) -> HTMLResponse:
    """Serve Swagger UI for interactive API documentation."""
    settings = get_settings()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Valence API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>
        html {{
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }}
        *,
        *:before,
        *:after {{
            box-sizing: inherit;
        }}
        body {{
            margin: 0;
            background: #fafafa;
        }}
        .swagger-ui .topbar {{
            display: none;
        }}
        .swagger-ui .info {{
            margin: 20px 0;
        }}
        .swagger-ui .info .title {{
            font-size: 36px;
            color: #1a1a2e;
        }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            window.ui = SwaggerUIBundle({{
                url: "{settings.base_url}/api/v1/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                persistAuthorization: true,
                displayRequestDuration: true,
                tryItOutEnabled: true,
                requestInterceptor: function(request) {{
                    // Add any custom request modifications here
                    return request;
                }},
            }});
        }};
    </script>
</body>
</html>"""
    return HTMLResponse(html)


async def info_endpoint(request: Request) -> JSONResponse:
    """Server info endpoint (no auth required)."""
    settings = get_settings()

    from ..substrate.tools import SUBSTRATE_TOOLS
    from ..vkb.tools import VKB_TOOLS

    response_data: dict[str, Any] = {
        "server": settings.server_name,
        "version": settings.server_version,
        "apiVersion": "v1",
        "protocol": "mcp",
        "protocolVersion": "2024-11-05",
        "transport": "http",
        "tools": {
            "substrate": len(SUBSTRATE_TOOLS),
            "vkb": len(VKB_TOOLS),
            "total": len(SUBSTRATE_TOOLS) + len(VKB_TOOLS),
        },
        "endpoints": {
            "mcp": "/api/v1/mcp",
            "health": "/api/v1/health",
            "info": "/",
            "federation": "/api/v1/federation/status",
            "beliefs": "/api/v1/beliefs",
            "openapi": "/api/v1/openapi.json",
            "docs": "/api/v1/docs",
            "metrics": "/metrics",
        },
        "authentication": {
            "methods": ["bearer"],
        },
    }

    # Add OAuth endpoints if enabled
    if settings.oauth_enabled:
        response_data["authentication"]["methods"].append("oauth2")
        response_data["authentication"]["oauth"] = {
            "authorization_server_metadata": "/.well-known/oauth-authorization-server",
            "protected_resource_metadata": "/.well-known/oauth-protected-resource",
            "authorization_endpoint": "/api/v1/oauth/authorize",
            "token_endpoint": "/api/v1/oauth/token",
            "registration_endpoint": "/api/v1/oauth/register",
        }

    return JSONResponse(response_data)


@asynccontextmanager
async def lifespan(app: Starlette):
    """Application lifespan handler."""
    settings = get_settings()
    logger.info(f"Starting Valence MCP server on {settings.host}:{settings.port}")

    # Initialize legacy token store
    get_token_store(settings.token_file)
    logger.info(f"Token store initialized from {settings.token_file}")

    # Initialize OAuth stores if enabled
    if settings.oauth_enabled:
        get_client_store()
        get_code_store()
        get_refresh_store()
        logger.info(f"OAuth stores initialized (clients file: {settings.oauth_clients_file})")

        if not settings.oauth_password:
            logger.warning(
                "OAuth password not configured! Set VALENCE_OAUTH_PASSWORD environment variable."
            )

    yield

    logger.info("Valence MCP server shutting down")


def create_app() -> Starlette:
    """Create the Starlette ASGI application."""
    settings = get_settings()

    # API version prefix for all REST endpoints
    API_V1 = "/api/v1"

    # Define routes
    routes = [
        # Root info (no version prefix - serves as discovery endpoint)
        Route("/", info_endpoint, methods=["GET"]),
        # Versioned API endpoints
        Route(f"{API_V1}/health", health_endpoint, methods=["GET"]),
        Route(f"{API_V1}/mcp", mcp_endpoint, methods=["POST"]),
        # OpenAPI documentation
        Route(f"{API_V1}/openapi.json", openapi_spec_endpoint, methods=["GET"]),
        Route(f"{API_V1}/docs", swagger_ui_endpoint, methods=["GET"]),
        # OAuth 2.1 endpoints (well-known paths per RFC, no version prefix)
        Route(
            "/.well-known/oauth-protected-resource", protected_resource_metadata, methods=["GET"]
        ),
        Route(
            "/.well-known/oauth-authorization-server",
            authorization_server_metadata,
            methods=["GET"],
        ),
        Route(f"{API_V1}/oauth/register", register_client, methods=["POST"]),
        Route(f"{API_V1}/oauth/authorize", authorize, methods=["GET", "POST"]),
        Route(f"{API_V1}/oauth/token", token, methods=["POST"]),
        # Federation discovery endpoints (well-known paths, no version prefix)
        Route("/.well-known/vfp-node-metadata", vfp_node_metadata, methods=["GET"]),
        Route("/.well-known/vfp-trust-anchors", vfp_trust_anchors, methods=["GET"]),
        # Federation API endpoints (versioned)
        Route(f"{API_V1}/federation/status", federation_status, methods=["GET"]),
        # Corroboration endpoints (versioned)
        Route(
            f"{API_V1}/beliefs/{{belief_id}}/corroboration",
            belief_corroboration_endpoint,
            methods=["GET"],
        ),
        Route(
            f"{API_V1}/beliefs/most-corroborated",
            most_corroborated_beliefs_endpoint,
            methods=["GET"],
        ),
        # Compliance endpoints (Issue #25: GDPR deletion)
        Route(f"{API_V1}/users/{{id}}/data", delete_user_data_endpoint, methods=["DELETE"]),
        Route(
            f"{API_V1}/tombstones/{{id}}/verification",
            get_deletion_verification_endpoint,
            methods=["GET"],
        ),
        # Sharing endpoints (Issue #50: share() API, Issue #54: revoke API)
        Route(f"{API_V1}/share", share_belief_endpoint, methods=["POST"]),
        Route(f"{API_V1}/shares", list_shares_endpoint, methods=["GET"]),
        Route(f"{API_V1}/shares/{{id}}", get_share_endpoint, methods=["GET"]),
        Route(f"{API_V1}/shares/{{id}}/revoke", revoke_share_endpoint, methods=["POST"]),
        # Notification endpoints (Issue #55: revocation propagation)
        Route(f"{API_V1}/notifications", list_notifications_endpoint, methods=["GET"]),
        Route(
            f"{API_V1}/notifications/{{id}}/acknowledge",
            acknowledge_notification_endpoint,
            methods=["POST"],
        ),
        # Prometheus metrics endpoint (Issue #138)
        Route("/metrics", metrics_endpoint, methods=["GET"]),
        Route(f"{API_V1}/metrics", metrics_endpoint, methods=["GET"]),
    ]

    # Define middleware
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "Mcp-Session-Id"],
            expose_headers=["Mcp-Session-Id"],
        ),
        Middleware(MetricsMiddleware),
    ]

    return Starlette(
        routes=routes,
        middleware=middleware,
        lifespan=lifespan,
    )


# Global app instance for uvicorn
app = create_app()


def run() -> None:
    """Run the server using uvicorn."""
    import uvicorn

    settings = get_settings()

    logger.info(f"Starting Valence HTTP MCP server on {settings.host}:{settings.port}")

    uvicorn.run(
        "valence.server.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    run()
