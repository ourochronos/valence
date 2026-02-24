# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

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
from pathlib import Path
from typing import Any

import yaml
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from .admin_endpoints import (
    admin_embeddings_backfill,
    admin_embeddings_migrate,
    admin_embeddings_status,
    admin_maintenance,
    admin_verify_chains,
)
from .auth import get_token_store, verify_token
from .auth_helpers import AuthenticatedClient  # noqa: F401 — re-exported for backwards compat
from .config import get_settings
from .endpoints.articles import (
    create_article_endpoint,
    get_article_endpoint,
    get_provenance_endpoint,
    link_provenance_endpoint,
    search_articles_endpoint,
    trace_claim_endpoint,
    update_article_endpoint,
)
from .endpoints.sources import (
    sources_create_endpoint,
    sources_get_endpoint,
    sources_list_endpoint,
    sources_search_endpoint,
)
from .metrics import MetricsMiddleware, metrics_endpoint
from .substrate_endpoints import (
    beliefs_create_endpoint,
    beliefs_get_endpoint,
    beliefs_list_endpoint,
    beliefs_search_endpoint,
    beliefs_supersede_endpoint,
    conflicts_endpoint,
    entities_get_endpoint,
    entities_list_endpoint,
    stats_endpoint,
    tensions_list_endpoint,
    tensions_resolve_endpoint,
)

logger = logging.getLogger(__name__)


class JSONLogFormatter(logging.Formatter):
    """JSON structured log formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        import json as _json

        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return _json.dumps(log_entry)


def configure_logging(log_format: str = "text") -> None:
    """Configure logging format for the application.

    Args:
        log_format: 'text' for human-readable, 'json' for structured JSON
    """
    root = logging.getLogger()
    handler = logging.StreamHandler()

    if log_format == "json":
        handler.setFormatter(JSONLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


# Rate limiting state (in-memory, per-instance)
_rate_limits: dict[str, list[float]] = defaultdict(list)


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
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute("SELECT 1")
        health_data["database"] = "connected"
    except Exception:
        health_data["database"] = "error: connection failed"
        health_data["status"] = "degraded"

    status_code = 200 if health_data["status"] == "healthy" else 503
    return JSONResponse(health_data, status_code=status_code)


def _authenticate_request(request: Request) -> AuthenticatedClient | None:
    """Authenticate a request using either Bearer token or OAuth JWT.

    Tries legacy Bearer token first, then OAuth JWT.

    Returns:
        AuthenticatedClient if valid, None otherwise
    """
    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return None

    # Try legacy Bearer token
    legacy_token = verify_token(auth_header)
    if legacy_token is not None:
        return AuthenticatedClient(
            client_id=legacy_token.client_id,
            user_id=None,
            scope=None,
            auth_method="bearer",
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
    except Exception:
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
            "error": {
                "code": -32600,
                "message": "Invalid Request: missing or invalid method",
            },
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
    except Exception:
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
    from ..mcp import SUBSTRATE_TOOLS, TOOL_HANDLERS

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
        for tool in SUBSTRATE_TOOLS:
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

        # Route to handler
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            raise MethodNotFoundError(f"Unknown tool: {tool_name}")

        result = handler(**arguments)

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
                            "text": f"Before responding, use knowledge_search to search for relevant knowledge about: {topic}",
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
    return """You have access to Valence v2, a personal knowledge substrate that stores sources,
compiles knowledge articles, tracks provenance, and surfaces contentions.

## Required Behaviors

### 1. Query Before Answering
BEFORE responding to questions about any topic that may have been discussed, documented, or
learned previously, you MUST call `knowledge_search` to search for relevant knowledge.
This ensures your responses are grounded in accumulated knowledge rather than assumptions.

### 2. Ingest Information Proactively
When new information arrives (documents, conversations, observations, tool outputs),
call `source_ingest` to store it. This is cheap and fast — do it before compilation.

### 3. Create Articles for Synthesised Knowledge
When you identify knowledge worth preserving as a durable article:
- Use `article_create` for direct article authorship
- Use `article_compile` to let the LLM synthesise from multiple sources

### 4. Surface and Resolve Contentions
Periodically call `contention_list` to identify contradictions in the knowledge base.
Resolve them with `contention_resolve` when you have enough context.

## Tool Usage Priority
1. `knowledge_search` — Always check first for relevant context
2. `source_ingest` — Capture raw information as it arrives
3. `article_create` / `article_compile` — Synthesise durable knowledge
4. `contention_list` / `contention_resolve` — Maintain knowledge consistency

## Key Principle
Search first, ingest early, compile deliberately, resolve contentions promptly."""


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

**FIRST** use `knowledge_search` to search for relevant knowledge, **THEN** respond with grounded information.

### 2. PROACTIVELY Capture Knowledge
Use `source_ingest` or `insight_extract` when:
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

### Knowledge Substrate (v2)
- `knowledge_search` - Search for existing knowledge (USE FIRST)
- `source_ingest` - Capture raw information as it arrives
- `article_create` - Store new synthesised knowledge
- `article_update` - Update existing knowledge with versioning
- `article_get` - Get detailed article information
- `article_compile` - Compile sources into an article via LLM
- `entity_get`, `entity_search` - Work with entities (people, tools, concepts)
- `contention_list`, `contention_resolve` - Handle contradictions

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
    from ..mcp import SUBSTRATE_TOOLS

    lines = ["# Valence Tool Reference\n"]

    lines.append("## Knowledge Substrate Tools\n")
    for tool in SUBSTRATE_TOOLS:
        # Just first line of description
        desc = tool.description.split("\n")[0] if tool.description else ""
        lines.append(f"- **{tool.name}**: {desc}")

    return "\n".join(lines)


def _get_context_prompt() -> str:
    """Get the context injection prompt."""
    return """You have access to Valence v2, a personal knowledge substrate.

**Core Behaviors:**
1. **Query First**: Before answering questions about past context, decisions, or preferences,
   use `knowledge_search` to search the knowledge base.
2. **Capture Proactively**: When valuable information emerges (decisions, preferences, insights),
   use `source_ingest` to store it for future reference.

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

    from ..mcp import SUBSTRATE_TOOLS

    response_data: dict[str, Any] = {
        "server": settings.server_name,
        "version": settings.server_version,
        "apiVersion": "v1",
        "protocol": "mcp",
        "protocolVersion": "2024-11-05",
        "transport": "http",
        "tools": {
            "total": len(SUBSTRATE_TOOLS),
        },
        "endpoints": {
            "mcp": "/api/v1/mcp",
            "health": "/api/v1/health",
            "info": "/",
            "sources": "/api/v1/sources",
            "articles": "/api/v1/articles",
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


async def _embedding_backfill_loop(interval_seconds: int = 300) -> None:
    """Periodically check for and backfill beliefs missing embeddings.

    Runs as a background task during server lifetime. Default interval: 5 minutes.
    Configurable via VALENCE_BACKFILL_INTERVAL env var (seconds, 0 to disable).
    """
    import asyncio
    import os

    interval = int(os.environ.get("VALENCE_BACKFILL_INTERVAL", str(interval_seconds)))
    if interval <= 0:
        logger.info("Embedding backfill loop disabled (interval=0)")
        return

    # Brief startup delay to let the server finish initializing
    await asyncio.sleep(5)

    while True:
        try:
            from valence.core.db import get_cursor
            from valence.core.embeddings import generate_embedding, vector_to_pgvector

            with get_cursor() as cur:
                cur.execute("SELECT id, content FROM articles WHERE embedding IS NULL AND status = 'active' LIMIT 50")
                rows = cur.fetchall()

            if rows:
                logger.info(f"Embedding backfill: found {len(rows)} articles without embeddings")
                backfilled = 0
                for row in rows:
                    try:
                        embedding = generate_embedding(row["content"])
                        embedding_str = vector_to_pgvector(embedding)
                        with get_cursor() as cur:
                            cur.execute(
                                "UPDATE articles SET embedding = %s::vector WHERE id = %s",
                                (embedding_str, row["id"]),
                            )
                        backfilled += 1
                    except Exception:
                        logger.warning(f"Embedding backfill failed for article {row['id']}", exc_info=True)
                logger.info(f"Embedding backfill: completed {backfilled}/{len(rows)}")
            else:
                logger.debug("Embedding backfill: all articles have embeddings")
        except ImportError:
            logger.debug("Embedding backfill: our_embeddings not available, skipping")
        except Exception:
            logger.warning("Embedding backfill loop error", exc_info=True)

        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: Starlette):
    """Application lifespan handler."""
    import asyncio

    settings = get_settings()
    logger.info(f"Starting Valence MCP server on {settings.host}:{settings.port}")

    # Initialize legacy token store
    get_token_store(settings.token_file)
    logger.info(f"Token store initialized from {settings.token_file}")

    # Start background embedding backfill task (#399)
    backfill_task = asyncio.create_task(_embedding_backfill_loop())

    yield

    # Cancel background tasks on shutdown
    backfill_task.cancel()
    logger.info("Valence MCP server shutting down")


def create_app() -> Starlette:
    """Create the Starlette ASGI application."""
    settings = get_settings()

    # API version prefix for all REST endpoints
    API_V1 = "/api/v1"  # noqa: N806 - intentionally constant-style naming

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
        # -----------------------------------------------------------------------
        # v2 Sources endpoints (C1)
        # -----------------------------------------------------------------------
        Route(f"{API_V1}/sources", sources_list_endpoint, methods=["GET"]),
        Route(f"{API_V1}/sources", sources_create_endpoint, methods=["POST"]),
        Route(f"{API_V1}/sources/search", sources_search_endpoint, methods=["POST"]),
        Route(f"{API_V1}/sources/{{source_id}}", sources_get_endpoint, methods=["GET"]),
        # -----------------------------------------------------------------------
        # v2 Articles endpoints (C2, C3, C5)
        # -----------------------------------------------------------------------
        Route(f"{API_V1}/articles", search_articles_endpoint, methods=["GET"]),
        Route(f"{API_V1}/articles", create_article_endpoint, methods=["POST"]),
        Route(f"{API_V1}/articles/search", search_articles_endpoint, methods=["POST"]),
        Route(f"{API_V1}/articles/{{article_id}}", get_article_endpoint, methods=["GET"]),
        Route(f"{API_V1}/articles/{{article_id}}", update_article_endpoint, methods=["PUT"]),
        Route(
            f"{API_V1}/articles/{{article_id}}/provenance",
            get_provenance_endpoint,
            methods=["GET"],
        ),
        Route(
            f"{API_V1}/articles/{{article_id}}/provenance/link",
            link_provenance_endpoint,
            methods=["POST"],
        ),
        Route(
            f"{API_V1}/articles/{{article_id}}/provenance/trace",
            trace_claim_endpoint,
            methods=["POST"],
        ),
        # -----------------------------------------------------------------------
        # Legacy Substrate REST endpoints (kept for backward compatibility)
        # -----------------------------------------------------------------------
        Route(f"{API_V1}/beliefs", beliefs_list_endpoint, methods=["GET"]),
        Route(f"{API_V1}/beliefs", beliefs_create_endpoint, methods=["POST"]),
        Route(f"{API_V1}/beliefs/search", beliefs_search_endpoint, methods=["GET"]),
        Route(f"{API_V1}/beliefs/conflicts", conflicts_endpoint, methods=["GET"]),
        Route(f"{API_V1}/beliefs/{{belief_id}}", beliefs_get_endpoint, methods=["GET"]),
        Route(f"{API_V1}/beliefs/{{belief_id}}/supersede", beliefs_supersede_endpoint, methods=["POST"]),
        Route(f"{API_V1}/entities", entities_list_endpoint, methods=["GET"]),
        Route(f"{API_V1}/entities/{{id}}", entities_get_endpoint, methods=["GET"]),
        Route(f"{API_V1}/tensions", tensions_list_endpoint, methods=["GET"]),
        Route(f"{API_V1}/tensions/{{id}}/resolve", tensions_resolve_endpoint, methods=["POST"]),
        # Stats (Issue #396)
        Route(f"{API_V1}/stats", stats_endpoint, methods=["GET"]),
        # Admin endpoints (Issue #396)
        Route(f"{API_V1}/admin/maintenance", admin_maintenance, methods=["POST"]),
        Route(f"{API_V1}/admin/embeddings/status", admin_embeddings_status, methods=["GET"]),
        Route(f"{API_V1}/admin/embeddings/backfill", admin_embeddings_backfill, methods=["POST"]),
        Route(f"{API_V1}/admin/embeddings/migrate", admin_embeddings_migrate, methods=["POST"]),
        Route(f"{API_V1}/admin/verify-chains", admin_verify_chains, methods=["GET"]),
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
    configure_logging(settings.log_format)

    logger.info(f"Starting Valence HTTP MCP server on {settings.host}:{settings.port}")

    uvicorn.run(
        "valence.server.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    run()
