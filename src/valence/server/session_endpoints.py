# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Session REST endpoints for Valence session ingestion.

Provides REST API for session management, message buffering, and conversation source ingestion.
Implements #470.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from starlette.requests import Request
from starlette.responses import JSONResponse

from valence.core.sessions import (
    append_message,
    append_messages,
    finalize_session,
    flush_session,
    flush_stale_sessions,
    get_messages,
    get_session,
    list_sessions,
    update_session,
    upsert_session,
)

from .session_models import (
    MessageBatchCreate,
    MessageCreate,
    SessionCreate,
    SessionUpdate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Session Management
# =============================================================================


async def upsert_session_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sessions — Upsert session (insert-if-new or touch last_activity_at)."""
    try:
        body_json = await request.json()
        body = SessionCreate(**body_json)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Invalid request body: {e}"}, status_code=400)

    resp = await upsert_session(
        session_id=body.session_id,
        platform=body.platform,
        channel=body.channel,
        participants=body.participants,
        metadata=body.metadata,
        parent_session_id=body.parent_session_id,
        subagent_label=body.subagent_label,
        subagent_model=body.subagent_model,
        subagent_task=body.subagent_task,
    )

    if not resp.success:
        logger.warning("Session upsert failed: %s", resp.error)
        return JSONResponse({"error": resp.error}, status_code=400)

    return JSONResponse(resp.data, status_code=201)


async def list_sessions_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sessions — List sessions with optional filters."""
    params = request.query_params
    status = params.get("status")
    platform = params.get("platform")
    since_str = params.get("since")
    limit = int(params.get("limit", "100"))

    since_dt = None
    if since_str:
        try:
            since_dt = datetime.fromisoformat(since_str)
        except ValueError:
            return JSONResponse({"error": f"Invalid ISO timestamp: {since_str}"}, status_code=400)

    resp = await list_sessions(status=status, platform=platform, since=since_dt)

    if not resp.success:
        logger.warning("Session list failed: %s", resp.error)
        return JSONResponse({"error": resp.error}, status_code=500)

    # Apply limit
    return JSONResponse(resp.data[:limit])


async def search_sessions_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sessions/search — Semantic search over conversation sources."""
    params = request.query_params
    query = params.get("q")
    if not query:
        return JSONResponse({"error": "Missing query parameter 'q'"}, status_code=400)

    limit = int(params.get("limit", "10"))

    try:
        from valence.core.sources import search_sources
    except ImportError:
        return JSONResponse({"error": "Source search not available"}, status_code=501)

    # Add conversation type filter to query
    full_query = f"{query} type:conversation"
    resp = await search_sources(full_query, limit=limit)

    if not resp.success:
        logger.warning("Session search failed: %s", resp.error)
        return JSONResponse({"error": resp.error}, status_code=500)

    return JSONResponse(resp.data)


async def flush_stale_sessions_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sessions/flush-stale — Find and flush all stale sessions."""
    params = request.query_params
    stale_minutes = int(params.get("stale_minutes", "30"))

    resp = await flush_stale_sessions(stale_minutes=stale_minutes)

    if not resp.success:
        logger.warning("Stale session flush failed: %s", resp.error)
        return JSONResponse({"error": resp.error}, status_code=500)

    return JSONResponse({"flushed": resp.data, "count": len(resp.data)})


async def get_session_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sessions/{session_id} — Get session details."""
    session_id = request.path_params["session_id"]

    resp = await get_session(session_id)

    if not resp.success:
        logger.warning("Session get failed for %s: %s", session_id, resp.error)
        return JSONResponse({"error": resp.error}, status_code=404)

    return JSONResponse(resp.data)


async def update_session_endpoint(request: Request) -> JSONResponse:
    """PATCH /api/v1/sessions/{session_id} — Update session metadata/status."""
    session_id = request.path_params["session_id"]

    try:
        body_json = await request.json()
        body = SessionUpdate(**body_json)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Invalid request body: {e}"}, status_code=400)

    resp = await update_session(
        session_id=session_id,
        status=body.status,
        metadata=body.metadata,
        participants=body.participants,
        ended_at=body.ended_at,
    )

    if not resp.success:
        logger.warning("Session update failed for %s: %s", session_id, resp.error)
        status_code = 404 if resp.error and "not found" in resp.error.lower() else 400
        return JSONResponse({"error": resp.error}, status_code=status_code)

    return JSONResponse(resp.data)


# =============================================================================
# Message Management
# =============================================================================


async def append_messages_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sessions/{session_id}/messages — Append message(s) to buffer."""
    session_id = request.path_params["session_id"]

    try:
        body_json = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # Check if it's a batch (has "messages" key) or single message
    if "messages" in body_json:
        try:
            batch = MessageBatchCreate(**body_json)
            message_dicts = [msg.model_dump() for msg in batch.messages]
        except Exception as e:
            return JSONResponse({"error": f"Invalid batch request: {e}"}, status_code=400)

        resp = await append_messages(session_id, message_dicts)

        if not resp.success:
            logger.warning("Message batch append failed for %s: %s", session_id, resp.error)
            return JSONResponse({"error": resp.error}, status_code=400)

        return JSONResponse(resp.data, status_code=201)
    else:
        try:
            msg = MessageCreate(**body_json)
        except Exception as e:
            return JSONResponse({"error": f"Invalid message request: {e}"}, status_code=400)

        resp = await append_message(
            session_id=session_id,
            speaker=msg.speaker,
            role=msg.role,
            content=msg.content,
            metadata=msg.metadata,
        )

        if not resp.success:
            logger.warning("Message append failed for %s: %s", session_id, resp.error)
            return JSONResponse({"error": resp.error}, status_code=400)

        return JSONResponse(resp.data, status_code=201)


async def get_messages_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/sessions/{session_id}/messages — List messages for session."""
    session_id = request.path_params["session_id"]
    params = request.query_params

    role = params.get("role")
    since_str = params.get("since")
    chunk_index_str = params.get("chunk_index")
    limit = int(params.get("limit", "1000"))

    since_dt = None
    if since_str:
        try:
            since_dt = datetime.fromisoformat(since_str)
        except ValueError:
            return JSONResponse({"error": f"Invalid ISO timestamp: {since_str}"}, status_code=400)

    chunk_index = None
    if chunk_index_str:
        try:
            chunk_index = int(chunk_index_str)
        except ValueError:
            return JSONResponse({"error": f"Invalid chunk_index: {chunk_index_str}"}, status_code=400)

    resp = await get_messages(session_id=session_id, role=role, since=since_dt, chunk_index=chunk_index)

    if not resp.success:
        logger.warning("Message get failed for %s: %s", session_id, resp.error)
        return JSONResponse({"error": resp.error}, status_code=500)

    # Apply limit
    return JSONResponse(resp.data[:limit])


# =============================================================================
# Flush & Compile
# =============================================================================


async def flush_session_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sessions/{session_id}/flush — Flush unflushed messages to source."""
    session_id = request.path_params["session_id"]
    params = request.query_params
    compile = params.get("compile", "true").lower() == "true"

    resp = await flush_session(session_id, compile=compile)

    if not resp.success:
        logger.warning("Session flush failed for %s: %s", session_id, resp.error)
        return JSONResponse({"error": resp.error}, status_code=500)

    return JSONResponse(resp.data)


async def compile_session_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sessions/{session_id}/compile — Compile session sources into article."""
    session_id = request.path_params["session_id"]

    try:
        from valence.core.compilation import compile_article
        from valence.core.sources import search_sources
    except ImportError:
        return JSONResponse({"error": "Compilation not available"}, status_code=501)

    # Get all sources for this session
    search_resp = await search_sources(f"session_key:{session_id}")
    if not search_resp.success or not search_resp.data:
        return JSONResponse({"error": "No sources found for session"}, status_code=404)

    source_ids = [s["id"] for s in search_resp.data]
    compile_resp = await compile_article(source_ids)

    if not compile_resp.success:
        logger.warning("Session compilation failed for %s: %s", session_id, compile_resp.error)
        return JSONResponse({"error": compile_resp.error}, status_code=500)

    return JSONResponse(compile_resp.data)


async def finalize_session_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/sessions/{session_id}/finalize — Flush final messages + complete + compile."""
    session_id = request.path_params["session_id"]

    resp = await finalize_session(session_id)

    if not resp.success:
        logger.warning("Session finalize failed for %s: %s", session_id, resp.error)
        return JSONResponse({"error": resp.error}, status_code=500)

    return JSONResponse(resp.data)
