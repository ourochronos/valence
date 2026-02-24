# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Session tool handlers."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from valence.core.compilation import compile_article
from valence.core.sessions import (
    append_message,
    append_messages,
    finalize_session,
    flush_session,
    flush_stale_sessions,
    get_messages,
    get_session,
    list_sessions,
    upsert_session,
)
from valence.core.sources import search_sources

logger = logging.getLogger(__name__)


def session_start(
    session_id: str,
    platform: str,
    channel: str | None = None,
    participants: list[str] | None = None,
    metadata: dict | None = None,
    parent_session_id: str | None = None,
    subagent_label: str | None = None,
    subagent_model: str | None = None,
    subagent_task: str | None = None,
) -> dict[str, Any]:
    """Upsert a session (insert-if-new or update last_activity_at).

    Args:
        session_id: Unique session identifier (platform-provided).
        platform: Platform name (e.g., 'openclaw', 'claude-code').
        channel: Optional channel (e.g., 'discord', 'telegram', 'cli').
        participants: List of participant names.
        metadata: Optional JSON metadata.
        parent_session_id: Parent session ID for subagents.
        subagent_label: Label for subagent sessions.
        subagent_model: Model used for subagent.
        subagent_task: Task description for subagent.

    Returns:
        dict with success flag and session data or error.
    """
    import asyncio

    result = asyncio.run(
        upsert_session(
            session_id=session_id,
            platform=platform,
            channel=channel,
            participants=participants,
            metadata=metadata,
            parent_session_id=parent_session_id,
            subagent_label=subagent_label,
            subagent_model=subagent_model,
            subagent_task=subagent_task,
        )
    )

    if result.success:
        return {"success": True, "session": result.data}
    else:
        return {"success": False, "error": result.error}


def session_append(
    session_id: str,
    messages: list[dict[str, Any]] | None = None,
    speaker: str | None = None,
    role: str | None = None,
    content: str | None = None,
    message_metadata: dict | None = None,
) -> dict[str, Any]:
    """Append message(s) to a session buffer.

    Supports two modes:
    1. Batch mode: pass messages as a list of dicts
    2. Single mode: pass speaker, role, content directly

    Args:
        session_id: Session identifier.
        messages: List of message dicts (batch mode).
        speaker: Speaker name (single mode).
        role: Message role (single mode).
        content: Message content (single mode).
        message_metadata: Optional message-specific metadata (single mode).

    Returns:
        dict with success flag and message data or error.
    """
    import asyncio

    if messages:
        # Batch mode
        result = asyncio.run(append_messages(session_id=session_id, messages=messages))
    elif speaker and role and content:
        # Single mode
        result = asyncio.run(
            append_message(
                session_id=session_id,
                speaker=speaker,
                role=role,
                content=content,
                metadata=message_metadata,
            )
        )
    else:
        return {
            "success": False,
            "error": "Either 'messages' (batch) or 'speaker'+'role'+'content' (single) required",
        }

    if result.success:
        return {"success": True, "messages": result.data if isinstance(result.data, list) else [result.data]}
    else:
        return {"success": False, "error": result.error}


def session_flush(session_id: str, compile: bool = True) -> dict[str, Any]:
    """Flush unflushed messages to a conversation source.

    Args:
        session_id: Session identifier.
        compile: Whether to trigger compilation after flush.

    Returns:
        dict with success flag and flush summary or error.
    """
    import asyncio

    result = asyncio.run(flush_session(session_id=session_id, compile=compile))

    if result.success:
        return {"success": True, "flush": result.data}
    else:
        return {"success": False, "error": result.error}


def session_finalize(session_id: str) -> dict[str, Any]:
    """Flush final messages and mark session as completed.

    Args:
        session_id: Session identifier.

    Returns:
        dict with success flag and finalization summary or error.
    """
    import asyncio

    result = asyncio.run(finalize_session(session_id=session_id))

    if result.success:
        return {"success": True, "finalization": result.data}
    else:
        return {"success": False, "error": result.error}


def session_search(query: str, limit: int = 20) -> dict[str, Any]:
    """Semantic search over conversation sources.

    Args:
        query: Search query.
        limit: Maximum results to return.

    Returns:
        dict with success flag and search results or error.
    """
    import asyncio

    # Search for conversation-type sources
    # We can use metadata filtering if the source_search supports it,
    # or post-filter results
    result = asyncio.run(search_sources(query=query, limit=limit))

    if result.success:
        # Filter to conversation type
        conversations = [s for s in result.data if s.get("type") == "conversation"]
        return {"success": True, "results": conversations[:limit]}
    else:
        return {"success": False, "error": result.error}


def session_list(
    status: str | None = None,
    platform: str | None = None,
    since: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """List sessions with optional filters.

    Args:
        status: Filter by status (e.g., 'active', 'stale', 'completed').
        platform: Filter by platform.
        since: Filter by started_at >= since (ISO timestamp).
        limit: Maximum results to return (not implemented yet).

    Returns:
        dict with success flag and session list or error.
    """
    import asyncio

    # Parse since if provided
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            return {"success": False, "error": f"Invalid ISO timestamp: {since}"}

    result = asyncio.run(list_sessions(status=status, platform=platform, since=since_dt))

    if result.success:
        sessions = result.data
        if limit:
            sessions = sessions[:limit]
        return {"success": True, "sessions": sessions}
    else:
        return {"success": False, "error": result.error}


def session_get(session_id: str, include_messages: bool = True) -> dict[str, Any]:
    """Get session details with optional messages.

    Args:
        session_id: Session identifier.
        include_messages: Whether to include messages in the response.

    Returns:
        dict with success flag and session data or error.
    """
    import asyncio

    session_result = asyncio.run(get_session(session_id=session_id))

    if not session_result.success:
        return {"success": False, "error": session_result.error}

    response = {"success": True, "session": session_result.data}

    if include_messages:
        messages_result = asyncio.run(get_messages(session_id=session_id))
        if messages_result.success:
            response["messages"] = messages_result.data
        else:
            response["messages_error"] = messages_result.error

    return response


def session_compile(session_id: str) -> dict[str, Any]:
    """Compile session sources into an article.

    Args:
        session_id: Session identifier.

    Returns:
        dict with success flag and compilation result or error.
    """
    import asyncio

    # Find all sources for this session
    result = asyncio.run(search_sources(f"session_key:{session_id}"))

    if not result.success:
        return {"success": False, "error": f"Failed to find sources: {result.error}"}

    if not result.data:
        return {"success": False, "error": f"No sources found for session {session_id}"}

    source_ids = [s["id"] for s in result.data]

    # Compile into article
    compile_result = asyncio.run(compile_article(source_ids=source_ids))

    if compile_result.success:
        return {"success": True, "article": compile_result.data}
    else:
        return {"success": False, "error": compile_result.error}


def session_flush_stale(stale_minutes: int = 30) -> dict[str, Any]:
    """Flush all stale sessions (no activity for stale_minutes).

    Args:
        stale_minutes: Inactivity threshold in minutes.

    Returns:
        dict with success flag and list of flushed sessions or error.
    """
    import asyncio

    result = asyncio.run(flush_stale_sessions(stale_minutes=stale_minutes))

    if result.success:
        return {"success": True, "flushed_sessions": result.data}
    else:
        return {"success": False, "error": result.error}
