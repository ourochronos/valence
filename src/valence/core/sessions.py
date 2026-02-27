# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Session ingestion and conversation buffering for Valence.

Sessions are first-class sources. Messages buffer in the database, then flush
to sources at natural boundaries (compaction, session end, or stale timeout).

Implements #469 (session schema) and #473 (flush logic).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from valence.core.db import get_cursor, serialize_row
from valence.core.response import ValenceResponse, err, ok
from valence.core.sources import ingest_source

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Access Layer
# ---------------------------------------------------------------------------


async def upsert_session(
    session_id: str,
    platform: str,
    channel: str | None = None,
    participants: list[str] | None = None,
    metadata: dict | None = None,
    parent_session_id: str | None = None,
    subagent_label: str | None = None,
    subagent_model: str | None = None,
    subagent_task: str | None = None,
) -> ValenceResponse:
    """Insert a new session or update last_activity_at if it already exists.

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
        ValenceResponse with data = session dict.
    """
    if not session_id or not session_id.strip():
        return err("session_id must be non-empty")
    if not platform or not platform.strip():
        return err("platform must be non-empty")

    participants_array = participants or []
    metadata_json = json.dumps(metadata or {})

    with get_cursor() as cur:
        # Validate parent exists if provided
        if parent_session_id:
            cur.execute("SELECT session_id FROM sessions WHERE session_id = %s", (parent_session_id,))
            if not cur.fetchone():
                return err(f"Parent session not found: {parent_session_id}")

        # Insert-if-new or update last_activity_at
        cur.execute(
            """
            INSERT INTO sessions (session_id, platform, channel, participants, metadata,
                                  parent_session_id, subagent_label, subagent_model, subagent_task)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE
            SET last_activity_at = NOW()
            RETURNING session_id, platform, channel, participants, started_at,
                      last_activity_at, ended_at, status, metadata, parent_session_id,
                      subagent_label, subagent_model, subagent_task, current_chunk_index
            """,
            (
                session_id,
                platform,
                channel,
                participants_array,
                metadata_json,
                parent_session_id,
                subagent_label,
                subagent_model,
                subagent_task,
            ),
        )
        row = cur.fetchone()

    result = serialize_row(row)
    logger.info("Upserted session %s (platform=%s)", session_id, platform)
    return ok(data=result)


async def get_session(session_id: str) -> ValenceResponse:
    """Retrieve a session by ID.

    Args:
        session_id: Session identifier.

    Returns:
        ValenceResponse with data = session dict, or error if not found.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT session_id, platform, channel, participants, started_at,
                   last_activity_at, ended_at, status, metadata, parent_session_id,
                   subagent_label, subagent_model, subagent_task, current_chunk_index
            FROM sessions
            WHERE session_id = %s
            """,
            (session_id,),
        )
        row = cur.fetchone()

    if not row:
        return err(f"Session not found: {session_id}")

    return ok(data=serialize_row(row))


async def list_sessions(
    status: str | None = None,
    platform: str | None = None,
    since: datetime | None = None,
) -> ValenceResponse:
    """List sessions with optional filters.

    Args:
        status: Filter by status (e.g., 'active', 'stale', 'completed').
        platform: Filter by platform.
        since: Filter by started_at >= since.

    Returns:
        ValenceResponse with data = list of session dicts.
    """
    with get_cursor() as cur:
        where_clauses = []
        params: list[Any] = []

        if status:
            where_clauses.append("status = %s")
            params.append(status)
        if platform:
            where_clauses.append("platform = %s")
            params.append(platform)
        if since:
            where_clauses.append("started_at >= %s")
            params.append(since)

        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

        cur.execute(
            f"""
            SELECT session_id, platform, channel, participants, started_at,
                   last_activity_at, ended_at, status, metadata, parent_session_id,
                   subagent_label, subagent_model, subagent_task, current_chunk_index
            FROM sessions
            WHERE {where_sql}
            ORDER BY last_activity_at DESC
            """,
            params,
        )
        rows = cur.fetchall()

    return ok(data=[serialize_row(row) for row in rows])


async def update_session(
    session_id: str,
    status: str | None = None,
    metadata: dict | None = None,
    participants: list[str] | None = None,
    ended_at: datetime | None = None,
) -> ValenceResponse:
    """Update session fields.

    Args:
        session_id: Session identifier.
        status: New status.
        metadata: Metadata to merge (replaces existing).
        participants: Participant list to set.
        ended_at: End timestamp.

    Returns:
        ValenceResponse with data = updated session dict.
    """
    updates = []
    params: list[Any] = []

    if status is not None:
        updates.append("status = %s")
        params.append(status)
    if metadata is not None:
        updates.append("metadata = %s::jsonb")
        params.append(json.dumps(metadata))
    if participants is not None:
        updates.append("participants = %s")
        params.append(participants)
    if ended_at is not None:
        updates.append("ended_at = %s")
        params.append(ended_at)

    if not updates:
        return err("No fields to update")

    params.append(session_id)

    with get_cursor() as cur:
        cur.execute(
            f"""
            UPDATE sessions
            SET {", ".join(updates)}
            WHERE session_id = %s
            RETURNING session_id, platform, channel, participants, started_at,
                      last_activity_at, ended_at, status, metadata, parent_session_id,
                      subagent_label, subagent_model, subagent_task, current_chunk_index
            """,
            params,
        )
        row = cur.fetchone()

    if not row:
        return err(f"Session not found: {session_id}")

    logger.info("Updated session %s", session_id)
    return ok(data=serialize_row(row))


async def append_message(
    session_id: str,
    speaker: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> ValenceResponse:
    """Append a message to the session buffer and update last_activity_at.

    Args:
        session_id: Session identifier.
        speaker: Speaker name (e.g., 'chris', 'jane', worker label).
        role: Message role ('user', 'assistant', 'system', 'tool').
        content: Message content.
        metadata: Optional message-specific metadata.

    Returns:
        ValenceResponse with data = message dict.
    """
    if role not in ("user", "assistant", "system", "tool"):
        return err(f"Invalid role: {role}")

    metadata_json = json.dumps(metadata or {})

    with get_cursor() as cur:
        # Insert message
        cur.execute(
            """
            INSERT INTO session_messages (session_id, speaker, role, content, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            RETURNING id, session_id, chunk_index, timestamp, speaker, role, content, metadata, flushed_at
            """,
            (session_id, speaker, role, content, metadata_json),
        )
        row = cur.fetchone()

        # Update session last_activity_at
        cur.execute(
            "UPDATE sessions SET last_activity_at = NOW() WHERE session_id = %s",
            (session_id,),
        )

    logger.debug("Appended message to session %s (speaker=%s, role=%s)", session_id, speaker, role)
    return ok(data=serialize_row(row))


async def append_messages(
    session_id: str,
    messages: list[dict[str, Any]],
) -> ValenceResponse:
    """Append multiple messages to the session buffer (batch version).

    Args:
        session_id: Session identifier.
        messages: List of message dicts with keys: speaker, role, content, metadata?.

    Returns:
        ValenceResponse with data = list of inserted message dicts.
    """
    if not messages:
        return ok(data=[])

    results = []
    for msg in messages:
        resp = await append_message(
            session_id,
            msg["speaker"],
            msg["role"],
            msg["content"],
            msg.get("metadata"),
        )
        if not resp.success:
            return resp
        results.append(resp.data)

    return ok(data=results)


async def get_messages(
    session_id: str,
    role: str | None = None,
    since: datetime | None = None,
    chunk_index: int | None = None,
) -> ValenceResponse:
    """Get messages for a session with optional filters.

    Args:
        session_id: Session identifier.
        role: Filter by role.
        since: Filter by timestamp >= since.
        chunk_index: Filter by chunk_index.

    Returns:
        ValenceResponse with data = list of message dicts.
    """
    with get_cursor() as cur:
        where_clauses = ["session_id = %s"]
        params: list[Any] = [session_id]

        if role:
            where_clauses.append("role = %s")
            params.append(role)
        if since:
            where_clauses.append("timestamp >= %s")
            params.append(since)
        if chunk_index is not None:
            where_clauses.append("chunk_index = %s")
            params.append(chunk_index)

        where_sql = " AND ".join(where_clauses)

        cur.execute(
            f"""
            SELECT id, session_id, chunk_index, timestamp, speaker, role, content, metadata, flushed_at
            FROM session_messages
            WHERE {where_sql}
            ORDER BY timestamp ASC
            """,
            params,
        )
        rows = cur.fetchall()

    return ok(data=[serialize_row(row) for row in rows])


async def get_unflushed_messages(session_id: str) -> ValenceResponse:
    """Get all unflushed messages for a session.

    Args:
        session_id: Session identifier.

    Returns:
        ValenceResponse with data = list of unflushed message dicts.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, session_id, chunk_index, timestamp, speaker, role, content, metadata, flushed_at
            FROM session_messages
            WHERE session_id = %s AND flushed_at IS NULL
            ORDER BY timestamp ASC
            """,
            (session_id,),
        )
        rows = cur.fetchall()

    return ok(data=[serialize_row(row) for row in rows])


async def find_stale_sessions(stale_minutes: int = 30) -> ValenceResponse:
    """Find active sessions with no activity for stale_minutes.

    Args:
        stale_minutes: Inactivity threshold in minutes.

    Returns:
        ValenceResponse with data = list of stale session_ids.
    """
    threshold = datetime.now(UTC) - timedelta(minutes=stale_minutes)

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE status = 'active' AND last_activity_at < %s
            ORDER BY last_activity_at ASC
            """,
            (threshold,),
        )
        rows = cur.fetchall()

    return ok(data=[row["session_id"] for row in rows])


# ---------------------------------------------------------------------------
# Flush Logic
# ---------------------------------------------------------------------------


async def flush_session(session_id: str, compile: bool = True) -> ValenceResponse:
    """Flush unflushed messages to a conversation source and mark them as flushed.

    Steps:
    1. Get unflushed messages
    2. If empty, return (nothing to flush)
    3. Serialize to markdown transcript
    4. Call ingest_source with type='conversation'
    5. Mark messages as flushed
    6. Increment current_chunk_index
    7. If compile=True: trigger compilation

    Args:
        session_id: Session identifier.
        compile: Whether to trigger compilation after flush.

    Returns:
        ValenceResponse with data = flush summary dict.
    """
    # Get session
    session_resp = await get_session(session_id)
    if not session_resp.success:
        return session_resp
    session = session_resp.data

    # Get unflushed messages
    msgs_resp = await get_unflushed_messages(session_id)
    if not msgs_resp.success:
        return msgs_resp
    messages = msgs_resp.data

    if not messages:
        logger.debug("No unflushed messages for session %s", session_id)
        return ok(data={"session_id": session_id, "message_count": 0, "flushed": False})

    # Serialize to markdown
    chunk_index = session["current_chunk_index"]
    transcript = _serialize_transcript(session, messages, chunk_index)

    # Ingest as conversation source
    source_metadata = {
        "session_key": session_id,
        "chunk_index": str(chunk_index),
        "platform": session["platform"],
        "channel": session.get("channel") or "",
        "participants": ",".join(session.get("participants") or []),
        "started_at": session["started_at"],
        "flushed_at": datetime.now(UTC).isoformat(),
        "message_count": str(len(messages)),
    }

    title = f"Session {session_id} — chunk {chunk_index}"
    ingest_resp = await ingest_source(
        content=transcript,
        source_type="conversation",
        title=title,
        metadata=source_metadata,
    )

    if not ingest_resp.success:
        return ingest_resp

    source_id = ingest_resp.data["id"]

    # Mark messages as flushed and increment chunk index
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE session_messages
            SET flushed_at = NOW()
            WHERE session_id = %s AND flushed_at IS NULL
            """,
            (session_id,),
        )

        cur.execute(
            """
            UPDATE sessions
            SET current_chunk_index = current_chunk_index + 1
            WHERE session_id = %s
            """,
            (session_id,),
        )

    logger.info(
        "Flushed session %s chunk %d (%d messages) → source %s",
        session_id,
        chunk_index,
        len(messages),
        source_id,
    )

    # Trigger compilation if requested
    if compile:
        from valence.core.compilation import compile_article

        # Query for all sources for this session
        from valence.core.sources import search_sources

        search_resp = await search_sources(f"session_key:{session_id}")
        if search_resp.success and search_resp.data:
            source_ids = [s["id"] for s in search_resp.data]
            compile_resp = await compile_article(source_ids)
            if not compile_resp.success:
                logger.warning("Compilation failed for session %s: %s", session_id, compile_resp.error)

    return ok(
        data={
            "session_id": session_id,
            "chunk_index": chunk_index,
            "message_count": len(messages),
            "source_id": source_id,
            "flushed": True,
        }
    )


async def finalize_session(session_id: str) -> ValenceResponse:
    """Flush final messages and mark session as completed.

    Args:
        session_id: Session identifier.

    Returns:
        ValenceResponse with data = finalization summary.
    """
    # Flush with compilation
    flush_resp = await flush_session(session_id, compile=True)
    if not flush_resp.success:
        return flush_resp

    # Mark as completed
    update_resp = await update_session(
        session_id,
        status="completed",
        ended_at=datetime.now(UTC),
    )
    if not update_resp.success:
        return update_resp

    logger.info("Finalized session %s", session_id)
    return ok(data={"session_id": session_id, "status": "completed", "flush": flush_resp.data})


async def flush_stale_sessions(stale_minutes: int = 30) -> ValenceResponse:
    """Find and flush stale sessions, then mark them as stale.

    Args:
        stale_minutes: Inactivity threshold in minutes.

    Returns:
        ValenceResponse with data = list of flushed session summaries.
    """
    # Find stale sessions
    stale_resp = await find_stale_sessions(stale_minutes)
    if not stale_resp.success:
        return stale_resp

    stale_sessions = stale_resp.data
    results = []

    for session_id in stale_sessions:
        # Flush with compilation
        flush_resp = await flush_session(session_id, compile=True)
        if not flush_resp.success:
            logger.warning("Failed to flush stale session %s: %s", session_id, flush_resp.error)
            continue

        # Mark as stale
        update_resp = await update_session(session_id, status="stale")
        if not update_resp.success:
            logger.warning("Failed to mark session %s as stale: %s", session_id, update_resp.error)

        results.append(flush_resp.data)

    logger.info("Flushed %d stale sessions", len(results))
    return ok(data=results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_transcript(session: dict, messages: list[dict], chunk_index: int) -> str:
    """Serialize session messages to markdown transcript format.

    Format:
        # Session: {session_id}
        Platform: {platform} | Channel: {channel}
        Participants: {participants joined}
        Time: {first_msg_timestamp} → {last_msg_timestamp}
        Chunk: {chunk_index}
        ---

        [{timestamp}] {speaker} ({role}):
        {content}

        [{timestamp}] {speaker} ({role}):
        {content}
    """
    session_id = session["session_id"]
    platform = session["platform"]
    channel = session.get("channel") or "unknown"
    participants = ", ".join(session.get("participants") or [])

    first_ts = messages[0]["timestamp"] if messages else ""
    last_ts = messages[-1]["timestamp"] if messages else ""

    header = [
        f"# Session: {session_id}",
        f"Platform: {platform} | Channel: {channel}",
        f"Participants: {participants}",
        f"Time: {first_ts} → {last_ts}",
        f"Chunk: {chunk_index}",
        "---",
    ]

    body = []
    for msg in messages:
        ts = msg["timestamp"]
        speaker = msg["speaker"]
        role = msg["role"]
        content = msg["content"]
        body.append(f"[{ts}] {speaker} ({role}):\n{content}")

    return "\n".join(header) + "\n\n" + "\n\n".join(body)
