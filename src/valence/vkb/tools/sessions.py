"""Session management tool implementations.

Functions: session_start, session_end, session_get, session_list, session_find_by_room
"""

from __future__ import annotations

import json
import logging
from typing import Any

from valence.lib.our_db import get_cursor
from valence.lib.our_models import Exchange, Session

logger = logging.getLogger(__name__)


def session_start(
    platform: str,
    project_context: str | None = None,
    external_room_id: str | None = None,
    claude_session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a new session."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO vkb_sessions (platform, project_context, external_room_id, claude_session_id, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                platform,
                project_context,
                external_room_id,
                claude_session_id,
                json.dumps(metadata or {}),
            ),
        )
        row = cur.fetchone()

        # Create source record for this session
        cur.execute(
            """
            INSERT INTO sources (type, session_id, title)
            VALUES ('conversation', %s, %s)
            """,
            (row["id"], f"Session: {row['id']}"),
        )

        session = Session.from_row(dict(row))
        return {
            "success": True,
            "session": session.to_dict(),
        }


def session_end(
    session_id: str,
    summary: str | None = None,
    themes: list[str] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    """End a session."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE vkb_sessions
            SET status = %s, summary = COALESCE(%s, summary), themes = COALESCE(%s, themes), ended_at = NOW()
            WHERE id = %s
            RETURNING *
            """,
            (status, summary, themes, session_id),
        )
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Session not found: {session_id}"}

        session = Session.from_row(dict(row))
        return {
            "success": True,
            "session": session.to_dict(),
        }


def session_get(
    session_id: str,
    include_exchanges: bool = False,
    exchange_limit: int = 10,
) -> dict[str, Any]:
    """Get a session by ID."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM vkb_sessions_overview WHERE id = %s", (session_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Session not found: {session_id}"}

        session = Session.from_row(dict(row))
        result: dict[str, Any] = {
            "success": True,
            "session": session.to_dict(),
        }

        if include_exchanges:
            cur.execute(
                """
                SELECT * FROM vkb_exchanges
                WHERE session_id = %s
                ORDER BY sequence DESC
                LIMIT %s
                """,
                (session_id, exchange_limit),
            )
            exchange_rows = cur.fetchall()
            result["exchanges"] = [
                Exchange.from_row(dict(r)).to_dict()
                for r in reversed(exchange_rows)  # Return in chronological order
            ]

        return result


def session_list(
    platform: str | None = None,
    project_context: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """List sessions."""
    with get_cursor() as cur:
        sql = "SELECT * FROM vkb_sessions_overview WHERE 1=1"
        params: list[Any] = []

        if platform:
            sql += " AND platform = %s"
            params.append(platform)

        if project_context:
            sql += " AND project_context = %s"
            params.append(project_context)

        if status:
            sql += " AND status = %s"
            params.append(status)

        sql += " ORDER BY started_at DESC LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        return {
            "success": True,
            "sessions": [Session.from_row(dict(r)).to_dict() for r in rows],
            "total_count": len(rows),
        }


def session_find_by_room(external_room_id: str) -> dict[str, Any]:
    """Find active session by external room ID."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM vkb_sessions
            WHERE external_room_id = %s AND status = 'active'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (external_room_id,),
        )
        row = cur.fetchone()

        if row:
            session = Session.from_row(dict(row))
            return {
                "success": True,
                "found": True,
                "session": session.to_dict(),
            }
        else:
            return {
                "success": True,
                "found": False,
                "session": None,
            }
