"""Exchange tool implementations.

Functions: exchange_add, exchange_list
"""

from __future__ import annotations

import json
import logging
from typing import Any

from valence.lib.our_db import get_cursor
from valence.lib.our_models import Exchange

logger = logging.getLogger(__name__)


def exchange_add(
    session_id: str,
    role: str,
    content: str,
    tokens_approx: int | None = None,
    tool_uses: list[dict] | None = None,
) -> dict[str, Any]:
    """Add an exchange to a session."""
    with get_cursor() as cur:
        # Get next sequence number
        cur.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 as next_seq FROM vkb_exchanges WHERE session_id = %s",
            (session_id,),
        )
        sequence = cur.fetchone()["next_seq"]

        cur.execute(
            """
            INSERT INTO vkb_exchanges (session_id, sequence, role, content, tokens_approx, tool_uses)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                session_id,
                sequence,
                role,
                content,
                tokens_approx,
                json.dumps(tool_uses or []),
            ),
        )
        row = cur.fetchone()

        exchange = Exchange.from_row(dict(row))
        return {
            "success": True,
            "exchange": exchange.to_dict(),
        }


def exchange_list(
    session_id: str,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Get exchanges from a session."""
    with get_cursor() as cur:
        sql = "SELECT * FROM vkb_exchanges WHERE session_id = %s ORDER BY sequence"
        params: list[Any] = [session_id]

        if limit:
            sql += " LIMIT %s OFFSET %s"
            params.extend([limit, offset])

        cur.execute(sql, params)
        rows = cur.fetchall()

        return {
            "success": True,
            "exchanges": [Exchange.from_row(dict(r)).to_dict() for r in rows],
            "total_count": len(rows),
        }
