"""Pattern management tool implementations.

Functions: pattern_record, pattern_reinforce, pattern_list, pattern_search
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from valence.core.db import get_cursor
from valence.lib.our_models import Pattern

from ...core.utils import escape_ilike

logger = logging.getLogger(__name__)


def pattern_record(
    type: str,
    description: str,
    evidence: list[str] | None = None,
    confidence: float = 0.5,
) -> dict[str, Any]:
    """Record a new pattern."""
    with get_cursor() as cur:
        # Validate and keep as strings -- psycopg2 can't adapt UUID objects in arrays,
        # but PostgreSQL will cast text[] to uuid[] via the explicit ::uuid[] cast.
        evidence_strs = [str(UUID(e)) for e in (evidence or [])]

        cur.execute(
            """
            INSERT INTO vkb_patterns (type, description, evidence, confidence)
            VALUES (%s, %s, %s::uuid[], %s)
            RETURNING *
            """,
            (type, description, evidence_strs, confidence),
        )
        row = cur.fetchone()

        pattern = Pattern.from_row(dict(row))
        return {
            "success": True,
            "pattern": pattern.to_dict(),
        }


def pattern_reinforce(
    pattern_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Reinforce an existing pattern."""
    with get_cursor() as cur:
        # Get current pattern
        cur.execute("SELECT * FROM vkb_patterns WHERE id = %s", (pattern_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Pattern not found: {pattern_id}"}

        pattern = Pattern.from_row(dict(row))
        evidence = pattern.evidence

        # Add session to evidence if not already present
        if session_id:
            session_uuid = UUID(session_id)
            if session_uuid not in evidence:
                evidence.append(session_uuid)

        # Increase confidence (asymptotic to 1.0)
        new_confidence = min(0.99, pattern.confidence + (1 - pattern.confidence) * 0.1)

        # Update status if appropriate
        new_status = pattern.status.value
        if pattern.occurrence_count >= 4 and pattern.status.value == "emerging":
            new_status = "established"

        # Convert UUIDs to strings for psycopg2 array adaptation
        evidence_strs = [str(e) for e in evidence]

        cur.execute(
            """
            UPDATE vkb_patterns
            SET evidence = %s::uuid[], occurrence_count = occurrence_count + 1,
                confidence = %s, last_observed = NOW(), status = %s
            WHERE id = %s
            RETURNING *
            """,
            (evidence_strs, new_confidence, new_status, pattern_id),
        )
        row = cur.fetchone()

        pattern = Pattern.from_row(dict(row))
        return {
            "success": True,
            "pattern": pattern.to_dict(),
        }


def pattern_list(
    type: str | None = None,
    status: str | None = None,
    min_confidence: float = 0,
    limit: int = 20,
) -> dict[str, Any]:
    """List patterns."""
    with get_cursor() as cur:
        sql = "SELECT * FROM vkb_patterns WHERE confidence >= %s"
        params: list[Any] = [min_confidence]

        if type:
            sql += " AND type = %s"
            params.append(type)

        if status:
            sql += " AND status = %s"
            params.append(status)

        sql += " ORDER BY occurrence_count DESC, confidence DESC LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        return {
            "success": True,
            "patterns": [Pattern.from_row(dict(r)).to_dict() for r in rows],
            "total_count": len(rows),
        }


def pattern_search(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Search patterns by description."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM vkb_patterns
            WHERE description ILIKE %s
            ORDER BY confidence DESC
            LIMIT %s
            """,
            (f"%{escape_ilike(query)}%", limit),
        )
        rows = cur.fetchall()

        return {
            "success": True,
            "patterns": [Pattern.from_row(dict(r)).to_dict() for r in rows],
            "total_count": len(rows),
        }
