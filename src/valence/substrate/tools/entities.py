"""Entity tool implementations.

Functions:
    entity_get, entity_search
"""

from __future__ import annotations

from typing import Any

from . import _common
from ._common import Belief, Entity, escape_ilike


def entity_get(
    entity_id: str,
    include_beliefs: bool = False,
    belief_limit: int = 10,
) -> dict[str, Any]:
    """Get an entity by ID."""
    with _common.get_cursor() as cur:
        cur.execute("SELECT * FROM entities WHERE id = %s", (entity_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Entity not found: {entity_id}"}

        entity = Entity.from_row(dict(row))
        result: dict[str, Any] = {
            "success": True,
            "entity": entity.to_dict(),
        }

        if include_beliefs:
            cur.execute(
                """
                SELECT b.*, be.role
                FROM articles b
                JOIN article_entities be ON b.id = be.article_id
                WHERE be.entity_id = %s
                AND b.status = 'active'
                AND b.superseded_by_id IS NULL
                ORDER BY b.created_at DESC
                LIMIT %s
                """,
                (entity_id, belief_limit),
            )
            belief_rows = cur.fetchall()
            result["beliefs"] = [{**Belief.from_row(dict(r)).to_dict(), "role": r["role"]} for r in belief_rows]

        return result


def entity_search(
    query: str,
    type: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Search for entities."""
    with _common.get_cursor() as cur:
        sql = """
            SELECT * FROM entities
            WHERE (name ILIKE %s OR %s = ANY(aliases))
            AND canonical_id IS NULL
        """
        params: list[Any] = [f"%{escape_ilike(query)}%", query]

        if type:
            sql += " AND type = %s"
            params.append(type)

        sql += " ORDER BY name LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        return {
            "success": True,
            "entities": [Entity.from_row(dict(r)).to_dict() for r in rows],
            "total_count": len(rows),
        }
