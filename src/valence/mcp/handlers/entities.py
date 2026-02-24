"""Entity tool handlers."""

from __future__ import annotations

import logging
from typing import Any

from valence.core.db import get_cursor

logger = logging.getLogger(__name__)


def entity_get(entity_id: str, include_beliefs: bool = False, belief_limit: int = 10) -> dict[str, Any]:
    """Get an entity by ID."""
    try:
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT id, type, name, canonical_id, aliases, metadata, created_at
                FROM entities
                WHERE id = %s
                """,
                (entity_id,),
            )
            row = cur.fetchone()

        if not row:
            return {"success": False, "error": f"Entity not found: {entity_id}"}

        entity = dict(row)
        entity["id"] = str(entity["id"])
        if entity.get("canonical_id"):
            entity["canonical_id"] = str(entity["canonical_id"])
        if entity.get("created_at"):
            entity["created_at"] = entity["created_at"].isoformat()

        if include_beliefs:
            entity["beliefs"] = []

        return {"success": True, "entity": entity}
    except Exception as exc:
        logger.exception("entity_get failed")
        return {"success": False, "error": str(exc)}


def entity_search(query: str, entity_type: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Search entities by name."""
    if not query or not query.strip():
        return {"success": False, "error": "query must be non-empty"}

    limit = max(1, min(int(limit), 200))

    try:
        with get_cursor() as cur:
            if entity_type:
                cur.execute(
                    """
                    SELECT id, type, name, canonical_id, aliases, metadata, created_at
                    FROM entities
                    WHERE name ILIKE %s AND type = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (f"%{query}%", entity_type, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, type, name, canonical_id, aliases, metadata, created_at
                    FROM entities
                    WHERE name ILIKE %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (f"%{query}%", limit),
                )
            rows = cur.fetchall()

        entities = []
        for row in rows:
            e = dict(row)
            e["id"] = str(e["id"])
            if e.get("canonical_id"):
                e["canonical_id"] = str(e["canonical_id"])
            if e.get("created_at"):
                e["created_at"] = e["created_at"].isoformat()
            entities.append(e)

        return {"success": True, "entities": entities, "total_count": len(entities)}
    except Exception as exc:
        logger.exception("entity_search failed")
        return {"success": False, "error": str(exc)}
