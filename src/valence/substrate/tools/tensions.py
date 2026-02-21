"""Tension tool implementations.

Functions:
    tension_list, tension_resolve
"""

from __future__ import annotations

from typing import Any

from . import _common
from ._common import Tension


def _supersede_article(article_id: str, new_content: str, reason: str) -> dict:
    """Supersede an article with new content (inline, no beliefs dependency)."""
    import json
    from ...core.articles import _compute_embedding, _count_tokens
    content_hash = __import__("hashlib").sha256(new_content.strip().lower().encode()).hexdigest()
    embedding_str = _compute_embedding(new_content)
    with _common.get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO articles (content, confidence, domain_path, size_tokens, content_hash, embedding,
                                  supersedes_id, compiled_at, extraction_method)
            SELECT %s, confidence, domain_path, %s, %s, %s::vector, id, NOW(), %s
            FROM articles WHERE id = %s
            RETURNING id
            """,
            (new_content, _count_tokens(new_content), content_hash, embedding_str,
             f"supersession: {reason}", article_id),
        )
        new_row = cur.fetchone()
        if not new_row:
            return {"success": False, "error": f"Article {article_id} not found"}
        new_id = str(new_row["id"])
        cur.execute(
            "UPDATE articles SET status='superseded', superseded_by_id=%s, modified_at=NOW() WHERE id=%s",
            (new_id, article_id),
        )
        return {"success": True, "new_id": new_id}


def tension_list(
    status: str | None = None,
    severity: str | None = None,
    entity_id: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """List tensions."""
    with _common.get_cursor() as cur:
        sql = "SELECT * FROM tensions WHERE 1=1"
        params: list[Any] = []

        if status:
            sql += " AND status = %s"
            params.append(status)

        if severity:
            severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            min_severity = severity_order.get(severity, 2)
            sql += " AND CASE severity WHEN 'low' THEN 1 WHEN 'medium' THEN 2 WHEN 'high' THEN 3 WHEN 'critical' THEN 4 END >= %s"
            params.append(min_severity)

        if entity_id:
            sql += """
                AND (
                    EXISTS (SELECT 1 FROM belief_entities WHERE belief_id = belief_a_id AND entity_id = %s)
                    OR EXISTS (SELECT 1 FROM belief_entities WHERE belief_id = belief_b_id AND entity_id = %s)
                )
            """
            params.extend([entity_id, entity_id])

        sql += " ORDER BY detected_at DESC LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        tensions = []
        for row in rows:
            tension = Tension.from_row(dict(row))
            tension_dict = tension.to_dict()

            # Load belief content for context
            cur.execute(
                "SELECT id, content FROM beliefs WHERE id IN (%s, %s)",
                (row["belief_a_id"], row["belief_b_id"]),
            )
            belief_rows = cur.fetchall()
            belief_map = {str(r["id"]): r["content"] for r in belief_rows}
            tension_dict["belief_a_content"] = belief_map.get(str(row["belief_a_id"]))
            tension_dict["belief_b_content"] = belief_map.get(str(row["belief_b_id"]))

            tensions.append(tension_dict)

        return {
            "success": True,
            "tensions": tensions,
            "total_count": len(tensions),
        }


def tension_resolve(
    tension_id: str,
    resolution: str,
    action: str,
) -> dict[str, Any]:
    """Resolve a tension."""
    with _common.get_cursor() as cur:
        # Get tension
        cur.execute("SELECT * FROM tensions WHERE id = %s", (tension_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Tension not found: {tension_id}"}

        belief_a_id = row["belief_a_id"]
        belief_b_id = row["belief_b_id"]

        # Perform action
        if action == "supersede_a":
            # Get belief B content and supersede A with it
            cur.execute("SELECT content FROM beliefs WHERE id = %s", (belief_b_id,))
            b_content = cur.fetchone()["content"]
            _supersede_article(str(belief_a_id), b_content, f"Tension resolution: {resolution}")

        elif action == "supersede_b":
            # Get belief A content and supersede B with it
            cur.execute("SELECT content FROM beliefs WHERE id = %s", (belief_a_id,))
            a_content = cur.fetchone()["content"]
            _supersede_article(str(belief_b_id), a_content, f"Tension resolution: {resolution}")

        elif action == "archive_both":
            cur.execute(
                "UPDATE beliefs SET status = 'archived', modified_at = NOW() WHERE id IN (%s, %s)",
                (belief_a_id, belief_b_id),
            )

        # Mark tension as resolved
        cur.execute(
            """
            UPDATE tensions
            SET status = %s, resolution = %s, resolved_at = NOW()
            WHERE id = %s
            """,
            (
                "resolved" if action != "keep_both" else "accepted",
                resolution,
                tension_id,
            ),
        )

        return {
            "success": True,
            "tension_id": tension_id,
            "action": action,
            "resolution": resolution,
        }
