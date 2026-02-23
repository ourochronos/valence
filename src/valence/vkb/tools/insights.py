"""Insight extraction tool implementations.

Functions: insight_extract, insight_list
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from valence.lib.our_confidence import DimensionalConfidence
from valence.lib.our_db import get_cursor

logger = logging.getLogger(__name__)


def insight_extract(
    session_id: str,
    content: str,
    domain_path: list[str] | None = None,
    confidence: dict[str, Any] | None = None,
    entities: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Extract an insight from a session and create a belief.

    Performs content-hash deduplication before creating. If an exact duplicate
    exists, reinforces the existing belief instead of creating a new one.
    """
    confidence_obj = DimensionalConfidence.from_dict(confidence or {"overall": 0.8})
    content_hash = hashlib.sha256(content.strip().lower().encode()).hexdigest()

    with get_cursor() as cur:
        # --- Dedup check: exact content hash ---
        cur.execute(
            "SELECT id, confidence FROM articles WHERE content_hash = %s AND status = 'active' AND superseded_by_id IS NULL",
            (content_hash,),
        )
        existing = cur.fetchone()
        if existing:
            from ...core.curation import corroboration_confidence

            existing_id = existing["id"]
            existing_conf = existing["confidence"] if isinstance(existing["confidence"], dict) else json.loads(existing["confidence"])

            # Record corroboration
            cur.execute("SELECT COUNT(*) as cnt FROM belief_corroborations WHERE article_id = %s", (existing_id,))
            count = cur.fetchone()["cnt"]
            cur.execute(
                "INSERT INTO belief_corroborations (article_id, source_session_id, source_type) VALUES (%s, %s, 'session')",
                (existing_id, session_id),
            )

            # Update confidence
            new_count = count + 1
            new_overall = corroboration_confidence(new_count)
            existing_conf["overall"] = max(existing_conf.get("overall", 0.5), new_overall)
            existing_conf["corroboration"] = new_overall
            cur.execute("UPDATE articles SET confidence = %s, modified_at = NOW() WHERE id = %s", (json.dumps(existing_conf), existing_id))

            # Link to session
            cur.execute(
                "INSERT INTO vkb_session_insights (session_id, article_id, extraction_method) VALUES (%s, %s, 'manual') ON CONFLICT DO NOTHING RETURNING id",
                (session_id, existing_id),
            )
            insight_row = cur.fetchone()

            return {
                "success": True,
                "deduplicated": True,
                "action": "reinforced",
                "corroboration_count": new_count,
                "insight_id": str(insight_row["id"]) if insight_row else None,
                "article_id": str(existing_id),
                "session_id": session_id,
            }

        # --- No duplicate: create new article ---
        cur.execute("SELECT id FROM sources WHERE session_id = %s LIMIT 1", (session_id,))
        source_row = cur.fetchone()
        source_id = source_row["id"] if source_row else None

        cur.execute(
            """
            INSERT INTO articles (content, confidence, domain_path, source_id, extraction_method, content_hash)
            VALUES (%s, %s, %s, %s, 'conversation_extraction', %s)
            RETURNING *
            """,
            (
                content,
                json.dumps(confidence_obj.to_dict()),
                domain_path or [],
                source_id,
                content_hash,
            ),
        )
        article_row = cur.fetchone()
        article_id = article_row["id"]

        # Link entities
        if entities:
            for entity in entities:
                cur.execute(
                    """
                    INSERT INTO entities (name, type)
                    VALUES (%s, %s)
                    ON CONFLICT (LOWER(name), type) WHERE canonical_id IS NULL
                    DO UPDATE SET modified_at = NOW()
                    RETURNING id
                    """,
                    (entity["name"], entity.get("type", "concept")),
                )
                entity_id = cur.fetchone()["id"]

                cur.execute(
                    """
                    INSERT INTO article_entities (article_id, entity_id, role)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (article_id, entity_id, entity.get("role", "subject")),
                )

        # Link to session
        cur.execute(
            """
            INSERT INTO vkb_session_insights (session_id, article_id, extraction_method)
            VALUES (%s, %s, 'manual')
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            (session_id, article_id),
        )
        insight_row = cur.fetchone()

        return {
            "success": True,
            "insight_id": str(insight_row["id"]) if insight_row else None,
            "article_id": str(article_id),
            "session_id": session_id,
        }


def insight_list(session_id: str) -> dict[str, Any]:
    """List insights from a session."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT si.*, b.content, b.confidence, b.domain_path, b.created_at as article_created_at
            FROM vkb_session_insights si
            JOIN articles b ON si.article_id = b.id
            WHERE si.session_id = %s
            ORDER BY si.extracted_at
            """,
            (session_id,),
        )
        rows = cur.fetchall()

        insights = []
        for row in rows:
            insights.append(
                {
                    "id": str(row["id"]),
                    "session_id": str(row["session_id"]),
                    "article_id": str(row["article_id"]),
                    "extraction_method": row["extraction_method"],
                    "extracted_at": row["extracted_at"].isoformat(),
                    "article": {
                        "content": row["content"],
                        "confidence": row["confidence"],
                        "domain_path": row["domain_path"],
                        "created_at": row["article_created_at"].isoformat(),
                    },
                }
            )

        return {
            "success": True,
            "insights": insights,
            "total_count": len(insights),
        }
