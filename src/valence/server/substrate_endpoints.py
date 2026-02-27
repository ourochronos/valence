# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""REST API endpoints for stats and conflict detection."""

from __future__ import annotations

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .auth_helpers import authenticate, require_scope
from .endpoint_utils import _parse_bool, _parse_float, format_response, parse_output_format
from .errors import internal_error
from .formatters import format_conflicts_text, format_stats_text

logger = logging.getLogger(__name__)


# =============================================================================
# STATS
# =============================================================================


async def stats_endpoint(request: Request) -> Response:
    """GET /api/v1/stats — Aggregate database statistics."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    output_format = parse_output_format(request)

    try:
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM articles")
            total = cur.fetchone()["total"]

            cur.execute("SELECT COUNT(*) as active FROM articles WHERE status = 'active' AND superseded_by_id IS NULL")
            active = cur.fetchone()["active"]

            cur.execute("SELECT COUNT(*) as with_emb FROM articles WHERE embedding IS NOT NULL")
            with_embedding = cur.fetchone()["with_emb"]

            cur.execute("SELECT COUNT(*) as cnt FROM contentions WHERE status = 'detected'")
            unresolved_contentions = cur.fetchone()["cnt"]

            try:
                cur.execute("SELECT COUNT(DISTINCT d) as count FROM articles, LATERAL unnest(domain_path) as d")
                domains = cur.fetchone()["count"]
            except Exception:
                domains = 0

            cur.execute("SELECT COUNT(*) as cnt FROM sources")
            source_count = cur.fetchone()["cnt"]

        result = {
            "success": True,
            "stats": {
                "total_articles": total,
                "active_articles": active,
                "total_sources": source_count,
                "with_embeddings": with_embedding,
                "unique_domains": domains,
                "unresolved_contentions": unresolved_contentions,
            },
        }
        return format_response(result, output_format, text_formatter=format_stats_text)
    except Exception:
        logger.exception("Error getting stats")
        return internal_error()


# =============================================================================
# CONFLICT DETECTION
# =============================================================================


async def conflicts_endpoint(request: Request) -> Response:
    """GET /api/v1/contentions/detect — Detect potential contradictions via semantic similarity."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, "substrate:read"):
        return err

    threshold = _parse_float(request.query_params.get("threshold"), 0.85) or 0.85
    auto_record = _parse_bool(request.query_params.get("auto_record"))
    output_format = parse_output_format(request)

    try:
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute(
                """
                WITH article_pairs AS (
                    SELECT
                        a1.id as id_a, a1.content as content_a, a1.confidence as confidence_a,
                        a2.id as id_b, a2.content as content_b, a2.confidence as confidence_b,
                        1 - (a1.embedding <=> a2.embedding) as similarity
                    FROM articles a1
                    CROSS JOIN articles a2
                    WHERE a1.id < a2.id
                      AND a1.embedding IS NOT NULL AND a2.embedding IS NOT NULL
                      AND a1.status = 'active' AND a2.status = 'active'
                      AND a1.superseded_by_id IS NULL AND a2.superseded_by_id IS NULL
                      AND 1 - (a1.embedding <=> a2.embedding) > %s
                    ORDER BY similarity DESC
                    LIMIT 50
                )
                SELECT * FROM article_pairs
                WHERE NOT EXISTS (
                    SELECT 1 FROM contentions t
                    WHERE (t.article_id = article_pairs.id_a AND t.related_article_id = article_pairs.id_b)
                       OR (t.article_id = article_pairs.id_b AND t.related_article_id = article_pairs.id_a)
                )
                """,
                (threshold,),
            )
            pairs = cur.fetchall()

            negation_words = {
                "not",
                "never",
                "no",
                "n't",
                "cannot",
                "without",
                "neither",
                "none",
                "nobody",
                "nothing",
                "nowhere",
                "false",
                "incorrect",
                "wrong",
                "fail",
                "reject",
                "deny",
                "refuse",
                "avoid",
            }
            opposites = [
                ("good", "bad"),
                ("right", "wrong"),
                ("true", "false"),
                ("should", "should not"),
                ("always", "never"),
                ("prefer", "avoid"),
                ("like", "dislike"),
                ("works", "fails"),
                ("correct", "incorrect"),
            ]

            conflicts = []
            for pair in pairs:
                content_a = pair["content_a"].lower()
                content_b = pair["content_b"].lower()
                words_a = set(content_a.split())
                words_b = set(content_b.split())

                neg_a = bool(words_a & negation_words)
                neg_b = bool(words_b & negation_words)

                conflict_signal = 0.0
                reason = []

                if neg_a != neg_b:
                    conflict_signal += 0.4
                    reason.append("negation asymmetry")

                for pos, neg in opposites:
                    if (pos in content_a and neg in content_b) or (neg in content_a and pos in content_b):
                        conflict_signal += 0.3
                        reason.append(f"opposite: {pos}/{neg}")
                        break

                if conflict_signal > 0.2 or pair["similarity"] > 0.92:
                    conflicts.append(
                        {
                            "id_a": str(pair["id_a"]),
                            "content_a": pair["content_a"],
                            "id_b": str(pair["id_b"]),
                            "content_b": pair["content_b"],
                            "similarity": float(pair["similarity"]),
                            "conflict_score": conflict_signal + (float(pair["similarity"]) - threshold) * 0.5,
                            "reason": ", ".join(reason) if reason else "high similarity",
                        }
                    )

            conflicts.sort(key=lambda x: x["conflict_score"], reverse=True)

            recorded = []
            if auto_record and conflicts:
                for c in conflicts:
                    cur.execute(
                        """
                        INSERT INTO contentions (article_id, related_article_id, type, description, severity)
                        VALUES (%s, %s, 'contradiction', %s, %s)
                        ON CONFLICT DO NOTHING
                        RETURNING id
                        """,
                        (
                            c["id_a"],
                            c["id_b"],
                            f"Auto-detected: {c['reason']}",
                            "high" if c["conflict_score"] > 0.5 else "medium",
                        ),
                    )
                    row = cur.fetchone()
                    if row:
                        recorded.append(str(row["id"]))

        result = {
            "success": True,
            "conflicts": conflicts,
            "count": len(conflicts),
            "threshold": threshold,
            "recorded_contentions": recorded,
        }
        return format_response(result, output_format, text_formatter=format_conflicts_text)
    except Exception:
        logger.exception("Error detecting conflicts")
        return internal_error()
