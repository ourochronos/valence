# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Admin REST endpoints for database management operations.

All endpoints require admin:write scope (bearer tokens get full access automatically).
"""

from __future__ import annotations

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .auth_helpers import authenticate, require_scope
from .endpoint_utils import format_response, parse_output_format
from .errors import internal_error, invalid_json_error, missing_field_error
from .formatters import format_embeddings_status_text, format_maintenance_text

logger = logging.getLogger(__name__)

ADMIN_SCOPE = "admin:write"


# =============================================================================
# MAINTENANCE
# =============================================================================


async def admin_maintenance(request: Request) -> Response:
    """POST /api/v1/admin/maintenance — Run maintenance operations.

    JSON body fields (all optional booleans):
        views, vacuum, all — legacy DB maintenance
        recompute_scores — batch-recompute usage_score for all articles
        process_queue — process pending mutation queue entries
        evict_if_over_capacity — run organic forgetting
        evict_count — max articles to evict (default 10)
        dry_run — preview without changes
    """
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, ADMIN_SCOPE):
        return err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    output_format = parse_output_format(request)
    dry_run = body.get("dry_run", False)
    run_all = body.get("all", False)

    # v2 knowledge operations
    v2_ops = any(body.get(op) for op in ("recompute_scores", "process_queue", "evict_if_over_capacity"))
    # Legacy DB operations
    legacy_ops = any(body.get(op) for op in ("views", "vacuum"))

    if not run_all and not v2_ops and not legacy_ops:
        return missing_field_error("at least one operation (all, views, vacuum, recompute_scores, process_queue, evict_if_over_capacity)")

    try:
        from ..core.maintenance import MaintenanceResult

        results: list[dict] = []

        # --- v2 knowledge operations ---
        if run_all or body.get("recompute_scores"):
            if dry_run:
                results.append({"operation": "recompute_scores", "dry_run": True, "note": "skipped in dry-run"})
            else:
                from ..core.usage import compute_usage_scores

                res = await compute_usage_scores()
                results.append(
                    {
                        "operation": "recompute_scores",
                        "dry_run": False,
                        "success": res.success,
                        **(res.data if isinstance(res.data, dict) and res.success else {"result": res.data} if res.success else {"error": res.error}),
                    }
                )

        if run_all or body.get("process_queue"):
            if dry_run:
                results.append({"operation": "process_queue", "dry_run": True, "note": "skipped in dry-run"})
            else:
                from ..core.compilation import process_mutation_queue

                res = await process_mutation_queue()
                results.append(
                    {
                        "operation": "process_queue",
                        "dry_run": False,
                        "success": res.success,
                        **(res.data if isinstance(res.data, dict) and res.success else {"result": res.data} if res.success else {"error": res.error}),
                    }
                )

        if body.get("evict_if_over_capacity"):
            evict_count = body.get("evict_count", 10)
            if dry_run:
                results.append({"operation": "evict_if_over_capacity", "dry_run": True, "note": "skipped in dry-run"})
            else:
                from ..core.forgetting import evict_lowest

                res = await evict_lowest(count=evict_count)
                results.append(
                    {
                        "operation": "evict_if_over_capacity",
                        "dry_run": False,
                        "success": res.success,
                        **(res.data if isinstance(res.data, dict) and res.success else {"result": res.data} if res.success else {"error": res.error}),
                    }
                )

        # --- Legacy DB maintenance ---
        if run_all or legacy_ops:
            import psycopg2
            import psycopg2.extras

            from ..core.db import get_connection_params
            from ..core.maintenance import (
                refresh_views,
                run_full_maintenance,
                vacuum_analyze,
            )

            params = get_connection_params()
            conn = psycopg2.connect(
                cursor_factory=psycopg2.extras.RealDictCursor,
                **params,
            )
            legacy_results: list[MaintenanceResult] = []
            try:
                if run_all:
                    if not dry_run:
                        conn.autocommit = True
                    cur = conn.cursor()
                    legacy_results = run_full_maintenance(cur, dry_run=dry_run)
                    cur.close()
                else:
                    cur = conn.cursor()
                    if body.get("views") and not dry_run:
                        legacy_results.append(refresh_views(cur))
                    elif body.get("views"):
                        legacy_results.append(MaintenanceResult(operation="refresh_views", details={"note": "skipped in dry-run"}, dry_run=True))
                    if body.get("vacuum") and not dry_run:
                        conn.autocommit = True
                        legacy_results.append(vacuum_analyze(cur))
                    elif body.get("vacuum"):
                        legacy_results.append(MaintenanceResult(operation="vacuum_analyze", details={"note": "skipped in dry-run"}, dry_run=True))
                    if not conn.autocommit:
                        conn.commit()
                    cur.close()

                for r in legacy_results:
                    results.append({"operation": r.operation, "dry_run": r.dry_run, **r.details})
            finally:
                conn.close()

        result = {
            "success": True,
            "results": results,
            "count": len(results),
            "dry_run": dry_run,
        }
        return format_response(result, output_format, text_formatter=format_maintenance_text)
    except Exception:
        logger.exception("Maintenance failed")
        return internal_error()


# =============================================================================
# EMBEDDINGS
# =============================================================================


async def admin_embeddings_status(request: Request) -> Response:
    """GET /api/v1/admin/embeddings/status — Embedding coverage status."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, ADMIN_SCOPE):
        return err

    output_format = parse_output_format(request)

    try:
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM articles")
            total = cur.fetchone()["total"]
            cur.execute("SELECT COUNT(*) as embedded FROM articles WHERE embedding IS NOT NULL")
            embedded = cur.fetchone()["embedded"]
            cur.execute("SELECT COUNT(*) as missing FROM articles WHERE embedding IS NULL")
            missing = cur.fetchone()["missing"]

        coverage = f"{embedded / total:.1%}" if total > 0 else "N/A"
        result = {
            "success": True,
            "stats": {
                "total_articles": total,
                "with_embeddings": embedded,
                "missing_embeddings": missing,
                "coverage": coverage,
            },
        }
        return format_response(result, output_format, text_formatter=format_embeddings_status_text)
    except Exception:
        logger.exception("Error getting embedding status")
        return internal_error()


async def admin_embeddings_backfill(request: Request) -> JSONResponse:
    """POST /api/v1/admin/embeddings/backfill — Backfill missing embeddings."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, ADMIN_SCOPE):
        return err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}

    batch_size = body.get("batch_size", 50)
    dry_run = body.get("dry_run", False)

    try:
        from valence.core.db import get_cursor
        from valence.core.embeddings import generate_embedding

        with get_cursor() as cur:
            cur.execute(
                "SELECT id, content FROM articles WHERE embedding IS NULL AND status = 'active' LIMIT %s",
                (batch_size,),
            )
            rows = cur.fetchall()

        if dry_run:
            return JSONResponse(
                {
                    "success": True,
                    "would_process": len(rows),
                    "dry_run": True,
                }
            )

        processed = 0
        errors = 0
        for row in rows:
            try:
                embedding = generate_embedding(row["content"])
                if embedding:
                    with get_cursor() as cur:
                        cur.execute("UPDATE articles SET embedding = %s WHERE id = %s", (embedding, row["id"]))
                    processed += 1
            except Exception:
                errors += 1

        return JSONResponse(
            {
                "success": True,
                "processed": processed,
                "errors": errors,
                "remaining": len(rows) - processed - errors,
            }
        )
    except Exception:
        logger.exception("Embedding backfill failed")
        return internal_error()


async def admin_embeddings_migrate(request: Request) -> JSONResponse:
    """POST /api/v1/admin/embeddings/migrate — Migrate embedding model/dimensions."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, ADMIN_SCOPE):
        return err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    model = body.get("model")
    dims = body.get("dims")
    if not model and not dims:
        return missing_field_error("model or dims")

    dry_run = body.get("dry_run", False)

    try:
        from valence.core.db import get_cursor

        if dry_run:
            with get_cursor() as cur:
                cur.execute("SELECT COUNT(*) as count FROM articles WHERE embedding IS NOT NULL")
                count = cur.fetchone()["count"]
            return JSONResponse(
                {
                    "success": True,
                    "would_affect": count,
                    "model": model,
                    "dims": dims,
                    "dry_run": True,
                }
            )

        with get_cursor() as cur:
            # NULL out all embeddings (they need regeneration with new model)
            cur.execute("UPDATE articles SET embedding = NULL WHERE embedding IS NOT NULL")
            affected = cur.rowcount

        return JSONResponse(
            {
                "success": True,
                "cleared_embeddings": affected,
                "model": model,
                "dims": dims,
                "note": "Run /admin/embeddings/backfill to regenerate with new model",
            }
        )
    except Exception:
        logger.exception("Embedding migration failed")
        return internal_error()


# =============================================================================
# CHAIN VERIFICATION
# =============================================================================


async def admin_verify_chains(request: Request) -> JSONResponse:
    """GET /api/v1/admin/verify-chains — Verify supersession chain integrity."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, ADMIN_SCOPE):
        return err

    try:
        from valence.core.db import get_cursor

        issues = []
        with get_cursor() as cur:
            # Check for broken supersession chains (pointing to non-existent articles)
            cur.execute("""
                SELECT b.id, b.superseded_by_id
                FROM articles b
                WHERE b.superseded_by_id IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM articles b2 WHERE b2.id = b.superseded_by_id)
            """)
            broken = cur.fetchall()
            for row in broken:
                issues.append(
                    {
                        "type": "broken_chain",
                        "belief_id": str(row["id"]),
                        "points_to": str(row["superseded_by_id"]),
                        "description": "Superseded belief points to non-existent belief",
                    }
                )

            # Check for cycles
            cur.execute("""
                WITH RECURSIVE chain AS (
                    SELECT id, superseded_by_id, 1 as depth, ARRAY[id] as path
                    FROM articles WHERE superseded_by_id IS NOT NULL
                    UNION ALL
                    SELECT b.id, b.superseded_by_id, c.depth + 1, c.path || b.id
                    FROM articles b JOIN chain c ON b.id = c.superseded_by_id
                    WHERE c.depth < 100 AND NOT (b.id = ANY(c.path))
                )
                SELECT id, depth FROM chain WHERE depth >= 50
            """)
            deep = cur.fetchall()
            for row in deep:
                issues.append(
                    {
                        "type": "deep_chain",
                        "belief_id": str(row["id"]),
                        "depth": row["depth"],
                        "description": "Supersession chain is unusually deep",
                    }
                )

        return JSONResponse(
            {
                "success": True,
                "issues": issues,
                "count": len(issues),
                "status": "healthy" if not issues else "issues_found",
            }
        )
    except Exception:
        logger.exception("Chain verification failed")
        return internal_error()
