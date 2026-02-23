"""Admin REST endpoints for database management operations.

All endpoints require admin:write scope (bearer tokens get full access automatically).
"""

from __future__ import annotations

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

from .auth_helpers import authenticate, require_scope
from .endpoint_utils import format_response, parse_output_format
from .errors import internal_error, invalid_json_error, missing_field_error
from .formatters import format_embeddings_status_text, format_maintenance_text, format_migration_status_text

logger = logging.getLogger(__name__)

ADMIN_SCOPE = "admin:write"


# =============================================================================
# MIGRATIONS
# =============================================================================


async def admin_migrate_status(request: Request) -> JSONResponse:
    """GET /api/v1/admin/migrate/status — Show migration status."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, ADMIN_SCOPE):
        return err

    output_format = parse_output_format(request)

    try:
        from valence.lib.our_db import get_cursor

        with get_cursor() as cur:
            cur.execute("""
                SELECT name, applied_at
                FROM schema_migrations
                ORDER BY applied_at
            """)
            applied = cur.fetchall()

        migrations = [{"name": m["name"], "status": "applied", "applied_at": str(m["applied_at"])} for m in applied]

        result = {"success": True, "migrations": migrations, "count": len(migrations)}
        return format_response(result, output_format, text_formatter=format_migration_status_text)
    except Exception:
        logger.exception("Error getting migration status")
        return internal_error()


async def admin_migrate_up(request: Request) -> JSONResponse:
    """POST /api/v1/admin/migrate/up — Apply pending migrations."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, ADMIN_SCOPE):
        return err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}

    dry_run = body.get("dry_run", False)

    try:
        from ..core.migrations import MigrationRunner

        runner = MigrationRunner()
        applied = runner.up(dry_run=dry_run)

        return JSONResponse(
            {
                "success": True,
                "applied": applied,
                "dry_run": dry_run,
            }
        )
    except Exception:
        logger.exception("Migration up failed")
        return internal_error()


async def admin_migrate_down(request: Request) -> JSONResponse:
    """POST /api/v1/admin/migrate/down — Rollback the last migration."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, ADMIN_SCOPE):
        return err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}

    dry_run = body.get("dry_run", False)

    try:
        from ..core.migrations import MigrationRunner

        runner = MigrationRunner()
        rolled_back = runner.down(dry_run=dry_run)

        return JSONResponse(
            {
                "success": True,
                "rolled_back": rolled_back,
                "dry_run": dry_run,
            }
        )
    except Exception:
        logger.exception("Migration down failed")
        return internal_error()


# =============================================================================
# MAINTENANCE
# =============================================================================


async def admin_maintenance(request: Request) -> JSONResponse:
    """POST /api/v1/admin/maintenance — Run maintenance operations.

    JSON body fields (all optional booleans):
        retention, archive, tombstones, compact, views, vacuum, all, dry_run
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

    ops_requested = any(body.get(op) for op in ("retention", "archive", "tombstones", "compact", "views", "vacuum"))
    if not run_all and not ops_requested:
        return missing_field_error("at least one operation (retention, archive, tombstones, compact, views, vacuum, all)")

    try:
        from ..cli.utils import get_db_connection
        from ..core.maintenance import (
            MaintenanceResult,
            apply_retention,
            archive_beliefs,
            cleanup_tombstones,
            compact_exchanges,
            refresh_views,
            run_full_maintenance,
            vacuum_analyze,
        )

        conn = get_db_connection()
        results: list[MaintenanceResult] = []

        try:
            if run_all:
                if not dry_run:
                    conn.autocommit = True
                cur = conn.cursor()
                results = run_full_maintenance(cur, dry_run=dry_run)
            else:
                cur = conn.cursor()

                if body.get("retention"):
                    results.extend(apply_retention(cur, dry_run=dry_run))
                if body.get("archive"):
                    results.append(archive_beliefs(cur, dry_run=dry_run))
                if body.get("tombstones"):
                    results.append(cleanup_tombstones(cur, dry_run=dry_run))
                if body.get("compact"):
                    results.append(compact_exchanges(cur, dry_run=dry_run))
                if body.get("views") and not dry_run:
                    results.append(refresh_views(cur))
                elif body.get("views"):
                    results.append(MaintenanceResult(operation="refresh_views", details={"note": "skipped in dry-run"}, dry_run=True))
                if body.get("vacuum") and not dry_run:
                    conn.autocommit = True
                    results.append(vacuum_analyze(cur))
                elif body.get("vacuum"):
                    results.append(MaintenanceResult(operation="vacuum_analyze", details={"note": "skipped in dry-run"}, dry_run=True))

                if not conn.autocommit:
                    conn.commit()
        finally:
            cur.close()
            conn.close()

        result_dicts = [{"operation": r.operation, "dry_run": r.dry_run, **r.details} for r in results]

        result = {
            "success": True,
            "results": result_dicts,
            "count": len(result_dicts),
            "dry_run": dry_run,
        }
        return format_response(result, output_format, text_formatter=format_maintenance_text)
    except Exception:
        logger.exception("Maintenance failed")
        return internal_error()


# =============================================================================
# EMBEDDINGS
# =============================================================================


async def admin_embeddings_status(request: Request) -> JSONResponse:
    """GET /api/v1/admin/embeddings/status — Embedding coverage status."""
    client = authenticate(request)
    if isinstance(client, JSONResponse):
        return client
    if err := require_scope(client, ADMIN_SCOPE):
        return err

    output_format = parse_output_format(request)

    try:
        from valence.lib.our_db import get_cursor

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
        from valence.lib.our_db import get_cursor
        from valence.lib.our_embeddings.service import generate_embedding

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
        from valence.lib.our_db import get_cursor

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
        from valence.lib.our_db import get_cursor

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
