# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Maintenance operations for the Valence knowledge system.

Provides vacuum, view refresh, and orchestrated maintenance cycles.
"""

from __future__ import annotations

import json
import logging as _logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg2 import sql

from valence.core.compilation import compile_article
from valence.core.db import get_cursor
from valence.core.sources import find_similar_ungrouped


@dataclass
class MaintenanceResult:
    """Result of a maintenance operation."""

    operation: str
    details: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def summary(self) -> str:
        """Get a summary of the maintenance operation.

        Returns:
            String describing the operation and its details.
        """
        return f"{self.operation}: {self.details}"


VACUUM_TABLES = [
    "articles",
    "sources",
    "article_sources",
    "usage_traces",
    "contentions",
    "entities",
    "article_entities",
    "system_config",
    "article_mutations",
    "mutation_queue",
]


def vacuum_analyze(cur) -> MaintenanceResult:
    """Run VACUUM ANALYZE on all known tables."""
    vacuumed = []
    for table in VACUUM_TABLES:
        cur.execute(sql.SQL("VACUUM ANALYZE {}").format(sql.Identifier(table)))
        vacuumed.append(table)

    return MaintenanceResult(
        operation="vacuum_analyze",
        details={"tables_vacuumed": len(vacuumed), "tables": vacuumed},
    )


def refresh_views(cur, concurrent: bool = True) -> MaintenanceResult:
    """Refresh materialized views if any exist."""
    cur.execute("SELECT matviewname FROM pg_matviews WHERE schemaname = 'public'")
    views = [row["matviewname"] for row in cur.fetchall()]

    refreshed = []
    for view in views:
        keyword = "CONCURRENTLY" if concurrent else ""
        cur.execute(
            sql.SQL("REFRESH MATERIALIZED VIEW {} {}").format(
                sql.SQL(keyword),
                sql.Identifier(view),
            )
        )
        refreshed.append(view)

    return MaintenanceResult(
        operation="refresh_views",
        details={"views_refreshed": len(refreshed), "views": refreshed},
    )


def run_full_maintenance(
    cur,
    dry_run: bool = False,
    skip_vacuum: bool = False,
    skip_views: bool = False,
) -> list[MaintenanceResult]:
    """Run full maintenance cycle.

    Order: refresh views -> vacuum analyze
    """
    results: list[MaintenanceResult] = []

    if not dry_run and not skip_views:
        results.append(refresh_views(cur))

    if not dry_run and not skip_vacuum:
        results.append(vacuum_analyze(cur))

    return results


def get_maintenance_schedule(cur) -> dict[str, Any] | None:
    """Get current maintenance schedule configuration from system_config.

    Returns:
        Dict with 'interval_hours' and 'last_run' (ISO timestamp), or None if disabled.
    """
    cur.execute("SELECT value FROM system_config WHERE key = 'maintenance_schedule'")
    row = cur.fetchone()
    if not row:
        return None

    schedule = row["value"]
    if isinstance(schedule, str):
        schedule = json.loads(schedule)

    # Handle disabled state
    if schedule.get("enabled") is False:
        return None

    return schedule


def set_maintenance_schedule(cur, interval_hours: int) -> dict[str, Any]:
    """Set maintenance schedule interval in system_config.

    Args:
        cur: Database cursor
        interval_hours: Hours between maintenance runs (must be > 0)

    Returns:
        Updated schedule configuration
    """
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")

    schedule = {
        "enabled": True,
        "interval_hours": interval_hours,
        "last_run": None,
    }

    cur.execute(
        """
        INSERT INTO system_config (key, value, updated_at)
        VALUES ('maintenance_schedule', %s, now())
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value, updated_at = now()
        """,
        (json.dumps(schedule),),
    )

    return schedule


def disable_maintenance_schedule(cur) -> None:
    """Disable scheduled maintenance."""
    cur.execute(
        """
        INSERT INTO system_config (key, value, updated_at)
        VALUES ('maintenance_schedule', %s, now())
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value, updated_at = now()
        """,
        (json.dumps({"enabled": False}),),
    )


def check_and_run_maintenance(cur) -> dict[str, Any] | None:
    """Check if maintenance should run and execute if needed.

    Lightweight check designed to be called on server startup or CLI invocation.
    Reads maintenance_schedule from system_config, compares timestamps, and runs
    if enough time has elapsed.

    Returns:
        Dict with maintenance results if run, None if skipped
    """
    schedule = get_maintenance_schedule(cur)

    # Not configured or disabled
    if not schedule:
        return None

    interval_hours = schedule.get("interval_hours", 24)
    last_run_str = schedule.get("last_run")

    # Determine if we should run
    should_run = False
    if last_run_str is None:
        # Never run before
        should_run = True
    else:
        # Parse last_run and check elapsed time
        last_run = datetime.fromisoformat(last_run_str.replace("Z", "+00:00"))
        now = datetime.now(UTC)
        elapsed = now - last_run

        if elapsed >= timedelta(hours=interval_hours):
            should_run = True

    if not should_run:
        return None

    # Run maintenance
    results = run_full_maintenance(cur, dry_run=False, skip_vacuum=False, skip_views=False)

    # Update last_run timestamp
    schedule["last_run"] = datetime.now(UTC).isoformat()
    cur.execute(
        """
        UPDATE system_config
        SET value = %s, updated_at = now()
        WHERE key = 'maintenance_schedule'
        """,
        (json.dumps(schedule),),
    )

    return {
        "maintenance_run": True,
        "timestamp": schedule["last_run"],
        "results": [{"operation": r.operation, "details": r.details} for r in results],
    }


# ---------------------------------------------------------------------------
# Recompile ungrouped sources
# ---------------------------------------------------------------------------


async def recompile_ungrouped_sources(limit: int = 50) -> dict:
    """Re-check ungrouped sources for compilation candidates.

    Finds sources with pipeline_status = 'complete' that have no article_sources
    row, then attempts to compile clusters of similar ungrouped sources.

    Args:
        limit: Maximum number of ungrouped sources to check.

    Returns:
        Dict with keys: checked, compiled, articles_created.
    """
    from valence.core.ingest_pipeline import AUTO_COMPILE_MIN_CLUSTER, COMPILE_SIMILARITY_THRESHOLD

    log = _logging.getLogger(__name__)

    # Fetch ungrouped sources
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT s.id
            FROM sources s
            WHERE s.pipeline_status = 'complete'
              AND NOT EXISTS (
                  SELECT 1 FROM article_sources a WHERE a.source_id = s.id
              )
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()

    ungrouped_ids = [row["id"] for row in rows]
    checked = len(ungrouped_ids)
    compiled = 0
    articles_created = 0
    processed: set[str] = set()

    for source_id in ungrouped_ids:
        sid = str(source_id)
        if sid in processed:
            continue

        try:
            similar_ids = await find_similar_ungrouped(sid, COMPILE_SIMILARITY_THRESHOLD)
            candidate_ids = [sid] + [s for s in similar_ids if s != sid]

            if len(candidate_ids) >= AUTO_COMPILE_MIN_CLUSTER:
                result = await compile_article(candidate_ids[:10])
                if result.success:
                    log.info(
                        "recompile_ungrouped: compiled %d sources → article %s",
                        len(candidate_ids),
                        (result.data or {}).get("id", "?"),
                    )
                    articles_created += 1
                    compiled += len(candidate_ids)
                    processed.update(candidate_ids)
                else:
                    log.warning("recompile_ungrouped: compile failed: %s", result.error)
            else:
                log.debug(
                    "recompile_ungrouped: only %d candidates for %s (need %d), skipping",
                    len(candidate_ids),
                    sid,
                    AUTO_COMPILE_MIN_CLUSTER,
                )
        except Exception as exc:
            log.warning("recompile_ungrouped: error processing %s: %s (non-fatal)", sid, exc)

    return {"checked": checked, "compiled": compiled, "articles_created": articles_created}
