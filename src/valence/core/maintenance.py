"""Maintenance operations for the Valence knowledge system.

Provides vacuum, view refresh, and orchestrated maintenance cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from psycopg2 import sql


@dataclass
class MaintenanceResult:
    """Result of a maintenance operation."""

    operation: str
    details: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def summary(self) -> str:
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
