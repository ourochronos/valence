"""Database maintenance operations: retention, archival, compaction.

Functions call stored procedures for the heavy lifting. This module provides
the Python interface and configuration for CLI and scheduled operations.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetentionConfig:
    """Per-table retention configuration (in days).

    None means keep forever. audit_log is always kept (GDPR 7-year minimum).
    """

    belief_retrievals_days: int = 90
    sync_events_days: int = 90
    embedding_coverage_days: int | None = None  # keep forever by default
    # audit_log: NEVER deleted (hardcoded in stored procedure)


@dataclass
class ArchivalConfig:
    """Belief archival configuration."""

    older_than_days: int = 180
    batch_size: int = 1000


@dataclass
class MaintenanceConfig:
    """Combined maintenance configuration."""

    retention: RetentionConfig = field(default_factory=RetentionConfig)
    archival: ArchivalConfig = field(default_factory=ArchivalConfig)


@dataclass
class MaintenanceResult:
    """Result from a maintenance operation."""

    operation: str
    details: dict[str, Any]
    dry_run: bool = False

    def __str__(self) -> str:
        status = " (dry run)" if self.dry_run else ""
        items = ", ".join(f"{k}={v}" for k, v in self.details.items())
        return f"{self.operation}{status}: {items}"


def apply_retention(cur, config: RetentionConfig | None = None, dry_run: bool = False) -> list[MaintenanceResult]:
    """Apply retention policies via stored procedure.

    Args:
        cur: Database cursor
        config: Retention configuration (uses defaults if None)
        dry_run: If True, only report what would be deleted

    Returns:
        List of MaintenanceResult with per-table deletion counts
    """
    config = config or RetentionConfig()

    cur.execute(
        "SELECT * FROM apply_retention_policies(%s, %s, %s, %s)",
        (
            config.belief_retrievals_days,
            config.sync_events_days,
            config.embedding_coverage_days,
            dry_run,
        ),
    )

    results = []
    for row in cur.fetchall():
        results.append(
            MaintenanceResult(
                operation="retention",
                details={"table": row["table_name"], "deleted": row["deleted_count"]},
                dry_run=dry_run,
            )
        )
    return results


def archive_beliefs(cur, config: ArchivalConfig | None = None, dry_run: bool = False) -> MaintenanceResult:
    """Archive stale superseded beliefs via stored procedure.

    Args:
        cur: Database cursor
        config: Archival configuration (uses defaults if None)
        dry_run: If True, only report what would be archived

    Returns:
        MaintenanceResult with archival counts
    """
    config = config or ArchivalConfig()

    cur.execute(
        "SELECT * FROM archive_stale_beliefs(%s, %s, %s)",
        (config.older_than_days, config.batch_size, dry_run),
    )
    row = cur.fetchone()

    return MaintenanceResult(
        operation="archival",
        details={"archived": row["archived_count"], "freed_embeddings": row["freed_embeddings"]},
        dry_run=dry_run,
    )


def cleanup_tombstones(cur, dry_run: bool = False) -> MaintenanceResult:
    """Remove expired tombstones.

    Args:
        cur: Database cursor
        dry_run: If True, only report what would be cleaned

    Returns:
        MaintenanceResult with cleanup count
    """
    cur.execute("SELECT cleanup_expired_tombstones(%s) as count", (dry_run,))
    row = cur.fetchone()

    return MaintenanceResult(
        operation="tombstone_cleanup",
        details={"removed": row["count"]},
        dry_run=dry_run,
    )


@dataclass
class CompactionConfig:
    """Exchange compaction configuration."""

    keep_first: int = 5  # Keep first N exchanges verbatim
    keep_last: int = 5  # Keep last N exchanges verbatim
    min_exchanges: int = 15  # Only compact sessions with more than this many exchanges


def compact_exchanges(
    cur,
    config: CompactionConfig | None = None,
    dry_run: bool = False,
) -> MaintenanceResult:
    """Compact exchanges in completed sessions.

    Keeps first+last N exchanges verbatim, replaces middle with a summary JSON
    containing: exchange_count, topics (from tool_uses), roles distribution,
    total_tokens. Never compacts active sessions.

    Args:
        cur: Database cursor
        config: Compaction settings (uses defaults if None)
        dry_run: If True, only report what would be compacted

    Returns:
        MaintenanceResult with compaction counts
    """
    config = config or CompactionConfig()
    total_compacted = 0
    total_exchanges_removed = 0

    # Find completed/abandoned sessions that haven't been compacted yet
    # and have enough exchanges to be worth compacting
    cur.execute(
        """
        SELECT s.id, COUNT(e.id) as exchange_count
        FROM vkb_sessions s
        JOIN vkb_exchanges e ON s.id = e.session_id
        WHERE s.status IN ('completed', 'abandoned')
          AND s.compacted_at IS NULL
        GROUP BY s.id
        HAVING COUNT(e.id) > %s
        """,
        (config.min_exchanges,),
    )
    candidates = cur.fetchall()

    for row in candidates:
        session_id = row["id"]
        exchange_count = row["exchange_count"]

        if dry_run:
            middle_count = exchange_count - config.keep_first - config.keep_last
            total_compacted += 1
            total_exchanges_removed += middle_count
            continue

        # Get all exchanges ordered by sequence
        cur.execute(
            "SELECT id, sequence, role, content, tokens_approx, tool_uses FROM vkb_exchanges WHERE session_id = %s ORDER BY sequence",
            (session_id,),
        )
        exchanges = cur.fetchall()

        if len(exchanges) <= config.keep_first + config.keep_last:
            continue

        # Split: first N, middle, last N
        exchanges[: config.keep_first]
        middle_exchanges = exchanges[config.keep_first : -config.keep_last]
        exchanges[-config.keep_last :]

        # Build summary from middle exchanges
        summary = _build_compaction_summary(middle_exchanges)

        # Store summary on session
        cur.execute(
            "UPDATE vkb_sessions SET compacted_summary = %s, compacted_at = NOW() WHERE id = %s",
            (json.dumps(summary), session_id),
        )

        # Delete middle exchanges
        middle_ids = [e["id"] for e in middle_exchanges]
        cur.execute(
            "DELETE FROM vkb_exchanges WHERE id = ANY(%s)",
            (middle_ids,),
        )

        total_compacted += 1
        total_exchanges_removed += len(middle_exchanges)

    return MaintenanceResult(
        operation="exchange_compaction",
        details={
            "sessions_compacted": total_compacted,
            "exchanges_removed": total_exchanges_removed,
        },
        dry_run=dry_run,
    )


def _build_compaction_summary(exchanges: list[dict]) -> dict[str, Any]:
    """Build a summary JSON from the middle exchanges being compacted.

    Returns:
        Dict with exchange_count, roles, topics, total_tokens, tool_uses
    """
    roles = Counter()
    tool_names: Counter = Counter()
    total_tokens = 0

    for ex in exchanges:
        roles[ex["role"]] += 1
        total_tokens += ex.get("tokens_approx") or 0
        tool_uses = ex.get("tool_uses") or []
        if isinstance(tool_uses, str):
            tool_uses = json.loads(tool_uses)
        for tool in tool_uses:
            name = tool if isinstance(tool, str) else tool.get("name", "unknown")
            tool_names[name] += 1

    return {
        "exchange_count": len(exchanges),
        "roles": dict(roles),
        "total_tokens": total_tokens,
        "tool_uses": dict(tool_names.most_common(20)),
    }


def vacuum_analyze(cur) -> MaintenanceResult:
    """Run VACUUM ANALYZE on key tables.

    Note: VACUUM cannot run inside a transaction, so the connection
    must have autocommit=True.
    """
    tables = ["beliefs", "vkb_exchanges", "vkb_sessions", "belief_retrievals", "embedding_coverage"]
    for table in tables:
        cur.execute(f"VACUUM ANALYZE {table}")  # noqa: S608 - table names are hardcoded, not user input

    return MaintenanceResult(
        operation="vacuum_analyze",
        details={"tables": len(tables)},
    )


def refresh_views(cur, concurrent: bool = True) -> MaintenanceResult:
    """Refresh materialized views via stored procedure.

    Args:
        cur: Database cursor
        concurrent: If True, use REFRESH CONCURRENTLY (no read lock)

    Returns:
        MaintenanceResult with per-view refresh status
    """
    cur.execute("SELECT * FROM refresh_materialized_views(%s)", (concurrent,))
    rows = cur.fetchall()

    refreshed = [r["view_name"] for r in rows if r["refreshed"]]
    failed = [{"view": r["view_name"], "error": r["error_msg"]} for r in rows if not r["refreshed"]]

    return MaintenanceResult(
        operation="refresh_views",
        details={"refreshed": len(refreshed), "failed": len(failed), "views": refreshed, "errors": failed},
    )


def run_full_maintenance(
    cur,
    config: MaintenanceConfig | None = None,
    dry_run: bool = False,
    skip_vacuum: bool = False,
    skip_views: bool = False,
) -> list[MaintenanceResult]:
    """Run full maintenance cycle in the correct order.

    Order: retention -> archival -> tombstone cleanup -> exchange compaction -> refresh views -> vacuum
    """
    config = config or MaintenanceConfig()
    results: list[MaintenanceResult] = []

    # 1. Retention policies
    results.extend(apply_retention(cur, config.retention, dry_run))

    # 2. Belief archival
    results.append(archive_beliefs(cur, config.archival, dry_run))

    # 3. Tombstone cleanup
    results.append(cleanup_tombstones(cur, dry_run))

    # 4. Exchange compaction
    results.append(compact_exchanges(cur, dry_run=dry_run))

    # 5. Refresh materialized views (skip in dry-run mode)
    if not dry_run and not skip_views:
        results.append(refresh_views(cur))

    # 6. VACUUM ANALYZE (skip in dry-run mode or if requested)
    if not dry_run and not skip_vacuum:
        results.append(vacuum_analyze(cur))

    return results
