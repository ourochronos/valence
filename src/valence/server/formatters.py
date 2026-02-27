# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Server-side text/table formatters for REST responses.

Each formatter takes a response data dict and returns a human-readable string.
Used by endpoints when ?output=text or ?output=table is requested.
"""

from __future__ import annotations

from typing import Any


def format_stats_text(data: dict[str, Any]) -> str:
    """Format stats response as human-readable text."""
    stats = data.get("stats", {})
    lines = ["Valence Statistics", "-" * 30]
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        lines.append(f"  {label}: {value}")
    return "\n".join(lines)


def format_conflicts_text(data: dict[str, Any]) -> str:
    """Format conflict detection results as text."""
    conflicts = data.get("conflicts", [])
    if not conflicts:
        return "No potential conflicts detected."

    lines = [f"Found {len(conflicts)} potential conflict(s)", ""]
    for i, c in enumerate(conflicts[:10], 1):
        lines.append("=" * 60)
        sim = c.get("similarity", 0)
        score = c.get("conflict_score", 0)
        lines.append(f"Conflict #{i} (similarity: {sim:.1%}, signal: {score:.2f})")
        lines.append(f"Reason: {c.get('reason', '?')}")
        lines.append(f"  A [{str(c.get('id_a', ''))[:8]}] {str(c.get('content_a', ''))[:70]}")
        lines.append(f"  B [{str(c.get('id_b', ''))[:8]}] {str(c.get('content_b', ''))[:70]}")
        lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_maintenance_text(data: dict[str, Any]) -> str:
    """Format maintenance results as text."""
    results = data.get("results", [])
    dry_run = data.get("dry_run", False)
    dry_label = " (dry run)" if dry_run else ""

    lines = [f"Maintenance Report{dry_label}", "=" * 50]
    for r in results:
        op = r.get("operation", "?")
        details = {k: v for k, v in r.items() if k not in ("operation", "dry_run")}
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items()) if details else ""
        lines.append(f"  {op}: {detail_str}")
    lines.append("=" * 50)
    lines.append(f"  {len(results)} operation(s) completed")
    return "\n".join(lines)


def format_sessions_list_text(data: dict[str, Any]) -> str:
    """Format sessions list as text."""
    sessions = data.get("sessions", [])
    if not sessions:
        return "No sessions found."

    lines = [f"Found {len(sessions)} session(s)", ""]
    for s in sessions:
        sid = str(s.get("id", ""))[:8]
        status = s.get("status", "?")
        platform = s.get("platform", "?")
        lines.append(f"  [{sid}] ({status}) platform={platform}")
    return "\n".join(lines)


def format_migration_status_text(data: dict[str, Any]) -> str:
    """Format migration status as text."""
    migrations = data.get("migrations", [])
    if not migrations:
        return "No migrations found."

    lines = ["Migration Status", "-" * 40]
    for m in migrations:
        name = m.get("name", "?")
        status = m.get("status", "?")
        lines.append(f"  [{status}] {name}")
    return "\n".join(lines)


def format_embeddings_status_text(data: dict[str, Any]) -> str:
    """Format embeddings status as text."""
    stats = data.get("stats", {})
    lines = ["Embedding Status", "-" * 30]
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        lines.append(f"  {label}: {value}")
    return "\n".join(lines)
