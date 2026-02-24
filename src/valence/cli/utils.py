# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Utility functions for Valence CLI."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def get_db_connection():  # type: ignore[no-untyped-def]
    """Get a database connection from the pool.

    This is a thin wrapper around core.db.get_connection() for backward compatibility.

    Returns:
        psycopg2 connection object

    Raises:
        DatabaseError: If connection fails
    """
    from valence.core import db

    return db.get_connection()


def format_confidence(conf: dict) -> str:
    """Format confidence for display."""
    if not conf:
        return "?"
    overall = conf.get("overall", 0)
    if isinstance(overall, int | float):
        return f"{overall:.0%}"
    return str(overall)[:5]


def format_age(dt: datetime) -> str:
    """Format datetime as human-readable age."""
    if not dt:
        return "?"

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    now = datetime.now(UTC)
    delta = now - dt

    if delta.days > 365:
        return f"{delta.days // 365}y"
    elif delta.days > 30:
        return f"{delta.days // 30}mo"
    elif delta.days > 0:
        return f"{delta.days}d"
    elif delta.seconds > 3600:
        return f"{delta.seconds // 3600}h"
    elif delta.seconds > 60:
        return f"{delta.seconds // 60}m"
    else:
        return "now"


# ============================================================================
# Multi-Signal Ranking (Valence Query Protocol)
# Re-exported from core.ranking for backward compatibility
# ============================================================================

from ..core.ranking import compute_confidence_score, compute_recency_score, multi_signal_rank  # noqa: E402, F401
