# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Confidence computation for knowledge articles.

Centralises the confidence formula so it's defined once and used by
compilation, recompilation, batch backfill, and any future paths.

Formula:
    confidence = min(0.95, avg_source_reliability + source_bonus)
    source_bonus = min(0.15, ln(1 + n_sources - 1) * 0.1)  when n > 1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    """Output of confidence computation."""

    overall: float  # clamped [0, 0.95]
    avg_reliability: float
    source_bonus: float
    corroboration_count: int

    def to_jsonb(self) -> dict[str, float]:
        """Return the JSONB payload for the confidence column."""
        return {"overall": round(self.overall, 3)}


def compute_confidence(sources: list[dict[str, Any]]) -> ConfidenceResult:
    """Compute article confidence from its linked sources.

    Args:
        sources: List of source dicts, each with an optional ``reliability``
                 key (float, default 0.5).

    Returns:
        ConfidenceResult with overall score, avg reliability, source bonus,
        and corroboration count.
    """
    if not sources:
        return ConfidenceResult(
            overall=0.5,
            avg_reliability=0.5,
            source_bonus=0.0,
            corroboration_count=0,
        )

    reliabilities = [float(s.get("reliability", 0.5)) for s in sources]
    avg_reliability = sum(reliabilities) / len(reliabilities)
    n = len(sources)

    if n > 1:
        source_bonus = min(0.15, math.log(1 + n - 1) * 0.1)
    else:
        source_bonus = 0.0

    overall = min(0.95, avg_reliability + source_bonus)

    return ConfidenceResult(
        overall=round(overall, 4),
        avg_reliability=round(avg_reliability, 4),
        source_bonus=round(source_bonus, 4),
        corroboration_count=n,
    )
