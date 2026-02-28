# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Multi-signal ranking for article retrieval.

Combines semantic similarity, confidence, and freshness into a single
final score. Used by MCP tools, CLI, and the retrieval layer.

Originally extracted from cli/utils.py for reuse across the codebase;
adapted in WU-05 to support articles (renamed from beliefs) and an
explicit freshness factor derived from source ages / article update times.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass
class RankingConfig:
    """Configuration for multi-signal ranking weights."""

    semantic_weight: float = 0.50
    confidence_weight: float = 0.35
    recency_weight: float = 0.15
    decay_rate: float = 0.01  # ~69 day half-life

    def normalized(self) -> RankingConfig:
        """Return a copy with weights normalized to sum to 1.0."""
        total = self.semantic_weight + self.confidence_weight + self.recency_weight
        if total <= 0:
            return RankingConfig()
        return RankingConfig(
            semantic_weight=self.semantic_weight / total,
            confidence_weight=self.confidence_weight / total,
            recency_weight=self.recency_weight / total,
            decay_rate=self.decay_rate,
        )


DEFAULT_RANKING = RankingConfig()

# ---------------------------------------------------------------------------
# Query intent detection (#70)
# ---------------------------------------------------------------------------

_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

_PROCEDURAL_PREFIXES = (
    "how to",
    "how should",
    "steps to",
    "process for",
    "what's the process",
    "whats the process",
)
_PROCEDURAL_CONTAINS = ("checklist", "workflow", "procedure")

_EPISODIC_PREFIXES = (
    "what happened",
    "when did",
)
_EPISODIC_CONTAINS = ("session", "last time")


def detect_query_intent(query: str) -> str:
    """Detect the intent of a query: "procedural", "episodic", or "general".

    Args:
        query: The search query string.

    Returns:
        One of "procedural", "episodic", or "general".
    """
    q = query.strip().lower()

    # Procedural checks
    for prefix in _PROCEDURAL_PREFIXES:
        if q.startswith(prefix):
            return "procedural"
    for term in _PROCEDURAL_CONTAINS:
        if term in q:
            return "procedural"

    # Episodic checks
    for prefix in _EPISODIC_PREFIXES:
        if q.startswith(prefix):
            return "episodic"
    for term in _EPISODIC_CONTAINS:
        if term in q:
            return "episodic"
    if _DATE_PATTERN.search(q):
        return "episodic"

    return "general"


# ---------------------------------------------------------------------------
# Cold-start helpers (#71)
# ---------------------------------------------------------------------------

_COLD_START_WINDOW_HOURS = 48
_COLD_START_FLOOR_FULL = 0.3  # floor at creation -> 24h
_COLD_START_FLOOR_HALF = 0.15  # floor at 24h -> 48h
_COLD_START_MIN_CONFIDENCE = 0.7


def _cold_start_floor(article: dict, confidence: float) -> float | None:
    """Return the cold-start score floor for a qualifying article, or None."""
    if confidence < _COLD_START_MIN_CONFIDENCE:
        return None

    created_at = article.get("created_at")
    if created_at is None:
        return None

    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    if isinstance(created_at, datetime):
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        age_hours = (now - created_at).total_seconds() / 3600.0
        if age_hours <= 24:
            return _COLD_START_FLOOR_FULL
        elif age_hours <= 48:
            return _COLD_START_FLOOR_HALF

    return None


def compute_confidence_score(article: dict) -> float:
    """Compute aggregated confidence score from 6D confidence vector.

    Works with both article dicts (v2) and legacy belief dicts (v1).
    Uses geometric mean to penalize articles with any weak dimension.
    Falls back to JSONB 'overall' field for backward compatibility.

    Args:
        article: Article (or legacy belief) dict from the database.
    """
    belief = article  # alias — same structure, renamed concept
    # Try 6D confidence columns first
    src = belief.get("confidence_source", 0.5)
    meth = belief.get("confidence_method", 0.5)
    cons = belief.get("confidence_consistency", 1.0)
    fresh = belief.get("confidence_freshness", 1.0)
    corr = belief.get("confidence_corroboration", 0.1)
    app = belief.get("confidence_applicability", 0.8)

    has_6d = any(
        [
            belief.get("confidence_source") is not None,
            belief.get("confidence_method") is not None,
        ]
    )

    if has_6d:
        # Geometric mean with spec weights
        # w_sr=0.25, w_mq=0.20, w_ic=0.15, w_tf=0.15, w_cor=0.15, w_da=0.10
        try:
            score = (src**0.25) * (meth**0.20) * (cons**0.15) * (fresh**0.15) * (corr**0.15) * (app**0.10)
            return min(1.0, max(0.0, score))
        except (ValueError, ZeroDivisionError):
            pass

    # Fallback to JSONB overall
    conf = belief.get("confidence", {})
    if isinstance(conf, dict):
        overall = conf.get("overall", 0.5)
        if isinstance(overall, int | float):
            return min(1.0, max(0.0, float(overall)))

    return 0.5


def compute_recency_score(created_at: datetime | str | None, decay_rate: float = 0.01) -> float:
    """Compute recency score with exponential decay.

    Default decay_rate=0.01 gives a half-life of ~69 days.
    Handles datetime objects, ISO format strings, and None.
    """
    if not created_at:
        return 0.5

    # Handle string dates (from serialized data)
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except (ValueError, TypeError):
            return 0.5

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)

    now = datetime.now(UTC)
    age_days = (now - created_at).total_seconds() / 86400

    recency = math.exp(-decay_rate * age_days)
    return min(1.0, max(0.0, recency))


def compute_freshness_score(
    article: dict,
    decay_rate: float = 0.01,
) -> float:
    """Compute freshness score for an article based on its last update time.

    Freshness is derived from ``compiled_at`` (preferred), ``modified_at``,
    or ``created_at``, in that priority order.  Uses the same exponential
    decay as ``compute_recency_score`` so weights are directly comparable.

    Args:
        article:    Article (or source) dict from the database.
        decay_rate: Exponential decay rate (default 0.01 ≈ 69-day half-life).

    Returns:
        Float in [0, 1]; 1.0 = just updated, approaches 0 for old articles.
    """
    for col in ("compiled_at", "modified_at", "created_at"):
        val = article.get(col)
        if val is None:
            continue
        return compute_recency_score(val, decay_rate)
    return 0.5  # Unknown age


def multi_signal_rank(
    results: list[dict],
    semantic_weight: float = 0.50,
    confidence_weight: float = 0.35,
    recency_weight: float = 0.15,
    decay_rate: float = 0.01,
    min_confidence: float | None = None,
    explain: bool = False,
    query_intent: str | None = None,
    cold_start_boost: bool = True,
) -> list[dict]:
    """Apply multi-signal ranking to query results.

    Formula: final_score = w_semantic * semantic + w_confidence * confidence + w_recency * recency

    The recency signal uses ``compiled_at`` (preferred for articles), then
    ``modified_at``, then ``created_at``, so freshness from source ages /
    article compilation time is properly captured.

    Args:
        results: List of article (or source) dicts with 'similarity' key.
        semantic_weight: Weight for semantic similarity (default 0.50).
        confidence_weight: Weight for confidence score (default 0.35).
        recency_weight: Weight for recency / freshness (default 0.15).
        decay_rate: Exponential decay rate for recency (default 0.01).
        min_confidence: Filter out articles below this confidence (optional).
        explain: Include score breakdown in each result dict.
        query_intent: Detected query intent ("procedural", "episodic", "general", or None).
            When an article's epistemic_type matches, apply 1.3x multiplier.
            When it conflicts (procedural<->episodic), apply 0.85x.
            "general" intent or "semantic" epistemic_type -> no adjustment.
        cold_start_boost: When True, apply a score floor to fresh high-confidence articles
            to prevent cold-start burial (default True).

    Returns:
        Sorted results with 'final_score' and optional 'score_breakdown'.
    """
    # Normalize weights to sum to 1.0
    total_weight = semantic_weight + confidence_weight + recency_weight
    if total_weight > 0:
        semantic_weight /= total_weight
        confidence_weight /= total_weight
        recency_weight /= total_weight

    ranked = []
    for r in results:
        # Semantic score (already computed from embedding similarity or ts_rank)
        semantic = r.get("similarity", 0.0)
        if isinstance(semantic, int | float):
            semantic = min(1.0, max(0.0, float(semantic)))
        else:
            semantic = 0.0

        # Confidence score
        confidence = compute_confidence_score(r)

        # Filter by minimum confidence if specified
        if min_confidence is not None and confidence < min_confidence:
            continue

        # Freshness / recency score — prefer compiled_at (articles) over created_at
        freshness_ts = r.get("compiled_at") or r.get("modified_at") or r.get("created_at")
        recency = compute_recency_score(freshness_ts, decay_rate) if freshness_ts else 0.5

        # Final score
        final_score = semantic_weight * semantic + confidence_weight * confidence + recency_weight * recency

        # --- Epistemic type awareness (#70) ---
        if query_intent and query_intent != "general":
            epistemic_type = r.get("epistemic_type")
            if epistemic_type and epistemic_type != "semantic":
                if epistemic_type == query_intent:
                    final_score *= 1.3
                elif {epistemic_type, query_intent} == {"procedural", "episodic"}:
                    final_score *= 0.85

        # --- Cold-start boost (#71) ---
        if cold_start_boost:
            floor = _cold_start_floor(r, confidence)
            if floor is not None:
                final_score = max(final_score, floor)

        r["final_score"] = final_score

        if explain:
            r["score_breakdown"] = {
                "semantic": {
                    "value": semantic,
                    "weight": semantic_weight,
                    "contribution": semantic_weight * semantic,
                },
                "confidence": {
                    "value": confidence,
                    "weight": confidence_weight,
                    "contribution": confidence_weight * confidence,
                },
                "recency": {
                    "value": recency,
                    "weight": recency_weight,
                    "contribution": recency_weight * recency,
                },
                "final": final_score,
            }

        ranked.append(r)

    ranked.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    return ranked
