"""Utility functions for Valence CLI."""

from __future__ import annotations

import logging
import math
import sys
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection using config."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    from ..core.config import get_config

    config = get_config()
    return psycopg2.connect(
        host=config.db_host,
        port=config.db_port,
        dbname=config.db_name,
        user=config.db_user,
        password=config.db_password,
        cursor_factory=RealDictCursor,
    )


def get_embedding(text: str) -> list[float] | None:
    """Generate embedding using configured provider (local or OpenAI)."""
    try:
        from oro_embeddings.service import generate_embedding

        return generate_embedding(text)
    except Exception as e:
        print(f"⚠️  Embedding failed: {e}", file=sys.stderr)
        return None


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
# ============================================================================


def compute_confidence_score(belief: dict) -> float:
    """
    Compute aggregated confidence score from 6D confidence vector.

    Uses geometric mean to penalize beliefs with any weak dimension.
    Falls back to JSONB 'overall' field for backward compatibility.
    """
    # Try 6D confidence columns first
    src = belief.get("confidence_source", 0.5)
    meth = belief.get("confidence_method", 0.5)
    cons = belief.get("confidence_consistency", 1.0)
    fresh = belief.get("confidence_freshness", 1.0)
    corr = belief.get("confidence_corroboration", 0.1)
    app = belief.get("confidence_applicability", 0.8)

    # Check if 6D columns are populated (not default placeholder)
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

    return 0.5  # Default


def compute_recency_score(created_at: datetime, decay_rate: float = 0.01) -> float:
    """
    Compute recency score with exponential decay.

    Default decay_rate=0.01 gives a half-life of ~69 days.
    """
    if not created_at:
        return 0.5

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)

    now = datetime.now(UTC)
    age_days = (now - created_at).total_seconds() / 86400

    # Exponential decay: e^(-λ × age)
    recency = math.exp(-decay_rate * age_days)
    return min(1.0, max(0.0, recency))


def multi_signal_rank(
    results: list[dict],
    semantic_weight: float = 0.50,
    confidence_weight: float = 0.35,
    recency_weight: float = 0.15,
    min_confidence: float | None = None,
    explain: bool = False,
) -> list[dict]:
    """
    Apply multi-signal ranking to query results.

    Formula: final_score = w_semantic × semantic + w_confidence × confidence + w_recency × recency

    Args:
        results: List of belief dicts with 'similarity' (semantic score)
        semantic_weight: Weight for semantic similarity (default 0.50)
        confidence_weight: Weight for confidence score (default 0.35)
        recency_weight: Weight for recency score (default 0.15)
        min_confidence: Filter out beliefs below this confidence (optional)
        explain: Include score breakdown in results

    Returns:
        Sorted results with 'final_score' and optional 'score_breakdown'
    """
    # Normalize weights to sum to 1.0
    total_weight = semantic_weight + confidence_weight + recency_weight
    if total_weight > 0:
        semantic_weight /= total_weight
        confidence_weight /= total_weight
        recency_weight /= total_weight

    ranked = []
    for r in results:
        # Semantic score (already computed from embedding similarity)
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

        # Recency score
        created_at = r.get("created_at")
        recency = compute_recency_score(created_at) if created_at else 0.5

        # Final score
        final_score = semantic_weight * semantic + confidence_weight * confidence + recency_weight * recency

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

    # Sort by final score descending
    ranked.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    return ranked
