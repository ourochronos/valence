"""Tests for Valence CLI.

Tests cover pure utility functions (format_confidence, format_age, ranking).
Command tests have been removed as part of v2 refactor (Phase 1).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from valence.cli.main import (
    compute_confidence_score,
    compute_recency_score,
    format_age,
    format_confidence,
    multi_signal_rank,
)

# ============================================================================
# Unit Tests - Pure Functions
# ============================================================================


class TestFormatConfidence:
    """Test confidence formatting."""

    def test_format_overall(self):
        """Format overall confidence."""
        assert format_confidence({"overall": 0.8}) == "80%"
        assert format_confidence({"overall": 0.95}) == "95%"
        assert format_confidence({"overall": 0.123}) == "12%"

    def test_format_empty(self):
        """Format empty confidence."""
        assert format_confidence({}) == "?"
        assert format_confidence(None) == "?"

    def test_format_non_numeric(self):
        """Format non-numeric overall."""
        # Should truncate to 5 chars
        result = format_confidence({"overall": "high"})
        assert len(result) <= 5


class TestFormatAge:
    """Test age formatting."""

    def test_format_recent(self):
        """Format very recent time."""
        now = datetime.now(UTC)
        assert format_age(now) == "now"
        assert format_age(now - timedelta(seconds=30)) == "now"

    def test_format_minutes(self):
        """Format minutes ago."""
        now = datetime.now(UTC)
        assert format_age(now - timedelta(minutes=5)) == "5m"
        assert format_age(now - timedelta(minutes=59)) == "59m"

    def test_format_hours(self):
        """Format hours ago."""
        now = datetime.now(UTC)
        assert format_age(now - timedelta(hours=3)) == "3h"
        assert format_age(now - timedelta(hours=23)) == "23h"

    def test_format_days(self):
        """Format days ago."""
        now = datetime.now(UTC)
        assert format_age(now - timedelta(days=5)) == "5d"
        assert format_age(now - timedelta(days=29)) == "29d"

    def test_format_months(self):
        """Format months ago."""
        now = datetime.now(UTC)
        assert format_age(now - timedelta(days=45)) == "1mo"
        assert format_age(now - timedelta(days=180)) == "6mo"

    def test_format_years(self):
        """Format years ago."""
        now = datetime.now(UTC)
        assert format_age(now - timedelta(days=400)) == "1y"
        assert format_age(now - timedelta(days=800)) == "2y"

    def test_format_none(self):
        """Format None datetime."""
        assert format_age(None) == "?"

    def test_format_naive_datetime(self):
        """Format naive datetime (no timezone) - gets treated as UTC."""
        now = datetime.now()
        result = format_age(now - timedelta(hours=2))
        assert result != "?"
        assert any(c in result for c in ["h", "m", "d", "y", "now", "mo"])


# ============================================================================
# Multi-Signal Ranking Tests (Issue #13)
# ============================================================================


class TestComputeConfidenceScore:
    """Test confidence score computation."""

    def test_6d_confidence_geometric_mean(self):
        """Compute confidence from 6D vector using geometric mean."""
        belief = {
            "confidence_source": 0.9,
            "confidence_method": 0.8,
            "confidence_consistency": 0.9,
            "confidence_freshness": 1.0,
            "confidence_corroboration": 0.5,
            "confidence_applicability": 0.8,
        }
        score = compute_confidence_score(belief)
        assert 0.7 < score < 0.9

    def test_6d_penalizes_weak_dimension(self):
        """Geometric mean penalizes beliefs with one weak dimension."""
        belief_weak = {
            "confidence_source": 0.9,
            "confidence_method": 0.9,
            "confidence_consistency": 0.9,
            "confidence_freshness": 0.9,
            "confidence_corroboration": 0.1,
            "confidence_applicability": 0.9,
        }
        belief_moderate = {
            "confidence_source": 0.7,
            "confidence_method": 0.7,
            "confidence_consistency": 0.7,
            "confidence_freshness": 0.7,
            "confidence_corroboration": 0.7,
            "confidence_applicability": 0.7,
        }

        score_weak = compute_confidence_score(belief_weak)
        score_moderate = compute_confidence_score(belief_moderate)
        assert score_moderate > score_weak

    def test_fallback_to_jsonb_overall(self):
        """Fall back to JSONB overall when 6D not populated."""
        belief = {"confidence": {"overall": 0.85}}
        score = compute_confidence_score(belief)
        assert score == 0.85

    def test_default_score(self):
        """Return default 0.5 when no confidence data."""
        belief = {}
        score = compute_confidence_score(belief)
        assert score == 0.5


class TestComputeRecencyScore:
    """Test recency score computation."""

    def test_recent_belief_high_score(self):
        """Recent beliefs get high recency score."""
        now = datetime.now(UTC)
        score = compute_recency_score(now)
        assert score > 0.99

    def test_old_belief_decays(self):
        """Old beliefs decay over time."""
        now = datetime.now(UTC)
        one_week_ago = now - timedelta(days=7)
        one_month_ago = now - timedelta(days=30)
        one_year_ago = now - timedelta(days=365)

        score_week = compute_recency_score(one_week_ago)
        score_month = compute_recency_score(one_month_ago)
        score_year = compute_recency_score(one_year_ago)

        assert score_week > score_month > score_year
        assert score_week > 0.9
        assert 0.6 < score_month < 0.8
        assert score_year < 0.05

    def test_custom_decay_rate(self):
        """Custom decay rate adjusts half-life."""
        now = datetime.now(UTC)
        one_week_ago = now - timedelta(days=7)

        score_high = compute_recency_score(one_week_ago, decay_rate=0.10)
        score_low = compute_recency_score(one_week_ago, decay_rate=0.002)

        assert score_low > score_high
        assert score_high < 0.6
        assert score_low > 0.98

    def test_none_returns_default(self):
        """None datetime returns default score."""
        score = compute_recency_score(None)
        assert score == 0.5

    def test_naive_datetime_handled(self):
        """Naive datetime (no timezone) is handled correctly."""
        naive_dt = datetime.now()
        score = compute_recency_score(naive_dt)
        assert 0.9 < score <= 1.0


class TestMultiSignalRank:
    """Test multi-signal ranking algorithm."""

    def test_default_weights(self):
        """Default weights: semantic=0.50, confidence=0.35, recency=0.15."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.9,
                "confidence": {"overall": 0.5},
                "created_at": now - timedelta(days=30),
            },
            {
                "id": uuid4(),
                "similarity": 0.7,
                "confidence": {"overall": 0.95},
                "created_at": now,
            },
        ]

        ranked = multi_signal_rank(results)

        assert all("final_score" in r for r in ranked)
        assert ranked[0]["similarity"] == 0.7

    def test_high_recency_weight(self):
        """High recency weight prefers newer beliefs."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.95,
                "confidence": {"overall": 0.9},
                "created_at": now - timedelta(days=60),
            },
            {
                "id": uuid4(),
                "similarity": 0.7,
                "confidence": {"overall": 0.7},
                "created_at": now,
            },
        ]

        ranked = multi_signal_rank(results, recency_weight=0.5)
        assert ranked[0]["created_at"] == now

    def test_min_confidence_filter(self):
        """Filter beliefs below minimum confidence."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.9,
                "confidence": {"overall": 0.3},
                "created_at": now,
            },
            {
                "id": uuid4(),
                "similarity": 0.8,
                "confidence": {"overall": 0.8},
                "created_at": now,
            },
        ]

        ranked = multi_signal_rank(results, min_confidence=0.5)
        assert len(ranked) == 1
        assert ranked[0]["confidence"]["overall"] == 0.8

    def test_explain_mode(self):
        """Explain mode includes score breakdown."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.85,
                "confidence": {"overall": 0.75},
                "created_at": now,
            },
        ]

        ranked = multi_signal_rank(results, explain=True)

        assert len(ranked) == 1
        assert "score_breakdown" in ranked[0]

        bd = ranked[0]["score_breakdown"]
        assert "semantic" in bd
        assert "confidence" in bd
        assert "recency" in bd
        assert "final" in bd

        assert "value" in bd["semantic"]
        assert "weight" in bd["semantic"]
        assert "contribution" in bd["semantic"]

        total = bd["semantic"]["contribution"] + bd["confidence"]["contribution"] + bd["recency"]["contribution"]
        assert abs(total - bd["final"]) < 0.001

    def test_weight_normalization(self):
        """Weights are normalized to sum to 1.0."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.8,
                "confidence": {"overall": 0.8},
                "created_at": now,
            },
        ]

        ranked = multi_signal_rank(
            results,
            semantic_weight=1.0,
            confidence_weight=1.0,
            recency_weight=1.0,
            explain=True,
        )

        bd = ranked[0]["score_breakdown"]
        assert abs(bd["semantic"]["weight"] - 0.333) < 0.01
        assert abs(bd["confidence"]["weight"] - 0.333) < 0.01
        assert abs(bd["recency"]["weight"] - 0.333) < 0.01

    def test_empty_results(self):
        """Handle empty results gracefully."""
        ranked = multi_signal_rank([])
        assert ranked == []

    def test_results_sorted_by_final_score(self):
        """Results are sorted by final_score descending."""
        now = datetime.now(UTC)

        results = [
            {"id": uuid4(), "similarity": 0.5, "confidence": {"overall": 0.5}, "created_at": now},
            {"id": uuid4(), "similarity": 0.9, "confidence": {"overall": 0.9}, "created_at": now},
            {"id": uuid4(), "similarity": 0.7, "confidence": {"overall": 0.7}, "created_at": now},
        ]

        ranked = multi_signal_rank(results)
        scores = [r["final_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)
