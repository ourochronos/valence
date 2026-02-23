"""Tests for core.ranking module."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from valence.core.ranking import (
    RankingConfig,
    compute_confidence_score,
    compute_recency_score,
    multi_signal_rank,
)


class TestRankingConfig:
    """Tests for RankingConfig dataclass."""

    def test_defaults(self):
        config = RankingConfig()
        assert config.semantic_weight == 0.50
        assert config.confidence_weight == 0.35
        assert config.recency_weight == 0.15
        assert config.decay_rate == 0.01

    def test_normalized_sums_to_one(self):
        config = RankingConfig(semantic_weight=1.0, confidence_weight=1.0, recency_weight=1.0)
        normed = config.normalized()
        total = normed.semantic_weight + normed.confidence_weight + normed.recency_weight
        assert abs(total - 1.0) < 1e-10

    def test_normalized_preserves_ratios(self):
        config = RankingConfig(semantic_weight=0.6, confidence_weight=0.3, recency_weight=0.1)
        normed = config.normalized()
        assert abs(normed.semantic_weight - 0.6) < 1e-10
        assert abs(normed.confidence_weight - 0.3) < 1e-10
        assert abs(normed.recency_weight - 0.1) < 1e-10

    def test_normalized_zero_weights_returns_defaults(self):
        config = RankingConfig(semantic_weight=0, confidence_weight=0, recency_weight=0)
        normed = config.normalized()
        assert normed.semantic_weight == 0.50


class TestComputeConfidenceScore:
    """Tests for compute_confidence_score."""

    def test_6d_confidence(self):
        """Should use 6D confidence when columns are populated."""
        belief = {
            "confidence_source": 0.8,
            "confidence_method": 0.7,
            "confidence_consistency": 0.9,
            "confidence_freshness": 0.85,
            "confidence_corroboration": 0.6,
            "confidence_applicability": 0.75,
        }
        score = compute_confidence_score(belief)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # High confidence values should produce high score

    def test_jsonb_fallback(self):
        """Should fall back to JSONB 'overall' when no 6D columns."""
        belief = {"confidence": {"overall": 0.85}}
        score = compute_confidence_score(belief)
        assert abs(score - 0.85) < 1e-10

    def test_missing_data_returns_default(self):
        """Should return 0.5 when no confidence data available."""
        score = compute_confidence_score({})
        assert abs(score - 0.5) < 1e-10

    def test_clamped_to_zero_one(self):
        """Score should always be in [0, 1]."""
        belief = {"confidence": {"overall": 1.5}}
        assert compute_confidence_score(belief) == 1.0

        belief = {"confidence": {"overall": -0.5}}
        assert compute_confidence_score(belief) == 0.0

    def test_non_numeric_overall(self):
        """Should handle non-numeric overall gracefully."""
        belief = {"confidence": {"overall": "high"}}
        score = compute_confidence_score(belief)
        assert abs(score - 0.5) < 1e-10


class TestComputeRecencyScore:
    """Tests for compute_recency_score."""

    def test_recent_scores_high(self):
        """A just-created belief should score near 1.0."""
        now = datetime.now(UTC)
        score = compute_recency_score(now)
        assert score > 0.99

    def test_old_scores_low(self):
        """A very old belief should score near 0."""
        old = datetime.now(UTC) - timedelta(days=1000)
        score = compute_recency_score(old)
        assert score < 0.01

    def test_half_life(self):
        """Score at ~69 days should be approximately 0.5 with default decay."""
        half_life = math.log(2) / 0.01  # ~69.3 days
        dt = datetime.now(UTC) - timedelta(days=half_life)
        score = compute_recency_score(dt)
        assert abs(score - 0.5) < 0.02

    def test_none_returns_default(self):
        """None created_at should return 0.5."""
        score = compute_recency_score(None)
        assert abs(score - 0.5) < 1e-10

    def test_naive_datetime_handled(self):
        """Should handle timezone-naive datetimes."""
        dt = datetime.now() - timedelta(days=10)
        score = compute_recency_score(dt)
        assert 0.0 <= score <= 1.0

    def test_custom_decay_rate(self):
        """Custom decay rate should change the half-life."""
        dt = datetime.now(UTC) - timedelta(days=10)
        fast = compute_recency_score(dt, decay_rate=0.1)
        slow = compute_recency_score(dt, decay_rate=0.001)
        assert fast < slow  # Faster decay = lower score at same age


class TestMultiSignalRank:
    """Tests for multi_signal_rank."""

    def _make_belief(self, similarity=0.5, confidence_overall=0.7, days_ago=0):
        return {
            "similarity": similarity,
            "confidence": {"overall": confidence_overall},
            "created_at": datetime.now(UTC) - timedelta(days=days_ago),
        }

    def test_sort_by_final_score(self):
        """Results should be sorted by final_score descending."""
        results = [
            self._make_belief(similarity=0.3, confidence_overall=0.3, days_ago=100),
            self._make_belief(similarity=0.9, confidence_overall=0.9, days_ago=1),
            self._make_belief(similarity=0.5, confidence_overall=0.5, days_ago=30),
        ]
        ranked = multi_signal_rank(results)
        scores = [r["final_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_weights_affect_ordering(self):
        """Different weights should produce different orderings."""
        high_sim = self._make_belief(similarity=0.9, confidence_overall=0.3, days_ago=50)
        high_conf = self._make_belief(similarity=0.3, confidence_overall=0.9, days_ago=50)

        # Heavy semantic weight
        sem_ranked = multi_signal_rank(
            [high_sim.copy(), high_conf.copy()],
            semantic_weight=0.9,
            confidence_weight=0.05,
            recency_weight=0.05,
        )
        assert sem_ranked[0]["similarity"] == 0.9

        # Heavy confidence weight
        conf_ranked = multi_signal_rank(
            [high_sim.copy(), high_conf.copy()],
            semantic_weight=0.05,
            confidence_weight=0.9,
            recency_weight=0.05,
        )
        assert conf_ranked[0]["confidence"]["overall"] == 0.9

    def test_min_confidence_filter(self):
        """min_confidence should filter low-confidence beliefs."""
        results = [
            self._make_belief(confidence_overall=0.8),
            self._make_belief(confidence_overall=0.3),
            self._make_belief(confidence_overall=0.6),
        ]
        ranked = multi_signal_rank(results, min_confidence=0.5)
        assert len(ranked) == 2
        for r in ranked:
            assert r["confidence"]["overall"] >= 0.5

    def test_explain_mode(self):
        """explain=True should include score_breakdown."""
        results = [self._make_belief()]
        ranked = multi_signal_rank(results, explain=True)
        assert "score_breakdown" in ranked[0]
        breakdown = ranked[0]["score_breakdown"]
        assert "semantic" in breakdown
        assert "confidence" in breakdown
        assert "recency" in breakdown
        assert "final" in breakdown
        # Contributions should sum to final
        total = sum(breakdown[k]["contribution"] for k in ("semantic", "confidence", "recency"))
        assert abs(total - breakdown["final"]) < 1e-10

    def test_explain_false_no_breakdown(self):
        """explain=False should not include score_breakdown."""
        results = [self._make_belief()]
        ranked = multi_signal_rank(results, explain=False)
        assert "score_breakdown" not in ranked[0]

    def test_normalization(self):
        """Weights should auto-normalize to sum to 1.0."""
        results = [self._make_belief(similarity=1.0, confidence_overall=1.0, days_ago=0)]
        ranked = multi_signal_rank(results, semantic_weight=2.0, confidence_weight=2.0, recency_weight=1.0)
        # With perfect scores and any normalized weights, final should be close to 1.0
        assert ranked[0]["final_score"] > 0.95

    def test_empty_results(self):
        """Should handle empty input gracefully."""
        ranked = multi_signal_rank([])
        assert ranked == []

    def test_missing_similarity_defaults_zero(self):
        """Beliefs without 'similarity' key should get 0.0 semantic score."""
        results = [{"confidence": {"overall": 0.8}, "created_at": datetime.now(UTC)}]
        ranked = multi_signal_rank(results, explain=True)
        assert ranked[0]["score_breakdown"]["semantic"]["value"] == 0.0
