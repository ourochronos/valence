# SPDX-License-Identifier: MIT
"""Tests for confidence computation."""

from __future__ import annotations

import math

from valence.core.confidence import compute_confidence


class TestComputeConfidence:
    def test_empty_sources(self):
        r = compute_confidence([])
        assert r.overall == 0.5
        assert r.avg_reliability == 0.5
        assert r.source_bonus == 0.0
        assert r.corroboration_count == 0

    def test_single_source_default_reliability(self):
        r = compute_confidence([{}])
        assert r.overall == 0.5
        assert r.corroboration_count == 1
        assert r.source_bonus == 0.0

    def test_single_source_high_reliability(self):
        r = compute_confidence([{"reliability": 0.9}])
        assert r.overall == 0.9
        assert r.avg_reliability == 0.9
        assert r.source_bonus == 0.0

    def test_two_sources_bonus(self):
        r = compute_confidence([{"reliability": 0.8}, {"reliability": 0.8}])
        assert r.corroboration_count == 2
        assert r.source_bonus > 0
        assert r.overall > 0.8  # bonus applied

    def test_many_sources_capped(self):
        sources = [{"reliability": 0.85}] * 20
        r = compute_confidence(sources)
        assert r.overall <= 0.95
        assert r.source_bonus <= 0.15

    def test_to_jsonb(self):
        r = compute_confidence([{"reliability": 0.75}])
        j = r.to_jsonb()
        assert "overall" in j
        assert isinstance(j["overall"], float)

    def test_matches_sql_formula(self):
        """Verify Python formula matches the SQL in usage.py backfill."""
        # SQL: LEAST(0.95, avg_rel + CASE WHEN n>1 THEN LEAST(0.15, LN(1+n-1)*0.1) ELSE 0 END)
        sources = [{"reliability": 0.7}, {"reliability": 0.8}, {"reliability": 0.6}]
        r = compute_confidence(sources)
        avg = (0.7 + 0.8 + 0.6) / 3
        bonus = min(0.15, math.log(1 + 3 - 1) * 0.1)
        expected = min(0.95, avg + bonus)
        assert abs(r.overall - round(expected, 4)) < 0.001
