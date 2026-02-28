# SPDX-License-Identifier: MIT
"""Tests for temporal_mode weight presets in knowledge_search."""

import pytest

from valence.core.retrieval import TEMPORAL_WEIGHT_PRESETS


def test_all_modes_defined():
    assert set(TEMPORAL_WEIGHT_PRESETS.keys()) == {"default", "prefer_recent", "prefer_stable"}


def test_weights_sum_to_one():
    for mode, weights in TEMPORAL_WEIGHT_PRESETS.items():
        total = weights["semantic"] + weights["confidence"] + weights["recency"]
        assert abs(total - 1.0) < 1e-9, f"Mode {mode!r} weights don't sum to 1.0: {total}"


def test_mode_values():
    assert TEMPORAL_WEIGHT_PRESETS["default"] == {"semantic": 0.50, "confidence": 0.35, "recency": 0.15}
    assert TEMPORAL_WEIGHT_PRESETS["prefer_recent"] == {"semantic": 0.35, "confidence": 0.15, "recency": 0.50}
    assert TEMPORAL_WEIGHT_PRESETS["prefer_stable"] == {"semantic": 0.40, "confidence": 0.45, "recency": 0.15}


def test_prefer_recent_has_highest_recency():
    recencies = {mode: w["recency"] for mode, w in TEMPORAL_WEIGHT_PRESETS.items()}
    assert recencies["prefer_recent"] == max(recencies.values())


def test_prefer_stable_has_highest_confidence():
    confidences = {mode: w["confidence"] for mode, w in TEMPORAL_WEIGHT_PRESETS.items()}
    assert confidences["prefer_stable"] == max(confidences.values())


def test_invalid_mode_falls_back_to_default():
    """_retrieve_sync should fall back to default for unknown modes."""
    invalid_weights = TEMPORAL_WEIGHT_PRESETS.get("nonexistent_mode", TEMPORAL_WEIGHT_PRESETS["default"])
    assert invalid_weights == TEMPORAL_WEIGHT_PRESETS["default"]


def test_modes_produce_different_weights():
    """The three modes must have distinct weight tuples."""
    tuples = [tuple(sorted(w.items())) for w in TEMPORAL_WEIGHT_PRESETS.values()]
    assert len(set(tuples)) == 3, "All three modes must have distinct weight configurations"
