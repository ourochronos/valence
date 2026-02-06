"""Tests for multi-dimensional epistemic trust relationships (Issue #268).

Tests cover:
- EpistemicTrustDimension enum and constants
- compute_epistemic_trust() — geometric and arithmetic mean
- TrustEdge with epistemic dimensions
- TrustEdge.with_epistemic() factory
- Serialization/deserialization with dimensions
- TrustService epistemic methods
- Delegation and transitive trust with epistemic dimensions
- Decay applied to epistemic dimensions
"""

from __future__ import annotations

import math

import pytest

from valence.privacy.trust import (
    DEFAULT_EPISTEMIC_WEIGHTS,
    EPISTEMIC_DIMENSIONS,
    TRUST_SCHEMA_CORE,
    TRUST_SCHEMA_EPISTEMIC,
    EpistemicTrustDimension,
    TrustEdge,
    TrustService,
    compute_epistemic_trust,
)

# ============================================================================
# Constants and Enum
# ============================================================================


class TestEpistemicTrustDimension:
    """Tests for the EpistemicTrustDimension enum."""

    def test_all_dimensions_present(self):
        assert len(EpistemicTrustDimension) == 6
        assert EpistemicTrustDimension.CONCLUSIONS == "conclusions"
        assert EpistemicTrustDimension.REASONING == "reasoning"
        assert EpistemicTrustDimension.PERSPECTIVE == "perspective"
        assert EpistemicTrustDimension.HONESTY == "honesty"
        assert EpistemicTrustDimension.METHODOLOGY == "methodology"
        assert EpistemicTrustDimension.PREDICTIVE == "predictive"

    def test_epistemic_dimensions_list(self):
        assert len(EPISTEMIC_DIMENSIONS) == 6
        assert "conclusions" in EPISTEMIC_DIMENSIONS
        assert "reasoning" in EPISTEMIC_DIMENSIONS

    def test_default_weights_sum_to_one(self):
        total = sum(DEFAULT_EPISTEMIC_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_default_weights_cover_all_dimensions(self):
        for dim in EpistemicTrustDimension:
            assert dim in DEFAULT_EPISTEMIC_WEIGHTS


# ============================================================================
# compute_epistemic_trust()
# ============================================================================


class TestComputeEpistemicTrust:
    """Tests for the standalone compute_epistemic_trust function."""

    def test_empty_returns_default(self):
        assert compute_epistemic_trust({}) == 0.5

    def test_uniform_high(self):
        dims = {d: 0.9 for d in EPISTEMIC_DIMENSIONS}
        result = compute_epistemic_trust(dims)
        assert 0.85 < result <= 0.95

    def test_uniform_low(self):
        dims = {d: 0.1 for d in EPISTEMIC_DIMENSIONS}
        result = compute_epistemic_trust(dims)
        assert 0.05 < result < 0.15

    def test_single_dimension(self):
        result = compute_epistemic_trust({"reasoning": 0.8})
        assert abs(result - 0.8) < 0.01

    def test_geometric_penalizes_imbalance(self):
        """Geometric mean penalizes one low dimension more than arithmetic."""
        balanced = {d: 0.7 for d in EPISTEMIC_DIMENSIONS}
        imbalanced = dict(balanced)
        imbalanced["honesty"] = 0.01  # One very low dimension

        geo_balanced = compute_epistemic_trust(balanced, use_geometric=True)
        geo_imbalanced = compute_epistemic_trust(imbalanced, use_geometric=True)
        arith_imbalanced = compute_epistemic_trust(imbalanced, use_geometric=False)

        # Geometric should penalize imbalance more
        assert geo_imbalanced < arith_imbalanced
        assert geo_balanced > geo_imbalanced

    def test_arithmetic_mode(self):
        dims = {"conclusions": 0.8, "reasoning": 0.2}
        result = compute_epistemic_trust(dims, use_geometric=False)
        # Should be close to weighted average
        assert 0.3 < result < 0.7

    def test_custom_weights(self):
        dims = {"conclusions": 1.0, "reasoning": 0.0}
        # Weight entirely on conclusions
        weights = {"conclusions": 1.0, "reasoning": 0.0}
        result = compute_epistemic_trust(dims, weights=weights, use_geometric=False)
        assert abs(result - 1.0) < 0.01

    def test_result_clamped_zero_one(self):
        dims = {d: 1.0 for d in EPISTEMIC_DIMENSIONS}
        assert compute_epistemic_trust(dims) <= 1.0
        dims = {d: 0.001 for d in EPISTEMIC_DIMENSIONS}
        assert compute_epistemic_trust(dims) >= 0.0

    def test_geometric_with_zero_uses_epsilon(self):
        """Zero values should use EPSILON floor, not produce -inf."""
        dims = {"conclusions": 0.0, "reasoning": 0.8}
        result = compute_epistemic_trust(dims, use_geometric=True)
        assert result > 0.0
        assert not math.isnan(result)
        assert not math.isinf(result)


# ============================================================================
# TrustEdge with epistemic dimensions
# ============================================================================


class TestTrustEdgeEpistemic:
    """Tests for TrustEdge's epistemic dimension support."""

    def test_default_no_dimensions(self):
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            integrity=0.7,
            confidentiality=0.6,
        )
        assert edge.dimensions == {}
        assert edge.schema == TRUST_SCHEMA_CORE
        assert edge.epistemic_trust is None

    def test_with_dimensions(self):
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            integrity=0.7,
            confidentiality=0.6,
            dimensions={"conclusions": 0.9, "reasoning": 0.85},
            schema=TRUST_SCHEMA_EPISTEMIC,
        )
        assert edge.dimensions["conclusions"] == 0.9
        assert edge.schema == TRUST_SCHEMA_EPISTEMIC

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            TrustEdge(
                source_did="did:key:alice",
                target_did="did:key:bob",
                competence=0.5,
                integrity=0.5,
                confidentiality=0.5,
                dimensions={"conclusions": 1.5},
            )

    def test_dimension_validation_negative(self):
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            TrustEdge(
                source_did="did:key:alice",
                target_did="did:key:bob",
                competence=0.5,
                integrity=0.5,
                confidentiality=0.5,
                dimensions={"conclusions": -0.1},
            )

    def test_get_set_dimension(self):
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.5,
            integrity=0.5,
            confidentiality=0.5,
        )
        assert edge.get_dimension("reasoning") is None
        assert not edge.has_dimension("reasoning")

        edge.set_dimension("reasoning", 0.8)
        assert edge.get_dimension("reasoning") == 0.8
        assert edge.has_dimension("reasoning")

        # Remove dimension
        edge.set_dimension("reasoning", None)
        assert not edge.has_dimension("reasoning")

    def test_set_dimension_validation(self):
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.5,
            integrity=0.5,
            confidentiality=0.5,
        )
        with pytest.raises(ValueError):
            edge.set_dimension("reasoning", 2.0)

    def test_epistemic_dimensions_property(self):
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.5,
            integrity=0.5,
            confidentiality=0.5,
            dimensions={
                "conclusions": 0.8,
                "reasoning": 0.7,
                "custom_dim": 0.9,  # Not an epistemic dimension
            },
        )
        eps = edge.epistemic_dimensions
        assert "conclusions" in eps
        assert "reasoning" in eps
        assert "custom_dim" not in eps

    def test_epistemic_trust_property(self):
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.5,
            integrity=0.5,
            confidentiality=0.5,
            dimensions={
                "conclusions": 0.8,
                "reasoning": 0.9,
                "honesty": 0.85,
            },
        )
        eps_trust = edge.epistemic_trust
        assert eps_trust is not None
        assert 0.0 < eps_trust < 1.0

    def test_epistemic_trust_none_without_dims(self):
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.5,
            integrity=0.5,
            confidentiality=0.5,
        )
        assert edge.epistemic_trust is None


# ============================================================================
# TrustEdge.with_epistemic() factory
# ============================================================================


class TestWithEpistemic:
    """Tests for the TrustEdge.with_epistemic() class method."""

    def test_basic_creation(self):
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            reasoning=0.9,
        )
        assert edge.source_did == "did:key:alice"
        assert edge.target_did == "did:key:bob"
        assert edge.dimensions["conclusions"] == 0.8
        assert edge.dimensions["reasoning"] == 0.9
        assert edge.schema == TRUST_SCHEMA_EPISTEMIC

    def test_all_epistemic_dimensions(self):
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            reasoning=0.9,
            perspective=0.7,
            honesty=0.85,
            methodology=0.75,
            predictive=0.6,
        )
        assert len(edge.epistemic_dimensions) == 6

    def test_with_core_overrides(self):
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            competence=0.9,
            integrity=0.95,
        )
        assert edge.competence == 0.9
        assert edge.integrity == 0.95

    def test_schema_core_when_no_dims(self):
        """If no epistemic dims provided, schema stays core."""
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
        )
        assert edge.schema == TRUST_SCHEMA_CORE
        assert edge.dimensions == {}

    def test_with_domain(self):
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            domain="epistemology",
        )
        assert edge.domain == "epistemology"

    def test_none_dims_excluded(self):
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            reasoning=None,
        )
        assert "conclusions" in edge.dimensions
        assert "reasoning" not in edge.dimensions


# ============================================================================
# Serialization roundtrip
# ============================================================================


class TestEpistemicSerialization:
    """Tests for to_dict/from_dict with epistemic dimensions."""

    def test_to_dict_includes_dimensions(self):
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            reasoning=0.9,
        )
        d = edge.to_dict()
        assert "dimensions" in d
        assert d["dimensions"]["conclusions"] == 0.8
        assert d["dimensions"]["reasoning"] == 0.9
        assert "epistemic_trust" in d
        assert d["schema"] == TRUST_SCHEMA_EPISTEMIC

    def test_to_dict_no_dimensions(self):
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.5,
            integrity=0.5,
            confidentiality=0.5,
        )
        d = edge.to_dict()
        # No dimensions or schema keys when using core defaults
        assert "dimensions" not in d
        assert "schema" not in d
        assert "epistemic_trust" not in d

    def test_roundtrip(self):
        original = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            reasoning=0.9,
            honesty=0.7,
            competence=0.85,
            domain="science",
        )
        d = original.to_dict()
        restored = TrustEdge.from_dict(d)

        assert restored.source_did == original.source_did
        assert restored.target_did == original.target_did
        assert restored.competence == original.competence
        assert restored.dimensions == original.dimensions
        assert restored.schema == original.schema
        assert restored.domain == original.domain

    def test_from_dict_missing_dimensions(self):
        """Legacy data without dimensions should work."""
        data = {
            "source_did": "did:key:alice",
            "target_did": "did:key:bob",
            "competence": 0.5,
            "integrity": 0.5,
            "confidentiality": 0.5,
            "judgment": 0.1,
        }
        edge = TrustEdge.from_dict(data)
        assert edge.dimensions == {}
        assert edge.schema == TRUST_SCHEMA_CORE

    def test_from_dict_bad_dimension_values_ignored(self):
        """Non-numeric dimension values should be ignored."""
        data = {
            "source_did": "did:key:alice",
            "target_did": "did:key:bob",
            "competence": 0.5,
            "integrity": 0.5,
            "confidentiality": 0.5,
            "judgment": 0.1,
            "dimensions": {"conclusions": 0.8, "bad_key": "not_a_number"},
        }
        edge = TrustEdge.from_dict(data)
        assert edge.dimensions == {"conclusions": 0.8}


# ============================================================================
# TrustService epistemic methods
# ============================================================================


class TestTrustServiceEpistemic:
    """Tests for TrustService.set_trust_dimensions and related methods."""

    def setup_method(self):
        self.service = TrustService(use_memory=True)

    def test_set_and_get_dimensions(self):
        self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={"conclusions": 0.8, "reasoning": 0.9},
        )

        dims = self.service.get_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
        )
        assert isinstance(dims, dict)
        assert dims["conclusions"] == 0.8
        assert dims["reasoning"] == 0.9

    def test_get_single_dimension(self):
        self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={"conclusions": 0.8, "reasoning": 0.9},
        )

        val = self.service.get_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimension="conclusions",
        )
        assert val == 0.8

    def test_get_missing_dimension(self):
        self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={"conclusions": 0.8},
        )

        val = self.service.get_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimension="honesty",
        )
        assert val is None

    def test_get_dimensions_no_edge(self):
        result = self.service.get_trust_dimensions(
            "did:key:nobody",
            "did:key:nope",
        )
        assert result is None

    def test_merge_dimensions(self):
        """Setting dimensions on existing edge merges them."""
        self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={"conclusions": 0.8},
        )
        self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={"reasoning": 0.9},
        )

        dims = self.service.get_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
        )
        assert isinstance(dims, dict)
        assert dims["conclusions"] == 0.8
        assert dims["reasoning"] == 0.9

    def test_update_existing_dimension(self):
        """Updating a dimension overwrites the old value."""
        self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={"conclusions": 0.8},
        )
        self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={"conclusions": 0.5},
        )

        dims = self.service.get_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
        )
        assert isinstance(dims, dict)
        assert dims["conclusions"] == 0.5

    def test_set_with_core_overrides(self):
        edge = self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={"conclusions": 0.8},
            competence=0.9,
        )
        assert edge.competence == 0.9
        assert edge.dimensions["conclusions"] == 0.8

    def test_set_preserves_core_defaults(self):
        """Core 4D should use defaults if not explicitly set."""
        edge = self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={"conclusions": 0.8},
        )
        assert edge.competence == 0.5
        assert edge.integrity == 0.5
        assert edge.confidentiality == 0.5
        assert edge.judgment == 0.1

    def test_compute_weighted_trust(self):
        self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={
                "conclusions": 0.8,
                "reasoning": 0.9,
                "honesty": 0.85,
            },
        )

        result = self.service.compute_weighted_trust(
            "did:key:alice",
            "did:key:bob",
        )
        assert result is not None
        assert 0.0 < result < 1.0

    def test_compute_weighted_trust_fallback_to_core(self):
        """Without epistemic dims, falls back to core overall_trust."""
        self.service.grant_trust(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            integrity=0.7,
            confidentiality=0.6,
        )

        result = self.service.compute_weighted_trust(
            "did:key:alice",
            "did:key:bob",
        )
        assert result is not None
        # Should be the core overall trust (geometric mean)
        assert 0.0 < result < 1.0

    def test_compute_weighted_trust_no_edge(self):
        result = self.service.compute_weighted_trust(
            "did:key:nobody",
            "did:key:nope",
        )
        assert result is None

    def test_domain_scoped(self):
        """Dimensions should be scoped per domain."""
        self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={"conclusions": 0.8},
            domain="science",
        )
        self.service.set_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimensions={"conclusions": 0.5},
            domain="politics",
        )

        sci = self.service.get_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimension="conclusions",
            domain="science",
        )
        pol = self.service.get_trust_dimensions(
            "did:key:alice",
            "did:key:bob",
            dimension="conclusions",
            domain="politics",
        )
        assert sci == 0.8
        assert pol == 0.5


# ============================================================================
# Decay with epistemic dimensions
# ============================================================================


class TestEpistemicDecay:
    """Tests for trust decay applied to epistemic dimensions."""

    def test_effective_trust_includes_epistemic(self):
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            reasoning=0.9,
        )
        result = edge.effective_trust()
        assert "dimensions" in result
        assert "epistemic_overall" in result
        assert result["dimensions"]["conclusions"] == pytest.approx(0.8, abs=0.01)

    def test_decay_reduces_epistemic_dimensions(self):
        from datetime import UTC, datetime, timedelta

        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            reasoning=0.9,
            decay_rate=0.5,
        )
        # Backdate the last_refreshed
        edge.last_refreshed = datetime.now(UTC) - timedelta(days=30)

        result = edge.effective_trust()
        assert result["dimensions"]["conclusions"] < 0.8
        assert result["dimensions"]["reasoning"] < 0.9


# ============================================================================
# Delegation with epistemic dimensions
# ============================================================================


class TestEpistemicDelegation:
    """Tests for delegated trust preserving epistemic dimensions."""

    def test_delegated_trust_includes_dimensions(self):
        from valence.privacy.trust.computation import compute_delegated_trust

        direct = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            reasoning=0.9,
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            judgment=0.7,
        )
        direct.can_delegate = True
        direct.delegation_depth = 1

        delegated = TrustEdge.with_epistemic(
            "did:key:bob",
            "did:key:carol",
            conclusions=0.7,
            reasoning=0.6,
            competence=0.7,
            integrity=0.7,
            confidentiality=0.7,
            judgment=0.5,
        )

        result = compute_delegated_trust(direct, delegated)
        assert result.source_did == "did:key:alice"
        assert result.target_did == "did:key:carol"
        assert "conclusions" in result.dimensions
        assert "reasoning" in result.dimensions
        # Delegated values should be less than or equal to both inputs
        assert result.dimensions["conclusions"] <= 0.8
        assert result.dimensions["reasoning"] <= 0.9


# ============================================================================
# Refresh trust with epistemic dimensions
# ============================================================================


class TestEpistemicRefresh:
    """Tests for refreshing trust including epistemic dimensions."""

    def test_refresh_with_extended_dimensions(self):
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.5,
            reasoning=0.5,
        )
        edge.refresh_trust(new_values={"conclusions": 0.9, "honesty": 0.8})
        assert edge.dimensions["conclusions"] == 0.9
        assert edge.dimensions["honesty"] == 0.8
        assert edge.dimensions["reasoning"] == 0.5  # unchanged

    def test_refresh_with_core_and_extended(self):
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.5,
        )
        edge.refresh_trust(new_values={"competence": 0.9, "conclusions": 0.8})
        assert edge.competence == 0.9
        assert edge.dimensions["conclusions"] == 0.8


# ============================================================================
# Copy methods preserve dimensions
# ============================================================================


class TestEpistemicCopy:
    """Tests that copy-like methods preserve epistemic dimensions."""

    def test_with_decay_preserves_dimensions(self):
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            reasoning=0.9,
        )
        decayed = edge.with_decay(decay_rate=0.5)
        assert "conclusions" in decayed.dimensions
        assert "reasoning" in decayed.dimensions
        assert decayed.schema == edge.schema

    def test_with_delegation_preserves_dimensions(self):
        edge = TrustEdge.with_epistemic(
            "did:key:alice",
            "did:key:bob",
            conclusions=0.8,
            reasoning=0.9,
        )
        delegated = edge.with_delegation(can_delegate=True, delegation_depth=2)
        assert delegated.dimensions == edge.dimensions
        assert delegated.schema == edge.schema
