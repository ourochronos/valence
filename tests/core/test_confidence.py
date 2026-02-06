"""Tests for valence.core.confidence module."""

from __future__ import annotations

import pytest
from valence.core.confidence import (
    DEFAULT_WEIGHTS,
    ConfidenceDimension,
    DimensionalConfidence,
    aggregate_confidence,
    confidence_label,
)

# ============================================================================
# ConfidenceDimension Tests
# ============================================================================


class TestConfidenceDimension:
    """Tests for ConfidenceDimension enum."""

    def test_all_dimensions_exist(self):
        """Verify all expected dimensions are defined."""
        expected = {
            "OVERALL",
            "SOURCE_RELIABILITY",
            "METHOD_QUALITY",
            "INTERNAL_CONSISTENCY",
            "TEMPORAL_FRESHNESS",
            "CORROBORATION",
            "DOMAIN_APPLICABILITY",
        }
        actual = {d.name for d in ConfidenceDimension}
        assert actual == expected

    def test_dimension_values_are_lowercase(self):
        """All dimension values should be lowercase."""
        for dim in ConfidenceDimension:
            assert dim.value == dim.value.lower()

    def test_overall_dimension_value(self):
        """OVERALL dimension should have 'overall' value."""
        assert ConfidenceDimension.OVERALL.value == "overall"

    def test_dimension_is_string_enum(self):
        """ConfidenceDimension should behave as string."""
        dim = ConfidenceDimension.SOURCE_RELIABILITY
        # As a StrEnum, the value can be accessed via .value
        assert dim.value == "source_reliability"
        # Comparison with string works due to StrEnum
        assert dim == "source_reliability"


# ============================================================================
# DEFAULT_WEIGHTS Tests
# ============================================================================


class TestDefaultWeights:
    """Tests for DEFAULT_WEIGHTS configuration."""

    def test_weights_sum_to_one(self):
        """All weights should sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_all_non_overall_dimensions_have_weights(self):
        """Every dimension except OVERALL should have a weight."""
        for dim in ConfidenceDimension:
            if dim != ConfidenceDimension.OVERALL:
                assert dim in DEFAULT_WEIGHTS

    def test_overall_not_in_weights(self):
        """OVERALL should not have a weight (it's computed)."""
        assert ConfidenceDimension.OVERALL not in DEFAULT_WEIGHTS

    def test_all_weights_are_positive(self):
        """All weights should be positive."""
        for weight in DEFAULT_WEIGHTS.values():
            assert weight > 0


# ============================================================================
# DimensionalConfidence Tests
# ============================================================================


class TestDimensionalConfidence:
    """Tests for DimensionalConfidence dataclass."""

    def test_default_values(self):
        """Test default confidence values."""
        conf = DimensionalConfidence()
        assert conf.overall == 0.7
        assert conf.source_reliability is None
        assert conf.method_quality is None
        assert conf.internal_consistency is None
        assert conf.temporal_freshness is None
        assert conf.corroboration is None
        assert conf.domain_applicability is None

    def test_create_with_overall(self):
        """Test creating with just overall score."""
        conf = DimensionalConfidence(overall=0.9)
        assert conf.overall == 0.9

    def test_create_with_all_dimensions(self):
        """Test creating with all dimensions."""
        conf = DimensionalConfidence(
            overall=0.8,
            source_reliability=0.9,
            method_quality=0.7,
            internal_consistency=0.85,
            temporal_freshness=0.6,
            corroboration=0.75,
            domain_applicability=0.8,
        )
        assert conf.overall == 0.8
        assert conf.source_reliability == 0.9
        assert conf.method_quality == 0.7
        assert conf.internal_consistency == 0.85
        assert conf.temporal_freshness == 0.6
        assert conf.corroboration == 0.75
        assert conf.domain_applicability == 0.8

    def test_validation_rejects_negative_overall(self):
        """Negative values should raise ValueError."""
        with pytest.raises(ValueError, match="overall must be between 0 and 1"):
            DimensionalConfidence(overall=-0.1)

    def test_validation_rejects_overall_above_one(self):
        """Values above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="overall must be between 0 and 1"):
            DimensionalConfidence(overall=1.5)

    def test_validation_rejects_negative_dimension(self):
        """Negative dimension values should raise ValueError."""
        with pytest.raises(ValueError, match="source_reliability must be between 0 and 1"):
            DimensionalConfidence(source_reliability=-0.5)

    def test_validation_rejects_dimension_above_one(self):
        """Dimension values above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="corroboration must be between 0 and 1"):
            DimensionalConfidence(corroboration=2.0)

    def test_validation_allows_boundary_values(self):
        """Boundary values 0 and 1 should be valid."""
        conf_zero = DimensionalConfidence(overall=0.0, source_reliability=0.0)
        assert conf_zero.overall == 0.0
        assert conf_zero.source_reliability == 0.0

        conf_one = DimensionalConfidence(overall=1.0, source_reliability=1.0)
        assert conf_one.overall == 1.0
        assert conf_one.source_reliability == 1.0


class TestDimensionalConfidenceSimple:
    """Tests for DimensionalConfidence.simple() factory."""

    def test_simple_creates_overall_only(self):
        """simple() should create confidence with just overall."""
        conf = DimensionalConfidence.simple(0.85)
        assert conf.overall == 0.85
        assert conf.source_reliability is None
        assert conf.method_quality is None

    def test_simple_with_zero(self):
        """simple() should work with zero."""
        conf = DimensionalConfidence.simple(0.0)
        assert conf.overall == 0.0

    def test_simple_with_one(self):
        """simple() should work with one."""
        conf = DimensionalConfidence.simple(1.0)
        assert conf.overall == 1.0


class TestDimensionalConfidenceFull:
    """Tests for DimensionalConfidence.full() factory."""

    def test_full_calculates_overall(self):
        """full() should calculate overall from dimensions."""
        conf = DimensionalConfidence.full(
            source_reliability=1.0,
            method_quality=1.0,
            internal_consistency=1.0,
            temporal_freshness=1.0,
            corroboration=1.0,
            domain_applicability=1.0,
        )
        # All dimensions at 1.0 should give overall of 1.0
        assert abs(conf.overall - 1.0) < 0.001

    def test_full_with_all_zeros(self):
        """full() with all zeros should give overall near epsilon (floor)."""
        conf = DimensionalConfidence.full(
            source_reliability=0.0,
            method_quality=0.0,
            internal_consistency=0.0,
            temporal_freshness=0.0,
            corroboration=0.0,
            domain_applicability=0.0,
        )
        # Geometric mean uses EPSILON (0.001) as floor to prevent log(0)
        assert conf.overall <= 0.001 + 1e-9

    def test_full_with_mixed_values(self):
        """full() should weight dimensions correctly using geometric mean."""
        conf = DimensionalConfidence.full(
            source_reliability=0.8,
            method_quality=0.6,
            internal_consistency=0.9,
            temporal_freshness=0.7,
            corroboration=0.5,
            domain_applicability=0.4,
        )
        # Geometric mean: exp(sum(w_i * ln(v_i)))
        # With default weights (0.25, 0.15, 0.20, 0.15, 0.15, 0.10)
        # = exp(0.25*ln(0.8) + 0.15*ln(0.6) + 0.20*ln(0.9) + 0.15*ln(0.7) + 0.15*ln(0.5) + 0.10*ln(0.4))
        # ≈ 0.669
        assert abs(conf.overall - 0.669) < 0.01

    def test_full_with_custom_weights(self):
        """full() should accept custom weights."""
        custom_weights = {
            ConfidenceDimension.SOURCE_RELIABILITY: 0.5,
            ConfidenceDimension.METHOD_QUALITY: 0.5,
            ConfidenceDimension.INTERNAL_CONSISTENCY: 0.0,
            ConfidenceDimension.TEMPORAL_FRESHNESS: 0.0,
            ConfidenceDimension.CORROBORATION: 0.0,
            ConfidenceDimension.DOMAIN_APPLICABILITY: 0.0,
        }
        conf = DimensionalConfidence.full(
            source_reliability=0.8,
            method_quality=0.4,
            internal_consistency=1.0,  # Should be ignored (weight=0)
            temporal_freshness=1.0,  # Should be ignored (weight=0)
            corroboration=1.0,  # Should be ignored (weight=0)
            domain_applicability=1.0,  # Should be ignored (weight=0)
            weights=custom_weights,
        )
        # Geometric mean: exp(0.5*ln(0.8) + 0.5*ln(0.4)) ≈ 0.566
        assert abs(conf.overall - 0.566) < 0.01


class TestDimensionalConfidenceRecalculate:
    """Tests for DimensionalConfidence.recalculate_overall()."""

    def test_recalculate_with_all_dimensions(self):
        """recalculate_overall should work when all dimensions are set."""
        conf = DimensionalConfidence(
            overall=0.5,  # Will be overwritten
            source_reliability=0.8,
            method_quality=0.6,
            internal_consistency=0.9,
            temporal_freshness=0.7,
            corroboration=0.5,
            domain_applicability=0.4,
        )
        result = conf.recalculate_overall()
        assert result is conf  # Returns self
        # Geometric mean ≈ 0.669 (same as test_full_with_mixed_values)
        assert abs(conf.overall - 0.669) < 0.01

    def test_recalculate_with_partial_dimensions(self):
        """recalculate_overall should handle partial dimensions."""
        conf = DimensionalConfidence(
            overall=0.5,
            source_reliability=0.8,
            method_quality=0.6,
            # Others are None
        )
        result = conf.recalculate_overall()
        assert result is conf
        # Should only use available dimensions
        assert conf.overall > 0

    def test_recalculate_with_no_dimensions(self):
        """recalculate_overall with no dimensions should keep original."""
        conf = DimensionalConfidence(overall=0.5)
        original = conf.overall
        conf.recalculate_overall()
        assert conf.overall == original


class TestDimensionalConfidenceWithDimension:
    """Tests for DimensionalConfidence.with_dimension()."""

    def test_with_dimension_returns_new_instance(self):
        """with_dimension should return a new instance."""
        original = DimensionalConfidence(overall=0.7)
        new = original.with_dimension(ConfidenceDimension.SOURCE_RELIABILITY, 0.9)
        assert new is not original
        assert original.source_reliability is None
        assert new.source_reliability == 0.9

    def test_with_dimension_preserves_other_values(self):
        """with_dimension should preserve other dimension values."""
        original = DimensionalConfidence(
            overall=0.7,
            source_reliability=0.8,
            method_quality=0.6,
        )
        new = original.with_dimension(ConfidenceDimension.CORROBORATION, 0.75)
        assert new.source_reliability == 0.8
        assert new.method_quality == 0.6
        assert new.corroboration == 0.75

    def test_with_dimension_recalculates_by_default(self):
        """with_dimension should recalculate overall by default."""
        original = DimensionalConfidence(overall=0.7, source_reliability=0.5)
        new = original.with_dimension(ConfidenceDimension.SOURCE_RELIABILITY, 0.9)
        assert new.overall != original.overall

    def test_with_dimension_skip_recalculate(self):
        """with_dimension can skip recalculation."""
        original = DimensionalConfidence(overall=0.7, source_reliability=0.5)
        new = original.with_dimension(ConfidenceDimension.SOURCE_RELIABILITY, 0.9, recalculate=False)
        assert new.overall == original.overall

    def test_with_dimension_overall_no_recalculate(self):
        """Setting OVERALL should not trigger recalculation."""
        original = DimensionalConfidence(overall=0.7)
        new = original.with_dimension(ConfidenceDimension.OVERALL, 0.9)
        assert new.overall == 0.9


class TestDimensionalConfidenceDecay:
    """Tests for DimensionalConfidence.decay()."""

    def test_decay_reduces_temporal_freshness(self):
        """decay() should reduce temporal_freshness."""
        conf = DimensionalConfidence(overall=0.8, temporal_freshness=0.9)
        decayed = conf.decay()
        assert decayed.temporal_freshness < 0.9
        assert decayed.temporal_freshness == pytest.approx(0.9 * 0.95)

    def test_decay_with_custom_factor(self):
        """decay() should accept custom factor."""
        conf = DimensionalConfidence(overall=0.8, temporal_freshness=1.0)
        decayed = conf.decay(factor=0.5)
        assert decayed.temporal_freshness == 0.5

    def test_decay_without_temporal_freshness(self):
        """decay() without temporal_freshness should decay overall."""
        conf = DimensionalConfidence(overall=0.8)
        decayed = conf.decay()
        assert decayed.overall < 0.8
        assert decayed.overall == pytest.approx(0.8 * 0.95)

    def test_decay_returns_new_instance(self):
        """decay() should return a new instance."""
        original = DimensionalConfidence(overall=0.8, temporal_freshness=0.9)
        decayed = original.decay()
        assert decayed is not original
        assert original.temporal_freshness == 0.9  # Unchanged


class TestDimensionalConfidenceBoostCorroboration:
    """Tests for DimensionalConfidence.boost_corroboration()."""

    def test_boost_increases_corroboration(self):
        """boost_corroboration() should increase corroboration."""
        conf = DimensionalConfidence(overall=0.7, corroboration=0.5)
        boosted = conf.boost_corroboration()
        assert boosted.corroboration > 0.5
        assert boosted.corroboration == 0.6

    def test_boost_with_custom_amount(self):
        """boost_corroboration() should accept custom amount."""
        conf = DimensionalConfidence(overall=0.7, corroboration=0.5)
        boosted = conf.boost_corroboration(amount=0.2)
        assert boosted.corroboration == 0.7

    def test_boost_caps_at_one(self):
        """boost_corroboration() should cap at 1.0."""
        conf = DimensionalConfidence(overall=0.7, corroboration=0.95)
        boosted = conf.boost_corroboration(amount=0.2)
        assert boosted.corroboration == 1.0

    def test_boost_with_none_corroboration(self):
        """boost_corroboration() with None should start at 0.5."""
        conf = DimensionalConfidence(overall=0.7)
        assert conf.corroboration is None
        boosted = conf.boost_corroboration()
        assert boosted.corroboration == 0.6  # 0.5 + 0.1

    def test_boost_returns_new_instance(self):
        """boost_corroboration() should return a new instance."""
        original = DimensionalConfidence(overall=0.7, corroboration=0.5)
        boosted = original.boost_corroboration()
        assert boosted is not original


class TestDimensionalConfidenceToDict:
    """Tests for DimensionalConfidence.to_dict()."""

    def test_to_dict_includes_overall(self):
        """to_dict() should always include overall."""
        conf = DimensionalConfidence(overall=0.8)
        d = conf.to_dict()
        assert "overall" in d
        assert d["overall"] == 0.8

    def test_to_dict_includes_set_dimensions(self):
        """to_dict() should include dimensions that are set."""
        conf = DimensionalConfidence(
            overall=0.8,
            source_reliability=0.9,
            corroboration=0.7,
        )
        d = conf.to_dict()
        assert d["source_reliability"] == 0.9
        assert d["corroboration"] == 0.7

    def test_to_dict_excludes_none_dimensions(self):
        """to_dict() should exclude dimensions that are None."""
        conf = DimensionalConfidence(overall=0.8, source_reliability=0.9)
        d = conf.to_dict()
        assert "method_quality" not in d
        assert "temporal_freshness" not in d


class TestDimensionalConfidenceFromDict:
    """Tests for DimensionalConfidence.from_dict()."""

    def test_from_dict_with_overall_only(self):
        """from_dict() with just overall."""
        conf = DimensionalConfidence.from_dict({"overall": 0.8})
        assert conf.overall == 0.8
        assert conf.source_reliability is None

    def test_from_dict_with_all_dimensions(self):
        """from_dict() with all dimensions."""
        data = {
            "overall": 0.8,
            "source_reliability": 0.9,
            "method_quality": 0.7,
            "internal_consistency": 0.85,
            "temporal_freshness": 0.6,
            "corroboration": 0.75,
            "domain_applicability": 0.8,
        }
        conf = DimensionalConfidence.from_dict(data)
        assert conf.overall == 0.8
        assert conf.source_reliability == 0.9
        assert conf.method_quality == 0.7
        assert conf.internal_consistency == 0.85
        assert conf.temporal_freshness == 0.6
        assert conf.corroboration == 0.75
        assert conf.domain_applicability == 0.8

    def test_from_dict_with_missing_overall(self):
        """from_dict() should default overall to 0.7."""
        conf = DimensionalConfidence.from_dict({})
        assert conf.overall == 0.7

    def test_from_dict_roundtrip(self):
        """to_dict and from_dict should roundtrip."""
        original = DimensionalConfidence(
            overall=0.8,
            source_reliability=0.9,
            corroboration=0.7,
        )
        roundtripped = DimensionalConfidence.from_dict(original.to_dict())
        assert roundtripped.overall == original.overall
        assert roundtripped.source_reliability == original.source_reliability
        assert roundtripped.corroboration == original.corroboration


class TestDimensionalConfidenceStr:
    """Tests for DimensionalConfidence.__str__()."""

    def test_str_includes_overall(self):
        """__str__() should include overall."""
        conf = DimensionalConfidence(overall=0.85)
        s = str(conf)
        assert "overall=0.85" in s
        assert "Confidence(" in s

    def test_str_includes_set_dimensions(self):
        """__str__() should include set dimensions."""
        conf = DimensionalConfidence(overall=0.8, source_reliability=0.9)
        s = str(conf)
        assert "overall=0.80" in s
        assert "source" in s  # Truncated name


# ============================================================================
# confidence_label Tests
# ============================================================================


class TestConfidenceLabel:
    """Tests for confidence_label function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.95, "very high"),
            (0.90, "very high"),
            (0.89, "high"),
            (0.75, "high"),
            (0.74, "moderate"),
            (0.50, "moderate"),
            (0.49, "low"),
            (0.25, "low"),
            (0.24, "very low"),
            (0.0, "very low"),
        ],
    )
    def test_confidence_label_boundaries(self, value, expected):
        """Test confidence label at various boundaries."""
        assert confidence_label(value) == expected

    def test_confidence_label_exact_boundary_very_high(self):
        """0.9 should be 'very high'."""
        assert confidence_label(0.9) == "very high"

    def test_confidence_label_just_below_very_high(self):
        """Just below 0.9 should be 'high'."""
        assert confidence_label(0.899) == "high"


# ============================================================================
# aggregate_confidence Tests
# ============================================================================


class TestAggregateConfidence:
    """Tests for aggregate_confidence function."""

    def test_empty_list_returns_default(self):
        """Empty list should return default confidence."""
        result = aggregate_confidence([])
        assert result.overall == 0.5

    def test_single_item_returns_same(self):
        """Single item should return that item."""
        conf = DimensionalConfidence(overall=0.8, source_reliability=0.9)
        result = aggregate_confidence([conf])
        assert result is conf

    def test_weighted_average_default(self):
        """Default method is weighted_average."""
        c1 = DimensionalConfidence(overall=0.8)
        c2 = DimensionalConfidence(overall=0.6)
        result = aggregate_confidence([c1, c2])
        # Weighted by overall scores
        assert result.overall > 0

    def test_minimum_method(self):
        """minimum method should use minimum of each dimension."""
        c1 = DimensionalConfidence(overall=0.8, source_reliability=0.9)
        c2 = DimensionalConfidence(overall=0.6, source_reliability=0.7)
        result = aggregate_confidence([c1, c2], method="minimum")
        assert result.overall == 0.6
        assert result.source_reliability == 0.7

    def test_maximum_method(self):
        """maximum method should use maximum of each dimension."""
        c1 = DimensionalConfidence(overall=0.8, source_reliability=0.7)
        c2 = DimensionalConfidence(overall=0.6, source_reliability=0.9)
        result = aggregate_confidence([c1, c2], method="maximum")
        assert result.overall == 0.8
        assert result.source_reliability == 0.9

    def test_minimum_with_none_values(self):
        """minimum should handle None values correctly."""
        c1 = DimensionalConfidence(overall=0.8, source_reliability=0.9)
        c2 = DimensionalConfidence(overall=0.6)  # No source_reliability
        result = aggregate_confidence([c1, c2], method="minimum")
        assert result.overall == 0.6
        assert result.source_reliability == 0.9  # Only one value, so it's the min

    def test_maximum_with_none_values(self):
        """maximum should handle None values correctly."""
        c1 = DimensionalConfidence(overall=0.8)
        c2 = DimensionalConfidence(overall=0.6, corroboration=0.7)
        result = aggregate_confidence([c1, c2], method="maximum")
        assert result.overall == 0.8
        assert result.corroboration == 0.7

    def test_weighted_average_with_dimensions(self):
        """weighted_average should handle dimensions."""
        c1 = DimensionalConfidence(overall=0.8, source_reliability=1.0)
        c2 = DimensionalConfidence(overall=0.2, source_reliability=0.0)
        result = aggregate_confidence([c1, c2], method="weighted_average")
        # c1 weighted 0.8, c2 weighted 0.2
        # source_reliability = (1.0*0.8 + 0.0*0.2) / (0.8 + 0.2) = 0.8
        assert result.source_reliability == pytest.approx(0.8, rel=0.01)

    def test_three_confidences(self):
        """Test with three confidences."""
        c1 = DimensionalConfidence(overall=0.9)
        c2 = DimensionalConfidence(overall=0.7)
        c3 = DimensionalConfidence(overall=0.5)
        result = aggregate_confidence([c1, c2, c3], method="minimum")
        assert result.overall == 0.5

    def test_weighted_average_with_all_zero_overall(self):
        """weighted_average should handle all zero overalls."""
        c1 = DimensionalConfidence(overall=0.0)
        c2 = DimensionalConfidence(overall=0.0)
        result = aggregate_confidence([c1, c2], method="weighted_average")
        # Should not divide by zero
        assert result.overall == 0.0


# ============================================================================
# Extensible Dimensions Tests (#266)
# ============================================================================


class TestExtensibleDimensionsSchema:
    """Tests for schema field in DimensionalConfidence."""

    def test_default_schema(self):
        """Default schema should be v1.confidence.core."""
        conf = DimensionalConfidence(overall=0.8)
        assert conf.schema == "v1.confidence.core"

    def test_custom_schema(self):
        """Schema can be set to custom value."""
        conf = DimensionalConfidence(overall=0.8, schema="v2.trust.social")
        assert conf.schema == "v2.trust.social"

    def test_schema_preserved_in_to_dict(self):
        """to_dict() should include schema."""
        conf = DimensionalConfidence(overall=0.8, schema="custom.schema")
        d = conf.to_dict()
        assert d["schema"] == "custom.schema"

    def test_schema_restored_from_dict(self):
        """from_dict() should restore schema."""
        data = {"overall": 0.8, "schema": "v2.trust.social"}
        conf = DimensionalConfidence.from_dict(data)
        assert conf.schema == "v2.trust.social"

    def test_missing_schema_defaults(self):
        """from_dict() with missing schema should use default."""
        data = {"overall": 0.8}
        conf = DimensionalConfidence.from_dict(data)
        assert conf.schema == "v1.confidence.core"


class TestExtensibleDimensionsCustom:
    """Tests for custom dimensions via dimensions dict."""

    def test_custom_dimensions_via_dict(self):
        """Custom dimensions can be passed via dimensions dict."""
        conf = DimensionalConfidence(
            overall=0.8,
            dimensions={"trust_conclusions": 0.9, "trust_reasoning": 0.7},
        )
        assert conf.dimensions["trust_conclusions"] == 0.9
        assert conf.dimensions["trust_reasoning"] == 0.7

    def test_custom_dimensions_preserved_in_to_dict(self):
        """to_dict() should include custom dimensions."""
        conf = DimensionalConfidence(
            overall=0.8,
            schema="v2.trust.social",
            dimensions={"trust_honesty": 0.85, "trust_predictive": 0.6},
        )
        d = conf.to_dict()
        assert d["trust_honesty"] == 0.85
        assert d["trust_predictive"] == 0.6

    def test_custom_dimensions_restored_from_dict(self):
        """from_dict() should restore custom dimensions."""
        data = {
            "overall": 0.8,
            "schema": "v2.trust.social",
            "trust_methodology": 0.75,
            "trust_domain": 0.9,
        }
        conf = DimensionalConfidence.from_dict(data)
        assert conf.dimensions["trust_methodology"] == 0.75
        assert conf.dimensions["trust_domain"] == 0.9

    def test_mixed_core_and_custom_dimensions(self):
        """Core and custom dimensions can coexist."""
        conf = DimensionalConfidence(
            overall=0.8,
            source_reliability=0.9,  # Core dimension via kwarg
            dimensions={"trust_honesty": 0.7},  # Custom dimension
        )
        assert conf.source_reliability == 0.9
        assert conf.dimensions["trust_honesty"] == 0.7
        # Core dimension should also be in dimensions dict
        assert conf.dimensions["source_reliability"] == 0.9


class TestExtensibleDimensionsCalculation:
    """Tests for overall calculation with custom dimensions."""

    def test_recalculate_includes_custom_dimensions(self):
        """recalculate_overall should include custom dimensions."""
        conf = DimensionalConfidence(
            overall=0.5,
            dimensions={
                "dim_a": 0.8,
                "dim_b": 0.6,
            },
        )
        conf.recalculate_overall()
        # With equal weights, geometric mean of 0.8 and 0.6 ≈ 0.693
        assert abs(conf.overall - 0.693) < 0.01

    def test_full_with_custom_dimensions(self):
        """full() should work with custom dimensions."""
        conf = DimensionalConfidence.full(
            source_reliability=0.8,
            method_quality=0.8,
            internal_consistency=0.8,
            temporal_freshness=0.8,
            corroboration=0.8,
            domain_applicability=0.8,
        )
        # All equal = geometric mean = 0.8
        assert abs(conf.overall - 0.8) < 0.01


class TestExtensibleDimensionsBackwardCompat:
    """Tests for backward compatibility with pre-extension code."""

    def test_old_style_construction(self):
        """Old-style construction with kwargs still works."""
        conf = DimensionalConfidence(
            overall=0.7,
            source_reliability=0.9,
            method_quality=0.8,
        )
        # Properties should work
        assert conf.overall == 0.7
        assert conf.source_reliability == 0.9
        assert conf.method_quality == 0.8
        # Schema should be default
        assert conf.schema == "v1.confidence.core"

    def test_old_style_to_dict_from_dict_roundtrip(self):
        """Old-style serialization should still work."""
        original = DimensionalConfidence(
            overall=0.8,
            source_reliability=0.9,
            corroboration=0.7,
        )
        data = original.to_dict()
        restored = DimensionalConfidence.from_dict(data)
        assert restored.overall == original.overall
        assert restored.source_reliability == original.source_reliability
        assert restored.corroboration == original.corroboration

    def test_legacy_data_without_schema(self):
        """Legacy data without schema field should work."""
        # Simulates data from before #266
        legacy_data = {
            "overall": 0.8,
            "source_reliability": 0.9,
        }
        conf = DimensionalConfidence.from_dict(legacy_data)
        assert conf.overall == 0.8
        assert conf.source_reliability == 0.9
        assert conf.schema == "v1.confidence.core"  # Default

    def test_with_dimension_preserves_schema(self):
        """with_dimension() should preserve custom schema."""
        original = DimensionalConfidence(overall=0.7, schema="custom.schema")
        new = original.with_dimension("source_reliability", 0.9)
        assert new.schema == "custom.schema"
