"""Dimensional confidence system for Valence beliefs.

Instead of a single confidence score, beliefs have multiple dimensions
that can be independently assessed and combined.

Per spec (MATH.md), geometric mean is preferred for overall confidence
calculation as it better penalizes imbalanced vectors.

This module supports extensible dimensions:
- Core dimensions (v1.confidence.core) are the original 6
- Additional dimensions can be added via the `dimensions` dict
- Schema field tracks which dimension set is in use
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any


class ConfidenceDimension(StrEnum):
    """Core dimensions of belief confidence (v1.confidence.core)."""

    OVERALL = "overall"
    SOURCE_RELIABILITY = "source_reliability"
    METHOD_QUALITY = "method_quality"
    INTERNAL_CONSISTENCY = "internal_consistency"
    TEMPORAL_FRESHNESS = "temporal_freshness"
    CORROBORATION = "corroboration"
    DOMAIN_APPLICABILITY = "domain_applicability"


# Core dimensions (excluding overall)
CORE_DIMENSIONS = [
    ConfidenceDimension.SOURCE_RELIABILITY,
    ConfidenceDimension.METHOD_QUALITY,
    ConfidenceDimension.INTERNAL_CONSISTENCY,
    ConfidenceDimension.TEMPORAL_FRESHNESS,
    ConfidenceDimension.CORROBORATION,
    ConfidenceDimension.DOMAIN_APPLICABILITY,
]

# Default schema identifier
DEFAULT_SCHEMA = "v1.confidence.core"

# Default weights for combining dimensions into overall score
DEFAULT_WEIGHTS: dict[str, float] = {
    ConfidenceDimension.SOURCE_RELIABILITY: 0.25,
    ConfidenceDimension.METHOD_QUALITY: 0.15,
    ConfidenceDimension.INTERNAL_CONSISTENCY: 0.20,
    ConfidenceDimension.TEMPORAL_FRESHNESS: 0.15,
    ConfidenceDimension.CORROBORATION: 0.15,
    ConfidenceDimension.DOMAIN_APPLICABILITY: 0.10,
}

# Floor value to prevent zeros in geometric mean calculations
EPSILON = 0.001


def _compute_overall(
    dims: dict[str, float],
    weights: dict[str, float] | None = None,
    use_geometric: bool = True,
) -> float:
    """Compute overall confidence from dimension values.

    Per spec (MATH.md):
    - Geometric mean: (∏v_i^{w_i})^{1/∑w_i} - penalizes imbalanced vectors
    - Arithmetic mean: ∑(w_i * v_i) / ∑w_i - traditional weighted average

    Args:
        dims: Dictionary of dimension name -> value
        weights: Dictionary of dimension name -> weight (defaults to equal weights)
        use_geometric: If True, use weighted geometric mean (recommended per spec)

    Returns:
        Overall confidence score in [0, 1]
    """
    if not dims:
        return 0.5  # Default when no dimensions present

    # Use provided weights, fall back to defaults, then equal weights
    effective_weights = weights or DEFAULT_WEIGHTS

    if use_geometric:
        # Weighted geometric mean: (∏v_i^{w_i})^{1/∑w_i}
        # Use log-space for numerical stability: exp(∑(w_i * log(v_i)) / ∑w_i)
        log_sum = 0.0
        total_weight = 0.0

        for dim, value in dims.items():
            # Get weight: try explicit weights, then default to 1.0
            w = effective_weights.get(dim, 1.0 / len(dims))
            if w > 0:
                # Use floor to prevent log(0)
                safe_value = max(EPSILON, value)
                log_sum += w * math.log(safe_value)
                total_weight += w

        if total_weight > 0:
            overall = math.exp(log_sum / total_weight)
        else:
            overall = 0.5
    else:
        # Weighted arithmetic mean: ∑(w_i * v_i) / ∑w_i
        weighted_sum = 0.0
        total_weight = 0.0

        for dim, value in dims.items():
            w = effective_weights.get(dim, 1.0 / len(dims))
            if w > 0:
                weighted_sum += w * value
                total_weight += w

        if total_weight > 0:
            overall = weighted_sum / total_weight
        else:
            overall = 0.5

    return min(1.0, max(0.0, overall))


class DimensionalConfidence:
    """Multi-dimensional confidence for a belief.

    Supports both core dimensions (v1.confidence.core) and extensible
    custom dimensions via the `dimensions` dict.

    Core dimensions:
    - overall: Combined score (0-1)
    - source_reliability: How trustworthy is the source?
    - method_quality: How rigorous was the extraction/inference?
    - internal_consistency: Does it align with other beliefs?
    - temporal_freshness: How recent is this information?
    - corroboration: Is it supported by multiple sources?
    - domain_applicability: How relevant is it to current context?

    Custom dimensions can be added via the `dimensions` dict and will
    be included in serialization and overall calculation.

    Backward compatible: accepts core dimension names as keyword arguments.
    """

    def __init__(
        self,
        overall: float = 0.7,
        schema: str = DEFAULT_SCHEMA,
        dimensions: dict[str, float] | None = None,
        # Backward-compatible kwargs for core dimensions
        source_reliability: float | None = None,
        method_quality: float | None = None,
        internal_consistency: float | None = None,
        temporal_freshness: float | None = None,
        corroboration: float | None = None,
        domain_applicability: float | None = None,
    ) -> None:
        """Initialize with optional core dimensions as kwargs for backward compat."""
        self.overall = overall
        self.schema = schema
        self.dimensions = dimensions.copy() if dimensions else {}

        # Merge in core dimension kwargs (backward compat)
        if source_reliability is not None:
            self.dimensions[ConfidenceDimension.SOURCE_RELIABILITY] = source_reliability
        if method_quality is not None:
            self.dimensions[ConfidenceDimension.METHOD_QUALITY] = method_quality
        if internal_consistency is not None:
            self.dimensions[ConfidenceDimension.INTERNAL_CONSISTENCY] = internal_consistency
        if temporal_freshness is not None:
            self.dimensions[ConfidenceDimension.TEMPORAL_FRESHNESS] = temporal_freshness
        if corroboration is not None:
            self.dimensions[ConfidenceDimension.CORROBORATION] = corroboration
        if domain_applicability is not None:
            self.dimensions[ConfidenceDimension.DOMAIN_APPLICABILITY] = domain_applicability

        # Validate
        if self.overall < 0 or self.overall > 1:
            raise ValueError(f"overall must be between 0 and 1, got {self.overall}")
        for dim, value in self.dimensions.items():
            if value < 0 or value > 1:
                raise ValueError(f"{dim} must be between 0 and 1, got {value}")

    # =========================================================================
    # Backward-compatible properties for core dimensions
    # =========================================================================

    @property
    def source_reliability(self) -> float | None:
        """How trustworthy is the source?"""
        return self.dimensions.get(ConfidenceDimension.SOURCE_RELIABILITY)

    @source_reliability.setter
    def source_reliability(self, value: float | None) -> None:
        if value is not None:
            self.dimensions[ConfidenceDimension.SOURCE_RELIABILITY] = value
        elif ConfidenceDimension.SOURCE_RELIABILITY in self.dimensions:
            del self.dimensions[ConfidenceDimension.SOURCE_RELIABILITY]

    @property
    def method_quality(self) -> float | None:
        """How rigorous was the extraction/inference?"""
        return self.dimensions.get(ConfidenceDimension.METHOD_QUALITY)

    @method_quality.setter
    def method_quality(self, value: float | None) -> None:
        if value is not None:
            self.dimensions[ConfidenceDimension.METHOD_QUALITY] = value
        elif ConfidenceDimension.METHOD_QUALITY in self.dimensions:
            del self.dimensions[ConfidenceDimension.METHOD_QUALITY]

    @property
    def internal_consistency(self) -> float | None:
        """Does it align with other beliefs?"""
        return self.dimensions.get(ConfidenceDimension.INTERNAL_CONSISTENCY)

    @internal_consistency.setter
    def internal_consistency(self, value: float | None) -> None:
        if value is not None:
            self.dimensions[ConfidenceDimension.INTERNAL_CONSISTENCY] = value
        elif ConfidenceDimension.INTERNAL_CONSISTENCY in self.dimensions:
            del self.dimensions[ConfidenceDimension.INTERNAL_CONSISTENCY]

    @property
    def temporal_freshness(self) -> float | None:
        """How recent is this information?"""
        return self.dimensions.get(ConfidenceDimension.TEMPORAL_FRESHNESS)

    @temporal_freshness.setter
    def temporal_freshness(self, value: float | None) -> None:
        if value is not None:
            self.dimensions[ConfidenceDimension.TEMPORAL_FRESHNESS] = value
        elif ConfidenceDimension.TEMPORAL_FRESHNESS in self.dimensions:
            del self.dimensions[ConfidenceDimension.TEMPORAL_FRESHNESS]

    @property
    def corroboration(self) -> float | None:
        """Is it supported by multiple sources?"""
        return self.dimensions.get(ConfidenceDimension.CORROBORATION)

    @corroboration.setter
    def corroboration(self, value: float | None) -> None:
        if value is not None:
            self.dimensions[ConfidenceDimension.CORROBORATION] = value
        elif ConfidenceDimension.CORROBORATION in self.dimensions:
            del self.dimensions[ConfidenceDimension.CORROBORATION]

    @property
    def domain_applicability(self) -> float | None:
        """How relevant is it to current context?"""
        return self.dimensions.get(ConfidenceDimension.DOMAIN_APPLICABILITY)

    @domain_applicability.setter
    def domain_applicability(self, value: float | None) -> None:
        if value is not None:
            self.dimensions[ConfidenceDimension.DOMAIN_APPLICABILITY] = value
        elif ConfidenceDimension.DOMAIN_APPLICABILITY in self.dimensions:
            del self.dimensions[ConfidenceDimension.DOMAIN_APPLICABILITY]

    # =========================================================================
    # Extensible dimension access
    # =========================================================================

    def get_dimension(self, name: str) -> float | None:
        """Get any dimension by name."""
        return self.dimensions.get(name)

    def set_dimension(self, name: str, value: float | None) -> None:
        """Set any dimension by name."""
        if value is not None:
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
            self.dimensions[name] = value
        elif name in self.dimensions:
            del self.dimensions[name]

    def has_dimension(self, name: str) -> bool:
        """Check if a dimension is set."""
        return name in self.dimensions

    # =========================================================================
    # Factory methods
    # =========================================================================

    @classmethod
    def simple(cls, overall: float) -> DimensionalConfidence:
        """Create a simple confidence with just overall score."""
        return cls(overall=overall)

    @classmethod
    def full(
        cls,
        source_reliability: float,
        method_quality: float,
        internal_consistency: float,
        temporal_freshness: float,
        corroboration: float,
        domain_applicability: float,
        weights: dict[str, float] | None = None,
        use_geometric: bool = True,
    ) -> DimensionalConfidence:
        """Create full dimensional confidence with calculated overall.

        Args:
            use_geometric: If True (default), use weighted geometric mean per spec.
                          This better penalizes imbalanced vectors.
        """
        dims: dict[str, float] = {
            str(ConfidenceDimension.SOURCE_RELIABILITY): source_reliability,
            str(ConfidenceDimension.METHOD_QUALITY): method_quality,
            str(ConfidenceDimension.INTERNAL_CONSISTENCY): internal_consistency,
            str(ConfidenceDimension.TEMPORAL_FRESHNESS): temporal_freshness,
            str(ConfidenceDimension.CORROBORATION): corroboration,
            str(ConfidenceDimension.DOMAIN_APPLICABILITY): domain_applicability,
        }
        overall = _compute_overall(dims, weights, use_geometric=use_geometric)
        return cls(overall=overall, dimensions=dims)

    @classmethod
    def from_dimensions(
        cls,
        dimensions: dict[str, float],
        schema: str = DEFAULT_SCHEMA,
        weights: dict[str, float] | None = None,
        use_geometric: bool = True,
    ) -> DimensionalConfidence:
        """Create confidence from arbitrary dimensions with calculated overall."""
        overall = _compute_overall(dimensions, weights, use_geometric=use_geometric)
        return cls(overall=overall, schema=schema, dimensions=dict(dimensions))

    # =========================================================================
    # Manipulation methods
    # =========================================================================

    def recalculate_overall(
        self,
        weights: dict[str, float] | None = None,
        use_geometric: bool = True,
    ) -> DimensionalConfidence:
        """Recalculate overall from dimensions.

        Args:
            weights: Custom weights for each dimension.
            use_geometric: If True (default), use weighted geometric mean per spec.
        """
        if self.dimensions:
            self.overall = _compute_overall(self.dimensions, weights, use_geometric=use_geometric)
        return self

    def with_dimension(
        self,
        dimension: str,
        value: float,
        recalculate: bool = True,
    ) -> DimensionalConfidence:
        """Return a new confidence with one dimension updated."""
        # Handle overall specially - it's not in the dimensions dict
        if dimension == ConfidenceDimension.OVERALL or dimension == "overall":
            return DimensionalConfidence(
                overall=value,
                schema=self.schema,
                dimensions=dict(self.dimensions),
            )

        new_dims = dict(self.dimensions)
        new_dims[dimension] = value
        result = DimensionalConfidence(
            overall=self.overall,
            schema=self.schema,
            dimensions=new_dims,
        )
        if recalculate:
            result.recalculate_overall()
        return result

    def decay(self, factor: float = 0.95) -> DimensionalConfidence:
        """Apply temporal decay to confidence.

        This reduces temporal_freshness and recalculates overall.
        """
        if self.temporal_freshness is not None:
            return self.with_dimension(
                ConfidenceDimension.TEMPORAL_FRESHNESS,
                self.temporal_freshness * factor,
            )
        # If no temporal dimension, decay overall directly
        return DimensionalConfidence(
            overall=self.overall * factor,
            schema=self.schema,
            dimensions=dict(self.dimensions),
        )

    def boost_corroboration(self, amount: float = 0.1) -> DimensionalConfidence:
        """Increase corroboration when additional sources confirm."""
        current = self.corroboration or 0.5
        new_value = min(1.0, current + amount)
        return self.with_dimension(ConfidenceDimension.CORROBORATION, new_value)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON/database storage.

        Output format:
        {
            "overall": 0.7,
            "schema": "v1.confidence.core",  # Only if not default
            "source_reliability": 0.8,       # Dimensions inline
            ...
        }
        """
        result: dict[str, Any] = {"overall": self.overall}

        # Include schema if not default
        if self.schema != DEFAULT_SCHEMA:
            result["schema"] = self.schema

        # Include all dimensions
        result.update(self.dimensions)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DimensionalConfidence:
        """Create from dictionary.

        Handles both old format (flat dimensions) and new format (with schema).
        """
        overall = data.get("overall", 0.7)
        schema = data.get("schema", DEFAULT_SCHEMA)

        # Extract dimensions (everything except overall and schema)
        dimensions: dict[str, float] = {}
        for key, value in data.items():
            if key in ("overall", "schema"):
                continue
            if isinstance(value, int | float):
                dimensions[key] = float(value)

        return cls(overall=overall, schema=schema, dimensions=dimensions)

    def __str__(self) -> str:
        """Human-readable representation."""
        parts = [f"overall={self.overall:.2f}"]
        for dim, value in sorted(self.dimensions.items()):
            short_name = dim.replace("_", "")[:6]
            parts.append(f"{short_name}={value:.2f}")
        if self.schema != DEFAULT_SCHEMA:
            parts.append(f"schema={self.schema}")
        return f"Confidence({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, DimensionalConfidence):
            return False
        return abs(self.overall - other.overall) < 0.0001 and self.schema == other.schema and self.dimensions == other.dimensions


def confidence_label(confidence: float) -> str:
    """Get a human-readable label for a confidence value."""
    if confidence >= 0.9:
        return "very high"
    elif confidence >= 0.75:
        return "high"
    elif confidence >= 0.5:
        return "moderate"
    elif confidence >= 0.25:
        return "low"
    else:
        return "very low"


def aggregate_confidence(
    confidences: list[DimensionalConfidence],
    method: str = "geometric",
) -> DimensionalConfidence:
    """Aggregate multiple confidences into one.

    Methods:
    - geometric (default): Weighted geometric mean per spec (penalizes imbalance)
    - weighted_average: Arithmetic average weighted by each confidence's overall score
    - minimum: Use the minimum of each dimension
    - maximum: Use the maximum of each dimension

    Per spec (MATH.md), geometric mean is preferred as it better handles
    imbalanced vectors and propagates low confidence appropriately.
    """
    if not confidences:
        return DimensionalConfidence.simple(0.5)

    if len(confidences) == 1:
        return confidences[0]

    # Collect all dimension names across all confidences
    all_dims: set[str] = set()
    for c in confidences:
        all_dims.update(c.dimensions.keys())

    # Use schema from first confidence (they should match in practice)
    schema = confidences[0].schema

    if method == "minimum":
        result_dims: dict[str, float] = {}
        for dim in all_dims:
            values = [c.dimensions[dim] for c in confidences if dim in c.dimensions]
            if values:
                result_dims[dim] = min(values)
        return DimensionalConfidence(
            overall=min(c.overall for c in confidences),
            schema=schema,
            dimensions=result_dims,
        )

    elif method == "maximum":
        result_dims = {}
        for dim in all_dims:
            values = [c.dimensions[dim] for c in confidences if dim in c.dimensions]
            if values:
                result_dims[dim] = max(values)
        return DimensionalConfidence(
            overall=max(c.overall for c in confidences),
            schema=schema,
            dimensions=result_dims,
        )

    elif method == "geometric":
        # Weighted geometric mean per spec (MATH.md)
        def weighted_geo(dim: str) -> float | None:
            """Compute weighted geometric mean for a dimension."""
            values = [(c.dimensions[dim], c.overall) for c in confidences if dim in c.dimensions]
            if not values:
                return None
            # Geometric mean in log space: exp(∑(w * log(v)) / ∑w)
            total_w = sum(w for _, w in values)
            if total_w == 0:
                total_w = len(values)
            log_sum = sum(w * math.log(max(EPSILON, v)) for v, w in values)
            return math.exp(log_sum / total_w)

        result_dims = {}
        for dim in all_dims:
            val = weighted_geo(dim)
            if val is not None:
                result_dims[dim] = val

        # For overall, use geometric mean of the overall scores
        overall_values = [c.overall for c in confidences]
        overall_weights = [c.overall for c in confidences]  # Self-weighted
        if sum(overall_weights) == 0:
            overall_weights = [1.0] * len(confidences)
        log_sum = sum(w * math.log(max(EPSILON, v)) for v, w in zip(overall_values, overall_weights))
        geo_overall = math.exp(log_sum / sum(overall_weights))

        return DimensionalConfidence(
            overall=min(1.0, max(0.0, geo_overall)),
            schema=schema,
            dimensions=result_dims,
        )

    else:  # weighted_average (arithmetic)

        def weighted_avg(dim: str) -> float | None:
            values = [(c.dimensions[dim], c.overall) for c in confidences if dim in c.dimensions]
            if not values:
                return None
            total_w = sum(w for _, w in values)
            if total_w == 0:
                total_w = len(values)
            return sum(v * w for v, w in values) / total_w

        result_dims = {}
        for dim in all_dims:
            val = weighted_avg(dim)
            if val is not None:
                result_dims[dim] = val

        total_weight = sum(c.overall for c in confidences)
        if total_weight == 0:
            total_weight = len(confidences)

        return DimensionalConfidence(
            overall=sum(c.overall * c.overall for c in confidences) / total_weight,
            schema=schema,
            dimensions=result_dims,
        )
