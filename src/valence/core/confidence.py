"""Dimensional confidence system for Valence beliefs.

Instead of a single confidence score, beliefs have multiple dimensions
that can be independently assessed and combined.

Per spec (MATH.md), geometric mean is preferred for overall confidence
calculation as it better penalizes imbalanced vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConfidenceDimension(str, Enum):
    """Dimensions of belief confidence."""

    OVERALL = "overall"
    SOURCE_RELIABILITY = "source_reliability"
    METHOD_QUALITY = "method_quality"
    INTERNAL_CONSISTENCY = "internal_consistency"
    TEMPORAL_FRESHNESS = "temporal_freshness"
    CORROBORATION = "corroboration"
    DOMAIN_APPLICABILITY = "domain_applicability"


# Default weights for combining dimensions into overall score
DEFAULT_WEIGHTS: dict[ConfidenceDimension, float] = {
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
    dims: dict[ConfidenceDimension, float],
    weights: dict[ConfidenceDimension, float],
    use_geometric: bool = True,
) -> float:
    """Compute overall confidence from dimension values.
    
    Per spec (MATH.md):
    - Geometric mean: (∏v_i^{w_i})^{1/∑w_i} - penalizes imbalanced vectors
    - Arithmetic mean: ∑(w_i * v_i) / ∑w_i - traditional weighted average
    
    Args:
        dims: Dictionary of dimension -> value
        weights: Dictionary of dimension -> weight
        use_geometric: If True, use weighted geometric mean (recommended per spec)
    
    Returns:
        Overall confidence score in [0, 1]
    """
    if not dims:
        return 0.5  # Default when no dimensions present
    
    if use_geometric:
        # Weighted geometric mean: (∏v_i^{w_i})^{1/∑w_i}
        # Use log-space for numerical stability: exp(∑(w_i * log(v_i)) / ∑w_i)
        log_sum = 0.0
        total_weight = 0.0
        
        for dim, value in dims.items():
            w = weights.get(dim, 0.0)
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
            w = weights.get(dim, 0.0)
            if w > 0:
                weighted_sum += w * value
                total_weight += w
        
        if total_weight > 0:
            overall = weighted_sum / total_weight
        else:
            overall = 0.5
    
    return min(1.0, max(0.0, overall))


@dataclass
class DimensionalConfidence:
    """Multi-dimensional confidence for a belief.

    Dimensions:
    - overall: Combined score (0-1)
    - source_reliability: How trustworthy is the source?
    - method_quality: How rigorous was the extraction/inference?
    - internal_consistency: Does it align with other beliefs?
    - temporal_freshness: How recent is this information?
    - corroboration: Is it supported by multiple sources?
    - domain_applicability: How relevant is it to current context?
    """

    overall: float = 0.7
    source_reliability: float | None = None
    method_quality: float | None = None
    internal_consistency: float | None = None
    temporal_freshness: float | None = None
    corroboration: float | None = None
    domain_applicability: float | None = None

    def __post_init__(self) -> None:
        """Validate confidence values."""
        for dim in ConfidenceDimension:
            value = getattr(self, dim.value, None)
            if value is not None and (value < 0 or value > 1):
                raise ValueError(f"{dim.value} must be between 0 and 1, got {value}")

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
        weights: dict[ConfidenceDimension, float] | None = None,
        use_geometric: bool = True,
    ) -> DimensionalConfidence:
        """Create full dimensional confidence with calculated overall.
        
        Args:
            use_geometric: If True (default), use weighted geometric mean per spec.
                          This better penalizes imbalanced vectors.
        """
        w = weights or DEFAULT_WEIGHTS
        dims = {
            ConfidenceDimension.SOURCE_RELIABILITY: source_reliability,
            ConfidenceDimension.METHOD_QUALITY: method_quality,
            ConfidenceDimension.INTERNAL_CONSISTENCY: internal_consistency,
            ConfidenceDimension.TEMPORAL_FRESHNESS: temporal_freshness,
            ConfidenceDimension.CORROBORATION: corroboration,
            ConfidenceDimension.DOMAIN_APPLICABILITY: domain_applicability,
        }
        overall = _compute_overall(dims, w, use_geometric=use_geometric)
        return cls(
            overall=overall,
            source_reliability=source_reliability,
            method_quality=method_quality,
            internal_consistency=internal_consistency,
            temporal_freshness=temporal_freshness,
            corroboration=corroboration,
            domain_applicability=domain_applicability,
        )

    def recalculate_overall(
        self,
        weights: dict[ConfidenceDimension, float] | None = None,
        use_geometric: bool = True,
    ) -> DimensionalConfidence:
        """Recalculate overall from dimensions.
        
        Args:
            weights: Custom weights for each dimension.
            use_geometric: If True (default), use weighted geometric mean per spec.
        """
        w = weights or DEFAULT_WEIGHTS
        dims = {}
        for dim in ConfidenceDimension:
            if dim == ConfidenceDimension.OVERALL:
                continue
            value = getattr(self, dim.value, None)
            if value is not None:
                dims[dim] = value

        if dims:
            self.overall = _compute_overall(dims, w, use_geometric=use_geometric)

        return self

    def with_dimension(
        self,
        dimension: ConfidenceDimension,
        value: float,
        recalculate: bool = True,
    ) -> DimensionalConfidence:
        """Return a new confidence with one dimension updated."""
        result = DimensionalConfidence(
            overall=self.overall,
            source_reliability=self.source_reliability,
            method_quality=self.method_quality,
            internal_consistency=self.internal_consistency,
            temporal_freshness=self.temporal_freshness,
            corroboration=self.corroboration,
            domain_applicability=self.domain_applicability,
        )
        setattr(result, dimension.value, value)
        if recalculate and dimension != ConfidenceDimension.OVERALL:
            result.recalculate_overall()
        return result

    def decay(self, factor: float = 0.95) -> DimensionalConfidence:
        """Apply temporal decay to confidence.

        This reduces temporal_freshness and recalculates overall.
        """
        if self.temporal_freshness is not None:
            new_freshness = self.temporal_freshness * factor
            return self.with_dimension(
                ConfidenceDimension.TEMPORAL_FRESHNESS,
                new_freshness,
            )
        # If no temporal dimension, decay overall directly
        return DimensionalConfidence(
            overall=self.overall * factor,
            source_reliability=self.source_reliability,
            method_quality=self.method_quality,
            internal_consistency=self.internal_consistency,
            temporal_freshness=self.temporal_freshness,
            corroboration=self.corroboration,
            domain_applicability=self.domain_applicability,
        )

    def boost_corroboration(self, amount: float = 0.1) -> DimensionalConfidence:
        """Increase corroboration when additional sources confirm."""
        current = self.corroboration or 0.5
        new_value = min(1.0, current + amount)
        return self.with_dimension(ConfidenceDimension.CORROBORATION, new_value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON/database storage."""
        result: dict[str, Any] = {"overall": self.overall}
        for dim in ConfidenceDimension:
            if dim == ConfidenceDimension.OVERALL:
                continue
            value = getattr(self, dim.value, None)
            if value is not None:
                result[dim.value] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DimensionalConfidence:
        """Create from dictionary."""
        return cls(
            overall=data.get("overall", 0.7),
            source_reliability=data.get("source_reliability"),
            method_quality=data.get("method_quality"),
            internal_consistency=data.get("internal_consistency"),
            temporal_freshness=data.get("temporal_freshness"),
            corroboration=data.get("corroboration"),
            domain_applicability=data.get("domain_applicability"),
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        parts = [f"overall={self.overall:.2f}"]
        for dim in ConfidenceDimension:
            if dim == ConfidenceDimension.OVERALL:
                continue
            value = getattr(self, dim.value, None)
            if value is not None:
                short_name = dim.value.replace("_", "")[:6]
                parts.append(f"{short_name}={value:.2f}")
        return f"Confidence({', '.join(parts)})"


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

    if method == "minimum":
        return DimensionalConfidence(
            overall=min(c.overall for c in confidences),
            source_reliability=min((c.source_reliability for c in confidences if c.source_reliability is not None), default=None),
            method_quality=min((c.method_quality for c in confidences if c.method_quality is not None), default=None),
            internal_consistency=min((c.internal_consistency for c in confidences if c.internal_consistency is not None), default=None),
            temporal_freshness=min((c.temporal_freshness for c in confidences if c.temporal_freshness is not None), default=None),
            corroboration=min((c.corroboration for c in confidences if c.corroboration is not None), default=None),
            domain_applicability=min((c.domain_applicability for c in confidences if c.domain_applicability is not None), default=None),
        )

    elif method == "maximum":
        return DimensionalConfidence(
            overall=max(c.overall for c in confidences),
            source_reliability=max((c.source_reliability for c in confidences if c.source_reliability is not None), default=None),
            method_quality=max((c.method_quality for c in confidences if c.method_quality is not None), default=None),
            internal_consistency=max((c.internal_consistency for c in confidences if c.internal_consistency is not None), default=None),
            temporal_freshness=max((c.temporal_freshness for c in confidences if c.temporal_freshness is not None), default=None),
            corroboration=max((c.corroboration for c in confidences if c.corroboration is not None), default=None),
            domain_applicability=max((c.domain_applicability for c in confidences if c.domain_applicability is not None), default=None),
        )

    elif method == "geometric":
        # Weighted geometric mean per spec (MATH.md)
        total_weight = sum(c.overall for c in confidences)
        if total_weight == 0:
            total_weight = len(confidences)

        def weighted_geo(getter) -> float | None:
            """Compute weighted geometric mean for a dimension."""
            values = [(getter(c), c.overall) for c in confidences if getter(c) is not None]
            if not values:
                return None
            # Geometric mean in log space: exp(∑(w * log(v)) / ∑w)
            total_w = sum(w for _, w in values)
            if total_w == 0:
                total_w = len(values)
            log_sum = sum(w * math.log(max(EPSILON, v)) for v, w in values)
            return math.exp(log_sum / total_w)

        # For overall, use geometric mean of the overall scores
        overall_values = [c.overall for c in confidences]
        overall_weights = [c.overall for c in confidences]  # Self-weighted
        if sum(overall_weights) == 0:
            overall_weights = [1.0] * len(confidences)
        log_sum = sum(w * math.log(max(EPSILON, v)) for v, w in zip(overall_values, overall_weights))
        geo_overall = math.exp(log_sum / sum(overall_weights))

        return DimensionalConfidence(
            overall=min(1.0, max(0.0, geo_overall)),
            source_reliability=weighted_geo(lambda c: c.source_reliability),
            method_quality=weighted_geo(lambda c: c.method_quality),
            internal_consistency=weighted_geo(lambda c: c.internal_consistency),
            temporal_freshness=weighted_geo(lambda c: c.temporal_freshness),
            corroboration=weighted_geo(lambda c: c.corroboration),
            domain_applicability=weighted_geo(lambda c: c.domain_applicability),
        )

    else:  # weighted_average (arithmetic)
        total_weight = sum(c.overall for c in confidences)
        if total_weight == 0:
            total_weight = len(confidences)

        def weighted_avg(getter) -> float | None:
            values = [(getter(c), c.overall) for c in confidences if getter(c) is not None]
            if not values:
                return None
            return sum(v * w for v, w in values) / sum(w for _, w in values)

        return DimensionalConfidence(
            overall=sum(c.overall * c.overall for c in confidences) / total_weight,
            source_reliability=weighted_avg(lambda c: c.source_reliability),
            method_quality=weighted_avg(lambda c: c.method_quality),
            internal_consistency=weighted_avg(lambda c: c.internal_consistency),
            temporal_freshness=weighted_avg(lambda c: c.temporal_freshness),
            corroboration=weighted_avg(lambda c: c.corroboration),
            domain_applicability=weighted_avg(lambda c: c.domain_applicability),
        )
