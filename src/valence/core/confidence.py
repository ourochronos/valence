"""Dimensional confidence system for Valence.

Multi-dimensional confidence scoring for knowledge units.
"""

import math
from enum import StrEnum
from typing import Any


class ConfidenceDimension(StrEnum):
    """Core dimensions of belief confidence."""

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

# Default weights for combining dimensions
DEFAULT_WEIGHTS: dict[str, float] = {
    ConfidenceDimension.SOURCE_RELIABILITY: 0.25,
    ConfidenceDimension.METHOD_QUALITY: 0.15,
    ConfidenceDimension.INTERNAL_CONSISTENCY: 0.20,
    ConfidenceDimension.TEMPORAL_FRESHNESS: 0.15,
    ConfidenceDimension.CORROBORATION: 0.15,
    ConfidenceDimension.DOMAIN_APPLICABILITY: 0.10,
}

# Floor value to prevent zeros in geometric mean
EPSILON = 0.001


def _compute_overall(
    dims: dict[str, float],
    weights: dict[str, float] | None = None,
    use_geometric: bool = True,
) -> float:
    """Compute overall confidence from dimension values using weighted geometric mean."""
    if not dims:
        return 0.5

    effective_weights = weights or DEFAULT_WEIGHTS

    if use_geometric:
        # Weighted geometric mean in log-space for numerical stability
        log_sum = 0.0
        total_weight = 0.0

        for dim, value in dims.items():
            w = effective_weights.get(dim, 1.0 / len(dims))
            if w > 0:
                safe_value = max(EPSILON, value)
                log_sum += w * math.log(safe_value)
                total_weight += w

        if total_weight > 0:
            overall = math.exp(log_sum / total_weight)
        else:
            overall = 0.5
    else:
        # Weighted arithmetic mean
        weighted_sum = 0.0
        total_weight = 0.0

        for dim, value in dims.items():
            w = effective_weights.get(dim, 1.0 / len(dims))
            if w > 0:
                weighted_sum += w * value
                total_weight += w

        overall = weighted_sum / total_weight if total_weight > 0 else 0.5

    return min(1.0, max(0.0, overall))


class DimensionalConfidence:
    """Multi-dimensional confidence for a belief.

    Attributes:
        overall: Combined score (0-1)
        schema: Schema identifier (default: v1.confidence.core)
        dimensions: Dict of dimension name -> value (0-1)
    """

    def __init__(
        self,
        overall: float = 0.7,
        schema: str = DEFAULT_SCHEMA,
        dimensions: dict[str, float] | None = None,
        # Backward-compatible kwargs
        source_reliability: float | None = None,
        method_quality: float | None = None,
        internal_consistency: float | None = None,
        temporal_freshness: float | None = None,
        corroboration: float | None = None,
        domain_applicability: float | None = None,
    ) -> None:
        """Initialize dimensional confidence.

        Args:
            overall: Overall confidence score (0-1)
            schema: Schema identifier
            dimensions: Dict of dimension values
            **kwargs: Core dimension values (backward compatible)
        """
        self.overall = overall
        self.schema = schema
        self.dimensions = dimensions.copy() if dimensions else {}

        # Merge core dimension kwargs
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

        # Validate ranges
        if not 0 <= self.overall <= 1:
            raise ValueError(f"overall must be in [0,1], got {self.overall}")
        for dim, value in self.dimensions.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{dim} must be in [0,1], got {value}")

    @property
    def source_reliability(self) -> float:
        """Get source reliability dimension."""
        return self.dimensions.get(ConfidenceDimension.SOURCE_RELIABILITY, self.overall)

    @property
    def method_quality(self) -> float:
        """Get method quality dimension."""
        return self.dimensions.get(ConfidenceDimension.METHOD_QUALITY, self.overall)

    @property
    def internal_consistency(self) -> float:
        """Get internal consistency dimension."""
        return self.dimensions.get(ConfidenceDimension.INTERNAL_CONSISTENCY, self.overall)

    @property
    def temporal_freshness(self) -> float:
        """Get temporal freshness dimension."""
        return self.dimensions.get(ConfidenceDimension.TEMPORAL_FRESHNESS, self.overall)

    @property
    def corroboration(self) -> float:
        """Get corroboration dimension."""
        return self.dimensions.get(ConfidenceDimension.CORROBORATION, self.overall)

    @property
    def domain_applicability(self) -> float:
        """Get domain applicability dimension."""
        return self.dimensions.get(ConfidenceDimension.DOMAIN_APPLICABILITY, self.overall)

    def recalculate_overall(self, weights: dict[str, float] | None = None, use_geometric: bool = True) -> None:
        """Recalculate overall score from dimensions."""
        if self.dimensions:
            self.overall = _compute_overall(self.dimensions, weights, use_geometric)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "schema": self.schema,
            "dimensions": self.dimensions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DimensionalConfidence":
        """Create from dictionary."""
        return cls(
            overall=data.get("overall", 0.7),
            schema=data.get("schema", DEFAULT_SCHEMA),
            dimensions=data.get("dimensions"),
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"DimensionalConfidence(overall={self.overall:.2f}, dims={len(self.dimensions)})"


def aggregate_confidence(confidences: list[DimensionalConfidence]) -> DimensionalConfidence:
    """Aggregate multiple confidence objects into one.

    Uses geometric mean across each dimension.

    Args:
        confidences: List of confidence objects to aggregate

    Returns:
        New DimensionalConfidence with aggregated values
    """
    if not confidences:
        return DimensionalConfidence()

    if len(confidences) == 1:
        return confidences[0]

    # Collect all dimension keys
    all_dims: set[str] = set()
    for conf in confidences:
        all_dims.update(conf.dimensions.keys())

    # Aggregate each dimension using geometric mean
    aggregated_dims: dict[str, float] = {}
    for dim in all_dims:
        values = [conf.dimensions.get(dim, conf.overall) for conf in confidences]
        # Geometric mean
        product = 1.0
        for v in values:
            product *= max(EPSILON, v)
        aggregated_dims[dim] = product ** (1.0 / len(values))

    # Calculate overall
    overall = _compute_overall(aggregated_dims)

    return DimensionalConfidence(overall=overall, dimensions=aggregated_dims)


def confidence_label(score: float) -> str:
    """Get human-readable label for confidence score.

    Args:
        score: Confidence score (0-1)

    Returns:
        Label: very low, low, medium, high, very high
    """
    if score < 0.2:
        return "very low"
    elif score < 0.4:
        return "low"
    elif score < 0.6:
        return "medium"
    elif score < 0.8:
        return "high"
    else:
        return "very high"
