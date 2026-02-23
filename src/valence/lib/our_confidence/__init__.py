"""our-confidence — Dimensional confidence system for belief scoring."""

__version__ = "0.1.0"

from .confidence import (
    CORE_DIMENSIONS,
    DEFAULT_SCHEMA,
    DEFAULT_WEIGHTS,
    EPSILON,
    ConfidenceDimension,
    DimensionalConfidence,
    aggregate_confidence,
    confidence_label,
)
from .dimension_registry import (
    DimensionRegistry,
    DimensionSchema,
    ValidationResult,
    get_registry,
    reset_registry,
)

__all__ = [
    "CORE_DIMENSIONS",
    "ConfidenceDimension",
    "DEFAULT_SCHEMA",
    "DEFAULT_WEIGHTS",
    "DimensionRegistry",
    "DimensionSchema",
    "DimensionalConfidence",
    "EPSILON",
    "ValidationResult",
    "aggregate_confidence",
    "confidence_label",
    "get_registry",
    "reset_registry",
]
