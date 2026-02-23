"""Public interface for our-confidence.

The primary public types are:
- DimensionalConfidence: Multi-dimensional confidence for beliefs
- ConfidenceDimension: Enum of core dimension names
- DimensionRegistry: Schema registry for dimension validation
- DimensionSchema: Schema definition for dimension sets
"""

# Re-exports for interface discovery
from .confidence import ConfidenceDimension, DimensionalConfidence, aggregate_confidence, confidence_label
from .dimension_registry import DimensionRegistry, DimensionSchema, ValidationResult, get_registry

__all__ = [
    "ConfidenceDimension",
    "DimensionRegistry",
    "DimensionSchema",
    "DimensionalConfidence",
    "ValidationResult",
    "aggregate_confidence",
    "confidence_label",
    "get_registry",
]
