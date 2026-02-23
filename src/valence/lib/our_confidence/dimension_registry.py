"""Dimension schema registry.

Network-maintained registry of recognized dimension schemas, enabling
emergent consensus on what dimensions matter.

Schemas define named sets of dimensions with validation rules and
optional inheritance, allowing domain-specific confidence models to
build on shared foundations.

Example:
    >>> from our_confidence.dimension_registry import get_registry
    >>> registry = get_registry()
    >>> schema = registry.get("v1.confidence.core")
    >>> result = registry.validate("v1.confidence.core", {"source_reliability": 0.8})
    >>> result.valid
    True
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DimensionSchema:
    """A named schema defining a set of recognized dimensions.

    Attributes:
        name: Unique schema identifier (e.g. 'v1.confidence.core').
        dimensions: List of dimension names in this schema.
        required: Dimensions that must be present for validity.
        value_range: Allowed (min, max) for dimension values.
        inherits: Optional parent schema name to inherit from.
        metadata: Arbitrary metadata about the schema.
    """

    name: str
    dimensions: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    value_range: tuple[float, float] = (0.0, 1.0)
    inherits: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate schema constraints."""
        if not self.name:
            raise ValueError("Schema name must not be empty")
        lo, hi = self.value_range
        if lo >= hi:
            raise ValueError(f"value_range lower ({lo}) must be less than upper ({hi})")
        for req in self.required:
            if req not in self.dimensions:
                raise ValueError(f"Required dimension '{req}' not in dimensions list")


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating dimension values against a schema.

    Attributes:
        valid: Whether the dimensions pass all checks.
        errors: List of human-readable error descriptions.
        schema_name: The schema that was validated against.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    schema_name: str = ""


class DimensionRegistry:
    """Thread-safe registry of dimension schemas.

    Provides registration, lookup, inheritance resolution, and
    validation of dimension values against schemas.

    Use :func:`get_registry` to obtain the singleton instance with
    built-in schemas pre-registered.
    """

    def __init__(self) -> None:
        self._schemas: dict[str, DimensionSchema] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, schema: DimensionSchema) -> None:
        """Register a schema.

        Raises:
            ValueError: If the schema's parent (``inherits``) is not
                registered yet.
        """
        with self._lock:
            if schema.inherits and schema.inherits not in self._schemas:
                raise ValueError(f"Parent schema '{schema.inherits}' not registered")
            self._schemas[schema.name] = schema

    def get(self, schema_name: str) -> DimensionSchema | None:
        """Look up a schema by name (``None`` if not found)."""
        with self._lock:
            return self._schemas.get(schema_name)

    def resolve(self, schema_name: str) -> DimensionSchema:
        """Resolve the full inheritance chain and return a flattened schema.

        The returned schema has:
        - ``dimensions``: parent dims + own dims (de-duplicated, order preserved).
        - ``required``: parent required + own required (de-duplicated).
        - ``value_range``: own range (overrides parent).
        - ``metadata``: parent metadata merged with own (own wins).
        - ``inherits``: ``None`` (fully resolved).

        Raises:
            KeyError: If *schema_name* is not registered.
            ValueError: If a circular inheritance chain is detected.
        """
        with self._lock:
            return self._resolve_locked(schema_name)

    def validate(
        self,
        schema_name: str,
        dimensions: dict[str, float],
    ) -> ValidationResult:
        """Validate dimension values against a schema.

        Checks:
        1. Schema exists.
        2. All required dimensions are present.
        3. All values are within ``value_range``.
        4. All dimension keys are recognized by the schema.

        Returns a :class:`ValidationResult` (always; never raises).
        """
        with self._lock:
            schema = self._schemas.get(schema_name)
            if schema is None:
                return ValidationResult(
                    valid=False,
                    errors=[f"Schema '{schema_name}' not found"],
                    schema_name=schema_name,
                )

            resolved = self._resolve_locked(schema_name)
            errors: list[str] = []

            # Required dimensions
            for req in resolved.required:
                if req not in dimensions:
                    errors.append(f"Missing required dimension: {req}")

            lo, hi = resolved.value_range
            allowed = set(resolved.dimensions)

            for dim_name, value in dimensions.items():
                # Range check
                if value < lo or value > hi:
                    errors.append(f"Dimension '{dim_name}' value {value} out of range [{lo}, {hi}]")
                # Recognised check
                if dim_name not in allowed:
                    errors.append(f"Unknown dimension '{dim_name}' for schema '{schema_name}'")

            return ValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                schema_name=schema_name,
            )

    def list_schemas(self) -> list[DimensionSchema]:
        """Return all registered schemas (snapshot, sorted by name)."""
        with self._lock:
            return sorted(self._schemas.values(), key=lambda s: s.name)

    def unregister(self, schema_name: str) -> bool:
        """Remove a schema.  Returns ``True`` if it existed."""
        with self._lock:
            return self._schemas.pop(schema_name, None) is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_locked(self, schema_name: str) -> DimensionSchema:
        """Resolve inheritance while already holding ``_lock``."""
        schema = self._schemas.get(schema_name)
        if schema is None:
            raise KeyError(f"Schema '{schema_name}' not registered")

        if schema.inherits is None:
            return schema

        # Walk the chain, detect cycles
        visited: set[str] = set()
        chain: list[DimensionSchema] = []
        current: DimensionSchema | None = schema

        while current is not None:
            if current.name in visited:
                raise ValueError(f"Circular inheritance detected: {' -> '.join(s.name for s in chain)} -> {current.name}")
            visited.add(current.name)
            chain.append(current)
            if current.inherits:
                current = self._schemas.get(current.inherits)
                if current is None:
                    raise KeyError(f"Parent schema '{chain[-1].inherits}' not registered")
            else:
                current = None

        # chain[0] = leaf, chain[-1] = root.  Merge root-first.
        chain.reverse()
        merged_dims: list[str] = []
        seen_dims: set[str] = set()
        merged_required: list[str] = []
        seen_required: set[str] = set()
        merged_metadata: dict[str, Any] = {}

        for link in chain:
            for d in link.dimensions:
                if d not in seen_dims:
                    merged_dims.append(d)
                    seen_dims.add(d)
            for r in link.required:
                if r not in seen_required:
                    merged_required.append(r)
                    seen_required.add(r)
            merged_metadata.update(link.metadata)

        # Leaf's value_range wins
        leaf = chain[-1]
        return DimensionSchema(
            name=schema_name,
            dimensions=merged_dims,
            required=merged_required,
            value_range=leaf.value_range,
            inherits=None,
            metadata=merged_metadata,
        )


# ======================================================================
# Built-in schemas
# ======================================================================

_BUILTIN_SCHEMAS: list[DimensionSchema] = [
    DimensionSchema(
        name="v1.confidence.core",
        dimensions=[
            "source_reliability",
            "method_quality",
            "internal_consistency",
            "temporal_freshness",
            "corroboration",
            "domain_applicability",
        ],
        required=[
            "source_reliability",
            "method_quality",
            "internal_consistency",
            "temporal_freshness",
            "corroboration",
            "domain_applicability",
        ],
        value_range=(0.0, 1.0),
        metadata={"description": "Core 6-dimensional confidence model"},
    ),
    DimensionSchema(
        name="v1.trust.core",
        dimensions=["conclusions", "reasoning", "perspective"],
        required=["conclusions"],
        value_range=(0.0, 1.0),
        metadata={"description": "Core trust dimensions"},
    ),
    DimensionSchema(
        name="v1.trust.extended",
        dimensions=["honesty", "methodology", "predictive"],
        required=[],
        value_range=(0.0, 1.0),
        inherits="v1.trust.core",
        metadata={"description": "Extended trust dimensions (inherits v1.trust.core)"},
    ),
]

# ======================================================================
# Singleton
# ======================================================================

_registry_instance: DimensionRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> DimensionRegistry:
    """Return the global singleton :class:`DimensionRegistry`.

    Built-in schemas are registered on first call.  Thread-safe.
    """
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                reg = DimensionRegistry()
                for schema in _BUILTIN_SCHEMAS:
                    reg.register(schema)
                _registry_instance = reg
    return _registry_instance


def reset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _registry_instance
    with _registry_lock:
        _registry_instance = None
