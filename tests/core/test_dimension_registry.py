"""Tests for valence.core.dimension_registry module."""

from __future__ import annotations

import threading

import pytest

from valence.core.dimension_registry import (
    DimensionRegistry,
    DimensionSchema,
    ValidationResult,
    get_registry,
    reset_registry,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Reset the global registry before each test."""
    reset_registry()
    yield
    reset_registry()


@pytest.fixture()
def registry() -> DimensionRegistry:
    """Return a fresh, empty registry (not the global singleton)."""
    return DimensionRegistry()


# ============================================================================
# DimensionSchema Tests
# ============================================================================


class TestDimensionSchema:
    """Tests for DimensionSchema dataclass."""

    def test_basic_creation(self):
        schema = DimensionSchema(
            name="test.schema",
            dimensions=["a", "b", "c"],
            required=["a"],
        )
        assert schema.name == "test.schema"
        assert schema.dimensions == ["a", "b", "c"]
        assert schema.required == ["a"]
        assert schema.value_range == (0.0, 1.0)
        assert schema.inherits is None
        assert schema.metadata == {}

    def test_custom_range(self):
        schema = DimensionSchema(
            name="custom.range",
            dimensions=["x"],
            value_range=(-1.0, 1.0),
        )
        assert schema.value_range == (-1.0, 1.0)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name must not be empty"):
            DimensionSchema(name="", dimensions=["a"])

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="lower.*must be less than upper"):
            DimensionSchema(name="bad", dimensions=["a"], value_range=(1.0, 0.0))

    def test_equal_range_raises(self):
        with pytest.raises(ValueError, match="lower.*must be less than upper"):
            DimensionSchema(name="bad", dimensions=["a"], value_range=(0.5, 0.5))

    def test_required_not_in_dimensions_raises(self):
        with pytest.raises(ValueError, match="Required dimension 'z' not in dimensions"):
            DimensionSchema(name="bad", dimensions=["a"], required=["z"])

    def test_frozen(self):
        schema = DimensionSchema(name="frozen", dimensions=["a"])
        with pytest.raises(AttributeError):
            schema.name = "changed"  # type: ignore[misc]

    def test_metadata(self):
        schema = DimensionSchema(
            name="meta",
            dimensions=["a"],
            metadata={"author": "test", "version": 2},
        )
        assert schema.metadata["author"] == "test"


# ============================================================================
# DimensionRegistry — Registration & Lookup
# ============================================================================


class TestRegistryBasics:
    """Basic registration, get, list, unregister."""

    def test_register_and_get(self, registry: DimensionRegistry):
        schema = DimensionSchema(name="s1", dimensions=["a", "b"])
        registry.register(schema)
        assert registry.get("s1") is schema

    def test_get_missing_returns_none(self, registry: DimensionRegistry):
        assert registry.get("nope") is None

    def test_list_schemas_empty(self, registry: DimensionRegistry):
        assert registry.list_schemas() == []

    def test_list_schemas_sorted(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="z", dimensions=["a"]))
        registry.register(DimensionSchema(name="a", dimensions=["b"]))
        names = [s.name for s in registry.list_schemas()]
        assert names == ["a", "z"]

    def test_register_overwrites(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="s", dimensions=["a"]))
        registry.register(DimensionSchema(name="s", dimensions=["b"]))
        assert registry.get("s").dimensions == ["b"]  # type: ignore[union-attr]

    def test_unregister(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="s", dimensions=["a"]))
        assert registry.unregister("s") is True
        assert registry.get("s") is None

    def test_unregister_missing(self, registry: DimensionRegistry):
        assert registry.unregister("nope") is False

    def test_register_with_missing_parent_raises(self, registry: DimensionRegistry):
        schema = DimensionSchema(name="child", dimensions=["x"], inherits="nonexistent")
        with pytest.raises(ValueError, match="Parent schema 'nonexistent' not registered"):
            registry.register(schema)


# ============================================================================
# Inheritance / Resolve
# ============================================================================


class TestRegistryResolve:
    """Tests for DimensionRegistry.resolve()."""

    def test_resolve_no_inheritance(self, registry: DimensionRegistry):
        schema = DimensionSchema(name="flat", dimensions=["a", "b"], required=["a"])
        registry.register(schema)
        resolved = registry.resolve("flat")
        assert resolved.dimensions == ["a", "b"]
        assert resolved.required == ["a"]
        assert resolved.inherits is None

    def test_resolve_single_inheritance(self, registry: DimensionRegistry):
        parent = DimensionSchema(
            name="parent",
            dimensions=["a", "b"],
            required=["a"],
            metadata={"origin": "parent"},
        )
        child = DimensionSchema(
            name="child",
            dimensions=["c"],
            required=["c"],
            inherits="parent",
            metadata={"origin": "child"},
        )
        registry.register(parent)
        registry.register(child)

        resolved = registry.resolve("child")
        assert resolved.dimensions == ["a", "b", "c"]
        assert set(resolved.required) == {"a", "c"}
        assert resolved.inherits is None
        # Child metadata overrides parent
        assert resolved.metadata["origin"] == "child"

    def test_resolve_multi_level_inheritance(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="l0", dimensions=["a"]))
        registry.register(DimensionSchema(name="l1", dimensions=["b"], inherits="l0"))
        registry.register(DimensionSchema(name="l2", dimensions=["c"], inherits="l1"))

        resolved = registry.resolve("l2")
        assert resolved.dimensions == ["a", "b", "c"]

    def test_resolve_deduplicates_dimensions(self, registry: DimensionRegistry):
        parent = DimensionSchema(name="p", dimensions=["a", "b"])
        child = DimensionSchema(name="c", dimensions=["b", "c"], inherits="p")
        registry.register(parent)
        registry.register(child)

        resolved = registry.resolve("c")
        assert resolved.dimensions == ["a", "b", "c"]

    def test_resolve_child_range_wins(self, registry: DimensionRegistry):
        parent = DimensionSchema(name="p", dimensions=["a"], value_range=(0.0, 1.0))
        child = DimensionSchema(name="c", dimensions=["b"], value_range=(-1.0, 1.0), inherits="p")
        registry.register(parent)
        registry.register(child)

        resolved = registry.resolve("c")
        assert resolved.value_range == (-1.0, 1.0)

    def test_resolve_missing_raises(self, registry: DimensionRegistry):
        with pytest.raises(KeyError, match="not registered"):
            registry.resolve("ghost")

    def test_resolve_circular_raises(self, registry: DimensionRegistry):
        """Circular inheritance must be detected."""
        # We need to bypass the register check to create a cycle.
        # Manually insert to simulate a corrupt state.
        s1 = DimensionSchema(name="s1", dimensions=["a"], inherits="s2")
        s2 = DimensionSchema(name="s2", dimensions=["b"], inherits="s1")
        # Force-insert (bypassing parent check)
        registry._schemas["s1"] = s1
        registry._schemas["s2"] = s2

        with pytest.raises(ValueError, match="Circular inheritance"):
            registry.resolve("s1")


# ============================================================================
# Validation
# ============================================================================


class TestRegistryValidation:
    """Tests for DimensionRegistry.validate()."""

    def test_valid_dimensions(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="s", dimensions=["a", "b"], required=["a"]))
        result = registry.validate("s", {"a": 0.5, "b": 0.8})
        assert result.valid is True
        assert result.errors == []
        assert result.schema_name == "s"

    def test_missing_required(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="s", dimensions=["a", "b"], required=["a"]))
        result = registry.validate("s", {"b": 0.5})
        assert result.valid is False
        assert any("Missing required dimension: a" in e for e in result.errors)

    def test_out_of_range(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="s", dimensions=["a"]))
        result = registry.validate("s", {"a": 1.5})
        assert result.valid is False
        assert any("out of range" in e for e in result.errors)

    def test_below_range(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="s", dimensions=["a"]))
        result = registry.validate("s", {"a": -0.1})
        assert result.valid is False

    def test_unknown_dimension(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="s", dimensions=["a"]))
        result = registry.validate("s", {"a": 0.5, "z": 0.3})
        assert result.valid is False
        assert any("Unknown dimension 'z'" in e for e in result.errors)

    def test_unknown_schema(self, registry: DimensionRegistry):
        result = registry.validate("nope", {"a": 0.5})
        assert result.valid is False
        assert any("not found" in e for e in result.errors)

    def test_validation_with_inheritance(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="p", dimensions=["a"], required=["a"]))
        registry.register(DimensionSchema(name="c", dimensions=["b"], inherits="p"))
        # Both parent's 'a' and child's 'b' should be accepted
        result = registry.validate("c", {"a": 0.5, "b": 0.8})
        assert result.valid is True

    def test_validation_inherited_required(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="p", dimensions=["a"], required=["a"]))
        registry.register(DimensionSchema(name="c", dimensions=["b"], inherits="p"))
        # Missing 'a' (required by parent)
        result = registry.validate("c", {"b": 0.8})
        assert result.valid is False
        assert any("Missing required dimension: a" in e for e in result.errors)

    def test_custom_range_validation(self, registry: DimensionRegistry):
        registry.register(
            DimensionSchema(
                name="bipolar",
                dimensions=["sentiment"],
                value_range=(-1.0, 1.0),
            )
        )
        assert registry.validate("bipolar", {"sentiment": -0.5}).valid is True
        assert registry.validate("bipolar", {"sentiment": -1.5}).valid is False

    def test_empty_dimensions_valid(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="s", dimensions=["a", "b"]))
        result = registry.validate("s", {})
        assert result.valid is True  # No required dims, no values to range-check

    def test_multiple_errors(self, registry: DimensionRegistry):
        registry.register(DimensionSchema(name="s", dimensions=["a", "b"], required=["a", "b"]))
        result = registry.validate("s", {"z": 2.0})
        assert result.valid is False
        assert len(result.errors) >= 3  # 2 missing + out of range + unknown


# ============================================================================
# Built-in Schemas via get_registry()
# ============================================================================


class TestBuiltinSchemas:
    """Tests for the built-in schemas registered by get_registry()."""

    def test_confidence_core_exists(self):
        reg = get_registry()
        schema = reg.get("v1.confidence.core")
        assert schema is not None
        assert "source_reliability" in schema.dimensions
        assert "method_quality" in schema.dimensions
        assert "internal_consistency" in schema.dimensions
        assert "temporal_freshness" in schema.dimensions
        assert "corroboration" in schema.dimensions
        assert "domain_applicability" in schema.dimensions
        assert len(schema.dimensions) == 6

    def test_trust_core_exists(self):
        reg = get_registry()
        schema = reg.get("v1.trust.core")
        assert schema is not None
        assert schema.dimensions == ["conclusions", "reasoning", "perspective"]
        assert schema.required == ["conclusions"]

    def test_trust_extended_exists(self):
        reg = get_registry()
        schema = reg.get("v1.trust.extended")
        assert schema is not None
        assert schema.inherits == "v1.trust.core"
        assert "honesty" in schema.dimensions

    def test_trust_extended_resolves(self):
        reg = get_registry()
        resolved = reg.resolve("v1.trust.extended")
        assert "conclusions" in resolved.dimensions  # From parent
        assert "honesty" in resolved.dimensions  # Own
        assert "reasoning" in resolved.dimensions  # From parent
        assert "perspective" in resolved.dimensions  # From parent
        assert "methodology" in resolved.dimensions
        assert "predictive" in resolved.dimensions
        assert len(resolved.dimensions) == 6

    def test_trust_extended_inherits_required(self):
        reg = get_registry()
        resolved = reg.resolve("v1.trust.extended")
        assert "conclusions" in resolved.required

    def test_validate_confidence_core(self):
        reg = get_registry()
        result = reg.validate(
            "v1.confidence.core",
            {
                "source_reliability": 0.8,
                "method_quality": 0.6,
                "internal_consistency": 0.9,
                "temporal_freshness": 0.7,
                "corroboration": 0.5,
                "domain_applicability": 0.4,
            },
        )
        assert result.valid is True

    def test_validate_trust_extended(self):
        reg = get_registry()
        result = reg.validate(
            "v1.trust.extended",
            {"conclusions": 0.8, "honesty": 0.7},
        )
        assert result.valid is True

    def test_validate_trust_extended_missing_required(self):
        reg = get_registry()
        result = reg.validate(
            "v1.trust.extended",
            {"honesty": 0.7},  # Missing 'conclusions'
        )
        assert result.valid is False

    def test_three_builtin_schemas(self):
        reg = get_registry()
        schemas = reg.list_schemas()
        assert len(schemas) == 3
        names = {s.name for s in schemas}
        assert names == {"v1.confidence.core", "v1.trust.core", "v1.trust.extended"}


# ============================================================================
# Singleton / Thread Safety
# ============================================================================


class TestSingleton:
    """Tests for the singleton pattern."""

    def test_get_registry_returns_same_instance(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_reset_registry_clears(self):
        r1 = get_registry()
        reset_registry()
        r2 = get_registry()
        assert r1 is not r2

    def test_concurrent_access(self):
        """Registry should be safe under concurrent access."""
        reg = get_registry()
        errors: list[Exception] = []

        def register_many(prefix: str):
            try:
                for i in range(50):
                    reg.register(
                        DimensionSchema(
                            name=f"{prefix}.{i}",
                            dimensions=[f"dim_{i}"],
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register_many, args=(f"t{t}",)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # 3 builtins + 4 threads * 50 = 203
        assert len(reg.list_schemas()) == 203


# ============================================================================
# ValidationResult
# ============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_defaults(self):
        r = ValidationResult(valid=True)
        assert r.valid is True
        assert r.errors == []
        assert r.schema_name == ""

    def test_with_errors(self):
        r = ValidationResult(
            valid=False,
            errors=["err1", "err2"],
            schema_name="test",
        )
        assert r.valid is False
        assert len(r.errors) == 2
