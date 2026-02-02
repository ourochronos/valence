"""Tests for valence.core.exceptions module."""

from __future__ import annotations

import pytest

from valence.core.exceptions import (
    ConfigException,
    ConflictError,
    DatabaseException,
    EmbeddingException,
    MCPException,
    NotFoundError,
    ValidationException,
    ValenceException,
)


# ============================================================================
# ValenceException Tests
# ============================================================================

class TestValenceException:
    """Tests for base ValenceException."""

    def test_create_with_message(self):
        """Create exception with just message."""
        exc = ValenceException("Something went wrong")
        assert str(exc) == "Something went wrong"
        assert exc.message == "Something went wrong"
        assert exc.details == {}

    def test_create_with_details(self):
        """Create exception with message and details."""
        details = {"key": "value", "count": 42}
        exc = ValenceException("Error occurred", details=details)
        assert exc.message == "Error occurred"
        assert exc.details == details

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        exc = ValenceException("Test error", details={"info": "extra"})
        d = exc.to_dict()
        assert d["error"] == "ValenceException"
        assert d["message"] == "Test error"
        assert d["details"] == {"info": "extra"}

    def test_to_dict_class_name(self):
        """to_dict should use actual class name."""
        exc = DatabaseException("DB error")
        d = exc.to_dict()
        assert d["error"] == "DatabaseException"

    def test_inherits_from_exception(self):
        """Should inherit from Exception."""
        exc = ValenceException("Test")
        assert isinstance(exc, Exception)

    def test_can_be_raised_and_caught(self):
        """Should be raisable and catchable."""
        with pytest.raises(ValenceException) as exc_info:
            raise ValenceException("Test raise")
        assert exc_info.value.message == "Test raise"


# ============================================================================
# DatabaseException Tests
# ============================================================================

class TestDatabaseException:
    """Tests for DatabaseException."""

    def test_inherits_from_valence_exception(self):
        """Should inherit from ValenceException."""
        exc = DatabaseException("DB error")
        assert isinstance(exc, ValenceException)

    def test_create_with_message(self):
        """Create with message."""
        exc = DatabaseException("Connection failed")
        assert exc.message == "Connection failed"

    def test_to_dict(self):
        """to_dict should include class name."""
        exc = DatabaseException("Query failed")
        d = exc.to_dict()
        assert d["error"] == "DatabaseException"
        assert d["message"] == "Query failed"


# ============================================================================
# ValidationException Tests
# ============================================================================

class TestValidationException:
    """Tests for ValidationException."""

    def test_create_with_message_only(self):
        """Create with just message."""
        exc = ValidationException("Invalid input")
        assert exc.message == "Invalid input"
        assert exc.field is None
        assert exc.value is None

    def test_create_with_field(self):
        """Create with field name."""
        exc = ValidationException("Field invalid", field="username")
        assert exc.field == "username"
        assert "field" in exc.details
        assert exc.details["field"] == "username"

    def test_create_with_field_and_value(self):
        """Create with field and value."""
        exc = ValidationException("Out of range", field="age", value=150)
        assert exc.field == "age"
        assert exc.value == 150
        assert exc.details["field"] == "age"
        assert exc.details["value"] == "150"

    def test_value_converted_to_string(self):
        """Value should be converted to string in details."""
        exc = ValidationException("Error", field="count", value=42)
        assert exc.details["value"] == "42"

    def test_to_dict(self):
        """to_dict should include field info."""
        exc = ValidationException("Bad value", field="email", value="invalid")
        d = exc.to_dict()
        assert d["error"] == "ValidationException"
        assert d["details"]["field"] == "email"
        assert d["details"]["value"] == "invalid"


# ============================================================================
# ConfigException Tests
# ============================================================================

class TestConfigException:
    """Tests for ConfigException."""

    def test_create_with_message_only(self):
        """Create with just message."""
        exc = ConfigException("Config error")
        assert exc.message == "Config error"
        assert exc.missing_vars == []

    def test_create_with_missing_vars(self):
        """Create with missing variables list."""
        missing = ["VAR1", "VAR2", "VAR3"]
        exc = ConfigException("Missing vars", missing_vars=missing)
        assert exc.missing_vars == missing
        assert exc.details["missing_vars"] == missing

    def test_empty_missing_vars_list(self):
        """Empty missing vars list should work."""
        exc = ConfigException("Error", missing_vars=[])
        assert exc.missing_vars == []
        # Empty list should not be in details
        assert "missing_vars" not in exc.details

    def test_to_dict(self):
        """to_dict should include missing vars."""
        exc = ConfigException("Config error", missing_vars=["DB_HOST"])
        d = exc.to_dict()
        assert d["error"] == "ConfigException"
        assert d["details"]["missing_vars"] == ["DB_HOST"]


# ============================================================================
# NotFoundError Tests
# ============================================================================

class TestNotFoundError:
    """Tests for NotFoundError."""

    def test_create(self):
        """Create with resource type and ID."""
        exc = NotFoundError("Belief", "abc-123")
        assert exc.resource_type == "Belief"
        assert exc.resource_id == "abc-123"
        assert exc.message == "Belief not found: abc-123"

    def test_details_include_resource_info(self):
        """Details should include resource info."""
        exc = NotFoundError("Entity", "xyz-789")
        assert exc.details["resource_type"] == "Entity"
        assert exc.details["resource_id"] == "xyz-789"

    def test_to_dict(self):
        """to_dict should include resource info."""
        exc = NotFoundError("Session", "session-1")
        d = exc.to_dict()
        assert d["error"] == "NotFoundError"
        assert d["details"]["resource_type"] == "Session"
        assert d["details"]["resource_id"] == "session-1"


# ============================================================================
# ConflictError Tests
# ============================================================================

class TestConflictError:
    """Tests for ConflictError."""

    def test_create_with_message_only(self):
        """Create with just message."""
        exc = ConflictError("Resource already exists")
        assert exc.message == "Resource already exists"
        assert exc.existing_id is None

    def test_create_with_existing_id(self):
        """Create with existing resource ID."""
        exc = ConflictError("Duplicate entry", existing_id="existing-123")
        assert exc.existing_id == "existing-123"
        assert exc.details["existing_id"] == "existing-123"

    def test_to_dict(self):
        """to_dict should include existing ID."""
        exc = ConflictError("Conflict", existing_id="id-456")
        d = exc.to_dict()
        assert d["error"] == "ConflictError"
        assert d["details"]["existing_id"] == "id-456"


# ============================================================================
# EmbeddingException Tests
# ============================================================================

class TestEmbeddingException:
    """Tests for EmbeddingException."""

    def test_create_with_message_only(self):
        """Create with just message."""
        exc = EmbeddingException("Embedding failed")
        assert exc.message == "Embedding failed"
        assert exc.provider is None

    def test_create_with_provider(self):
        """Create with provider name."""
        exc = EmbeddingException("API error", provider="openai")
        assert exc.provider == "openai"
        assert exc.details["provider"] == "openai"

    def test_to_dict(self):
        """to_dict should include provider."""
        exc = EmbeddingException("Timeout", provider="cohere")
        d = exc.to_dict()
        assert d["error"] == "EmbeddingException"
        assert d["details"]["provider"] == "cohere"


# ============================================================================
# MCPException Tests
# ============================================================================

class TestMCPException:
    """Tests for MCPException."""

    def test_create_with_message_only(self):
        """Create with just message."""
        exc = MCPException("MCP error")
        assert exc.message == "MCP error"
        assert exc.tool_name is None

    def test_create_with_tool_name(self):
        """Create with tool name."""
        exc = MCPException("Tool failed", tool_name="belief_query")
        assert exc.tool_name == "belief_query"
        assert exc.details["tool_name"] == "belief_query"

    def test_to_dict(self):
        """to_dict should include tool name."""
        exc = MCPException("Execution error", tool_name="entity_search")
        d = exc.to_dict()
        assert d["error"] == "MCPException"
        assert d["details"]["tool_name"] == "entity_search"


# ============================================================================
# Exception Hierarchy Tests
# ============================================================================

class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_all_inherit_from_valence_exception(self):
        """All custom exceptions should inherit from ValenceException."""
        exceptions = [
            DatabaseException("test"),
            ValidationException("test"),
            ConfigException("test"),
            NotFoundError("Type", "id"),
            ConflictError("test"),
            EmbeddingException("test"),
            MCPException("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, ValenceException)

    def test_can_catch_by_base_class(self):
        """Should be able to catch all by ValenceException."""
        exceptions = [
            DatabaseException,
            ValidationException,
            ConfigException,
            EmbeddingException,
            MCPException,
        ]
        for exc_class in exceptions:
            with pytest.raises(ValenceException):
                raise exc_class("test error")

    def test_specific_catch_works(self):
        """Should be able to catch specific exception types."""
        with pytest.raises(DatabaseException):
            raise DatabaseException("DB error")

        # But not catch unrelated types
        with pytest.raises(DatabaseException):
            try:
                raise DatabaseException("DB error")
            except ValidationException:
                pytest.fail("Should not catch ValidationException")


# ============================================================================
# Serialization Tests
# ============================================================================

class TestExceptionSerialization:
    """Tests for exception serialization."""

    def test_to_dict_always_has_required_fields(self):
        """All exceptions should have error, message, details in to_dict."""
        exceptions = [
            ValenceException("test"),
            DatabaseException("test"),
            ValidationException("test"),
            ConfigException("test"),
            NotFoundError("Type", "id"),
            ConflictError("test"),
            EmbeddingException("test"),
            MCPException("test"),
        ]
        for exc in exceptions:
            d = exc.to_dict()
            assert "error" in d
            assert "message" in d
            assert "details" in d

    def test_to_dict_json_serializable(self):
        """to_dict output should be JSON serializable."""
        import json
        exc = ValidationException("Bad input", field="email", value={"nested": "data"})
        d = exc.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert json_str is not None
