"""Tests for valence.core.response module - ValenceResponse dataclass and helpers.

Tests cover:
- ValenceResponse dataclass
- to_dict() serialization
- ok() helper
- err() helper
- degraded flag handling
"""

from __future__ import annotations


class TestValenceResponse:
    """Test ValenceResponse dataclass."""

    def test_success_response(self):
        """Test creating a successful response."""
        from valence.core.response import ValenceResponse

        resp = ValenceResponse(success=True, data={"id": "123"})

        assert resp.success is True
        assert resp.data == {"id": "123"}
        assert resp.error is None
        assert resp.degraded is False

    def test_error_response(self):
        """Test creating an error response."""
        from valence.core.response import ValenceResponse

        resp = ValenceResponse(success=False, error="Something went wrong")

        assert resp.success is False
        assert resp.data is None
        assert resp.error == "Something went wrong"
        assert resp.degraded is False

    def test_degraded_response(self):
        """Test creating a degraded response."""
        from valence.core.response import ValenceResponse

        resp = ValenceResponse(success=True, data={"result": "ok"}, degraded=True)

        assert resp.success is True
        assert resp.data == {"result": "ok"}
        assert resp.degraded is True
        assert resp.error is None

    def test_defaults(self):
        """Test default values."""
        from valence.core.response import ValenceResponse

        resp = ValenceResponse(success=True)

        assert resp.success is True
        assert resp.data is None
        assert resp.error is None
        assert resp.degraded is False


class TestToDict:
    """Test to_dict() serialization."""

    def test_to_dict_success_with_data(self):
        """Test to_dict includes data when present."""
        from valence.core.response import ValenceResponse

        resp = ValenceResponse(success=True, data={"id": "abc123"})
        result = resp.to_dict()

        assert result == {"success": True, "data": {"id": "abc123"}}

    def test_to_dict_success_without_data(self):
        """Test to_dict omits None data."""
        from valence.core.response import ValenceResponse

        resp = ValenceResponse(success=True, data=None)
        result = resp.to_dict()

        assert result == {"success": True}
        assert "data" not in result

    def test_to_dict_error(self):
        """Test to_dict includes error when present."""
        from valence.core.response import ValenceResponse

        resp = ValenceResponse(success=False, error="Invalid input")
        result = resp.to_dict()

        assert result == {"success": False, "error": "Invalid input"}
        assert "data" not in result

    def test_to_dict_degraded(self):
        """Test to_dict includes degraded flag when True."""
        from valence.core.response import ValenceResponse

        resp = ValenceResponse(success=True, data={"msg": "ok"}, degraded=True)
        result = resp.to_dict()

        assert result == {"success": True, "data": {"msg": "ok"}, "degraded": True}

    def test_to_dict_not_degraded(self):
        """Test to_dict omits degraded when False."""
        from valence.core.response import ValenceResponse

        resp = ValenceResponse(success=True, data={"msg": "ok"}, degraded=False)
        result = resp.to_dict()

        assert result == {"success": True, "data": {"msg": "ok"}}
        assert "degraded" not in result

    def test_to_dict_empty_error(self):
        """Test to_dict omits empty error string."""
        from valence.core.response import ValenceResponse

        resp = ValenceResponse(success=False, error="")
        result = resp.to_dict()

        assert result == {"success": False}
        assert "error" not in result

    def test_to_dict_complex_data(self):
        """Test to_dict with complex nested data."""
        from valence.core.response import ValenceResponse

        data = {
            "articles": [
                {"id": "1", "title": "First"},
                {"id": "2", "title": "Second"},
            ],
            "metadata": {"count": 2, "page": 1},
        }
        resp = ValenceResponse(success=True, data=data)
        result = resp.to_dict()

        assert result == {"success": True, "data": data}


class TestOkHelper:
    """Test ok() helper function."""

    def test_ok_with_data(self):
        """Test ok() creates success response with data."""
        from valence.core.response import ok

        resp = ok(data={"id": "123", "name": "test"})

        assert resp.success is True
        assert resp.data == {"id": "123", "name": "test"}
        assert resp.error is None
        assert resp.degraded is False

    def test_ok_without_data(self):
        """Test ok() creates success response without data."""
        from valence.core.response import ok

        resp = ok()

        assert resp.success is True
        assert resp.data is None
        assert resp.error is None
        assert resp.degraded is False

    def test_ok_with_degraded(self):
        """Test ok() creates degraded response."""
        from valence.core.response import ok

        resp = ok(data={"result": "fallback used"}, degraded=True)

        assert resp.success is True
        assert resp.data == {"result": "fallback used"}
        assert resp.degraded is True
        assert resp.error is None

    def test_ok_with_none_data(self):
        """Test ok() with explicit None data."""
        from valence.core.response import ok

        resp = ok(data=None)

        assert resp.success is True
        assert resp.data is None

    def test_ok_with_list_data(self):
        """Test ok() with list data."""
        from valence.core.response import ok

        data = [1, 2, 3, 4, 5]
        resp = ok(data=data)

        assert resp.success is True
        assert resp.data == data

    def test_ok_with_string_data(self):
        """Test ok() with string data."""
        from valence.core.response import ok

        resp = ok(data="operation completed")

        assert resp.success is True
        assert resp.data == "operation completed"

    def test_ok_with_zero_data(self):
        """Test ok() with zero (falsy but valid data)."""
        from valence.core.response import ok

        resp = ok(data=0)

        assert resp.success is True
        assert resp.data == 0

    def test_ok_with_empty_list(self):
        """Test ok() with empty list (falsy but valid data)."""
        from valence.core.response import ok

        resp = ok(data=[])

        assert resp.success is True
        assert resp.data == []


class TestErrHelper:
    """Test err() helper function."""

    def test_err_basic(self):
        """Test err() creates error response."""
        from valence.core.response import err

        resp = err("Something went wrong")

        assert resp.success is False
        assert resp.error == "Something went wrong"
        assert resp.data is None
        assert resp.degraded is False

    def test_err_detailed_message(self):
        """Test err() with detailed error message."""
        from valence.core.response import err

        resp = err("Database connection failed: timeout after 30s")

        assert resp.success is False
        assert resp.error == "Database connection failed: timeout after 30s"

    def test_err_short_message(self):
        """Test err() with short error message."""
        from valence.core.response import err

        resp = err("Invalid")

        assert resp.success is False
        assert resp.error == "Invalid"

    def test_err_empty_string(self):
        """Test err() with empty error string."""
        from valence.core.response import err

        resp = err("")

        assert resp.success is False
        assert resp.error == ""


class TestResponseIntegration:
    """Integration tests for response patterns."""

    def test_success_to_dict_roundtrip(self):
        """Test success response to_dict matches expected format."""
        from valence.core.response import ok

        resp = ok(data={"article_id": "abc", "created": True})
        d = resp.to_dict()

        assert d["success"] is True
        assert d["data"]["article_id"] == "abc"
        assert "error" not in d
        assert "degraded" not in d

    def test_error_to_dict_roundtrip(self):
        """Test error response to_dict matches expected format."""
        from valence.core.response import err

        resp = err("Validation failed: missing title")
        d = resp.to_dict()

        assert d["success"] is False
        assert d["error"] == "Validation failed: missing title"
        assert "data" not in d
        assert "degraded" not in d

    def test_degraded_to_dict_roundtrip(self):
        """Test degraded response to_dict matches expected format."""
        from valence.core.response import ok

        resp = ok(data={"content": "fallback"}, degraded=True)
        d = resp.to_dict()

        assert d["success"] is True
        assert d["data"]["content"] == "fallback"
        assert d["degraded"] is True
        assert "error" not in d

    def test_void_operation_pattern(self):
        """Test pattern for void operations (no meaningful return data)."""
        from valence.core.response import ok

        # Void operation (e.g., delete)
        resp = ok()
        d = resp.to_dict()

        assert d == {"success": True}

    def test_list_result_pattern(self):
        """Test pattern for list results."""
        from valence.core.response import ok

        items = [{"id": "1"}, {"id": "2"}]
        resp = ok(data=items)
        d = resp.to_dict()

        assert d["success"] is True
        assert len(d["data"]) == 2

    def test_error_with_context_pattern(self):
        """Test error with context information."""
        from valence.core.response import err

        resp = err("Article not found: abc-123")
        d = resp.to_dict()

        assert d["success"] is False
        assert "abc-123" in d["error"]
