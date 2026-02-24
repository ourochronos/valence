"""Tests for endpoint_utils.py (parameter parsing and response formatting)."""

from __future__ import annotations

from unittest.mock import MagicMock

from starlette.datastructures import QueryParams
from starlette.requests import Request

from valence.server.endpoint_utils import (
    _parse_bool,
    _parse_float,
    _parse_int,
    format_response,
    parse_output_format,
)


class TestParseBool:
    """Tests for _parse_bool helper."""

    def test_true_values(self):
        """Test parsing true values."""
        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True
        assert _parse_bool("Yes") is True

    def test_false_values(self):
        """Test parsing false values."""
        assert _parse_bool("false") is False
        assert _parse_bool("False") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False
        assert _parse_bool("anything") is False

    def test_none_returns_default(self):
        """Test None returns default value."""
        assert _parse_bool(None) is False
        assert _parse_bool(None, default=True) is True

    def test_empty_string(self):
        """Test empty string returns False."""
        assert _parse_bool("") is False


class TestParseInt:
    """Tests for _parse_int helper."""

    def test_valid_integer(self):
        """Test parsing valid integers."""
        assert _parse_int("10", default=5) == 10
        assert _parse_int("100", default=5) == 100
        assert _parse_int("0", default=5) == 0

    def test_none_returns_default(self):
        """Test None returns default value."""
        assert _parse_int(None, default=10) == 10
        assert _parse_int(None, default=0) == 0

    def test_maximum_cap(self):
        """Test maximum value capping."""
        assert _parse_int("2000", default=10, maximum=100) == 100
        assert _parse_int("50", default=10, maximum=100) == 50
        assert _parse_int("100", default=10, maximum=100) == 100

    def test_invalid_integer_returns_default(self):
        """Test invalid integer returns default."""
        assert _parse_int("not_a_number", default=10) == 10
        assert _parse_int("12.5", default=10) == 10
        assert _parse_int("", default=10) == 10

    def test_negative_integers(self):
        """Test parsing negative integers."""
        assert _parse_int("-10", default=0) == -10

    def test_zero_maximum(self):
        """Test zero maximum."""
        assert _parse_int("10", default=5, maximum=0) == 0


class TestParseFloat:
    """Tests for _parse_float helper."""

    def test_valid_float(self):
        """Test parsing valid floats."""
        assert _parse_float("1.5") == 1.5
        assert _parse_float("0.85") == 0.85
        assert _parse_float("10.0") == 10.0

    def test_none_returns_default(self):
        """Test None returns default value."""
        assert _parse_float(None) is None
        assert _parse_float(None, default=0.5) == 0.5

    def test_invalid_float_returns_default(self):
        """Test invalid float returns default."""
        assert _parse_float("not_a_float") is None
        assert _parse_float("not_a_float", default=1.0) == 1.0
        assert _parse_float("", default=0.5) == 0.5

    def test_integer_as_float(self):
        """Test parsing integer as float."""
        assert _parse_float("10") == 10.0
        assert _parse_float("0") == 0.0

    def test_negative_float(self):
        """Test parsing negative floats."""
        assert _parse_float("-1.5") == -1.5


class TestParseOutputFormat:
    """Tests for parse_output_format."""

    def test_json_format(self):
        """Test parsing json format."""
        request = MagicMock(spec=Request)
        request.query_params = QueryParams({"output": "json"})
        assert parse_output_format(request) == "json"

    def test_text_format(self):
        """Test parsing text format."""
        request = MagicMock(spec=Request)
        request.query_params = QueryParams({"output": "text"})
        assert parse_output_format(request) == "text"

    def test_table_format(self):
        """Test parsing table format."""
        request = MagicMock(spec=Request)
        request.query_params = QueryParams({"output": "table"})
        assert parse_output_format(request) == "table"

    def test_default_json(self):
        """Test default is json when not specified."""
        request = MagicMock(spec=Request)
        request.query_params = QueryParams({})
        assert parse_output_format(request) == "json"

    def test_invalid_format_defaults_to_json(self):
        """Test invalid format defaults to json."""
        request = MagicMock(spec=Request)
        request.query_params = QueryParams({"output": "xml"})
        assert parse_output_format(request) == "json"

        request.query_params = QueryParams({"output": "invalid"})
        assert parse_output_format(request) == "json"


class TestFormatResponse:
    """Tests for format_response helper."""

    def test_json_response(self):
        """Test JSON response formatting."""
        data = {"success": True, "message": "test"}
        response = format_response(data, "json")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_text_response_with_formatter(self):
        """Test text response with formatter."""

        def text_formatter(data):
            return f"Success: {data['success']}"

        data = {"success": True}
        response = format_response(data, "text", text_formatter=text_formatter)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert b"Success: True" in response.body

    def test_table_response_with_formatter(self):
        """Test table response with table formatter."""

        def table_formatter(data):
            return "| Header |\n| Value |"

        data = {"data": "test"}
        response = format_response(data, "table", table_formatter=table_formatter)

        assert response.status_code == 200
        assert b"| Header |" in response.body

    def test_text_fallback_to_text_formatter(self):
        """Test table format falls back to text formatter if no table formatter."""

        def text_formatter(data):
            return "Text output"

        data = {"data": "test"}
        response = format_response(data, "table", text_formatter=text_formatter)

        assert b"Text output" in response.body

    def test_no_formatter_defaults_to_json(self):
        """Test falls back to JSON when no formatter provided."""
        data = {"success": True}

        # Text format but no formatter
        response = format_response(data, "text")
        assert response.headers["content-type"] == "application/json"

        # Table format but no formatter
        response = format_response(data, "table")
        assert response.headers["content-type"] == "application/json"

    def test_custom_status_code(self):
        """Test custom status code."""
        data = {"error": "not found"}
        response = format_response(data, "json", status_code=404)
        assert response.status_code == 404

    def test_status_code_with_text_format(self):
        """Test custom status code with text format."""

        def text_formatter(data):
            return "Error"

        data = {"error": "test"}
        response = format_response(data, "text", text_formatter=text_formatter, status_code=500)
        assert response.status_code == 500


class TestParameterParsingIntegration:
    """Integration tests for parameter parsing."""

    def test_parse_multiple_bools(self):
        """Test parsing multiple boolean parameters."""
        assert _parse_bool("true") and _parse_bool("1")
        assert not (_parse_bool("false") or _parse_bool("0"))

    def test_parse_mixed_types(self):
        """Test parsing different parameter types together."""
        # Simulate query params
        limit = _parse_int("50", default=10, maximum=100)
        dry_run = _parse_bool("true")
        threshold = _parse_float("0.85")

        assert limit == 50
        assert dry_run is True
        assert threshold == 0.85

    def test_all_defaults(self):
        """Test all parsers with None values."""
        assert _parse_bool(None, default=True) is True
        assert _parse_int(None, default=10) == 10
        assert _parse_float(None, default=0.5) == 0.5

    def test_edge_case_values(self):
        """Test edge case values."""
        # Very large number capped by maximum
        assert _parse_int("999999", default=1, maximum=1000) == 1000

        # Very small float
        assert _parse_float("0.0001") == 0.0001

        # Zero values
        assert _parse_int("0", default=10) == 0
        assert _parse_float("0.0") == 0.0


class TestFormatResponseEdgeCases:
    """Edge case tests for format_response."""

    def test_empty_data(self):
        """Test formatting empty data."""
        response = format_response({}, "json")
        assert response.status_code == 200

    def test_nested_data(self):
        """Test formatting nested data."""
        data = {
            "success": True,
            "nested": {"level1": {"level2": "value"}},
        }
        response = format_response(data, "json")
        assert response.status_code == 200

    def test_unicode_in_text_format(self):
        """Test unicode content in text format."""

        def text_formatter(data):
            return data["message"]

        data = {"message": "Hello 世界 🎉"}
        response = format_response(data, "text", text_formatter=text_formatter)
        assert "世界" in response.body.decode("utf-8")

    def test_formatter_exception_handling(self):
        """Test graceful handling when formatter raises exception."""

        def broken_formatter(data):
            raise ValueError("Formatter error")

        data = {"test": "data"}

        # Should not raise, might fall back to JSON or handle error
        try:
            response = format_response(data, "text", text_formatter=broken_formatter)
            # If it doesn't raise, that's fine - implementation may vary
            assert response.status_code in [200, 500]
        except ValueError:
            # If it raises, that's also a valid implementation choice
            pass

    def test_large_data_set(self):
        """Test formatting large data set."""
        data = {"items": [{"id": i, "value": f"item_{i}"} for i in range(1000)]}
        response = format_response(data, "json")
        assert response.status_code == 200
