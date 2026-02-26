"""Tests for MCP handler utilities."""

from __future__ import annotations

import asyncio
import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

from valence.mcp.handlers._utils import run_async, validate_enum


class TestRunAsync:
    """Tests for run_async utility."""

    def test_run_async_no_loop(self):
        """Test run_async when no event loop exists."""

        async def sample_coro():
            return "result"

        # Create coroutine and ensure it's closed to avoid RuntimeWarning
        coro = sample_coro()
        try:
            with patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")):
                with patch("asyncio.run", return_value="result") as mock_run:
                    result = run_async(coro)
        finally:
            # Close the coroutine to prevent RuntimeWarning about unawaited coroutine
            coro.close()

        assert result == "result"
        mock_run.assert_called_once()

    def test_run_async_existing_not_running(self):
        """Test run_async with existing but not running loop."""

        async def sample_coro():
            return "result"

        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.return_value = "result"

        # Create coroutine and ensure it's closed to avoid RuntimeWarning
        coro = sample_coro()
        try:
            with patch("asyncio.get_event_loop", return_value=mock_loop):
                result = run_async(coro)
        finally:
            # Close the coroutine to prevent RuntimeWarning about unawaited coroutine
            coro.close()

        assert result == "result"
        mock_loop.run_until_complete.assert_called_once()

    def test_run_async_running_loop(self):
        """Test run_async when loop is already running."""

        async def sample_coro():
            return "result"

        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True

        mock_future = MagicMock()
        mock_future.result.return_value = "result"

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.submit.return_value = mock_future

        # Create coroutine and ensure it's closed to avoid RuntimeWarning
        coro = sample_coro()
        try:
            with patch("asyncio.get_event_loop", return_value=mock_loop):
                with patch("concurrent.futures.ThreadPoolExecutor", return_value=mock_pool):
                    result = run_async(coro)
        finally:
            # Close the coroutine to prevent RuntimeWarning about unawaited coroutine
            coro.close()

        assert result == "result"
        mock_pool.submit.assert_called_once()
        mock_future.result.assert_called_once_with(timeout=600)

    def test_run_async_timeout(self):
        """Test run_async with timeout."""

        async def slow_coro():
            await asyncio.sleep(100)
            return "result"

        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True

        mock_future = MagicMock()
        mock_future.result.side_effect = concurrent.futures.TimeoutError()

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.submit.return_value = mock_future

        # Create coroutine and ensure it's closed to avoid RuntimeWarning
        coro = slow_coro()
        try:
            with patch("asyncio.get_event_loop", return_value=mock_loop):
                with patch("concurrent.futures.ThreadPoolExecutor", return_value=mock_pool):
                    with pytest.raises(concurrent.futures.TimeoutError):
                        run_async(coro)
        finally:
            # Close the coroutine to prevent RuntimeWarning about unawaited coroutine
            coro.close()

    def test_run_async_exception_in_coro(self):
        """Test run_async when coroutine raises exception."""

        async def failing_coro():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            run_async(failing_coro())

    def test_run_async_with_return_value(self):
        """Test run_async with various return values."""

        async def coro_with_dict():
            return {"key": "value"}

        result = run_async(coro_with_dict())
        assert result == {"key": "value"}

        async def coro_with_list():
            return [1, 2, 3]

        result = run_async(coro_with_list())
        assert result == [1, 2, 3]

        async def coro_with_none():
            return None

        result = run_async(coro_with_none())
        assert result is None


class TestValidateEnum:
    """Tests for validate_enum utility."""

    def test_validate_enum_valid_value(self):
        """Test validate_enum with valid value."""
        result = validate_enum("apple", ["apple", "banana", "cherry"], "fruit")

        assert result is None

    def test_validate_enum_invalid_value(self):
        """Test validate_enum with invalid value."""
        result = validate_enum("grape", ["apple", "banana", "cherry"], "fruit")

        assert result is not None
        assert result["success"] is False
        assert "Invalid fruit" in result["error"]
        assert "grape" in result["error"]
        assert "apple" in result["error"]

    def test_validate_enum_empty_list(self):
        """Test validate_enum with empty valid_values."""
        result = validate_enum("value", [], "param")

        assert result is not None
        assert result["success"] is False

    def test_validate_enum_case_sensitive(self):
        """Test validate_enum is case-sensitive."""
        result = validate_enum("Apple", ["apple", "banana"], "fruit")

        assert result is not None
        assert result["success"] is False

    def test_validate_enum_single_valid_value(self):
        """Test validate_enum with single valid value."""
        result = validate_enum("only", ["only"], "param")

        assert result is None

    def test_validate_enum_numeric_values(self):
        """Test validate_enum with numeric string values."""
        result = validate_enum("1", ["1", "2", "3"], "number")

        assert result is None

        result = validate_enum("4", ["1", "2", "3"], "number")
        assert result is not None
        assert result["success"] is False

    def test_validate_enum_error_format(self):
        """Test validate_enum error message format."""
        result = validate_enum("invalid", ["valid1", "valid2"], "type")

        assert result["success"] is False
        assert "Invalid type 'invalid'" in result["error"]
        assert "Must be one of:" in result["error"]
        assert "valid1" in result["error"]
        assert "valid2" in result["error"]

    def test_validate_enum_special_characters(self):
        """Test validate_enum with special characters."""
        result = validate_enum("foo-bar", ["foo-bar", "baz_qux"], "identifier")

        assert result is None

    def test_validate_enum_whitespace(self):
        """Test validate_enum with whitespace in values."""
        result = validate_enum("hello world", ["hello world", "goodbye"], "greeting")

        assert result is None

        result = validate_enum("hello", ["hello world"], "greeting")
        assert result is not None
