"""Tests for valence.core.logging module."""

from __future__ import annotations

import json
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Correlation ID Tests
# ============================================================================


class TestCorrelationId:
    """Tests for correlation ID functionality."""

    def test_get_correlation_id_default_none(self):
        """Should return None when no correlation ID is set."""
        from valence.core.logging import get_correlation_id, set_correlation_id

        # Ensure clean state
        set_correlation_id(None)
        assert get_correlation_id() is None

    def test_set_and_get_correlation_id(self):
        """Should set and retrieve correlation ID."""
        from valence.core.logging import get_correlation_id, set_correlation_id

        set_correlation_id("test-correlation-123")
        assert get_correlation_id() == "test-correlation-123"

        # Clean up
        set_correlation_id(None)

    def test_generate_correlation_id(self):
        """Should generate unique correlation IDs."""
        from valence.core.logging import generate_correlation_id

        id1 = generate_correlation_id()
        id2 = generate_correlation_id()

        assert id1 != id2
        assert len(id1) == 36  # UUID format
        assert "-" in id1

    def test_correlation_context_generates_id(self):
        """Context manager should generate ID if not provided."""
        from valence.core.logging import (
            correlation_context,
            get_correlation_id,
            set_correlation_id,
        )

        set_correlation_id(None)

        with correlation_context() as cid:
            assert cid is not None
            assert len(cid) == 36
            assert get_correlation_id() == cid

        # Should be reset after context
        assert get_correlation_id() is None

    def test_correlation_context_uses_provided_id(self):
        """Context manager should use provided ID."""
        from valence.core.logging import (
            correlation_context,
            get_correlation_id,
            set_correlation_id,
        )

        set_correlation_id(None)

        with correlation_context("my-custom-id") as cid:
            assert cid == "my-custom-id"
            assert get_correlation_id() == "my-custom-id"

        assert get_correlation_id() is None

    def test_correlation_context_restores_previous_id(self):
        """Context manager should restore previous ID on exit."""
        from valence.core.logging import (
            correlation_context,
            get_correlation_id,
            set_correlation_id,
        )

        set_correlation_id("outer-id")

        with correlation_context("inner-id"):
            assert get_correlation_id() == "inner-id"

        assert get_correlation_id() == "outer-id"

        # Clean up
        set_correlation_id(None)

    def test_nested_correlation_contexts(self):
        """Should handle nested correlation contexts correctly."""
        from valence.core.logging import (
            correlation_context,
            get_correlation_id,
            set_correlation_id,
        )

        set_correlation_id(None)

        with correlation_context("outer"):
            assert get_correlation_id() == "outer"

            with correlation_context("inner"):
                assert get_correlation_id() == "inner"

            assert get_correlation_id() == "outer"

        assert get_correlation_id() is None


# ============================================================================
# JSONFormatter Tests
# ============================================================================


class TestJSONFormatter:
    """Tests for JSONFormatter class."""

    def test_format_basic_message(self):
        """Should format basic log message as JSON."""
        from valence.core.logging import JSONFormatter, set_correlation_id

        set_correlation_id(None)
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "timestamp" in data
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"

    def test_format_includes_correlation_id(self):
        """Should include correlation ID when present."""
        from valence.core.logging import JSONFormatter, set_correlation_id

        set_correlation_id("test-correlation-abc")
        try:
            formatter = JSONFormatter()
            record = logging.LogRecord(
                name="test.logger",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg="Test message",
                args=(),
                exc_info=None,
            )

            output = formatter.format(record)
            data = json.loads(output)

            assert "correlation_id" in data
            assert data["correlation_id"] == "test-correlation-abc"
        finally:
            set_correlation_id(None)

    def test_format_excludes_correlation_id_when_none(self):
        """Should not include correlation_id field when not set."""
        from valence.core.logging import JSONFormatter, set_correlation_id

        set_correlation_id(None)
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "correlation_id" not in data

    def test_format_includes_source_for_warnings(self):
        """Should include source info for warnings and above."""
        from valence.core.logging import JSONFormatter, set_correlation_id

        set_correlation_id(None)
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_function"

        output = formatter.format(record)
        data = json.loads(output)

        assert "source" in data
        assert data["source"]["file"] == "/path/to/file.py"
        assert data["source"]["line"] == 42
        assert data["source"]["function"] == "test_function"

    def test_format_excludes_source_for_info(self):
        """Should not include source for INFO level."""
        from valence.core.logging import JSONFormatter, set_correlation_id

        set_correlation_id(None)
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "source" not in data

    def test_format_includes_exception(self):
        """Should include exception info when present."""
        from valence.core.logging import JSONFormatter, set_correlation_id

        set_correlation_id(None)
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "Test error" in data["exception"]

    def test_format_includes_extra_data(self):
        """Should include extra_data field if present."""
        from valence.core.logging import JSONFormatter, set_correlation_id

        set_correlation_id(None)
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Message",
            args=(),
            exc_info=None,
        )
        record.extra_data = {"key": "value", "count": 42}

        output = formatter.format(record)
        data = json.loads(output)

        assert "extra" in data
        assert data["extra"]["key"] == "value"
        assert data["extra"]["count"] == 42


# ============================================================================
# StandardFormatter Tests
# ============================================================================


class TestStandardFormatter:
    """Tests for StandardFormatter class."""

    def test_format_basic_message(self):
        """Should format basic log message."""
        from valence.core.logging import StandardFormatter, set_correlation_id

        set_correlation_id(None)
        formatter = StandardFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        assert "test.logger" in output
        assert "INFO" in output
        assert "Test message" in output

    def test_format_includes_correlation_id(self):
        """Should include correlation ID in message when present."""
        from valence.core.logging import StandardFormatter, set_correlation_id

        set_correlation_id("abcd1234-5678-90ab-cdef")
        try:
            formatter = StandardFormatter(use_colors=False)
            record = logging.LogRecord(
                name="test.logger",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg="Test message",
                args=(),
                exc_info=None,
            )

            output = formatter.format(record)
            # Should include first 8 chars of correlation ID
            assert "[abcd1234]" in output
            assert "Test message" in output
        finally:
            set_correlation_id(None)

    def test_uses_colors_when_enabled(self):
        """Should use colors when enabled and isatty."""
        from valence.core.logging import StandardFormatter, set_correlation_id

        set_correlation_id(None)
        # Mock stderr as a tty
        with patch.object(sys.stderr, "isatty", return_value=True):
            formatter = StandardFormatter(use_colors=True)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        # Should contain color escape codes
        assert "\033[" in output

    def test_no_colors_when_not_tty(self):
        """Should not use colors when not a tty."""
        from valence.core.logging import StandardFormatter

        # Mock stderr as not a tty
        with patch.object(sys.stderr, "isatty", return_value=False):
            formatter = StandardFormatter(use_colors=True)

        assert formatter.use_colors is False

    def test_no_colors_when_disabled(self):
        """Should not use colors when explicitly disabled."""
        from valence.core.logging import StandardFormatter

        with patch.object(sys.stderr, "isatty", return_value=True):
            formatter = StandardFormatter(use_colors=False)

        assert formatter.use_colors is False


# ============================================================================
# configure_logging Tests
# ============================================================================


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_sets_log_level(self):
        """Should set the log level."""
        from valence.core.logging import configure_logging

        configure_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_level_from_string(self):
        """Should accept level as string."""
        from valence.core.logging import configure_logging

        configure_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_level_from_int(self):
        """Should accept level as int."""
        from valence.core.logging import configure_logging

        configure_logging(level=logging.ERROR)
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_level_from_env_var(self, monkeypatch):
        """Should read level from environment variable."""
        from valence.core.logging import configure_logging

        monkeypatch.setenv("VALENCE_LOG_LEVEL", "CRITICAL")
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.CRITICAL

    def test_json_format_explicit(self):
        """Should use JSON format when explicitly set."""
        from valence.core.logging import JSONFormatter, configure_logging

        configure_logging(json_format=True)
        root = logging.getLogger()
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_json_format_from_env(self, monkeypatch):
        """Should use JSON format from VALENCE_LOG_FORMAT=json."""
        from valence.core.logging import JSONFormatter, configure_logging

        monkeypatch.setenv("VALENCE_LOG_FORMAT", "json")
        configure_logging()
        root = logging.getLogger()
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_text_format_from_env(self, monkeypatch):
        """Should use text format from VALENCE_LOG_FORMAT=text."""
        from valence.core.logging import StandardFormatter, configure_logging

        monkeypatch.setenv("VALENCE_LOG_FORMAT", "text")
        configure_logging()
        root = logging.getLogger()
        handler = root.handlers[0]
        assert isinstance(handler.formatter, StandardFormatter)

    def test_standard_format_when_tty(self):
        """Should use standard format when in terminal."""
        from valence.core.logging import StandardFormatter, configure_logging

        with patch.object(sys.stderr, "isatty", return_value=True):
            configure_logging(json_format=False)

        root = logging.getLogger()
        handler = root.handlers[0]
        assert isinstance(handler.formatter, StandardFormatter)

    def test_adds_file_handler(self, tmp_path):
        """Should add file handler when log_file specified."""
        from valence.core.logging import configure_logging

        log_file = tmp_path / "test.log"
        configure_logging(log_file=str(log_file))

        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

    def test_file_handler_uses_json(self, tmp_path):
        """File handler should always use JSON format."""
        from valence.core.logging import JSONFormatter, configure_logging

        log_file = tmp_path / "test.log"
        configure_logging(log_file=str(log_file), json_format=False)

        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        assert isinstance(file_handlers[0].formatter, JSONFormatter)

    def test_removes_existing_handlers(self):
        """Should remove existing handlers before configuring."""
        from valence.core.logging import configure_logging

        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        initial_count = len(root.handlers)

        configure_logging()

        # Should have replaced existing handlers
        assert len(root.handlers) < initial_count + 2

    def test_sets_library_log_levels(self):
        """Should set noisy library log levels to WARNING."""
        from valence.core.logging import configure_logging

        configure_logging()

        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("asyncio").level == logging.WARNING


# ============================================================================
# get_logger Tests
# ============================================================================


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self):
        """Should return a logger instance."""
        from valence.core.logging import get_logger

        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_same_name_returns_same_logger(self):
        """Same name should return same logger instance."""
        from valence.core.logging import get_logger

        logger1 = get_logger("my.logger")
        logger2 = get_logger("my.logger")
        assert logger1 is logger2


# ============================================================================
# ToolCallLogger Tests
# ============================================================================


class TestToolCallLogger:
    """Tests for ToolCallLogger class."""

    def test_sanitizes_sensitive_params(self):
        """Should redact sensitive parameters."""
        from valence.core.logging import ToolCallLogger

        tcl = ToolCallLogger()
        arguments = {
            "query": "test query",
            "password": "secret123",
            "api_key": "sk-xxxx",
            "normal_param": "visible",
        }

        sanitized = tcl._sanitize(arguments)

        assert sanitized["query"] == "test query"
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["normal_param"] == "visible"

    def test_sanitizes_nested_structures(self):
        """Should sanitize nested dicts and lists."""
        from valence.core.logging import ToolCallLogger

        tcl = ToolCallLogger()
        arguments = {
            "config": {
                "host": "localhost",
                "password": "secret",
            },
            "items": [
                {"name": "item1", "auth_token": "token123"},
            ],
        }

        sanitized = tcl._sanitize(arguments)

        assert sanitized["config"]["host"] == "localhost"
        assert sanitized["config"]["password"] == "[REDACTED]"
        assert sanitized["items"][0]["name"] == "item1"
        assert sanitized["items"][0]["auth_token"] == "[REDACTED]"

    def test_truncates_long_strings(self):
        """Should truncate very long strings."""
        from valence.core.logging import ToolCallLogger

        tcl = ToolCallLogger()
        long_string = "x" * 1000

        sanitized = tcl._sanitize(long_string)

        assert len(sanitized) < len(long_string)
        assert sanitized.endswith("...")

    def test_log_call(self):
        """Should log tool calls."""
        from valence.core.logging import ToolCallLogger

        mock_logger = MagicMock()
        tcl = ToolCallLogger(logger=mock_logger)

        tcl.log_call("belief_query", {"query": "test"})

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.DEBUG
        assert "belief_query" in call_args[0][1]

    def test_log_result(self):
        """Should log tool results."""
        from valence.core.logging import ToolCallLogger

        mock_logger = MagicMock()
        tcl = ToolCallLogger(logger=mock_logger)

        tcl.log_result("belief_query", success=True, duration_ms=42.5)

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert "success" in call_args[0][1]
        assert "42.5" in call_args[0][1]

    def test_log_result_failure(self):
        """Should log failure results."""
        from valence.core.logging import ToolCallLogger

        mock_logger = MagicMock()
        tcl = ToolCallLogger(logger=mock_logger)

        tcl.log_result("belief_query", success=False)

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert "failure" in call_args[0][1]

    def test_case_insensitive_sensitive_detection(self):
        """Should detect sensitive params case-insensitively."""
        from valence.core.logging import ToolCallLogger

        tcl = ToolCallLogger()
        arguments = {
            "PASSWORD": "secret",
            "Api_Key": "key123",
            "AUTH_TOKEN": "token",
        }

        sanitized = tcl._sanitize(arguments)

        assert sanitized["PASSWORD"] == "[REDACTED]"
        assert sanitized["Api_Key"] == "[REDACTED]"
        assert sanitized["AUTH_TOKEN"] == "[REDACTED]"

    def test_default_logger(self):
        """Should use valence.tools logger by default."""
        from valence.core.logging import ToolCallLogger

        tcl = ToolCallLogger()
        assert tcl.logger.name == "valence.tools"


# ============================================================================
# module-level tool_logger Tests
# ============================================================================


class TestModuleLevelToolLogger:
    """Tests for module-level tool_logger instance."""

    def test_tool_logger_exists(self):
        """Should have a module-level tool_logger."""
        from valence.core.logging import tool_logger

        assert tool_logger is not None

    def test_tool_logger_is_tool_call_logger(self):
        """Should be a ToolCallLogger instance."""
        from valence.core.logging import ToolCallLogger, tool_logger

        assert isinstance(tool_logger, ToolCallLogger)


# ============================================================================
# Integration Tests
# ============================================================================


class TestLoggingIntegration:
    """Integration tests for logging with correlation IDs."""

    def test_json_logging_with_correlation_context(self, tmp_path):
        """Full integration: JSON logging with correlation context."""
        from valence.core.logging import (
            configure_logging,
            correlation_context,
            get_logger,
            set_correlation_id,
        )

        set_correlation_id(None)
        log_file = tmp_path / "integration.log"

        configure_logging(json_format=True, log_file=str(log_file))
        logger = get_logger("integration.test")

        with correlation_context("integration-test-123"):
            logger.info("Test message inside context")

        logger.info("Test message outside context")

        # Read and verify log file
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 2

        log1 = json.loads(lines[0])
        assert log1["correlation_id"] == "integration-test-123"
        assert log1["message"] == "Test message inside context"

        log2 = json.loads(lines[1])
        assert "correlation_id" not in log2
        assert log2["message"] == "Test message outside context"
