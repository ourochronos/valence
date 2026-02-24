# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Structured logging configuration for Valence.

Provides:
- Consistent log formatting across services
- JSON formatter for production (machine-parseable)
- Standard formatter for development (human-readable)
- Correlation IDs for request tracing
- Sanitized tool call logging
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# Context variable for correlation ID (thread/async-safe)
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Get the current correlation ID.

    Returns:
        Current correlation ID or None if not set.
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: str | None) -> None:
    """Set the correlation ID for the current context.

    Args:
        correlation_id: The correlation ID to set, or None to clear.
    """
    _correlation_id.set(correlation_id)


def generate_correlation_id() -> str:
    """Generate a new unique correlation ID.

    Returns:
        A new UUID-based correlation ID.
    """
    return str(uuid.uuid4())


@contextmanager
def correlation_context(
    correlation_id: str | None = None,
) -> Generator[str, None, None]:
    """Context manager for correlation ID scope.

    Args:
        correlation_id: Optional correlation ID to use. If None, generates a new one.

    Yields:
        The correlation ID being used.

    Example:
        with correlation_context() as cid:
            logger.info("Processing request")  # Will include cid
    """
    cid = correlation_id or generate_correlation_id()
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production environments.

    Produces structured logs that can be parsed by log aggregation tools.
    Includes correlation ID when present in context.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.

        Args:
            record: LogRecord to format.

        Returns:
            JSON string with timestamp, level, logger, message, and optional correlation ID.
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if present
        correlation_id = get_correlation_id()
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        # Add source location for errors
        if record.levelno >= logging.WARNING:
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data)


class StandardFormatter(logging.Formatter):
    """Standard log formatter for development.

    Human-readable format with colors for terminal output.
    Includes correlation ID when present in context.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    CORRELATION_COLOR = "\033[90m"  # Gray

    def __init__(self, use_colors: bool = True):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with optional color coding.

        Args:
            record: LogRecord to format.

        Returns:
            Formatted log string with timestamp, logger name, level, and message.
            Includes color coding if enabled and writing to a TTY.
        """
        # Make a copy to avoid mutating the original record
        record = logging.makeLogRecord(record.__dict__)

        # Add correlation ID to message if present
        correlation_id = get_correlation_id()
        if correlation_id:
            short_cid = correlation_id[:8]  # Use first 8 chars for readability
            if self.use_colors:
                cid_str = f"{self.CORRELATION_COLOR}[{short_cid}]{self.RESET} "
            else:
                cid_str = f"[{short_cid}] "
            record.msg = cid_str + str(record.msg)

        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"

        return super().format(record)


def configure_logging(
    level: str | int = "INFO",
    json_format: bool | None = None,
    log_file: str | None = None,
) -> None:
    """Configure logging for Valence services.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (auto-detect if None)
        log_file: Optional file to write logs to

    Environment variables:
        VALENCE_LOG_LEVEL: Override log level
        VALENCE_LOG_FORMAT: Log format ("json" or "text", auto-detect if unset)
        VALENCE_LOG_FILE: Log file path
    """
    # Get settings from config
    from .config import get_config

    config = get_config()

    level = config.log_level if level == "INFO" else level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if json_format is None:
        format_env = config.log_format.lower()
        if format_env == "json":
            json_format = True
        elif format_env == "text":
            json_format = False
        else:
            # Auto-detect: use JSON if not in a terminal
            json_format = not sys.stderr.isatty()

    log_file = config.log_file if log_file is None else log_file

    # Create formatter
    formatter: logging.Formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = StandardFormatter()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        # Always use JSON for file output
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Set levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger
    """
    return logging.getLogger(name)


class ToolCallLogger:
    """Logger for MCP tool calls.

    Logs tool calls with sanitized parameters to avoid exposing
    sensitive data in logs.
    """

    # Parameters that should be sanitized
    SENSITIVE_PARAMS = {
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "auth",
        "credential",
    }

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("valence.tools")

    def log_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        level: int = logging.DEBUG,
    ) -> None:
        """Log a tool call with sanitized parameters.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments (will be sanitized)
            level: Log level
        """
        sanitized = self._sanitize(arguments)
        self.logger.log(
            level,
            f"Tool call: {tool_name}",
            extra={
                "extra_data": {
                    "tool": tool_name,
                    "arguments": sanitized,
                }
            },
        )

    def log_result(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float | None = None,
        level: int = logging.DEBUG,
    ) -> None:
        """Log a tool result.

        Args:
            tool_name: Name of the tool
            success: Whether the call succeeded
            duration_ms: Call duration in milliseconds
            level: Log level
        """
        status = "success" if success else "failure"
        msg = f"Tool result: {tool_name} -> {status}"
        if duration_ms is not None:
            msg += f" ({duration_ms:.1f}ms)"

        self.logger.log(
            level,
            msg,
            extra={
                "extra_data": {
                    "tool": tool_name,
                    "success": success,
                    "duration_ms": duration_ms,
                }
            },
        )

    def _sanitize(self, data: Any) -> Any:
        """Recursively sanitize sensitive data.

        Args:
            data: Data to sanitize

        Returns:
            Sanitized copy of data
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if any(s in key.lower() for s in self.SENSITIVE_PARAMS):
                    result[key] = "[REDACTED]"
                else:
                    result[key] = self._sanitize(value)
            return result
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        elif isinstance(data, str) and len(data) > 500:
            # Truncate very long strings
            return data[:500] + "..."
        else:
            return data


# Default tool call logger
tool_logger = ToolCallLogger()
