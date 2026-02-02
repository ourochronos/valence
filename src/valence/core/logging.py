"""Structured logging configuration for Valence.

Provides:
- Consistent log formatting across services
- JSON formatter for production (machine-parseable)
- Standard formatter for development (human-readable)
- Sanitized tool call logging
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production environments.

    Produces structured logs that can be parsed by log aggregation tools.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

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
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
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
        VALENCE_LOG_JSON: Force JSON format (true/false)
        VALENCE_LOG_FILE: Log file path
    """
    # Get settings from environment
    level = os.environ.get("VALENCE_LOG_LEVEL", level)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if json_format is None:
        json_env = os.environ.get("VALENCE_LOG_JSON", "").lower()
        if json_env in ("true", "1", "yes"):
            json_format = True
        elif json_env in ("false", "0", "no"):
            json_format = False
        else:
            # Auto-detect: use JSON if not in a terminal
            json_format = not sys.stderr.isatty()

    log_file = os.environ.get("VALENCE_LOG_FILE", log_file)

    # Create formatter
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
        level: int = logging.DEBUG
    ) -> None:
        """Log a tool call with sanitized parameters.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments (will be sanitized)
            level: Log level
        """
        sanitized = self._sanitize(arguments)
        self.logger.log(level, f"Tool call: {tool_name}", extra={
            "extra_data": {
                "tool": tool_name,
                "arguments": sanitized,
            }
        })

    def log_result(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float | None = None,
        level: int = logging.DEBUG
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

        self.logger.log(level, msg, extra={
            "extra_data": {
                "tool": tool_name,
                "success": success,
                "duration_ms": duration_ms,
            }
        })

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
