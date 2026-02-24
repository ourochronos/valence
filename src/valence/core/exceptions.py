# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Custom exception hierarchy for Valence.

Provides specific exception types for different error categories,
enabling better error handling and clearer error messages.
"""

from __future__ import annotations

from typing import Any


class ValenceException(Exception):  # noqa: N818 - kept for backwards compatibility
    """Base exception for all Valence errors.

    All Valence-specific exceptions should inherit from this class.
    """

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class DatabaseException(ValenceException):
    """Exception for database-related errors.

    Raised when:
    - Database connection fails
    - Query execution fails
    - Schema validation fails
    - Transaction errors occur
    """

    pass


class ValidationException(ValenceException):
    """Exception for validation errors.

    Raised when:
    - Input validation fails
    - Data integrity constraints are violated
    - Required fields are missing
    - Field values are out of range
    """

    def __init__(self, message: str, field: str | None = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, details)
        self.field = field
        self.value = value


class ConfigException(ValenceException):
    """Exception for configuration errors.

    Raised when:
    - Required environment variables are missing
    - Configuration files are invalid
    - Service configuration is incomplete
    """

    def __init__(self, message: str, missing_vars: list[str] | None = None):
        details = {}
        if missing_vars:
            details["missing_vars"] = missing_vars
        super().__init__(message, details)
        self.missing_vars = missing_vars or []


class NotFoundError(ValenceException):
    """Exception for resource not found errors.

    Raised when:
    - Requested belief doesn't exist
    - Requested entity doesn't exist
    - Requested session doesn't exist
    """

    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} not found: {resource_id}"
        details = {
            "resource_type": resource_type,
            "resource_id": resource_id,
        }
        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConflictError(ValenceException):
    """Exception for conflict errors.

    Raised when:
    - Attempting to create a duplicate resource
    - Optimistic locking fails
    - State conflict during update
    """

    def __init__(self, message: str, existing_id: str | None = None):
        details = {}
        if existing_id:
            details["existing_id"] = existing_id
        super().__init__(message, details)
        self.existing_id = existing_id


class EmbeddingException(ValenceException):
    """Exception for embedding-related errors.

    Raised when:
    - Embedding generation fails
    - Embedding provider is unavailable
    - Vector dimension mismatch
    """

    def __init__(self, message: str, provider: str | None = None):
        details = {}
        if provider:
            details["provider"] = provider
        super().__init__(message, details)
        self.provider = provider


class MCPException(ValenceException):
    """Exception for MCP protocol errors.

    Raised when:
    - Tool execution fails
    - Invalid tool parameters
    - Protocol-level errors
    """

    def __init__(self, message: str, tool_name: str | None = None):
        details = {}
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(message, details)
        self.tool_name = tool_name
