"""Exception hierarchy for our-db.

Provides specific exception types for different error categories,
enabling better error handling and clearer error messages.
"""

from __future__ import annotations

from typing import Any


class OroDbError(Exception):
    """Base exception for all our-db errors.

    All our-db-specific exceptions should inherit from this class.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class DatabaseError(OroDbError):
    """Exception for database-related errors.

    Raised when:
    - Database connection fails
    - Query execution fails
    - Schema validation fails
    - Transaction errors occur
    """


class ValidationError(OroDbError):
    """Exception for validation errors.

    Raised when:
    - Input validation fails
    - Data integrity constraints are violated
    - Required fields are missing
    - Field values are out of range
    """

    def __init__(self, message: str, field: str | None = None, value: Any = None):
        details: dict[str, Any] = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, details)
        self.field = field
        self.value = value


class ConfigError(OroDbError):
    """Exception for configuration errors.

    Raised when:
    - Required environment variables are missing
    - Configuration files are invalid
    - Service configuration is incomplete
    """

    def __init__(self, message: str, missing_vars: list[str] | None = None):
        details: dict[str, Any] = {}
        if missing_vars:
            details["missing_vars"] = missing_vars
        super().__init__(message, details)
        self.missing_vars = missing_vars or []


class NotFoundError(OroDbError):
    """Exception for resource not found errors."""

    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} not found: {resource_id}"
        details = {
            "resource_type": resource_type,
            "resource_id": resource_id,
        }
        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConflictError(OroDbError):
    """Exception for conflict errors.

    Raised when:
    - Attempting to create a duplicate resource
    - Optimistic locking fails
    - State conflict during update
    """

    def __init__(self, message: str, existing_id: str | None = None):
        details: dict[str, Any] = {}
        if existing_id:
            details["existing_id"] = existing_id
        super().__init__(message, details)
        self.existing_id = existing_id
