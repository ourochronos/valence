"""Valence Core - Shared primitives for the knowledge substrate."""

from .models import (
    Belief,
    Entity,
    Source,
    Tension,
    Session,
    Exchange,
    Pattern,
    BeliefEntity,
    SessionInsight,
)
from .confidence import DimensionalConfidence, ConfidenceDimension
from .temporal import TemporalValidity
from .db import get_connection, generate_id
from .exceptions import (
    ValenceException,
    DatabaseException,
    ValidationException,
    ConfigException,
    NotFoundError,
    ConflictError,
    EmbeddingException,
    MCPException,
)
from .health import (
    HealthStatus,
    run_health_check,
    require_healthy,
    startup_checks,
    validate_environment,
    validate_database,
)
from .logging import (
    configure_logging,
    get_logger,
    ToolCallLogger,
    tool_logger,
)
from .mcp_base import (
    MCPServerBase,
    ToolRouter,
    success_response,
    error_response,
    not_found_response,
)

__all__ = [
    # Models
    "Belief",
    "Entity",
    "Source",
    "Tension",
    "Session",
    "Exchange",
    "Pattern",
    "BeliefEntity",
    "SessionInsight",
    # Confidence
    "DimensionalConfidence",
    "ConfidenceDimension",
    # Temporal
    "TemporalValidity",
    # Database
    "get_connection",
    "generate_id",
    # Exceptions
    "ValenceException",
    "DatabaseException",
    "ValidationException",
    "ConfigException",
    "NotFoundError",
    "ConflictError",
    "EmbeddingException",
    "MCPException",
    # Health
    "HealthStatus",
    "run_health_check",
    "require_healthy",
    "startup_checks",
    "validate_environment",
    "validate_database",
    # Logging
    "configure_logging",
    "get_logger",
    "ToolCallLogger",
    "tool_logger",
    # MCP Base
    "MCPServerBase",
    "ToolRouter",
    "success_response",
    "error_response",
    "not_found_response",
]
