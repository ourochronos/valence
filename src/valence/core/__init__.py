# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Valence Core - Shared primitives for the knowledge substrate."""

from .config import (
    clear_config_cache,
    get_config,
)
from .exceptions import (
    ConfigException,
    ConflictError,
    DatabaseException,
    EmbeddingException,
    MCPException,
    NotFoundError,
    ValenceException,
    ValidationException,
)
from .health import (
    HealthStatus,
    require_healthy,
    run_health_check,
    startup_checks,
    validate_database,
    validate_environment,
)
from .logging import (
    ToolCallLogger,
    configure_logging,
    get_logger,
    tool_logger,
)
from .lru_cache import (
    DEFAULT_CACHE_MAX_SIZE,
    BoundedList,
    LRUDict,
    get_cache_max_size,
)
from .resources import (
    Resource,
    ResourceReport,
    ResourceType,
    SafetyStatus,
    UsageAttestation,
)
from .temporal import TemporalValidity

__all__ = [
    # Config
    "get_config",
    "clear_config_cache",
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
    # LRU Cache
    "LRUDict",
    "BoundedList",
    "get_cache_max_size",
    "DEFAULT_CACHE_MAX_SIZE",
    # Resources
    "Resource",
    "ResourceType",
    "SafetyStatus",
    "ResourceReport",
    "UsageAttestation",
    # Temporal
    "TemporalValidity",
]
