"""Public interface for our-db.

Re-exports the primary types and functions that consumers should use.
"""

from .config import CoreSettings, get_config, set_config
from .db import (
    AsyncConnectionPool,
    ConnectionPool,
    async_cursor,
    get_connection,
    get_cursor,
    put_connection,
)
from .exceptions import (
    ConfigError,
    ConflictError,
    DatabaseError,
    NotFoundError,
    OroDbError,
    ValidationError,
)
from .migrations import MigrationRunner

__all__ = [
    # Config
    "CoreSettings",
    "get_config",
    "set_config",
    # DB
    "ConnectionPool",
    "AsyncConnectionPool",
    "get_cursor",
    "async_cursor",
    "get_connection",
    "put_connection",
    # Exceptions
    "OroDbError",
    "DatabaseError",
    "ValidationError",
    "ConfigError",
    "NotFoundError",
    "ConflictError",
    # Migrations
    "MigrationRunner",
]
