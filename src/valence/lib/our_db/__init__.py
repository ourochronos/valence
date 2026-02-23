"""our-db: Database connectivity, configuration, and migration brick."""

__version__ = "0.1.0"

# Configuration
from .config import CoreSettings, clear_config_cache, get_config, set_config

# Database connectivity (sync)
from .db import (
    ASYNCPG_AVAILABLE,
    AsyncConnectionPool,
    ConnectionPool,
    async_check_connection,
    async_close_pool,
    async_connection_context,
    async_count_rows,
    async_cursor,
    async_get_connection,
    async_get_schema_version,
    async_init_schema,
    async_put_connection,
    async_table_exists,
    check_connection,
    close_pool,
    count_rows,
    generate_id,
    get_async_pool_stats,
    get_connection,
    get_connection_context,
    get_cursor,
    get_pool_stats,
    get_schema_version,
    init_schema,
    put_connection,
    table_exists,
)

# Exceptions
from .exceptions import (
    ConfigError,
    ConflictError,
    DatabaseError,
    NotFoundError,
    OroDbError,
    ValidationError,
)

# Migrations
from .migrations import (
    AppliedMigration,
    MigrationInfo,
    MigrationRunner,
    MigrationStatus,
)

# Utilities
from .utils import escape_ilike

__all__ = [
    "__version__",
    # Config
    "CoreSettings",
    "get_config",
    "set_config",
    "clear_config_cache",
    # Exceptions
    "OroDbError",
    "DatabaseError",
    "ValidationError",
    "ConfigError",
    "NotFoundError",
    "ConflictError",
    # Sync DB
    "ConnectionPool",
    "get_connection",
    "put_connection",
    "get_cursor",
    "get_connection_context",
    "close_pool",
    "get_pool_stats",
    # Async DB
    "ASYNCPG_AVAILABLE",
    "AsyncConnectionPool",
    "async_get_connection",
    "async_put_connection",
    "async_cursor",
    "async_connection_context",
    "async_close_pool",
    "get_async_pool_stats",
    # Utilities
    "generate_id",
    "init_schema",
    "async_init_schema",
    "check_connection",
    "async_check_connection",
    "get_schema_version",
    "async_get_schema_version",
    "table_exists",
    "async_table_exists",
    "count_rows",
    "async_count_rows",
    "escape_ilike",
    # Migrations
    "MigrationRunner",
    "MigrationInfo",
    "MigrationStatus",
    "AppliedMigration",
]
