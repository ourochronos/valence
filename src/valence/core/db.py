"""Database connection utilities for Valence.

Provides both synchronous and asynchronous database access:

Sync (psycopg2):
    - ConnectionPool with ThreadedConnectionPool
    - get_connection() / put_connection()
    - get_cursor() context manager

Async (asyncpg):
    - AsyncConnectionPool with native async pool
    - async_get_connection() / async_put_connection()
    - async_cursor() context manager

Pool size is configured via environment variables:
    - VALENCE_DB_POOL_MIN: Minimum connections (default: 5)
    - VALENCE_DB_POOL_MAX: Maximum connections (default: 20)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import pool as psycopg2_pool
from psycopg2.extras import RealDictCursor

from .config import get_config
from .exceptions import DatabaseException

logger = logging.getLogger(__name__)

# Optional asyncpg import - gracefully degrade if not installed
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore
    ASYNCPG_AVAILABLE = False
    logger.debug("asyncpg not installed, async database features unavailable")


def get_connection_params() -> dict[str, Any]:
    """Get database connection parameters from config."""
    config = get_config()
    return config.connection_params


def get_async_connection_params() -> dict[str, Any]:
    """Get database connection parameters for asyncpg (uses 'database' not 'dbname')."""
    params = get_connection_params()
    # asyncpg uses 'database' instead of 'dbname'
    params["database"] = params.pop("dbname")
    return params


def get_pool_config() -> dict[str, int]:
    """Get connection pool configuration from config.

    Environment variables:
        VALENCE_DB_POOL_MIN: Minimum pool size (default: 5)
        VALENCE_DB_POOL_MAX: Maximum pool size (default: 20)
    """
    config = get_config()
    return config.pool_config


# =============================================================================
# Synchronous Connection Pool (psycopg2)
# =============================================================================


class ConnectionPool:
    """Thread-safe connection pool manager.

    Uses psycopg2's ThreadedConnectionPool for concurrent access.
    Pool is lazily initialized on first connection request.

    This is a singleton - use get_instance() to access.
    """

    _instance: ConnectionPool | None = None
    _lock = threading.Lock()

    def __init__(self):
        self._pool: psycopg2_pool.ThreadedConnectionPool | None = None
        self._pool_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> ConnectionPool:
        """Get the singleton pool instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_pool(self) -> psycopg2_pool.ThreadedConnectionPool:
        """Ensure pool is initialized, creating it if necessary."""
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    conn_params = get_connection_params()
                    pool_config = get_pool_config()
                    try:
                        self._pool = psycopg2_pool.ThreadedConnectionPool(
                            minconn=pool_config["minconn"],
                            maxconn=pool_config["maxconn"],
                            **conn_params,
                        )
                        logger.info(
                            f"Connection pool initialized: "
                            f"min={pool_config['minconn']}, max={pool_config['maxconn']}"
                        )
                    except psycopg2.OperationalError as e:
                        logger.error(f"Failed to create connection pool: {e}")
                        raise DatabaseException(f"Failed to create connection pool: {e}")
        return self._pool

    def get_connection(self):
        """Get a connection from the pool.

        Raises:
            DatabaseException: If pool is exhausted or connection fails
        """
        pool = self._ensure_pool()
        try:
            conn = pool.getconn()
            if conn is None:
                raise DatabaseException("Connection pool exhausted")
            return conn
        except psycopg2_pool.PoolError as e:
            logger.error(f"Pool error getting connection: {e}")
            raise DatabaseException(f"Failed to get connection from pool: {e}")
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            raise DatabaseException(f"Database error: {e}")

    def put_connection(self, conn) -> None:
        """Return a connection to the pool."""
        if self._pool is not None and conn is not None:
            try:
                self._pool.putconn(conn)
            except psycopg2_pool.PoolError as e:
                logger.warning(f"Error returning connection to pool: {e}")
                # Try to close the connection if we can't return it
                try:
                    conn.close()
                except Exception:
                    pass

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._pool_lock:
            if self._pool is not None:
                self._pool.closeall()
                self._pool = None
                logger.info("Connection pool closed")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        pool_config = get_pool_config()

        if self._pool is None:
            return {
                "initialized": False,
                "min_connections": pool_config["minconn"],
                "max_connections": pool_config["maxconn"],
            }

        return {
            "initialized": True,
            "min_connections": pool_config["minconn"],
            "max_connections": pool_config["maxconn"],
        }


# Module-level sync pool instance
_pool = ConnectionPool.get_instance()


def get_connection():
    """Get a database connection from the pool.

    Raises:
        DatabaseException: If connection fails
    """
    return _pool.get_connection()


def put_connection(conn) -> None:
    """Return a connection to the pool."""
    _pool.put_connection(conn)


def close_pool() -> None:
    """Close the connection pool."""
    _pool.close_all()


def get_pool_stats() -> dict[str, Any]:
    """Get connection pool statistics."""
    return _pool.get_stats()


@contextmanager
def get_cursor(dict_cursor: bool = True) -> Generator:
    """Context manager for database cursor.

    Usage:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM beliefs")
            rows = cur.fetchall()

    Raises:
        DatabaseException: On database errors
    """
    conn = get_connection()
    cur = None
    try:
        cursor_factory = RealDictCursor if dict_cursor else None
        cur = conn.cursor(cursor_factory=cursor_factory)
        yield cur
        conn.commit()
    except psycopg2.IntegrityError as e:
        conn.rollback()
        logger.error(f"Database integrity error: {e}")
        raise DatabaseException(f"Integrity constraint violation: {e}")
    except psycopg2.ProgrammingError as e:
        conn.rollback()
        logger.error(f"Database programming error: {e}")
        raise DatabaseException(f"SQL error: {e}")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise DatabaseException(f"Database error: {e}")
    finally:
        if cur is not None:
            cur.close()
        put_connection(conn)


@contextmanager
def get_connection_context():
    """Context manager for database connection.

    Usage:
        with get_connection_context() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM beliefs")

    Raises:
        DatabaseException: On database errors
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except psycopg2.IntegrityError as e:
        conn.rollback()
        logger.error(f"Database integrity error: {e}")
        raise DatabaseException(f"Integrity constraint violation: {e}")
    except psycopg2.ProgrammingError as e:
        conn.rollback()
        logger.error(f"Database programming error: {e}")
        raise DatabaseException(f"SQL error: {e}")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise DatabaseException(f"Database error: {e}")
    finally:
        put_connection(conn)


# =============================================================================
# Asynchronous Connection Pool (asyncpg)
# =============================================================================


class AsyncConnectionPool:
    """Async connection pool manager using asyncpg.

    Pool is lazily initialized on first connection request.
    This is a singleton - use get_instance() to access.

    Requires asyncpg to be installed.
    """

    _instance: AsyncConnectionPool | None = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls) -> AsyncConnectionPool:
        """Get the singleton pool instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def get_instance_sync(cls) -> AsyncConnectionPool:
        """Get the singleton pool instance synchronously (for setup).

        Note: The pool itself is still initialized lazily via async.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _ensure_pool(self) -> asyncpg.Pool:
        """Ensure pool is initialized, creating it if necessary."""
        if not ASYNCPG_AVAILABLE:
            raise DatabaseException(
                "asyncpg not installed. Install with: pip install asyncpg"
            )

        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    conn_params = get_async_connection_params()
                    pool_config = get_pool_config()
                    try:
                        self._pool = await asyncpg.create_pool(
                            min_size=pool_config["minconn"],
                            max_size=pool_config["maxconn"],
                            **conn_params,
                        )
                        logger.info(
                            f"Async connection pool initialized: "
                            f"min={pool_config['minconn']}, max={pool_config['maxconn']}"
                        )
                    except asyncpg.PostgresError as e:
                        logger.error(f"Failed to create async connection pool: {e}")
                        raise DatabaseException(f"Failed to create async connection pool: {e}")
                    except OSError as e:
                        logger.error(f"Failed to connect to database: {e}")
                        raise DatabaseException(f"Failed to connect to database: {e}")
        return self._pool

    async def get_connection(self) -> asyncpg.Connection:
        """Get a connection from the pool.

        Raises:
            DatabaseException: If pool is exhausted or connection fails
        """
        pool = await self._ensure_pool()
        try:
            conn = await pool.acquire()
            return conn
        except asyncpg.PostgresError as e:
            logger.error(f"Database error: {e}")
            raise DatabaseException(f"Database error: {e}")

    async def put_connection(self, conn: asyncpg.Connection) -> None:
        """Return a connection to the pool."""
        if self._pool is not None and conn is not None:
            try:
                await self._pool.release(conn)
            except Exception as e:
                logger.warning(f"Error returning connection to pool: {e}")
                try:
                    await conn.close()
                except Exception:
                    pass

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self._pool_lock:
            if self._pool is not None:
                await self._pool.close()
                self._pool = None
                logger.info("Async connection pool closed")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        pool_config = get_pool_config()

        if self._pool is None:
            return {
                "initialized": False,
                "min_connections": pool_config["minconn"],
                "max_connections": pool_config["maxconn"],
                "type": "async",
            }

        return {
            "initialized": True,
            "min_connections": pool_config["minconn"],
            "max_connections": pool_config["maxconn"],
            "size": self._pool.get_size(),
            "free_size": self._pool.get_idle_size(),
            "type": "async",
        }


# Module-level async pool instance (lazy)
_async_pool: AsyncConnectionPool | None = None


def _get_async_pool() -> AsyncConnectionPool:
    """Get the async pool instance (creates if needed)."""
    global _async_pool
    if _async_pool is None:
        _async_pool = AsyncConnectionPool.get_instance_sync()
    return _async_pool


async def async_get_connection() -> asyncpg.Connection:
    """Get an async database connection from the pool.

    Raises:
        DatabaseException: If connection fails
    """
    pool = _get_async_pool()
    return await pool.get_connection()


async def async_put_connection(conn: asyncpg.Connection) -> None:
    """Return an async connection to the pool."""
    pool = _get_async_pool()
    await pool.put_connection(conn)


async def async_close_pool() -> None:
    """Close the async connection pool."""
    pool = _get_async_pool()
    await pool.close_all()


def get_async_pool_stats() -> dict[str, Any]:
    """Get async connection pool statistics."""
    pool = _get_async_pool()
    return pool.get_stats()


@asynccontextmanager
async def async_cursor() -> AsyncGenerator[asyncpg.Connection, None]:
    """Async context manager for database operations.

    asyncpg doesn't use cursors like psycopg2 - you execute directly on the connection.
    This context manager handles transaction management.

    Usage:
        async with async_cursor() as conn:
            rows = await conn.fetch("SELECT * FROM beliefs")
            # or
            await conn.execute("INSERT INTO beliefs ...")

    Raises:
        DatabaseException: On database errors
    """
    conn = await async_get_connection()
    try:
        async with conn.transaction():
            yield conn
    except asyncpg.UniqueViolationError as e:
        logger.error(f"Database integrity error: {e}")
        raise DatabaseException(f"Integrity constraint violation: {e}")
    except asyncpg.PostgresSyntaxError as e:
        logger.error(f"Database programming error: {e}")
        raise DatabaseException(f"SQL error: {e}")
    except asyncpg.PostgresError as e:
        logger.error(f"Database error: {e}")
        raise DatabaseException(f"Database error: {e}")
    finally:
        await async_put_connection(conn)


@asynccontextmanager
async def async_connection_context() -> AsyncGenerator[asyncpg.Connection, None]:
    """Async context manager for database connection.

    Similar to async_cursor but without automatic transaction.
    Useful when you need explicit transaction control.

    Usage:
        async with async_connection_context() as conn:
            async with conn.transaction():
                await conn.execute("...")

    Raises:
        DatabaseException: On database errors
    """
    conn = await async_get_connection()
    try:
        yield conn
    except asyncpg.UniqueViolationError as e:
        logger.error(f"Database integrity error: {e}")
        raise DatabaseException(f"Integrity constraint violation: {e}")
    except asyncpg.PostgresSyntaxError as e:
        logger.error(f"Database programming error: {e}")
        raise DatabaseException(f"SQL error: {e}")
    except asyncpg.PostgresError as e:
        logger.error(f"Database error: {e}")
        raise DatabaseException(f"Database error: {e}")
    finally:
        await async_put_connection(conn)


# =============================================================================
# Utility Functions (sync, with async variants)
# =============================================================================


def generate_id() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def init_schema() -> None:
    """Initialize database schema from SQL files.

    Raises:
        DatabaseException: If schema initialization fails
    """
    schema_dir = Path(__file__).parent.parent / "substrate"
    schema_files = [
        "schema.sql",
        "procedures.sql",
    ]

    conn = get_connection()
    cur = None
    try:
        cur = conn.cursor()

        for schema_file in schema_files:
            schema_path = schema_dir / schema_file
            if schema_path.exists():
                with open(schema_path) as f:
                    sql = f.read()
                    cur.execute(sql)

        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Schema initialization failed: {e}")
        raise DatabaseException(f"Failed to initialize schema: {e}")
    finally:
        if cur is not None:
            cur.close()
        put_connection(conn)


async def async_init_schema() -> None:
    """Initialize database schema from SQL files (async version).

    Raises:
        DatabaseException: If schema initialization fails
    """
    schema_dir = Path(__file__).parent.parent / "substrate"
    schema_files = [
        "schema.sql",
        "procedures.sql",
    ]

    async with async_connection_context() as conn:
        try:
            for schema_file in schema_files:
                schema_path = schema_dir / schema_file
                if schema_path.exists():
                    with open(schema_path) as f:
                        sql = f.read()
                        await conn.execute(sql)
        except asyncpg.PostgresError as e:
            logger.error(f"Schema initialization failed: {e}")
            raise DatabaseException(f"Failed to initialize schema: {e}")


def check_connection() -> bool:
    """Check if database connection is working."""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
            return True
    except (DatabaseException, psycopg2.Error):
        return False


async def async_check_connection() -> bool:
    """Check if async database connection is working."""
    try:
        async with async_cursor() as conn:
            await conn.fetchval("SELECT 1")
            return True
    except (DatabaseException, Exception):
        return False


def get_schema_version() -> str | None:
    """Get the current schema version if tracked."""
    try:
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'schema_version'
                )
            """
            )
            if cur.fetchone()[0]:
                cur.execute("SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1")
                row = cur.fetchone()
                return row["version"] if row else None
            return None
    except (DatabaseException, psycopg2.Error) as e:
        logger.debug(f"Could not get schema version: {e}")
        return None


async def async_get_schema_version() -> str | None:
    """Get the current schema version if tracked (async version)."""
    try:
        async with async_cursor() as conn:
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'schema_version'
                )
            """
            )
            if exists:
                row = await conn.fetchrow(
                    "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
                )
                return row["version"] if row else None
            return None
    except (DatabaseException, Exception) as e:
        logger.debug(f"Could not get schema version: {e}")
        return None


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database.

    Raises:
        DatabaseException: If query fails
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = %s
            ) as exists
        """,
            (table_name,),
        )
        row = cur.fetchone()
        return row["exists"] if row else False


async def async_table_exists(table_name: str) -> bool:
    """Check if a table exists in the database (async version).

    Raises:
        DatabaseException: If query fails
    """
    async with async_cursor() as conn:
        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = $1
            )
        """,
            table_name,
        )
        return exists or False


# Allowlist of valid tables for count_rows() to prevent SQL injection
# Only tables that are safe to query should be listed here
VALID_TABLES = frozenset(
    [
        "beliefs",
        "entities",
        "vkb_sessions",
        "vkb_exchanges",
        "vkb_patterns",
        "tensions",
        "federation_nodes",
        "node_trust",
        "user_node_trust",
        "sync_state",
        "schema_version",
    ]
)


def count_rows(table_name: str) -> int:
    """Get row count for a table.

    Args:
        table_name: Name of the table (must be in VALID_TABLES allowlist)

    Raises:
        ValueError: If table is not in allowlist or does not exist
        DatabaseException: If query fails
    """
    # Security: Validate against allowlist BEFORE any database query
    # This prevents SQL injection via table name manipulation
    if table_name not in VALID_TABLES:
        raise ValueError(
            f"Table not in allowlist: {table_name}. "
            f"Valid tables: {', '.join(sorted(VALID_TABLES))}"
        )

    with get_cursor() as cur:
        # Double-check table exists (defense in depth)
        cur.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_name = %s
        """,
            (table_name,),
        )
        if not cur.fetchone():
            raise ValueError(f"Table does not exist: {table_name}")

        # Safe to interpolate because we validated against allowlist
        cur.execute(f"SELECT COUNT(*) as count FROM {table_name}")  # nosec B608
        row = cur.fetchone()
        return row["count"] if row else 0


async def async_count_rows(table_name: str) -> int:
    """Get row count for a table (async version).

    Args:
        table_name: Name of the table (must be in VALID_TABLES allowlist)

    Raises:
        ValueError: If table is not in allowlist or does not exist
        DatabaseException: If query fails
    """
    # Security: Validate against allowlist BEFORE any database query
    if table_name not in VALID_TABLES:
        raise ValueError(
            f"Table not in allowlist: {table_name}. "
            f"Valid tables: {', '.join(sorted(VALID_TABLES))}"
        )

    async with async_cursor() as conn:
        # Double-check table exists (defense in depth)
        exists = await conn.fetchval(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_name = $1
        """,
            table_name,
        )
        if not exists:
            raise ValueError(f"Table does not exist: {table_name}")

        # Safe to interpolate because we validated against allowlist
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")  # nosec B608
        return count or 0


class DatabaseStats:
    """Statistics about the database."""

    def __init__(self):
        self.beliefs_count: int = 0
        self.entities_count: int = 0
        self.sessions_count: int = 0
        self.exchanges_count: int = 0
        self.patterns_count: int = 0
        self.tensions_count: int = 0

    @classmethod
    def collect(cls) -> DatabaseStats:
        """Collect current database statistics."""
        stats = cls()

        tables = [
            ("beliefs", "beliefs_count"),
            ("entities", "entities_count"),
            ("vkb_sessions", "sessions_count"),
            ("vkb_exchanges", "exchanges_count"),
            ("vkb_patterns", "patterns_count"),
            ("tensions", "tensions_count"),
        ]

        for table, attr in tables:
            try:
                setattr(stats, attr, count_rows(table))
            except (ValueError, DatabaseException, psycopg2.Error) as e:
                logger.debug(f"Could not count rows in {table}: {e}")

        return stats

    @classmethod
    async def async_collect(cls) -> DatabaseStats:
        """Collect current database statistics (async version)."""
        stats = cls()

        tables = [
            ("beliefs", "beliefs_count"),
            ("entities", "entities_count"),
            ("vkb_sessions", "sessions_count"),
            ("vkb_exchanges", "exchanges_count"),
            ("vkb_patterns", "patterns_count"),
            ("tensions", "tensions_count"),
        ]

        for table, attr in tables:
            try:
                setattr(stats, attr, await async_count_rows(table))
            except (ValueError, DatabaseException, Exception) as e:
                logger.debug(f"Could not count rows in {table}: {e}")

        return stats

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "beliefs": self.beliefs_count,
            "entities": self.entities_count,
            "sessions": self.sessions_count,
            "exchanges": self.exchanges_count,
            "patterns": self.patterns_count,
            "tensions": self.tensions_count,
        }
