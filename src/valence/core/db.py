"""Database connection utilities for Valence.

Provides connection pooling via psycopg2's ThreadedConnectionPool.
Pool size is configured via environment variables:
    - VALENCE_DB_POOL_MIN: Minimum connections (default: 5)
    - VALENCE_DB_POOL_MAX: Maximum connections (default: 20)
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import pool as psycopg2_pool
from psycopg2.extras import RealDictCursor

from .config import get_config
from .exceptions import DatabaseException

logger = logging.getLogger(__name__)


def get_connection_params() -> dict[str, Any]:
    """Get database connection parameters from config."""
    config = get_config()
    return config.connection_params


def get_pool_config() -> dict[str, int]:
    """Get connection pool configuration from config.

    Environment variables:
        VALENCE_DB_POOL_MIN: Minimum pool size (default: 5)
        VALENCE_DB_POOL_MAX: Maximum pool size (default: 20)
    """
    config = get_config()
    return config.pool_config


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


# Module-level pool instance
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


def check_connection() -> bool:
    """Check if database connection is working."""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
            return True
    except (DatabaseException, psycopg2.Error):
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
