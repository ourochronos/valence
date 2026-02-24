# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Database connection management for Valence.

Config via VALENCE_DB_* environment variables.
"""

import os
import queue
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from psycopg2 import pool as psycopg2_pool
from psycopg2.extras import RealDictCursor
from psycopg2.pool import PoolError


def _get_db_config() -> dict[str, Any]:
    """Get database connection parameters from environment."""
    return {
        "host": os.environ.get("VALENCE_DB_HOST", "127.0.0.1"),
        "port": int(os.environ.get("VALENCE_DB_PORT", "5433")),
        "dbname": os.environ.get("VALENCE_DB_NAME", "valence"),
        "user": os.environ.get("VALENCE_DB_USER", "valence"),
        "password": os.environ.get("VALENCE_DB_PASSWORD", "valence"),
    }


# Connection pool (lazy init, thread-safe)
_pool: psycopg2_pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> psycopg2_pool.ThreadedConnectionPool:
    """Get or create the connection pool."""
    from valence.core.config import get_config

    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                cfg = _get_db_config()
                config = get_config()
                _pool = psycopg2_pool.ThreadedConnectionPool(
                    minconn=config.db_pool_min,
                    maxconn=config.db_pool_max,
                    **cfg,
                )
    return _pool


def _get_conn_with_timeout(pool: psycopg2_pool.ThreadedConnectionPool, timeout: int) -> Any:
    """Get a connection from pool with timeout.

    Args:
        pool: The connection pool
        timeout: Timeout in seconds

    Returns:
        A database connection

    Raises:
        PoolError: If timeout expires before connection is available
    """
    result_queue: queue.Queue = queue.Queue()

    def _get_conn():
        try:
            conn = pool.getconn()
            result_queue.put(("success", conn))
        except Exception as e:
            result_queue.put(("error", e))

    thread = threading.Thread(target=_get_conn, daemon=True)
    thread.start()

    try:
        result_type, result_value = result_queue.get(timeout=timeout)
        if result_type == "error":
            raise result_value
        return result_value
    except queue.Empty:
        raise PoolError(f"Connection pool timeout after {timeout} seconds")


def _validate_connection(conn: Any) -> bool:
    """Check if a connection is valid and healthy.

    Args:
        conn: Database connection to validate

    Returns:
        True if connection is valid, False otherwise
    """
    if conn.closed:
        return False

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return True
    except Exception:
        return False


def _get_healthy_connection(pool: psycopg2_pool.ThreadedConnectionPool, timeout: int) -> Any:
    """Get a healthy connection from pool, discarding stale ones.

    Args:
        pool: The connection pool
        timeout: Timeout in seconds

    Returns:
        A validated database connection

    Raises:
        PoolError: If unable to get a healthy connection
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        conn = _get_conn_with_timeout(pool, timeout)
        if _validate_connection(conn):
            return conn

        # Connection is stale, close it and try again
        try:
            conn.close()
        except Exception:
            pass
        pool.putconn(conn, close=True)

        if attempt == max_attempts - 1:
            raise PoolError("Failed to get healthy connection after multiple attempts")

    raise PoolError("Failed to get healthy connection")


@contextmanager
def get_cursor() -> Generator[Any, None, None]:
    """Get a database cursor with auto-commit on success, rollback on error.

    Usage:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM sources")
            rows = cur.fetchall()
    """
    from valence.core.config import get_config

    pool = _get_pool()
    config = get_config()
    conn = _get_healthy_connection(pool, config.db_pool_timeout)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


@contextmanager
def get_connection() -> Generator[Any, None, None]:
    """Get a database connection from the pool.

    For cases that need connection-level control (like migrations).

    Usage:
        with get_connection() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE DATABASE test")
    """
    from valence.core.config import get_config

    pool = _get_pool()
    config = get_config()
    conn = _get_healthy_connection(pool, config.db_pool_timeout)
    try:
        yield conn
    finally:
        pool.putconn(conn)


def put_connection(conn: Any) -> None:
    """Return a connection to the pool (compatibility wrapper)."""
    pool = _get_pool()
    pool.putconn(conn)


def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


def generate_id() -> str:
    """Generate a UUID for database records."""
    return str(uuid.uuid4())


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            )
            """,
            (table_name,),
        )
        result = cur.fetchone()
        return result["exists"] if result else False


def count_rows(table_name: str) -> int:
    """Count rows in a table."""
    from psycopg2 import sql as psql

    with get_cursor() as cur:
        cur.execute(psql.SQL("SELECT COUNT(*) as count FROM {}").format(psql.Identifier(table_name)))
        result = cur.fetchone()
        return result["count"] if result else 0


def get_schema_version() -> str | None:
    """Get the current schema version from system_config."""
    if not table_exists("system_config"):
        return None

    with get_cursor() as cur:
        cur.execute("SELECT value FROM system_config WHERE key = 'schema_version'")
        result = cur.fetchone()
        return result["value"] if result else None


def init_schema(schema_path: str | None = None) -> None:
    """Initialize database schema from schema.sql.

    Args:
        schema_path: Path to schema.sql file (optional, auto-detects if None)
    """
    from pathlib import Path

    if schema_path is None:
        # Auto-detect schema.sql relative to this file
        # src/valence/core/db.py -> src/valence/schema.sql or ./schema.sql
        core_dir = Path(__file__).parent
        valence_dir = core_dir.parent

        candidates = [
            valence_dir / "schema.sql",
            valence_dir.parent / "schema.sql",
            Path.cwd() / "schema.sql",
        ]

        for candidate in candidates:
            if candidate.exists():
                schema_path = str(candidate)
                break

        if schema_path is None:
            raise FileNotFoundError("schema.sql not found. Checked: " + ", ".join(str(c) for c in candidates))

    with open(schema_path) as f:
        schema_sql = f.read()

    with get_connection() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(schema_sql)


def check_connection() -> bool:
    """Check if database connection is working."""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def get_connection_params() -> dict[str, Any]:
    """Get database connection parameters (for compatibility)."""
    return _get_db_config()


def serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a DB row dict: UUID → str, datetime → ISO string."""
    import uuid as _uuid
    from datetime import datetime as _dt

    d = dict(row)
    for key, val in list(d.items()):
        if isinstance(val, _dt):
            d[key] = val.isoformat()
        elif isinstance(val, _uuid.UUID):
            d[key] = str(val)
    return d
