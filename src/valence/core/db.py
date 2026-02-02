"""Database connection utilities for Valence."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import psycopg2
from psycopg2.extras import RealDictCursor

from .exceptions import DatabaseException

logger = logging.getLogger(__name__)


def get_connection_params() -> dict[str, Any]:
    """Get database connection parameters from environment."""
    return {
        "host": os.environ.get("VKB_DB_HOST", "localhost"),
        "port": int(os.environ.get("VKB_DB_PORT", "5432")),
        "dbname": os.environ.get("VKB_DB_NAME", "valence"),
        "user": os.environ.get("VKB_DB_USER", "valence"),
        "password": os.environ.get("VKB_DB_PASSWORD", ""),
    }


def get_connection():
    """Get a database connection with dict cursor.

    Raises:
        DatabaseException: If connection fails
    """
    params = get_connection_params()
    try:
        return psycopg2.connect(**params)
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        raise DatabaseException(f"Failed to connect to database: {e}")
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        raise DatabaseException(f"Database error: {e}")


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
        cur.close()
        conn.close()


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
        conn.close()


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

    try:
        conn = get_connection()
        cur = conn.cursor()

        for schema_file in schema_files:
            schema_path = schema_dir / schema_file
            if schema_path.exists():
                with open(schema_path) as f:
                    sql = f.read()
                    cur.execute(sql)

        conn.commit()
        cur.close()
        conn.close()
    except psycopg2.Error as e:
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


def get_schema_version() -> str | None:
    """Get the current schema version if tracked."""
    try:
        with get_cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'schema_version'
                )
            """)
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
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = %s
            ) as exists
        """, (table_name,))
        row = cur.fetchone()
        return row["exists"] if row else False


def count_rows(table_name: str) -> int:
    """Get row count for a table.

    Raises:
        ValueError: If table does not exist
        DatabaseException: If query fails
    """
    with get_cursor() as cur:
        # Use parameterized query for safety, but table names can't be parameterized
        # so we validate the table name first
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = %s
        """, (table_name,))
        if not cur.fetchone():
            raise ValueError(f"Table does not exist: {table_name}")

        cur.execute(f"SELECT COUNT(*) as count FROM {table_name}")
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
