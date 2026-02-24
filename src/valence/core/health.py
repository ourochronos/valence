"""Health check utilities for Valence services.

Provides startup validation, health probes, and diagnostic information.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from valence.core.db import count_rows, get_connection_params, table_exists

from .exceptions import ConfigException, DatabaseException

logger = logging.getLogger(__name__)


class DatabaseStats:
    """Statistics about the valence database tables."""

    def __init__(self):
        self.beliefs_count: int = 0
        self.entities_count: int = 0
        self.sessions_count: int = 0
        self.exchanges_count: int = 0
        self.patterns_count: int = 0
        self.tensions_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "beliefs_count": self.beliefs_count,
            "entities_count": self.entities_count,
            "sessions_count": self.sessions_count,
            "exchanges_count": self.exchanges_count,
            "patterns_count": self.patterns_count,
            "tensions_count": self.tensions_count,
        }

    @classmethod
    def collect(cls) -> DatabaseStats:
        """Collect current database statistics."""
        import psycopg2

        stats = cls()
        tables = [
            ("articles", "beliefs_count"),
            ("entities", "entities_count"),
            ("contentions", "tensions_count"),
        ]
        for table, attr in tables:
            try:
                setattr(stats, attr, count_rows(table))
            except (ValueError, DatabaseException, psycopg2.Error) as e:
                logger.debug(f"Could not count rows in {table}: {e}")
        return stats


# Required environment variables for operation
REQUIRED_ENV_VARS = [
    "VALENCE_DB_HOST",
    "VALENCE_DB_NAME",
    "VALENCE_DB_USER",
]

# Optional but recommended environment variables
OPTIONAL_ENV_VARS = [
    "VALENCE_DB_PASSWORD",
    "OPENAI_API_KEY",  # For embeddings
]

# Core tables that must exist for the system to function
REQUIRED_TABLES = [
    "articles",
    "entities",
    "article_entities",
    "sources",
    "contentions",
]


@dataclass
class HealthStatus:
    """Overall health status of the system."""

    healthy: bool = False
    database_connected: bool = False
    schema_valid: bool = False
    pgvector_available: bool = False
    env_vars_present: bool = False
    missing_env_vars: list[str] = field(default_factory=list)
    missing_tables: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "healthy": self.healthy,
            "database_connected": self.database_connected,
            "schema_valid": self.schema_valid,
            "pgvector_available": self.pgvector_available,
            "env_vars_present": self.env_vars_present,
            "missing_env_vars": self.missing_env_vars,
            "missing_tables": self.missing_tables,
            "warnings": self.warnings,
            "stats": self.stats,
            "error": self.error,
        }


def check_env_vars() -> tuple[bool, list[str], list[str]]:
    """Check for required and optional environment variables.

    Returns:
        Tuple of (all_required_present, missing_required, missing_optional)
    """
    missing_required = []
    missing_optional = []

    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            missing_required.append(var)

    for var in OPTIONAL_ENV_VARS:
        if not os.environ.get(var):
            missing_optional.append(var)

    return len(missing_required) == 0, missing_required, missing_optional


def check_database_connection() -> tuple[bool, str | None]:
    """Check if database is accessible.

    Returns:
        Tuple of (connected, error_message)
    """
    try:
        import psycopg2

        params = get_connection_params()
        conn = psycopg2.connect(**params)
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return True, None
    except ImportError:
        return False, "psycopg2 not installed"
    except psycopg2.OperationalError as e:
        return False, f"Connection failed: {e}"
    except psycopg2.Error as e:
        return False, f"Database error: {e}"


def check_pgvector() -> tuple[bool, str | None]:
    """Check if pgvector extension is available.

    Returns:
        Tuple of (available, error_message)
    """
    try:
        import psycopg2

        params = get_connection_params()
        conn = psycopg2.connect(**params)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                )
            """
            )
            result = cur.fetchone()
            if result and result[0]:
                conn.close()
                return True, None
            else:
                conn.close()
                return False, "pgvector extension not installed"
    except psycopg2.Error as e:
        return False, f"Database error: {e}"


def check_schema() -> tuple[bool, list[str]]:
    """Check if all required tables exist.

    Returns:
        Tuple of (all_present, missing_tables)
    """
    missing = []
    for table in REQUIRED_TABLES:
        try:
            if not table_exists(table):
                missing.append(table)
        except DatabaseException as e:
            logger.debug(f"Error checking table {table}: {e}")
            missing.append(table)

    return len(missing) == 0, missing


def run_health_check() -> HealthStatus:
    """Run comprehensive health check.

    Returns:
        HealthStatus with all check results
    """
    status = HealthStatus()
    warnings = []

    # Check environment variables
    env_ok, missing_required, missing_optional = check_env_vars()
    status.env_vars_present = env_ok
    status.missing_env_vars = missing_required

    if missing_optional:
        warnings.append(f"Optional env vars not set: {', '.join(missing_optional)}")

    if not env_ok:
        status.error = f"Missing required environment variables: {', '.join(missing_required)}"
        status.warnings = warnings
        return status

    # Check database connection
    db_ok, db_error = check_database_connection()
    status.database_connected = db_ok

    if not db_ok:
        status.error = f"Database connection failed: {db_error}"
        status.warnings = warnings
        return status

    # Check pgvector
    pgvector_ok, pgvector_error = check_pgvector()
    status.pgvector_available = pgvector_ok

    if not pgvector_ok:
        warnings.append(f"pgvector not available: {pgvector_error}")

    # Check schema
    schema_ok, missing_tables = check_schema()
    status.schema_valid = schema_ok
    status.missing_tables = missing_tables

    if not schema_ok:
        status.error = f"Missing required tables: {', '.join(missing_tables)}"
        status.warnings = warnings
        return status

    # Collect stats
    try:
        stats = DatabaseStats.collect()
        status.stats = stats.to_dict()
    except DatabaseException as e:
        warnings.append(f"Could not collect stats: {e}")

    # All checks passed
    status.healthy = True
    status.warnings = warnings
    return status


def require_healthy(fail_fast: bool = True) -> HealthStatus:
    """Check health and optionally exit if unhealthy.

    Args:
        fail_fast: If True, exit with error code if unhealthy

    Returns:
        HealthStatus

    Raises:
        SystemExit: If fail_fast and unhealthy
    """
    status = run_health_check()

    if status.healthy:
        logger.info("Health check passed")
        if status.warnings:
            for warning in status.warnings:
                logger.warning(warning)
        if status.stats:
            logger.info(f"Database stats: {status.stats}")
    else:
        logger.error(f"Health check failed: {status.error}")
        if fail_fast:
            sys.exit(1)

    return status


def validate_environment() -> None:
    """Validate that all required environment variables are set.

    Raises:
        ConfigException: If required variables are missing
    """
    env_ok, missing_required, _ = check_env_vars()
    if not env_ok:
        raise ConfigException(f"Missing required environment variables: {', '.join(missing_required)}")


def validate_database() -> None:
    """Validate database connection and schema.

    Raises:
        DatabaseException: If connection fails or schema is invalid
    """
    # Check connection
    db_ok, db_error = check_database_connection()
    if not db_ok:
        raise DatabaseException(f"Database connection failed: {db_error}")

    # Check schema
    schema_ok, missing_tables = check_schema()
    if not schema_ok:
        raise DatabaseException(f"Database schema invalid. Missing tables: {', '.join(missing_tables)}")


def startup_checks(fail_fast: bool = True) -> HealthStatus:
    """Run all startup checks.

    This should be called at service startup to ensure everything is ready.

    Args:
        fail_fast: If True, exit with error code on failure

    Returns:
        HealthStatus
    """
    logger.info("Running startup health checks...")

    status = run_health_check()

    if status.healthy:
        logger.info("All startup checks passed")
        logger.info("  Database: connected")
        logger.info(f"  Schema: valid ({len(REQUIRED_TABLES)} tables)")
        logger.info(f"  pgvector: {'available' if status.pgvector_available else 'not available'}")

        if status.stats:
            logger.info(f"  Beliefs: {status.stats.get('beliefs', 0)}")
            logger.info(f"  Entities: {status.stats.get('entities', 0)}")
            logger.info(f"  Sessions: {status.stats.get('sessions', 0)}")

        if status.warnings:
            logger.warning("Warnings:")
            for warning in status.warnings:
                logger.warning(f"  - {warning}")
    else:
        logger.error("Startup checks FAILED")
        logger.error(f"  Error: {status.error}")

        if status.missing_env_vars:
            logger.error(f"  Missing env vars: {', '.join(status.missing_env_vars)}")

        if status.missing_tables:
            logger.error(f"  Missing tables: {', '.join(status.missing_tables)}")

        if fail_fast:
            logger.error("Exiting due to failed health checks")
            sys.exit(1)

    return status


def cli_health_check() -> int:
    """CLI entry point for health check.

    Returns:
        Exit code (0 for healthy, 1 for unhealthy)
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    status = run_health_check()

    print(f"Healthy: {status.healthy}")
    print(f"Database connected: {status.database_connected}")
    print(f"Schema valid: {status.schema_valid}")
    print(f"pgvector available: {status.pgvector_available}")
    print(f"Environment valid: {status.env_vars_present}")

    if status.missing_env_vars:
        print(f"Missing env vars: {', '.join(status.missing_env_vars)}")

    if status.missing_tables:
        print(f"Missing tables: {', '.join(status.missing_tables)}")

    if status.error:
        print(f"Error: {status.error}")

    if status.warnings:
        print("Warnings:")
        for warning in status.warnings:
            print(f"  - {warning}")

    if status.stats:
        print("Stats:")
        for key, value in status.stats.items():
            print(f"  {key}: {value}")

    return 0 if status.healthy else 1
