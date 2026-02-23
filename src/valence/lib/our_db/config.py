"""Core configuration - centralized config for our-db.

All environment-based configuration should flow through this module.
This provides a single source of truth and consistent defaults.

Usage:
    from our_db import get_config
    config = get_config()

    # Access settings
    db_host = config.db_host
    log_level = config.log_level
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    """Core configuration settings.

    Settings can be configured via environment variables.
    Database settings support ORO_DB_ prefix.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ==========================================================================
    # DATABASE SETTINGS
    # ==========================================================================

    db_host: str = Field(
        default="localhost",
        description="Database host",
        validation_alias="ORO_DB_HOST",
    )
    db_port: int = Field(
        default=5432,
        description="Database port",
        validation_alias="ORO_DB_PORT",
    )
    db_name: str = Field(
        default="postgres",
        description="Database name",
        validation_alias="ORO_DB_NAME",
    )
    db_user: str = Field(
        default="postgres",
        description="Database user",
        validation_alias="ORO_DB_USER",
    )
    db_password: str = Field(
        default="",
        description="Database password",
        validation_alias="ORO_DB_PASSWORD",
    )

    # Connection pool settings
    db_pool_min: int = Field(
        default=5,
        description="Minimum pool connections",
        validation_alias="ORO_DB_POOL_MIN",
    )
    db_pool_max: int = Field(
        default=20,
        description="Maximum pool connections",
        validation_alias="ORO_DB_POOL_MAX",
    )

    # ==========================================================================
    # LOGGING SETTINGS
    # ==========================================================================

    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
        validation_alias="ORO_LOG_LEVEL",
    )
    log_format: str = Field(
        default="",
        description="Log format: 'json', 'text', or '' (auto-detect)",
        validation_alias="ORO_LOG_FORMAT",
    )
    log_file: str | None = Field(
        default=None,
        description="Log file path (optional)",
        validation_alias="ORO_LOG_FILE",
    )

    # ==========================================================================
    # COMPUTED PROPERTIES
    # ==========================================================================

    @property
    def database_url(self) -> str:
        """Construct database URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def connection_params(self) -> dict[str, str | int]:
        """Get database connection parameters dict."""
        return {
            "host": self.db_host,
            "port": self.db_port,
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
        }

    @property
    def pool_config(self) -> dict[str, int]:
        """Get connection pool configuration."""
        return {
            "minconn": self.db_pool_min,
            "maxconn": self.db_pool_max,
        }


# ==========================================================================
# GLOBAL CONFIG INSTANCE (lazy loaded)
# ==========================================================================

_config: CoreSettings | None = None


def get_config() -> CoreSettings:
    """Get the global configuration instance.

    Returns:
        The singleton CoreSettings instance.
    """
    global _config
    if _config is None:
        _config = CoreSettings()
    return _config


def set_config(config: CoreSettings) -> None:
    """Set the global configuration instance.

    Useful for testing or custom configuration.

    Args:
        config: The CoreSettings instance to use.
    """
    global _config
    _config = config


def clear_config_cache() -> None:
    """Clear the config cache. Useful for testing."""
    global _config
    _config = None
