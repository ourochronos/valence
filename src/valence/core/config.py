"""Core configuration - centralized config for the valence package.

All environment-based configuration should flow through this module.
This provides a single source of truth and consistent defaults.

Usage:
    from valence.core.config import get_config
    config = get_config()

    # Access settings
    db_host = config.db_host
    log_level = config.log_level
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    """Core configuration settings for Valence.

    Settings can be configured via environment variables.
    Most use VALENCE_ prefix, but database settings also support
    legacy VKB_ and PG* prefixes for backward compatibility.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ==========================================================================
    # DATABASE SETTINGS
    # ==========================================================================
    # Support both VKB_ (legacy) and VALENCE_ prefixes, plus PG* fallbacks

    db_host: str = Field(
        default="localhost",
        description="Database host",
        validation_alias="VKB_DB_HOST",
    )
    db_port: int = Field(
        default=5432,
        description="Database port",
        validation_alias="VKB_DB_PORT",
    )
    db_name: str = Field(
        default="valence",
        description="Database name",
        validation_alias="VKB_DB_NAME",
    )
    db_user: str = Field(
        default="valence",
        description="Database user",
        validation_alias="VKB_DB_USER",
    )
    db_password: str = Field(
        default="",
        description="Database password",
        validation_alias="VKB_DB_PASSWORD",
    )

    # Connection pool settings
    db_pool_min: int = Field(
        default=5,
        description="Minimum pool connections",
        validation_alias="VALENCE_DB_POOL_MIN",
    )
    db_pool_max: int = Field(
        default=20,
        description="Maximum pool connections",
        validation_alias="VALENCE_DB_POOL_MAX",
    )

    # ==========================================================================
    # EMBEDDING SETTINGS
    # ==========================================================================

    embedding_provider: str = Field(
        default="local",
        description="Embedding provider: 'local' or 'openai'",
        validation_alias="VALENCE_EMBEDDING_PROVIDER",
    )
    embedding_model_path: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local embedding model name or path",
        validation_alias="VALENCE_EMBEDDING_MODEL_PATH",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for local embeddings: 'cpu' or 'cuda'",
        validation_alias="VALENCE_EMBEDDING_DEVICE",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for embeddings",
        validation_alias="OPENAI_API_KEY",
    )

    # ==========================================================================
    # LOGGING SETTINGS
    # ==========================================================================

    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
        validation_alias="VALENCE_LOG_LEVEL",
    )
    log_format: str = Field(
        default="",
        description="Log format: 'json', 'text', or '' (auto-detect)",
        validation_alias="VALENCE_LOG_FORMAT",
    )
    log_file: str | None = Field(
        default=None,
        description="Log file path (optional)",
        validation_alias="VALENCE_LOG_FILE",
    )

    # ==========================================================================
    # CACHE SETTINGS
    # ==========================================================================

    cache_max_size: int = Field(
        default=1000,
        description="Maximum size for in-memory LRU caches",
        validation_alias="VALENCE_CACHE_MAX_SIZE",
    )

    # ==========================================================================
    # SEED NODE SETTINGS (for federation bootstrap)
    # ==========================================================================

    seed_host: str | None = Field(
        default=None,
        description="Seed node host override",
        validation_alias="VALENCE_SEED_HOST",
    )
    seed_port: int = Field(
        default=8470,
        description="Seed node port",
        validation_alias="VALENCE_SEED_PORT",
    )
    seed_id: str | None = Field(
        default=None,
        description="Seed node ID override",
        validation_alias="VALENCE_SEED_ID",
    )
    seed_peers: str | None = Field(
        default=None,
        description="Comma-separated list of seed peers",
        validation_alias="VALENCE_SEED_PEERS",
    )

    # ==========================================================================
    # FEDERATION IDENTITY SETTINGS (used by CLI and peer_sync)
    # ==========================================================================

    federation_private_key: str | None = Field(
        default=None,
        description="Ed25519 private key hex for signing",
        validation_alias="VALENCE_FEDERATION_PRIVATE_KEY",
    )
    federation_public_key: str | None = Field(
        default=None,
        description="Ed25519 public key in multibase format",
        validation_alias="VALENCE_FEDERATION_PUBLIC_KEY",
    )
    federation_did: str | None = Field(
        default=None,
        description="Node DID for federation",
        validation_alias="VALENCE_FEDERATION_DID",
    )
    trust_registry_path: str | None = Field(
        default=None,
        description="Path to trust registry file",
        validation_alias="VALENCE_TRUST_REGISTRY",
    )

    # ==========================================================================
    # FEDERATION SECURITY SETTINGS
    # ==========================================================================

    require_tls: bool = Field(
        default=False,
        description="Require TLS (HTTPS) for all federation peer connections. Set to true in production to prevent MITM attacks.",
        validation_alias="VALENCE_REQUIRE_TLS",
    )
    federation_require_auth: bool = Field(
        default=False,
        description="Require authentication for all federation/sync requests. When true, unauthenticated federation requests are rejected with 401.",
        validation_alias="VALENCE_FEDERATION_REQUIRE_AUTH",
    )

    # ==========================================================================
    # HEALTH CHECK SETTINGS
    # ==========================================================================

    # Environment variables to check in health checks
    # (not actual settings, but documented here for consistency)
    # These are checked via os.environ in health.py

    # ==========================================================================
    # COMPUTED PROPERTIES
    # ==========================================================================

    @property
    def database_url(self) -> str:
        """Construct database URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def connection_params(self) -> dict:
        """Get database connection parameters dict."""
        return {
            "host": self.db_host,
            "port": self.db_port,
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
        }

    @property
    def pool_config(self) -> dict:
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


def clear_config_cache() -> None:
    """Clear the config cache. Useful for testing."""
    global _config
    _config = None


# ==========================================================================
# FEDERATION CONFIG PROTOCOL (for backward compatibility)
# ==========================================================================


@runtime_checkable
class FederationConfigProtocol(Protocol):
    """Protocol defining federation configuration requirements.

    This allows federation to depend on a config interface rather than
    the concrete server.config.ServerSettings class.
    """

    @property
    def federation_node_did(self) -> str | None:
        """Node DID for this federation instance."""
        ...

    @property
    def federation_private_key(self) -> str | None:
        """Ed25519 private key hex for signing."""
        ...

    @property
    def federation_sync_interval_seconds(self) -> int:
        """Interval between sync operations."""
        ...

    @property
    def port(self) -> int:
        """Server port (used as fallback for DID generation)."""
        ...


@dataclass
class FederationConfig:
    """Concrete federation configuration.

    This can be instantiated directly for testing or populated from ServerSettings.
    """

    federation_node_did: str | None = None
    federation_private_key: str | None = None
    federation_sync_interval_seconds: int = 300
    port: int = 8420


# Global federation config - set by server layer at startup
_federation_config: FederationConfigProtocol | None = None


def set_federation_config(config: FederationConfigProtocol) -> None:
    """Set the global federation config.

    Called by server layer at startup to inject its settings.

    Args:
        config: An object implementing FederationConfigProtocol
                (typically ServerSettings from server.config)
    """
    global _federation_config
    _federation_config = config


def get_federation_config() -> FederationConfigProtocol:
    """Get the global federation config.

    Returns:
        The configured federation settings.

    Raises:
        RuntimeError: If federation config hasn't been set yet.
    """
    if _federation_config is None:
        raise RuntimeError(
            "Federation config not initialized. Call set_federation_config() at application startup, or ensure server.config is imported first."
        )
    return _federation_config


def get_federation_config_or_none() -> FederationConfigProtocol | None:
    """Get the global federation config, or None if not set.

    Useful for code that needs to check if federation is configured
    without raising an exception.

    Returns:
        The configured federation settings, or None.
    """
    return _federation_config


def clear_federation_config() -> None:
    """Clear the global federation config.

    Primarily for testing purposes.
    """
    global _federation_config
    _federation_config = None
