# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

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

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    """Core configuration settings for Valence.

    Settings can be configured via environment variables using the VALENCE_ prefix.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ==========================================================================
    # DATABASE SETTINGS
    # ==========================================================================

    db_host: str = Field(
        default="localhost",
        description="Database host",
        validation_alias="VALENCE_DB_HOST",
    )
    db_port: int = Field(
        default=5433,
        description="Database port",
        validation_alias="VALENCE_DB_PORT",
    )
    db_name: str = Field(
        default="valence",
        description="Database name",
        validation_alias="VALENCE_DB_NAME",
    )
    db_user: str = Field(
        default="valence",
        description="Database user",
        validation_alias="VALENCE_DB_USER",
    )
    db_password: str = Field(
        default="valence",
        description="Database password",
        validation_alias="VALENCE_DB_PASSWORD",
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
    db_pool_timeout: int = Field(
        default=30,
        description="Connection pool timeout in seconds",
        validation_alias="VALENCE_DB_POOL_TIMEOUT",
    )

    # ==========================================================================
    # EMBEDDING SETTINGS
    # ==========================================================================

    embedding_provider: str = Field(
        default="local",
        description="Embedding provider: 'local' or 'openai'",
        validation_alias="VALENCE_EMBEDDING_PROVIDER",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model name (local model path or OpenAI model name)",
        validation_alias="VALENCE_EMBEDDING_MODEL",
    )
    embedding_model_path: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local embedding model name or path",
        validation_alias="VALENCE_EMBEDDING_MODEL_PATH",
    )
    embedding_dims: int = Field(
        default=384,
        description="Embedding vector dimensions. Must match the configured model output. Source of truth is embedding_types table.",
        validation_alias="VALENCE_EMBEDDING_DIMS",
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
    # CURATION SETTINGS
    # ==========================================================================

    min_capture_confidence: float = Field(
        default=0.50,
        description="Minimum confidence threshold for auto-capture",
        validation_alias="VALENCE_MIN_CAPTURE_CONFIDENCE",
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
    # VALIDATORS
    # ==========================================================================

    @model_validator(mode="after")
    def _auto_select_embedding_provider(self) -> CoreSettings:
        """Auto-select openai embedding provider when API key is available."""
        # Only auto-select if no explicit provider was set (still default "local")
        # and an OpenAI API key is available
        if self.embedding_provider == "local" and self.openai_api_key:
            import os

            # Check if VALENCE_EMBEDDING_PROVIDER was explicitly set
            if not os.environ.get("VALENCE_EMBEDDING_PROVIDER"):
                self.embedding_provider = "openai"
        return self

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
