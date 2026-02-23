"""Centralized configuration registry for the Valence v2 knowledge system.

``ConfigRegistry`` is the single source of truth for all runtime configuration.
It reads from environment variables matching existing VKB_DB_* / ORO_DB_* patterns
so that existing deployment configs continue to work unchanged.

Usage::

    from valence.core.config_registry import ConfigRegistry

    cfg = ConfigRegistry.from_env()
    print(cfg.db_host, cfg.db_port, cfg.db_name)

Implements WU-14 (C12, DR-10).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ConfigRegistry:
    """Single source of truth for all Valence configuration.

    Covers DB connection, embedding provider, and system-level parameters.
    Reads from environment variables via :meth:`from_env`.

    All fields have sensible defaults for local development.
    """

    # -----------------------------------------------------------------------
    # Database
    # -----------------------------------------------------------------------

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "valence"
    db_user: str = "valence"
    db_password: str = ""

    # Connection pool
    db_pool_min: int = 5
    db_pool_max: int = 20

    # -----------------------------------------------------------------------
    # Embedding
    # -----------------------------------------------------------------------

    embedding_provider: str = "local"
    embedding_model_path: str = "BAAI/bge-small-en-v1.5"
    embedding_dims: int = 384
    embedding_device: str = "cpu"
    openai_api_key: str = ""

    # -----------------------------------------------------------------------
    # Async embedding flag (WU-12)
    # -----------------------------------------------------------------------

    async_embeddings: bool = False

    # -----------------------------------------------------------------------
    # Usage / forgetting
    # -----------------------------------------------------------------------

    usage_decay_rate: float = 0.01  # λ per day — half-life ≈ 69 days

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    log_level: str = "INFO"

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ConfigRegistry:
        """Load configuration from environment variables.

        Variable priority (first match wins):

        DB:
          - ``VKB_DB_HOST`` / ``ORO_DB_HOST``
          - ``VKB_DB_PORT`` / ``ORO_DB_PORT``
          - ``VKB_DB_NAME`` / ``ORO_DB_NAME``
          - ``VKB_DB_USER`` / ``ORO_DB_USER``
          - ``VKB_DB_PASSWORD`` / ``ORO_DB_PASSWORD``

        Embedding:
          - ``VALENCE_EMBEDDING_PROVIDER``
          - ``VALENCE_EMBEDDING_MODEL_PATH``
          - ``VALENCE_EMBEDDING_DIMS``
          - ``VALENCE_EMBEDDING_DEVICE``
          - ``OPENAI_API_KEY``

        Other:
          - ``VALENCE_ASYNC_EMBEDDINGS`` (truthy string → True)
          - ``VALENCE_USAGE_DECAY_RATE``
          - ``VALENCE_LOG_LEVEL``
          - ``VALENCE_DB_POOL_MIN`` / ``VALENCE_DB_POOL_MAX``

        Returns:
            Populated ConfigRegistry instance.
        """

        def _get(*keys: str, default: str = "") -> str:
            for key in keys:
                val = os.environ.get(key)
                if val is not None:
                    return val
            return default

        def _get_int(*keys: str, default: int) -> int:
            val = _get(*keys)
            if val:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
            return default

        def _get_float(*keys: str, default: float) -> float:
            val = _get(*keys)
            if val:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
            return default

        def _get_bool(*keys: str, default: bool = False) -> bool:
            val = _get(*keys).strip().lower()
            if val in ("1", "true", "yes", "on"):
                return True
            if val in ("0", "false", "no", "off", ""):
                return default
            return default

        return cls(
            db_host=_get("VKB_DB_HOST", "ORO_DB_HOST", default="localhost"),
            db_port=_get_int("VKB_DB_PORT", "ORO_DB_PORT", default=5432),
            db_name=_get("VKB_DB_NAME", "ORO_DB_NAME", default="valence"),
            db_user=_get("VKB_DB_USER", "ORO_DB_USER", default="valence"),
            db_password=_get("VKB_DB_PASSWORD", "ORO_DB_PASSWORD", default=""),
            db_pool_min=_get_int("VALENCE_DB_POOL_MIN", default=5),
            db_pool_max=_get_int("VALENCE_DB_POOL_MAX", default=20),
            embedding_provider=_get("VALENCE_EMBEDDING_PROVIDER", default="local"),
            embedding_model_path=_get("VALENCE_EMBEDDING_MODEL_PATH", default="BAAI/bge-small-en-v1.5"),
            embedding_dims=_get_int("VALENCE_EMBEDDING_DIMS", default=384),
            embedding_device=_get("VALENCE_EMBEDDING_DEVICE", default="cpu"),
            openai_api_key=_get("OPENAI_API_KEY", default=""),
            async_embeddings=_get_bool("VALENCE_ASYNC_EMBEDDINGS"),
            usage_decay_rate=_get_float("VALENCE_USAGE_DECAY_RATE", default=0.01),
            log_level=_get("VALENCE_LOG_LEVEL", default="INFO"),
        )

    # -----------------------------------------------------------------------
    # Computed helpers
    # -----------------------------------------------------------------------

    @property
    def database_url(self) -> str:
        """PostgreSQL DSN string."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def connection_params(self) -> dict:
        """Parameters dict compatible with psycopg2.connect(**params)."""
        return {
            "host": self.db_host,
            "port": self.db_port,
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
        }

    @property
    def pool_config(self) -> dict:
        """Pool size dict compatible with psycopg2.pool.*ConnectionPool."""
        return {
            "minconn": self.db_pool_min,
            "maxconn": self.db_pool_max,
        }


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_registry: ConfigRegistry | None = None


def get_config_registry() -> ConfigRegistry:
    """Return the module-level singleton ConfigRegistry.

    Instantiated from environment on first call; subsequent calls return the
    cached instance.  Call :func:`reset_config_registry` to force a reload
    (useful in tests).
    """
    global _registry
    if _registry is None:
        _registry = ConfigRegistry.from_env()
    return _registry


def reset_config_registry() -> None:
    """Clear the cached registry.  Useful for testing environment changes."""
    global _registry
    _registry = None
