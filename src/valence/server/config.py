"""Server configuration using pydantic-settings."""

from __future__ import annotations

import logging
import secrets
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def get_package_version() -> str:
    """Get the package version from installed metadata.

    Returns the version from pyproject.toml when installed,
    or a dev fallback when running from source without install.
    """
    try:
        return version("ourochronos-valence")
    except PackageNotFoundError:
        return "0.0.0-dev"


class ServerSettings(BaseSettings):
    """Configuration for the Valence HTTP MCP server.

    Settings can be configured via environment variables with VALENCE_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="VALENCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="127.0.0.1", description="Host to bind to")
    port: int = Field(default=8420, description="Port to bind to")

    # External URL (for OAuth redirects) - if not set, constructed from host:port
    external_url: str | None = Field(
        default=None,
        description="External URL for OAuth (e.g., https://pod.zonk1024.info)",
    )

    # Token authentication (legacy Bearer tokens)
    token_file: Path = Field(
        default=Path("/opt/valence/config/tokens.json"),
        description="Path to token storage file",
    )

    # OAuth 2.1 settings
    oauth_enabled: bool = Field(default=True, description="Enable OAuth 2.1 authentication")
    oauth_jwt_secret: str | None = Field(
        default=None,
        description="Secret for signing JWTs (REQUIRED in production - set VALENCE_OAUTH_JWT_SECRET)",
    )
    oauth_jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    oauth_access_token_expiry: int = Field(default=3600, description="Access token expiry in seconds (default: 1 hour)")
    oauth_refresh_token_expiry: int = Field(
        default=86400 * 30,
        description="Refresh token expiry in seconds (default: 30 days)",
    )
    oauth_code_expiry: int = Field(
        default=600,
        description="Authorization code expiry in seconds (default: 10 minutes)",
    )

    # OAuth user credentials (simple single-user setup)
    oauth_username: str = Field(default="admin", description="OAuth login username")
    oauth_password: str = Field(
        default="",
        description="OAuth login password (REQUIRED for OAuth - set VALENCE_OAUTH_PASSWORD)",
    )

    # OAuth client storage
    oauth_clients_file: Path = Field(
        default=Path("/opt/valence/config/oauth_clients.json"),
        description="Path to OAuth clients storage file",
    )

    # Database settings (inherited from existing VKB config)
    db_host: str = Field(default="localhost", alias="VALENCE_DB_HOST")
    db_port: int = Field(default=5433, alias="VALENCE_DB_PORT")
    db_name: str = Field(default="valence", alias="VALENCE_DB_NAME")
    db_user: str = Field(default="valence", alias="VALENCE_DB_USER")
    db_password: str = Field(default="", alias="VALENCE_DB_PASSWORD")

    # OpenAI for embeddings
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # CORS settings
    allowed_origins: list[str] = Field(
        default=[],
        description="Allowed CORS origins. Empty = same-origin only. Set to ['*'] for development.",
    )

    # Logging
    log_format: str = Field(
        default="text",
        description="Log format: 'text' for human-readable, 'json' for structured JSON logging",
    )

    # Rate limiting (requests per minute per client)
    rate_limit_rpm: int = Field(default=60, description="Rate limit per minute")

    # Cache settings (Issue #147)
    cache_max_size: int = Field(
        default=1000,
        description="Maximum size for in-memory LRU caches (router_cache, failure_events, etc.)",
    )

    # Server name for MCP
    server_name: str = Field(default="valence", description="MCP server name")
    server_version: str = Field(default_factory=get_package_version, description="Server version")

    # Production mode flag (explicit override)
    production: bool = Field(
        default=False,
        description="Force production mode (stricter security requirements)",
    )

    # ==========================================================================
    # FEDERATION SETTINGS
    # ==========================================================================

    # Enable/disable federation
    federation_enabled: bool = Field(
        default=False,
        description="Enable federation protocol endpoints",
    )

    # Node identity
    federation_node_did: str | None = Field(
        default=None,
        description="Node DID (e.g., did:vkb:web:valence.example.com). If not set, derived from external_url.",
    )
    federation_node_name: str | None = Field(
        default=None,
        description="Human-readable node name",
    )

    # Cryptographic identity (Ed25519 public key in multibase format)
    federation_public_key: str | None = Field(
        default=None,
        description="Ed25519 public key in multibase format (z6Mk...)",
    )
    federation_private_key: str | None = Field(
        default=None,
        description="Ed25519 private key hex (for signing - keep secret!)",
    )

    # Capabilities
    federation_capabilities: list[str] = Field(
        default=["belief_sync"],
        description="Federation capabilities: belief_sync, aggregation_participate, aggregation_publish",
    )

    # Knowledge domains
    federation_domains: list[str] = Field(
        default=[],
        description="Knowledge domains this node specializes in",
    )

    # Trust anchor publishing
    federation_publish_trust_anchors: bool = Field(
        default=False,
        description="Publish trust anchors at /.well-known/vfp-trust-anchors",
    )

    # Bootstrap nodes for discovery
    federation_bootstrap_nodes: list[str] = Field(
        default=[],
        description="Initial nodes to connect to for federation (list of DIDs or URLs)",
    )

    # Sync settings
    federation_sync_interval_seconds: int = Field(
        default=300,
        description="Interval between sync operations (default: 5 minutes)",
    )
    federation_max_hop_count: int = Field(
        default=3,
        description="Maximum hops for belief propagation",
    )

    # Privacy settings
    federation_default_visibility: str = Field(
        default="private",
        description="Default visibility for new beliefs: private, trusted, federated, public",
    )
    federation_privacy_epsilon: float = Field(
        default=0.1,
        description="Differential privacy epsilon for aggregation queries",
    )
    federation_privacy_delta: float = Field(
        default=1e-6,
        description="Differential privacy delta for aggregation queries",
    )
    federation_min_aggregation_contributors: int = Field(
        default=5,
        description="Minimum contributors required for aggregation results",
    )
    federation_require_auth: bool = Field(
        default=False,
        description="Require authentication for all federation/sync requests. When true, unauthenticated federation requests are rejected with 401.",
    )

    @model_validator(mode="after")
    def validate_production_settings(self) -> ServerSettings:
        """Validate security settings for production environments.

        In production (when host is not localhost/127.0.0.1 or external_url is set),
        require explicit JWT secret configuration.
        """
        is_production = (
            self.external_url is not None
            or self.host not in ("localhost", "127.0.0.1", "0.0.0.0")  # nosec B104
            or self.production
        )

        if is_production and self.oauth_enabled:
            if not self.oauth_jwt_secret:
                logger.warning(
                    "VALENCE_OAUTH_JWT_SECRET not set — disabling OAuth. "
                    "Bearer token auth still works. Generate a secret with: "
                    "python -c 'import secrets; print(secrets.token_hex(32))'"
                )
                object.__setattr__(self, "oauth_enabled", False)
            elif len(self.oauth_jwt_secret) < 32:
                raise ValueError(
                    "VALENCE_OAUTH_JWT_SECRET must be at least 32 characters. "
                    "Generate a secure one with: python -c 'import secrets; print(secrets.token_hex(32))'"
                )

        # For development, generate a random secret if not provided
        if not self.oauth_jwt_secret:
            logger.warning("Auto-generating JWT secret - tokens will not persist across restarts")
            object.__setattr__(self, "oauth_jwt_secret", secrets.token_hex(32))

        return self

    @property
    def database_url(self) -> str:
        """Construct database URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def base_url(self) -> str:
        """Get the base URL for the server."""
        if self.external_url:
            return self.external_url.rstrip("/")
        return f"http://{self.host}:{self.port}"

    @property
    def issuer_url(self) -> str:
        """Get the OAuth issuer URL."""
        return self.base_url

    @property
    def mcp_resource_url(self) -> str:
        """Get the MCP resource URL."""
        return f"{self.base_url}/api/v1/mcp"


# Global settings instance - lazy loaded
_settings: ServerSettings | None = None


def get_settings() -> ServerSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = ServerSettings()
    return _settings


def clear_settings_cache() -> None:
    """Clear the settings cache. Useful for testing."""
    global _settings
    _settings = None
