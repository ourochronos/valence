"""Core configuration - shared config interfaces for the valence package.

This module provides a layer-appropriate way for lower-level modules (like federation)
to access configuration without importing from higher-level modules (like server).

The server layer sets the config at startup; federation reads it through this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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
            "Federation config not initialized. "
            "Call set_federation_config() at application startup, "
            "or ensure server.config is imported first."
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
