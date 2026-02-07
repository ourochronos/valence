"""Transport configuration for Valence P2P networking.

Centralises all transport-related settings and supports environment
variable overrides via ``VALENCE_TRANSPORT_*`` prefix.

Issue #300 — P2P: Integrate py-libp2p as transport backend.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class TransportType(StrEnum):
    """Supported transport backends."""

    LIBP2P = "libp2p"
    LEGACY = "legacy"


# Default listen address: all interfaces, TCP 4001 (libp2p convention)
DEFAULT_LISTEN_ADDRS: list[str] = ["/ip4/0.0.0.0/tcp/4001"]

# Well-known bootstrap peers (empty by default — configure per deployment)
DEFAULT_BOOTSTRAP_PEERS: list[str] = []


@dataclass
class TransportConfig:
    """Configuration for the Valence transport layer.

    All fields can be overridden via environment variables with the
    ``VALENCE_TRANSPORT_`` prefix.  Lists are JSON-encoded strings.

    Examples::

        export VALENCE_TRANSPORT_TYPE=libp2p
        export VALENCE_TRANSPORT_LISTEN_ADDRS='["/ip4/0.0.0.0/tcp/9000"]'
        export VALENCE_TRANSPORT_BOOTSTRAP_PEERS='["/ip4/1.2.3.4/tcp/4001/p2p/QmPeer"]'
        export VALENCE_TRANSPORT_DHT_ENABLED=true
    """

    # -- backend selection ---------------------------------------------------
    transport_type: TransportType = TransportType.LIBP2P

    # -- network addresses ---------------------------------------------------
    listen_addrs: list[str] = field(default_factory=lambda: list(DEFAULT_LISTEN_ADDRS))
    bootstrap_peers: list[str] = field(default_factory=lambda: list(DEFAULT_BOOTSTRAP_PEERS))

    # -- feature toggles -----------------------------------------------------
    dht_enabled: bool = True
    gossipsub_enabled: bool = True
    relay_enabled: bool = True

    # -- libp2p-specific tunables -------------------------------------------
    gossipsub_degree: int = 6
    gossipsub_degree_low: int = 4
    gossipsub_degree_high: int = 12
    gossipsub_heartbeat_interval: float = 1.0  # seconds

    # -- key management ------------------------------------------------------
    # Path to an Ed25519 private key file (PEM).  If unset, a new key is
    # generated on each start (fine for dev, bad for production).
    private_key_path: str | None = None

    # -- misc ----------------------------------------------------------------
    connection_timeout: float = 30.0  # seconds
    max_peers: int = 50

    # -- extra ---------------------------------------------------------------
    extra: dict[str, Any] = field(default_factory=dict)

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_env(cls, prefix: str = "VALENCE_TRANSPORT_") -> TransportConfig:
        """Build a config from environment variables.

        Environment variables are upper-cased field names prefixed with
        *prefix*.  Boolean fields accept ``true/1/yes`` (case-insensitive).
        List fields expect JSON arrays.

        Unrecognised ``VALENCE_TRANSPORT_*`` vars are silently ignored
        so that downstream tools can set their own without breaking us.
        """
        kwargs: dict[str, Any] = {}

        # transport_type
        raw = os.environ.get(f"{prefix}TYPE")
        if raw:
            try:
                kwargs["transport_type"] = TransportType(raw.lower())
            except ValueError:
                logger.warning("Unknown transport type %r, falling back to default", raw)

        # listen_addrs / bootstrap_peers (JSON lists)
        for list_field in ("LISTEN_ADDRS", "BOOTSTRAP_PEERS"):
            raw = os.environ.get(f"{prefix}{list_field}")
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        kwargs[list_field.lower()] = parsed
                    else:
                        logger.warning("%s%s is not a JSON array, ignoring", prefix, list_field)
                except json.JSONDecodeError:
                    logger.warning("%s%s is not valid JSON, ignoring", prefix, list_field)

        # boolean toggles
        for bool_field in ("DHT_ENABLED", "GOSSIPSUB_ENABLED", "RELAY_ENABLED"):
            raw = os.environ.get(f"{prefix}{bool_field}")
            if raw is not None:
                kwargs[bool_field.lower()] = raw.lower() in ("true", "1", "yes")

        # numeric tunables
        for num_field, cast in [
            ("GOSSIPSUB_DEGREE", int),
            ("GOSSIPSUB_DEGREE_LOW", int),
            ("GOSSIPSUB_DEGREE_HIGH", int),
            ("GOSSIPSUB_HEARTBEAT_INTERVAL", float),
            ("CONNECTION_TIMEOUT", float),
            ("MAX_PEERS", int),
        ]:
            raw = os.environ.get(f"{prefix}{num_field}")
            if raw is not None:
                try:
                    kwargs[num_field.lower()] = cast(raw)
                except (ValueError, TypeError):
                    logger.warning("%s%s=%r is not a valid number, ignoring", prefix, num_field, raw)

        # private_key_path
        raw = os.environ.get(f"{prefix}PRIVATE_KEY_PATH")
        if raw:
            kwargs["private_key_path"] = raw

        return cls(**kwargs)

    def validate(self) -> list[str]:
        """Return a list of validation warnings (empty = all good)."""
        warnings: list[str] = []

        if not self.listen_addrs:
            warnings.append("No listen addresses configured — node will not accept connections")

        if self.gossipsub_degree_low > self.gossipsub_degree:
            warnings.append(f"gossipsub_degree_low ({self.gossipsub_degree_low}) > gossipsub_degree ({self.gossipsub_degree})")

        if self.gossipsub_degree > self.gossipsub_degree_high:
            warnings.append(f"gossipsub_degree ({self.gossipsub_degree}) > gossipsub_degree_high ({self.gossipsub_degree_high})")

        if self.max_peers < 1:
            warnings.append(f"max_peers ({self.max_peers}) must be >= 1")

        return warnings
