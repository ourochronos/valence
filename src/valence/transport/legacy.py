"""Legacy transport adapter — wraps existing federation networking.

This module bridges the current Valence federation stack
(``valence.federation``) into the new :class:`TransportAdapter` interface
so that the existing networking "just works" through the pluggable layer
while new backends (libp2p, QUIC, …) are developed in parallel.

It is deliberately *thin*: every method delegates to the existing code and
performs only the minimal translation needed to satisfy the protocol.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from .adapter import (
    Connection,
    PeerInfo,
    TransportConfig,
    TransportState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal subscription bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _Subscription:
    """Tracks a single topic subscription."""

    topic: str
    handler: Callable[[str, bytes], Any]


# ---------------------------------------------------------------------------
# Legacy Transport Adapter
# ---------------------------------------------------------------------------


class LegacyTransportAdapter:
    """Wraps the existing federation transport to satisfy :class:`TransportAdapter`.

    Lifecycle
    ---------
    * ``start()``  — initialises the peer store, optionally contacts
      bootstrap peers via the federation discovery module.
    * ``stop()``   — tears down connections and clears subscriptions.

    Messaging
    ---------
    * ``send()`` uses the federation protocol handler to deliver a message
      to a peer endpoint over HTTP (the current federation wire format).
    * ``broadcast()`` fans out to all known peers.
    * ``subscribe()`` records a handler that ``broadcast`` invokes locally
      (full gossip is not yet implemented in the federation layer).

    Discovery
    ---------
    * ``discover_peers()`` delegates to the peer exchange protocol and the
      in-memory peer store.
    """

    def __init__(self, config: TransportConfig | None = None) -> None:
        self._config = config or TransportConfig(backend="legacy")
        self._state: TransportState = TransportState.STOPPED
        self._subscriptions: dict[str, list[_Subscription]] = {}
        self._connections: dict[str, Connection] = {}  # peer_id → Connection
        self._peer_store: Any = None  # Lazy — avoids importing federation at module level

    # -- properties --------------------------------------------------------

    @property
    def state(self) -> TransportState:
        return self._state

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Initialise the legacy transport."""
        if self._state == TransportState.RUNNING:
            return

        self._state = TransportState.STARTING
        try:
            # Lazy-import so the transport package can be loaded without
            # pulling in the full federation dependency graph at import time.
            from oro_federation.peers import get_peer_store

            self._peer_store = get_peer_store()

            # If bootstrap peers were provided, register them.
            for addr in self._config.bootstrap_peers:
                # Minimal registration — just ensure the peer store knows the
                # address.  We use the address itself as a provisional DID
                # because the legacy layer identifies peers by DID.
                self._peer_store.add_peer(
                    did=addr,
                    endpoint=addr,
                    public_key_multibase="",
                )

            self._state = TransportState.RUNNING
            logger.info("Legacy transport started (peers: %d)", len(self._peer_store.list_peers()))
        except Exception:
            self._state = TransportState.ERROR
            logger.exception("Failed to start legacy transport")
            raise

    async def stop(self) -> None:
        """Shut down the legacy transport."""
        if self._state == TransportState.STOPPED:
            return

        self._state = TransportState.STOPPING

        # Close tracked connections
        for conn in list(self._connections.values()):
            try:
                await conn.close()
            except Exception:
                logger.debug("Error closing connection to %s", conn.peer_id, exc_info=True)
        self._connections.clear()
        self._subscriptions.clear()

        self._state = TransportState.STOPPED
        logger.info("Legacy transport stopped")

    # -- connections -------------------------------------------------------

    async def connect(self, peer_id: str, addrs: list[str]) -> Connection:
        """Create a logical connection to a peer.

        The legacy federation layer is stateless HTTP, so "connecting"
        simply records the peer in the store and returns a
        :class:`Connection` handle.
        """
        if self._state != TransportState.RUNNING:
            raise RuntimeError("Transport is not running")

        addr = addrs[0] if addrs else ""
        if self._peer_store is not None:
            self._peer_store.add_peer(
                did=peer_id,
                endpoint=addr,
                public_key_multibase="",
            )

        conn = Connection(
            peer_id=peer_id,
            remote_addr=addr,
            local_addr=self._config.node_id,
            _handle={"peer_id": peer_id, "addr": addr},
        )
        self._connections[peer_id] = conn
        return conn

    async def listen(self, addr: str) -> AsyncIterator[Connection]:
        """Yield incoming connections.

        The legacy federation layer uses an HTTP server (Starlette) rather
        than a raw listener, so this is a no-op iterator that never yields.
        A future version could wrap the federation server's request cycle
        into Connection objects.
        """
        # Yield nothing — the federation server handles inbound requests
        # through its own HTTP endpoint machinery.
        return
        yield  # make this a valid async generator  # noqa: RET504

    # -- messaging ---------------------------------------------------------

    async def send(self, peer_id: str, protocol: str, message: bytes) -> bytes:
        """Send *message* to *peer_id* and return the response.

        Delegates to an HTTP POST against the peer's federation endpoint.
        The ``protocol`` parameter selects the federation protocol path
        (e.g. ``"share_belief"``, ``"sync_request"``).
        """
        if self._state != TransportState.RUNNING:
            raise RuntimeError("Transport is not running")

        # Resolve endpoint
        peer = self._peer_store.get_peer(peer_id) if self._peer_store else None
        endpoint = peer.endpoint if peer else peer_id  # fall back to raw id

        import aiohttp

        url = f"{endpoint.rstrip('/')}/federation/{protocol}"
        try:
            timeout = aiohttp.ClientTimeout(total=self._config.send_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    data=message,
                    headers={"Content-Type": "application/octet-stream"},
                ) as resp:
                    return await resp.read()
        except Exception as exc:
            logger.warning("send to %s failed: %s", peer_id, exc)
            raise

    async def broadcast(self, topic: str, message: bytes) -> None:
        """Publish *message* to all local subscribers of *topic*.

        Full gossip dissemination is not yet supported by the legacy
        layer — this only invokes locally registered handlers.
        """
        handlers = self._subscriptions.get(topic, [])
        for sub in handlers:
            try:
                result = sub.handler(self._config.node_id, message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Broadcast handler error for topic %s", topic)

    async def subscribe(self, topic: str, handler: Callable[[str, bytes], Any]) -> None:
        """Register *handler* for *topic*."""
        sub = _Subscription(topic=topic, handler=handler)
        self._subscriptions.setdefault(topic, []).append(sub)
        logger.debug("Subscribed to topic %s (total handlers: %d)", topic, len(self._subscriptions[topic]))

    # -- discovery ---------------------------------------------------------

    async def discover_peers(self) -> list[PeerInfo]:
        """Return currently known peers from the federation peer store."""
        if self._peer_store is None:
            return []

        peers: list[PeerInfo] = []
        for p in self._peer_store.list_peers():
            peers.append(
                PeerInfo(
                    peer_id=p.did,
                    addrs=[p.endpoint] if p.endpoint else [],
                    metadata={
                        "name": p.name,
                        "trust_score": p.trust_score,
                    },
                    last_seen=p.last_seen if hasattr(p, "last_seen") else None,
                )
            )
        return peers
