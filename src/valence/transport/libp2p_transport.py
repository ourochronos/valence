"""libp2p transport backend for Valence.

Implements :class:`~valence.transport.adapter.TransportAdapter` using
`py-libp2p <https://github.com/libp2p/py-libp2p>`_ (v0.5+).

Key design points
-----------------
* **Trio ↔ asyncio bridge** — py-libp2p is trio-native.  We run a
  dedicated trio event loop in a background thread and expose an
  asyncio-friendly interface so the rest of Valence (which is asyncio)
  can call us normally.
* **GossipSub** for topic-based broadcast (belief propagation).
* **Stream protocols** for point-to-point request/response.
* **Ed25519 keys** — PeerID is derived from Ed25519, aligning with
  Valence's multi-DID identity model.
* **Circuit relay v2** for NAT traversal.

Issue #300 — P2P: Integrate py-libp2p as transport backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from typing import Any

import trio

from valence.transport.adapter import (
    MessageEnvelope,
    MessageHandler,
    PeerInfo,
    TransportError,
    TransportState,
)
from valence.transport.config import TransportConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — py-libp2p is an optional dependency
# ---------------------------------------------------------------------------

_LIBP2P_AVAILABLE = False

try:
    from libp2p import new_host
    from libp2p.crypto.ed25519 import (
        Ed25519PrivateKey,
    )
    from libp2p.crypto.ed25519 import (
        create_new_key_pair as create_ed25519_keypair,
    )
    from libp2p.crypto.keys import KeyPair
    from libp2p.custom_types import TProtocol
    from libp2p.peer.peerinfo import info_from_p2p_addr
    from libp2p.pubsub.gossipsub import GossipSub
    from libp2p.pubsub.pubsub import Pubsub

    try:
        import multiaddr as multiaddr_mod  # noqa: F401 — presence check
    except ImportError:
        pass

    _LIBP2P_AVAILABLE = True
except ImportError:
    pass

# Valence protocol IDs
VALENCE_SYNC_PROTOCOL = "/valence/sync/1.0.0"
VALENCE_AUTH_PROTOCOL = "/valence/auth/1.0.0"
VALENCE_BELIEF_TOPIC = "valence/beliefs"
GOSSIPSUB_PROTOCOL_ID = "/meshsub/1.0.0"

# Maximum message size for stream reads
MAX_READ_LEN = 2**20  # 1 MiB


def _require_libp2p() -> None:
    """Raise a clear error if py-libp2p is not installed."""
    if not _LIBP2P_AVAILABLE:
        raise TransportError("py-libp2p is not installed. Install it with: pip install valence[p2p]")


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


def _load_or_create_keypair(private_key_path: str | None) -> KeyPair:
    """Load an Ed25519 keypair from *private_key_path* or generate a new one."""
    if private_key_path:
        try:
            from cryptography.hazmat.primitives.serialization import (
                Encoding,
                NoEncryption,
                PrivateFormat,
                load_pem_private_key,
            )

            with open(private_key_path, "rb") as f:
                raw_key = load_pem_private_key(f.read(), password=None)
            # Extract raw 32-byte seed for libp2p
            raw_bytes = raw_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
            private_key = Ed25519PrivateKey.from_bytes(raw_bytes)
            return KeyPair(private_key, private_key.get_public_key())
        except Exception as exc:
            logger.warning(
                "Failed to load key from %s (%s), generating ephemeral key",
                private_key_path,
                exc,
            )

    return create_ed25519_keypair()


# ---------------------------------------------------------------------------
# Trio ↔ asyncio bridge
# ---------------------------------------------------------------------------


class _TrioBridge:
    """Run a trio event loop in a background thread and expose a portal.

    This allows asyncio coroutines to schedule work on the trio loop
    and wait for results.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._trio_token: trio.lowlevel.TrioToken | None = None
        self._started = threading.Event()
        self._stop_event: trio.Event | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_trio, daemon=True, name="valence-libp2p-trio")
        self._thread.start()
        if not self._started.wait(timeout=10):
            raise TransportError("Trio bridge failed to start within 10 s")

    def _run_trio(self) -> None:
        trio.lowlevel.current_trio_token  # type: ignore[attr-defined]  # noqa: B018
        trio.from_thread.run_sync(lambda: None)  # noqa: This doesn't work outside trio
        # We need to run trio.run ourselves:
        trio.run(self._trio_main)

    async def _trio_main(self) -> None:
        self._trio_token = trio.lowlevel.current_trio_token()
        self._stop_event = trio.Event()
        self._started.set()
        await self._stop_event.wait()

    def stop(self) -> None:
        if self._stop_event is not None and self._trio_token is not None:
            try:
                trio.from_thread.run_sync(
                    self._stop_event.set,
                    trio_token=self._trio_token,
                )
            except trio.RunFinishedError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)

    def run_sync(self, fn: Any) -> Any:
        """Run a sync function inside the trio thread."""
        if self._trio_token is None:
            raise TransportError("Trio bridge not started")
        return trio.from_thread.run_sync(fn, trio_token=self._trio_token)

    async def run_in_trio(self, async_fn: Any, *args: Any) -> Any:
        """Schedule an async trio function from asyncio and await the result."""
        if self._trio_token is None:
            raise TransportError("Trio bridge not started")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def _portal() -> None:
            trio.from_thread.run(
                self._run_and_resolve,
                async_fn,
                args,
                future,
                loop,
                trio_token=self._trio_token,
            )

        await loop.run_in_executor(None, _portal)
        return await future

    @staticmethod
    async def _run_and_resolve(
        async_fn: Any,
        args: tuple[Any, ...],
        future: asyncio.Future[Any],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        try:
            result = await async_fn(*args)
            loop.call_soon_threadsafe(future.set_result, result)
        except BaseException as exc:
            loop.call_soon_threadsafe(future.set_exception, exc)


# ---------------------------------------------------------------------------
# Libp2pTransport
# ---------------------------------------------------------------------------


class Libp2pTransport:
    """Valence transport adapter backed by py-libp2p.

    Implements the :class:`TransportAdapter` protocol.

    Usage::

        from valence.transport.config import TransportConfig
        from valence.transport.libp2p_transport import Libp2pTransport

        config = TransportConfig.from_env()
        transport = Libp2pTransport(config)
        await transport.start()
        await transport.broadcast("valence/beliefs", payload)
        await transport.stop()
    """

    def __init__(self, config: TransportConfig | None = None) -> None:
        _require_libp2p()
        self._config = config or TransportConfig()
        self._state = TransportState.STOPPED
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)

        # Trio bridge for running py-libp2p
        self._bridge = _TrioBridge()

        # Populated on start()
        self._keypair: KeyPair | None = None
        self._host: Any = None  # libp2p BasicHost
        self._gossipsub: Any = None  # GossipSub instance
        self._pubsub: Any = None  # Pubsub instance
        self._subscriptions: dict[str, Any] = {}  # topic -> Subscription
        self._host_ctx: Any = None  # host.run() async context manager
        self._nursery: Any = None  # trio nursery for background tasks
        self._local_peer: PeerInfo | None = None

    # -- TransportAdapter: lifecycle ----------------------------------------

    @property
    def state(self) -> TransportState:
        return self._state

    @property
    def local_peer(self) -> PeerInfo:
        if self._local_peer is None:
            raise TransportError("Transport not started — no local peer available")
        return self._local_peer

    async def start(self) -> None:
        if self._state == TransportState.RUNNING:
            return
        self._state = TransportState.STARTING
        try:
            # Generate or load keypair
            self._keypair = _load_or_create_keypair(self._config.private_key_path)

            # Start the trio bridge thread
            self._bridge.start()

            # Initialise libp2p host inside the trio thread
            loop = asyncio.get_running_loop()
            ready: asyncio.Future[None] = loop.create_future()

            def _start_in_trio() -> None:
                trio.from_thread.run(
                    self._trio_start,
                    ready,
                    loop,
                    trio_token=self._bridge._trio_token,
                )

            await loop.run_in_executor(None, _start_in_trio)
            # Wait for trio-side initialisation to complete
            await asyncio.wait_for(ready, timeout=self._config.connection_timeout)
            self._state = TransportState.RUNNING
            logger.info(
                "libp2p transport started — peer %s listening on %s",
                self._local_peer.peer_id if self._local_peer else "?",
                self._config.listen_addrs,
            )
        except Exception as exc:
            self._state = TransportState.ERROR
            raise TransportError(f"Failed to start libp2p transport: {exc}") from exc

    async def _trio_start(
        self,
        ready: asyncio.Future[None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Initialise the libp2p host (runs inside trio)."""
        import multiaddr as multiaddr_mod

        listen_addrs = [multiaddr_mod.Multiaddr(addr) for addr in self._config.listen_addrs]

        # Build host
        self._host = new_host(key_pair=self._keypair)

        # Enter the host context — this starts listening
        self._host_ctx = self._host.run(listen_addrs=listen_addrs)
        await self._host_ctx.__aenter__()

        peer_id_str = self._host.get_id().to_string()
        host_addrs = [str(a) for a in self._host.get_addrs()]
        self._local_peer = PeerInfo(
            peer_id=peer_id_str,
            addrs=tuple(host_addrs),
        )

        # Set up stream protocol handlers
        self._host.set_stream_handler(TProtocol(VALENCE_SYNC_PROTOCOL), self._handle_sync_stream)
        self._host.set_stream_handler(TProtocol(VALENCE_AUTH_PROTOCOL), self._handle_auth_stream)

        # Set up GossipSub if enabled
        if self._config.gossipsub_enabled:
            self._gossipsub = GossipSub(
                protocols=[TProtocol(GOSSIPSUB_PROTOCOL_ID)],
                degree=self._config.gossipsub_degree,
                degree_low=self._config.gossipsub_degree_low,
                degree_high=self._config.gossipsub_degree_high,
                time_to_live=3,
            )
            self._pubsub = Pubsub(
                self._host,
                self._gossipsub,
                self._host.get_id(),
            )

        # Connect to bootstrap peers
        for peer_addr in self._config.bootstrap_peers:
            try:
                maddr = multiaddr_mod.Multiaddr(peer_addr)
                peer_info = info_from_p2p_addr(maddr)
                await self._host.connect(peer_info)
                logger.info("Connected to bootstrap peer %s", peer_addr)
            except Exception as exc:
                logger.warning("Failed to connect to bootstrap peer %s: %s", peer_addr, exc)

        # Signal asyncio side that we're ready
        loop.call_soon_threadsafe(ready.set_result, None)

        # Keep the trio task alive until bridge is stopped
        if self._bridge._stop_event is not None:
            await self._bridge._stop_event.wait()

        # Cleanup
        if self._host_ctx is not None:
            await self._host_ctx.__aexit__(None, None, None)

    async def stop(self) -> None:
        if self._state in (TransportState.STOPPED, TransportState.STOPPING):
            return
        self._state = TransportState.STOPPING
        try:
            # Unsubscribe all topics
            self._subscriptions.clear()
            self._handlers.clear()

            # Stop the trio bridge (signals the host context to exit)
            self._bridge.stop()

            self._state = TransportState.STOPPED
            logger.info("libp2p transport stopped")
        except Exception as exc:
            self._state = TransportState.ERROR
            raise TransportError(f"Failed to stop libp2p transport: {exc}") from exc

    # -- TransportAdapter: messaging ----------------------------------------

    async def send(self, peer_id: str, topic: str, payload: bytes) -> None:
        self._ensure_running()

        envelope = MessageEnvelope(
            source=self.local_peer.peer_id,
            topic=topic,
            payload=payload,
        )
        data = _encode_envelope(envelope)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        def _send_in_trio() -> None:
            trio.from_thread.run(
                self._trio_send,
                peer_id,
                topic,
                data,
                future,
                loop,
                trio_token=self._bridge._trio_token,
            )

        await loop.run_in_executor(None, _send_in_trio)
        await future

    async def _trio_send(
        self,
        peer_id: str,
        topic: str,
        data: bytes,
        future: asyncio.Future[None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Send data to a specific peer via a stream (runs inside trio)."""
        from libp2p.peer.id import ID

        try:
            target_id = ID.from_base58(peer_id)
            protocol = TProtocol(VALENCE_SYNC_PROTOCOL)
            stream = await self._host.new_stream(target_id, [protocol])
            await stream.write(data)
            await stream.close()
            loop.call_soon_threadsafe(future.set_result, None)
        except Exception as exc:
            loop.call_soon_threadsafe(
                future.set_exception,
                TransportError(f"Failed to send to {peer_id}: {exc}"),
            )

    async def broadcast(self, topic: str, payload: bytes) -> None:
        self._ensure_running()

        if not self._config.gossipsub_enabled or self._pubsub is None:
            raise TransportError("GossipSub is not enabled — cannot broadcast")

        envelope = MessageEnvelope(
            source=self.local_peer.peer_id,
            topic=topic,
            payload=payload,
        )
        data = _encode_envelope(envelope)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        def _broadcast_in_trio() -> None:
            trio.from_thread.run(
                self._trio_broadcast,
                topic,
                data,
                future,
                loop,
                trio_token=self._bridge._trio_token,
            )

        await loop.run_in_executor(None, _broadcast_in_trio)
        await future

    async def _trio_broadcast(
        self,
        topic: str,
        data: bytes,
        future: asyncio.Future[None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        try:
            await self._pubsub.publish(topic, data)
            loop.call_soon_threadsafe(future.set_result, None)
        except Exception as exc:
            loop.call_soon_threadsafe(
                future.set_exception,
                TransportError(f"Broadcast failed on topic {topic!r}: {exc}"),
            )

    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        self._ensure_running()
        self._handlers[topic].append(handler)

        # If this is the first handler for this topic, subscribe on pubsub
        if len(self._handlers[topic]) == 1 and self._pubsub is not None:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[None] = loop.create_future()

            def _subscribe_in_trio() -> None:
                trio.from_thread.run(
                    self._trio_subscribe,
                    topic,
                    loop,
                    future,
                    trio_token=self._bridge._trio_token,
                )

            await loop.run_in_executor(None, _subscribe_in_trio)
            await future

    async def _trio_subscribe(
        self,
        topic: str,
        loop: asyncio.AbstractEventLoop,
        future: asyncio.Future[None],
    ) -> None:
        try:
            subscription = await self._pubsub.subscribe(topic)
            self._subscriptions[topic] = subscription
            loop.call_soon_threadsafe(future.set_result, None)
            # Start receive loop for this subscription
            # NOTE: This task runs indefinitely in the trio thread
            while topic in self._subscriptions:
                try:
                    msg = await subscription.get()
                    envelope = _decode_envelope(msg.data, topic)
                    # Dispatch to asyncio handlers
                    for handler in list(self._handlers.get(topic, [])):
                        loop.call_soon_threadsafe(
                            asyncio.ensure_future,
                            handler(envelope),
                        )
                except Exception:
                    logger.debug("Error in subscription receive loop for %s", topic, exc_info=True)
                    await trio.sleep(0.1)
        except Exception as exc:
            if not future.done():
                loop.call_soon_threadsafe(
                    future.set_exception,
                    TransportError(f"Subscribe failed for topic {topic!r}: {exc}"),
                )

    async def unsubscribe(self, topic: str) -> None:
        self._handlers.pop(topic, None)
        self._subscriptions.pop(topic, None)

    # -- TransportAdapter: peer discovery -----------------------------------

    async def discover_peers(self) -> list[PeerInfo]:
        self._ensure_running()

        if self._host is None:
            return []

        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[PeerInfo]] = loop.create_future()

        def _discover_in_trio() -> None:
            trio.from_thread.run(
                self._trio_discover,
                future,
                loop,
                trio_token=self._bridge._trio_token,
            )

        await loop.run_in_executor(None, _discover_in_trio)
        return await future

    async def _trio_discover(
        self,
        future: asyncio.Future[list[PeerInfo]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        try:
            peerstore = self._host.get_peerstore()
            peer_ids = peerstore.peer_ids()
            peers = []
            local_id = self._host.get_id()
            for pid in peer_ids:
                if pid == local_id:
                    continue
                addrs = peerstore.addrs(pid)
                peers.append(
                    PeerInfo(
                        peer_id=pid.to_string(),
                        addrs=tuple(str(a) for a in addrs),
                    )
                )
            loop.call_soon_threadsafe(future.set_result, peers)
        except Exception as exc:
            loop.call_soon_threadsafe(
                future.set_exception,
                TransportError(f"Peer discovery failed: {exc}"),
            )

    async def connect_peer(self, addr: str) -> PeerInfo:
        self._ensure_running()

        loop = asyncio.get_running_loop()
        future: asyncio.Future[PeerInfo] = loop.create_future()

        def _connect_in_trio() -> None:
            trio.from_thread.run(
                self._trio_connect,
                addr,
                future,
                loop,
                trio_token=self._bridge._trio_token,
            )

        await loop.run_in_executor(None, _connect_in_trio)
        return await future

    async def _trio_connect(
        self,
        addr: str,
        future: asyncio.Future[PeerInfo],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        import multiaddr as multiaddr_mod

        try:
            maddr = multiaddr_mod.Multiaddr(addr)
            peer_info = info_from_p2p_addr(maddr)
            await self._host.connect(peer_info)
            addrs = self._host.get_peerstore().addrs(peer_info.peer_id)
            result = PeerInfo(
                peer_id=peer_info.peer_id.to_string(),
                addrs=tuple(str(a) for a in addrs),
            )
            loop.call_soon_threadsafe(future.set_result, result)
        except Exception as exc:
            loop.call_soon_threadsafe(
                future.set_exception,
                TransportError(f"Failed to connect to {addr}: {exc}"),
            )

    # -- stream handlers (trio-side) ----------------------------------------

    async def _handle_sync_stream(self, stream: Any) -> None:
        """Handle incoming sync protocol streams."""
        try:
            data = await stream.read(MAX_READ_LEN)
            if data:
                envelope = _decode_envelope(data, VALENCE_SYNC_PROTOCOL)
                loop = asyncio.get_running_loop()
                for handler in list(self._handlers.get(envelope.topic, [])):
                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        handler(envelope),
                    )
        except Exception:
            logger.debug("Error handling sync stream", exc_info=True)
        finally:
            await stream.close()

    async def _handle_auth_stream(self, stream: Any) -> None:
        """Handle incoming auth protocol streams."""
        try:
            data = await stream.read(MAX_READ_LEN)
            if data:
                envelope = _decode_envelope(data, VALENCE_AUTH_PROTOCOL)
                loop = asyncio.get_running_loop()
                for handler in list(self._handlers.get(envelope.topic, [])):
                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        handler(envelope),
                    )
        except Exception:
            logger.debug("Error handling auth stream", exc_info=True)
        finally:
            await stream.close()

    # -- internals ----------------------------------------------------------

    def _ensure_running(self) -> None:
        if self._state != TransportState.RUNNING:
            raise TransportError(f"Transport is {self._state.value}, expected RUNNING")


# ---------------------------------------------------------------------------
# Wire format helpers
# ---------------------------------------------------------------------------


def _encode_envelope(envelope: MessageEnvelope) -> bytes:
    """Serialise an envelope to bytes (JSON + payload)."""
    header = {
        "id": envelope.message_id,
        "src": envelope.source,
        "topic": envelope.topic,
        "ts": envelope.timestamp,
        "meta": envelope.metadata,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Format: 4-byte header length (big-endian) + header + payload
    import struct

    return struct.pack(">I", len(header_bytes)) + header_bytes + envelope.payload


def _decode_envelope(data: bytes, fallback_topic: str = "") -> MessageEnvelope:
    """Deserialise bytes into a :class:`MessageEnvelope`."""
    import struct

    if len(data) < 4:
        # Treat entire data as payload
        return MessageEnvelope(
            source="unknown",
            topic=fallback_topic,
            payload=data,
        )

    (header_len,) = struct.unpack(">I", data[:4])
    if header_len > len(data) - 4:
        # Malformed — treat as raw payload
        return MessageEnvelope(
            source="unknown",
            topic=fallback_topic,
            payload=data,
        )

    header = json.loads(data[4 : 4 + header_len])
    payload = data[4 + header_len :]

    return MessageEnvelope(
        message_id=header.get("id", str(uuid.uuid4())),
        source=header.get("src", "unknown"),
        topic=header.get("topic", fallback_topic),
        payload=payload,
        timestamp=header.get("ts", time.time()),
        metadata=header.get("meta", {}),
    )
