"""
Protocol handlers that translate between VFP message types and libp2p streams.

Each handler owns one libp2p protocol ID and implements:

* ``protocol_id`` — the protocol string to register with the libp2p host.
* ``handle_stream(stream)`` — called when a remote peer opens a stream for
  this protocol (inbound).
* ``open_stream(host, peer_id, request)`` — opens a new stream to *peer_id*
  and sends *request*, returning the response (outbound).

GossipSub handlers implement ``handle_message`` / ``publish`` instead.

All handlers reuse the existing VFP request/response dataclasses from
``valence.federation.protocol`` and the codecs from
``valence.transport.message_codec``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from oro_federation.protocol import (
    AuthChallengeRequest,
    AuthVerifyRequest,
    ErrorMessage,
    MessageType,
    ProtocolMessage,
    RequestBeliefsRequest,
    ShareBeliefRequest,
    SyncRequest,
    TrustAttestationRequest,
    parse_message,
)

from valence.transport.message_codec import (
    CodecError,
    GossipSubCodec,
    StreamBuffer,
    StreamCodec,
)
from valence.transport.protocols import (
    VALENCE_AUTH_PROTOCOL,
    VALENCE_BELIEFS_TOPIC,
    VALENCE_PEERS_TOPIC,
    VALENCE_SYNC_PROTOCOL,
    VALENCE_TRUST_PROTOCOL,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Abstract stream / libp2p interfaces (thin typing shims)
# ============================================================================
# These protocols define the minimal surface we need from a libp2p
# implementation (e.g. py-libp2p, a Rust FFI bridge, or a mock in tests).


@runtime_checkable
class IStream(Protocol):
    """Minimal async stream interface expected by handlers."""

    async def read(self, n: int = -1) -> bytes: ...
    async def write(self, data: bytes) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class IHost(Protocol):
    """Minimal libp2p host interface expected by handlers."""

    async def new_stream(self, peer_id: str, protocols: list[str]) -> IStream: ...


@runtime_checkable
class IPubSub(Protocol):
    """Minimal GossipSub interface expected by handlers."""

    async def publish(self, topic: str, data: bytes) -> None: ...
    async def subscribe(self, topic: str) -> Any: ...


# ============================================================================
# Base handler
# ============================================================================


class BaseStreamHandler(ABC):
    """Base class for libp2p stream protocol handlers.

    Subclasses must implement :meth:`_dispatch` which receives a parsed
    request dict and returns a response dict (or ``None`` to send nothing).
    """

    @property
    @abstractmethod
    def protocol_id(self) -> str:
        """The libp2p protocol string this handler serves."""

    async def handle_stream(self, stream: IStream) -> None:
        """Read one request from *stream*, dispatch it, and write the response.

        This follows the simple request/response pattern used by VFP stream
        protocols.  Multi-message exchanges (e.g. auth challenge + verify)
        are modelled as separate stream opens by the initiator.
        """
        buf = StreamBuffer()
        try:
            # Read until we have one complete frame
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                buf.feed(chunk)
                frames = buf.drain()
                if frames:
                    break

            if not frames:
                logger.warning("%s: stream closed before a complete frame", self.protocol_id)
                return

            request_dict = frames[0]
            response_dict = await self._dispatch(request_dict)

            if response_dict is not None:
                await stream.write(StreamCodec.encode(response_dict))
        except CodecError as exc:
            logger.warning("%s: codec error: %s", self.protocol_id, exc)
            err = ErrorMessage(message=str(exc)).to_dict()
            try:
                await stream.write(StreamCodec.encode(err))
            except Exception:
                pass
        except Exception:
            logger.exception("%s: unhandled error in stream handler", self.protocol_id)
        finally:
            try:
                await stream.close()
            except Exception:
                pass

    async def send_request(
        self,
        host: IHost,
        peer_id: str,
        request: ProtocolMessage,
    ) -> dict[str, Any] | None:
        """Open a stream to *peer_id*, send *request*, and return the response dict.

        Returns ``None`` if the peer closes the stream without a response.
        """
        stream = await host.new_stream(peer_id, [self.protocol_id])
        try:
            await stream.write(StreamCodec.encode_message(request))

            buf = StreamBuffer()
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                buf.feed(chunk)
                frames = buf.drain()
                if frames:
                    return frames[0]
            return None
        finally:
            try:
                await stream.close()
            except Exception:
                pass

    @abstractmethod
    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle a parsed request dict and return the response dict."""


# ============================================================================
# Sync protocol handler
# ============================================================================


class SyncProtocolHandler(BaseStreamHandler):
    """Handles belief sync, share, and request over ``/valence/sync/1.0.0``.

    Dispatches to the existing VFP handlers in ``valence.federation.protocol``:

    * ``SYNC_REQUEST`` → ``handle_sync_request``
    * ``SHARE_BELIEF`` → ``handle_share_belief``
    * ``REQUEST_BELIEFS`` → ``handle_request_beliefs``
    """

    def __init__(
        self,
        *,
        get_sender_node_id: Any | None = None,
        get_sender_trust: Any | None = None,
    ) -> None:
        self._get_sender_node_id = get_sender_node_id
        self._get_sender_trust = get_sender_trust

    @property
    def protocol_id(self) -> str:
        return VALENCE_SYNC_PROTOCOL

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        msg = parse_message(request)
        if msg is None:
            return ErrorMessage(message="Unparseable message").to_dict()

        sender_node_id = self._resolve_sender_node_id(request)
        sender_trust = self._resolve_sender_trust(sender_node_id)

        if sender_node_id is None:
            return ErrorMessage(message="Sender identification required").to_dict()

        if isinstance(msg, SyncRequest):
            from oro_federation.protocol import handle_sync_request

            result = handle_sync_request(msg, sender_node_id, sender_trust)
            return result.to_dict()

        if isinstance(msg, ShareBeliefRequest):
            from oro_federation.protocol import handle_share_belief

            share_result = handle_share_belief(msg, sender_node_id, sender_trust)
            return share_result.to_dict()

        if isinstance(msg, RequestBeliefsRequest):
            from oro_federation.protocol import handle_request_beliefs

            beliefs_result = handle_request_beliefs(msg, sender_node_id, sender_trust)
            return beliefs_result.to_dict()

        return ErrorMessage(message=f"Unexpected message type for sync protocol: {msg.type}").to_dict()

    # -- sender resolution helpers (pluggable via constructor) ---------------

    def _resolve_sender_node_id(self, request: dict[str, Any]) -> UUID | None:
        if self._get_sender_node_id is not None:
            return self._get_sender_node_id(request)
        raw = request.get("sender_node_id")
        if raw is not None:
            return UUID(raw) if isinstance(raw, str) else raw
        return None

    def _resolve_sender_trust(self, node_id: UUID | None) -> float:
        if self._get_sender_trust is not None and node_id is not None:
            return self._get_sender_trust(node_id)
        return 0.1  # Observer-level default


# ============================================================================
# Auth protocol handler
# ============================================================================


class AuthProtocolHandler(BaseStreamHandler):
    """Handles DID auth challenge/verify over ``/valence/auth/1.0.0``.

    Dispatches:

    * ``AUTH_CHALLENGE`` → ``create_auth_challenge``
    * ``AUTH_VERIFY`` → ``verify_auth_challenge``
    """

    @property
    def protocol_id(self) -> str:
        return VALENCE_AUTH_PROTOCOL

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        msg = parse_message(request)
        if msg is None:
            return ErrorMessage(message="Unparseable auth message").to_dict()

        if isinstance(msg, AuthChallengeRequest):
            from oro_federation.protocol import create_auth_challenge

            response = create_auth_challenge(msg.client_did)
            return response.to_dict()

        if isinstance(msg, AuthVerifyRequest):
            # Look up the client's public key to verify the signature
            public_key = self._lookup_public_key(msg.client_did)
            if public_key is None:
                return ErrorMessage(message=f"Unknown node: {msg.client_did}").to_dict()

            from oro_federation.protocol import verify_auth_challenge

            verify_response = verify_auth_challenge(
                client_did=msg.client_did,
                challenge=msg.challenge,
                signature=msg.signature,
                public_key_multibase=public_key,
            )
            return verify_response.to_dict()

        return ErrorMessage(message=f"Unexpected message type for auth protocol: {msg.type}").to_dict()

    @staticmethod
    def _lookup_public_key(client_did: str) -> str | None:
        """Look up a node's public key by DID.

        Attempts a database lookup first; returns ``None`` if the node
        is not known.
        """
        try:
            from oro_db import get_cursor

            with get_cursor() as cur:
                cur.execute(
                    "SELECT public_key_multibase FROM federation_nodes WHERE did = %s",
                    (client_did,),
                )
                row = cur.fetchone()
                return row["public_key_multibase"] if row else None
        except Exception:
            logger.debug("Could not look up public key for %s", client_did)
            return None


# ============================================================================
# Trust protocol handler
# ============================================================================


class TrustProtocolHandler(BaseStreamHandler):
    """Handles trust attestation exchange over ``/valence/trust/1.0.0``.

    Dispatches:

    * ``TRUST_ATTESTATION`` → internal trust processing
    """

    def __init__(
        self,
        *,
        get_sender_node_id: Any | None = None,
        get_sender_trust: Any | None = None,
    ) -> None:
        self._get_sender_node_id = get_sender_node_id
        self._get_sender_trust = get_sender_trust

    @property
    def protocol_id(self) -> str:
        return VALENCE_TRUST_PROTOCOL

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        msg = parse_message(request)
        if msg is None:
            return ErrorMessage(message="Unparseable trust message").to_dict()

        if isinstance(msg, TrustAttestationRequest):
            sender_node_id = self._resolve_sender_node_id(request)
            sender_trust = self._resolve_sender_trust(sender_node_id)

            if sender_node_id is None:
                return ErrorMessage(message="Sender identification required").to_dict()

            # Delegate to the existing VFP handler (sync, not async)
            from oro_federation.protocol import _handle_trust_attestation

            result = _handle_trust_attestation(msg, sender_node_id, sender_trust)
            return result.to_dict()

        return ErrorMessage(message=f"Unexpected message type for trust protocol: {msg.type}").to_dict()

    def _resolve_sender_node_id(self, request: dict[str, Any]) -> UUID | None:
        if self._get_sender_node_id is not None:
            return self._get_sender_node_id(request)
        raw = request.get("sender_node_id")
        if raw is not None:
            return UUID(raw) if isinstance(raw, str) else raw
        return None

    def _resolve_sender_trust(self, node_id: UUID | None) -> float:
        if self._get_sender_trust is not None and node_id is not None:
            return self._get_sender_trust(node_id)
        return 0.1


# ============================================================================
# GossipSub handler: belief propagation
# ============================================================================


@dataclass
class BeliefPropagationHandler:
    """GossipSub handler for the ``/valence/beliefs`` topic.

    Encodes outbound beliefs as JSON for GossipSub publish and decodes
    inbound messages, dispatching them through the VFP ``SHARE_BELIEF``
    handler path.
    """

    topic: str = VALENCE_BELIEFS_TOPIC
    _pubsub: IPubSub | None = None

    # Optional hooks for sender resolution (same pattern as stream handlers)
    get_sender_node_id: Any | None = None
    get_sender_trust: Any | None = None

    async def handle_message(self, data: bytes) -> dict[str, Any] | None:
        """Process an inbound GossipSub message.

        Args:
            data: Raw bytes from the GossipSub subscription.

        Returns:
            The response dict from belief processing, or ``None`` on error.
        """
        try:
            msg_dict = GossipSubCodec.decode(data)
        except CodecError as exc:
            logger.warning("beliefs topic: codec error: %s", exc)
            return None

        msg = parse_message(msg_dict)
        if msg is None:
            # Might be a raw belief dict rather than a wrapped VFP message.
            # Wrap it as a SHARE_BELIEF for processing.
            if "content" in msg_dict and "origin_node_did" in msg_dict:
                msg_dict = {
                    "type": MessageType.SHARE_BELIEF.value,
                    "beliefs": [msg_dict],
                }
                msg = parse_message(msg_dict)

        if not isinstance(msg, ShareBeliefRequest):
            logger.debug("beliefs topic: ignoring non-SHARE_BELIEF message")
            return None

        sender_node_id = self._resolve_sender_node_id(msg_dict)
        sender_trust = self._resolve_sender_trust(sender_node_id)

        if sender_node_id is None:
            logger.warning("beliefs topic: could not identify sender")
            return None

        from oro_federation.protocol import handle_share_belief

        result = handle_share_belief(msg, sender_node_id, sender_trust)
        return result.to_dict()

    async def publish_belief(
        self,
        pubsub: IPubSub | None = None,
        belief_dict: dict[str, Any] | None = None,
        message: ProtocolMessage | None = None,
    ) -> None:
        """Publish a belief (or VFP message) to the beliefs topic.

        Provide *either* ``belief_dict`` (raw belief data that will be
        wrapped in a ``SHARE_BELIEF`` envelope) or ``message`` (a fully
        formed ``ProtocolMessage``).
        """
        ps = pubsub or self._pubsub
        if ps is None:
            raise RuntimeError("No PubSub instance configured")

        if message is not None:
            payload = GossipSubCodec.encode_message(message)
        elif belief_dict is not None:
            envelope = {
                "type": MessageType.SHARE_BELIEF.value,
                "beliefs": [belief_dict],
            }
            payload = GossipSubCodec.encode(envelope)
        else:
            raise ValueError("Provide belief_dict or message")

        await ps.publish(self.topic, payload)

    def _resolve_sender_node_id(self, msg_dict: dict[str, Any]) -> UUID | None:
        if self.get_sender_node_id is not None:
            return self.get_sender_node_id(msg_dict)
        raw = msg_dict.get("sender_node_id")
        if raw is not None:
            return UUID(raw) if isinstance(raw, str) else raw
        return None

    def _resolve_sender_trust(self, node_id: UUID | None) -> float:
        if self.get_sender_trust is not None and node_id is not None:
            return self.get_sender_trust(node_id)
        return 0.1


# ============================================================================
# GossipSub handler: peer discovery
# ============================================================================


@dataclass
class PeerDiscoveryHandler:
    """GossipSub handler for the ``/valence/peers`` topic.

    Handles peer announcement messages (node joining, capabilities update).
    """

    topic: str = VALENCE_PEERS_TOPIC
    _pubsub: IPubSub | None = None

    async def handle_message(self, data: bytes) -> dict[str, Any] | None:
        """Process an inbound peer announcement.

        Returns the parsed announcement dict for the caller to act on.
        """
        try:
            return GossipSubCodec.decode(data)
        except CodecError as exc:
            logger.warning("peers topic: codec error: %s", exc)
            return None

    async def publish_announcement(
        self,
        pubsub: IPubSub | None = None,
        announcement: dict[str, Any] | None = None,
    ) -> None:
        """Publish a peer announcement to the peers topic."""
        ps = pubsub or self._pubsub
        if ps is None:
            raise RuntimeError("No PubSub instance configured")
        if announcement is None:
            raise ValueError("Provide announcement dict")
        await ps.publish(self.topic, GossipSubCodec.encode(announcement))


# ============================================================================
# Registry convenience
# ============================================================================


def create_handlers(
    *,
    get_sender_node_id: Any | None = None,
    get_sender_trust: Any | None = None,
) -> dict[str, BaseStreamHandler | BeliefPropagationHandler | PeerDiscoveryHandler]:
    """Create all protocol handlers with shared sender resolution hooks.

    Returns a dict keyed by protocol ID / topic string.
    """
    sync = SyncProtocolHandler(
        get_sender_node_id=get_sender_node_id,
        get_sender_trust=get_sender_trust,
    )
    auth = AuthProtocolHandler()
    trust = TrustProtocolHandler(
        get_sender_node_id=get_sender_node_id,
        get_sender_trust=get_sender_trust,
    )
    beliefs = BeliefPropagationHandler(
        get_sender_node_id=get_sender_node_id,
        get_sender_trust=get_sender_trust,
    )
    peers = PeerDiscoveryHandler()

    return {
        sync.protocol_id: sync,
        auth.protocol_id: auth,
        trust.protocol_id: trust,
        beliefs.topic: beliefs,
        peers.topic: peers,
    }
