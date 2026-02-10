"""Tests for valence.transport.protocol_handler — handler dispatch."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from oro_federation.protocol import (
    AuthChallengeRequest,
    ShareBeliefRequest,
    SyncRequest,
    TrustAttestationRequest,
)

from valence.transport.message_codec import GossipSubCodec, StreamCodec
from valence.transport.protocol_handler import (
    AuthProtocolHandler,
    BeliefPropagationHandler,
    PeerDiscoveryHandler,
    SyncProtocolHandler,
    TrustProtocolHandler,
    create_handlers,
)
from valence.transport.protocols import (
    VALENCE_AUTH_PROTOCOL,
    VALENCE_BELIEFS_TOPIC,
    VALENCE_PEERS_TOPIC,
    VALENCE_SYNC_PROTOCOL,
    VALENCE_TRUST_PROTOCOL,
)

# ============================================================================
# Helpers — fake stream / host / pubsub
# ============================================================================


class FakeStream:
    """In-memory async stream for testing."""

    def __init__(self, inbound: bytes = b"") -> None:
        self._inbound = bytearray(inbound)
        self._outbound = bytearray()
        self._closed = False

    async def read(self, n: int = -1) -> bytes:
        if not self._inbound:
            return b""
        if n < 0:
            data = bytes(self._inbound)
            self._inbound.clear()
            return data
        data = bytes(self._inbound[:n])
        del self._inbound[:n]
        return data

    async def write(self, data: bytes) -> None:
        self._outbound.extend(data)

    async def close(self) -> None:
        self._closed = True

    @property
    def outbound_bytes(self) -> bytes:
        return bytes(self._outbound)

    @property
    def closed(self) -> bool:
        return self._closed


class FakeHost:
    """In-memory host that returns a FakeStream with predetermined response."""

    def __init__(self, response_data: bytes = b"") -> None:
        self._response_data = response_data
        self.opened_streams: list[tuple[str, list[str]]] = []

    async def new_stream(self, peer_id: str, protocols: list[str]) -> FakeStream:
        self.opened_streams.append((peer_id, protocols))
        return FakeStream(inbound=self._response_data)


class FakePubSub:
    """In-memory PubSub for testing."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(self, topic: str, data: bytes) -> None:
        self.published.append((topic, data))

    async def subscribe(self, topic: str) -> None:
        return None


# ============================================================================
# Protocol IDs
# ============================================================================


class TestProtocolIds:
    """Each handler exposes the correct protocol_id."""

    def test_sync_handler_protocol_id(self) -> None:
        handler = SyncProtocolHandler()
        assert handler.protocol_id == VALENCE_SYNC_PROTOCOL

    def test_auth_handler_protocol_id(self) -> None:
        handler = AuthProtocolHandler()
        assert handler.protocol_id == VALENCE_AUTH_PROTOCOL

    def test_trust_handler_protocol_id(self) -> None:
        handler = TrustProtocolHandler()
        assert handler.protocol_id == VALENCE_TRUST_PROTOCOL

    def test_belief_propagation_topic(self) -> None:
        handler = BeliefPropagationHandler()
        assert handler.topic == VALENCE_BELIEFS_TOPIC

    def test_peer_discovery_topic(self) -> None:
        handler = PeerDiscoveryHandler()
        assert handler.topic == VALENCE_PEERS_TOPIC


# ============================================================================
# SyncProtocolHandler dispatch
# ============================================================================


class TestSyncProtocolHandler:
    """SyncProtocolHandler._dispatch with mocked sender resolution."""

    @pytest.fixture()
    def handler(self) -> SyncProtocolHandler:
        node_id = uuid4()
        return SyncProtocolHandler(
            get_sender_node_id=lambda req: node_id,
            get_sender_trust=lambda nid: 0.5,
        )

    @pytest.mark.asyncio
    async def test_dispatch_unparseable_returns_error(self, handler: SyncProtocolHandler) -> None:
        result = await handler._dispatch({"type": "GARBAGE"})
        assert result is not None
        assert result["type"] == "ERROR"

    @pytest.mark.asyncio
    async def test_dispatch_without_sender_returns_error(self) -> None:
        handler = SyncProtocolHandler()  # No sender hooks
        request = SyncRequest(page_size=10).to_dict()
        result = await handler._dispatch(request)
        assert result is not None
        assert result["type"] == "ERROR"
        assert "Sender" in result.get("message", "")

    @pytest.mark.asyncio
    async def test_dispatch_wrong_message_type_returns_error(self, handler: SyncProtocolHandler) -> None:
        # AUTH_CHALLENGE is not a sync message
        request = AuthChallengeRequest(client_did="did:vkb:web:test").to_dict()
        result = await handler._dispatch(request)
        assert result is not None
        assert result["type"] == "ERROR"
        assert "Unexpected" in result.get("message", "")


# ============================================================================
# AuthProtocolHandler dispatch
# ============================================================================


class TestAuthProtocolHandler:
    """AuthProtocolHandler._dispatch."""

    @pytest.fixture()
    def handler(self) -> AuthProtocolHandler:
        return AuthProtocolHandler()

    @pytest.mark.asyncio
    async def test_dispatch_unparseable_returns_error(self, handler: AuthProtocolHandler) -> None:
        result = await handler._dispatch({"type": "GARBAGE"})
        assert result is not None
        assert result["type"] == "ERROR"

    @pytest.mark.asyncio
    async def test_dispatch_wrong_type_returns_error(self, handler: AuthProtocolHandler) -> None:
        request = SyncRequest(page_size=10).to_dict()
        result = await handler._dispatch(request)
        assert result is not None
        assert result["type"] == "ERROR"


# ============================================================================
# TrustProtocolHandler dispatch
# ============================================================================


class TestTrustProtocolHandler:
    """TrustProtocolHandler._dispatch."""

    @pytest.fixture()
    def handler(self) -> TrustProtocolHandler:
        node_id = uuid4()
        return TrustProtocolHandler(
            get_sender_node_id=lambda req: node_id,
            get_sender_trust=lambda nid: 0.5,
        )

    @pytest.mark.asyncio
    async def test_dispatch_unparseable_returns_error(self, handler: TrustProtocolHandler) -> None:
        result = await handler._dispatch({"type": "GARBAGE"})
        assert result is not None
        assert result["type"] == "ERROR"

    @pytest.mark.asyncio
    async def test_dispatch_without_sender_returns_error(self) -> None:
        handler = TrustProtocolHandler()
        request = TrustAttestationRequest(
            attestation={"subject_did": "did:vkb:web:bob"},
            issuer_signature="abc",
        ).to_dict()
        result = await handler._dispatch(request)
        assert result is not None
        assert result["type"] == "ERROR"
        assert "Sender" in result.get("message", "")


# ============================================================================
# BeliefPropagationHandler
# ============================================================================


class TestBeliefPropagationHandler:
    """BeliefPropagationHandler GossipSub handling."""

    @pytest.fixture()
    def handler(self) -> BeliefPropagationHandler:
        node_id = uuid4()
        return BeliefPropagationHandler(
            get_sender_node_id=lambda msg: node_id,
            get_sender_trust=lambda nid: 0.4,
        )

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, handler: BeliefPropagationHandler) -> None:
        result = await handler.handle_message(b"not json")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_non_share_belief_ignored(self, handler: BeliefPropagationHandler) -> None:
        data = GossipSubCodec.encode({"type": "SYNC_REQUEST"})
        result = await handler.handle_message(data)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_without_sender_returns_none(self) -> None:
        handler = BeliefPropagationHandler()  # No sender hooks
        data = GossipSubCodec.encode(
            {
                "type": "SHARE_BELIEF",
                "beliefs": [{"content": "test", "origin_node_did": "did:vkb:web:x"}],
            }
        )
        result = await handler.handle_message(data)
        assert result is None

    @pytest.mark.asyncio
    async def test_publish_belief_dict(self) -> None:
        pubsub = FakePubSub()
        handler = BeliefPropagationHandler()
        await handler.publish_belief(
            pubsub=pubsub,
            belief_dict={"content": "Earth is round", "origin_node_did": "did:vkb:web:me"},
        )
        assert len(pubsub.published) == 1
        topic, data = pubsub.published[0]
        assert topic == VALENCE_BELIEFS_TOPIC
        decoded = json.loads(data)
        assert decoded["type"] == "SHARE_BELIEF"
        assert len(decoded["beliefs"]) == 1

    @pytest.mark.asyncio
    async def test_publish_message(self) -> None:
        pubsub = FakePubSub()
        handler = BeliefPropagationHandler()
        msg = ShareBeliefRequest(beliefs=[{"content": "test"}])
        await handler.publish_belief(pubsub=pubsub, message=msg)
        assert len(pubsub.published) == 1

    @pytest.mark.asyncio
    async def test_publish_without_pubsub_raises(self) -> None:
        handler = BeliefPropagationHandler()
        with pytest.raises(RuntimeError, match="No PubSub"):
            await handler.publish_belief(belief_dict={"content": "x"})

    @pytest.mark.asyncio
    async def test_publish_without_data_raises(self) -> None:
        pubsub = FakePubSub()
        handler = BeliefPropagationHandler()
        with pytest.raises(ValueError, match="Provide"):
            await handler.publish_belief(pubsub=pubsub)


# ============================================================================
# PeerDiscoveryHandler
# ============================================================================


class TestPeerDiscoveryHandler:
    """PeerDiscoveryHandler GossipSub handling."""

    @pytest.mark.asyncio
    async def test_handle_valid_announcement(self) -> None:
        handler = PeerDiscoveryHandler()
        announcement = {"did": "did:vkb:web:new-node", "capabilities": ["sync"]}
        data = GossipSubCodec.encode(announcement)
        result = await handler.handle_message(data)
        assert result == announcement

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self) -> None:
        handler = PeerDiscoveryHandler()
        result = await handler.handle_message(b"bad")
        assert result is None

    @pytest.mark.asyncio
    async def test_publish_announcement(self) -> None:
        pubsub = FakePubSub()
        handler = PeerDiscoveryHandler()
        ann = {"did": "did:vkb:web:me", "status": "active"}
        await handler.publish_announcement(pubsub=pubsub, announcement=ann)
        assert len(pubsub.published) == 1
        assert pubsub.published[0][0] == VALENCE_PEERS_TOPIC

    @pytest.mark.asyncio
    async def test_publish_without_pubsub_raises(self) -> None:
        handler = PeerDiscoveryHandler()
        with pytest.raises(RuntimeError, match="No PubSub"):
            await handler.publish_announcement(announcement={"did": "x"})

    @pytest.mark.asyncio
    async def test_publish_without_data_raises(self) -> None:
        pubsub = FakePubSub()
        handler = PeerDiscoveryHandler()
        with pytest.raises(ValueError, match="Provide"):
            await handler.publish_announcement(pubsub=pubsub)


# ============================================================================
# BaseStreamHandler — handle_stream and send_request
# ============================================================================


class TestBaseStreamHandler:
    """Test the stream I/O wiring via handle_stream / send_request."""

    @pytest.mark.asyncio
    async def test_handle_stream_codec_error(self) -> None:
        """Handler sends an error frame on bad input and closes the stream."""
        handler = SyncProtocolHandler(
            get_sender_node_id=lambda r: uuid4(),
            get_sender_trust=lambda n: 0.5,
        )
        # Feed garbage that has a valid length header but bad JSON
        import struct

        bad_payload = b"{bad json"
        bad_frame = struct.pack("!I", len(bad_payload)) + bad_payload
        stream = FakeStream(inbound=bad_frame)
        await handler.handle_stream(stream)
        assert stream.closed

    @pytest.mark.asyncio
    async def test_handle_stream_empty_close(self) -> None:
        """Handler handles a stream that closes without sending data."""
        handler = SyncProtocolHandler()
        stream = FakeStream(inbound=b"")
        await handler.handle_stream(stream)
        assert stream.closed

    @pytest.mark.asyncio
    async def test_send_request_returns_none_on_empty_response(self) -> None:
        """send_request returns None when the peer closes without responding."""
        handler = SyncProtocolHandler()
        host = FakeHost(response_data=b"")
        msg = SyncRequest(page_size=10)
        result = await handler.send_request(host, "peer123", msg)
        assert result is None
        assert len(host.opened_streams) == 1
        assert host.opened_streams[0] == ("peer123", [VALENCE_SYNC_PROTOCOL])

    @pytest.mark.asyncio
    async def test_send_request_returns_response(self) -> None:
        """send_request decodes a valid response from the peer."""
        handler = SyncProtocolHandler()
        response_dict = {"type": "SYNC_RESPONSE", "changes": [], "has_more": False}
        response_frame = StreamCodec.encode(response_dict)
        host = FakeHost(response_data=response_frame)
        msg = SyncRequest(page_size=10)
        result = await handler.send_request(host, "peer456", msg)
        assert result is not None
        assert result["type"] == "SYNC_RESPONSE"


# ============================================================================
# create_handlers registry
# ============================================================================


class TestCreateHandlers:
    """create_handlers factory function."""

    def test_returns_all_handlers(self) -> None:
        handlers = create_handlers()
        assert VALENCE_SYNC_PROTOCOL in handlers
        assert VALENCE_AUTH_PROTOCOL in handlers
        assert VALENCE_TRUST_PROTOCOL in handlers
        assert VALENCE_BELIEFS_TOPIC in handlers
        assert VALENCE_PEERS_TOPIC in handlers

    def test_handlers_have_correct_types(self) -> None:
        handlers = create_handlers()
        assert isinstance(handlers[VALENCE_SYNC_PROTOCOL], SyncProtocolHandler)
        assert isinstance(handlers[VALENCE_AUTH_PROTOCOL], AuthProtocolHandler)
        assert isinstance(handlers[VALENCE_TRUST_PROTOCOL], TrustProtocolHandler)
        assert isinstance(handlers[VALENCE_BELIEFS_TOPIC], BeliefPropagationHandler)
        assert isinstance(handlers[VALENCE_PEERS_TOPIC], PeerDiscoveryHandler)

    def test_shared_hooks_propagated(self) -> None:
        node_id = uuid4()
        hook = lambda req: node_id  # noqa: E731
        handlers = create_handlers(get_sender_node_id=hook)
        sync = handlers[VALENCE_SYNC_PROTOCOL]
        assert isinstance(sync, SyncProtocolHandler)
        assert sync._get_sender_node_id is hook
