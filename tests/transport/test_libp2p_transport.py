"""Tests for the libp2p transport backend.

These tests mock py-libp2p internals so they can run without a real
network.  Integration tests with actual peers are in tests/integration/.

Issue #300 — P2P: Integrate py-libp2p as transport backend.
"""

from __future__ import annotations

import struct
from unittest.mock import AsyncMock, patch

import pytest

from valence.transport.adapter import (
    MessageEnvelope,
    TransportError,
    TransportState,
)
from valence.transport.config import TransportConfig
from valence.transport.libp2p_transport import (
    VALENCE_BELIEF_TOPIC,
    VALENCE_SYNC_PROTOCOL,
    Libp2pTransport,
    _decode_envelope,
    _encode_envelope,
    _require_libp2p,
)

# ---------------------------------------------------------------------------
# Wire format tests (no network needed)
# ---------------------------------------------------------------------------


class TestEnvelopeEncoding:
    """Test the encode/decode round-trip for MessageEnvelope."""

    def test_round_trip(self):
        original = MessageEnvelope(
            source="peer-A",
            topic="valence/beliefs",
            payload=b"hello world",
            message_id="msg-001",
            timestamp=1700000000.0,
            metadata={"ttl": 60},
        )
        encoded = _encode_envelope(original)
        decoded = _decode_envelope(encoded)

        assert decoded.message_id == original.message_id
        assert decoded.source == original.source
        assert decoded.topic == original.topic
        assert decoded.payload == original.payload
        assert decoded.timestamp == original.timestamp
        assert decoded.metadata == original.metadata

    def test_empty_payload(self):
        original = MessageEnvelope(
            source="peer-B",
            topic="sync",
            payload=b"",
        )
        encoded = _encode_envelope(original)
        decoded = _decode_envelope(encoded)
        assert decoded.payload == b""
        assert decoded.source == "peer-B"

    def test_binary_payload(self):
        payload = bytes(range(256))
        original = MessageEnvelope(
            source="peer-C",
            topic="auth",
            payload=payload,
        )
        encoded = _encode_envelope(original)
        decoded = _decode_envelope(encoded)
        assert decoded.payload == payload

    def test_decode_short_data(self):
        """Data shorter than 4 bytes should be treated as raw payload."""
        decoded = _decode_envelope(b"\x01\x02", "fallback-topic")
        assert decoded.source == "unknown"
        assert decoded.topic == "fallback-topic"
        assert decoded.payload == b"\x01\x02"

    def test_decode_malformed_header_length(self):
        """Header length > remaining data should be treated as raw."""
        data = struct.pack(">I", 9999) + b"tiny"
        decoded = _decode_envelope(data, "fallback")
        assert decoded.source == "unknown"
        assert decoded.topic == "fallback"


# ---------------------------------------------------------------------------
# Transport instantiation (mocked)
# ---------------------------------------------------------------------------


class TestLibp2pTransportInit:
    def test_initial_state(self):
        transport = Libp2pTransport()
        assert transport.state == TransportState.STOPPED

    def test_custom_config(self):
        config = TransportConfig(
            listen_addrs=["/ip4/127.0.0.1/tcp/5555"],
            gossipsub_degree=3,
        )
        transport = Libp2pTransport(config)
        assert transport._config.gossipsub_degree == 3

    def test_local_peer_before_start_raises(self):
        transport = Libp2pTransport()
        with pytest.raises(TransportError, match="not started"):
            _ = transport.local_peer


class TestLibp2pTransportNotRunning:
    """Operations that require RUNNING state should fail cleanly."""

    @pytest.mark.asyncio
    async def test_send_before_start(self):
        transport = Libp2pTransport()
        with pytest.raises(TransportError, match="RUNNING"):
            await transport.send("peer", "topic", b"data")

    @pytest.mark.asyncio
    async def test_broadcast_before_start(self):
        transport = Libp2pTransport()
        with pytest.raises(TransportError, match="RUNNING"):
            await transport.broadcast("topic", b"data")

    @pytest.mark.asyncio
    async def test_subscribe_before_start(self):
        transport = Libp2pTransport()
        with pytest.raises(TransportError, match="RUNNING"):
            await transport.subscribe("topic", AsyncMock())

    @pytest.mark.asyncio
    async def test_discover_before_start(self):
        transport = Libp2pTransport()
        with pytest.raises(TransportError, match="RUNNING"):
            await transport.discover_peers()

    @pytest.mark.asyncio
    async def test_connect_peer_before_start(self):
        transport = Libp2pTransport()
        with pytest.raises(TransportError, match="RUNNING"):
            await transport.connect_peer("/ip4/1.2.3.4/tcp/4001/p2p/QmPeer")


class TestLibp2pTransportStopIdempotent:
    @pytest.mark.asyncio
    async def test_stop_when_already_stopped(self):
        transport = Libp2pTransport()
        # Should not raise
        await transport.stop()
        assert transport.state == TransportState.STOPPED


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------


class TestProtocolConstants:
    def test_sync_protocol(self):
        assert VALENCE_SYNC_PROTOCOL == "/valence/sync/1.0.0"

    def test_belief_topic(self):
        assert VALENCE_BELIEF_TOPIC == "valence/beliefs"


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestLibp2pTransportConfig:
    def test_gossipsub_disabled(self):
        config = TransportConfig(gossipsub_enabled=False)
        transport = Libp2pTransport(config)
        assert transport._config.gossipsub_enabled is False

    def test_from_env_integration(self, monkeypatch):
        monkeypatch.setenv("VALENCE_TRANSPORT_TYPE", "libp2p")
        monkeypatch.setenv("VALENCE_TRANSPORT_DHT_ENABLED", "true")
        monkeypatch.setenv("VALENCE_TRANSPORT_GOSSIPSUB_DEGREE", "8")

        config = TransportConfig.from_env()
        transport = Libp2pTransport(config)
        assert transport._config.gossipsub_degree == 8
        assert transport._config.dht_enabled is True


# ---------------------------------------------------------------------------
# require_libp2p guard
# ---------------------------------------------------------------------------


class TestRequireLibp2p:
    def test_available(self):
        """Should not raise when libp2p is installed."""
        _require_libp2p()

    def test_unavailable(self):
        with patch("valence.transport.libp2p_transport._LIBP2P_AVAILABLE", False):
            with pytest.raises(TransportError, match="not installed"):
                _require_libp2p()

    def test_transport_init_without_libp2p(self):
        with patch("valence.transport.libp2p_transport._LIBP2P_AVAILABLE", False):
            with pytest.raises(TransportError, match="not installed"):
                Libp2pTransport()
