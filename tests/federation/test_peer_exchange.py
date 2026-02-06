"""Tests for peer exchange protocol (Issue #263).

Tests cover:
- PeerInfo validation and serialization
- PeerExchangeMessage serialization/deserialization
- PeerExchangeResponse serialization/deserialization
- PeerExchangeProtocol:
  - Rate limiting
  - Shareable peers filtering (trust, opt-out, status)
  - Handling incoming peer exchange requests
  - Merging received peers (dedup, validation, anti-spam)
  - Requesting peers from remote nodes (HTTP)
- Privacy controls (opt-out, trust filtering)
- Anti-spam (address validation, self-rejection)
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from valence.federation.discovery import (
    PEER_EXCHANGE_RATE_LIMIT,
    PEER_EXCHANGE_RATE_WINDOW,
    PeerExchangeMessage,
    PeerExchangeProtocol,
    PeerExchangeResponse,
    PeerInfo,
    exchange_peers,
    get_peer_exchange_protocol,
    set_peer_exchange_protocol,
)
from valence.federation.models import (
    FederationNode,
    NodeStatus,
    TrustPhase,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_cursor():
    """Mock database cursor."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Mock the get_cursor context manager."""

    @contextmanager
    def _mock_get_cursor(dict_cursor: bool = True) -> Generator:
        yield mock_cursor

    with patch("valence.federation.discovery.get_cursor", _mock_get_cursor):
        yield mock_cursor


@pytest.fixture
def protocol():
    """Create a PeerExchangeProtocol for testing."""
    return PeerExchangeProtocol(
        local_did="did:vkb:web:local.example.com",
        max_peers=50,
        rate_limit=10,
        rate_window=3600,
        min_trust=0.2,
    )


@pytest.fixture
def sample_peer_info():
    """Factory for PeerInfo instances."""

    def _factory(**kwargs):
        return PeerInfo(
            node_id=kwargs.get("node_id", "did:vkb:web:peer1.example.com"),
            address=kwargs.get("address", "https://peer1.example.com/federation"),
            last_seen=kwargs.get("last_seen", datetime.now()),
            capabilities=kwargs.get("capabilities", ["belief_sync"]),
            domains=kwargs.get("domains", ["science"]),
            trust_phase=kwargs.get("trust_phase", "contributor"),
        )

    return _factory


@pytest.fixture
def sample_node():
    """Create a sample FederationNode."""

    def _factory(**kwargs):
        return FederationNode(
            id=kwargs.get("id", uuid4()),
            did=kwargs.get("did", "did:vkb:web:target.example.com"),
            federation_endpoint=kwargs.get(
                "federation_endpoint",
                "https://target.example.com/federation",
            ),
            status=NodeStatus(kwargs.get("status", "active")),
            trust_phase=TrustPhase(kwargs.get("trust_phase", "contributor")),
            capabilities=kwargs.get("capabilities", ["belief_sync"]),
            domains=kwargs.get("domains", ["test"]),
        )

    return _factory


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global protocol singleton between tests."""
    set_peer_exchange_protocol(None)
    yield
    set_peer_exchange_protocol(None)


# =============================================================================
# PEER INFO TESTS
# =============================================================================


class TestPeerInfo:
    """Tests for PeerInfo dataclass."""

    def test_create_peer_info(self, sample_peer_info):
        """Test basic PeerInfo creation."""
        peer = sample_peer_info()
        assert peer.node_id == "did:vkb:web:peer1.example.com"
        assert peer.address == "https://peer1.example.com/federation"
        assert peer.capabilities == ["belief_sync"]

    def test_to_dict(self, sample_peer_info):
        """Test serialization to dict."""
        peer = sample_peer_info()
        d = peer.to_dict()
        assert d["node_id"] == "did:vkb:web:peer1.example.com"
        assert d["address"] == "https://peer1.example.com/federation"
        assert d["capabilities"] == ["belief_sync"]
        assert d["domains"] == ["science"]
        assert d["trust_phase"] == "contributor"
        assert d["last_seen"] is not None

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "node_id": "did:vkb:web:peer2.example.com",
            "address": "https://peer2.example.com/federation",
            "last_seen": "2026-01-15T10:30:00",
            "capabilities": ["belief_sync", "aggregation"],
            "domains": ["tech"],
            "trust_phase": "participant",
        }
        peer = PeerInfo.from_dict(data)
        assert peer.node_id == "did:vkb:web:peer2.example.com"
        assert peer.address == "https://peer2.example.com/federation"
        assert peer.capabilities == ["belief_sync", "aggregation"]
        assert peer.last_seen is not None

    def test_from_dict_missing_fields(self):
        """Test deserialization with missing optional fields."""
        data = {
            "node_id": "did:vkb:web:minimal.example.com",
            "address": "https://minimal.example.com/federation",
        }
        peer = PeerInfo.from_dict(data)
        assert peer.node_id == "did:vkb:web:minimal.example.com"
        assert peer.capabilities == []
        assert peer.domains == []
        assert peer.last_seen is None

    def test_from_dict_invalid_timestamp(self):
        """Test deserialization with invalid timestamp."""
        data = {
            "node_id": "did:vkb:web:peer.example.com",
            "address": "https://peer.example.com/federation",
            "last_seen": "not-a-date",
        }
        peer = PeerInfo.from_dict(data)
        assert peer.last_seen is None

    def test_is_valid(self, sample_peer_info):
        """Test validation of valid peer info."""
        peer = sample_peer_info()
        assert peer.is_valid() is True

    def test_is_valid_missing_node_id(self, sample_peer_info):
        """Test validation with missing node_id."""
        peer = sample_peer_info(node_id="")
        assert peer.is_valid() is False

    def test_is_valid_missing_address(self, sample_peer_info):
        """Test validation with missing address."""
        peer = sample_peer_info(address="")
        assert peer.is_valid() is False

    def test_is_valid_non_did_node_id(self, sample_peer_info):
        """Test validation with non-DID node_id."""
        peer = sample_peer_info(node_id="not-a-did")
        assert peer.is_valid() is False

    def test_roundtrip_serialization(self, sample_peer_info):
        """Test that to_dict -> from_dict is lossless."""
        original = sample_peer_info()
        reconstructed = PeerInfo.from_dict(original.to_dict())
        assert reconstructed.node_id == original.node_id
        assert reconstructed.address == original.address
        assert reconstructed.capabilities == original.capabilities
        assert reconstructed.domains == original.domains
        assert reconstructed.trust_phase == original.trust_phase


# =============================================================================
# PEER EXCHANGE MESSAGE TESTS
# =============================================================================


class TestPeerExchangeMessage:
    """Tests for PeerExchangeMessage dataclass."""

    def test_create_message(self, sample_peer_info):
        """Test creating a peer exchange message."""
        peers = [sample_peer_info(), sample_peer_info(node_id="did:vkb:web:peer2.example.com")]
        msg = PeerExchangeMessage(
            sender_did="did:vkb:web:sender.example.com",
            peers=peers,
        )
        assert msg.sender_did == "did:vkb:web:sender.example.com"
        assert len(msg.peers) == 2
        assert msg.max_hops == 2

    def test_to_dict(self, sample_peer_info):
        """Test serialization."""
        msg = PeerExchangeMessage(
            sender_did="did:vkb:web:sender.example.com",
            peers=[sample_peer_info()],
            max_hops=3,
        )
        d = msg.to_dict()
        assert d["type"] == "peer_exchange"
        assert d["sender_did"] == "did:vkb:web:sender.example.com"
        assert len(d["peers"]) == 1
        assert d["max_hops"] == 3
        assert "timestamp" in d

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "type": "peer_exchange",
            "sender_did": "did:vkb:web:sender.example.com",
            "peers": [
                {
                    "node_id": "did:vkb:web:peer1.example.com",
                    "address": "https://peer1.example.com/federation",
                }
            ],
            "timestamp": "2026-01-15T10:30:00",
            "max_hops": 1,
        }
        msg = PeerExchangeMessage.from_dict(data)
        assert msg.sender_did == "did:vkb:web:sender.example.com"
        assert len(msg.peers) == 1
        assert msg.max_hops == 1

    def test_from_dict_empty_peers(self):
        """Test deserialization with no peers."""
        data = {
            "sender_did": "did:vkb:web:empty.example.com",
        }
        msg = PeerExchangeMessage.from_dict(data)
        assert msg.peers == []
        assert msg.max_hops == 2  # default

    def test_roundtrip(self, sample_peer_info):
        """Test to_dict -> from_dict roundtrip."""
        original = PeerExchangeMessage(
            sender_did="did:vkb:web:test.example.com",
            peers=[sample_peer_info()],
            max_hops=3,
        )
        reconstructed = PeerExchangeMessage.from_dict(original.to_dict())
        assert reconstructed.sender_did == original.sender_did
        assert len(reconstructed.peers) == len(original.peers)
        assert reconstructed.max_hops == original.max_hops


# =============================================================================
# PEER EXCHANGE RESPONSE TESTS
# =============================================================================


class TestPeerExchangeResponse:
    """Tests for PeerExchangeResponse dataclass."""

    def test_create_response(self, sample_peer_info):
        """Test creating a response."""
        resp = PeerExchangeResponse(
            responder_did="did:vkb:web:responder.example.com",
            peers=[sample_peer_info()],
            accepted=3,
            rejected=1,
        )
        assert resp.responder_did == "did:vkb:web:responder.example.com"
        assert resp.accepted == 3
        assert resp.rejected == 1

    def test_to_dict(self, sample_peer_info):
        """Test serialization."""
        resp = PeerExchangeResponse(
            responder_did="did:vkb:web:responder.example.com",
            peers=[sample_peer_info()],
            accepted=2,
            rejected=0,
        )
        d = resp.to_dict()
        assert d["type"] == "peer_exchange_response"
        assert d["accepted"] == 2
        assert d["rejected"] == 0

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "responder_did": "did:vkb:web:responder.example.com",
            "peers": [
                {
                    "node_id": "did:vkb:web:peer.example.com",
                    "address": "https://peer.example.com/federation",
                }
            ],
            "accepted": 5,
            "rejected": 2,
        }
        resp = PeerExchangeResponse.from_dict(data)
        assert resp.responder_did == "did:vkb:web:responder.example.com"
        assert resp.accepted == 5
        assert resp.rejected == 2
        assert len(resp.peers) == 1

    def test_roundtrip(self, sample_peer_info):
        """Test to_dict -> from_dict roundtrip."""
        original = PeerExchangeResponse(
            responder_did="did:vkb:web:test.example.com",
            peers=[sample_peer_info()],
            accepted=1,
            rejected=0,
        )
        reconstructed = PeerExchangeResponse.from_dict(original.to_dict())
        assert reconstructed.responder_did == original.responder_did
        assert reconstructed.accepted == original.accepted
        assert reconstructed.rejected == original.rejected


# =============================================================================
# PEER EXCHANGE PROTOCOL TESTS
# =============================================================================


class TestPeerExchangeProtocolRateLimiting:
    """Tests for rate limiting in PeerExchangeProtocol."""

    def test_rate_limit_allows_initial_exchange(self, protocol):
        """First exchange should always be allowed."""
        assert protocol._check_rate_limit("did:vkb:web:peer.example.com") is True

    def test_rate_limit_blocks_after_max(self, protocol):
        """Should block after exceeding rate limit."""
        peer_did = "did:vkb:web:spammer.example.com"
        for _ in range(protocol.rate_limit):
            protocol._record_exchange(peer_did)

        assert protocol._check_rate_limit(peer_did) is False

    def test_rate_limit_allows_after_window(self, protocol):
        """Should allow exchanges after the window expires."""
        peer_did = "did:vkb:web:peer.example.com"

        # Fill up rate limit with old timestamps
        old_time = time.monotonic() - protocol.rate_window - 1
        protocol._exchange_timestamps[peer_did] = [old_time] * protocol.rate_limit

        assert protocol._check_rate_limit(peer_did) is True

    def test_rate_limit_per_peer(self, protocol):
        """Rate limits should be independent per peer."""
        peer_a = "did:vkb:web:a.example.com"
        peer_b = "did:vkb:web:b.example.com"

        for _ in range(protocol.rate_limit):
            protocol._record_exchange(peer_a)

        assert protocol._check_rate_limit(peer_a) is False
        assert protocol._check_rate_limit(peer_b) is True

    def test_record_exchange_tracks_timestamp(self, protocol):
        """Record should add a timestamp entry."""
        peer_did = "did:vkb:web:peer.example.com"
        protocol._record_exchange(peer_did)
        assert len(protocol._exchange_timestamps[peer_did]) == 1

    def test_rate_limit_prunes_old_timestamps(self, protocol):
        """Old timestamps should be pruned during check."""
        peer_did = "did:vkb:web:peer.example.com"
        old_time = time.monotonic() - protocol.rate_window - 100
        protocol._exchange_timestamps[peer_did] = [old_time] * 20

        protocol._check_rate_limit(peer_did)
        assert len(protocol._exchange_timestamps[peer_did]) == 0


class TestPeerExchangeProtocolShareablePeers:
    """Tests for get_shareable_peers."""

    def test_returns_active_peers(self, protocol, mock_get_cursor):
        """Should return active, trusted peers."""
        now = datetime.now()
        mock_get_cursor.fetchall.return_value = [
            {
                "did": "did:vkb:web:peer1.example.com",
                "federation_endpoint": "https://peer1.example.com/federation",
                "last_seen_at": now,
                "capabilities": ["belief_sync"],
                "domains": ["science"],
                "trust_phase": "contributor",
                "metadata": {},
                "trust_overall": 0.5,
            }
        ]

        peers = protocol.get_shareable_peers()
        assert len(peers) == 1
        assert peers[0].node_id == "did:vkb:web:peer1.example.com"

    def test_excludes_opted_out_peers(self, protocol, mock_get_cursor):
        """Should skip peers that opted out of discovery."""
        mock_get_cursor.fetchall.return_value = [
            {
                "did": "did:vkb:web:private.example.com",
                "federation_endpoint": "https://private.example.com/federation",
                "last_seen_at": datetime.now(),
                "capabilities": ["belief_sync"],
                "domains": ["private"],
                "trust_phase": "contributor",
                "metadata": {"discovery_opt_out": True},
                "trust_overall": 0.8,
            }
        ]

        peers = protocol.get_shareable_peers()
        assert len(peers) == 0

    def test_excludes_low_trust_peers(self, protocol, mock_get_cursor):
        """Should skip peers below minimum trust."""
        mock_get_cursor.fetchall.return_value = [
            {
                "did": "did:vkb:web:untrusted.example.com",
                "federation_endpoint": "https://untrusted.example.com/federation",
                "last_seen_at": datetime.now(),
                "capabilities": [],
                "domains": [],
                "trust_phase": "contributor",
                "metadata": {},
                "trust_overall": 0.1,  # Below min_trust of 0.2
            }
        ]

        peers = protocol.get_shareable_peers()
        assert len(peers) == 0

    def test_handles_none_metadata(self, protocol, mock_get_cursor):
        """Should handle peers with None metadata gracefully."""
        mock_get_cursor.fetchall.return_value = [
            {
                "did": "did:vkb:web:peer.example.com",
                "federation_endpoint": "https://peer.example.com/federation",
                "last_seen_at": datetime.now(),
                "capabilities": [],
                "domains": [],
                "trust_phase": "participant",
                "metadata": None,
                "trust_overall": 0.6,
            }
        ]

        peers = protocol.get_shareable_peers()
        assert len(peers) == 1

    def test_handles_db_error(self, protocol, mock_get_cursor):
        """Should return empty list on database error."""
        mock_get_cursor.fetchall.side_effect = Exception("DB error")

        peers = protocol.get_shareable_peers()
        assert peers == []


class TestPeerExchangeProtocolMergePeers:
    """Tests for merge_peers."""

    def test_merge_valid_new_peers(self, protocol, mock_get_cursor):
        """Should accept valid new peers."""
        # get_node_by_did returns None (peer not known)
        mock_get_cursor.fetchone.side_effect = [
            None,  # get_node_by_did check
            {"id": uuid4()},  # INSERT result
        ]

        peers = [
            PeerInfo(
                node_id="did:vkb:web:new-peer.example.com",
                address="https://new-peer.example.com/federation",
            )
        ]

        accepted, rejected = protocol.merge_peers(peers)
        assert accepted == 1
        assert rejected == 0

    def test_reject_invalid_peers(self, protocol, mock_get_cursor):
        """Should reject peers with invalid info."""
        peers = [
            PeerInfo(node_id="", address="https://example.com/federation"),
            PeerInfo(node_id="did:vkb:web:valid.com", address=""),
            PeerInfo(node_id="not-a-did", address="https://example.com"),
        ]

        accepted, rejected = protocol.merge_peers(peers)
        assert accepted == 0
        assert rejected == 3

    def test_reject_self(self, protocol, mock_get_cursor):
        """Should reject our own DID."""
        peers = [
            PeerInfo(
                node_id="did:vkb:web:local.example.com",  # same as protocol.local_did
                address="https://local.example.com/federation",
            )
        ]

        accepted, rejected = protocol.merge_peers(peers)
        assert accepted == 0
        assert rejected == 1

    def test_reject_already_known(self, protocol, mock_get_cursor):
        """Should reject peers we already know."""
        existing_node = MagicMock(spec=FederationNode)

        with patch("valence.federation.discovery.get_node_by_did", return_value=existing_node):
            peers = [
                PeerInfo(
                    node_id="did:vkb:web:known.example.com",
                    address="https://known.example.com/federation",
                )
            ]

            accepted, rejected = protocol.merge_peers(peers)
            assert accepted == 0
            assert rejected == 1

    def test_reject_invalid_address(self, protocol, mock_get_cursor):
        """Should reject peers with invalid addresses."""
        with patch("valence.federation.discovery.get_node_by_did", return_value=None):
            peers = [
                PeerInfo(
                    node_id="did:vkb:web:badaddr.example.com",
                    address="ftp://invalid-protocol.example.com",
                ),
            ]

            accepted, rejected = protocol.merge_peers(peers)
            assert accepted == 0
            assert rejected == 1

    def test_merge_handles_db_error(self, protocol, mock_get_cursor):
        """Should handle database errors gracefully during merge."""
        mock_get_cursor.fetchone.side_effect = Exception("DB connection lost")

        with patch("valence.federation.discovery.get_node_by_did", return_value=None):
            peers = [
                PeerInfo(
                    node_id="did:vkb:web:peer.example.com",
                    address="https://peer.example.com/federation",
                )
            ]

            accepted, rejected = protocol.merge_peers(peers)
            assert accepted == 0
            assert rejected == 1

    def test_merge_on_conflict_does_nothing(self, protocol, mock_get_cursor):
        """Should handle ON CONFLICT (race condition) gracefully."""
        mock_get_cursor.fetchone.return_value = None  # INSERT returned no row (conflict)

        with patch("valence.federation.discovery.get_node_by_did", return_value=None):
            peers = [
                PeerInfo(
                    node_id="did:vkb:web:race.example.com",
                    address="https://race.example.com/federation",
                )
            ]

            accepted, rejected = protocol.merge_peers(peers)
            assert accepted == 0
            assert rejected == 1

    def test_merge_multiple_peers(self, protocol, mock_get_cursor):
        """Should process multiple peers correctly."""
        # First peer: new, gets accepted
        # Second peer: already known, rejected
        # Third peer: invalid, rejected
        call_count = 0

        def fake_get_node(did):
            if did == "did:vkb:web:known.example.com":
                return MagicMock(spec=FederationNode)
            return None

        def fake_fetchone():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"id": uuid4()}
            return None

        mock_get_cursor.fetchone.side_effect = fake_fetchone

        with patch("valence.federation.discovery.get_node_by_did", side_effect=fake_get_node):
            peers = [
                PeerInfo(
                    node_id="did:vkb:web:new.example.com",
                    address="https://new.example.com/federation",
                ),
                PeerInfo(
                    node_id="did:vkb:web:known.example.com",
                    address="https://known.example.com/federation",
                ),
                PeerInfo(node_id="", address=""),
            ]

            accepted, rejected = protocol.merge_peers(peers)
            assert accepted == 1
            assert rejected == 2


class TestPeerExchangeProtocolHandleExchange:
    """Tests for handle_peer_exchange."""

    def test_handle_exchange_normal(self, protocol, mock_get_cursor):
        """Should process incoming exchange and return our peers."""
        # Setup: get_shareable_peers returns some peers
        mock_get_cursor.fetchall.return_value = [
            {
                "did": "did:vkb:web:our-peer.example.com",
                "federation_endpoint": "https://our-peer.example.com/federation",
                "last_seen_at": datetime.now(),
                "capabilities": ["belief_sync"],
                "domains": ["test"],
                "trust_phase": "contributor",
                "metadata": {},
                "trust_overall": 0.5,
            }
        ]

        request = PeerExchangeMessage(
            sender_did="did:vkb:web:sender.example.com",
            peers=[],  # No peers sent
        )

        response = protocol.handle_peer_exchange(request)
        assert response.responder_did == "did:vkb:web:local.example.com"
        assert len(response.peers) == 1
        assert response.accepted == 0
        assert response.rejected == 0

    def test_handle_exchange_rate_limited(self, protocol):
        """Should reject exchange when rate limited."""
        peer_did = "did:vkb:web:spammer.example.com"

        # Exhaust rate limit
        for _ in range(protocol.rate_limit):
            protocol._record_exchange(peer_did)

        request = PeerExchangeMessage(
            sender_did=peer_did,
            peers=[
                PeerInfo(
                    node_id="did:vkb:web:spam-peer.example.com",
                    address="https://spam.example.com/federation",
                )
            ],
        )

        response = protocol.handle_peer_exchange(request)
        assert response.peers == []
        assert response.accepted == 0
        assert response.rejected == 1

    def test_handle_exchange_with_incoming_peers(self, protocol, mock_get_cursor):
        """Should merge incoming peers and return our own."""
        # For get_shareable_peers query
        mock_get_cursor.fetchall.return_value = []

        # For merge_peers INSERT returning a new node id
        mock_get_cursor.fetchone.return_value = {"id": uuid4()}

        request = PeerExchangeMessage(
            sender_did="did:vkb:web:sender.example.com",
            peers=[
                PeerInfo(
                    node_id="did:vkb:web:new-peer.example.com",
                    address="https://new-peer.example.com/federation",
                )
            ],
        )

        with patch("valence.federation.discovery.get_node_by_did", return_value=None):
            response = protocol.handle_peer_exchange(request)

        assert response.accepted == 1
        assert response.rejected == 0


class TestPeerExchangeProtocolRequestPeers:
    """Tests for request_peers (HTTP client)."""

    @pytest.mark.asyncio
    async def test_request_peers_no_endpoint(self, protocol, sample_node):
        """Should return empty list if node has no endpoint."""
        node = sample_node(federation_endpoint=None)
        result = await protocol.request_peers(node)
        assert result == []

    @pytest.mark.asyncio
    async def test_request_peers_rate_limited(self, protocol, sample_node):
        """Should return empty list when rate limited."""
        node = sample_node()
        for _ in range(protocol.rate_limit):
            protocol._record_exchange(node.did)

        result = await protocol.request_peers(node)
        assert result == []

    @pytest.mark.asyncio
    async def test_request_peers_success(self, protocol, sample_node, mock_get_cursor):
        """Should exchange peers over HTTP and return new nodes."""
        node = sample_node()

        # Mock get_shareable_peers (returns empty for simplicity)
        mock_get_cursor.fetchall.return_value = []

        # Mock HTTP response
        response_data = PeerExchangeResponse(
            responder_did=node.did,
            peers=[
                PeerInfo(
                    node_id="did:vkb:web:discovered.example.com",
                    address="https://discovered.example.com/federation",
                )
            ],
            accepted=0,
            rejected=0,
        ).to_dict()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        discovered_node = sample_node(did="did:vkb:web:discovered.example.com")

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch("valence.federation.discovery.get_node_by_did") as mock_get_node,
        ):
            # merge_peers check: first call returns None (new), second returns the node
            mock_get_node.side_effect = [None, discovered_node]
            # DB insert for merge
            mock_get_cursor.fetchone.return_value = {"id": uuid4()}

            result = await protocol.request_peers(node)

        assert len(result) == 1
        assert result[0].did == "did:vkb:web:discovered.example.com"

    @pytest.mark.asyncio
    async def test_request_peers_http_error(self, protocol, sample_node, mock_get_cursor):
        """Should handle HTTP errors gracefully."""
        node = sample_node()
        mock_get_cursor.fetchall.return_value = []

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await protocol.request_peers(node)

        assert result == []

    @pytest.mark.asyncio
    async def test_request_peers_network_error(self, protocol, sample_node, mock_get_cursor):
        """Should handle network errors gracefully."""
        import aiohttp

        node = sample_node()
        mock_get_cursor.fetchall.return_value = []

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await protocol.request_peers(node)

        assert result == []


class TestAddressValidation:
    """Tests for _validate_address."""

    def test_valid_https_url(self):
        """Should accept valid HTTPS URLs."""
        assert PeerExchangeProtocol._validate_address("https://example.com/federation") is True

    def test_valid_http_url(self):
        """Should accept HTTP URLs (dev environments)."""
        assert PeerExchangeProtocol._validate_address("http://localhost:8080/federation") is True

    def test_reject_empty(self):
        """Should reject empty address."""
        assert PeerExchangeProtocol._validate_address("") is False

    def test_reject_non_http(self):
        """Should reject non-HTTP protocols."""
        assert PeerExchangeProtocol._validate_address("ftp://example.com") is False
        assert PeerExchangeProtocol._validate_address("ws://example.com") is False

    def test_reject_too_long(self):
        """Should reject absurdly long URLs."""
        long_url = "https://example.com/" + "a" * 2100
        assert PeerExchangeProtocol._validate_address(long_url) is False

    def test_reject_no_hostname(self):
        """Should reject URLs without hostname."""
        assert PeerExchangeProtocol._validate_address("https://") is False


# =============================================================================
# INTEGRATION: exchange_peers function
# =============================================================================


class TestExchangePeersFunction:
    """Tests for the top-level exchange_peers function."""

    @pytest.mark.asyncio
    async def test_exchange_peers_no_endpoint(self, sample_node):
        """Should return empty list for node without endpoint."""
        node = sample_node(federation_endpoint=None)
        result = await exchange_peers(node)
        assert result == []

    @pytest.mark.asyncio
    async def test_exchange_peers_uses_protocol(self, sample_node, mock_get_cursor):
        """Should delegate to PeerExchangeProtocol."""
        node = sample_node()
        mock_get_cursor.fetchall.return_value = []

        mock_protocol = MagicMock()
        mock_protocol.request_peers = AsyncMock(return_value=[])
        set_peer_exchange_protocol(mock_protocol)

        result = await exchange_peers(node)
        mock_protocol.request_peers.assert_called_once_with(node)
        assert result == []


# =============================================================================
# SINGLETON MANAGEMENT
# =============================================================================


class TestSingletonManagement:
    """Tests for get/set_peer_exchange_protocol."""

    def test_get_creates_default(self):
        """Should create a default instance if none set."""
        with patch("valence.federation.discovery.get_config") as mock_config:
            mock_config.return_value = MagicMock(node_did="did:vkb:web:test.example.com")
            protocol = get_peer_exchange_protocol()
            assert isinstance(protocol, PeerExchangeProtocol)
            assert protocol.local_did == "did:vkb:web:test.example.com"

    def test_set_and_get(self):
        """Should return the protocol that was set."""
        custom = PeerExchangeProtocol(local_did="did:vkb:web:custom.example.com")
        set_peer_exchange_protocol(custom)
        assert get_peer_exchange_protocol() is custom

    def test_set_none_clears(self):
        """Setting None should clear the singleton."""
        custom = PeerExchangeProtocol(local_did="test")
        set_peer_exchange_protocol(custom)
        set_peer_exchange_protocol(None)

        with patch("valence.federation.discovery.get_config") as mock_config:
            mock_config.return_value = MagicMock(node_did="")
            result = get_peer_exchange_protocol()
            assert result is not custom
