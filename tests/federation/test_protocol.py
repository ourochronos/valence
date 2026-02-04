"""Tests for federation protocol handlers.

Tests cover:
- Message type enums
- Protocol message dataclasses
- Authentication flow (challenge/verify)
- Belief handlers (share, request)
- Sync handlers
- Message parsing and dispatch
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4, UUID

import pytest

from valence.federation.protocol import (
    MessageType,
    ErrorCode,
    ProtocolMessage,
    ErrorMessage,
    AuthChallengeRequest,
    AuthChallengeResponse,
    AuthVerifyRequest,
    AuthVerifyResponse,
    ShareBeliefRequest,
    ShareBeliefResponse,
    RequestBeliefsRequest,
    BeliefsResponse,
    SyncRequest,
    SyncChange,
    SyncResponse,
    TrustAttestationRequest,
    TrustAttestationResponse,
    create_auth_challenge,
    verify_auth_challenge,
    handle_share_belief,
    handle_request_beliefs,
    handle_sync_request,
    parse_message,
    handle_message,
    _pending_challenges,
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

    with patch("valence.federation.protocol.get_cursor", _mock_get_cursor):
        yield mock_cursor


@pytest.fixture(autouse=True)
def clear_pending_challenges():
    """Clear pending challenges before each test."""
    _pending_challenges.clear()
    yield
    _pending_challenges.clear()


# =============================================================================
# MESSAGE TYPE TESTS
# =============================================================================


class TestMessageType:
    """Tests for MessageType enum."""

    def test_auth_message_types(self):
        """Test authentication message types exist."""
        assert MessageType.AUTH_CHALLENGE is not None
        assert MessageType.AUTH_CHALLENGE_RESPONSE is not None
        assert MessageType.AUTH_VERIFY is not None
        assert MessageType.AUTH_VERIFY_RESPONSE is not None

    def test_belief_message_types(self):
        """Test belief message types exist."""
        assert MessageType.SHARE_BELIEF is not None
        assert MessageType.SHARE_BELIEF_RESPONSE is not None
        assert MessageType.REQUEST_BELIEFS is not None
        assert MessageType.BELIEFS_RESPONSE is not None

    def test_sync_message_types(self):
        """Test sync message types exist."""
        assert MessageType.SYNC_REQUEST is not None
        assert MessageType.SYNC_RESPONSE is not None

    def test_error_message_type(self):
        """Test error message type exists."""
        assert MessageType.ERROR is not None


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_error_codes_defined(self):
        """Test all error codes are defined."""
        expected_codes = [
            "AUTH_FAILED",
            "TRUST_INSUFFICIENT",
            "VISIBILITY_DENIED",
            "RATE_LIMITED",
            "SYNC_CURSOR_INVALID",
            "SIGNATURE_INVALID",
            "INTERNAL_ERROR",
            "INVALID_REQUEST",
            "NODE_NOT_FOUND",
        ]
        
        for code_name in expected_codes:
            assert hasattr(ErrorCode, code_name)


# =============================================================================
# PROTOCOL MESSAGE TESTS
# =============================================================================


class TestProtocolMessage:
    """Tests for ProtocolMessage base class."""

    def test_protocol_message_creation(self):
        """Test creating a protocol message."""
        msg = ProtocolMessage(type=MessageType.ERROR)
        
        assert msg.type == MessageType.ERROR
        assert msg.request_id is not None
        assert msg.timestamp is not None

    def test_protocol_message_to_dict(self):
        """Test converting message to dict."""
        msg = ProtocolMessage(type=MessageType.ERROR)
        result = msg.to_dict()
        
        assert result["type"] == "ERROR"
        assert "request_id" in result
        assert "timestamp" in result


class TestErrorMessage:
    """Tests for ErrorMessage dataclass."""

    def test_error_message_creation(self):
        """Test creating an error message."""
        msg = ErrorMessage(
            error_code=ErrorCode.AUTH_FAILED,
            message="Authentication failed",
        )
        
        assert msg.type == MessageType.ERROR
        assert msg.error_code == ErrorCode.AUTH_FAILED
        assert msg.message == "Authentication failed"

    def test_error_message_to_dict(self):
        """Test converting error message to dict."""
        msg = ErrorMessage(
            error_code=ErrorCode.TRUST_INSUFFICIENT,
            message="Trust too low",
            details={"required": 0.5, "current": 0.1},
        )
        result = msg.to_dict()
        
        assert result["error_code"] == "TRUST_INSUFFICIENT"
        assert result["message"] == "Trust too low"
        assert result["details"]["required"] == 0.5


# =============================================================================
# AUTHENTICATION MESSAGE TESTS
# =============================================================================


class TestAuthChallengeRequest:
    """Tests for AuthChallengeRequest."""

    def test_challenge_request_creation(self):
        """Test creating a challenge request."""
        msg = AuthChallengeRequest(client_did="did:vkb:web:test.example.com")
        
        assert msg.type == MessageType.AUTH_CHALLENGE
        assert msg.client_did == "did:vkb:web:test.example.com"

    def test_challenge_request_to_dict(self):
        """Test converting challenge request to dict."""
        msg = AuthChallengeRequest(client_did="did:vkb:web:test.example.com")
        result = msg.to_dict()
        
        assert result["client_did"] == "did:vkb:web:test.example.com"


class TestAuthChallengeResponse:
    """Tests for AuthChallengeResponse."""

    def test_challenge_response_creation(self):
        """Test creating a challenge response."""
        msg = AuthChallengeResponse(challenge="abc123")
        
        assert msg.type == MessageType.AUTH_CHALLENGE_RESPONSE
        assert msg.challenge == "abc123"
        assert msg.expires_at is not None

    def test_challenge_response_to_dict(self):
        """Test converting challenge response to dict."""
        msg = AuthChallengeResponse(challenge="xyz789")
        result = msg.to_dict()
        
        assert result["challenge"] == "xyz789"
        assert "expires_at" in result


class TestAuthVerifyRequest:
    """Tests for AuthVerifyRequest."""

    def test_verify_request_creation(self):
        """Test creating a verify request."""
        msg = AuthVerifyRequest(
            client_did="did:vkb:web:test.example.com",
            challenge="abc123",
            signature="sig_data"
        )
        
        assert msg.type == MessageType.AUTH_VERIFY
        assert msg.client_did == "did:vkb:web:test.example.com"
        assert msg.challenge == "abc123"
        assert msg.signature == "sig_data"


class TestAuthVerifyResponse:
    """Tests for AuthVerifyResponse."""

    def test_verify_response_creation(self):
        """Test creating a verify response."""
        msg = AuthVerifyResponse(session_token="token123")
        
        assert msg.type == MessageType.AUTH_VERIFY_RESPONSE
        assert msg.session_token == "token123"


# =============================================================================
# BELIEF MESSAGE TESTS
# =============================================================================


class TestShareBeliefRequest:
    """Tests for ShareBeliefRequest."""

    def test_share_belief_request_creation(self):
        """Test creating a share belief request."""
        beliefs = [{"content": "Test belief", "confidence": 0.8}]
        msg = ShareBeliefRequest(beliefs=beliefs)
        
        assert msg.type == MessageType.SHARE_BELIEF
        assert len(msg.beliefs) == 1
        assert msg.beliefs[0]["content"] == "Test belief"


class TestShareBeliefResponse:
    """Tests for ShareBeliefResponse."""

    def test_share_belief_response_creation(self):
        """Test creating a share belief response."""
        msg = ShareBeliefResponse(
            accepted=5,
            rejected=2,
            rejection_reasons={"belief1": "Invalid signature"},
        )
        
        assert msg.type == MessageType.SHARE_BELIEF_RESPONSE
        assert msg.accepted == 5
        assert msg.rejected == 2


class TestRequestBeliefsRequest:
    """Tests for RequestBeliefsRequest."""

    def test_request_beliefs_creation(self):
        """Test creating a request beliefs message."""
        msg = RequestBeliefsRequest(
            requester_did="did:vkb:web:test.example.com",
            domain_filter=["science"],
            min_confidence=0.5,
            limit=20,
        )
        
        assert msg.type == MessageType.REQUEST_BELIEFS
        assert msg.domain_filter == ["science"]
        assert msg.min_confidence == 0.5


class TestBeliefsResponse:
    """Tests for BeliefsResponse."""

    def test_beliefs_response_creation(self):
        """Test creating a beliefs response."""
        beliefs = [{"id": str(uuid4()), "content": "Belief"}]
        msg = BeliefsResponse(
            beliefs=beliefs,
            total_available=100,
            cursor="cursor123",
        )
        
        assert msg.type == MessageType.BELIEFS_RESPONSE
        assert len(msg.beliefs) == 1
        assert msg.total_available == 100


# =============================================================================
# SYNC MESSAGE TESTS
# =============================================================================


class TestSyncRequest:
    """Tests for SyncRequest."""

    def test_sync_request_creation(self):
        """Test creating a sync request."""
        since = datetime.now() - timedelta(days=1)
        msg = SyncRequest(
            since=since,
            domains=["science", "tech"],
            cursor="cursor123",
        )
        
        assert msg.type == MessageType.SYNC_REQUEST
        assert msg.since == since
        assert msg.domains == ["science", "tech"]


class TestSyncChange:
    """Tests for SyncChange dataclass."""

    def test_sync_change_creation(self):
        """Test creating a sync change."""
        change = SyncChange(
            change_type="belief_created",
            belief={"id": str(uuid4()), "content": "New belief"},
        )
        
        assert change.change_type == "belief_created"
        assert change.belief is not None

    def test_sync_change_to_dict(self):
        """Test converting sync change to dict."""
        change = SyncChange(
            change_type="belief_superseded",
            belief={"id": "123"},
            old_belief_id="old_123",
        )
        result = change.to_dict()
        
        assert result["type"] == "belief_superseded"
        assert result["old_belief_id"] == "old_123"


class TestSyncResponse:
    """Tests for SyncResponse."""

    def test_sync_response_creation(self):
        """Test creating a sync response."""
        changes = [
            SyncChange(change_type="belief_created", belief={}),
            SyncChange(change_type="belief_superseded", belief={}),
        ]
        msg = SyncResponse(
            changes=changes,
            cursor="new_cursor",
            has_more=True,
        )
        
        assert msg.type == MessageType.SYNC_RESPONSE
        assert len(msg.changes) == 2
        assert msg.has_more is True


# =============================================================================
# AUTHENTICATION FLOW TESTS
# =============================================================================


class TestCreateAuthChallenge:
    """Tests for create_auth_challenge function."""

    def test_create_challenge(self):
        """Test creating an auth challenge."""
        client_did = "did:vkb:web:test.example.com"
        
        response = create_auth_challenge(client_did)
        
        assert response.challenge is not None
        assert len(response.challenge) == 64  # 32 bytes hex = 64 chars
        assert response.expires_at > datetime.now()

    def test_challenge_stored(self):
        """Test that challenge is stored in pending challenges."""
        client_did = "did:vkb:web:test.example.com"
        
        response = create_auth_challenge(client_did)
        
        assert client_did in _pending_challenges
        stored_challenge, expires_at = _pending_challenges[client_did]
        assert stored_challenge == response.challenge


class TestVerifyAuthChallenge:
    """Tests for verify_auth_challenge function."""

    def test_verify_no_pending_challenge(self):
        """Test verify with no pending challenge."""
        result = verify_auth_challenge(
            client_did="did:vkb:web:test.example.com",
            challenge="abc123",
            signature="sig",
            public_key_multibase="z6Mk...",
        )
        
        assert isinstance(result, ErrorMessage)
        assert result.error_code == ErrorCode.AUTH_FAILED
        assert "No pending challenge" in result.message

    def test_verify_expired_challenge(self):
        """Test verify with expired challenge."""
        client_did = "did:vkb:web:test.example.com"
        # Add expired challenge
        _pending_challenges[client_did] = ("abc123", datetime.now() - timedelta(minutes=10))
        
        result = verify_auth_challenge(
            client_did=client_did,
            challenge="abc123",
            signature="sig",
            public_key_multibase="z6Mk...",
        )
        
        assert isinstance(result, ErrorMessage)
        assert result.error_code == ErrorCode.AUTH_FAILED
        assert "expired" in result.message.lower()

    def test_verify_challenge_mismatch(self):
        """Test verify with wrong challenge."""
        client_did = "did:vkb:web:test.example.com"
        _pending_challenges[client_did] = ("correct_challenge", datetime.now() + timedelta(minutes=5))
        
        result = verify_auth_challenge(
            client_did=client_did,
            challenge="wrong_challenge",
            signature="sig",
            public_key_multibase="z6Mk...",
        )
        
        assert isinstance(result, ErrorMessage)
        assert result.error_code == ErrorCode.AUTH_FAILED
        assert "mismatch" in result.message.lower()


# =============================================================================
# BELIEF HANDLER TESTS
# =============================================================================


class TestHandleShareBelief:
    """Tests for handle_share_belief function."""

    def test_share_belief_empty_list(self):
        """Test sharing empty belief list."""
        request = ShareBeliefRequest(beliefs=[])
        
        response = handle_share_belief(
            request,
            sender_node_id=uuid4(),
            sender_trust=0.5,
        )
        
        assert isinstance(response, ShareBeliefResponse)
        assert response.accepted == 0
        assert response.rejected == 0

    def test_share_belief_missing_fields(self, mock_get_cursor):
        """Test sharing belief with missing fields."""
        beliefs = [{"content": "Test"}]  # Missing required fields
        request = ShareBeliefRequest(beliefs=beliefs)
        
        response = handle_share_belief(
            request,
            sender_node_id=uuid4(),
            sender_trust=0.5,
        )
        
        assert response.rejected == 1
        assert "Missing required field" in list(response.rejection_reasons.values())[0]


class TestHandleRequestBeliefs:
    """Tests for handle_request_beliefs function."""

    def test_request_beliefs_low_trust(self):
        """Test requesting beliefs with insufficient trust."""
        request = RequestBeliefsRequest(
            requester_did="did:vkb:web:test.example.com",
        )
        
        response = handle_request_beliefs(
            request,
            requester_node_id=uuid4(),
            requester_trust=0.05,  # Too low
        )
        
        assert isinstance(response, ErrorMessage)
        assert response.error_code == ErrorCode.TRUST_INSUFFICIENT

    def test_request_beliefs_success(self, mock_get_cursor):
        """Test successful belief request."""
        mock_get_cursor.fetchall.return_value = []
        mock_get_cursor.fetchone.return_value = {"total": 0}
        
        request = RequestBeliefsRequest(
            requester_did="did:vkb:web:test.example.com",
        )
        
        with patch("valence.server.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                federation_node_did="did:vkb:web:local",
                federation_private_key=None,
            )
            
            response = handle_request_beliefs(
                request,
                requester_node_id=uuid4(),
                requester_trust=0.5,
            )
        
        assert isinstance(response, BeliefsResponse)


# =============================================================================
# SYNC HANDLER TESTS
# =============================================================================


class TestHandleSyncRequest:
    """Tests for handle_sync_request function."""

    def test_sync_request_low_trust(self):
        """Test sync request with insufficient trust."""
        request = SyncRequest()
        
        response = handle_sync_request(
            request,
            requester_node_id=uuid4(),
            requester_trust=0.1,  # Too low for sync
        )
        
        assert isinstance(response, ErrorMessage)
        assert response.error_code == ErrorCode.TRUST_INSUFFICIENT

    def test_sync_request_success(self, mock_get_cursor):
        """Test successful sync request."""
        mock_get_cursor.fetchall.return_value = []
        
        request = SyncRequest(
            since=datetime.now() - timedelta(days=1),
            domains=["test"],
        )
        
        with patch("valence.server.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                federation_node_did="did:vkb:web:local",
                federation_private_key=None,
            )
            
            response = handle_sync_request(
                request,
                requester_node_id=uuid4(),
                requester_trust=0.5,  # Enough for sync
            )
        
        assert isinstance(response, SyncResponse)
        assert response.has_more is False


# =============================================================================
# MESSAGE PARSING TESTS
# =============================================================================


class TestParseMessage:
    """Tests for parse_message function."""

    def test_parse_auth_challenge(self):
        """Test parsing AUTH_CHALLENGE message."""
        data = {
            "type": "AUTH_CHALLENGE",
            "request_id": str(uuid4()),
            "client_did": "did:vkb:web:test.example.com",
        }
        
        result = parse_message(data)
        
        assert isinstance(result, AuthChallengeRequest)
        assert result.client_did == "did:vkb:web:test.example.com"

    def test_parse_auth_verify(self):
        """Test parsing AUTH_VERIFY message."""
        data = {
            "type": "AUTH_VERIFY",
            "request_id": str(uuid4()),
            "client_did": "did:vkb:web:test.example.com",
            "challenge": "abc123",
            "signature": "sig_data",
        }
        
        result = parse_message(data)
        
        assert isinstance(result, AuthVerifyRequest)
        assert result.challenge == "abc123"

    def test_parse_share_belief(self):
        """Test parsing SHARE_BELIEF message."""
        data = {
            "type": "SHARE_BELIEF",
            "request_id": str(uuid4()),
            "beliefs": [{"content": "Test"}],
        }
        
        result = parse_message(data)
        
        assert isinstance(result, ShareBeliefRequest)
        assert len(result.beliefs) == 1

    def test_parse_request_beliefs(self):
        """Test parsing REQUEST_BELIEFS message."""
        data = {
            "type": "REQUEST_BELIEFS",
            "request_id": str(uuid4()),
            "requester_did": "did:vkb:web:test.example.com",
            "query": {
                "domain_filter": ["science"],
                "min_confidence": 0.5,
                "limit": 20,
            },
        }
        
        result = parse_message(data)
        
        assert isinstance(result, RequestBeliefsRequest)
        assert result.domain_filter == ["science"]

    def test_parse_sync_request(self):
        """Test parsing SYNC_REQUEST message."""
        now = datetime.now()
        data = {
            "type": "SYNC_REQUEST",
            "request_id": str(uuid4()),
            "since": now.isoformat(),
            "domains": ["test"],
        }
        
        result = parse_message(data)
        
        assert isinstance(result, SyncRequest)
        assert result.domains == ["test"]

    def test_parse_invalid_type(self):
        """Test parsing message with invalid type."""
        data = {
            "type": "INVALID_TYPE",
        }
        
        result = parse_message(data)
        
        assert result is None

    def test_parse_missing_type(self):
        """Test parsing message without type."""
        data = {
            "request_id": str(uuid4()),
        }
        
        result = parse_message(data)
        
        assert result is None


# =============================================================================
# MESSAGE DISPATCH TESTS
# =============================================================================


class TestHandleMessage:
    """Tests for handle_message dispatch function."""

    @pytest.mark.asyncio
    async def test_handle_auth_challenge(self):
        """Test handling AUTH_CHALLENGE message."""
        msg = AuthChallengeRequest(client_did="did:vkb:web:test.example.com")
        
        result = await handle_message(msg)
        
        assert isinstance(result, AuthChallengeResponse)

    @pytest.mark.asyncio
    async def test_handle_auth_verify_no_public_key(self, mock_get_cursor):
        """Test handling AUTH_VERIFY with unknown node."""
        mock_get_cursor.fetchone.return_value = None
        
        msg = AuthVerifyRequest(
            client_did="did:vkb:web:unknown.example.com",
            challenge="abc123",
            signature="sig",
        )
        
        result = await handle_message(msg)
        
        assert isinstance(result, ErrorMessage)
        assert result.error_code == ErrorCode.NODE_NOT_FOUND

    @pytest.mark.asyncio
    async def test_handle_message_no_sender(self, mock_get_cursor):
        """Test handling message without sender identification."""
        mock_get_cursor.fetchone.return_value = None
        
        msg = ShareBeliefRequest(beliefs=[])
        
        result = await handle_message(msg)
        
        assert isinstance(result, ErrorMessage)
        assert result.error_code == ErrorCode.AUTH_FAILED

    @pytest.mark.asyncio
    async def test_handle_share_belief_with_sender(self, mock_get_cursor):
        """Test handling SHARE_BELIEF with sender context."""
        sender_node_id = uuid4()
        mock_get_cursor.fetchone.return_value = {"id": sender_node_id}
        
        msg = ShareBeliefRequest(beliefs=[])
        
        with patch("valence.federation.trust.get_effective_trust") as mock_trust:
            mock_trust.return_value = 0.5
            
            result = await handle_message(
                msg,
                sender_did="did:vkb:web:test.example.com",
            )
        
        assert isinstance(result, ShareBeliefResponse)

    @pytest.mark.asyncio
    async def test_handle_request_beliefs_with_sender(self, mock_get_cursor):
        """Test handling REQUEST_BELIEFS with sender context."""
        sender_node_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": sender_node_id},  # First call for sender lookup
            {"total": 0},  # For counting query
        ]
        mock_get_cursor.fetchall.return_value = []
        
        msg = RequestBeliefsRequest(
            requester_did="did:vkb:web:test.example.com",
        )
        
        with patch("valence.federation.trust.get_effective_trust") as mock_trust, \
             patch("valence.server.config.get_settings") as mock_settings:
            mock_trust.return_value = 0.5
            mock_settings.return_value = MagicMock(
                federation_node_did="did:vkb:web:local",
                federation_private_key=None,
            )
            
            result = await handle_message(
                msg,
                sender_did="did:vkb:web:test.example.com",
            )
        
        assert isinstance(result, BeliefsResponse)

    @pytest.mark.asyncio
    async def test_handle_sync_request_with_sender(self, mock_get_cursor):
        """Test handling SYNC_REQUEST with sender context."""
        sender_node_id = uuid4()
        mock_get_cursor.fetchone.return_value = {"id": sender_node_id}
        mock_get_cursor.fetchall.return_value = []
        
        msg = SyncRequest()
        
        with patch("valence.federation.trust.get_effective_trust") as mock_trust, \
             patch("valence.server.config.get_settings") as mock_settings:
            mock_trust.return_value = 0.5
            mock_settings.return_value = MagicMock(
                federation_node_did="did:vkb:web:local",
                federation_private_key=None,
            )
            
            result = await handle_message(
                msg,
                sender_did="did:vkb:web:test.example.com",
            )
        
        assert isinstance(result, SyncResponse)


# =============================================================================
# TRUST ATTESTATION TESTS
# =============================================================================


class TestTrustAttestationMessages:
    """Tests for trust attestation messages."""

    def test_attestation_request_creation(self):
        """Test creating a trust attestation request."""
        msg = TrustAttestationRequest(
            attestation={"subject_did": "did:vkb:web:target"},
            issuer_signature="sig_data",
        )
        
        assert msg.type == MessageType.TRUST_ATTESTATION
        assert msg.attestation["subject_did"] == "did:vkb:web:target"

    def test_attestation_response_creation(self):
        """Test creating a trust attestation response."""
        msg = TrustAttestationResponse(
            accepted=True,
            reason=None,
        )
        
        assert msg.type == MessageType.TRUST_ATTESTATION_RESPONSE
        assert msg.accepted is True
