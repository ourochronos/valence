"""Tests for receive and reshare service (#344).

Tests cover:
1. validate_consent_chain valid
2. validate_consent_chain missing field
3. validate_consent_chain revoked
4. validate_consent_chain max hops exceeded
5. receive_share accepts valid
6. receive_share rejects invalid chain
7. receive_share creates notification
8. reshare_belief success
9. reshare_belief share not found
10. reshare_belief know_me blocked
11. reshare_belief max hops exceeded
12. reshare_belief duplicate rejected
13. get_notifications
14. mark_notification_read
15. IncomingShare serialization
16. ReshareResult serialization
17. ShareNotification serialization
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from valence.core.receive_service import (
    IncomingShare,
    ReshareResult,
    ShareNotification,
    get_notifications,
    mark_notification_read,
    receive_share,
    reshare_belief,
    validate_consent_chain,
)


@pytest.fixture
def mock_cur():
    return MagicMock()


class TestValidateConsentChain:
    """Test consent chain validation."""

    def test_valid_chain(self):
        chain = {
            "origin_sharer": "did:alice",
            "chain_hash": "abc123",
            "hop_count": 1,
            "max_hops": 3,
        }
        valid, error = validate_consent_chain(chain)
        assert valid is True
        assert error == ""

    def test_missing_origin(self):
        chain = {"chain_hash": "abc"}
        valid, error = validate_consent_chain(chain)
        assert valid is False
        assert "origin_sharer" in error

    def test_missing_chain_hash(self):
        chain = {"origin_sharer": "did:alice"}
        valid, error = validate_consent_chain(chain)
        assert valid is False
        assert "chain_hash" in error

    def test_revoked_chain(self):
        chain = {
            "origin_sharer": "did:alice",
            "chain_hash": "abc",
            "revoked": True,
        }
        valid, error = validate_consent_chain(chain)
        assert valid is False
        assert "revoked" in error.lower()

    def test_max_hops_exceeded(self):
        chain = {
            "origin_sharer": "did:alice",
            "chain_hash": "abc",
            "hop_count": 3,
            "max_hops": 3,
        }
        valid, error = validate_consent_chain(chain)
        assert valid is False
        assert "hops" in error.lower()

    def test_no_max_hops_unlimited(self):
        chain = {
            "origin_sharer": "did:alice",
            "chain_hash": "abc",
            "hop_count": 100,
        }
        valid, error = validate_consent_chain(chain)
        assert valid is True


class TestReceiveShare:
    """Test incoming share processing."""

    def test_accepts_valid(self, mock_cur):
        chain = {
            "origin_sharer": "did:alice",
            "chain_hash": "abc",
            "belief_id": "b1",
        }
        result = receive_share(
            mock_cur,
            sender_did="did:alice",
            belief_content="Test belief",
            confidence={"overall": 0.8},
            intent="learn_from_me",
            consent_chain=chain,
            recipient_did="did:bob",
        )
        assert isinstance(result, IncomingShare)
        assert result.status == "accepted"
        assert result.sender_did == "did:alice"
        assert result.content == "Test belief"

    def test_rejects_invalid_chain(self, mock_cur):
        chain = {"revoked": True, "origin_sharer": "x", "chain_hash": "y"}
        result = receive_share(
            mock_cur,
            sender_did="did:alice",
            belief_content="Bad",
            confidence={},
            intent="learn_from_me",
            consent_chain=chain,
            recipient_did="did:bob",
        )
        assert isinstance(result, str)
        assert "revoked" in result.lower()

    def test_persists_to_db(self, mock_cur):
        chain = {"origin_sharer": "did:alice", "chain_hash": "abc"}
        receive_share(
            mock_cur, "did:alice", "Content", {"overall": 0.5},
            "learn_from_me", chain, "did:bob",
        )
        sql_calls = [c[0][0] for c in mock_cur.execute.call_args_list]
        assert any("INSERT INTO received_shares" in s for s in sql_calls)
        assert any("INSERT INTO share_notifications" in s for s in sql_calls)


class TestReshareBeliefBasic:
    """Test reshare validation."""

    def test_share_not_found(self, mock_cur):
        mock_cur.fetchone.return_value = None
        result = reshare_belief(mock_cur, "missing", "did:alice", "did:carol")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_know_me_blocked(self, mock_cur):
        mock_cur.fetchone.return_value = {
            "id": "s1", "belief_id": "b1", "intent": "know_me",
            "status": "accepted", "consent_chain": json.dumps({"origin_sharer": "did:alice", "chain_hash": "abc"}),
            "content": "test", "confidence": "{}",
        }
        result = reshare_belief(mock_cur, "s1", "did:bob", "did:carol")
        assert result.success is False
        assert "know_me" in result.error.lower()

    def test_not_accepted_status(self, mock_cur):
        mock_cur.fetchone.return_value = {
            "id": "s1", "status": "rejected",
        }
        result = reshare_belief(mock_cur, "s1", "did:bob", "did:carol")
        assert result.success is False
        assert "accepted" in result.error.lower()


class TestReshareBeliefSuccess:
    """Test successful reshare flow."""

    def test_reshare_success(self, mock_cur):
        chain = {
            "origin_sharer": "did:alice",
            "chain_hash": "original_hash",
            "hop_count": 0,
            "max_hops": 3,
        }
        mock_cur.fetchone.side_effect = [
            {  # original share
                "id": "s1", "belief_id": "b1", "intent": "learn_from_me",
                "status": "accepted", "consent_chain": json.dumps(chain),
                "content": "Test content", "confidence": '{"overall": 0.8}',
            },
            None,  # not already shared to recipient
        ]
        result = reshare_belief(mock_cur, "s1", "did:bob", "did:carol")
        assert result.success is True
        assert result.share_id is not None
        assert result.new_chain_hash is not None

    def test_max_hops_exceeded(self, mock_cur):
        chain = {
            "origin_sharer": "did:alice",
            "chain_hash": "abc",
            "hop_count": 2,
            "max_hops": 3,
        }
        mock_cur.fetchone.return_value = {
            "id": "s1", "belief_id": "b1", "intent": "learn_from_me",
            "status": "accepted", "consent_chain": json.dumps(chain),
            "content": "test", "confidence": "{}",
        }
        result = reshare_belief(mock_cur, "s1", "did:bob", "did:carol")
        assert result.success is False
        assert "max hops" in result.error.lower()

    def test_duplicate_rejected(self, mock_cur):
        chain = {
            "origin_sharer": "did:alice",
            "chain_hash": "abc",
            "hop_count": 0,
            "max_hops": 3,
        }
        mock_cur.fetchone.side_effect = [
            {
                "id": "s1", "belief_id": "b1", "intent": "learn_from_me",
                "status": "accepted", "consent_chain": json.dumps(chain),
                "content": "test", "confidence": "{}",
            },
            {"exists": True},  # already shared
        ]
        result = reshare_belief(mock_cur, "s1", "did:bob", "did:carol")
        assert result.success is False
        assert "already shared" in result.error.lower()


class TestGetNotifications:
    """Test notification retrieval."""

    def test_returns_notifications(self, mock_cur):
        now = datetime.now(timezone.utc)
        mock_cur.fetchall.return_value = [
            {
                "id": "n1", "share_id": "s1", "recipient_did": "did:bob",
                "sender_did": "did:alice", "belief_id": "b1",
                "intent": "learn_from_me", "read": False, "created_at": now,
            },
        ]
        notifications = get_notifications(mock_cur, "did:bob")
        assert len(notifications) == 1
        assert notifications[0].sender_did == "did:alice"
        assert notifications[0].read is False

    def test_queries_with_unread_filter(self, mock_cur):
        mock_cur.fetchall.return_value = []
        get_notifications(mock_cur, "did:bob", unread_only=True)
        sql = mock_cur.execute.call_args[0][0]
        assert "read = FALSE" in sql

    def test_queries_all(self, mock_cur):
        mock_cur.fetchall.return_value = []
        get_notifications(mock_cur, "did:bob", unread_only=False)
        sql = mock_cur.execute.call_args[0][0]
        assert "read = FALSE" not in sql


class TestMarkNotificationRead:
    """Test marking notifications as read."""

    def test_marks_read(self, mock_cur):
        mock_cur.rowcount = 1
        result = mark_notification_read(mock_cur, "n1")
        assert result is True

    def test_not_found(self, mock_cur):
        mock_cur.rowcount = 0
        result = mark_notification_read(mock_cur, "missing")
        assert result is False


class TestSerialization:
    """Test dataclass serialization."""

    def test_incoming_share(self):
        share = IncomingShare(
            id="s1", belief_id="b1", sender_did="did:alice",
            content="Test", confidence={"overall": 0.8},
            intent="learn_from_me",
            consent_chain={"origin": "did:alice"},
            status="accepted",
        )
        d = share.to_dict()
        assert d["sender_did"] == "did:alice"
        assert d["status"] == "accepted"

    def test_reshare_result_success(self):
        r = ReshareResult(success=True, share_id="s2", new_chain_hash="abc")
        d = r.to_dict()
        assert d["success"] is True
        assert d["share_id"] == "s2"

    def test_reshare_result_failure(self):
        r = ReshareResult(success=False, error="not allowed")
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "not allowed"

    def test_notification(self):
        n = ShareNotification(
            id="n1", share_id="s1", recipient_did="did:bob",
            sender_did="did:alice", belief_id="b1",
            intent="learn_from_me",
        )
        d = n.to_dict()
        assert d["read"] is False
        assert d["intent"] == "learn_from_me"
