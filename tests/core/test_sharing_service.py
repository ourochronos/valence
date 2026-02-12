"""Tests for SharingService (#341).

Tests cover:
1. share_belief creates consent chain and share
2. share_belief raises BeliefNotFoundError
3. share_belief raises OwnershipError
4. share_belief raises DuplicateShareError
5. list_shares returns ShareInfo objects
6. revoke_share updates consent chain
7. revoke_share raises on not found / already revoked / wrong owner
8. Service is composable (no global state)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from valence.core.sharing_service import (
    BeliefNotFoundError,
    DuplicateShareError,
    OwnershipError,
    PolicyViolationError,
    RevokeResult,
    ShareInfo,
    ShareResult,
    SharingError,
    SharingService,
    get_sharing_service,
)


@pytest.fixture
def service():
    return SharingService()


@pytest.fixture
def mock_cur():
    return MagicMock()


class TestShareBelief:
    """Test share_belief service method."""

    @patch("valence.core.crypto_service.encrypt_for_sharing")
    @patch("valence.core.identity_service.sign_data")
    def test_creates_share(self, mock_sign, mock_encrypt, service, mock_cur):
        # Setup mocks
        mock_sign.return_value = MagicMock(signature=b"x" * 64)
        mock_encrypt.return_value = {"algorithm": "none", "content": "test"}

        mock_cur.fetchone.side_effect = [
            # Belief lookup
            {"id": "b1", "content": "test belief", "share_policy": None, "holder_id": None},
            # consent_chain INSERT RETURNING id
            {"id": "cc1"},
            # share INSERT RETURNING id
            {"id": "s1"},
        ]

        result = service.share_belief(
            mock_cur,
            belief_id="b1",
            recipient_did="did:valence:bob",
            intent="know_me",
            local_did="did:valence:alice",
        )

        assert isinstance(result, ShareResult)
        assert result.share_id == "s1"
        assert result.consent_chain_id == "cc1"
        assert result.recipient_did == "did:valence:bob"
        assert result.belief_content == "test belief"

    def test_raises_belief_not_found(self, service, mock_cur):
        mock_cur.fetchone.return_value = None

        with pytest.raises(BeliefNotFoundError, match="Belief not found"):
            service.share_belief(mock_cur, "bad-id", "did:bob", "know_me", "did:alice")

    def test_raises_ownership_error(self, service, mock_cur):
        mock_cur.fetchone.return_value = {
            "id": "b1", "content": "test", "share_policy": None, "holder_id": "entity-999",
        }

        with pytest.raises(OwnershipError, match="not the holder"):
            service.share_belief(
                mock_cur, "b1", "did:bob", "know_me", "did:alice",
                local_entity_id="entity-123",
            )

    def test_ownership_check_skipped_for_null_holder(self, service, mock_cur):
        """NULL holder_id = permissive (backward compat)."""
        mock_cur.fetchone.side_effect = [
            {"id": "b1", "content": "test", "share_policy": None, "holder_id": None},
            {"id": "cc1"},
            {"id": "s1"},
        ]

        with patch("valence.core.identity_service.sign_data") as mock_sign, \
             patch("valence.core.crypto_service.encrypt_for_sharing") as mock_enc:
            mock_sign.return_value = MagicMock(signature=b"x" * 64)
            mock_enc.return_value = {"algorithm": "none", "content": "test"}

            result = service.share_belief(mock_cur, "b1", "did:bob", "know_me", "did:alice")
            assert result.share_id == "s1"

    @patch("valence.core.crypto_service.encrypt_for_sharing")
    @patch("valence.core.identity_service.sign_data")
    def test_raises_duplicate_share(self, mock_sign, mock_encrypt, service, mock_cur):
        mock_sign.return_value = MagicMock(signature=b"x" * 64)
        mock_encrypt.return_value = {"algorithm": "none", "content": "test"}

        mock_cur.fetchone.side_effect = [
            {"id": "b1", "content": "test", "share_policy": None, "holder_id": None},
            {"id": "cc1"},
        ]

        # Simulate unique violation on share INSERT (3rd execute call)
        call_count = 0
        original_execute = mock_cur.execute

        def execute_with_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # The share INSERT
                raise Exception("duplicate key value violates unique constraint")
            return original_execute(*args, **kwargs)

        mock_cur.execute = MagicMock(side_effect=execute_with_error)

        with pytest.raises(DuplicateShareError, match="already shared"):
            service.share_belief(mock_cur, "b1", "did:bob", "know_me", "did:alice")


class TestListShares:
    """Test list_shares service method."""

    def test_returns_share_info_list(self, service, mock_cur):
        mock_cur.fetchall.return_value = [
            {
                "share_id": "s1", "belief_id": "b1", "belief_content": "test",
                "recipient_did": "did:bob", "origin_sharer": "did:alice",
                "intent": "know_me", "created_at": datetime(2024, 1, 1),
                "access_count": 3, "revoked": False, "revoked_at": None,
                "revocation_reason": None,
            },
        ]

        shares = service.list_shares(mock_cur, local_did="did:alice", direction="outgoing")

        assert len(shares) == 1
        assert isinstance(shares[0], ShareInfo)
        assert shares[0].share_id == "s1"
        assert shares[0].intent == "know_me"

    def test_empty_list(self, service, mock_cur):
        mock_cur.fetchall.return_value = []

        shares = service.list_shares(mock_cur, local_did="did:alice")

        assert shares == []

    def test_incoming_direction(self, service, mock_cur):
        mock_cur.fetchall.return_value = []
        service.list_shares(mock_cur, local_did="did:bob", direction="incoming")

        sql = mock_cur.execute.call_args[0][0]
        assert "s.recipient_did = %s" in sql


class TestRevokeShare:
    """Test revoke_share service method."""

    def test_revokes_share(self, service, mock_cur):
        mock_cur.fetchone.return_value = {
            "id": "s1", "consent_chain_id": "cc1",
            "origin_sharer": "did:alice", "revoked": False,
        }

        result = service.revoke_share(mock_cur, share_id="s1", local_did="did:alice", reason="no longer needed")

        assert isinstance(result, RevokeResult)
        assert result.share_id == "s1"
        assert result.consent_chain_id == "cc1"
        assert result.reason == "no longer needed"

    def test_raises_not_found(self, service, mock_cur):
        mock_cur.fetchone.return_value = None

        with pytest.raises(SharingError, match="Share not found"):
            service.revoke_share(mock_cur, "bad-id", "did:alice")

    def test_raises_already_revoked(self, service, mock_cur):
        mock_cur.fetchone.return_value = {
            "id": "s1", "consent_chain_id": "cc1",
            "origin_sharer": "did:alice", "revoked": True,
        }

        with pytest.raises(SharingError, match="already revoked"):
            service.revoke_share(mock_cur, "s1", "did:alice")

    def test_raises_wrong_owner(self, service, mock_cur):
        mock_cur.fetchone.return_value = {
            "id": "s1", "consent_chain_id": "cc1",
            "origin_sharer": "did:alice", "revoked": False,
        }

        with pytest.raises(SharingError, match="did not create"):
            service.revoke_share(mock_cur, "s1", "did:eve")


class TestGetSharingService:
    """Test singleton accessor."""

    def test_returns_same_instance(self):
        s1 = get_sharing_service()
        s2 = get_sharing_service()
        assert s1 is s2

    def test_returns_sharing_service(self):
        assert isinstance(get_sharing_service(), SharingService)
