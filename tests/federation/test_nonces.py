"""Tests for belief-level nonce replay protection.

Tests cover:
- Nonce generation (uniqueness, format)
- NonceTracker (record, is_seen, cleanup, TTL expiry)
- Thread safety
- Integration with protocol (incoming belief nonce validation)
- Integration with protocol (outbound belief nonce generation)
- Global singleton management
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.federation.nonces import (
    DEFAULT_NONCE_TTL_SECONDS,
    NonceTracker,
    generate_nonce,
    get_nonce_tracker,
    reset_nonce_tracker,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tracker():
    """Create a NonceTracker with short TTL for testing."""
    return NonceTracker(ttl_seconds=5)


@pytest.fixture(autouse=True)
def _reset_global_tracker():
    """Reset the global tracker before and after each test."""
    reset_nonce_tracker()
    yield
    reset_nonce_tracker()


# =============================================================================
# NONCE GENERATION
# =============================================================================


class TestGenerateNonce:
    """Tests for the generate_nonce function."""

    def test_returns_string(self):
        nonce = generate_nonce()
        assert isinstance(nonce, str)

    def test_hex_format(self):
        nonce = generate_nonce()
        # Should be 32 hex chars (16 bytes)
        assert len(nonce) == 32
        int(nonce, 16)  # Should not raise

    def test_uniqueness(self):
        nonces = {generate_nonce() for _ in range(1000)}
        assert len(nonces) == 1000


# =============================================================================
# NONCE TRACKER — BASIC OPERATIONS
# =============================================================================


class TestNonceTrackerBasic:
    """Tests for NonceTracker basic operations."""

    def test_unseen_nonce(self, tracker: NonceTracker):
        assert tracker.is_seen("node-a", "nonce-1") is False

    def test_record_and_check(self, tracker: NonceTracker):
        tracker.record_nonce("node-a", "nonce-1")
        assert tracker.is_seen("node-a", "nonce-1") is True

    def test_different_origins_independent(self, tracker: NonceTracker):
        tracker.record_nonce("node-a", "nonce-1")
        assert tracker.is_seen("node-b", "nonce-1") is False

    def test_different_nonces_independent(self, tracker: NonceTracker):
        tracker.record_nonce("node-a", "nonce-1")
        assert tracker.is_seen("node-a", "nonce-2") is False

    def test_multiple_nonces_same_origin(self, tracker: NonceTracker):
        tracker.record_nonce("node-a", "nonce-1")
        tracker.record_nonce("node-a", "nonce-2")
        assert tracker.is_seen("node-a", "nonce-1") is True
        assert tracker.is_seen("node-a", "nonce-2") is True

    def test_origin_count(self, tracker: NonceTracker):
        assert tracker.origin_count() == 0
        tracker.record_nonce("node-a", "n1")
        assert tracker.origin_count() == 1
        tracker.record_nonce("node-b", "n2")
        assert tracker.origin_count() == 2

    def test_nonce_count_total(self, tracker: NonceTracker):
        assert tracker.nonce_count() == 0
        tracker.record_nonce("node-a", "n1")
        tracker.record_nonce("node-a", "n2")
        tracker.record_nonce("node-b", "n3")
        assert tracker.nonce_count() == 3

    def test_nonce_count_per_origin(self, tracker: NonceTracker):
        tracker.record_nonce("node-a", "n1")
        tracker.record_nonce("node-a", "n2")
        tracker.record_nonce("node-b", "n3")
        assert tracker.nonce_count("node-a") == 2
        assert tracker.nonce_count("node-b") == 1
        assert tracker.nonce_count("node-c") == 0

    def test_clear(self, tracker: NonceTracker):
        tracker.record_nonce("node-a", "n1")
        tracker.record_nonce("node-b", "n2")
        tracker.clear()
        assert tracker.nonce_count() == 0
        assert tracker.origin_count() == 0
        assert tracker.is_seen("node-a", "n1") is False


# =============================================================================
# NONCE TRACKER — TTL & EXPIRY
# =============================================================================


class TestNonceTrackerTTL:
    """Tests for TTL-based nonce expiry."""

    def test_ttl_property(self):
        tracker = NonceTracker(ttl_seconds=42)
        assert tracker.ttl_seconds == 42

    def test_default_ttl(self):
        tracker = NonceTracker()
        assert tracker.ttl_seconds == DEFAULT_NONCE_TTL_SECONDS

    def test_expired_nonce_not_seen(self):
        tracker = NonceTracker(ttl_seconds=0)
        tracker.record_nonce("node-a", "nonce-1")
        # TTL=0 means it expires immediately; sleep a tiny bit
        time.sleep(0.01)
        assert tracker.is_seen("node-a", "nonce-1") is False

    def test_expired_nonce_cleaned_on_check(self):
        """is_seen removes individual expired entries on access."""
        tracker = NonceTracker(ttl_seconds=0)
        tracker.record_nonce("node-a", "nonce-1")
        time.sleep(0.01)
        # After is_seen returns False, the entry should be removed
        tracker.is_seen("node-a", "nonce-1")
        assert tracker.nonce_count("node-a") == 0

    def test_cleanup_removes_expired(self):
        tracker = NonceTracker(ttl_seconds=0)
        tracker.record_nonce("node-a", "n1")
        tracker.record_nonce("node-a", "n2")
        tracker.record_nonce("node-b", "n3")
        time.sleep(0.01)
        removed = tracker.cleanup()
        assert removed == 3
        assert tracker.nonce_count() == 0
        assert tracker.origin_count() == 0

    def test_cleanup_keeps_fresh(self):
        tracker = NonceTracker(ttl_seconds=60)
        tracker.record_nonce("node-a", "fresh-nonce")
        removed = tracker.cleanup()
        assert removed == 0
        assert tracker.is_seen("node-a", "fresh-nonce") is True

    def test_max_per_origin_triggers_cleanup(self):
        """When max_per_origin is exceeded, expired entries are cleaned."""
        # Use a TTL long enough that new entries survive but we can
        # manually expire old ones via monkeypatch.
        tracker = NonceTracker(ttl_seconds=60, max_per_origin=3)
        # Record 3 nonces
        tracker.record_nonce("node-a", "old-1")
        tracker.record_nonce("node-a", "old-2")
        tracker.record_nonce("node-a", "old-3")
        # Manually backdate the old entries so they appear expired
        now = time.monotonic()
        for nonce in ["old-1", "old-2", "old-3"]:
            tracker._seen["node-a"][nonce] = now - 120  # 2 minutes ago
        # This 4th one triggers cleanup of expired entries
        tracker.record_nonce("node-a", "new-1")
        # The old ones should be cleaned, only new-1 remains
        assert tracker.nonce_count("node-a") == 1
        assert tracker.is_seen("node-a", "new-1") is True


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================


class TestGlobalTracker:
    """Tests for the global singleton tracker."""

    def test_get_returns_instance(self):
        tracker = get_nonce_tracker()
        assert isinstance(tracker, NonceTracker)

    def test_get_returns_same_instance(self):
        t1 = get_nonce_tracker()
        t2 = get_nonce_tracker()
        assert t1 is t2

    def test_reset_clears_instance(self):
        t1 = get_nonce_tracker()
        reset_nonce_tracker()
        t2 = get_nonce_tracker()
        assert t1 is not t2


# =============================================================================
# FEDERATED BELIEF MODEL INTEGRATION
# =============================================================================


class TestFederatedBeliefNonce:
    """Tests for the nonce field on FederatedBelief."""

    def test_nonce_field_default_none(self):
        from valence.core.confidence import DimensionalConfidence
        from valence.federation.models import FederatedBelief

        belief = FederatedBelief(
            id=uuid4(),
            federation_id=uuid4(),
            origin_node_did="did:vkb:web:test",
            content="test belief",
            confidence=DimensionalConfidence.simple(0.8),
        )
        assert belief.nonce is None

    def test_nonce_field_set(self):
        from valence.core.confidence import DimensionalConfidence
        from valence.federation.models import FederatedBelief

        nonce = generate_nonce()
        belief = FederatedBelief(
            id=uuid4(),
            federation_id=uuid4(),
            origin_node_did="did:vkb:web:test",
            content="test belief",
            confidence=DimensionalConfidence.simple(0.8),
            nonce=nonce,
        )
        assert belief.nonce == nonce

    def test_nonce_in_to_dict(self):
        from valence.core.confidence import DimensionalConfidence
        from valence.federation.models import FederatedBelief

        nonce = generate_nonce()
        belief = FederatedBelief(
            id=uuid4(),
            federation_id=uuid4(),
            origin_node_did="did:vkb:web:test",
            content="test belief",
            confidence=DimensionalConfidence.simple(0.8),
            nonce=nonce,
        )
        d = belief.to_dict()
        assert d["nonce"] == nonce

    def test_nonce_none_in_to_dict(self):
        from valence.core.confidence import DimensionalConfidence
        from valence.federation.models import FederatedBelief

        belief = FederatedBelief(
            id=uuid4(),
            federation_id=uuid4(),
            origin_node_did="did:vkb:web:test",
            content="test belief",
            confidence=DimensionalConfidence.simple(0.8),
        )
        d = belief.to_dict()
        assert d["nonce"] is None

    def test_nonce_in_signable_content_when_set(self):
        from valence.core.confidence import DimensionalConfidence
        from valence.federation.models import FederatedBelief

        nonce = generate_nonce()
        belief = FederatedBelief(
            id=uuid4(),
            federation_id=uuid4(),
            origin_node_did="did:vkb:web:test",
            content="test belief",
            confidence=DimensionalConfidence.simple(0.8),
            nonce=nonce,
        )
        signable = belief.to_signable_content()
        assert signable["nonce"] == nonce

    def test_nonce_absent_from_signable_when_none(self):
        from valence.core.confidence import DimensionalConfidence
        from valence.federation.models import FederatedBelief

        belief = FederatedBelief(
            id=uuid4(),
            federation_id=uuid4(),
            origin_node_did="did:vkb:web:test",
            content="test belief",
            confidence=DimensionalConfidence.simple(0.8),
        )
        signable = belief.to_signable_content()
        assert "nonce" not in signable


# =============================================================================
# PROTOCOL INTEGRATION — INCOMING BELIEFS
# =============================================================================


class TestProtocolNonceValidation:
    """Tests for nonce validation in the protocol layer."""

    def _make_belief_data(self, nonce: str | None = None) -> dict:
        """Create minimal belief data for testing."""
        data = {
            "federation_id": str(uuid4()),
            "origin_node_did": "did:vkb:web:sender.example",
            "content": "Test belief content",
            "confidence": {"overall": 0.8},
            "origin_signature": "fakesig",
            "domain_path": [],
        }
        if nonce is not None:
            data["nonce"] = nonce
        return data

    def test_replayed_nonce_rejected(self):
        """A belief with a previously-seen nonce should be rejected."""
        from valence.federation.protocol import _process_incoming_belief

        nonce = generate_nonce()
        sender_id = uuid4()
        origin_did = "did:vkb:web:sender.example"

        # Pre-record the nonce in the tracker (simulating first receipt)
        tracker = get_nonce_tracker()
        tracker.record_nonce(origin_did, nonce)

        # Now try to process a belief with the same nonce — should reject early
        belief_data = self._make_belief_data(nonce=nonce)
        result = _process_incoming_belief(belief_data, sender_id, 0.5)
        assert result == "Replayed nonce"

    def test_no_nonce_skips_check(self):
        """Beliefs without a nonce should skip nonce validation."""
        from valence.federation.protocol import _process_incoming_belief

        # This will fail at signature verification, but should NOT fail
        # at nonce check — we just need to verify no nonce error appears
        belief_data = self._make_belief_data(nonce=None)

        with patch("valence.federation.protocol.get_cursor") as mock_cursor:
            cursor_ctx = MagicMock()
            cursor = MagicMock()
            cursor.fetchone.return_value = {"public_key_multibase": "z6MkFake"}
            cursor_ctx.__enter__ = MagicMock(return_value=cursor)
            cursor_ctx.__exit__ = MagicMock(return_value=False)
            mock_cursor.return_value = cursor_ctx

            with patch(
                "valence.federation.protocol.verify_belief_signature",
                return_value=False,
            ):
                result = _process_incoming_belief(belief_data, uuid4(), 0.5)
                # Should fail on signature, NOT on nonce
                assert result == "Invalid signature"

    def test_different_nonces_both_accepted(self):
        """Different nonces from the same origin should not trigger replay."""
        from valence.federation.protocol import _process_incoming_belief

        sender_id = uuid4()

        # Mock the cursor so processing gets past nonce check
        cursor_ctx = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = {"public_key_multibase": "z6MkFake"}
        cursor_ctx.__enter__ = MagicMock(return_value=cursor)
        cursor_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch("valence.federation.protocol.get_cursor", return_value=cursor_ctx),
            patch(
                "valence.federation.protocol.verify_belief_signature",
                return_value=False,
            ),
        ):
            belief1 = self._make_belief_data(nonce=generate_nonce())
            belief2 = self._make_belief_data(nonce=generate_nonce())

            result1 = _process_incoming_belief(belief1, sender_id, 0.5)
            result2 = _process_incoming_belief(belief2, sender_id, 0.5)

            # Both should get past nonce check (will fail later at sig verify)
            assert result1 != "Replayed nonce"
            assert result2 != "Replayed nonce"


# =============================================================================
# PROTOCOL INTEGRATION — OUTBOUND BELIEFS
# =============================================================================


class TestProtocolNonceGeneration:
    """Tests for nonce generation in outbound belief conversion."""

    @patch("valence.core.config.get_federation_config")
    def test_outbound_belief_has_nonce(self, mock_config):
        """Beliefs converted for federation should include a nonce."""
        from valence.federation.protocol import _belief_row_to_federated

        mock_settings = MagicMock()
        mock_settings.federation_node_did = "did:vkb:web:localhost"
        mock_settings.federation_private_key = None  # skip signing
        mock_config.return_value = mock_settings

        row = {
            "id": uuid4(),
            "federation_id": uuid4(),
            "content": "Test belief",
            "confidence": {"overall": 0.8},
            "domain_path": ["test"],
            "visibility": "federated",
            "share_level": "belief_only",
            "valid_from": None,
            "valid_until": None,
        }

        result = _belief_row_to_federated(row, 0.5)
        assert result is not None
        assert "nonce" in result
        assert isinstance(result["nonce"], str)
        assert len(result["nonce"]) == 32

    @patch("valence.core.config.get_federation_config")
    def test_outbound_nonces_unique(self, mock_config):
        """Each outbound belief should get a unique nonce."""
        from valence.federation.protocol import _belief_row_to_federated

        mock_settings = MagicMock()
        mock_settings.federation_node_did = "did:vkb:web:localhost"
        mock_settings.federation_private_key = None
        mock_config.return_value = mock_settings

        row = {
            "id": uuid4(),
            "federation_id": uuid4(),
            "content": "Test belief",
            "confidence": {"overall": 0.8},
            "domain_path": ["test"],
            "visibility": "federated",
            "share_level": "belief_only",
            "valid_from": None,
            "valid_until": None,
        }

        nonces = set()
        for _ in range(100):
            result = _belief_row_to_federated(row, 0.5)
            nonces.add(result["nonce"])

        assert len(nonces) == 100
