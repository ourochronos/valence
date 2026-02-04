"""Tests for user data deletion (Issue #25).

Verifies:
- Tombstone creation
- Cryptographic erasure
- GDPR Article 17 compliance
- Deletion verification reports
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

from valence.compliance.deletion import (
    DeletionReason,
    Tombstone,
    DeletionResult,
    create_tombstone,
    perform_cryptographic_erasure,
    delete_user_data,
    get_deletion_verification,
    _hash_user_id,
)


class TestDeletionReason:
    """Test deletion reason enum."""

    def test_all_reasons_exist(self):
        """Verify all expected deletion reasons."""
        expected = {
            "user_request",
            "consent_withdrawal",
            "legal_order",
            "policy_violation",
            "data_accuracy",
            "security_incident",
        }
        actual = {r.value for r in DeletionReason}
        assert actual == expected

    def test_gdpr_article_17_reason(self):
        """USER_REQUEST corresponds to GDPR Article 17."""
        reason = DeletionReason.USER_REQUEST
        assert reason.value == "user_request"


class TestTombstone:
    """Test tombstone data structure."""

    def test_tombstone_creation(self):
        """Should create tombstone with required fields."""
        tombstone = Tombstone(
            id=uuid4(),
            target_type="belief",
            target_id=uuid4(),
            created_at=MagicMock(),
            created_by="user_hash",
            reason=DeletionReason.USER_REQUEST,
        )
        
        assert tombstone.target_type == "belief"
        assert tombstone.reason == DeletionReason.USER_REQUEST
        assert not tombstone.encryption_key_revoked

    def test_tombstone_to_dict(self):
        """Should serialize tombstone to dict."""
        tombstone_id = uuid4()
        target_id = uuid4()
        
        tombstone = Tombstone(
            id=tombstone_id,
            target_type="user",
            target_id=target_id,
            created_at=MagicMock(isoformat=lambda: "2024-01-01T00:00:00"),
            created_by="hashed_user",
            reason=DeletionReason.CONSENT_WITHDRAWAL,
            legal_basis="GDPR Article 7",
        )
        
        data = tombstone.to_dict()
        
        assert data["id"] == str(tombstone_id)
        assert data["target_type"] == "user"
        assert data["target_id"] == str(target_id)
        assert data["reason"] == "consent_withdrawal"
        assert data["legal_basis"] == "GDPR Article 7"


class TestUserIdHashing:
    """Test user ID privacy hashing."""

    def test_hash_consistency(self):
        """Same user ID should produce same hash."""
        user_id = "test-user-123"
        hash1 = _hash_user_id(user_id)
        hash2 = _hash_user_id(user_id)
        
        assert hash1 == hash2

    def test_hash_different_users(self):
        """Different users should have different hashes."""
        hash1 = _hash_user_id("user-1")
        hash2 = _hash_user_id("user-2")
        
        assert hash1 != hash2

    def test_hash_length(self):
        """Hash should be truncated to 32 characters."""
        hash_val = _hash_user_id("any-user")
        assert len(hash_val) == 32


class TestDeletionResult:
    """Test deletion result structure."""

    def test_success_result(self):
        """Should represent successful deletion."""
        result = DeletionResult(
            success=True,
            tombstone_id=uuid4(),
            beliefs_deleted=5,
            sessions_deleted=2,
        )
        
        assert result.success
        assert result.beliefs_deleted == 5
        assert result.error is None

    def test_failure_result(self):
        """Should represent failed deletion."""
        result = DeletionResult(
            success=False,
            error="Database connection failed",
        )
        
        assert not result.success
        assert result.error == "Database connection failed"

    def test_to_dict(self):
        """Should serialize result to dict."""
        tombstone_id = uuid4()
        result = DeletionResult(
            success=True,
            tombstone_id=tombstone_id,
            beliefs_deleted=3,
            entities_anonymized=1,
            sessions_deleted=2,
            exchanges_deleted=10,
            patterns_deleted=0,
        )
        
        data = result.to_dict()
        
        assert data["success"]
        assert data["tombstone_id"] == str(tombstone_id)
        assert data["deleted_counts"]["beliefs"] == 3
        assert data["deleted_counts"]["entities_anonymized"] == 1
        assert data["deleted_counts"]["sessions"] == 2
        assert data["error"] is None


class TestTombstoneCreation:
    """Test tombstone creation with database (requires test DB)."""

    @pytest.fixture
    def mock_cursor(self):
        """Mock database cursor."""
        with patch("valence.compliance.deletion.get_cursor") as mock:
            cursor = MagicMock()
            mock.return_value.__enter__ = MagicMock(return_value=cursor)
            mock.return_value.__exit__ = MagicMock(return_value=False)
            yield cursor

    def test_create_tombstone_stores_record(self, mock_cursor):
        """Should store tombstone in database."""
        target_id = uuid4()
        
        tombstone = create_tombstone(
            target_type="belief",
            target_id=target_id,
            created_by="test-user",
            reason=DeletionReason.USER_REQUEST,
            legal_basis="GDPR Article 17",
        )
        
        assert tombstone.target_type == "belief"
        assert tombstone.target_id == target_id
        assert tombstone.reason == DeletionReason.USER_REQUEST
        
        # Verify database call
        mock_cursor.execute.assert_called_once()

    def test_create_tombstone_hashes_user(self, mock_cursor):
        """Should hash user ID for privacy."""
        tombstone = create_tombstone(
            target_type="user",
            target_id=uuid4(),
            created_by="sensitive-user-id",
            reason=DeletionReason.USER_REQUEST,
        )
        
        # User ID should be hashed, not stored raw
        assert tombstone.created_by != "sensitive-user-id"
        assert len(tombstone.created_by) == 32


class TestCryptographicErasure:
    """Test cryptographic erasure functionality."""

    @pytest.fixture
    def mock_cursor(self):
        """Mock database cursor."""
        with patch("valence.compliance.deletion.get_cursor") as mock:
            cursor = MagicMock()
            mock.return_value.__enter__ = MagicMock(return_value=cursor)
            mock.return_value.__exit__ = MagicMock(return_value=False)
            yield cursor

    def test_marks_key_revoked(self, mock_cursor):
        """Should mark encryption key as revoked."""
        tombstone_id = uuid4()
        
        # Mock return value for tombstone lookup
        mock_cursor.fetchone.return_value = {
            "target_type": "belief",
            "target_id": uuid4(),
        }
        
        result = perform_cryptographic_erasure(tombstone_id)
        
        assert result
        # Should have updated tombstones and beliefs tables
        assert mock_cursor.execute.call_count >= 2

    def test_handles_missing_tombstone(self, mock_cursor):
        """Should handle missing tombstone gracefully."""
        mock_cursor.fetchone.return_value = None
        
        result = perform_cryptographic_erasure(uuid4())
        
        assert not result


class TestUserDataDeletion:
    """Test full user data deletion flow."""

    @pytest.fixture
    def mock_all_db(self):
        """Mock all database operations."""
        with patch("valence.compliance.deletion.get_cursor") as mock_cursor, \
             patch("valence.compliance.deletion.create_tombstone") as mock_tombstone, \
             patch("valence.compliance.deletion.perform_cryptographic_erasure") as mock_erasure, \
             patch("valence.compliance.deletion._start_tombstone_propagation") as mock_propagate:
            
            cursor = MagicMock()
            mock_cursor.return_value.__enter__ = MagicMock(return_value=cursor)
            mock_cursor.return_value.__exit__ = MagicMock(return_value=False)
            
            # Setup mock tombstone
            tombstone = MagicMock()
            tombstone.id = uuid4()
            mock_tombstone.return_value = tombstone
            
            # Setup cursor responses
            cursor.fetchall.return_value = []  # No beliefs/sessions found
            cursor.rowcount = 0
            
            yield {
                "cursor": cursor,
                "tombstone": mock_tombstone,
                "erasure": mock_erasure,
                "propagate": mock_propagate,
            }

    def test_creates_master_tombstone(self, mock_all_db):
        """Should create master tombstone for deletion."""
        result = delete_user_data("test-user")
        
        mock_all_db["tombstone"].assert_called()
        assert result.tombstone_id is not None

    def test_supports_all_deletion_reasons(self, mock_all_db):
        """Should accept all valid deletion reasons."""
        for reason in DeletionReason:
            result = delete_user_data("test-user", reason=reason)
            assert result.success

    def test_triggers_cryptographic_erasure(self, mock_all_db):
        """Should trigger cryptographic erasure."""
        delete_user_data("test-user")
        
        mock_all_db["erasure"].assert_called_once()

    def test_starts_federation_propagation(self, mock_all_db):
        """Should start tombstone propagation for federation."""
        delete_user_data("test-user")
        
        mock_all_db["propagate"].assert_called_once()

    def test_returns_deletion_counts(self, mock_all_db):
        """Should return counts of deleted items."""
        result = delete_user_data("test-user")
        
        assert result.success
        assert isinstance(result.beliefs_deleted, int)
        assert isinstance(result.sessions_deleted, int)


class TestDeletionVerification:
    """Test deletion verification reports."""

    @pytest.fixture
    def mock_cursor(self):
        """Mock database cursor."""
        with patch("valence.compliance.deletion.get_cursor") as mock:
            cursor = MagicMock()
            mock.return_value.__enter__ = MagicMock(return_value=cursor)
            mock.return_value.__exit__ = MagicMock(return_value=False)
            yield cursor

    def test_returns_verification_report(self, mock_cursor):
        """Should return verification report."""
        tombstone_id = uuid4()
        
        # Mock tombstone data
        from datetime import datetime
        mock_cursor.fetchone.return_value = {
            "id": tombstone_id,
            "target_type": "user",
            "target_id": uuid4(),
            "created_at": datetime.now(),
            "created_by": "hashed",
            "reason": "user_request",
            "legal_basis": "GDPR Article 17",
            "encryption_key_revoked": True,
            "key_revocation_timestamp": datetime.now(),
            "propagation_started": datetime.now(),
            "acknowledged_by": {},
        }
        
        report = get_deletion_verification(tombstone_id)
        
        assert report is not None
        assert report["tombstone_id"] == str(tombstone_id)
        assert report["status"] == "complete"
        assert report["key_revoked"]

    def test_returns_none_for_missing_tombstone(self, mock_cursor):
        """Should return None for unknown tombstone."""
        mock_cursor.fetchone.return_value = None
        
        report = get_deletion_verification(uuid4())
        
        assert report is None

    def test_status_processing_when_key_not_revoked(self, mock_cursor):
        """Should show 'processing' status when key not yet revoked."""
        from datetime import datetime
        mock_cursor.fetchone.return_value = {
            "id": uuid4(),
            "target_type": "user",
            "target_id": uuid4(),
            "created_at": datetime.now(),
            "created_by": "hashed",
            "reason": "user_request",
            "encryption_key_revoked": False,  # Not yet revoked
            "key_revocation_timestamp": None,
            "propagation_started": None,
            "acknowledged_by": {},
        }
        
        report = get_deletion_verification(uuid4())
        
        assert report["status"] == "processing"
        assert not report["key_revoked"]
