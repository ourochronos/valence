"""Tests for compliance API endpoints.

Covers:
- DELETE /api/v1/users/{id}/data (GDPR Article 17 deletion)
- GET /api/v1/tombstones/{id}/verification
- Error handling paths
- Input validation
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from starlette.requests import Request
from valence.compliance.deletion import (
    DeletionReason,
    DeletionResult,
    Tombstone,
)
from valence.server.compliance_endpoints import (
    delete_user_data_endpoint,
    get_deletion_verification_endpoint,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_request():
    """Create a mock Starlette request."""

    def _make_request(
        path_params: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
    ) -> MagicMock:
        request = MagicMock(spec=Request)
        request.path_params = path_params or {}

        # Mock query_params as a dict-like object with .get()
        qp = query_params or {}
        mock_qp = MagicMock()
        mock_qp.get = lambda key, default=None: qp.get(key, default)
        request.query_params = mock_qp

        return request

    return _make_request


@pytest.fixture
def mock_deletion_result():
    """Create a successful deletion result."""
    return DeletionResult(
        success=True,
        tombstone_id=uuid4(),
        beliefs_deleted=5,
        entities_anonymized=1,
        sessions_deleted=2,
        exchanges_deleted=10,
        patterns_deleted=3,
    )


@pytest.fixture
def mock_tombstone():
    """Create a mock tombstone for testing."""
    return Tombstone(
        id=uuid4(),
        target_type="user",
        target_id=uuid4(),
        created_at=datetime.now(),
        created_by="hashed_user_id",
        reason=DeletionReason.USER_REQUEST,
        legal_basis="GDPR Article 17 - user_request",
        encryption_key_revoked=True,
        key_revocation_timestamp=datetime.now(),
        propagation_started=datetime.now(),
        acknowledged_by={"peer1": datetime.now()},
    )


# ============================================================================
# Test delete_user_data_endpoint
# ============================================================================


class TestDeleteUserDataEndpoint:
    """Test DELETE /api/v1/users/{id}/data endpoint."""

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_request, mock_deletion_result):
        """Successful user data deletion."""
        request = mock_request(
            path_params={"id": "user123"},
            query_params={"reason": "user_request"},
        )

        with patch("valence.server.compliance_endpoints.delete_user_data") as mock_delete:
            mock_delete.return_value = mock_deletion_result

            response = await delete_user_data_endpoint(request)

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["success"] is True
        assert body["tombstone_id"] is not None
        assert body["deleted_counts"]["beliefs"] == 5
        assert body["deleted_counts"]["sessions"] == 2

    @pytest.mark.asyncio
    async def test_delete_missing_user_id(self, mock_request):
        """Missing user ID returns 400."""
        request = mock_request(path_params={})

        response = await delete_user_data_endpoint(request)

        assert response.status_code == 400
        body = json.loads(response.body)
        assert "error" in body
        assert "User ID is required" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_delete_invalid_reason(self, mock_request):
        """Invalid deletion reason returns 400."""
        request = mock_request(
            path_params={"id": "user123"},
            query_params={"reason": "invalid_reason"},
        )

        response = await delete_user_data_endpoint(request)

        assert response.status_code == 400
        body = json.loads(response.body)
        assert "error" in body
        assert "Invalid reason" in body["error"]["message"]
        # Valid reasons are now in the error message
        assert "user_request" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_delete_default_reason(self, mock_request, mock_deletion_result):
        """Default reason is 'user_request' when not specified."""
        request = mock_request(
            path_params={"id": "user123"},
            query_params={},  # No reason specified
        )

        with patch("valence.server.compliance_endpoints.delete_user_data") as mock_delete:
            mock_delete.return_value = mock_deletion_result

            response = await delete_user_data_endpoint(request)

        assert response.status_code == 200
        # Verify default reason was used
        mock_delete.assert_called_once()
        call_kwargs = mock_delete.call_args[1]
        assert call_kwargs["reason"] == DeletionReason.USER_REQUEST

    @pytest.mark.asyncio
    async def test_delete_with_legal_basis(self, mock_request, mock_deletion_result):
        """Delete with legal basis parameter."""
        request = mock_request(
            path_params={"id": "user123"},
            query_params={
                "reason": "legal_order",
                "legal_basis": "Court order #12345",
            },
        )

        with patch("valence.server.compliance_endpoints.delete_user_data") as mock_delete:
            mock_delete.return_value = mock_deletion_result

            response = await delete_user_data_endpoint(request)

        assert response.status_code == 200
        mock_delete.assert_called_once()
        call_kwargs = mock_delete.call_args[1]
        assert call_kwargs["legal_basis"] == "Court order #12345"
        assert call_kwargs["reason"] == DeletionReason.LEGAL_ORDER

    @pytest.mark.asyncio
    async def test_delete_all_valid_reasons(self, mock_request, mock_deletion_result):
        """Test all valid deletion reasons are accepted."""
        valid_reasons = [
            "user_request",
            "consent_withdrawal",
            "legal_order",
            "policy_violation",
            "data_accuracy",
            "security_incident",
        ]

        for reason in valid_reasons:
            request = mock_request(
                path_params={"id": f"user_{reason}"},
                query_params={"reason": reason},
            )

            with patch("valence.server.compliance_endpoints.delete_user_data") as mock_delete:
                mock_delete.return_value = mock_deletion_result

                response = await delete_user_data_endpoint(request)

            assert response.status_code == 200, f"Failed for reason: {reason}"

    @pytest.mark.asyncio
    async def test_delete_failure_returns_500(self, mock_request):
        """Deletion failure returns 500."""
        request = mock_request(
            path_params={"id": "user123"},
            query_params={"reason": "user_request"},
        )

        failed_result = DeletionResult(
            success=False,
            error="Database connection failed",
        )

        with patch("valence.server.compliance_endpoints.delete_user_data") as mock_delete:
            mock_delete.return_value = failed_result

            response = await delete_user_data_endpoint(request)

        assert response.status_code == 500
        body = json.loads(response.body)
        assert body["success"] is False
        assert "error" in body

    @pytest.mark.asyncio
    async def test_delete_exception_returns_500(self, mock_request):
        """Unexpected exception returns 500."""
        request = mock_request(
            path_params={"id": "user123"},
            query_params={"reason": "user_request"},
        )

        with patch("valence.server.compliance_endpoints.delete_user_data") as mock_delete:
            mock_delete.side_effect = Exception("Unexpected database error")

            response = await delete_user_data_endpoint(request)

        assert response.status_code == 500
        body = json.loads(response.body)
        assert body["success"] is False
        assert "Unexpected database error" in body["error"]["message"]


# ============================================================================
# Test get_deletion_verification_endpoint
# ============================================================================


class TestGetDeletionVerificationEndpoint:
    """Test GET /api/v1/tombstones/{id}/verification endpoint."""

    @pytest.mark.asyncio
    async def test_verification_success(self, mock_request):
        """Successful verification report retrieval."""
        tombstone_id = uuid4()
        request = mock_request(path_params={"id": str(tombstone_id)})

        mock_report = {
            "tombstone_id": str(tombstone_id),
            "status": "complete",
            "tombstone_created": "2026-02-04T00:00:00",
            "key_revoked": True,
            "key_revocation_timestamp": "2026-02-04T00:01:00",
            "propagation_status": {
                "started": True,
                "acknowledged_count": 3,
            },
            "legal_basis": "GDPR Article 17",
            "reason": "user_request",
        }

        with patch("valence.server.compliance_endpoints.get_deletion_verification") as mock_get:
            mock_get.return_value = mock_report

            response = await get_deletion_verification_endpoint(request)

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["tombstone_id"] == str(tombstone_id)
        assert body["status"] == "complete"
        assert body["key_revoked"] is True

    @pytest.mark.asyncio
    async def test_verification_not_found(self, mock_request):
        """Tombstone not found returns 404."""
        tombstone_id = uuid4()
        request = mock_request(path_params={"id": str(tombstone_id)})

        with patch("valence.server.compliance_endpoints.get_deletion_verification") as mock_get:
            mock_get.return_value = None

            response = await get_deletion_verification_endpoint(request)

        assert response.status_code == 404
        body = json.loads(response.body)
        assert "error" in body
        assert "not found" in body["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_verification_invalid_uuid(self, mock_request):
        """Invalid UUID returns 400."""
        request = mock_request(path_params={"id": "not-a-valid-uuid"})

        response = await get_deletion_verification_endpoint(request)

        assert response.status_code == 400
        body = json.loads(response.body)
        assert "error" in body
        assert "Invalid tombstone ID" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_verification_missing_id(self, mock_request):
        """Missing tombstone ID returns 400."""
        request = mock_request(path_params={})

        response = await get_deletion_verification_endpoint(request)

        assert response.status_code == 400
        body = json.loads(response.body)
        assert "Invalid tombstone ID" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_verification_empty_string_id(self, mock_request):
        """Empty string ID returns 400."""
        request = mock_request(path_params={"id": ""})

        response = await get_deletion_verification_endpoint(request)

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_verification_processing_status(self, mock_request):
        """Verification shows 'processing' status when key not yet revoked."""
        tombstone_id = uuid4()
        request = mock_request(path_params={"id": str(tombstone_id)})

        mock_report = {
            "tombstone_id": str(tombstone_id),
            "status": "processing",  # Key not yet revoked
            "tombstone_created": "2026-02-04T00:00:00",
            "key_revoked": False,
            "key_revocation_timestamp": None,
            "propagation_status": {
                "started": False,
                "acknowledged_count": 0,
            },
            "legal_basis": None,
            "reason": "user_request",
        }

        with patch("valence.server.compliance_endpoints.get_deletion_verification") as mock_get:
            mock_get.return_value = mock_report

            response = await get_deletion_verification_endpoint(request)

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["status"] == "processing"
        assert body["key_revoked"] is False


# ============================================================================
# Test DeletionReason enum
# ============================================================================


class TestDeletionReasonEnum:
    """Test DeletionReason enum values."""

    def test_all_reasons_have_values(self):
        """All deletion reasons have string values."""
        for reason in DeletionReason:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0

    def test_user_request_value(self):
        """USER_REQUEST has expected value."""
        assert DeletionReason.USER_REQUEST.value == "user_request"

    def test_consent_withdrawal_value(self):
        """CONSENT_WITHDRAWAL has expected value."""
        assert DeletionReason.CONSENT_WITHDRAWAL.value == "consent_withdrawal"

    def test_legal_order_value(self):
        """LEGAL_ORDER has expected value."""
        assert DeletionReason.LEGAL_ORDER.value == "legal_order"

    def test_invalid_reason_raises(self):
        """Invalid reason raises ValueError."""
        with pytest.raises(ValueError):
            DeletionReason("not_a_real_reason")


# ============================================================================
# Test DeletionResult
# ============================================================================


class TestDeletionResult:
    """Test DeletionResult dataclass."""

    def test_successful_result_to_dict(self):
        """Successful result serializes correctly."""
        tombstone_id = uuid4()
        result = DeletionResult(
            success=True,
            tombstone_id=tombstone_id,
            beliefs_deleted=10,
            entities_anonymized=2,
            sessions_deleted=3,
            exchanges_deleted=15,
            patterns_deleted=5,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["tombstone_id"] == str(tombstone_id)
        assert data["deleted_counts"]["beliefs"] == 10
        assert data["deleted_counts"]["entities_anonymized"] == 2
        assert data["deleted_counts"]["sessions"] == 3
        assert data["deleted_counts"]["exchanges"] == 15
        assert data["deleted_counts"]["patterns"] == 5
        assert data["error"] is None

    def test_failed_result_to_dict(self):
        """Failed result serializes with error."""
        result = DeletionResult(
            success=False,
            error="Database connection failed",
        )

        data = result.to_dict()

        assert data["success"] is False
        assert data["tombstone_id"] is None
        assert data["error"] == "Database connection failed"

    def test_default_counts_are_zero(self):
        """Default deletion counts are zero."""
        result = DeletionResult(success=True)

        data = result.to_dict()

        assert data["deleted_counts"]["beliefs"] == 0
        assert data["deleted_counts"]["sessions"] == 0
        assert data["deleted_counts"]["exchanges"] == 0
        assert data["deleted_counts"]["patterns"] == 0


# ============================================================================
# Test Tombstone
# ============================================================================


class TestTombstone:
    """Test Tombstone dataclass."""

    def test_tombstone_to_dict(self, mock_tombstone):
        """Tombstone serializes to dict."""
        data = mock_tombstone.to_dict()

        assert "id" in data
        assert data["target_type"] == "user"
        assert data["reason"] == "user_request"
        assert data["encryption_key_revoked"] is True
        assert data["legal_basis"] is not None

    def test_tombstone_from_row(self):
        """Tombstone deserializes from database row."""
        tombstone_id = uuid4()
        target_id = uuid4()
        now = datetime.now()

        row = {
            "id": str(tombstone_id),
            "target_type": "belief",
            "target_id": str(target_id),
            "created_at": now,
            "created_by": "hashed_id",
            "reason": "consent_withdrawal",
            "legal_basis": "User withdrew consent",
            "encryption_key_revoked": True,
            "key_revocation_timestamp": now,
            "propagation_started": now,
            "acknowledged_by": json.dumps({"peer1": now.isoformat()}),
            "signature": None,
        }

        tombstone = Tombstone.from_row(row)

        assert tombstone.id == tombstone_id
        assert tombstone.target_type == "belief"
        assert tombstone.target_id == target_id
        assert tombstone.reason == DeletionReason.CONSENT_WITHDRAWAL
        assert tombstone.encryption_key_revoked is True

    def test_tombstone_from_row_with_uuid_objects(self):
        """Tombstone handles UUID objects in row."""
        tombstone_id = uuid4()
        target_id = uuid4()
        now = datetime.now()

        row = {
            "id": tombstone_id,  # UUID object, not string
            "target_type": "user",
            "target_id": target_id,  # UUID object
            "created_at": now,
            "created_by": "hashed",
            "reason": "user_request",
            "acknowledged_by": {},
        }

        tombstone = Tombstone.from_row(row)

        assert tombstone.id == tombstone_id
        assert tombstone.target_id == target_id

    def test_tombstone_signature_serialization(self):
        """Tombstone signature serializes as hex."""
        tombstone = Tombstone(
            id=uuid4(),
            target_type="belief",
            target_id=uuid4(),
            created_at=datetime.now(),
            created_by="user",
            reason=DeletionReason.USER_REQUEST,
            signature=b"\x00\x01\x02\x03\xff",
        )

        data = tombstone.to_dict()

        assert data["signature"] == "00010203ff"


# ============================================================================
# Integration-style tests (with mocked dependencies)
# ============================================================================


class TestEndpointIntegration:
    """Integration tests for compliance endpoints."""

    @pytest.mark.asyncio
    async def test_full_deletion_flow(self, mock_request):
        """Test full deletion and verification flow."""
        user_id = "test-user-123"
        tombstone_id = uuid4()

        # Step 1: Delete user data
        delete_request = mock_request(
            path_params={"id": user_id},
            query_params={"reason": "user_request"},
        )

        deletion_result = DeletionResult(
            success=True,
            tombstone_id=tombstone_id,
            beliefs_deleted=3,
            sessions_deleted=1,
        )

        with patch("valence.server.compliance_endpoints.delete_user_data") as mock_delete:
            mock_delete.return_value = deletion_result
            delete_response = await delete_user_data_endpoint(delete_request)

        assert delete_response.status_code == 200
        delete_body = json.loads(delete_response.body)
        returned_tombstone_id = delete_body["tombstone_id"]

        # Step 2: Verify deletion
        verify_request = mock_request(
            path_params={"id": returned_tombstone_id},
        )

        verification_report = {
            "tombstone_id": returned_tombstone_id,
            "status": "complete",
            "tombstone_created": "2026-02-04T00:00:00",
            "key_revoked": True,
            "key_revocation_timestamp": "2026-02-04T00:01:00",
            "propagation_status": {
                "started": True,
                "acknowledged_count": 0,
            },
            "legal_basis": "GDPR Article 17 - user_request",
            "reason": "user_request",
        }

        with patch("valence.server.compliance_endpoints.get_deletion_verification") as mock_verify:
            mock_verify.return_value = verification_report
            verify_response = await get_deletion_verification_endpoint(verify_request)

        assert verify_response.status_code == 200
        verify_body = json.loads(verify_response.body)
        assert verify_body["status"] == "complete"
        assert verify_body["key_revoked"] is True
