"""Tests for Sharing API endpoints (server/sharing_endpoints.py).

Tests cover:
1. share_belief_endpoint - POST /api/v1/share
2. list_shares_endpoint - GET /api/v1/shares
3. get_share_endpoint - GET /api/v1/shares/{id}
4. revoke_share_endpoint - POST /api/v1/shares/{id}/revoke
5. Error handling (missing fields, invalid JSON, service unavailable)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from valence.server.auth_helpers import AuthenticatedClient
from valence.server.sharing_endpoints import (
    _share_to_dict,
    get_share_endpoint,
    get_sharing_service,
    list_shares_endpoint,
    revoke_share_endpoint,
    set_sharing_service,
    share_belief_endpoint,
)

MOCK_CLIENT = AuthenticatedClient(client_id="test", auth_method="bearer")

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_sharing_service():
    """Create a mock sharing service."""
    service = MagicMock()
    service.identity = MagicMock()
    service.identity.get_did.return_value = "did:vkb:key:z6MkSharer"
    return service


@pytest.fixture
def app(mock_sharing_service):
    """Create a test Starlette app with sharing endpoints."""
    routes = [
        Route("/api/v1/share", share_belief_endpoint, methods=["POST"]),
        Route("/api/v1/shares", list_shares_endpoint, methods=["GET"]),
        Route("/api/v1/shares/{id}", get_share_endpoint, methods=["GET"]),
        Route("/api/v1/shares/{id}/revoke", revoke_share_endpoint, methods=["POST"]),
    ]

    set_sharing_service(mock_sharing_service)
    return Starlette(routes=routes)


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_auth():
    with patch("valence.server.sharing_endpoints.authenticate", return_value=MOCK_CLIENT):
        yield


# ============================================================================
# Auth Tests
# ============================================================================


class TestShareAuth:
    """Test authentication requirement on sharing endpoints."""

    def test_unauthenticated_returns_401(self, client):
        from starlette.responses import JSONResponse

        with patch(
            "valence.server.sharing_endpoints.authenticate",
            return_value=JSONResponse({"error": "unauthorized"}, status_code=401),
        ):
            resp = client.post("/api/v1/share", json={"belief_id": "x", "recipient_did": "y"})
            assert resp.status_code == 401

    def test_wrong_scope_returns_403(self, client):
        oauth_client = AuthenticatedClient(client_id="test", auth_method="oauth", scope="vkb:read")
        with patch("valence.server.sharing_endpoints.authenticate", return_value=oauth_client):
            resp = client.post("/api/v1/share", json={"belief_id": "x", "recipient_did": "y"})
            assert resp.status_code == 403


# ============================================================================
# Service Management Tests
# ============================================================================


class TestServiceManagement:
    """Test sharing service get/set functions."""

    def test_set_and_get_service(self):
        """Set and get sharing service."""
        # Clear any existing service
        set_sharing_service(None)
        assert get_sharing_service() is None

        mock_service = MagicMock()
        set_sharing_service(mock_service)

        assert get_sharing_service() == mock_service

        # Clean up
        set_sharing_service(None)


# ============================================================================
# share_belief_endpoint Tests
# ============================================================================


class TestShareBeliefEndpoint:
    """Test POST /api/v1/share endpoint."""

    def test_share_success(self, client, mock_sharing_service):
        """Successfully share a belief."""
        mock_result = MagicMock()
        mock_result.share_id = "share-001"
        mock_result.consent_chain_id = "chain-001"
        mock_result.encrypted_for = "did:vkb:key:z6MkRecipient"
        mock_result.created_at = "2024-01-15T10:30:00Z"

        mock_sharing_service.share = AsyncMock(return_value=mock_result)

        response = client.post(
            "/api/v1/share",
            json={
                "belief_id": "belief-001",
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["share_id"] == "share-001"
        assert data["consent_chain_id"] == "chain-001"

    def test_share_with_policy(self, client, mock_sharing_service):
        """Share with explicit policy."""
        mock_result = MagicMock()
        mock_result.share_id = "share-002"
        mock_result.consent_chain_id = "chain-002"
        mock_result.encrypted_for = "did:vkb:key:z6MkRecipient"
        mock_result.created_at = "2024-01-15T10:30:00Z"

        mock_sharing_service.share = AsyncMock(return_value=mock_result)

        response = client.post(
            "/api/v1/share",
            json={
                "belief_id": "belief-002",
                "recipient_did": "did:vkb:key:z6MkRecipient",
                "policy": {
                    "level": "direct",
                    "enforcement": "cryptographic",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_share_missing_belief_id(self, client, mock_sharing_service):
        """Missing belief_id returns 400."""
        response = client.post(
            "/api/v1/share",
            json={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "belief_id" in data.get("error", {}).get("message", data.get("message", ""))

    def test_share_missing_recipient_did(self, client, mock_sharing_service):
        """Missing recipient_did returns 400."""
        response = client.post(
            "/api/v1/share",
            json={
                "belief_id": "belief-001",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "recipient_did" in data.get("error", {}).get("message", data.get("message", ""))

    def test_share_invalid_json(self, client, mock_sharing_service):
        """Invalid JSON returns 400."""
        response = client.post(
            "/api/v1/share",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "JSON" in data.get("error", {}).get("message", data.get("message", ""))

    def test_share_invalid_policy(self, client, mock_sharing_service):
        """Invalid policy returns 400."""
        # Mock SharePolicy.from_dict to raise
        with patch("valence.server.sharing_endpoints.SharePolicy") as mock_policy:
            mock_policy.from_dict.side_effect = ValueError("Invalid level")

            response = client.post(
                "/api/v1/share",
                json={
                    "belief_id": "belief-001",
                    "recipient_did": "did:vkb:key:z6MkRecipient",
                    "policy": {"level": "invalid"},
                },
            )

        assert response.status_code == 400
        data = response.json()
        assert "policy" in data.get("error", {}).get("message", data.get("message", "")).lower()

    def test_share_service_unavailable(self, client):
        """Service unavailable returns 503."""
        set_sharing_service(None)

        response = client.post(
            "/api/v1/share",
            json={
                "belief_id": "belief-001",
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 503

    def test_share_validation_error(self, client, mock_sharing_service):
        """Service validation error returns 400."""
        mock_sharing_service.share = AsyncMock(side_effect=ValueError("Belief not found"))

        response = client.post(
            "/api/v1/share",
            json={
                "belief_id": "nonexistent",
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "Belief not found" in data.get("error", {}).get("message", data.get("message", ""))

    def test_share_internal_error(self, client, mock_sharing_service):
        """Internal error returns 500."""
        mock_sharing_service.share = AsyncMock(side_effect=Exception("Database error"))

        response = client.post(
            "/api/v1/share",
            json={
                "belief_id": "belief-001",
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 500


# ============================================================================
# list_shares_endpoint Tests
# ============================================================================


class TestListSharesEndpoint:
    """Test GET /api/v1/shares endpoint."""

    def test_list_shares_success(self, client, mock_sharing_service):
        """Successfully list shares."""
        mock_share = MagicMock()
        mock_share.id = "share-001"
        mock_share.consent_chain_id = "chain-001"
        mock_share.recipient_did = "did:vkb:key:z6MkRecipient"
        mock_share.created_at = "2024-01-15T10:30:00Z"
        mock_share.accessed_at = None

        mock_sharing_service.list_shares = AsyncMock(return_value=[mock_share])

        response = client.get("/api/v1/shares")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["count"] == 1
        assert len(data["shares"]) == 1
        assert data["shares"][0]["id"] == "share-001"

    def test_list_shares_with_filters(self, client, mock_sharing_service):
        """List shares with query filters."""
        mock_sharing_service.list_shares = AsyncMock(return_value=[])

        response = client.get(
            "/api/v1/shares",
            params={
                "sharer_did": "did:vkb:key:z6MkSharer",
                "recipient_did": "did:vkb:key:z6MkRecipient",
                "limit": "50",
                "revoked": "true",
            },
        )

        assert response.status_code == 200

        # Verify filters were passed
        call_kwargs = mock_sharing_service.list_shares.call_args[1]
        assert call_kwargs["sharer_did"] == "did:vkb:key:z6MkSharer"
        assert call_kwargs["recipient_did"] == "did:vkb:key:z6MkRecipient"
        assert call_kwargs["limit"] == 50
        assert call_kwargs["include_revoked"] is True

    def test_list_shares_limit_capped(self, client, mock_sharing_service):
        """Limit is capped at 1000."""
        mock_sharing_service.list_shares = AsyncMock(return_value=[])

        response = client.get("/api/v1/shares", params={"limit": "5000"})

        assert response.status_code == 200

        call_kwargs = mock_sharing_service.list_shares.call_args[1]
        assert call_kwargs["limit"] == 1000

    def test_list_shares_invalid_limit(self, client, mock_sharing_service):
        """Invalid limit defaults to 100."""
        mock_sharing_service.list_shares = AsyncMock(return_value=[])

        response = client.get("/api/v1/shares", params={"limit": "abc"})

        assert response.status_code == 200

        call_kwargs = mock_sharing_service.list_shares.call_args[1]
        assert call_kwargs["limit"] == 100

    def test_list_shares_service_unavailable(self, client):
        """Service unavailable returns 503."""
        set_sharing_service(None)

        response = client.get("/api/v1/shares")

        assert response.status_code == 503

    def test_list_shares_internal_error(self, client, mock_sharing_service):
        """Internal error returns 500."""
        mock_sharing_service.list_shares = AsyncMock(side_effect=Exception("Database error"))

        response = client.get("/api/v1/shares")

        assert response.status_code == 500


# ============================================================================
# get_share_endpoint Tests
# ============================================================================


class TestGetShareEndpoint:
    """Test GET /api/v1/shares/{id} endpoint."""

    def test_get_share_success(self, client, mock_sharing_service):
        """Successfully get a share."""
        mock_share = MagicMock()
        mock_share.id = "share-001"
        mock_share.consent_chain_id = "chain-001"
        mock_share.recipient_did = "did:vkb:key:z6MkRecipient"
        mock_share.created_at = "2024-01-15T10:30:00Z"
        mock_share.accessed_at = "2024-01-16T08:00:00Z"

        mock_sharing_service.get_share = AsyncMock(return_value=mock_share)

        response = client.get("/api/v1/shares/share-001")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["share"]["id"] == "share-001"

    def test_get_share_not_found(self, client, mock_sharing_service):
        """Share not found returns 404."""
        mock_sharing_service.get_share = AsyncMock(return_value=None)

        response = client.get("/api/v1/shares/nonexistent")

        assert response.status_code == 404

    def test_get_share_service_unavailable(self, client):
        """Service unavailable returns 503."""
        set_sharing_service(None)

        response = client.get("/api/v1/shares/share-001")

        assert response.status_code == 503

    def test_get_share_internal_error(self, client, mock_sharing_service):
        """Internal error returns 500."""
        mock_sharing_service.get_share = AsyncMock(side_effect=Exception("Database error"))

        response = client.get("/api/v1/shares/share-001")

        assert response.status_code == 500


# ============================================================================
# revoke_share_endpoint Tests
# ============================================================================


class TestRevokeShareEndpoint:
    """Test POST /api/v1/shares/{id}/revoke endpoint."""

    def test_revoke_success(self, client, mock_sharing_service):
        """Successfully revoke a share."""
        mock_result = MagicMock()
        mock_result.share_id = "share-001"
        mock_result.consent_chain_id = "chain-001"
        mock_result.revoked_at = "2024-01-17T12:00:00Z"
        mock_result.affected_recipients = ["did:vkb:key:z6MkRecipient"]

        mock_sharing_service.revoke_share = AsyncMock(return_value=mock_result)

        response = client.post("/api/v1/shares/share-001/revoke", json={})

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["share_id"] == "share-001"
        assert data["revoked_at"] == "2024-01-17T12:00:00Z"

    def test_revoke_with_reason(self, client, mock_sharing_service):
        """Revoke with reason."""
        mock_result = MagicMock()
        mock_result.share_id = "share-001"
        mock_result.consent_chain_id = "chain-001"
        mock_result.revoked_at = "2024-01-17T12:00:00Z"
        mock_result.affected_recipients = []

        mock_sharing_service.revoke_share = AsyncMock(return_value=mock_result)

        response = client.post(
            "/api/v1/shares/share-001/revoke",
            json={
                "reason": "Privacy concern",
            },
        )

        assert response.status_code == 200

        # Verify reason was passed
        call_args = mock_sharing_service.revoke_share.call_args[0]
        assert call_args[0].reason == "Privacy concern"

    def test_revoke_empty_body(self, client, mock_sharing_service):
        """Revoke with empty body is OK (reason is optional)."""
        mock_result = MagicMock()
        mock_result.share_id = "share-001"
        mock_result.consent_chain_id = "chain-001"
        mock_result.revoked_at = "2024-01-17T12:00:00Z"
        mock_result.affected_recipients = []

        mock_sharing_service.revoke_share = AsyncMock(return_value=mock_result)

        # Send without body
        response = client.post("/api/v1/shares/share-001/revoke")

        assert response.status_code == 200

    def test_revoke_not_found(self, client, mock_sharing_service):
        """Share not found returns 404."""
        mock_sharing_service.revoke_share = AsyncMock(side_effect=ValueError("Share not found"))

        response = client.post("/api/v1/shares/nonexistent/revoke", json={})

        assert response.status_code == 404

    def test_revoke_already_revoked(self, client, mock_sharing_service):
        """Already revoked returns 409."""
        mock_sharing_service.revoke_share = AsyncMock(side_effect=ValueError("Share already revoked"))

        response = client.post("/api/v1/shares/share-001/revoke", json={})

        assert response.status_code == 409
        data = response.json()
        assert "already revoked" in data.get("error", {}).get("message", data.get("message", "")).lower()

    def test_revoke_permission_denied(self, client, mock_sharing_service):
        """Not owner returns 403."""
        mock_sharing_service.revoke_share = AsyncMock(side_effect=PermissionError("Not the original sharer"))

        response = client.post("/api/v1/shares/share-001/revoke", json={})

        assert response.status_code == 403

    def test_revoke_service_unavailable(self, client):
        """Service unavailable returns 503."""
        set_sharing_service(None)

        response = client.post("/api/v1/shares/share-001/revoke", json={})

        assert response.status_code == 503

    def test_revoke_internal_error(self, client, mock_sharing_service):
        """Internal error returns 500."""
        mock_sharing_service.revoke_share = AsyncMock(side_effect=Exception("Database error"))

        response = client.post("/api/v1/shares/share-001/revoke", json={})

        assert response.status_code == 500


# ============================================================================
# _share_to_dict Helper Tests
# ============================================================================


class TestShareToDict:
    """Test _share_to_dict helper function."""

    def test_share_to_dict_full(self):
        """Convert share with all fields."""
        share = MagicMock()
        share.id = "share-001"
        share.consent_chain_id = "chain-001"
        share.recipient_did = "did:vkb:key:z6MkRecipient"
        share.created_at = "2024-01-15T10:30:00Z"
        share.accessed_at = "2024-01-16T08:00:00Z"

        result = _share_to_dict(share)

        assert result["id"] == "share-001"
        assert result["consent_chain_id"] == "chain-001"
        assert result["recipient_did"] == "did:vkb:key:z6MkRecipient"
        assert result["created_at"] == "2024-01-15T10:30:00Z"
        assert result["accessed_at"] == "2024-01-16T08:00:00Z"

    def test_share_to_dict_null_accessed(self):
        """Convert share with null accessed_at."""
        share = MagicMock()
        share.id = "share-002"
        share.consent_chain_id = "chain-002"
        share.recipient_did = "did:vkb:key:z6MkRecipient"
        share.created_at = "2024-01-15T10:30:00Z"
        share.accessed_at = None

        result = _share_to_dict(share)

        assert result["accessed_at"] is None
