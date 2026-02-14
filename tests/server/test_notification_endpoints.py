"""Tests for Notification API endpoints (server/notification_endpoints.py).

Tests cover:
1. list_notifications_endpoint - GET /api/v1/notifications
2. acknowledge_notification_endpoint - POST /api/v1/notifications/{id}/acknowledge
3. Error handling (missing fields, invalid JSON, service unavailable)
4. _notification_to_dict helper
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from valence.server.auth_helpers import AuthenticatedClient
from valence.server.notification_endpoints import (
    _notification_to_dict,
    acknowledge_notification_endpoint,
    list_notifications_endpoint,
)
from valence.server.sharing_endpoints import set_sharing_service

MOCK_CLIENT = AuthenticatedClient(client_id="test", auth_method="bearer")

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_sharing_service():
    """Create a mock sharing service."""
    service = MagicMock()
    return service


@pytest.fixture
def app(mock_sharing_service):
    """Create a test Starlette app with notification endpoints."""
    routes = [
        Route("/api/v1/notifications", list_notifications_endpoint, methods=["GET"]),
        Route(
            "/api/v1/notifications/{id}/acknowledge",
            acknowledge_notification_endpoint,
            methods=["POST"],
        ),
    ]

    set_sharing_service(mock_sharing_service)
    return Starlette(routes=routes)


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_auth():
    with patch("valence.server.notification_endpoints.authenticate", return_value=MOCK_CLIENT):
        yield


# ============================================================================
# Auth Tests
# ============================================================================


class TestNotificationAuth:
    """Test authentication requirement on notification endpoints."""

    def test_unauthenticated_returns_401(self, client):
        from starlette.responses import JSONResponse

        with patch(
            "valence.server.notification_endpoints.authenticate",
            return_value=JSONResponse({"error": "unauthorized"}, status_code=401),
        ):
            resp = client.get("/api/v1/notifications", params={"recipient_did": "did:test"})
            assert resp.status_code == 401


# ============================================================================
# list_notifications_endpoint Tests
# ============================================================================


class TestListNotificationsEndpoint:
    """Test GET /api/v1/notifications endpoint."""

    def test_list_notifications_success(self, client, mock_sharing_service):
        """Successfully list notifications."""
        mock_notification = MagicMock()
        mock_notification.id = "notif-001"
        mock_notification.recipient_did = "did:vkb:key:z6MkRecipient"
        mock_notification.notification_type = "share_received"
        mock_notification.payload = {"share_id": "share-001"}
        mock_notification.created_at = "2024-01-15T10:30:00Z"
        mock_notification.acknowledged_at = None

        mock_sharing_service.get_pending_notifications = AsyncMock(return_value=[mock_notification])

        response = client.get(
            "/api/v1/notifications",
            params={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["count"] == 1
        assert len(data["notifications"]) == 1
        assert data["notifications"][0]["id"] == "notif-001"
        assert data["notifications"][0]["notification_type"] == "share_received"

    def test_list_notifications_empty(self, client, mock_sharing_service):
        """List notifications returns empty list."""
        mock_sharing_service.get_pending_notifications = AsyncMock(return_value=[])

        response = client.get(
            "/api/v1/notifications",
            params={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["count"] == 0
        assert data["notifications"] == []

    def test_list_notifications_multiple(self, client, mock_sharing_service):
        """List multiple notifications."""
        mock_notifications = []
        for i in range(3):
            n = MagicMock()
            n.id = f"notif-00{i + 1}"
            n.recipient_did = "did:vkb:key:z6MkRecipient"
            n.notification_type = "share_received"
            n.payload = {"share_id": f"share-00{i + 1}"}
            n.created_at = f"2024-01-1{5 + i}T10:30:00Z"
            n.acknowledged_at = None
            mock_notifications.append(n)

        mock_sharing_service.get_pending_notifications = AsyncMock(return_value=mock_notifications)

        response = client.get(
            "/api/v1/notifications",
            params={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 3
        assert len(data["notifications"]) == 3

    def test_list_notifications_missing_recipient_did(self, client, mock_sharing_service):
        """Missing recipient_did returns 400."""
        response = client.get("/api/v1/notifications")

        assert response.status_code == 400
        data = response.json()
        # Handle both old and new error response formats
        error_msg = data.get("error", {}).get("message", data.get("message", ""))
        assert "recipient_did" in error_msg

    def test_list_notifications_service_unavailable(self, client):
        """Service unavailable returns 503."""
        set_sharing_service(None)

        response = client.get(
            "/api/v1/notifications",
            params={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 503

    def test_list_notifications_internal_error(self, client, mock_sharing_service):
        """Internal error returns 500."""
        mock_sharing_service.get_pending_notifications = AsyncMock(side_effect=Exception("Database error"))

        response = client.get(
            "/api/v1/notifications",
            params={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 500


# ============================================================================
# acknowledge_notification_endpoint Tests
# ============================================================================


class TestAcknowledgeNotificationEndpoint:
    """Test POST /api/v1/notifications/{id}/acknowledge endpoint."""

    def test_acknowledge_success(self, client, mock_sharing_service):
        """Successfully acknowledge a notification."""
        mock_sharing_service.acknowledge_notification = AsyncMock(return_value=True)

        response = client.post(
            "/api/v1/notifications/notif-001/acknowledge",
            json={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["notification_id"] == "notif-001"
        assert data["acknowledged"] is True

    def test_acknowledge_not_found(self, client, mock_sharing_service):
        """Notification not found returns 404."""
        mock_sharing_service.acknowledge_notification = AsyncMock(return_value=False)

        response = client.post(
            "/api/v1/notifications/nonexistent/acknowledge",
            json={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 404

    def test_acknowledge_missing_recipient_did(self, client, mock_sharing_service):
        """Missing recipient_did returns 400."""
        response = client.post("/api/v1/notifications/notif-001/acknowledge", json={})

        assert response.status_code == 400
        data = response.json()
        assert "recipient_did" in data.get("error", {}).get("message", data.get("message", ""))

    def test_acknowledge_invalid_json(self, client, mock_sharing_service):
        """Invalid JSON returns 400."""
        response = client.post(
            "/api/v1/notifications/notif-001/acknowledge",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "JSON" in data.get("error", {}).get("message", data.get("message", ""))

    def test_acknowledge_permission_denied(self, client, mock_sharing_service):
        """Permission denied returns 403."""
        mock_sharing_service.acknowledge_notification = AsyncMock(side_effect=PermissionError("Not the notification recipient"))

        response = client.post(
            "/api/v1/notifications/notif-001/acknowledge",
            json={
                "recipient_did": "did:vkb:key:z6MkWrongRecipient",
            },
        )

        assert response.status_code == 403

    def test_acknowledge_service_unavailable(self, client):
        """Service unavailable returns 503."""
        set_sharing_service(None)

        response = client.post(
            "/api/v1/notifications/notif-001/acknowledge",
            json={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 503

    def test_acknowledge_internal_error(self, client, mock_sharing_service):
        """Internal error returns 500."""
        mock_sharing_service.acknowledge_notification = AsyncMock(side_effect=Exception("Database error"))

        response = client.post(
            "/api/v1/notifications/notif-001/acknowledge",
            json={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 500


# ============================================================================
# _notification_to_dict Helper Tests
# ============================================================================


class TestNotificationToDict:
    """Test _notification_to_dict helper function."""

    def test_notification_to_dict_full(self):
        """Convert notification with all fields."""
        notification = MagicMock()
        notification.id = "notif-001"
        notification.recipient_did = "did:vkb:key:z6MkRecipient"
        notification.notification_type = "share_received"
        notification.payload = {"share_id": "share-001", "sharer_name": "Alice"}
        notification.created_at = "2024-01-15T10:30:00Z"
        notification.acknowledged_at = "2024-01-16T08:00:00Z"

        result = _notification_to_dict(notification)

        assert result["id"] == "notif-001"
        assert result["recipient_did"] == "did:vkb:key:z6MkRecipient"
        assert result["notification_type"] == "share_received"
        assert result["payload"] == {"share_id": "share-001", "sharer_name": "Alice"}
        assert result["created_at"] == "2024-01-15T10:30:00Z"
        assert result["acknowledged_at"] == "2024-01-16T08:00:00Z"

    def test_notification_to_dict_unacknowledged(self):
        """Convert unacknowledged notification."""
        notification = MagicMock()
        notification.id = "notif-002"
        notification.recipient_did = "did:vkb:key:z6MkRecipient"
        notification.notification_type = "share_revoked"
        notification.payload = {"share_id": "share-001"}
        notification.created_at = "2024-01-15T10:30:00Z"
        notification.acknowledged_at = None

        result = _notification_to_dict(notification)

        assert result["id"] == "notif-002"
        assert result["notification_type"] == "share_revoked"
        assert result["acknowledged_at"] is None

    def test_notification_to_dict_empty_payload(self):
        """Convert notification with empty payload."""
        notification = MagicMock()
        notification.id = "notif-003"
        notification.recipient_did = "did:vkb:key:z6MkRecipient"
        notification.notification_type = "consent_expired"
        notification.payload = {}
        notification.created_at = "2024-01-15T10:30:00Z"
        notification.acknowledged_at = None

        result = _notification_to_dict(notification)

        assert result["payload"] == {}

    def test_notification_to_dict_complex_payload(self):
        """Convert notification with complex payload."""
        notification = MagicMock()
        notification.id = "notif-004"
        notification.recipient_did = "did:vkb:key:z6MkRecipient"
        notification.notification_type = "batch_share"
        notification.payload = {
            "shares": [
                {"id": "share-001", "belief_id": "belief-001"},
                {"id": "share-002", "belief_id": "belief-002"},
            ],
            "total": 2,
            "from_did": "did:vkb:key:z6MkSharer",
        }
        notification.created_at = "2024-01-15T10:30:00Z"
        notification.acknowledged_at = None

        result = _notification_to_dict(notification)

        assert len(result["payload"]["shares"]) == 2
        assert result["payload"]["total"] == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestNotificationIntegration:
    """Integration tests for notification flow."""

    def test_list_then_acknowledge_flow(self, client, mock_sharing_service):
        """Test complete flow: list notifications, then acknowledge one."""
        # Setup: Create pending notifications
        mock_notification = MagicMock()
        mock_notification.id = "notif-flow-001"
        mock_notification.recipient_did = "did:vkb:key:z6MkRecipient"
        mock_notification.notification_type = "share_received"
        mock_notification.payload = {"share_id": "share-001"}
        mock_notification.created_at = "2024-01-15T10:30:00Z"
        mock_notification.acknowledged_at = None

        mock_sharing_service.get_pending_notifications = AsyncMock(return_value=[mock_notification])
        mock_sharing_service.acknowledge_notification = AsyncMock(return_value=True)

        # Step 1: List notifications
        list_response = client.get(
            "/api/v1/notifications",
            params={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert list_response.status_code == 200
        notifications = list_response.json()["notifications"]
        assert len(notifications) == 1
        notification_id = notifications[0]["id"]

        # Step 2: Acknowledge the notification
        ack_response = client.post(
            f"/api/v1/notifications/{notification_id}/acknowledge",
            json={"recipient_did": "did:vkb:key:z6MkRecipient"},
        )

        assert ack_response.status_code == 200
        assert ack_response.json()["acknowledged"] is True

        # Verify acknowledge was called with correct args
        mock_sharing_service.acknowledge_notification.assert_called_once_with(
            "notif-flow-001",
            "did:vkb:key:z6MkRecipient",
        )

    def test_different_notification_types(self, client, mock_sharing_service):
        """Test different notification types are returned correctly."""
        notification_types = [
            ("share_received", {"share_id": "s1"}),
            ("share_revoked", {"share_id": "s2", "reason": "Privacy"}),
            ("consent_expired", {"chain_id": "c1"}),
            ("access_requested", {"requester_did": "did:x"}),
        ]

        mock_notifications = []
        for i, (ntype, payload) in enumerate(notification_types):
            n = MagicMock()
            n.id = f"notif-type-{i}"
            n.recipient_did = "did:vkb:key:z6MkRecipient"
            n.notification_type = ntype
            n.payload = payload
            n.created_at = "2024-01-15T10:30:00Z"
            n.acknowledged_at = None
            mock_notifications.append(n)

        mock_sharing_service.get_pending_notifications = AsyncMock(return_value=mock_notifications)

        response = client.get(
            "/api/v1/notifications",
            params={
                "recipient_did": "did:vkb:key:z6MkRecipient",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 4

        # Verify each type is present
        types_returned = {n["notification_type"] for n in data["notifications"]}
        expected_types = {
            "share_received",
            "share_revoked",
            "consent_expired",
            "access_requested",
        }
        assert types_returned == expected_types
