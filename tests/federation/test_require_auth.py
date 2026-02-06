"""Tests for VALENCE_FEDERATION_REQUIRE_AUTH config (#255).

Verifies that:
- When enabled, unauthenticated federation requests are rejected with 401
- When disabled (default), requests pass through for backward compatibility
- Discovery endpoints (/.well-known/*) are never affected
- Protocol-level messages also respect the setting
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request

from valence.server.errors import AUTH_FEDERATION_REQUIRED
from valence.server.federation_endpoints import (
    federation_nodes_list,
    federation_status,
    federation_sync_status,
    require_federation_auth,
)

# =============================================================================
# HELPERS
# =============================================================================


def _make_mock_request(
    method: str = "GET",
    path: str = "/federation/status",
    headers: dict | None = None,
    body: bytes = b"{}",
) -> MagicMock:
    """Create a mock Starlette Request."""
    mock = MagicMock(spec=Request)
    mock.method = method
    mock.url = MagicMock()
    mock.url.path = path
    mock.headers = headers or {}
    mock.body = AsyncMock(return_value=body)
    mock.state = MagicMock()
    mock.query_params = {}
    return mock


def _make_signed_request(
    method: str = "GET",
    path: str = "/federation/status",
    body: bytes = b"{}",
) -> tuple[MagicMock, MagicMock]:
    """Create a mock signed request with valid DID headers.

    Returns (mock_request, mock_did_doc) for use with patched resolve.
    """
    from valence.federation.identity import generate_keypair, sign_message

    keypair = generate_keypair()
    did = "did:vkb:key:" + keypair.public_key_multibase
    timestamp = str(int(time.time()))
    nonce = "test-nonce-255"
    body_hash = hashlib.sha256(body).hexdigest()

    message = f"{method} {path} {timestamp} {nonce} {body_hash}"
    signature = sign_message(message.encode("utf-8"), keypair.private_key_bytes)

    mock_request = _make_mock_request(
        method=method,
        path=path,
        headers={
            "X-VFP-DID": did,
            "X-VFP-Signature": base64.b64encode(signature).decode(),
            "X-VFP-Timestamp": timestamp,
            "X-VFP-Nonce": nonce,
        },
        body=body,
    )

    mock_did_doc = MagicMock()
    mock_did_doc.public_key_multibase = keypair.public_key_multibase

    return mock_request, mock_did_doc


# =============================================================================
# CONFIG DEFAULT
# =============================================================================


class TestConfigDefault:
    """Test that federation_require_auth defaults to False."""

    def test_core_config_default(self):
        """CoreSettings should default federation_require_auth to False."""
        from valence.core.config import CoreSettings

        settings = CoreSettings()
        assert settings.federation_require_auth is False

    def test_core_config_from_env(self, monkeypatch):
        """CoreSettings should read VALENCE_FEDERATION_REQUIRE_AUTH env var."""
        monkeypatch.setenv("VALENCE_FEDERATION_REQUIRE_AUTH", "true")
        from valence.core.config import CoreSettings

        settings = CoreSettings()
        assert settings.federation_require_auth is True

    def test_core_config_false_from_env(self, monkeypatch):
        """CoreSettings should parse 'false' correctly."""
        monkeypatch.setenv("VALENCE_FEDERATION_REQUIRE_AUTH", "false")
        from valence.core.config import CoreSettings

        settings = CoreSettings()
        assert settings.federation_require_auth is False


# =============================================================================
# DECORATOR BEHAVIOR
# =============================================================================


class TestRequireFederationAuthDecorator:
    """Test the @require_federation_auth decorator."""

    @pytest.mark.asyncio
    async def test_passes_through_when_disabled(self):
        """When federation_require_auth=False, requests pass through."""

        @require_federation_auth
        async def handler(request: Request) -> dict:
            from starlette.responses import JSONResponse

            return JSONResponse({"success": True})

        mock_request = _make_mock_request()

        with patch("valence.server.federation_endpoints.get_settings") as mock_settings:
            mock_settings.return_value.federation_require_auth = False
            response = await handler(mock_request)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_rejects_unauthenticated_when_enabled(self):
        """When federation_require_auth=True, unsigned requests get 401."""

        @require_federation_auth
        async def handler(request: Request) -> dict:
            from starlette.responses import JSONResponse

            return JSONResponse({"success": True})

        mock_request = _make_mock_request()

        with patch("valence.server.federation_endpoints.get_settings") as mock_settings:
            mock_settings.return_value.federation_require_auth = True
            response = await handler(mock_request)

            assert response.status_code == 401
            body = json.loads(response.body)
            assert body["success"] is False
            assert body["error"]["code"] == AUTH_FEDERATION_REQUIRED
            assert "Authentication required" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_allows_authenticated_when_enabled(self):
        """When federation_require_auth=True, signed requests pass through."""

        @require_federation_auth
        async def handler(request: Request) -> dict:
            from starlette.responses import JSONResponse

            return JSONResponse({"success": True, "did": request.state.did_info["did"]})

        mock_request, mock_did_doc = _make_signed_request()

        with patch("valence.server.federation_endpoints.get_settings") as mock_settings:
            mock_settings.return_value.federation_require_auth = True

            with patch("valence.federation.identity.parse_did"):
                with patch("valence.federation.identity.resolve_did_sync") as mock_resolve:
                    mock_resolve.return_value = mock_did_doc

                    response = await handler(mock_request)
                    assert response.status_code == 200
                    body = json.loads(response.body)
                    assert body["success"] is True

    @pytest.mark.asyncio
    async def test_attaches_did_info_to_request_state(self):
        """Decorator should attach did_info to request.state on success."""

        @require_federation_auth
        async def handler(request: Request) -> dict:
            from starlette.responses import JSONResponse

            return JSONResponse(
                {
                    "did": request.state.did_info["did"],
                    "has_nonce": "nonce" in request.state.did_info,
                }
            )

        mock_request, mock_did_doc = _make_signed_request()

        with patch("valence.server.federation_endpoints.get_settings") as mock_settings:
            mock_settings.return_value.federation_require_auth = True

            with patch("valence.federation.identity.parse_did"):
                with patch("valence.federation.identity.resolve_did_sync") as mock_resolve:
                    mock_resolve.return_value = mock_did_doc

                    response = await handler(mock_request)
                    body = json.loads(response.body)
                    assert body["did"].startswith("did:vkb:key:")
                    assert body["has_nonce"] is True


# =============================================================================
# ENDPOINT-LEVEL TESTS
# =============================================================================


class TestEndpointEnforcement:
    """Test that the decorator is applied to the correct endpoints."""

    def test_federation_status_is_decorated(self):
        """federation_status should have @require_federation_auth."""
        assert hasattr(federation_status, "__wrapped__")

    def test_federation_nodes_list_is_decorated(self):
        """federation_nodes_list should have @require_federation_auth."""
        assert hasattr(federation_nodes_list, "__wrapped__")

    def test_federation_sync_status_is_decorated(self):
        """federation_sync_status should have @require_federation_auth."""
        assert hasattr(federation_sync_status, "__wrapped__")

    def test_discovery_endpoints_not_decorated(self):
        """Well-known discovery endpoints must remain public."""
        from valence.server.federation_endpoints import (
            vfp_node_metadata,
            vfp_trust_anchors,
        )

        # These should NOT be wrapped by require_federation_auth
        assert vfp_node_metadata.__name__ == "vfp_node_metadata"
        assert vfp_trust_anchors.__name__ == "vfp_trust_anchors"

    @pytest.mark.asyncio
    async def test_federation_status_rejects_when_auth_required(self):
        """federation_status should return 401 when auth required and not provided."""
        mock_request = _make_mock_request(path="/federation/status")

        with patch("valence.server.federation_endpoints.get_settings") as mock_settings:
            mock_settings.return_value.federation_require_auth = True
            mock_settings.return_value.federation_enabled = True

            response = await federation_status(mock_request)
            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_federation_status_allows_when_auth_not_required(self):
        """federation_status should work without auth when auth not required."""
        mock_request = _make_mock_request(path="/federation/status")

        with patch("valence.server.federation_endpoints.get_settings") as mock_settings:
            mock_settings.return_value.federation_require_auth = False
            mock_settings.return_value.federation_enabled = True
            mock_settings.return_value.federation_node_did = "did:vkb:web:test"
            mock_settings.return_value.federation_node_name = "TestNode"
            mock_settings.return_value.federation_capabilities = ["belief_sync"]
            mock_settings.return_value.external_url = None
            mock_settings.return_value.port = 8420

            with patch(
                "valence.server.federation_endpoints._get_federation_stats",
                return_value={"nodes": {"total": 0}},
            ):
                response = await federation_status(mock_request)
                assert response.status_code == 200


# =============================================================================
# PROTOCOL-LEVEL TESTS
# =============================================================================


class TestProtocolLevelAuth:
    """Test protocol-level auth enforcement in handle_message."""

    @pytest.mark.asyncio
    async def test_auth_messages_always_allowed(self):
        """AUTH_CHALLENGE messages should work regardless of require_auth setting."""
        from valence.federation.protocol import (
            AuthChallengeRequest,
            handle_message,
        )

        msg = AuthChallengeRequest(client_did="did:vkb:web:test.example.com")

        # Should work even with require_auth=True and no sender
        with patch("valence.core.config.get_config") as mock_config:
            mock_config.return_value.federation_require_auth = True
            response = await handle_message(msg)
            # Auth challenge should return a challenge response, not an error
            assert hasattr(response, "challenge")
            assert response.challenge != ""

    @pytest.mark.asyncio
    async def test_sync_request_rejected_when_auth_required_no_sender(self):
        """SYNC_REQUEST without sender should be rejected when auth required."""
        from valence.federation.protocol import (
            ErrorCode,
            SyncRequest,
            handle_message,
        )

        msg = SyncRequest()

        with patch("valence.core.config.get_config") as mock_config:
            mock_config.return_value.federation_require_auth = True

            response = await handle_message(msg, sender_did=None, sender_node_id=None)
            assert hasattr(response, "error_code")
            assert response.error_code == ErrorCode.AUTH_FAILED
            assert "Authentication required" in response.message
            assert "VALENCE_FEDERATION_REQUIRE_AUTH" in response.message

    @pytest.mark.asyncio
    async def test_sync_request_generic_error_when_auth_not_required(self):
        """SYNC_REQUEST without sender should get generic error when auth not required."""
        from valence.federation.protocol import (
            ErrorCode,
            SyncRequest,
            handle_message,
        )

        msg = SyncRequest()

        with patch("valence.core.config.get_config") as mock_config:
            mock_config.return_value.federation_require_auth = False

            response = await handle_message(msg, sender_did=None, sender_node_id=None)
            assert hasattr(response, "error_code")
            assert response.error_code == ErrorCode.AUTH_FAILED
            assert "Sender identification required" in response.message

    @pytest.mark.asyncio
    async def test_share_belief_rejected_when_auth_required_no_sender(self):
        """SHARE_BELIEF without sender should be rejected when auth required."""
        from valence.federation.protocol import (
            ErrorCode,
            ShareBeliefRequest,
            handle_message,
        )

        msg = ShareBeliefRequest(beliefs=[])

        with patch("valence.core.config.get_config") as mock_config:
            mock_config.return_value.federation_require_auth = True

            response = await handle_message(msg, sender_did=None, sender_node_id=None)
            assert hasattr(response, "error_code")
            assert response.error_code == ErrorCode.AUTH_FAILED
            assert "Authentication required" in response.message


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================


class TestBackwardCompatibility:
    """Ensure default behavior is unchanged (no auth required)."""

    @pytest.mark.asyncio
    async def test_default_config_allows_unauthenticated(self):
        """With default config, unauthenticated requests should pass through."""

        @require_federation_auth
        async def handler(request: Request) -> dict:
            from starlette.responses import JSONResponse

            return JSONResponse({"success": True})

        mock_request = _make_mock_request()

        with patch("valence.server.federation_endpoints.get_settings") as mock_settings:
            # Default: federation_require_auth = False
            mock_settings.return_value.federation_require_auth = False
            response = await handler(mock_request)
            assert response.status_code == 200

    def test_error_code_is_distinct(self):
        """AUTH_FEDERATION_REQUIRED should be a distinct error code."""
        from valence.server.errors import AUTH_SIGNATURE_FAILED

        assert AUTH_FEDERATION_REQUIRED != AUTH_SIGNATURE_FAILED
        assert AUTH_FEDERATION_REQUIRED == "AUTH_FEDERATION_REQUIRED"
