"""Tests for auth_helpers module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from valence.server.auth_helpers import AuthenticatedClient, authenticate, require_scope


@pytest.fixture
def mock_request():
    """Create a mock request with an auth header."""

    def _make(auth_header: str = "") -> MagicMock:
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": auth_header} if auth_header else {}
        return request

    return _make


class TestAuthenticate:
    """Tests for the authenticate() function."""

    def test_missing_auth_header_returns_401(self, mock_request):
        request = mock_request("")
        result = authenticate(request)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 401

    def test_invalid_auth_scheme_returns_401(self, mock_request):
        request = mock_request("Basic dXNlcjpwYXNz")
        result = authenticate(request)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 401

    @patch("valence.server.auth_helpers.verify_token")
    def test_valid_bearer_token_returns_client(self, mock_verify, mock_request):
        mock_token = MagicMock()
        mock_token.client_id = "test-client"
        mock_verify.return_value = mock_token

        request = mock_request("Bearer vt_abc123")
        result = authenticate(request)
        assert isinstance(result, AuthenticatedClient)
        assert result.client_id == "test-client"
        assert result.auth_method == "bearer"

    @patch("valence.server.auth_helpers.verify_access_token")
    @patch("valence.server.auth_helpers.verify_token")
    @patch("valence.server.auth_helpers.get_settings")
    def test_valid_oauth_token_returns_client(self, mock_settings, mock_verify_token, mock_verify_jwt, mock_request):
        mock_verify_token.return_value = None
        mock_settings.return_value.oauth_enabled = True
        mock_settings.return_value.mcp_resource_url = "http://localhost:8420"
        mock_verify_jwt.return_value = {"client_id": "oauth-client", "sub": "user1", "scope": "substrate:read vkb:read"}

        request = mock_request("Bearer eyJhbGciOiJIUzI1NiJ9.test")
        result = authenticate(request)
        assert isinstance(result, AuthenticatedClient)
        assert result.client_id == "oauth-client"
        assert result.auth_method == "oauth"
        assert result.scope == "substrate:read vkb:read"

    @patch("valence.server.auth_helpers.verify_token")
    def test_invalid_token_returns_401(self, mock_verify, mock_request):
        mock_verify.return_value = None

        with patch("valence.server.auth_helpers.get_settings") as mock_settings:
            mock_settings.return_value.oauth_enabled = False
            request = mock_request("Bearer invalid_token")
            result = authenticate(request)
            assert isinstance(result, JSONResponse)
            assert result.status_code == 401


class TestRequireScope:
    """Tests for the require_scope() function."""

    def test_bearer_token_always_passes(self):
        client = AuthenticatedClient(client_id="test", auth_method="bearer")
        assert require_scope(client, "substrate:read") is None
        assert require_scope(client, "vkb:write") is None

    def test_oauth_with_matching_scope_passes(self):
        client = AuthenticatedClient(client_id="test", auth_method="oauth", scope="substrate:read vkb:read")
        assert require_scope(client, "substrate:read") is None
        assert require_scope(client, "vkb:read") is None

    def test_oauth_with_mcp_access_passes_all(self):
        client = AuthenticatedClient(client_id="test", auth_method="oauth", scope="mcp:access")
        assert require_scope(client, "substrate:read") is None
        assert require_scope(client, "substrate:write") is None
        assert require_scope(client, "vkb:read") is None

    def test_oauth_missing_scope_returns_403(self):
        client = AuthenticatedClient(client_id="test", auth_method="oauth", scope="substrate:read")
        result = require_scope(client, "substrate:write")
        assert isinstance(result, JSONResponse)
        assert result.status_code == 403

    def test_oauth_no_scope_returns_403(self):
        client = AuthenticatedClient(client_id="test", auth_method="oauth", scope=None)
        result = require_scope(client, "substrate:read")
        assert isinstance(result, JSONResponse)
        assert result.status_code == 403
