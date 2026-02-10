"""Comprehensive OAuth 2.1 and Auth test coverage.

This module provides additional test coverage for Issue #129:
1. Token issuance flow (end-to-end)
2. Token refresh
3. Token validation
4. PKCE verification
5. Error cases (invalid client, expired token, etc.)

Uses pytest fixtures to mock OAuth state.
"""

from __future__ import annotations

import base64
import hashlib
import time
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_all_stores():
    """Reset all OAuth and auth stores between tests."""
    import valence.server.auth as auth_module
    import valence.server.config as config_module
    import valence.server.oauth_models as oauth_module

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
def oauth_config(monkeypatch, tmp_path):
    """Configure OAuth environment for tests."""
    token_file = tmp_path / "tokens.json"
    clients_file = tmp_path / "clients.json"

    token_file.write_text('{"tokens": []}')
    clients_file.write_text('{"clients": []}')

    monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("VALENCE_OAUTH_ENABLED", "true")
    monkeypatch.setenv("VALENCE_OAUTH_CLIENTS_FILE", str(clients_file))
    monkeypatch.setenv(
        "VALENCE_OAUTH_JWT_SECRET",
        "test-jwt-secret-for-testing-must-be-at-least-32-chars",
    )
    monkeypatch.setenv("VALENCE_OAUTH_USERNAME", "admin")
    monkeypatch.setenv("VALENCE_OAUTH_PASSWORD", "testpass")
    monkeypatch.setenv("VALENCE_EXTERNAL_URL", "http://localhost:8420")
    monkeypatch.setenv("VALENCE_FEDERATION_ENABLED", "false")
    monkeypatch.setenv("VALENCE_OAUTH_ACCESS_TOKEN_EXPIRY", "3600")
    monkeypatch.setenv("VALENCE_OAUTH_REFRESH_TOKEN_EXPIRY", "2592000")
    monkeypatch.setenv("VALENCE_OAUTH_CODE_EXPIRY", "600")

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

    with patch("oro_db.get_cursor", mock_context):
        yield mock_cursor


@pytest.fixture
def client(oauth_config, mock_db) -> TestClient:
    """Create test client with OAuth configured."""
    from valence.server.app import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


def generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code_verifier and code_challenge pair."""
    import secrets

    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")

    return verifier, challenge


API_V1 = "/api/v1"


# ============================================================================
# End-to-End Token Issuance Flow Tests
# ============================================================================


class TestE2ETokenIssuance:
    """End-to-end tests for complete OAuth token issuance flow."""

    def test_complete_oauth_flow(self, client):
        """Test complete OAuth 2.1 flow from registration to token use."""
        # Step 1: Dynamic Client Registration
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={
                "redirect_uris": ["http://localhost:3000/callback"],
                "client_name": "Test Application",
                "grant_types": ["authorization_code", "refresh_token"],
                "scope": "mcp:tools mcp:resources",
            },
        )
        assert reg_response.status_code == 201
        client_id = reg_response.json()["client_id"]

        # Step 2: Generate PKCE pair
        verifier, challenge = generate_pkce_pair()

        # Step 3: Authorization request (shows login page)
        auth_response = client.get(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost:3000/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "scope": "mcp:tools",
                "state": "random-state-value",
            },
        )
        assert auth_response.status_code == 200
        assert "Valence" in auth_response.text

        # Step 4: Submit credentials
        auth_submit = client.post(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost:3000/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "scope": "mcp:tools",
                "state": "random-state-value",
            },
            data={
                "username": "admin",
                "password": "testpass",
            },
            follow_redirects=False,
        )
        assert auth_submit.status_code == 302
        location = auth_submit.headers["location"]
        assert "code=" in location
        assert "state=random-state-value" in location

        # Extract authorization code
        parsed = urllib.parse.urlparse(location)
        params = urllib.parse.parse_qs(parsed.query)
        code = params["code"][0]

        # Step 5: Exchange code for tokens
        token_response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://localhost:3000/callback",
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )
        assert token_response.status_code == 200
        tokens = token_response.json()

        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "Bearer"
        assert tokens["expires_in"] == 3600
        assert tokens["scope"] == "mcp:tools"

    def test_oauth_flow_preserves_scope(self, client):
        """Test that requested scope is preserved through the flow."""
        # Register client with wide scope support
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={
                "redirect_uris": ["http://localhost/callback"],
                "scope": "mcp:tools mcp:resources mcp:prompts",
            },
        )
        client_id = reg_response.json()["client_id"]

        verifier, challenge = generate_pkce_pair()

        # Request specific scope
        auth_response = client.post(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "scope": "mcp:resources",
            },
            data={"username": "admin", "password": "testpass"},
            follow_redirects=False,
        )

        code = urllib.parse.parse_qs(urllib.parse.urlparse(auth_response.headers["location"]).query)["code"][0]

        token_response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://localhost/callback",
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )

        assert token_response.json()["scope"] == "mcp:resources"


# ============================================================================
# Token Refresh Tests
# ============================================================================


class TestTokenRefresh:
    """Tests for OAuth token refresh functionality."""

    def _get_tokens(self, client) -> dict:
        """Helper to get initial access and refresh tokens."""
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
        )
        client_id = reg_response.json()["client_id"]

        verifier, challenge = generate_pkce_pair()

        auth_response = client.post(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            data={"username": "admin", "password": "testpass"},
            follow_redirects=False,
        )

        code = urllib.parse.parse_qs(urllib.parse.urlparse(auth_response.headers["location"]).query)["code"][0]

        token_response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://localhost/callback",
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )

        return {
            "client_id": client_id,
            **token_response.json(),
        }

    def test_refresh_token_issues_new_access_token(self, client):
        """Test that refresh token grants new access token."""
        tokens = self._get_tokens(client)
        original_access_token = tokens["access_token"]

        refresh_response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": tokens["client_id"],
            },
        )

        assert refresh_response.status_code == 200
        new_tokens = refresh_response.json()

        assert "access_token" in new_tokens
        assert new_tokens["access_token"] != original_access_token
        assert new_tokens["token_type"] == "Bearer"

    def test_refresh_preserves_scope(self, client):
        """Test that refreshed token preserves original scope."""
        tokens = self._get_tokens(client)
        original_scope = tokens["scope"]

        refresh_response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
            },
        )

        assert refresh_response.json()["scope"] == original_scope

    def test_refresh_without_client_id(self, client):
        """Test refresh works without client_id (uses token's client_id)."""
        tokens = self._get_tokens(client)

        refresh_response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                # No client_id provided
            },
        )

        assert refresh_response.status_code == 200

    def test_refresh_with_wrong_client_id(self, client):
        """Test refresh fails with mismatched client_id."""
        tokens = self._get_tokens(client)

        refresh_response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": "different-client-id",
            },
        )

        assert refresh_response.status_code == 400
        assert refresh_response.json()["error"] == "invalid_grant"


# ============================================================================
# Token Validation Tests
# ============================================================================


class TestTokenValidation:
    """Tests for access token validation."""

    @pytest.fixture(autouse=True)
    def setup_jwt_config(self, monkeypatch):
        """Set up JWT configuration for tests."""
        monkeypatch.setenv(
            "VALENCE_OAUTH_JWT_SECRET",
            "test-secret-for-jwt-testing-must-be-at-least-32-chars",
        )
        monkeypatch.setenv("VALENCE_EXTERNAL_URL", "http://localhost:8420")
        monkeypatch.setenv("VALENCE_OAUTH_ACCESS_TOKEN_EXPIRY", "3600")

        import valence.server.config as config_module

        config_module._settings = None
        yield
        config_module._settings = None

    def test_valid_token_contains_claims(self):
        """Test valid token contains expected claims."""
        from valence.server.oauth_models import create_access_token, verify_access_token

        token = create_access_token(
            client_id="test-client",
            user_id="test-user",
            scope="mcp:tools mcp:resources",
            audience="http://localhost:8420/api/v1/mcp",
        )

        payload = verify_access_token(token, "http://localhost:8420/api/v1/mcp")

        assert payload is not None
        assert payload["client_id"] == "test-client"
        assert payload["sub"] == "test-user"
        assert payload["scope"] == "mcp:tools mcp:resources"
        assert "iat" in payload
        assert "exp" in payload
        assert "jti" in payload  # Token ID for revocation

    def test_token_with_wrong_issuer_rejected(self, monkeypatch):
        """Test token with wrong issuer is rejected."""
        import valence.server.config as config_module
        from valence.server.oauth_models import create_access_token, verify_access_token

        # Create token with one issuer
        monkeypatch.setenv("VALENCE_EXTERNAL_URL", "http://original-issuer.com")
        config_module._settings = None

        token = create_access_token(
            client_id="test-client",
            user_id="test-user",
            scope="mcp:tools",
            audience="http://original-issuer.com/api/v1/mcp",
        )

        # Change issuer and try to verify
        monkeypatch.setenv("VALENCE_EXTERNAL_URL", "http://different-issuer.com")
        config_module._settings = None

        payload = verify_access_token(token, "http://original-issuer.com/api/v1/mcp")

        assert payload is None  # Should reject

    def test_tampered_token_rejected(self):
        """Test tampered token is rejected."""
        from valence.server.oauth_models import create_access_token, verify_access_token

        token = create_access_token(
            client_id="test-client",
            user_id="test-user",
            scope="mcp:tools",
            audience="http://localhost:8420/api/v1/mcp",
        )

        # Tamper with token payload
        parts = token.split(".")
        # Modify payload (base64url decode, change, re-encode)
        import base64

        payload = base64.urlsafe_b64decode(parts[1] + "==")
        # Just flip a byte
        tampered_payload = payload[:10] + bytes([payload[10] ^ 0xFF]) + payload[11:]
        parts[1] = base64.urlsafe_b64encode(tampered_payload).rstrip(b"=").decode()
        tampered_token = ".".join(parts)

        result = verify_access_token(tampered_token, "http://localhost:8420/api/v1/mcp")

        assert result is None


# ============================================================================
# PKCE Verification Tests
# ============================================================================


class TestPKCEVerification:
    """Tests for PKCE code challenge verification."""

    def test_pkce_valid_s256(self):
        """Test valid S256 PKCE verification."""
        from valence.server.oauth_models import verify_pkce

        # RFC 7636 example values
        verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")

        assert verify_pkce(verifier, challenge, "S256") is True

    def test_pkce_invalid_verifier(self):
        """Test PKCE fails with wrong verifier."""
        from valence.server.oauth_models import verify_pkce

        verifier = "correct-verifier"
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")

        assert verify_pkce("wrong-verifier", challenge, "S256") is False

    def test_pkce_rejects_plain_method(self):
        """Test PKCE rejects insecure plain method."""
        from valence.server.oauth_models import verify_pkce

        assert verify_pkce("verifier", "verifier", "plain") is False

    def test_pkce_rejects_unknown_method(self):
        """Test PKCE rejects unknown challenge method."""
        from valence.server.oauth_models import verify_pkce

        assert verify_pkce("verifier", "challenge", "SHA512") is False

    def test_pkce_required_in_flow(self, client):
        """Test that authorization fails without PKCE."""
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
                # Missing code_challenge
            },
            follow_redirects=False,
        )

        assert response.status_code == 302
        location = response.headers.get("location", "")
        assert "error" in location or "PKCE" in location.lower() or "code_challenge" in location.lower()


# ============================================================================
# Error Cases
# ============================================================================


class TestErrorCases:
    """Tests for OAuth error handling."""

    def test_invalid_client_id(self, client):
        """Test authorization fails with invalid client_id."""
        _, challenge = generate_pkce_pair()

        response = client.get(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": "nonexistent-client-id",
                "redirect_uri": "http://localhost/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
        )

        assert response.status_code == 400

    def test_expired_authorization_code(self, client, oauth_config):
        """Test token exchange fails with expired authorization code."""
        from valence.server.oauth_models import get_code_store

        # Register client and get auth code
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
        )
        client_id = reg_response.json()["client_id"]

        verifier, challenge = generate_pkce_pair()

        auth_response = client.post(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            data={"username": "admin", "password": "testpass"},
            follow_redirects=False,
        )

        code = urllib.parse.parse_qs(urllib.parse.urlparse(auth_response.headers["location"]).query)["code"][0]

        # Manually expire the code
        code_store = get_code_store()
        if code in code_store._codes:
            code_store._codes[code].expires_at = time.time() - 100

        # Try to exchange expired code
        token_response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://localhost/callback",
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )

        assert token_response.status_code == 400
        assert token_response.json()["error"] == "invalid_grant"

    def test_redirect_uri_mismatch(self, client):
        """Test token exchange fails with redirect_uri mismatch."""
        reg_response = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
        )
        client_id = reg_response.json()["client_id"]

        verifier, challenge = generate_pkce_pair()

        auth_response = client.post(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            data={"username": "admin", "password": "testpass"},
            follow_redirects=False,
        )

        code = urllib.parse.parse_qs(urllib.parse.urlparse(auth_response.headers["location"]).query)["code"][0]

        # Try to exchange with different redirect_uri
        token_response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://different-uri.com/callback",
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )

        assert token_response.status_code == 400
        assert token_response.json()["error"] == "invalid_grant"

    def test_client_id_mismatch_in_token_exchange(self, client):
        """Test token exchange fails with client_id mismatch."""
        # Register two clients
        reg1 = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
        )
        client1_id = reg1.json()["client_id"]

        reg2 = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
        )
        client2_id = reg2.json()["client_id"]

        verifier, challenge = generate_pkce_pair()

        # Get code for client1
        auth_response = client.post(
            f"{API_V1}/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client1_id,
                "redirect_uri": "http://localhost/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            data={"username": "admin", "password": "testpass"},
            follow_redirects=False,
        )

        code = urllib.parse.parse_qs(urllib.parse.urlparse(auth_response.headers["location"]).query)["code"][0]

        # Try to exchange with client2's ID
        token_response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://localhost/callback",
                "client_id": client2_id,  # Wrong client
                "code_verifier": verifier,
            },
        )

        assert token_response.status_code == 400
        assert token_response.json()["error"] == "invalid_grant"

    def test_invalid_json_in_registration(self, client):
        """Test registration fails with invalid JSON."""
        response = client.post(
            f"{API_V1}/oauth/register",
            content=b"not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        assert "invalid" in response.json()["error_description"].lower()

    def test_empty_redirect_uris(self, client):
        """Test registration fails with empty redirect_uris."""
        response = client.post(
            f"{API_V1}/oauth/register",
            json={"redirect_uris": []},
        )

        assert response.status_code == 400

    def test_invalid_form_data_in_token(self, client):
        """Test token endpoint handles invalid form data gracefully."""
        # Send with wrong content type
        response = client.post(
            f"{API_V1}/oauth/token",
            content=b"\x00\x01\x02invalid",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        # Should not crash, should return error
        assert response.status_code in (400, 422)


# ============================================================================
# Legacy Bearer Token Tests
# ============================================================================


class TestLegacyBearerTokens:
    """Tests for legacy Bearer token authentication (auth.py)."""

    @pytest.fixture
    def temp_token_file(self, tmp_path) -> Path:
        """Create a temporary token file."""
        token_file = tmp_path / "tokens.json"
        token_file.write_text('{"tokens": []}')
        return token_file

    def test_token_creation_and_verification(self, temp_token_file):
        """Test creating and verifying a bearer token."""
        from valence.server.auth import TokenStore

        store = TokenStore(temp_token_file)
        raw_token = store.create(
            client_id="test-client",
            description="Test token",
            scopes=["mcp:access", "mcp:admin"],
        )

        token = store.verify(raw_token)

        assert token is not None
        assert token.client_id == "test-client"
        assert token.has_scope("mcp:access")
        assert token.has_scope("mcp:admin")
        assert not token.has_scope("mcp:superadmin")

    def test_token_with_expiry(self, temp_token_file):
        """Test token expiration."""
        from valence.server.auth import TokenStore

        store = TokenStore(temp_token_file)

        # Create expired token
        raw_token = store.create(
            client_id="test-client",
            expires_at=time.time() - 3600,  # Expired 1 hour ago
        )

        token = store.verify(raw_token)

        assert token is None  # Should not verify

    def test_token_revocation(self, temp_token_file):
        """Test token revocation."""
        from valence.server.auth import TokenStore, hash_token

        store = TokenStore(temp_token_file)
        raw_token = store.create(client_id="test-client")
        token_hash = hash_token(raw_token)

        # Verify before revocation
        assert store.verify(raw_token) is not None

        # Revoke
        assert store.revoke(token_hash) is True

        # Verify after revocation
        assert store.verify(raw_token) is None

    def test_verify_token_with_scope_check(self, temp_token_file):
        """Test verify_token function with scope requirements."""
        from unittest.mock import patch

        from valence.server.auth import TokenStore, verify_token

        store = TokenStore(temp_token_file)
        raw_token = store.create(
            client_id="test-client",
            scopes=["mcp:access"],  # Only access scope
        )

        with patch("valence.server.auth.get_token_store", return_value=store):
            # Should pass for mcp:access
            token = verify_token(raw_token, required_scope="mcp:access")
            assert token is not None

            # Should fail for mcp:admin
            token = verify_token(raw_token, required_scope="mcp:admin")
            assert token is None


# ============================================================================
# Expired Refresh Token Tests
# ============================================================================


class TestExpiredRefreshToken:
    """Tests for expired refresh token handling."""

    @pytest.fixture(autouse=True)
    def setup_config(self, monkeypatch):
        """Set up configuration."""
        monkeypatch.setenv(
            "VALENCE_OAUTH_JWT_SECRET",
            "test-secret-for-jwt-testing-must-be-at-least-32-chars",
        )
        monkeypatch.setenv("VALENCE_EXTERNAL_URL", "http://localhost:8420")
        monkeypatch.setenv("VALENCE_OAUTH_REFRESH_TOKEN_EXPIRY", "2592000")

        import valence.server.config as config_module

        config_module._settings = None
        yield
        config_module._settings = None

    def test_expired_refresh_token_rejected(self):
        """Test that expired refresh token is rejected."""
        import hashlib

        from valence.server.oauth_models import RefreshTokenStore

        store = RefreshTokenStore()

        # Create a token
        raw_token = store.create(
            client_id="test-client",
            user_id="test-user",
            scope="mcp:tools",
        )

        # Manually expire it
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        if token_hash in store._tokens:
            store._tokens[token_hash].expires_at = time.time() - 100

        # Try to validate
        result = store.validate(raw_token)

        assert result is None

    def test_expired_refresh_token_cleaned_up(self):
        """Test that expired refresh token is removed from store."""
        import hashlib

        from valence.server.oauth_models import RefreshTokenStore

        store = RefreshTokenStore()

        raw_token = store.create(
            client_id="test-client",
            user_id="test-user",
            scope="mcp:tools",
        )

        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        # Token exists before expiration validation
        assert token_hash in store._tokens

        # Expire and validate
        store._tokens[token_hash].expires_at = time.time() - 100
        store.validate(raw_token)

        # Token should be removed after failed validation
        assert token_hash not in store._tokens
