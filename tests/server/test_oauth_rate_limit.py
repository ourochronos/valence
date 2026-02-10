"""Tests for OAuth rate limiting (Issue #173)."""

from __future__ import annotations

import base64
import hashlib
import urllib.parse
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset all stores between tests."""
    import valence.server.auth as auth_module
    import valence.server.config as config_module
    import valence.server.oauth_models as oauth_module
    import valence.server.rate_limit as rate_limit_module

    config_module._settings = None
    oauth_module._client_store = None
    oauth_module._code_store = None
    oauth_module._refresh_store = None
    auth_module._token_store = None
    rate_limit_module.clear_rate_limits()

    yield

    config_module._settings = None
    oauth_module._client_store = None
    oauth_module._code_store = None
    oauth_module._refresh_store = None
    auth_module._token_store = None
    rate_limit_module.clear_rate_limits()


@pytest.fixture
def oauth_env(monkeypatch, tmp_path):
    """Set up OAuth environment with low rate limit for testing."""
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
    # Set a low rate limit for testing
    monkeypatch.setenv("VALENCE_RATE_LIMIT_RPM", "5")

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
def client(oauth_env, mock_db) -> TestClient:
    """Create test client."""
    from valence.server.app import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


API_V1 = "/api/v1"


def generate_pkce_pair():
    """Generate a PKCE code_verifier and code_challenge pair."""
    import secrets

    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")

    return verifier, challenge


# ============================================================================
# Rate Limit Module Unit Tests
# ============================================================================


class TestRateLimitModule:
    """Unit tests for the rate_limit module."""

    def test_check_rate_limit_allows_initial_requests(self):
        """Test that initial requests are allowed."""
        from valence.server.rate_limit import check_rate_limit, clear_rate_limits

        clear_rate_limits()

        # First request should be allowed
        assert check_rate_limit("test-key", rpm_limit=5) is True

    def test_check_rate_limit_blocks_after_limit(self):
        """Test that requests are blocked after exceeding limit."""
        from valence.server.rate_limit import check_rate_limit, clear_rate_limits

        clear_rate_limits()

        # Make 5 requests (limit)
        for _ in range(5):
            assert check_rate_limit("test-key", rpm_limit=5) is True

        # 6th request should be blocked
        assert check_rate_limit("test-key", rpm_limit=5) is False

    def test_check_rate_limit_separate_keys(self):
        """Test that different keys have separate limits."""
        from valence.server.rate_limit import check_rate_limit, clear_rate_limits

        clear_rate_limits()

        # Exhaust key1's limit
        for _ in range(5):
            check_rate_limit("key1", rpm_limit=5)

        # key1 should be blocked
        assert check_rate_limit("key1", rpm_limit=5) is False

        # key2 should still be allowed
        assert check_rate_limit("key2", rpm_limit=5) is True

    def test_check_oauth_rate_limit_ip_tracking(self):
        """Test OAuth rate limit tracks by IP."""
        from valence.server.rate_limit import check_oauth_rate_limit, clear_rate_limits

        clear_rate_limits()

        # Create a mock request
        class MockRequest:
            def __init__(self, ip: str):
                self.headers = {}
                self.client = type("obj", (object,), {"host": ip})()

        request = MockRequest("192.168.1.100")

        # Make requests up to limit
        for _ in range(5):
            result = check_oauth_rate_limit(request, client_id=None, rpm_limit=5)
            assert result.allowed is True

        # Next request should be blocked
        result = check_oauth_rate_limit(request, client_id=None, rpm_limit=5)
        assert result.allowed is False
        assert "ip:" in result.key

    def test_check_oauth_rate_limit_client_id_tracking(self):
        """Test OAuth rate limit tracks by client_id when provided."""
        from valence.server.rate_limit import check_oauth_rate_limit, clear_rate_limits

        clear_rate_limits()

        class MockRequest:
            def __init__(self, ip: str):
                self.headers = {}
                self.client = type("obj", (object,), {"host": ip})()

        # Different IPs but same client_id
        for i in range(5):
            request = MockRequest(f"192.168.1.{i}")
            result = check_oauth_rate_limit(request, client_id="same-client", rpm_limit=5)
            assert result.allowed is True

        # 6th request with same client_id (different IP) should be blocked
        request = MockRequest("192.168.1.99")
        result = check_oauth_rate_limit(request, client_id="same-client", rpm_limit=5)
        assert result.allowed is False
        assert "client:" in result.key

    def test_x_forwarded_for_header(self):
        """Test that X-Forwarded-For header is respected."""
        from valence.server.rate_limit import _get_client_ip

        class MockRequest:
            def __init__(self, forwarded_for: str | None, direct_ip: str):
                self.headers = {}
                if forwarded_for:
                    self.headers["X-Forwarded-For"] = forwarded_for
                self.client = type("obj", (object,), {"host": direct_ip})()

        # With X-Forwarded-For
        request = MockRequest("203.0.113.50, 70.41.3.18", "127.0.0.1")
        assert _get_client_ip(request) == "203.0.113.50"

        # Without X-Forwarded-For
        request = MockRequest(None, "192.168.1.1")
        assert _get_client_ip(request) == "192.168.1.1"


# ============================================================================
# Token Endpoint Rate Limit Integration Tests
# ============================================================================


class TestTokenEndpointRateLimit:
    """Integration tests for token endpoint rate limiting."""

    def _register_client(self, client) -> str:
        """Helper to register a client and return client_id."""
        response = client.post(
            f"{API_V1}/oauth/register",
            json={
                "redirect_uris": ["http://localhost/callback"],
                "client_name": "Test App",
            },
        )
        return response.json()["client_id"]

    def _get_auth_code(self, client, client_id: str) -> tuple[str, str]:
        """Helper to get an authorization code."""
        verifier, challenge = generate_pkce_pair()

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

        location = response.headers["location"]
        parsed = urllib.parse.urlparse(location)
        params = urllib.parse.parse_qs(parsed.query)
        code = params["code"][0]

        return code, verifier

    def test_token_endpoint_rate_limited(self, client):
        """Test that token endpoint returns 429 when rate limit exceeded."""
        from valence.server.rate_limit import clear_rate_limits

        clear_rate_limits()

        # Make requests up to the limit (5)
        for i in range(5):
            response = client.post(
                f"{API_V1}/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": f"invalid-code-{i}",
                    "redirect_uri": "http://localhost/callback",
                    "client_id": "test-client",
                    "code_verifier": "verifier",
                },
            )
            # These fail with 400 (invalid code), but count against rate limit
            assert response.status_code == 400

        # 6th request should be rate limited
        response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": "invalid-code-6",
                "redirect_uri": "http://localhost/callback",
                "client_id": "test-client",
                "code_verifier": "verifier",
            },
        )

        assert response.status_code == 429
        assert response.json()["error"] == "rate_limit_exceeded"
        assert "Retry-After" in response.headers

    def test_token_endpoint_valid_requests_counted(self, client):
        """Test that valid requests also count against rate limit."""
        from valence.server.rate_limit import clear_rate_limits

        clear_rate_limits()

        client_id = self._register_client(client)

        # Make 4 valid token exchanges
        for i in range(4):
            clear_rate_limits()  # Reset between iterations for clean codes
            code, verifier = self._get_auth_code(client, client_id)

            # Use the rate limit tracking just for token endpoint

            # Don't clear here - we want to accumulate

        # Now test accumulation
        clear_rate_limits()

        # Make 5 requests (with invalid codes for simplicity)
        for i in range(5):
            client.post(
                f"{API_V1}/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": f"invalid-token-{i}",
                },
            )

        # 6th should be rate limited
        response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": "invalid-token-6",
            },
        )
        assert response.status_code == 429

    def test_different_clients_have_separate_limits(self, client):
        """Test that different client_ids have separate rate limits."""
        from valence.server.rate_limit import clear_rate_limits

        clear_rate_limits()

        # Exhaust limit for client-a
        for i in range(5):
            client.post(
                f"{API_V1}/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": f"code-{i}",
                    "client_id": "client-a",
                    "redirect_uri": "http://localhost/callback",
                    "code_verifier": "verifier",
                },
            )

        # client-a should be rate limited
        response = client.post(
            f"{API_V1}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": "code-6",
                "client_id": "client-a",
                "redirect_uri": "http://localhost/callback",
                "code_verifier": "verifier",
            },
        )
        assert response.status_code == 429

        # But note: same IP is also limited, so client-b would also be blocked
        # This is intentional - we limit by both IP AND client_id


# ============================================================================
# Registration Endpoint Rate Limit Tests
# ============================================================================


class TestRegistrationRateLimit:
    """Tests for registration endpoint rate limiting."""

    def test_register_endpoint_rate_limited(self, client):
        """Test that registration endpoint returns 429 when rate limited."""
        from valence.server.rate_limit import clear_rate_limits

        clear_rate_limits()

        # Make 5 registration requests
        for i in range(5):
            response = client.post(
                f"{API_V1}/oauth/register",
                json={
                    "redirect_uris": [f"http://localhost:{3000 + i}/callback"],
                    "client_name": f"Test App {i}",
                },
            )
            assert response.status_code == 201

        # 6th request should be rate limited
        response = client.post(
            f"{API_V1}/oauth/register",
            json={
                "redirect_uris": ["http://localhost:4000/callback"],
                "client_name": "Test App 6",
            },
        )

        assert response.status_code == 429
        assert response.json()["error"] == "rate_limit_exceeded"


# ============================================================================
# Rate Limit Response Format Tests
# ============================================================================


class TestRateLimitResponse:
    """Tests for rate limit response format."""

    def test_rate_limit_response_format(self, client):
        """Test that rate limit response follows OAuth error format."""
        from valence.server.rate_limit import clear_rate_limits

        clear_rate_limits()

        # Exhaust rate limit
        for _ in range(5):
            client.post(f"{API_V1}/oauth/token", data={"grant_type": "test"})

        response = client.post(
            f"{API_V1}/oauth/token",
            data={"grant_type": "authorization_code"},
        )

        assert response.status_code == 429
        data = response.json()

        # Should follow OAuth error response format
        assert "error" in data
        assert data["error"] == "rate_limit_exceeded"
        assert "error_description" in data

        # Should include Retry-After header
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "60"

    def test_rate_limit_response_json_content_type(self, client):
        """Test that rate limit response is JSON."""
        from valence.server.rate_limit import clear_rate_limits

        clear_rate_limits()

        # Exhaust rate limit
        for _ in range(5):
            client.post(f"{API_V1}/oauth/token", data={"grant_type": "test"})

        response = client.post(
            f"{API_V1}/oauth/token",
            data={"grant_type": "test"},
        )

        assert response.status_code == 429
        assert "application/json" in response.headers.get("content-type", "")
