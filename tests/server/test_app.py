"""Tests for the main Starlette application."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset all stores between tests."""
    import valence.server.app as app_module
    import valence.server.auth as auth_module
    import valence.server.config as config_module
    import valence.server.oauth_models as oauth_module

    config_module._settings = None
    oauth_module._client_store = None
    oauth_module._code_store = None
    oauth_module._refresh_store = None
    auth_module._token_store = None
    # Reset rate limits
    app_module._rate_limits.clear()

    yield

    config_module._settings = None
    oauth_module._client_store = None
    oauth_module._code_store = None
    oauth_module._refresh_store = None
    auth_module._token_store = None
    app_module._rate_limits.clear()


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    """Set up application environment."""
    token_file = tmp_path / "tokens.json"
    clients_file = tmp_path / "clients.json"

    token_file.write_text('{"tokens": []}')
    clients_file.write_text('{"clients": []}')

    monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("VALENCE_OAUTH_ENABLED", "true")
    monkeypatch.setenv("VALENCE_OAUTH_CLIENTS_FILE", str(clients_file))
    # JWT secret must be at least 32 characters
    monkeypatch.setenv(
        "VALENCE_OAUTH_JWT_SECRET",
        "test-jwt-secret-for-testing-must-be-at-least-32-chars",
    )
    monkeypatch.setenv("VALENCE_OAUTH_USERNAME", "admin")
    monkeypatch.setenv("VALENCE_OAUTH_PASSWORD", "testpass")
    monkeypatch.setenv("VALENCE_EXTERNAL_URL", "http://localhost:8420")
    monkeypatch.setenv("VALENCE_FEDERATION_ENABLED", "false")
    monkeypatch.setenv("VALENCE_RATE_LIMIT_RPM", "60")
    monkeypatch.setenv("VALENCE_ALLOWED_ORIGINS", '["http://example.com"]')

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

    with patch("our_db.get_cursor", mock_context):
        yield mock_cursor


@pytest.fixture
def client(app_env, mock_db) -> TestClient:
    """Create test client."""
    from valence.server.app import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_token(app_env) -> str:
    """Create a valid authentication token."""
    from valence.server.auth import get_token_store

    store = get_token_store(app_env["token_file"])
    return store.create(client_id="test-client")


# ============================================================================
# Health Endpoint Tests
# ============================================================================

# API prefix for v1 endpoints
API_V1 = "/api/v1"


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_healthy(self, client):
        """Test healthy health check response."""
        response = client.get(f"{API_V1}/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "server" in data
        assert "version" in data

    def test_health_check_includes_db_status(self, client):
        """Test health check includes database status."""
        response = client.get(f"{API_V1}/health")
        data = response.json()

        assert "database" in data

    def test_health_check_degraded_on_db_error(self, app_env, tmp_path):
        """Test health check returns degraded on database error."""
        # Set up files
        token_file = tmp_path / "tokens.json"
        clients_file = tmp_path / "clients.json"
        token_file.write_text('{"tokens": []}')
        clients_file.write_text('{"clients": []}')

        # Mock database error
        def mock_context(*args, **kwargs):
            class CM:
                def __enter__(self):
                    raise Exception("Database connection failed")

                def __exit__(self, *args):
                    pass

            return CM()

        # Patch at valence.core.db where get_cursor is defined
        with patch("our_db.get_cursor", mock_context):
            from valence.server.app import create_app

            app = create_app()
            test_client = TestClient(app, raise_server_exceptions=False)

            response = test_client.get(f"{API_V1}/health")
            data = response.json()

            assert data["status"] == "degraded"
            assert response.status_code == 503


# ============================================================================
# Info Endpoint Tests
# ============================================================================


class TestInfoEndpoint:
    """Tests for server info endpoint."""

    def test_info_no_auth_required(self, client):
        """Test info endpoint doesn't require auth."""
        response = client.get("/")

        assert response.status_code == 200

    def test_info_returns_server_details(self, client):
        """Test info endpoint returns server details."""
        response = client.get("/")
        data = response.json()

        assert "server" in data
        assert "version" in data
        assert "protocol" in data
        assert data["protocol"] == "mcp"
        assert "tools" in data
        assert "endpoints" in data

    def test_info_shows_oauth_when_enabled(self, client):
        """Test info shows OAuth info when enabled."""
        response = client.get("/")
        data = response.json()

        assert "oauth2" in data["authentication"]["methods"]
        assert "oauth" in data["authentication"]


# ============================================================================
# MCP Endpoint Tests
# ============================================================================


class TestMCPEndpoint:
    """Tests for MCP JSON-RPC endpoint."""

    def test_mcp_requires_auth(self, client):
        """Test MCP endpoint requires authentication."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "ping",
                "id": 1,
            },
        )

        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_mcp_accepts_bearer_token(self, client, auth_token):
        """Test MCP accepts Bearer token authentication."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "ping",
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200

    def test_mcp_ping(self, client, auth_token):
        """Test MCP ping method."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "ping",
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "result" in data

    def test_mcp_initialize(self, client, auth_token):
        """Test MCP initialize method."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {},
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        result = data["result"]

        assert "protocolVersion" in result
        assert "capabilities" in result
        assert "serverInfo" in result
        assert "instructions" in result

    def test_mcp_initialized(self, client, auth_token):
        """Test MCP initialized notification."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "initialized",
                "params": {},
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200

    def test_mcp_tools_list(self, client, auth_token):
        """Test MCP tools/list method."""
        # Test that tools/list returns a valid response
        # The actual tool list depends on what's available in the environment
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "tools" in data["result"]
        # tools can be empty if the modules aren't fully configured
        assert isinstance(data["result"]["tools"], list)

    def test_mcp_resources_list(self, client, auth_token):
        """Test MCP resources/list method."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "resources/list",
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "resources" in data["result"]

    def test_mcp_resources_read_instructions(self, client, auth_token):
        """Test reading instructions resource."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "resources/read",
                "params": {"uri": "valence://instructions"},
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "contents" in data["result"]

    def test_mcp_prompts_list(self, client, auth_token):
        """Test MCP prompts/list method."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/list",
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "prompts" in data["result"]

    def test_mcp_prompts_get_context(self, client, auth_token):
        """Test getting valence-context prompt."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/get",
                "params": {"name": "valence-context"},
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "messages" in data["result"]

    def test_mcp_method_not_found(self, client, auth_token):
        """Test unknown method returns error."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "nonexistent/method",
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32601

    def test_mcp_invalid_json(self, client, auth_token):
        """Test invalid JSON returns parse error."""
        response = client.post(
            f"{API_V1}/mcp",
            content="not valid json",
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == -32700

    def test_mcp_invalid_jsonrpc_version(self, client, auth_token):
        """Test wrong JSON-RPC version returns error."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "1.0",  # Wrong version
                "method": "ping",
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32600

    def test_mcp_notification_no_response(self, client, auth_token):
        """Test notifications (no id) return 204."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "initialized",
                # No id - this is a notification
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 204

    def test_mcp_batch_request(self, client, auth_token):
        """Test batch JSON-RPC request."""
        response = client.post(
            f"{API_V1}/mcp",
            json=[
                {"jsonrpc": "2.0", "method": "ping", "id": 1},
                {"jsonrpc": "2.0", "method": "ping", "id": 2},
            ],
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_under_limit(self, client, auth_token):
        """Test requests under rate limit succeed."""
        for _ in range(5):
            response = client.post(
                f"{API_V1}/mcp",
                json={"jsonrpc": "2.0", "method": "ping", "id": 1},
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            assert response.status_code == 200

    def test_rate_limit_exceeded(self, client, auth_token, monkeypatch):
        """Test rate limit exceeded returns 429."""
        # Set very low rate limit
        monkeypatch.setenv("VALENCE_RATE_LIMIT_RPM", "2")

        # Reset settings to pick up new limit
        import valence.server.config as config_module

        config_module._settings = None

        # Make requests until rate limited
        responses = []
        for _ in range(5):
            response = client.post(
                f"{API_V1}/mcp",
                json={"jsonrpc": "2.0", "method": "ping", "id": 1},
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            responses.append(response)

        # At least one should be rate limited
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes


# ============================================================================
# Authentication Tests
# ============================================================================


class TestAuthentication:
    """Tests for authentication handling."""

    def test_invalid_bearer_token(self, client):
        """Test invalid bearer token is rejected."""
        response = client.post(
            f"{API_V1}/mcp",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response.status_code == 401

    def test_missing_authorization_header(self, client):
        """Test missing auth header is rejected."""
        response = client.post(
            f"{API_V1}/mcp",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
        )

        assert response.status_code == 401

    def test_malformed_authorization_header(self, client):
        """Test malformed auth header is rejected."""
        response = client.post(
            f"{API_V1}/mcp",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
            headers={"Authorization": "NotBearer token"},
        )

        assert response.status_code == 401

    def test_www_authenticate_header(self, client):
        """Test 401 includes WWW-Authenticate header."""
        response = client.post(
            f"{API_V1}/mcp",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
        )

        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
        assert "Bearer" in response.headers["WWW-Authenticate"]


# ============================================================================
# CORS Tests
# ============================================================================


class TestCORS:
    """Tests for CORS middleware."""

    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            f"{API_V1}/mcp",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization",
            },
        )

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    def test_cors_actual_request(self, client, auth_token):
        """Test CORS headers on actual request."""
        response = client.post(
            f"{API_V1}/mcp",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Origin": "http://example.com",
            },
        )

        assert "Access-Control-Allow-Origin" in response.headers


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestCheckRateLimit:
    """Tests for rate limit checking function."""

    def test_check_rate_limit_allow(self):
        """Test rate limit allows under limit."""
        from valence.server.app import _check_rate_limit, _rate_limits

        _rate_limits.clear()

        assert _check_rate_limit("client1", rpm_limit=10) is True
        assert _check_rate_limit("client1", rpm_limit=10) is True

    def test_check_rate_limit_deny(self):
        """Test rate limit denies over limit."""
        from valence.server.app import _check_rate_limit, _rate_limits

        _rate_limits.clear()

        # Make requests up to limit
        for _ in range(5):
            _check_rate_limit("client2", rpm_limit=5)

        # Next should be denied
        assert _check_rate_limit("client2", rpm_limit=5) is False

    def test_check_rate_limit_cleans_old_entries(self):
        """Test rate limit cleans old entries."""
        from valence.server.app import _check_rate_limit, _rate_limits

        _rate_limits.clear()

        # Add old entries manually
        old_time = time.time() - 120  # 2 minutes ago
        _rate_limits["client3"] = [old_time] * 100

        # Should clean old entries and allow new request
        assert _check_rate_limit("client3", rpm_limit=5) is True


class TestAuthenticatedClient:
    """Tests for AuthenticatedClient dataclass."""

    def test_authenticated_client_creation(self):
        """Test creating authenticated client."""
        from valence.server.app import AuthenticatedClient

        client = AuthenticatedClient(
            client_id="test-id",
            user_id="admin",
            scope="mcp:tools",
            auth_method="oauth",
        )

        assert client.client_id == "test-id"
        assert client.user_id == "admin"
        assert client.scope == "mcp:tools"
        assert client.auth_method == "oauth"

    def test_authenticated_client_defaults(self):
        """Test authenticated client defaults."""
        from valence.server.app import AuthenticatedClient

        client = AuthenticatedClient(client_id="test-id")

        assert client.user_id is None
        assert client.scope is None
        assert client.auth_method == "bearer"
