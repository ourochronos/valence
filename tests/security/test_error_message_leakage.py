"""Security tests for error message handling.

Verifies that internal error responses don't leak implementation details
like exception messages, stack traces, or internal paths.

References issue #174: Security - Don't expose exception details in error responses.
"""

from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from valence.server.auth import TokenStore
from valence.server.auth_helpers import AuthenticatedClient

MOCK_CLIENT = AuthenticatedClient(client_id="test", auth_method="bearer")

# =============================================================================
# SENSITIVE PATTERNS TO CHECK FOR LEAKAGE
# =============================================================================

# These patterns should NEVER appear in error responses
SENSITIVE_PATTERNS = [
    "Traceback",
    'File "',
    "/home/",
    "/usr/",
    "site-packages",
    ".py:",  # Stack trace line numbers
    "connection refused",
    "database",
    "password",
    "secret",
    "token",
    "credential",
    "PostgreSQL",
    "psycopg",
    "sqlalchemy",
]


def assert_no_sensitive_info(response_body: dict | str, context: str = "") -> None:
    """Assert that response body doesn't contain sensitive information.

    Args:
        response_body: The response JSON or string to check
        context: Optional context for error messages
    """
    body_str = json.dumps(response_body) if isinstance(response_body, dict) else response_body
    body_lower = body_str.lower()

    for pattern in SENSITIVE_PATTERNS:
        assert pattern.lower() not in body_lower, (
            f"Sensitive pattern '{pattern}' found in error response{' (' + context + ')' if context else ''}: {body_str[:200]}"
        )


# =============================================================================
# CORROBORATION ENDPOINTS SECURITY TESTS
# =============================================================================


class TestCorroborationEndpointsSecurity:
    """Test that corroboration endpoints don't leak error details."""

    @pytest.fixture(autouse=True)
    def mock_auth(self):
        with patch("valence.server.corroboration_endpoints.authenticate", return_value=MOCK_CLIENT):
            yield

    @pytest.fixture
    def app(self):
        """Create test app with corroboration endpoints."""
        from valence.server.corroboration_endpoints import (
            belief_corroboration_endpoint,
            most_corroborated_beliefs_endpoint,
        )

        routes = [
            Route(
                "/beliefs/{belief_id}/corroboration",
                belief_corroboration_endpoint,
                methods=["GET"],
            ),
            Route(
                "/beliefs/most-corroborated",
                most_corroborated_beliefs_endpoint,
                methods=["GET"],
            ),
        ]
        return Starlette(routes=routes)

    @pytest.fixture
    def client(self, app):
        return TestClient(app, raise_server_exceptions=False)

    def test_corroboration_db_error_no_leakage(self, client):
        """Database errors should not leak connection details."""
        belief_id = uuid4()

        with patch("our_federation.corroboration.get_corroboration") as mock_get:
            # Simulate a database error with sensitive details
            mock_get.side_effect = Exception(
                'psycopg2.OperationalError: connection to server at "localhost" (127.0.0.1), port 5432 failed: Connection refused'
            )

            response = client.get(f"/beliefs/{belief_id}/corroboration")

        assert response.status_code == 500
        data = response.json()

        # Should have generic error message
        assert data["success"] is False
        assert "internal" in data["error"]["message"].lower()

        # Should NOT leak sensitive database details
        assert_no_sensitive_info(data, "corroboration endpoint")
        assert "psycopg" not in json.dumps(data).lower()
        assert "connection" not in data["error"]["message"].lower()
        assert "localhost" not in json.dumps(data)
        assert "5432" not in json.dumps(data)

    def test_corroboration_file_error_no_leakage(self, client):
        """File errors should not leak path information."""
        belief_id = uuid4()

        with patch("our_federation.corroboration.get_corroboration") as mock_get:
            mock_get.side_effect = FileNotFoundError("[Errno 2] No such file or directory: '/home/user/app/data/config.json'")

            response = client.get(f"/beliefs/{belief_id}/corroboration")

        assert response.status_code == 500
        data = response.json()

        assert_no_sensitive_info(data, "corroboration file error")
        assert "/home/" not in json.dumps(data)
        assert "config.json" not in json.dumps(data)

    def test_most_corroborated_error_no_leakage(self, client):
        """Most corroborated endpoint errors should not leak details."""
        with patch("our_federation.corroboration.get_most_corroborated_beliefs") as mock_get:
            mock_get.side_effect = RuntimeError("SQLAlchemy error: Table 'beliefs' doesn't exist in schema 'public'")

            response = client.get("/beliefs/most-corroborated")

        assert response.status_code == 500
        data = response.json()

        assert_no_sensitive_info(data, "most corroborated endpoint")
        assert "sqlalchemy" not in json.dumps(data).lower()
        assert "schema" not in json.dumps(data).lower()
        assert "table" not in json.dumps(data).lower()


# =============================================================================
# COMPLIANCE ENDPOINTS SECURITY TESTS
# =============================================================================


class TestComplianceEndpointsSecurity:
    """Test that compliance endpoints don't leak error details."""

    @pytest.fixture(autouse=True)
    def mock_auth(self):
        with patch("valence.server.compliance_endpoints.authenticate", return_value=MOCK_CLIENT):
            yield

    @pytest.fixture
    def app(self):
        """Create test app with compliance endpoints."""
        from valence.server.compliance_endpoints import delete_user_data_endpoint

        routes = [
            Route(
                "/compliance/delete/{id}",
                delete_user_data_endpoint,
                methods=["DELETE"],
            ),
        ]
        return Starlette(routes=routes)

    @pytest.fixture
    def client(self, app):
        return TestClient(app, raise_server_exceptions=False)

    def test_deletion_error_no_leakage(self, client):
        """User data deletion errors should not leak details."""
        with patch("valence.server.compliance_endpoints.delete_user_data") as mock_delete:
            mock_delete.side_effect = Exception("Failed to connect to PostgreSQL at 192.168.1.100:5432 with user 'valence_admin'")

            response = client.delete("/compliance/delete/user123?reason=user_request")

        assert response.status_code == 500
        data = response.json()

        assert_no_sensitive_info(data, "compliance deletion")
        assert "192.168" not in json.dumps(data)
        assert "valence_admin" not in json.dumps(data)
        assert "postgresql" not in json.dumps(data).lower()


# =============================================================================
# MCP/JSON-RPC ENDPOINTS SECURITY TESTS
# =============================================================================


class TestMCPEndpointsSecurity:
    """Test that MCP JSON-RPC endpoints don't leak error details."""

    @pytest.fixture
    def token_file(self, tmp_path):
        """Create a temporary token file."""
        token_path = tmp_path / "tokens.json"
        token_path.write_text('{"tokens": []}')
        return token_path

    @pytest.fixture
    def test_token(self, token_file):
        """Create a valid test token."""
        store = TokenStore(token_file)
        raw_token = store.create(client_id="test-client", description="Test token")
        return raw_token

    @pytest.fixture
    def auth_headers(self, test_token):
        """Create auth headers with a valid token."""
        return {"Authorization": f"Bearer {test_token}"}

    @pytest.fixture
    def app(self, token_file, monkeypatch):
        """Create test app with MCP endpoint."""
        # Clear cached settings and token store
        import valence.server.auth as auth_module
        import valence.server.config as config_module

        config_module._settings = None
        auth_module._token_store = None

        # Set token file path in environment
        monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))

        from valence.server.app import create_app

        return create_app()

    @pytest.fixture
    def client(self, app):
        return TestClient(app, raise_server_exceptions=False)

    def test_mcp_parse_error_no_leakage(self, client, auth_headers):
        """JSON parse errors should not leak parsing details."""
        response = client.post(
            "/api/v1/mcp",
            content=b"{ invalid json here }}}",
            headers={**auth_headers, "Content-Type": "application/json"},
        )

        # JSON-RPC parse errors return 400
        assert response.status_code == 400
        data = response.json()

        # Should have generic error message
        assert data["error"]["code"] == -32700
        assert data["error"]["message"] == "Parse error"

        # Should NOT include the actual parsing exception details
        assert "Expecting" not in data["error"]["message"]
        assert "line" not in data["error"]["message"].lower()
        assert "column" not in data["error"]["message"].lower()

    def test_mcp_internal_error_no_leakage(self, client, auth_headers):
        """Internal MCP errors should not leak details."""
        # Valid JSON-RPC request that will cause an internal error
        with patch("valence.server.app._dispatch_method") as mock_dispatch:
            mock_dispatch.side_effect = Exception("KeyError: 'embedding_model' not found in config at /etc/valence/secrets.yaml")

            response = client.post(
                "/api/v1/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 1,
                },
                headers=auth_headers,
            )

        # JSON-RPC returns 200 with error in body for method errors
        # (500 is only for transport-level errors like JSON parse failure)
        assert response.status_code == 200
        data = response.json()

        assert data["error"]["code"] == -32603
        assert data["error"]["message"] == "Internal error"

        # Should NOT leak the actual exception
        assert_no_sensitive_info(data, "MCP internal error")
        assert "KeyError" not in json.dumps(data)
        assert "secrets.yaml" not in json.dumps(data)
        assert "/etc/" not in json.dumps(data)


# =============================================================================
# FEDERATION ENDPOINTS SECURITY TESTS
# =============================================================================


class TestFederationEndpointsSecurity:
    """Test that federation endpoints don't leak error details."""

    @pytest.fixture(autouse=True)
    def reset_stores(self):
        """Reset config between tests."""
        import valence.server.config as config_module

        config_module._settings = None
        yield
        config_module._settings = None

    @pytest.fixture
    def federation_env(self, monkeypatch, tmp_path):
        """Set up federation-enabled environment."""
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
        monkeypatch.setenv("VALENCE_EXTERNAL_URL", "https://valence.example.com")
        monkeypatch.setenv("VALENCE_FEDERATION_ENABLED", "true")
        monkeypatch.setenv("VALENCE_FEDERATION_NODE_DID", "did:vkb:web:valence.example.com")
        monkeypatch.setenv("VALENCE_FEDERATION_NODE_NAME", "TestNode")
        monkeypatch.setenv(
            "VALENCE_FEDERATION_PUBLIC_KEY",
            "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
        )

    def test_federation_status_error_no_leakage(self, federation_env, monkeypatch):
        """Federation status errors should not leak sensitive details in error field."""
        from valence.server.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Mock the internal stats function to return stats with an error field
        # The _get_federation_stats function has its own try/except that catches errors
        # and returns them in an 'error' field
        with patch("valence.server.federation_endpoints._get_federation_stats") as mock_stats:
            # Return stats with an error field (simulating caught exception)
            mock_stats.return_value = {
                "nodes": {"total": 0, "by_status": {}},
                "sync": {"active_peers": 0, "beliefs_sent": 0, "beliefs_received": 0},
                "beliefs": {"local": 0, "federated": 0},
                "error": "Failed to fetch federation stats",
            }

            response = client.get("/api/v1/federation/status")

        assert response.status_code == 200
        data = response.json()

        assert "federation" in data
        assert "error" in data["federation"]

        # Error message should be generic, not expose implementation details
        error_str = data["federation"]["error"]
        assert error_str == "Failed to fetch federation stats"
        assert "cryptography" not in error_str.lower()
        assert "signature" not in error_str.lower()

    def test_vfp_node_metadata_error_no_leakage(self, federation_env, monkeypatch):
        """VFP node metadata errors should not leak details."""
        from valence.server.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Mock the internal build function to raise an error with sensitive info
        with patch("valence.server.federation_endpoints._build_did_document") as mock_build:
            mock_build.side_effect = Exception("psycopg2.OperationalError: connection to database at 192.168.1.100 failed")

            response = client.get("/.well-known/vfp-node-metadata")

        # Should return 500 but not leak connection details
        assert response.status_code == 500
        response_text = response.text
        assert "psycopg2" not in response_text.lower()
        assert "192.168" not in response_text
        assert "OperationalError" not in response_text


# =============================================================================
# UNIFIED SERVER (MCP TOOLS) SECURITY TESTS
# =============================================================================


class TestUnifiedServerSecurity:
    """Test that unified MCP server tool errors don't leak details."""

    def test_tool_internal_error_structure(self):
        """Verify tool errors use generic messages."""
        # This is a structural test - we check the code directly
        import inspect

        from valence.server import unified_server

        source = inspect.getsource(unified_server)

        # Should NOT have f"Internal error: {str(e)}" pattern
        assert 'f"Internal error: {str(e)}"' not in source
        assert "f'Internal error: {str(e)}'" not in source

        # Should have generic "Internal error" (without exception details)
        assert '"Internal error"' in source or "'Internal error'" in source


# =============================================================================
# GENERIC ERROR RESPONSE FORMAT TESTS
# =============================================================================


class TestErrorResponseFormat:
    """Test that all error responses use consistent generic messages."""

    def test_internal_error_helper_exists(self):
        """Verify internal_error helper function returns generic message."""
        from valence.server.errors import internal_error

        response = internal_error("Internal server error")

        assert response.status_code == 500

        # The message should be what we passed, not auto-generated from exception
        body = json.loads(response.body)
        assert body["error"]["message"] == "Internal server error"

    def test_internal_error_codes(self):
        """Verify internal_error uses correct error codes."""
        from valence.server.errors import internal_error

        response = internal_error("Test")

        body = json.loads(response.body)

        # Should have standard error structure
        assert "error" in body
        assert "code" in body["error"]
        assert "message" in body["error"]
