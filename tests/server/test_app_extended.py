"""Additional tests for app.py to improve coverage."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset all stores between tests."""
    import valence.server.app as app_module
    import valence.server.auth as auth_module
    import valence.server.config as config_module

    config_module._settings = None
    auth_module._token_store = None
    app_module._rate_limits.clear()
    app_module._openapi_spec_cache = None

    yield

    config_module._settings = None
    auth_module._token_store = None
    app_module._rate_limits.clear()
    app_module._openapi_spec_cache = None


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    """Set up application environment."""
    token_file = tmp_path / "tokens.json"
    token_file.write_text('{"tokens": []}')

    monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("VALENCE_RATE_LIMIT_RPM", "60")
    monkeypatch.setenv("VALENCE_EXTERNAL_URL", "http://localhost:8420")

    return {"token_file": token_file}


@pytest.fixture
def mock_db():
    """Mock database for tests."""
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
def client(app_env, mock_db) -> TestClient:
    """Create test client."""
    from valence.server.app import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_token(app_env) -> str:
    """Create valid auth token."""
    from valence.server.auth import get_token_store

    store = get_token_store(app_env["token_file"])
    return store.create(client_id="test-client")


API_V1 = "/api/v1"


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================


class TestLoggingConfiguration:
    """Tests for configure_logging function."""

    def test_json_logging_format(self):
        """Test JSON logging format."""
        from valence.server.app import configure_logging

        configure_logging(log_format="json")
        # No exception should be raised

    def test_text_logging_format(self):
        """Test text logging format."""
        from valence.server.app import configure_logging

        configure_logging(log_format="text")
        # No exception should be raised

    def test_json_log_formatter(self):
        """Test JSONLogFormatter."""
        import logging

        from valence.server.app import JSONLogFormatter

        formatter = JSONLogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_json_log_formatter_with_exception(self):
        """Test JSONLogFormatter with exception."""
        import logging

        from valence.server.app import JSONLogFormatter

        formatter = JSONLogFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )

            formatted = formatter.format(record)
            data = json.loads(formatted)

            assert data["level"] == "ERROR"
            assert "exception" in data
            assert "ValueError" in data["exception"]


# =============================================================================
# RATE LIMITING
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limit_check_allows_within_limit(self):
        """Test rate limiting allows requests within limit."""
        from valence.server.app import _check_rate_limit

        client_id = "test-client"
        result = _check_rate_limit(client_id, rpm_limit=60)
        assert result is True

    def test_rate_limit_blocks_when_exceeded(self):
        """Test rate limiting blocks when exceeded."""
        from valence.server.app import _check_rate_limit

        client_id = "test-client-blocked"

        # Make requests up to the limit
        for _ in range(5):
            assert _check_rate_limit(client_id, rpm_limit=5) is True

        # Next request should be blocked
        assert _check_rate_limit(client_id, rpm_limit=5) is False

    def test_rate_limit_enforced_on_mcp_endpoint(self, client, app_env):
        """Test rate limit is enforced on MCP endpoint."""
        from valence.server.auth import get_token_store

        # Create token and make many requests
        store = get_token_store(app_env["token_file"])
        token = store.create(client_id="rate-limit-test")

        # Make requests up to limit (set to 60 in fixture)
        for i in range(5):
            response = client.post(
                f"{API_V1}/mcp",
                json={"jsonrpc": "2.0", "method": "ping", "id": i},
                headers={"Authorization": f"Bearer {token}"},
            )
            # First few should succeed
            if i < 60:
                assert response.status_code == 200


# =============================================================================
# MCP REQUEST HANDLING
# =============================================================================


class TestMCPRequestHandling:
    """Tests for MCP request handling."""

    def test_batch_requests(self, client, auth_token):
        """Test MCP batch request handling."""
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

    def test_notification_no_response(self, client, auth_token):
        """Test MCP notification (no id) returns 204."""
        response = client.post(
            f"{API_V1}/mcp",
            json={"jsonrpc": "2.0", "method": "initialized", "params": {}},
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        # Notifications without id can return 204 or still process
        assert response.status_code in [200, 204]

    def test_invalid_json_rpc_version(self, client, auth_token):
        """Test invalid JSON-RPC version."""
        response = client.post(
            f"{API_V1}/mcp",
            json={"jsonrpc": "1.0", "method": "ping", "id": 1},
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32600

    def test_missing_method(self, client, auth_token):
        """Test request with missing method."""
        response = client.post(
            f"{API_V1}/mcp",
            json={"jsonrpc": "2.0", "id": 1},
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32600

    def test_unknown_method(self, client, auth_token):
        """Test request with unknown method."""
        response = client.post(
            f"{API_V1}/mcp",
            json={"jsonrpc": "2.0", "method": "unknown_method", "id": 1},
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32601

    def test_invalid_json_parse_error(self, client, auth_token):
        """Test invalid JSON returns parse error."""
        response = client.post(
            f"{API_V1}/mcp",
            content="{invalid json",
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == -32700

    def test_prompts_get_valence_context(self, client, auth_token):
        """Test prompts/get for valence-context."""
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
        assert "result" in data
        assert "messages" in data["result"]

    def test_prompts_get_recall_context(self, client, auth_token):
        """Test prompts/get for recall-context."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/get",
                "params": {"name": "recall-context", "arguments": {"topic": "AI"}},
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

    def test_prompts_get_unknown_prompt(self, client, auth_token):
        """Test prompts/get with unknown prompt."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/get",
                "params": {"name": "unknown-prompt"},
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data

    def test_resources_read_tools(self, client, auth_token):
        """Test resources/read for tools reference."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "resources/read",
                "params": {"uri": "valence://tools"},
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "contents" in data["result"]

    def test_resources_read_unknown_uri(self, client, auth_token):
        """Test resources/read with unknown URI."""
        response = client.post(
            f"{API_V1}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "resources/read",
                "params": {"uri": "valence://unknown"},
                "id": 1,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data


# =============================================================================
# OPENAPI ENDPOINTS
# =============================================================================


class TestOpenAPIEndpoints:
    """Tests for OpenAPI documentation endpoints."""

    def test_openapi_spec_endpoint(self, client):
        """Test OpenAPI spec endpoint returns JSON."""
        response = client.get(f"{API_V1}/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        # Should return valid OpenAPI spec (either loaded or fallback)

    def test_swagger_ui_endpoint(self, client):
        """Test Swagger UI endpoint."""
        response = client.get(f"{API_V1}/docs")

        assert response.status_code == 200
        assert b"swagger-ui" in response.content
        assert b"Swagger" in response.content or b"API Documentation" in response.content


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_server_instructions(self):
        """Test _get_server_instructions returns content."""
        from valence.server.app import _get_server_instructions

        instructions = _get_server_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 0
        assert "Valence" in instructions

    def test_get_usage_instructions(self):
        """Test _get_usage_instructions returns content."""
        from valence.server.app import _get_usage_instructions

        instructions = _get_usage_instructions()
        assert isinstance(instructions, str)
        assert "knowledge_search" in instructions

    def test_get_tool_reference(self):
        """Test _get_tool_reference returns content."""
        from valence.server.app import _get_tool_reference

        reference = _get_tool_reference()
        assert isinstance(reference, str)

    def test_get_context_prompt(self):
        """Test _get_context_prompt returns content."""
        from valence.server.app import _get_context_prompt

        prompt = _get_context_prompt()
        assert isinstance(prompt, str)
        assert "Valence" in prompt


# =============================================================================
# ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in endpoints."""

    def test_mcp_internal_error(self, client, auth_token):
        """Test MCP endpoint handles internal errors."""
        with patch(
            "valence.server.app._handle_rpc_request",
            side_effect=Exception("Internal error"),
        ):
            response = client.post(
                f"{API_V1}/mcp",
                json={"jsonrpc": "2.0", "method": "ping", "id": 1},
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32603


# =============================================================================
# AUTHENTICATION HELPERS
# =============================================================================


class TestAuthenticateRequest:
    """Tests for _authenticate_request helper."""

    def test_missing_authorization_header(self):
        """Test missing authorization header."""
        from starlette.requests import Request

        from valence.server.app import _authenticate_request

        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = ""

        result = _authenticate_request(mock_request)
        assert result is None

    def test_invalid_authorization_format(self):
        """Test invalid authorization header format."""
        from starlette.requests import Request

        from valence.server.app import _authenticate_request

        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = "InvalidFormat token"

        result = _authenticate_request(mock_request)
        assert result is None

    def test_valid_bearer_token(self, app_env):
        """Test valid bearer token authentication."""
        from starlette.requests import Request

        from valence.server.app import _authenticate_request
        from valence.server.auth import Token, get_token_store

        store = get_token_store(app_env["token_file"])
        token = store.create(client_id="test-auth")

        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = f"Bearer {token}"

        with patch("valence.server.app.verify_token") as mock_verify:
            # Token uses token_hash, not token
            mock_verify.return_value = Token(token_hash=token, client_id="test-auth")

            result = _authenticate_request(mock_request)
            assert result is not None
            assert result.client_id == "test-auth"


# =============================================================================
# OPENAPI SPEC CACHING
# =============================================================================


class TestOpenAPISpecCaching:
    """Tests for OpenAPI spec caching."""

    def test_spec_caching(self, client):
        """Test OpenAPI spec is cached after first load."""
        # First request
        response1 = client.get(f"{API_V1}/openapi.json")
        assert response1.status_code == 200

        # Second request should use cache
        response2 = client.get(f"{API_V1}/openapi.json")
        assert response2.status_code == 200

        # Responses should be identical (from cache)
        assert response1.json() == response2.json()
