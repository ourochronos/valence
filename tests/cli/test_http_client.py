"""Tests for the Valence HTTP client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from valence.cli.config import CLIConfig, reset_cli_config, set_cli_config
from valence.cli.http_client import ValenceAPIError, ValenceClient, ValenceConnectionError, get_client


@pytest.fixture(autouse=True)
def _setup_config():
    """Set up a test config."""
    set_cli_config(CLIConfig(server_url="http://test:8420", token="vt_test123", timeout=5.0))
    yield
    reset_cli_config()


class TestValenceClient:
    def test_url_construction(self):
        client = ValenceClient()
        assert client._url("/beliefs") == "http://test:8420/api/v1/beliefs"
        assert client._url("/admin/migrate/up") == "http://test:8420/api/v1/admin/migrate/up"

    def test_url_strips_trailing_slash(self):
        client = ValenceClient(server_url="http://test:8420/")
        assert client._url("/beliefs") == "http://test:8420/api/v1/beliefs"

    def test_headers_with_token(self):
        client = ValenceClient()
        headers = client._headers()
        assert headers["Authorization"] == "Bearer vt_test123"

    def test_headers_without_token(self):
        client = ValenceClient(token="")
        headers = client._headers()
        assert "Authorization" not in headers

    def test_constructor_overrides_config(self):
        client = ValenceClient(server_url="http://override:9999", token="vt_override", timeout=99.0)
        assert client.base_url == "http://override:9999"
        assert client.token == "vt_override"
        assert client.timeout == 99.0


class TestHandleResponse:
    def test_success_json(self):
        client = ValenceClient()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.headers = {"content-type": "application/json"}
        resp.json.return_value = {"success": True, "beliefs": []}
        result = client._handle_response(resp)
        assert result == {"success": True, "beliefs": []}

    def test_success_plain_text(self):
        client = ValenceClient()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.headers = {"content-type": "text/plain; charset=utf-8"}
        resp.text = "Valence Statistics\n---------\nBeliefs: 42"
        result = client._handle_response(resp)
        assert result == {"formatted": "Valence Statistics\n---------\nBeliefs: 42"}

    def test_error_response_raises(self):
        client = ValenceClient()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 404
        resp.json.return_value = {"success": False, "error": {"code": "NOT_FOUND_BELIEF", "message": "Belief not found"}}
        with pytest.raises(ValenceAPIError) as exc_info:
            client._handle_response(resp)
        assert exc_info.value.status_code == 404
        assert exc_info.value.code == "NOT_FOUND_BELIEF"
        assert exc_info.value.message == "Belief not found"

    def test_error_response_non_json(self):
        client = ValenceClient()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 500
        resp.json.side_effect = json.JSONDecodeError("", "", 0)
        resp.text = "Internal Server Error"
        with pytest.raises(ValenceAPIError) as exc_info:
            client._handle_response(resp)
        assert exc_info.value.status_code == 500
        assert exc_info.value.message == "Internal Server Error"

    def test_auth_error(self):
        client = ValenceClient()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 401
        resp.json.return_value = {"success": False, "error": {"code": "AUTH_MISSING_TOKEN", "message": "Missing or invalid authentication token"}}
        with pytest.raises(ValenceAPIError) as exc_info:
            client._handle_response(resp)
        assert exc_info.value.status_code == 401
        assert exc_info.value.code == "AUTH_MISSING_TOKEN"

    def test_error_response_string_error(self):
        """Errors returned as plain strings should be handled gracefully."""
        client = ValenceClient()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 400
        resp.json.return_value = {"success": False, "error": "Source already has tree_index."}
        with pytest.raises(ValenceAPIError) as exc_info:
            client._handle_response(resp)
        assert exc_info.value.status_code == 400
        assert exc_info.value.code == "ERROR"
        assert "tree_index" in exc_info.value.message


class TestClientRequests:
    @patch("valence.cli.http_client.httpx.Client")
    def test_get(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {"success": True, "data": "test"}
        mock_client.request.return_value = mock_resp

        client = ValenceClient()
        result = client.get("/beliefs", params={"query": "test"})
        assert result == {"success": True, "data": "test"}
        mock_client.request.assert_called_once_with(
            "GET",
            "http://test:8420/api/v1/beliefs",
            headers={"Authorization": "Bearer vt_test123"},
            params={"query": "test"},
            json=None,
        )

    @patch("valence.cli.http_client.httpx.Client")
    def test_post(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 201
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {"success": True, "id": "abc123"}
        mock_client.request.return_value = mock_resp

        client = ValenceClient()
        result = client.post("/beliefs", body={"content": "test belief"})
        assert result == {"success": True, "id": "abc123"}

    @patch("valence.cli.http_client.httpx.Client")
    def test_connection_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.request.side_effect = httpx.ConnectError("Connection refused")

        client = ValenceClient()
        with pytest.raises(ValenceConnectionError) as exc_info:
            client.get("/health")
        assert "http://test:8420" in str(exc_info.value)
        assert "valence-server" in str(exc_info.value)

    @patch("valence.cli.http_client.httpx.Client")
    def test_timeout_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.request.side_effect = httpx.TimeoutException("timed out")

        client = ValenceClient()
        with pytest.raises(ValenceConnectionError) as exc_info:
            client.get("/health")
        assert "timed out" in str(exc_info.value)


class TestGetClient:
    def test_get_client_uses_config(self):
        client = get_client()
        assert client.base_url == "http://test:8420"
        assert client.token == "vt_test123"
