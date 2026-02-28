"""Tests for parity endpoints (knowledge_search, article_compile, admin_forget)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_config():
    import valence.server.config as config_module

    config_module._settings = None
    yield
    config_module._settings = None


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    token_file = tmp_path / "tokens.json"
    token_file.write_text('{"tokens": []}')
    monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("VALENCE_RATE_LIMIT_RPM", "60")
    return {"token_file": token_file}


@pytest.fixture
def client(app_env) -> TestClient:
    from valence.server.app import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_token(app_env) -> str:
    from valence.server.auth import get_token_store

    store = get_token_store(app_env["token_file"])
    return store.create(client_id="test-client")


API_V1 = "/api/v1"


class TestKnowledgeSearchEndpoint:
    """Tests for GET /api/v1/search."""

    def test_requires_auth(self, client):
        response = client.get(f"{API_V1}/search?query=test")
        assert response.status_code == 401

    def test_missing_query(self, client, auth_token):
        response = client.get(f"{API_V1}/search", headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 400

    @patch("valence.mcp.handlers.articles.knowledge_search")
    def test_search_success(self, mock_search, client, auth_token):
        mock_search.return_value = {"success": True, "results": [{"title": "test"}], "count": 1}

        response = client.get(
            f"{API_V1}/search?query=python",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_search.assert_called_once_with(query="python", limit=10, include_sources=False, session_id=None, temporal_mode="default")

    @patch("valence.mcp.handlers.articles.knowledge_search")
    def test_search_with_params(self, mock_search, client, auth_token):
        mock_search.return_value = {"success": True, "results": [], "count": 0}

        response = client.get(
            f"{API_V1}/search?query=test&limit=5&include_sources=true&session_id=abc",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        mock_search.assert_called_once_with(query="test", limit=5, include_sources=True, session_id="abc", temporal_mode="default")


class TestCompileArticleEndpoint:
    """Tests for POST /api/v1/articles/compile."""

    def test_requires_auth(self, client):
        response = client.post(f"{API_V1}/articles/compile", json={"source_ids": ["abc"]})
        assert response.status_code == 401

    def test_missing_source_ids(self, client, auth_token):
        response = client.post(
            f"{API_V1}/articles/compile",
            json={},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 400

    def test_invalid_json(self, client, auth_token):
        response = client.post(
            f"{API_V1}/articles/compile",
            content=b"not json",
            headers={"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"},
        )
        assert response.status_code == 400

    @patch("valence.mcp.handlers.articles.article_compile")
    def test_compile_success(self, mock_compile, client, auth_token):
        mock_compile.return_value = {"success": True, "article": {"id": "art-1"}}

        response = client.post(
            f"{API_V1}/articles/compile",
            json={"source_ids": ["src-1", "src-2"], "title_hint": "My Article"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        mock_compile.assert_called_once_with(source_ids=["src-1", "src-2"], title_hint="My Article")


class TestAdminForgetEndpoint:
    """Tests for DELETE /api/v1/admin/forget/:target_type/:target_id."""

    def test_requires_auth(self, client):
        response = client.delete(f"{API_V1}/admin/forget/source/abc")
        assert response.status_code == 401

    def test_invalid_target_type(self, client, auth_token):
        response = client.delete(
            f"{API_V1}/admin/forget/invalid/abc",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 400
        assert "target_type" in response.json()["error"]["message"]

    @patch("valence.mcp.handlers.admin.admin_forget")
    def test_forget_source(self, mock_forget, client, auth_token):
        mock_forget.return_value = {"success": True, "deleted": "source", "id": "src-1"}

        response = client.delete(
            f"{API_V1}/admin/forget/source/src-1",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        mock_forget.assert_called_once_with(target_type="source", target_id="src-1")

    @patch("valence.mcp.handlers.admin.admin_forget")
    def test_forget_article(self, mock_forget, client, auth_token):
        mock_forget.return_value = {"success": True, "deleted": "article", "id": "art-1"}

        response = client.delete(
            f"{API_V1}/admin/forget/article/art-1",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        mock_forget.assert_called_once_with(target_type="article", target_id="art-1")
