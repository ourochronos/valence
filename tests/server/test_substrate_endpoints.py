"""Tests for substrate_endpoints.py (stats and conflict detection)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config between tests."""
    import valence.server.config as config_module

    config_module._settings = None
    yield
    config_module._settings = None


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    """Set up application environment."""
    token_file = tmp_path / "tokens.json"
    token_file.write_text('{"tokens": []}')

    monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("VALENCE_RATE_LIMIT_RPM", "60")
    return {"token_file": token_file}


@pytest.fixture
def mock_db():
    """Mock database cursor."""
    mock_cursor = MagicMock()

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
# STATS
# =============================================================================


class TestStatsEndpoint:
    """Tests for GET /api/v1/stats."""

    def test_requires_auth(self, client):
        response = client.get(f"{API_V1}/stats")
        assert response.status_code == 401

    def test_stats_success(self, client, auth_token, mock_db):
        mock_db.fetchone.side_effect = [
            {"total": 100},
            {"active": 80},
            {"with_emb": 60},
            {"cnt": 5},
            {"count": 10},
            {"cnt": 50},
        ]

        response = client.get(f"{API_V1}/stats", headers={"Authorization": f"Bearer {auth_token}"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        stats = data["stats"]
        assert stats["total_articles"] == 100
        assert stats["active_articles"] == 80
        assert stats["with_embeddings"] == 60
        assert stats["unresolved_contentions"] == 5

    def test_stats_json_format(self, client, auth_token, mock_db):
        mock_db.fetchone.side_effect = [
            {"total": 10},
            {"active": 8},
            {"with_emb": 5},
            {"cnt": 2},
            {"count": 3},
            {"cnt": 15},
        ]

        response = client.get(
            f"{API_V1}/stats?output=json",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_stats_text_format(self, client, auth_token, mock_db):
        mock_db.fetchone.side_effect = [
            {"total": 10},
            {"active": 8},
            {"with_emb": 5},
            {"cnt": 2},
            {"count": 3},
            {"cnt": 15},
        ]

        response = client.get(
            f"{API_V1}/stats?output=text",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        assert b"Valence Statistics" in response.content


# =============================================================================
# CONFLICT DETECTION
# =============================================================================


class TestConflictsEndpoint:
    """Tests for GET /api/v1/contentions/detect."""

    def test_requires_auth(self, client):
        response = client.get(f"{API_V1}/contentions/detect")
        assert response.status_code == 401

    def test_conflicts_detection(self, client, auth_token, mock_db):
        mock_db.fetchall.return_value = []

        response = client.get(
            f"{API_V1}/contentions/detect",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 0

    def test_conflicts_with_threshold(self, client, auth_token, mock_db):
        mock_db.fetchall.return_value = []

        response = client.get(
            f"{API_V1}/contentions/detect?threshold=0.9",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["threshold"] == 0.9

    def test_conflicts_auto_record(self, client, auth_token, mock_db):
        mock_db.fetchall.return_value = []

        response = client.get(
            f"{API_V1}/contentions/detect?auto_record=true",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "recorded_contentions" in data

    def test_conflicts_text_format(self, client, auth_token, mock_db):
        mock_db.fetchall.return_value = []

        response = client.get(
            f"{API_V1}/contentions/detect?output=text",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        assert b"No potential conflicts" in response.content
