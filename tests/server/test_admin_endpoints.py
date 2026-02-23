"""Tests for admin REST endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from valence.server.admin_endpoints import (
    admin_embeddings_backfill,
    admin_embeddings_migrate,
    admin_embeddings_status,
    admin_maintenance,
    admin_migrate_down,
    admin_migrate_status,
    admin_migrate_up,
    admin_verify_chains,
)
from valence.server.auth_helpers import AuthenticatedClient

MOCK_CLIENT = AuthenticatedClient(client_id="test", auth_method="bearer")


@pytest.fixture
def app():
    routes = [
        Route("/api/v1/admin/migrate/status", admin_migrate_status, methods=["GET"]),
        Route("/api/v1/admin/migrate/up", admin_migrate_up, methods=["POST"]),
        Route("/api/v1/admin/migrate/down", admin_migrate_down, methods=["POST"]),
        Route("/api/v1/admin/maintenance", admin_maintenance, methods=["POST"]),
        Route("/api/v1/admin/embeddings/status", admin_embeddings_status, methods=["GET"]),
        Route("/api/v1/admin/embeddings/backfill", admin_embeddings_backfill, methods=["POST"]),
        Route("/api/v1/admin/embeddings/migrate", admin_embeddings_migrate, methods=["POST"]),
        Route("/api/v1/admin/verify-chains", admin_verify_chains, methods=["GET"]),
    ]
    return Starlette(routes=routes)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_auth():
    with patch("valence.server.admin_endpoints.authenticate", return_value=MOCK_CLIENT):
        yield


def _mock_cursor():
    """Create a mock cursor context manager."""
    mock_cur = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cur)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm, mock_cur


# =============================================================================
# AUTH
# =============================================================================


class TestAdminAuth:
    def test_unauthenticated_returns_401(self, client):
        with patch(
            "valence.server.admin_endpoints.authenticate",
            return_value=JSONResponse({"error": "unauthorized"}, status_code=401),
        ):
            resp = client.get("/api/v1/admin/migrate/status")
            assert resp.status_code == 401

    def test_wrong_scope_returns_403(self, client):
        oauth_client = AuthenticatedClient(client_id="test", auth_method="oauth", scope="substrate:read")
        with patch("valence.server.admin_endpoints.authenticate", return_value=oauth_client):
            resp = client.get("/api/v1/admin/migrate/status")
            assert resp.status_code == 403


# =============================================================================
# MIGRATION ENDPOINTS
# =============================================================================


class TestMigrateStatus:
    @patch("our_db.get_cursor")
    def test_happy_path(self, mock_gc, client):
        mock_cm, mock_cur = _mock_cursor()
        mock_gc.return_value = mock_cm
        mock_cur.fetchall.return_value = [
            {"name": "001_init", "applied_at": "2024-01-01 00:00:00"},
            {"name": "002_add_embeddings", "applied_at": "2024-01-02 00:00:00"},
        ]

        resp = client.get("/api/v1/admin/migrate/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["count"] == 2

    @patch("our_db.get_cursor")
    def test_text_output(self, mock_gc, client):
        mock_cm, mock_cur = _mock_cursor()
        mock_gc.return_value = mock_cm
        mock_cur.fetchall.return_value = [{"name": "001_init", "applied_at": "2024-01-01"}]

        resp = client.get("/api/v1/admin/migrate/status", params={"output": "text"})
        assert resp.status_code == 200
        assert "Migration Status" in resp.text


class TestMigrateUp:
    @patch("valence.core.migrations.MigrationRunner")
    def test_happy_path(self, mock_runner_cls, client):
        mock_runner = MagicMock()
        mock_runner.up.return_value = ["001_init", "002_embeddings"]
        mock_runner_cls.return_value = mock_runner

        resp = client.post("/api/v1/admin/migrate/up", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["applied"]) == 2

    @patch("valence.core.migrations.MigrationRunner")
    def test_dry_run(self, mock_runner_cls, client):
        mock_runner = MagicMock()
        mock_runner.up.return_value = ["001_init"]
        mock_runner_cls.return_value = mock_runner

        resp = client.post("/api/v1/admin/migrate/up", json={"dry_run": True})
        assert resp.status_code == 200
        assert resp.json()["dry_run"] is True


# =============================================================================
# MAINTENANCE
# =============================================================================


class TestMaintenance:
    def test_no_operation_returns_error(self, client):
        resp = client.post("/api/v1/admin/maintenance", json={})
        assert resp.status_code == 400

    @patch("valence.cli.utils.get_db_connection")
    @patch("valence.core.maintenance.apply_retention")
    def test_retention(self, mock_retention, mock_conn, client):
        from valence.core.maintenance import MaintenanceResult

        mock_retention.return_value = [MaintenanceResult(operation="apply_retention", details={"deleted": 5})]
        conn = MagicMock()
        conn.autocommit = False
        conn.cursor.return_value = MagicMock()
        mock_conn.return_value = conn

        resp = client.post("/api/v1/admin/maintenance", json={"retention": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["count"] == 1

    @patch("valence.cli.utils.get_db_connection")
    @patch("valence.core.maintenance.run_full_maintenance")
    def test_run_all(self, mock_full, mock_conn, client):
        from valence.core.maintenance import MaintenanceResult

        mock_full.return_value = [
            MaintenanceResult(operation="retention", details={}),
            MaintenanceResult(operation="archive", details={}),
        ]
        conn = MagicMock()
        conn.autocommit = False
        conn.cursor.return_value = MagicMock()
        mock_conn.return_value = conn

        resp = client.post("/api/v1/admin/maintenance", json={"all": True})
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    @patch("valence.cli.utils.get_db_connection")
    @patch("valence.core.maintenance.apply_retention")
    def test_dry_run(self, mock_retention, mock_conn, client):
        from valence.core.maintenance import MaintenanceResult

        mock_retention.return_value = [MaintenanceResult(operation="retention", details={"would_delete": 5}, dry_run=True)]
        conn = MagicMock()
        conn.autocommit = False
        conn.cursor.return_value = MagicMock()
        mock_conn.return_value = conn

        resp = client.post("/api/v1/admin/maintenance", json={"retention": True, "dry_run": True})
        assert resp.status_code == 200
        assert resp.json()["dry_run"] is True


# =============================================================================
# EMBEDDINGS
# =============================================================================


class TestEmbeddingsStatus:
    @patch("our_db.get_cursor")
    def test_happy_path(self, mock_gc, client):
        mock_cm, mock_cur = _mock_cursor()
        mock_gc.return_value = mock_cm
        mock_cur.fetchone.side_effect = [
            {"total": 100},
            {"embedded": 80},
            {"missing": 20},
        ]

        resp = client.get("/api/v1/admin/embeddings/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["stats"]["total_articles"] == 100
        assert data["stats"]["with_embeddings"] == 80


class TestEmbeddingsBackfill:
    @patch("our_embeddings.service.generate_embedding")
    @patch("our_db.get_cursor")
    def test_dry_run(self, mock_gc, mock_embed, client):
        mock_cm, mock_cur = _mock_cursor()
        mock_gc.return_value = mock_cm
        mock_cur.fetchall.return_value = [{"id": "1", "content": "test"}]

        resp = client.post("/api/v1/admin/embeddings/backfill", json={"dry_run": True})
        assert resp.status_code == 200
        assert resp.json()["dry_run"] is True
        assert resp.json()["would_process"] == 1


class TestEmbeddingsMigrate:
    def test_missing_params(self, client):
        resp = client.post("/api/v1/admin/embeddings/migrate", json={})
        assert resp.status_code == 400

    @patch("our_db.get_cursor")
    def test_dry_run(self, mock_gc, client):
        mock_cm, mock_cur = _mock_cursor()
        mock_gc.return_value = mock_cm
        mock_cur.fetchone.return_value = {"count": 50}

        resp = client.post("/api/v1/admin/embeddings/migrate", json={"model": "text-embedding-3-small", "dry_run": True})
        assert resp.status_code == 200
        assert resp.json()["would_affect"] == 50


# =============================================================================
# CHAIN VERIFICATION
# =============================================================================


class TestVerifyChains:
    @patch("our_db.get_cursor")
    def test_healthy(self, mock_gc, client):
        mock_cm, mock_cur = _mock_cursor()
        mock_gc.return_value = mock_cm
        mock_cur.fetchall.side_effect = [[], []]

        resp = client.get("/api/v1/admin/verify-chains")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "healthy"
        assert data["count"] == 0

    @patch("our_db.get_cursor")
    def test_issues_found(self, mock_gc, client):
        mock_cm, mock_cur = _mock_cursor()
        mock_gc.return_value = mock_cm
        mock_cur.fetchall.side_effect = [
            [{"id": "abc", "superseded_by_id": "xyz"}],
            [],
        ]

        resp = client.get("/api/v1/admin/verify-chains")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "issues_found"
        assert data["count"] == 1
