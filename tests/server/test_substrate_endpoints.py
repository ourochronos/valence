"""Tests for substrate REST endpoints (v2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from valence.server.auth_helpers import AuthenticatedClient
from valence.server.substrate_endpoints import (
    beliefs_create_endpoint,
    beliefs_get_endpoint,
    beliefs_list_endpoint,
    beliefs_search_endpoint,
    beliefs_supersede_endpoint,
    conflicts_endpoint,
    entities_get_endpoint,
    entities_list_endpoint,
    stats_endpoint,
    tensions_list_endpoint,
    tensions_resolve_endpoint,
)

MOCK_CLIENT = AuthenticatedClient(client_id="test", auth_method="bearer")


@pytest.fixture
def app():
    routes = [
        Route("/api/v1/beliefs", beliefs_list_endpoint, methods=["GET"]),
        Route("/api/v1/beliefs", beliefs_create_endpoint, methods=["POST"]),
        Route("/api/v1/beliefs/search", beliefs_search_endpoint, methods=["GET"]),
        Route("/api/v1/beliefs/conflicts", conflicts_endpoint, methods=["GET"]),
        Route("/api/v1/beliefs/{belief_id}", beliefs_get_endpoint, methods=["GET"]),
        Route("/api/v1/beliefs/{belief_id}/supersede", beliefs_supersede_endpoint, methods=["POST"]),
        Route("/api/v1/entities", entities_list_endpoint, methods=["GET"]),
        Route("/api/v1/entities/{id}", entities_get_endpoint, methods=["GET"]),
        Route("/api/v1/tensions", tensions_list_endpoint, methods=["GET"]),
        Route("/api/v1/tensions/{id}/resolve", tensions_resolve_endpoint, methods=["POST"]),
        Route("/api/v1/stats", stats_endpoint, methods=["GET"]),
    ]
    return Starlette(routes=routes)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_auth():
    with patch("valence.server.substrate_endpoints.authenticate", return_value=MOCK_CLIENT):
        yield


# =============================================================================
# AUTH
# =============================================================================


class TestAuth:
    def test_unauthenticated_returns_401(self, client):
        from starlette.responses import JSONResponse

        with patch(
            "valence.server.substrate_endpoints.authenticate",
            return_value=JSONResponse({"error": "unauthorized"}, status_code=401),
        ):
            resp = client.get("/api/v1/beliefs", params={"query": "test"})
            assert resp.status_code == 401

    def test_wrong_scope_returns_403(self, client):
        oauth_client = AuthenticatedClient(client_id="test", auth_method="oauth", scope="vkb:read")
        with patch("valence.server.substrate_endpoints.authenticate", return_value=oauth_client):
            resp = client.get("/api/v1/beliefs", params={"query": "test"})
            assert resp.status_code == 403


# =============================================================================
# BELIEFS (legacy endpoints → v2 article/retrieval tools)
# =============================================================================


class TestBeliefsListEndpoint:
    def test_missing_query_returns_400(self, client):
        resp = client.get("/api/v1/beliefs")
        assert resp.status_code == 400

    @patch("valence.substrate.tools.articles.article_search")
    def test_happy_path(self, mock_search, client):
        mock_search.return_value = {"success": True, "articles": [], "total_count": 0}
        resp = client.get("/api/v1/beliefs", params={"query": "test"})
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock_search.assert_called_once()
        assert mock_search.call_args.kwargs["query"] == "test"

    @patch("valence.substrate.tools.articles.article_search")
    def test_with_filters(self, mock_search, client):
        mock_search.return_value = {"success": True, "articles": [], "total_count": 0}
        resp = client.get(
            "/api/v1/beliefs",
            params={"query": "test", "domain_filter": "tech,python", "limit": "5"},
        )
        assert resp.status_code == 200
        kwargs = mock_search.call_args.kwargs
        assert kwargs["domain_filter"] == ["tech", "python"]
        assert kwargs["limit"] == 5


class TestBeliefsCreateEndpoint:
    def test_missing_content_returns_400(self, client):
        resp = client.post("/api/v1/beliefs", json={})
        assert resp.status_code == 400

    @patch("valence.substrate.tools.articles.article_create")
    def test_happy_path(self, mock_create, client):
        mock_create.return_value = {"success": True, "article": {"id": "abc"}}
        resp = client.post("/api/v1/beliefs", json={"content": "Test article"})
        assert resp.status_code == 201
        assert resp.json()["success"] is True
        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["content"] == "Test article"

    @patch("valence.substrate.tools.articles.article_create")
    def test_with_optional_fields(self, mock_create, client):
        mock_create.return_value = {"success": True, "article": {"id": "abc"}}
        resp = client.post(
            "/api/v1/beliefs",
            json={
                "content": "Test",
                "domain_path": ["tech"],
            },
        )
        assert resp.status_code == 201
        kwargs = mock_create.call_args.kwargs
        assert kwargs["domain_path"] == ["tech"]


class TestBeliefsSearchEndpoint:
    def test_missing_query_returns_400(self, client):
        resp = client.get("/api/v1/beliefs/search")
        assert resp.status_code == 400

    @patch("valence.substrate.tools.articles.article_search")
    def test_happy_path(self, mock_search, client):
        mock_search.return_value = {"success": True, "articles": [], "total_count": 0}
        resp = client.get("/api/v1/beliefs/search", params={"query": "embeddings"})
        assert resp.status_code == 200
        mock_search.assert_called_once()


class TestBeliefsGetEndpoint:
    @patch("valence.substrate.tools.articles.article_get")
    def test_happy_path(self, mock_get, client):
        mock_get.return_value = {"success": True, "article": {"id": "abc"}}
        resp = client.get("/api/v1/beliefs/abc-123")
        assert resp.status_code == 200

    @patch("valence.substrate.tools.articles.article_get")
    def test_not_found(self, mock_get, client):
        mock_get.return_value = {"success": False, "error": "Article not found"}
        resp = client.get("/api/v1/beliefs/nonexistent")
        assert resp.status_code == 404


class TestBeliefsSupersede:
    def test_missing_fields_returns_400(self, client):
        resp = client.post("/api/v1/beliefs/abc/supersede", json={})
        assert resp.status_code == 400

    @patch("valence.substrate.tools.articles.article_update")
    def test_happy_path(self, mock_update, client):
        mock_update.return_value = {"success": True, "article": {"id": "abc", "version": 2}}
        resp = client.post(
            "/api/v1/beliefs/abc/supersede",
            json={"new_content": "Updated", "reason": "Better info"},
        )
        assert resp.status_code == 200
        assert mock_update.call_args.kwargs["article_id"] == "abc"
        assert mock_update.call_args.kwargs["content"] == "Updated"


# =============================================================================
# ENTITIES
# =============================================================================


class TestEntitiesListEndpoint:
    def test_missing_query_returns_400(self, client):
        resp = client.get("/api/v1/entities")
        assert resp.status_code == 400

    @patch("valence.substrate.tools.entities.entity_search")
    def test_happy_path(self, mock_search, client):
        mock_search.return_value = {"success": True, "entities": []}
        resp = client.get("/api/v1/entities", params={"query": "claude"})
        assert resp.status_code == 200


class TestEntitiesGetEndpoint:
    @patch("valence.substrate.tools.entities.entity_get")
    def test_happy_path(self, mock_get, client):
        mock_get.return_value = {"success": True, "entity": {"id": "abc"}}
        resp = client.get("/api/v1/entities/abc")
        assert resp.status_code == 200

    @patch("valence.substrate.tools.entities.entity_get")
    def test_not_found(self, mock_get, client):
        mock_get.return_value = {"success": False, "error": "Entity not found"}
        resp = client.get("/api/v1/entities/nonexistent")
        assert resp.status_code == 404


# =============================================================================
# CONTENTIONS (legacy /tensions routes)
# =============================================================================


class TestTensionsListEndpoint:
    @patch("valence.substrate.tools.contention.contention_list")
    def test_happy_path(self, mock_list, client):
        mock_list.return_value = {"success": True, "contentions": [], "total_count": 0}
        resp = client.get("/api/v1/tensions")
        assert resp.status_code == 200

    @patch("valence.substrate.tools.contention.contention_list")
    def test_with_filters(self, mock_list, client):
        mock_list.return_value = {"success": True, "contentions": [], "total_count": 0}
        resp = client.get("/api/v1/tensions", params={"status": "detected"})
        assert resp.status_code == 200
        kwargs = mock_list.call_args.kwargs
        assert kwargs["status"] == "detected"


class TestTensionsResolveEndpoint:
    def test_missing_fields_returns_400(self, client):
        resp = client.post("/api/v1/tensions/abc/resolve", json={})
        assert resp.status_code == 400

    @patch("valence.substrate.tools.contention.contention_resolve")
    def test_happy_path(self, mock_resolve, client):
        mock_resolve.return_value = {"success": True, "contention": {"id": "abc", "status": "resolved"}}
        resp = client.post(
            "/api/v1/tensions/abc/resolve",
            json={"resolution": "A is correct", "action": "supersede_a"},
        )
        assert resp.status_code == 200


# =============================================================================
# STATS
# =============================================================================


def _mock_cursor():
    """Create a mock cursor context manager."""
    mock_cur = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cur)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm, mock_cur


class TestStatsEndpoint:
    @patch("our_db.get_cursor")
    def test_happy_path(self, mock_gc, client):
        mock_cm, mock_cur = _mock_cursor()
        mock_gc.return_value = mock_cm
        mock_cur.fetchone.side_effect = [
            {"total": 100},
            {"active": 80},
            {"with_emb": 60},
            {"cnt": 5},
            {"count": 10},
            {"cnt": 25},
        ]

        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["stats"]["total_articles"] == 100
        assert data["stats"]["active_articles"] == 80
        assert data["stats"]["with_embeddings"] == 60
        assert data["stats"]["unresolved_contentions"] == 5

    @patch("our_db.get_cursor")
    def test_text_output(self, mock_gc, client):
        mock_cm, mock_cur = _mock_cursor()
        mock_gc.return_value = mock_cm
        mock_cur.fetchone.side_effect = [
            {"total": 10}, {"active": 8}, {"with_emb": 6},
            {"cnt": 1}, {"count": 3}, {"cnt": 5},
        ]

        resp = client.get("/api/v1/stats", params={"output": "text"})
        assert resp.status_code == 200
        assert "Valence Statistics" in resp.text


# =============================================================================
# CONFLICT DETECTION
# =============================================================================


class TestConflictsEndpoint:
    @patch("our_db.get_cursor")
    def test_no_conflicts(self, mock_gc, client):
        mock_cm, mock_cur = _mock_cursor()
        mock_gc.return_value = mock_cm
        mock_cur.fetchall.return_value = []

        resp = client.get("/api/v1/beliefs/conflicts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["count"] == 0

    @patch("our_db.get_cursor")
    def test_with_threshold(self, mock_gc, client):
        mock_cm, mock_cur = _mock_cursor()
        mock_gc.return_value = mock_cm
        mock_cur.fetchall.return_value = []

        resp = client.get("/api/v1/beliefs/conflicts", params={"threshold": "0.9"})
        assert resp.status_code == 200
