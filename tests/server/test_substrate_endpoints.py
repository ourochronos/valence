"""Tests for substrate REST endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from valence.server.auth_helpers import AuthenticatedClient
from valence.server.substrate_endpoints import (
    beliefs_confidence_endpoint,
    beliefs_create_endpoint,
    beliefs_get_endpoint,
    beliefs_list_endpoint,
    beliefs_search_endpoint,
    beliefs_supersede_endpoint,
    entities_get_endpoint,
    entities_list_endpoint,
    tensions_list_endpoint,
    tensions_resolve_endpoint,
    trust_check_endpoint,
)

MOCK_CLIENT = AuthenticatedClient(client_id="test", auth_method="bearer")


@pytest.fixture
def app():
    routes = [
        Route("/api/v1/beliefs", beliefs_list_endpoint, methods=["GET"]),
        Route("/api/v1/beliefs", beliefs_create_endpoint, methods=["POST"]),
        Route("/api/v1/beliefs/search", beliefs_search_endpoint, methods=["GET"]),
        Route("/api/v1/beliefs/{belief_id}", beliefs_get_endpoint, methods=["GET"]),
        Route("/api/v1/beliefs/{belief_id}/supersede", beliefs_supersede_endpoint, methods=["POST"]),
        Route("/api/v1/beliefs/{belief_id}/confidence", beliefs_confidence_endpoint, methods=["GET"]),
        Route("/api/v1/entities", entities_list_endpoint, methods=["GET"]),
        Route("/api/v1/entities/{id}", entities_get_endpoint, methods=["GET"]),
        Route("/api/v1/tensions", tensions_list_endpoint, methods=["GET"]),
        Route("/api/v1/tensions/{id}/resolve", tensions_resolve_endpoint, methods=["POST"]),
        Route("/api/v1/trust", trust_check_endpoint, methods=["GET"]),
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
# BELIEFS
# =============================================================================


class TestBeliefsListEndpoint:
    def test_missing_query_returns_400(self, client):
        resp = client.get("/api/v1/beliefs")
        assert resp.status_code == 400

    @patch("valence.substrate.tools.beliefs.belief_query")
    def test_happy_path(self, mock_query, client):
        mock_query.return_value = {"success": True, "beliefs": []}
        resp = client.get("/api/v1/beliefs", params={"query": "test"})
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock_query.assert_called_once()
        assert mock_query.call_args.kwargs["query"] == "test"

    @patch("valence.substrate.tools.beliefs.belief_query")
    def test_with_filters(self, mock_query, client):
        mock_query.return_value = {"success": True, "beliefs": []}
        resp = client.get(
            "/api/v1/beliefs",
            params={"query": "test", "domain_filter": "tech,python", "limit": "5", "include_archived": "true"},
        )
        assert resp.status_code == 200
        kwargs = mock_query.call_args.kwargs
        assert kwargs["domain_filter"] == ["tech", "python"]
        assert kwargs["limit"] == 5
        assert kwargs["include_archived"] is True


class TestBeliefsCreateEndpoint:
    def test_missing_content_returns_400(self, client):
        resp = client.post("/api/v1/beliefs", json={})
        assert resp.status_code == 400

    @patch("valence.substrate.tools.beliefs.belief_create")
    def test_happy_path(self, mock_create, client):
        mock_create.return_value = {"success": True, "belief": {"id": "abc"}}
        resp = client.post("/api/v1/beliefs", json={"content": "Test belief"})
        assert resp.status_code == 201
        assert resp.json()["success"] is True
        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["content"] == "Test belief"

    @patch("valence.substrate.tools.beliefs.belief_create")
    def test_with_optional_fields(self, mock_create, client):
        mock_create.return_value = {"success": True, "belief": {"id": "abc"}}
        resp = client.post(
            "/api/v1/beliefs",
            json={
                "content": "Test",
                "domain_path": ["tech"],
                "confidence": {"overall": 0.9},
                "visibility": "federated",
            },
        )
        assert resp.status_code == 201
        kwargs = mock_create.call_args.kwargs
        assert kwargs["domain_path"] == ["tech"]
        assert kwargs["visibility"] == "federated"


class TestBeliefsSearchEndpoint:
    def test_missing_query_returns_400(self, client):
        resp = client.get("/api/v1/beliefs/search")
        assert resp.status_code == 400

    @patch("valence.substrate.tools.beliefs.belief_search")
    def test_happy_path(self, mock_search, client):
        mock_search.return_value = {"success": True, "results": []}
        resp = client.get("/api/v1/beliefs/search", params={"query": "embeddings"})
        assert resp.status_code == 200
        mock_search.assert_called_once()


class TestBeliefsGetEndpoint:
    @patch("valence.substrate.tools.beliefs.belief_get")
    def test_happy_path(self, mock_get, client):
        mock_get.return_value = {"success": True, "belief": {"id": "abc"}}
        resp = client.get("/api/v1/beliefs/abc-123")
        assert resp.status_code == 200

    @patch("valence.substrate.tools.beliefs.belief_get")
    def test_not_found(self, mock_get, client):
        mock_get.return_value = {"success": False, "error": "Belief not found"}
        resp = client.get("/api/v1/beliefs/nonexistent")
        assert resp.status_code == 404


class TestBeliefsSupersede:
    def test_missing_fields_returns_400(self, client):
        resp = client.post("/api/v1/beliefs/abc/supersede", json={})
        assert resp.status_code == 400

    @patch("valence.substrate.tools.beliefs.belief_supersede")
    def test_happy_path(self, mock_supersede, client):
        mock_supersede.return_value = {"success": True, "new_belief": {"id": "def"}}
        resp = client.post(
            "/api/v1/beliefs/abc/supersede",
            json={"new_content": "Updated", "reason": "Better info"},
        )
        assert resp.status_code == 200
        assert mock_supersede.call_args.kwargs["old_belief_id"] == "abc"


class TestBeliefsConfidence:
    @patch("valence.substrate.tools.confidence.confidence_explain")
    def test_happy_path(self, mock_explain, client):
        mock_explain.return_value = {"success": True, "overall": 0.8}
        resp = client.get("/api/v1/beliefs/abc/confidence")
        assert resp.status_code == 200


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
# TENSIONS
# =============================================================================


class TestTensionsListEndpoint:
    @patch("valence.substrate.tools.tensions.tension_list")
    def test_happy_path(self, mock_list, client):
        mock_list.return_value = {"success": True, "tensions": []}
        resp = client.get("/api/v1/tensions")
        assert resp.status_code == 200

    @patch("valence.substrate.tools.tensions.tension_list")
    def test_with_filters(self, mock_list, client):
        mock_list.return_value = {"success": True, "tensions": []}
        resp = client.get("/api/v1/tensions", params={"status": "detected", "severity": "high"})
        assert resp.status_code == 200
        kwargs = mock_list.call_args.kwargs
        assert kwargs["status"] == "detected"
        assert kwargs["severity"] == "high"


class TestTensionsResolveEndpoint:
    def test_missing_fields_returns_400(self, client):
        resp = client.post("/api/v1/tensions/abc/resolve", json={})
        assert resp.status_code == 400

    @patch("valence.substrate.tools.tensions.tension_resolve")
    def test_happy_path(self, mock_resolve, client):
        mock_resolve.return_value = {"success": True}
        resp = client.post(
            "/api/v1/tensions/abc/resolve",
            json={"resolution": "A is correct", "action": "supersede_b"},
        )
        assert resp.status_code == 200


# =============================================================================
# TRUST
# =============================================================================


class TestTrustCheckEndpoint:
    def test_missing_topic_returns_400(self, client):
        resp = client.get("/api/v1/trust")
        assert resp.status_code == 400

    @patch("valence.substrate.tools.trust.trust_check")
    def test_happy_path(self, mock_check, client):
        mock_check.return_value = {"success": True, "trusted_entities": []}
        resp = client.get("/api/v1/trust", params={"topic": "python"})
        assert resp.status_code == 200
