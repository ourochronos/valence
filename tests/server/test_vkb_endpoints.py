"""Tests for VKB REST endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from valence.server.auth_helpers import AuthenticatedClient
from valence.server.vkb_endpoints import (
    exchanges_add_endpoint,
    exchanges_list_endpoint,
    insights_extract_endpoint,
    insights_list_endpoint,
    patterns_create_endpoint,
    patterns_list_endpoint,
    patterns_reinforce_endpoint,
    patterns_search_endpoint,
    sessions_by_room_endpoint,
    sessions_create_endpoint,
    sessions_end_endpoint,
    sessions_get_endpoint,
    sessions_list_endpoint,
)

MOCK_CLIENT = AuthenticatedClient(client_id="test", auth_method="bearer")


@pytest.fixture
def app():
    routes = [
        Route("/api/v1/sessions", sessions_create_endpoint, methods=["POST"]),
        Route("/api/v1/sessions", sessions_list_endpoint, methods=["GET"]),
        Route("/api/v1/sessions/by-room/{room_id}", sessions_by_room_endpoint, methods=["GET"]),
        Route("/api/v1/sessions/{id}", sessions_get_endpoint, methods=["GET"]),
        Route("/api/v1/sessions/{id}/end", sessions_end_endpoint, methods=["POST"]),
        Route("/api/v1/sessions/{id}/exchanges", exchanges_add_endpoint, methods=["POST"]),
        Route("/api/v1/sessions/{id}/exchanges", exchanges_list_endpoint, methods=["GET"]),
        Route("/api/v1/sessions/{id}/insights", insights_extract_endpoint, methods=["POST"]),
        Route("/api/v1/sessions/{id}/insights", insights_list_endpoint, methods=["GET"]),
        Route("/api/v1/patterns", patterns_create_endpoint, methods=["POST"]),
        Route("/api/v1/patterns", patterns_list_endpoint, methods=["GET"]),
        Route("/api/v1/patterns/search", patterns_search_endpoint, methods=["GET"]),
        Route("/api/v1/patterns/{id}/reinforce", patterns_reinforce_endpoint, methods=["POST"]),
    ]
    return Starlette(routes=routes)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_auth():
    with patch("valence.server.vkb_endpoints.authenticate", return_value=MOCK_CLIENT):
        yield


# =============================================================================
# AUTH
# =============================================================================


class TestVkbAuth:
    def test_unauthenticated_returns_401(self, client):
        from starlette.responses import JSONResponse

        with patch(
            "valence.server.vkb_endpoints.authenticate",
            return_value=JSONResponse({"error": "unauthorized"}, status_code=401),
        ):
            resp = client.get("/api/v1/sessions")
            assert resp.status_code == 401

    def test_wrong_scope_returns_403(self, client):
        oauth_client = AuthenticatedClient(client_id="test", auth_method="oauth", scope="substrate:read")
        with patch("valence.server.vkb_endpoints.authenticate", return_value=oauth_client):
            resp = client.get("/api/v1/sessions")
            assert resp.status_code == 403


# =============================================================================
# SESSIONS
# =============================================================================


class TestSessionsCreateEndpoint:
    def test_missing_platform_returns_400(self, client):
        resp = client.post("/api/v1/sessions", json={})
        assert resp.status_code == 400

    @patch("valence.vkb.tools.sessions.session_start")
    def test_happy_path(self, mock_start, client):
        mock_start.return_value = {"success": True, "session": {"id": "sess-1"}}
        resp = client.post("/api/v1/sessions", json={"platform": "api"})
        assert resp.status_code == 201
        assert resp.json()["success"] is True
        mock_start.assert_called_once()
        assert mock_start.call_args.kwargs["platform"] == "api"

    @patch("valence.vkb.tools.sessions.session_start")
    def test_with_optional_fields(self, mock_start, client):
        mock_start.return_value = {"success": True, "session": {"id": "sess-1"}}
        resp = client.post(
            "/api/v1/sessions",
            json={
                "platform": "openclaw",
                "external_room_id": "room-42",
                "project_context": "test-project",
                "metadata": {"key": "value"},
            },
        )
        assert resp.status_code == 201
        kwargs = mock_start.call_args.kwargs
        assert kwargs["platform"] == "openclaw"
        assert kwargs["external_room_id"] == "room-42"
        assert kwargs["project_context"] == "test-project"
        assert kwargs["metadata"] == {"key": "value"}


class TestSessionsListEndpoint:
    @patch("valence.vkb.tools.sessions.session_list")
    def test_happy_path(self, mock_list, client):
        mock_list.return_value = {"success": True, "sessions": []}
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 200

    @patch("valence.vkb.tools.sessions.session_list")
    def test_with_filters(self, mock_list, client):
        mock_list.return_value = {"success": True, "sessions": []}
        resp = client.get("/api/v1/sessions", params={"platform": "api", "status": "active", "limit": "5"})
        assert resp.status_code == 200
        kwargs = mock_list.call_args.kwargs
        assert kwargs["platform"] == "api"
        assert kwargs["status"] == "active"
        assert kwargs["limit"] == 5


class TestSessionsByRoomEndpoint:
    @patch("valence.vkb.tools.sessions.session_find_by_room")
    def test_happy_path(self, mock_find, client):
        mock_find.return_value = {"success": True, "session": {"id": "sess-1"}}
        resp = client.get("/api/v1/sessions/by-room/room-42")
        assert resp.status_code == 200
        assert mock_find.call_args.kwargs["external_room_id"] == "room-42"

    @patch("valence.vkb.tools.sessions.session_find_by_room")
    def test_not_found(self, mock_find, client):
        mock_find.return_value = {"success": False, "error": "No active session found"}
        resp = client.get("/api/v1/sessions/by-room/nonexistent")
        assert resp.status_code == 404


class TestSessionsGetEndpoint:
    @patch("valence.vkb.tools.sessions.session_get")
    def test_happy_path(self, mock_get, client):
        mock_get.return_value = {"success": True, "session": {"id": "sess-1"}}
        resp = client.get("/api/v1/sessions/sess-1")
        assert resp.status_code == 200

    @patch("valence.vkb.tools.sessions.session_get")
    def test_not_found(self, mock_get, client):
        mock_get.return_value = {"success": False, "error": "Session not found"}
        resp = client.get("/api/v1/sessions/nonexistent")
        assert resp.status_code == 404

    @patch("valence.vkb.tools.sessions.session_get")
    def test_with_exchanges(self, mock_get, client):
        mock_get.return_value = {"success": True, "session": {"id": "sess-1"}, "exchanges": []}
        resp = client.get("/api/v1/sessions/sess-1", params={"include_exchanges": "true", "exchange_limit": "5"})
        assert resp.status_code == 200
        kwargs = mock_get.call_args.kwargs
        assert kwargs["include_exchanges"] is True
        assert kwargs["exchange_limit"] == 5


class TestSessionsEndEndpoint:
    @patch("valence.vkb.tools.sessions.session_end")
    def test_happy_path(self, mock_end, client):
        mock_end.return_value = {"success": True}
        resp = client.post("/api/v1/sessions/sess-1/end", json={"summary": "Done", "themes": ["testing"]})
        assert resp.status_code == 200
        kwargs = mock_end.call_args.kwargs
        assert kwargs["session_id"] == "sess-1"
        assert kwargs["summary"] == "Done"
        assert kwargs["themes"] == ["testing"]

    @patch("valence.vkb.tools.sessions.session_end")
    def test_not_found(self, mock_end, client):
        mock_end.return_value = {"success": False, "error": "Session not found"}
        resp = client.post("/api/v1/sessions/nonexistent/end", json={})
        assert resp.status_code == 404


# =============================================================================
# EXCHANGES
# =============================================================================


class TestExchangesAddEndpoint:
    def test_missing_role_returns_400(self, client):
        resp = client.post("/api/v1/sessions/sess-1/exchanges", json={"content": "hello"})
        assert resp.status_code == 400

    def test_missing_content_returns_400(self, client):
        resp = client.post("/api/v1/sessions/sess-1/exchanges", json={"role": "user"})
        assert resp.status_code == 400

    @patch("valence.vkb.tools.exchanges.exchange_add")
    def test_happy_path(self, mock_add, client):
        mock_add.return_value = {"success": True, "exchange": {"id": "ex-1"}}
        resp = client.post(
            "/api/v1/sessions/sess-1/exchanges",
            json={"role": "user", "content": "Hello world"},
        )
        assert resp.status_code == 201
        kwargs = mock_add.call_args.kwargs
        assert kwargs["session_id"] == "sess-1"
        assert kwargs["role"] == "user"
        assert kwargs["content"] == "Hello world"

    @patch("valence.vkb.tools.exchanges.exchange_add")
    def test_with_optional_fields(self, mock_add, client):
        mock_add.return_value = {"success": True, "exchange": {"id": "ex-1"}}
        resp = client.post(
            "/api/v1/sessions/sess-1/exchanges",
            json={"role": "assistant", "content": "Hi", "tokens_approx": 100, "tool_uses": [{"name": "read"}]},
        )
        assert resp.status_code == 201
        kwargs = mock_add.call_args.kwargs
        assert kwargs["tokens_approx"] == 100
        assert kwargs["tool_uses"] == [{"name": "read"}]


class TestExchangesListEndpoint:
    @patch("valence.vkb.tools.exchanges.exchange_list")
    def test_happy_path(self, mock_list, client):
        mock_list.return_value = {"success": True, "exchanges": []}
        resp = client.get("/api/v1/sessions/sess-1/exchanges")
        assert resp.status_code == 200
        assert mock_list.call_args.kwargs["session_id"] == "sess-1"

    @patch("valence.vkb.tools.exchanges.exchange_list")
    def test_with_pagination(self, mock_list, client):
        mock_list.return_value = {"success": True, "exchanges": []}
        resp = client.get("/api/v1/sessions/sess-1/exchanges", params={"limit": "10", "offset": "5"})
        assert resp.status_code == 200
        kwargs = mock_list.call_args.kwargs
        assert kwargs["limit"] == 10
        assert kwargs["offset"] == 5


# =============================================================================
# INSIGHTS
# =============================================================================


class TestInsightsExtractEndpoint:
    def test_missing_content_returns_400(self, client):
        resp = client.post("/api/v1/sessions/sess-1/insights", json={})
        assert resp.status_code == 400

    @patch("valence.vkb.tools.insights.insight_extract")
    def test_happy_path(self, mock_extract, client):
        mock_extract.return_value = {"success": True, "belief_id": "bel-1"}
        resp = client.post(
            "/api/v1/sessions/sess-1/insights",
            json={"content": "Python is great for scripting"},
        )
        assert resp.status_code == 201
        kwargs = mock_extract.call_args.kwargs
        assert kwargs["session_id"] == "sess-1"
        assert kwargs["content"] == "Python is great for scripting"

    @patch("valence.vkb.tools.insights.insight_extract")
    def test_with_optional_fields(self, mock_extract, client):
        mock_extract.return_value = {"success": True, "belief_id": "bel-1"}
        resp = client.post(
            "/api/v1/sessions/sess-1/insights",
            json={
                "content": "Test insight",
                "domain_path": ["tech", "python"],
                "confidence": {"overall": 0.9},
                "entities": [{"name": "Python", "type": "tool"}],
            },
        )
        assert resp.status_code == 201
        kwargs = mock_extract.call_args.kwargs
        assert kwargs["domain_path"] == ["tech", "python"]
        assert kwargs["confidence"] == {"overall": 0.9}


class TestInsightsListEndpoint:
    @patch("valence.vkb.tools.insights.insight_list")
    def test_happy_path(self, mock_list, client):
        mock_list.return_value = {"success": True, "insights": []}
        resp = client.get("/api/v1/sessions/sess-1/insights")
        assert resp.status_code == 200
        assert mock_list.call_args.kwargs["session_id"] == "sess-1"


# =============================================================================
# PATTERNS
# =============================================================================


class TestPatternsCreateEndpoint:
    def test_missing_type_returns_400(self, client):
        resp = client.post("/api/v1/patterns", json={"description": "test"})
        assert resp.status_code == 400

    def test_missing_description_returns_400(self, client):
        resp = client.post("/api/v1/patterns", json={"type": "preference"})
        assert resp.status_code == 400

    @patch("valence.vkb.tools.patterns.pattern_record")
    def test_happy_path(self, mock_record, client):
        mock_record.return_value = {"success": True, "pattern": {"id": "pat-1"}}
        resp = client.post(
            "/api/v1/patterns",
            json={"type": "preference", "description": "Prefers dark mode"},
        )
        assert resp.status_code == 201
        kwargs = mock_record.call_args.kwargs
        assert kwargs["type"] == "preference"
        assert kwargs["description"] == "Prefers dark mode"

    @patch("valence.vkb.tools.patterns.pattern_record")
    def test_with_optional_fields(self, mock_record, client):
        mock_record.return_value = {"success": True, "pattern": {"id": "pat-1"}}
        resp = client.post(
            "/api/v1/patterns",
            json={
                "type": "working_style",
                "description": "Prefers TDD",
                "evidence": ["sess-1", "sess-2"],
                "confidence": 0.8,
            },
        )
        assert resp.status_code == 201
        kwargs = mock_record.call_args.kwargs
        assert kwargs["evidence"] == ["sess-1", "sess-2"]
        assert kwargs["confidence"] == 0.8


class TestPatternsListEndpoint:
    @patch("valence.vkb.tools.patterns.pattern_list")
    def test_happy_path(self, mock_list, client):
        mock_list.return_value = {"success": True, "patterns": []}
        resp = client.get("/api/v1/patterns")
        assert resp.status_code == 200

    @patch("valence.vkb.tools.patterns.pattern_list")
    def test_with_filters(self, mock_list, client):
        mock_list.return_value = {"success": True, "patterns": []}
        resp = client.get("/api/v1/patterns", params={"type": "preference", "status": "established", "limit": "5"})
        assert resp.status_code == 200
        kwargs = mock_list.call_args.kwargs
        assert kwargs["type"] == "preference"
        assert kwargs["status"] == "established"
        assert kwargs["limit"] == 5


class TestPatternsSearchEndpoint:
    def test_missing_query_returns_400(self, client):
        resp = client.get("/api/v1/patterns/search")
        assert resp.status_code == 400

    @patch("valence.vkb.tools.patterns.pattern_search")
    def test_happy_path(self, mock_search, client):
        mock_search.return_value = {"success": True, "patterns": []}
        resp = client.get("/api/v1/patterns/search", params={"query": "dark mode"})
        assert resp.status_code == 200
        mock_search.assert_called_once()


class TestPatternsReinforceEndpoint:
    @patch("valence.vkb.tools.patterns.pattern_reinforce")
    def test_happy_path(self, mock_reinforce, client):
        mock_reinforce.return_value = {"success": True}
        resp = client.post("/api/v1/patterns/pat-1/reinforce", json={"session_id": "sess-1"})
        assert resp.status_code == 200
        kwargs = mock_reinforce.call_args.kwargs
        assert kwargs["pattern_id"] == "pat-1"
        assert kwargs["session_id"] == "sess-1"

    @patch("valence.vkb.tools.patterns.pattern_reinforce")
    def test_not_found(self, mock_reinforce, client):
        mock_reinforce.return_value = {"success": False, "error": "Pattern not found"}
        resp = client.post("/api/v1/patterns/nonexistent/reinforce", json={})
        assert resp.status_code == 404
