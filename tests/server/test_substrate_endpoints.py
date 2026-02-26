"""Tests for substrate_endpoints.py (legacy belief/entity/tension REST endpoints)."""

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
# BELIEFS
# =============================================================================


class TestBeliefsListEndpoint:
    """Tests for GET /api/v1/beliefs (search/list)."""

    def test_requires_auth(self, client):
        """Test endpoint requires authentication."""
        response = client.get(f"{API_V1}/beliefs?query=test")
        assert response.status_code == 401

    def test_requires_query_param(self, client, auth_token):
        """Test endpoint requires query parameter."""
        response = client.get(f"{API_V1}/beliefs", headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "query" in data["error"]["message"].lower()

    @patch("valence.mcp.handlers.articles.article_search")
    def test_search_beliefs_success(self, mock_search, client, auth_token):
        """Test successful belief search."""
        mock_search.return_value = {
            "success": True,
            "results": [{"id": "art1", "content": "test belief"}],
        }

        response = client.get(
            f"{API_V1}/beliefs?query=test",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        mock_search.assert_called_once()

    @patch("valence.mcp.handlers.articles.article_search")
    def test_search_with_domain_filter(self, mock_search, client, auth_token):
        """Test search with domain filter."""
        mock_search.return_value = {"success": True, "results": []}

        response = client.get(
            f"{API_V1}/beliefs?query=test&domain_filter=tech,ai",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["domain_filter"] == ["tech", "ai"]

    @patch("valence.mcp.handlers.articles.article_search")
    def test_search_with_limit(self, mock_search, client, auth_token):
        """Test search respects limit parameter."""
        mock_search.return_value = {"success": True, "results": []}

        response = client.get(
            f"{API_V1}/beliefs?query=test&limit=50",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["limit"] == 50


class TestBeliefsCreateEndpoint:
    """Tests for POST /api/v1/beliefs."""

    def test_requires_auth(self, client):
        """Test endpoint requires authentication."""
        response = client.post(f"{API_V1}/beliefs", json={"content": "test"})
        assert response.status_code == 401

    def test_requires_content(self, client, auth_token):
        """Test endpoint requires content field."""
        response = client.post(
            f"{API_V1}/beliefs",
            json={},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "content" in data["error"]["message"].lower()

    @patch("valence.mcp.handlers.articles.article_create")
    def test_create_belief_success(self, mock_create, client, auth_token):
        """Test successful belief creation."""
        mock_create.return_value = {"success": True, "id": "art1"}

        response = client.post(
            f"{API_V1}/beliefs",
            json={"content": "New belief"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["id"] == "art1"
        mock_create.assert_called_once()

    @patch("valence.mcp.handlers.articles.article_create")
    def test_create_with_optional_fields(self, mock_create, client, auth_token):
        """Test creation with optional fields."""
        mock_create.return_value = {"success": True, "id": "art1"}

        response = client.post(
            f"{API_V1}/beliefs",
            json={
                "content": "New belief",
                "title": "Title",
                "source_ids": ["src1"],
                "domain_path": ["tech"],
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 201
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["title"] == "Title"
        assert call_kwargs["source_ids"] == ["src1"]
        assert call_kwargs["domain_path"] == ["tech"]

    def test_invalid_json(self, client, auth_token):
        """Test invalid JSON returns 400."""
        response = client.post(
            f"{API_V1}/beliefs",
            content="not json",
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 400


class TestBeliefsSearchEndpoint:
    """Tests for GET /api/v1/beliefs/search (semantic search)."""

    def test_requires_auth(self, client):
        """Test endpoint requires authentication."""
        response = client.get(f"{API_V1}/beliefs/search?query=test")
        assert response.status_code == 401

    def test_requires_query(self, client, auth_token):
        """Test endpoint requires query parameter."""
        response = client.get(
            f"{API_V1}/beliefs/search",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 400

    @patch("valence.mcp.handlers.articles.article_search")
    def test_semantic_search_success(self, mock_search, client, auth_token):
        """Test successful semantic search."""
        mock_search.return_value = {
            "success": True,
            "results": [{"id": "art1", "similarity": 0.95}],
        }

        response = client.get(
            f"{API_V1}/beliefs/search?query=semantic",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) > 0


class TestBeliefsGetEndpoint:
    """Tests for GET /api/v1/beliefs/{belief_id}."""

    def test_requires_auth(self, client):
        """Test endpoint requires authentication."""
        response = client.get(f"{API_V1}/beliefs/test-id")
        assert response.status_code == 401

    @patch("valence.mcp.handlers.articles.article_get")
    def test_get_belief_success(self, mock_get, client, auth_token):
        """Test successful belief retrieval."""
        mock_get.return_value = {
            "success": True,
            "article": {"id": "art1", "content": "test"},
        }

        response = client.get(
            f"{API_V1}/beliefs/art1",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_get.assert_called_once_with(article_id="art1", include_provenance=False)

    @patch("valence.mcp.handlers.articles.article_get")
    def test_get_belief_not_found(self, mock_get, client, auth_token):
        """Test 404 when belief not found."""
        mock_get.return_value = {"success": False, "error": "Not found"}

        response = client.get(
            f"{API_V1}/beliefs/missing",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 404

    @patch("valence.mcp.handlers.articles.article_get")
    def test_include_provenance(self, mock_get, client, auth_token):
        """Test include_provenance query param."""
        mock_get.return_value = {"success": True, "article": {}}

        response = client.get(
            f"{API_V1}/beliefs/art1?include_provenance=true",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        mock_get.assert_called_once_with(article_id="art1", include_provenance=True)


class TestBeliefsSupersede:
    """Tests for POST /api/v1/beliefs/{belief_id}/supersede."""

    def test_requires_auth(self, client):
        """Test endpoint requires authentication."""
        response = client.post(
            f"{API_V1}/beliefs/art1/supersede",
            json={"new_content": "updated", "reason": "correction"},
        )
        assert response.status_code == 401

    def test_requires_new_content(self, client, auth_token):
        """Test endpoint requires new_content."""
        response = client.post(
            f"{API_V1}/beliefs/art1/supersede",
            json={"reason": "update"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 400

    def test_requires_reason(self, client, auth_token):
        """Test endpoint requires reason."""
        response = client.post(
            f"{API_V1}/beliefs/art1/supersede",
            json={"new_content": "updated"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 400

    @patch("valence.mcp.handlers.articles.article_update")
    def test_supersede_success(self, mock_update, client, auth_token):
        """Test successful supersede."""
        mock_update.return_value = {"success": True, "id": "art2"}

        response = client.post(
            f"{API_V1}/beliefs/art1/supersede",
            json={"new_content": "updated belief", "reason": "correction"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_update.assert_called_once_with(article_id="art1", content="updated belief", epistemic_type=None)


# =============================================================================
# ENTITIES
# =============================================================================


class TestEntitiesListEndpoint:
    """Tests for GET /api/v1/entities."""

    def test_requires_auth(self, client):
        """Test endpoint requires authentication."""
        response = client.get(f"{API_V1}/entities?query=test")
        assert response.status_code == 401

    def test_requires_query(self, client, auth_token):
        """Test endpoint requires query parameter."""
        response = client.get(f"{API_V1}/entities", headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 400

    @patch("valence.mcp.handlers.entities.entity_search")
    def test_search_entities_success(self, mock_search, client, auth_token):
        """Test successful entity search."""
        mock_search.return_value = {
            "success": True,
            "entities": [{"id": "ent1", "name": "Test Entity"}],
        }

        response = client.get(
            f"{API_V1}/entities?query=test",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "entities" in data

    @patch("valence.mcp.handlers.entities.entity_search")
    def test_search_with_type_filter(self, mock_search, client, auth_token):
        """Test search with type filter."""
        mock_search.return_value = {"success": True, "entities": []}

        response = client.get(
            f"{API_V1}/entities?query=test&type=person",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["entity_type"] == "person"


class TestEntitiesGetEndpoint:
    """Tests for GET /api/v1/entities/{id}."""

    def test_requires_auth(self, client):
        """Test endpoint requires authentication."""
        response = client.get(f"{API_V1}/entities/ent1")
        assert response.status_code == 401

    @patch("valence.mcp.handlers.entities.entity_get")
    def test_get_entity_success(self, mock_get, client, auth_token):
        """Test successful entity retrieval."""
        mock_get.return_value = {
            "success": True,
            "entity": {"id": "ent1", "name": "Test"},
        }

        response = client.get(
            f"{API_V1}/entities/ent1",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch("valence.mcp.handlers.entities.entity_get")
    def test_get_entity_not_found(self, mock_get, client, auth_token):
        """Test 404 when entity not found."""
        mock_get.return_value = {"success": False}

        response = client.get(
            f"{API_V1}/entities/missing",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 404


# =============================================================================
# TENSIONS
# =============================================================================


class TestTensionsListEndpoint:
    """Tests for GET /api/v1/tensions."""

    def test_requires_auth(self, client):
        """Test endpoint requires authentication."""
        response = client.get(f"{API_V1}/tensions")
        assert response.status_code == 401

    @patch("valence.mcp.handlers.contention.contention_list")
    def test_list_tensions_success(self, mock_list, client, auth_token):
        """Test successful tensions list."""
        mock_list.return_value = {
            "success": True,
            "contentions": [{"id": "con1", "status": "detected"}],
        }

        response = client.get(f"{API_V1}/tensions", headers={"Authorization": f"Bearer {auth_token}"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch("valence.mcp.handlers.contention.contention_list")
    def test_filter_by_status(self, mock_list, client, auth_token):
        """Test filtering tensions by status."""
        mock_list.return_value = {"success": True, "contentions": []}

        response = client.get(
            f"{API_V1}/tensions?status=resolved",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["status"] == "resolved"


class TestTensionsResolveEndpoint:
    """Tests for POST /api/v1/tensions/{id}/resolve."""

    def test_requires_auth(self, client):
        """Test endpoint requires authentication."""
        response = client.post(
            f"{API_V1}/tensions/con1/resolve",
            json={"resolution": "fixed", "action": "merge"},
        )
        assert response.status_code == 401

    def test_requires_resolution(self, client, auth_token):
        """Test endpoint requires resolution field."""
        response = client.post(
            f"{API_V1}/tensions/con1/resolve",
            json={"action": "merge"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 400

    def test_requires_action(self, client, auth_token):
        """Test endpoint requires action field."""
        response = client.post(
            f"{API_V1}/tensions/con1/resolve",
            json={"resolution": "fixed"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 400

    @patch("valence.mcp.handlers.contention.contention_resolve")
    def test_resolve_tension_success(self, mock_resolve, client, auth_token):
        """Test successful tension resolution."""
        mock_resolve.return_value = {"success": True}

        response = client.post(
            f"{API_V1}/tensions/con1/resolve",
            json={"resolution": "Fixed conflict", "action": "merge"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_resolve.assert_called_once_with(contention_id="con1", resolution="merge", rationale="Fixed conflict")


# =============================================================================
# STATS
# =============================================================================


class TestStatsEndpoint:
    """Tests for GET /api/v1/stats."""

    def test_requires_auth(self, client):
        """Test endpoint requires authentication."""
        response = client.get(f"{API_V1}/stats")
        assert response.status_code == 401

    def test_stats_success(self, client, auth_token, mock_db):
        """Test successful stats retrieval."""
        # Mock database responses
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
        assert "stats" in data
        stats = data["stats"]
        assert stats["total_articles"] == 100
        assert stats["active_articles"] == 80
        assert stats["with_embeddings"] == 60
        assert stats["unresolved_contentions"] == 5

    def test_stats_json_format(self, client, auth_token, mock_db):
        """Test stats in JSON format."""
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


# =============================================================================
# CONFLICTS
# =============================================================================


class TestConflictsEndpoint:
    """Tests for GET /api/v1/beliefs/conflicts."""

    def test_requires_auth(self, client):
        """Test endpoint requires authentication."""
        response = client.get(f"{API_V1}/beliefs/conflicts")
        assert response.status_code == 401

    def test_conflicts_detection(self, client, auth_token, mock_db):
        """Test conflict detection."""
        # Mock database to return no pairs
        mock_db.fetchall.return_value = []

        response = client.get(
            f"{API_V1}/beliefs/conflicts",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "conflicts" in data
        assert data["count"] == 0

    def test_conflicts_with_threshold(self, client, auth_token, mock_db):
        """Test conflicts with custom threshold."""
        mock_db.fetchall.return_value = []

        response = client.get(
            f"{API_V1}/beliefs/conflicts?threshold=0.9",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["threshold"] == 0.9

    def test_conflicts_auto_record(self, client, auth_token, mock_db):
        """Test auto-recording contentions."""
        mock_db.fetchall.return_value = []

        response = client.get(
            f"{API_V1}/beliefs/conflicts?auto_record=true",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "recorded_contentions" in data


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================


class TestBeliefsListEdgeCases:
    """Edge case tests for beliefs list endpoint."""

    @patch("valence.mcp.handlers.articles.article_search")
    def test_invalid_ranking_json(self, mock_search, client, auth_token):
        """Test invalid ranking parameter."""
        response = client.get(
            f"{API_V1}/beliefs?query=test&ranking=not-json",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "ranking" in data["error"]["message"].lower()

    @patch("valence.mcp.handlers.articles.article_search")
    def test_ranking_not_dict(self, mock_search, client, auth_token):
        """Test ranking parameter that's not a dict."""
        response = client.get(
            f"{API_V1}/beliefs?query=test&ranking=[]",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 400

    @patch("valence.mcp.handlers.articles.article_search")
    def test_internal_error_handling(self, mock_search, client, auth_token):
        """Test internal error handling."""
        mock_search.side_effect = Exception("Database error")

        response = client.get(
            f"{API_V1}/beliefs?query=test",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 500


class TestBeliefsCreateEdgeCases:
    """Edge case tests for beliefs create endpoint."""

    @patch("valence.mcp.handlers.articles.article_create")
    def test_create_failure(self, mock_create, client, auth_token):
        """Test creation failure returns 400."""
        mock_create.return_value = {"success": False, "error": "Validation failed"}

        response = client.post(
            f"{API_V1}/beliefs",
            json={"content": "test"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 400

    @patch("valence.mcp.handlers.articles.article_create")
    def test_create_internal_error(self, mock_create, client, auth_token):
        """Test internal error on create."""
        mock_create.side_effect = Exception("DB error")

        response = client.post(
            f"{API_V1}/beliefs",
            json={"content": "test"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 500


class TestStatsTextFormat:
    """Test stats endpoint with text format."""

    def test_stats_text_format(self, client, auth_token, mock_db):
        """Test stats with text output format."""
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


class TestConflictsTextFormat:
    """Test conflicts endpoint with text format."""

    def test_conflicts_text_format(self, client, auth_token, mock_db):
        """Test conflicts with text output format."""
        mock_db.fetchall.return_value = []

        response = client.get(
            f"{API_V1}/beliefs/conflicts?output=text",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        assert b"No potential conflicts" in response.content
