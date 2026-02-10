"""Tests for corroboration API endpoints.

Tests cover:
- GET /beliefs/{belief_id}/corroboration - Get corroboration details
- GET /beliefs/most-corroborated - Get most corroborated beliefs
- Error handling (404, 400, 500)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from valence.server.corroboration_endpoints import (
    belief_corroboration_endpoint,
    most_corroborated_beliefs_endpoint,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_corroboration_info():
    """Create mock corroboration info."""

    def _factory(
        belief_id: UUID | None = None,
        count: int = 3,
        confidence: float = 0.47,
        sources: list | None = None,
    ):
        return MagicMock(
            belief_id=belief_id or uuid4(),
            corroboration_count=count,
            confidence_corroboration=confidence,
            sources=sources or [{"source_did": "did:vkb:web:example.com", "similarity": 0.95}],
        )

    return _factory


@pytest.fixture
def app():
    """Create test Starlette app with corroboration endpoints."""
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
def client(app):
    """Create test client."""
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# BELIEF CORROBORATION ENDPOINT TESTS
# =============================================================================


class TestBeliefCorroborationEndpoint:
    """Tests for belief_corroboration_endpoint."""

    def test_corroboration_success(self, client, mock_corroboration_info):
        """Test successful corroboration retrieval."""
        belief_id = uuid4()

        with patch("oro_federation.corroboration.get_corroboration") as mock_get:
            mock_get.return_value = mock_corroboration_info(
                belief_id=belief_id,
                count=3,
                confidence=0.47,
            )

            response = client.get(f"/beliefs/{belief_id}/corroboration")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["belief_id"] == str(belief_id)
        assert data["corroboration_count"] == 3
        assert data["confidence_corroboration"] == 0.47
        assert data["confidence_label"] == "moderately corroborated"
        assert len(data["corroborating_sources"]) == 1

    def test_corroboration_uncorroborated(self, client, mock_corroboration_info):
        """Test uncorroborated belief."""
        belief_id = uuid4()

        with patch("oro_federation.corroboration.get_corroboration") as mock_get:
            mock_get.return_value = mock_corroboration_info(
                belief_id=belief_id,
                count=0,
                confidence=0.0,
                sources=[],
            )

            response = client.get(f"/beliefs/{belief_id}/corroboration")

        assert response.status_code == 200
        data = response.json()

        assert data["corroboration_count"] == 0
        assert data["confidence_label"] == "uncorroborated"

    def test_corroboration_single(self, client, mock_corroboration_info):
        """Test single corroboration."""
        belief_id = uuid4()

        with patch("oro_federation.corroboration.get_corroboration") as mock_get:
            mock_get.return_value = mock_corroboration_info(count=1)

            response = client.get(f"/beliefs/{belief_id}/corroboration")

        data = response.json()
        assert data["confidence_label"] == "single corroboration"

    def test_corroboration_well_corroborated(self, client, mock_corroboration_info):
        """Test well corroborated belief."""
        belief_id = uuid4()

        with patch("oro_federation.corroboration.get_corroboration") as mock_get:
            mock_get.return_value = mock_corroboration_info(count=5)

            response = client.get(f"/beliefs/{belief_id}/corroboration")

        data = response.json()
        assert data["confidence_label"] == "well corroborated"

    def test_corroboration_highly_corroborated(self, client, mock_corroboration_info):
        """Test highly corroborated belief."""
        belief_id = uuid4()

        with patch("oro_federation.corroboration.get_corroboration") as mock_get:
            mock_get.return_value = mock_corroboration_info(count=10)

            response = client.get(f"/beliefs/{belief_id}/corroboration")

        data = response.json()
        assert data["confidence_label"] == "highly corroborated"

    def test_corroboration_not_found(self, client):
        """Test 404 when belief not found."""
        belief_id = uuid4()

        with patch("oro_federation.corroboration.get_corroboration") as mock_get:
            mock_get.return_value = None

            response = client.get(f"/beliefs/{belief_id}/corroboration")

        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["error"]["message"].lower()

    def test_corroboration_invalid_uuid(self, client):
        """Test 400 for invalid UUID format."""
        response = client.get("/beliefs/not-a-valid-uuid/corroboration")

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "Invalid" in data["error"]["message"]

    def test_corroboration_internal_error(self, client):
        """Test 500 on internal error."""
        belief_id = uuid4()

        with patch("oro_federation.corroboration.get_corroboration") as mock_get:
            mock_get.side_effect = Exception("Database connection failed")

            response = client.get(f"/beliefs/{belief_id}/corroboration")

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False


# =============================================================================
# MOST CORROBORATED BELIEFS ENDPOINT TESTS
# =============================================================================


class TestMostCorroboratedBeliefsEndpoint:
    """Tests for most_corroborated_beliefs_endpoint."""

    def test_most_corroborated_success(self, client):
        """Test successful retrieval of most corroborated beliefs."""
        mock_beliefs = [
            {
                "id": str(uuid4()),
                "content": "Belief 1",
                "corroboration_count": 5,
                "confidence_corroboration": 0.6,
                "sources": [],
                "domain_path": ["tech"],
                "created_at": "2024-01-15T10:00:00",
            },
            {
                "id": str(uuid4()),
                "content": "Belief 2",
                "corroboration_count": 3,
                "confidence_corroboration": 0.47,
                "sources": [],
                "domain_path": [],
                "created_at": "2024-01-15T11:00:00",
            },
        ]

        with patch("oro_federation.corroboration.get_most_corroborated_beliefs") as mock_get:
            mock_get.return_value = mock_beliefs

            response = client.get("/beliefs/most-corroborated")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["beliefs"]) == 2
        assert data["total_count"] == 2
        # Should be ordered by corroboration count
        assert data["beliefs"][0]["corroboration_count"] == 5

    def test_most_corroborated_with_limit(self, client):
        """Test with custom limit."""
        with patch("oro_federation.corroboration.get_most_corroborated_beliefs") as mock_get:
            mock_get.return_value = []

            response = client.get("/beliefs/most-corroborated?limit=5")

        assert response.status_code == 200
        # Verify limit was passed to function
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["limit"] == 5

    def test_most_corroborated_with_min_count(self, client):
        """Test with minimum corroboration count."""
        with patch("oro_federation.corroboration.get_most_corroborated_beliefs") as mock_get:
            mock_get.return_value = []

            response = client.get("/beliefs/most-corroborated?min_count=3")

        assert response.status_code == 200
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["min_count"] == 3

    def test_most_corroborated_with_domain_filter(self, client):
        """Test with domain filter."""
        with patch("oro_federation.corroboration.get_most_corroborated_beliefs") as mock_get:
            mock_get.return_value = []

            response = client.get("/beliefs/most-corroborated?domain=tech")

        assert response.status_code == 200
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["domain_filter"] == ["tech"]

    def test_most_corroborated_empty_result(self, client):
        """Test when no corroborated beliefs exist."""
        with patch("oro_federation.corroboration.get_most_corroborated_beliefs") as mock_get:
            mock_get.return_value = []

            response = client.get("/beliefs/most-corroborated")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["beliefs"] == []
        assert data["total_count"] == 0

    def test_most_corroborated_internal_error(self, client):
        """Test 500 on internal error."""
        with patch("oro_federation.corroboration.get_most_corroborated_beliefs") as mock_get:
            mock_get.side_effect = Exception("Database error")

            response = client.get("/beliefs/most-corroborated")

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False


# =============================================================================
# CONFIDENCE LABEL TESTS
# =============================================================================


class TestConfidenceLabelLogic:
    """Test the confidence labeling logic via endpoint."""

    @pytest.mark.parametrize(
        "count,expected_label",
        [
            (0, "uncorroborated"),
            (1, "single corroboration"),
            (2, "moderately corroborated"),
            (3, "moderately corroborated"),
            (4, "well corroborated"),
            (5, "well corroborated"),
            (6, "well corroborated"),
            (7, "highly corroborated"),
            (10, "highly corroborated"),
            (100, "highly corroborated"),
        ],
    )
    def test_confidence_labels(self, client, mock_corroboration_info, count, expected_label):
        """Test all confidence label thresholds."""
        belief_id = uuid4()

        with patch("oro_federation.corroboration.get_corroboration") as mock_get:
            mock_get.return_value = mock_corroboration_info(count=count)

            response = client.get(f"/beliefs/{belief_id}/corroboration")

        data = response.json()
        assert data["confidence_label"] == expected_label
