"""Tests for contention MCP handlers."""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

from valence.core.response import ValenceResponse
from valence.mcp.handlers.contention import contention_detect, contention_list, contention_resolve


class TestContentionDetect:
    """Tests for contention_detect handler."""

    def test_detect_success(self):
        """Test successful contention detection."""
        contention_data = {
            "id": str(uuid4()),
            "article_id": str(uuid4()),
            "source_id": str(uuid4()),
            "severity": "high",
            "description": "Test contention",
        }
        mock_response = ValenceResponse(success=True, data=contention_data)

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_detect(
                article_id=str(uuid4()),
                source_id=str(uuid4()),
            )

        assert result["success"] is True
        assert result["contention"]["id"] == contention_data["id"]
        assert result["contention"]["severity"] == "high"

    def test_detect_failure(self):
        """Test contention detection failure."""
        mock_response = ValenceResponse(
            success=False,
            error="Detection failed",
        )

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_detect(
                article_id=str(uuid4()),
                source_id=str(uuid4()),
            )

        assert result["success"] is False
        assert "Detection failed" in result["error"]

    def test_detect_article_not_found(self):
        """Test detection with non-existent article."""
        mock_response = ValenceResponse(
            success=False,
            error="Article not found",
        )

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_detect(
                article_id=str(uuid4()),
                source_id=str(uuid4()),
            )

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_detect_source_not_found(self):
        """Test detection with non-existent source."""
        mock_response = ValenceResponse(
            success=False,
            error="Source not found",
        )

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_detect(
                article_id=str(uuid4()),
                source_id=str(uuid4()),
            )

        assert result["success"] is False
        assert "not found" in result["error"]


class TestContentionList:
    """Tests for contention_list handler."""

    def test_list_all_success(self):
        """Test listing all contentions."""
        contentions = [
            {
                "id": str(uuid4()),
                "article_id": str(uuid4()),
                "status": "active",
                "severity": "medium",
            },
            {
                "id": str(uuid4()),
                "article_id": str(uuid4()),
                "status": "active",
                "severity": "high",
            },
        ]
        mock_response = ValenceResponse(success=True, data=contentions)

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_list()

        assert result["success"] is True
        assert len(result["contentions"]) == 2
        assert result["total_count"] == 2

    def test_list_by_article_id(self):
        """Test listing contentions for specific article."""
        article_id = str(uuid4())
        contentions = [
            {
                "id": str(uuid4()),
                "article_id": article_id,
                "status": "active",
            }
        ]
        mock_response = ValenceResponse(success=True, data=contentions)

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_list(article_id=article_id)

        assert result["success"] is True
        assert len(result["contentions"]) == 1
        assert result["contentions"][0]["article_id"] == article_id

    def test_list_by_status(self):
        """Test listing contentions by status."""
        contentions = [
            {
                "id": str(uuid4()),
                "status": "resolved",
            }
        ]
        mock_response = ValenceResponse(success=True, data=contentions)

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_list(status="resolved")

        assert result["success"] is True
        assert result["contentions"][0]["status"] == "resolved"

    def test_list_by_article_and_status(self):
        """Test listing contentions by article_id and status."""
        article_id = str(uuid4())
        contentions = [
            {
                "id": str(uuid4()),
                "article_id": article_id,
                "status": "active",
            }
        ]
        mock_response = ValenceResponse(success=True, data=contentions)

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_list(article_id=article_id, status="active")

        assert result["success"] is True
        assert len(result["contentions"]) == 1

    def test_list_empty_results(self):
        """Test listing with no contentions."""
        mock_response = ValenceResponse(success=True, data=[])

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_list()

        assert result["success"] is True
        assert len(result["contentions"]) == 0
        assert result["total_count"] == 0

    def test_list_none_data(self):
        """Test listing when data is None."""
        mock_response = ValenceResponse(success=True, data=None)

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_list()

        assert result["success"] is True
        assert result["contentions"] == []
        assert result["total_count"] == 0

    def test_list_failure(self):
        """Test listing contentions failure."""
        mock_response = ValenceResponse(
            success=False,
            error="Database error",
        )

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_list()

        assert result["success"] is False
        assert "Database error" in result["error"]


class TestContentionResolve:
    """Tests for contention_resolve handler."""

    def test_resolve_success(self):
        """Test successful contention resolution."""
        contention_data = {
            "id": str(uuid4()),
            "status": "resolved",
            "resolution": "accepted",
            "rationale": "Source is reliable",
        }
        mock_response = ValenceResponse(success=True, data=contention_data)

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_resolve(
                contention_id=str(uuid4()),
                resolution="accepted",
                rationale="Source is reliable",
            )

        assert result["success"] is True
        assert result["contention"]["status"] == "resolved"
        assert result["contention"]["resolution"] == "accepted"

    def test_resolve_failure(self):
        """Test contention resolution failure."""
        mock_response = ValenceResponse(
            success=False,
            error="Resolution failed",
        )

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_resolve(
                contention_id=str(uuid4()),
                resolution="rejected",
                rationale="Source unreliable",
            )

        assert result["success"] is False
        assert "Resolution failed" in result["error"]

    def test_resolve_not_found(self):
        """Test resolving non-existent contention."""
        mock_response = ValenceResponse(
            success=False,
            error="Contention not found",
        )

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_resolve(
                contention_id=str(uuid4()),
                resolution="accepted",
                rationale="Test",
            )

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_resolve_invalid_resolution(self):
        """Test resolving with invalid resolution type."""
        mock_response = ValenceResponse(
            success=False,
            error="Invalid resolution type",
        )

        with patch("valence.mcp.handlers.contention.run_async", return_value=mock_response):
            result = contention_resolve(
                contention_id=str(uuid4()),
                resolution="invalid",
                rationale="Test",
            )

        assert result["success"] is False
        assert "Invalid" in result["error"]
