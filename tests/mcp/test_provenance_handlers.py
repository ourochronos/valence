"""Tests for provenance MCP handlers."""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest

from valence.core.response import ValenceResponse
from valence.mcp.handlers.provenance import provenance_get, provenance_link, provenance_trace


class TestProvenanceLink:
    """Tests for provenance_link handler."""

    def test_link_success(self):
        """Test successful provenance link creation."""
        link_data = {
            "id": str(uuid4()),
            "article_id": str(uuid4()),
            "source_id": str(uuid4()),
            "relationship": "confirms",
        }
        mock_response = ValenceResponse(success=True, data=link_data)

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_link(
                article_id=str(uuid4()),
                source_id=str(uuid4()),
                relationship="confirms",
            )

        assert result["success"] is True
        assert result["link"]["relationship"] == "confirms"

    def test_link_default_relationship(self):
        """Test link creation with default relationship."""
        link_data = {
            "id": str(uuid4()),
            "article_id": str(uuid4()),
            "source_id": str(uuid4()),
            "relationship": "confirms",
        }
        mock_response = ValenceResponse(success=True, data=link_data)

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_link(
                article_id=str(uuid4()),
                source_id=str(uuid4()),
            )

        assert result["success"] is True
        assert result["link"]["relationship"] == "confirms"

    def test_link_custom_relationship(self):
        """Test link creation with custom relationship."""
        link_data = {
            "id": str(uuid4()),
            "article_id": str(uuid4()),
            "source_id": str(uuid4()),
            "relationship": "contradicts",
        }
        mock_response = ValenceResponse(success=True, data=link_data)

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_link(
                article_id=str(uuid4()),
                source_id=str(uuid4()),
                relationship="contradicts",
            )

        assert result["success"] is True
        assert result["link"]["relationship"] == "contradicts"

    def test_link_failure(self):
        """Test provenance link creation failure."""
        mock_response = ValenceResponse(
            success=False,
            error="Link creation failed",
        )

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_link(
                article_id=str(uuid4()),
                source_id=str(uuid4()),
            )

        assert result["success"] is False
        assert "failed" in result["error"]

    def test_link_article_not_found(self):
        """Test linking with non-existent article."""
        mock_response = ValenceResponse(
            success=False,
            error="Article not found",
        )

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_link(
                article_id=str(uuid4()),
                source_id=str(uuid4()),
            )

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_link_source_not_found(self):
        """Test linking with non-existent source."""
        mock_response = ValenceResponse(
            success=False,
            error="Source not found",
        )

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_link(
                article_id=str(uuid4()),
                source_id=str(uuid4()),
            )

        assert result["success"] is False
        assert "not found" in result["error"]


class TestProvenanceGet:
    """Tests for provenance_get handler."""

    def test_get_success(self):
        """Test successful provenance retrieval."""
        provenance_data = [
            {
                "source_id": str(uuid4()),
                "relationship": "confirms",
                "confidence": 0.9,
            },
            {
                "source_id": str(uuid4()),
                "relationship": "supports",
                "confidence": 0.8,
            },
        ]
        mock_response = ValenceResponse(success=True, data=provenance_data)

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_get(article_id=str(uuid4()))

        assert result["success"] is True
        assert len(result["provenance"]) == 2
        assert result["count"] == 2

    def test_get_empty_provenance(self):
        """Test getting provenance with no sources."""
        mock_response = ValenceResponse(success=True, data=[])

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_get(article_id=str(uuid4()))

        assert result["success"] is True
        assert len(result["provenance"]) == 0
        assert result["count"] == 0

    def test_get_none_data(self):
        """Test getting provenance when data is None."""
        mock_response = ValenceResponse(success=True, data=None)

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_get(article_id=str(uuid4()))

        assert result["success"] is True
        assert result["provenance"] == []
        assert result["count"] == 0

    def test_get_failure(self):
        """Test provenance retrieval failure."""
        mock_response = ValenceResponse(
            success=False,
            error="Retrieval failed",
        )

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_get(article_id=str(uuid4()))

        assert result["success"] is False
        assert "failed" in result["error"]

    def test_get_article_not_found(self):
        """Test getting provenance for non-existent article."""
        mock_response = ValenceResponse(
            success=False,
            error="Article not found",
        )

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_get(article_id=str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]


class TestProvenanceTrace:
    """Tests for provenance_trace handler."""

    def test_trace_success(self):
        """Test successful claim tracing."""
        sources_data = [
            {
                "source_id": str(uuid4()),
                "title": "Source 1",
                "confidence": 0.95,
            },
            {
                "source_id": str(uuid4()),
                "title": "Source 2",
                "confidence": 0.85,
            },
        ]
        mock_response = ValenceResponse(success=True, data=sources_data)

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_trace(
                article_id=str(uuid4()),
                claim_text="Test claim",
            )

        assert result["success"] is True
        assert len(result["sources"]) == 2
        assert result["count"] == 2

    def test_trace_no_sources_found(self):
        """Test tracing claim with no matching sources."""
        mock_response = ValenceResponse(success=True, data=[])

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_trace(
                article_id=str(uuid4()),
                claim_text="Unique claim",
            )

        assert result["success"] is True
        assert len(result["sources"]) == 0
        assert result["count"] == 0

    def test_trace_none_data(self):
        """Test tracing when data is None."""
        mock_response = ValenceResponse(success=True, data=None)

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_trace(
                article_id=str(uuid4()),
                claim_text="Test claim",
            )

        assert result["success"] is True
        assert result["sources"] == []
        assert result["count"] == 0

    def test_trace_failure(self):
        """Test claim tracing failure."""
        mock_response = ValenceResponse(
            success=False,
            error="Trace failed",
        )

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_trace(
                article_id=str(uuid4()),
                claim_text="Test claim",
            )

        assert result["success"] is False
        assert "failed" in result["error"]

    def test_trace_article_not_found(self):
        """Test tracing for non-existent article."""
        mock_response = ValenceResponse(
            success=False,
            error="Article not found",
        )

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_trace(
                article_id=str(uuid4()),
                claim_text="Test claim",
            )

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_trace_single_source(self):
        """Test tracing claim to single source."""
        sources_data = [
            {
                "source_id": str(uuid4()),
                "title": "Primary Source",
                "confidence": 1.0,
            }
        ]
        mock_response = ValenceResponse(success=True, data=sources_data)

        with patch("valence.mcp.handlers.provenance.run_async", return_value=mock_response):
            result = provenance_trace(
                article_id=str(uuid4()),
                claim_text="Specific claim",
            )

        assert result["success"] is True
        assert len(result["sources"]) == 1
        assert result["sources"][0]["confidence"] == 1.0
