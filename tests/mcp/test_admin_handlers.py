"""Tests for admin MCP handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from valence.core.response import ValenceResponse
from valence.mcp.handlers.admin import admin_forget, admin_maintenance, admin_stats


class TestAdminForget:
    """Tests for admin_forget handler."""

    def test_forget_source_success(self):
        """Test successfully forgetting a source."""
        source_id = str(uuid4())
        mock_response = ValenceResponse(
            success=True,
            data={"removed_id": source_id, "type": "source"},
        )

        with patch("valence.core.forgetting.remove_source"):
            with patch("valence.mcp.handlers.admin.run_async") as mock_run:
                mock_run.return_value = mock_response
                result = admin_forget("source", source_id)

        assert result["success"] is True
        assert result["removed_id"] == source_id

    def test_forget_article_success(self):
        """Test successfully forgetting an article."""
        article_id = str(uuid4())
        mock_response = ValenceResponse(
            success=True,
            data={"removed_id": article_id, "type": "article"},
        )

        with patch("valence.core.forgetting.remove_article"):
            with patch("valence.mcp.handlers.admin.run_async") as mock_run:
                mock_run.return_value = mock_response
                result = admin_forget("article", article_id)

        assert result["success"] is True
        assert result["removed_id"] == article_id

    def test_forget_invalid_target_type(self):
        """Test admin_forget with invalid target_type."""
        result = admin_forget("invalid_type", str(uuid4()))

        assert result["success"] is False
        assert "must be 'source' or 'article'" in result["error"]

    def test_forget_source_not_found(self):
        """Test forgetting non-existent source."""
        mock_response = ValenceResponse(
            success=False,
            error="Source not found",
        )

        with patch("valence.mcp.handlers.admin.run_async", return_value=mock_response):
            result = admin_forget("source", str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_forget_exception_handling(self):
        """Test admin_forget exception handling."""
        with patch("valence.mcp.handlers.admin.run_async", side_effect=Exception("Database error")):
            result = admin_forget("source", str(uuid4()))

        assert result["success"] is False
        assert "Database error" in result["error"]


class TestAdminStats:
    """Tests for admin_stats handler."""

    def test_stats_success(self):
        """Test successful stats collection."""
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {
            "sources_count": 100,
            "articles_count": 50,
            "capacity_used": 0.75,
        }

        with patch("valence.mcp.handlers.admin.DatabaseStats.collect", return_value=mock_stats):
            result = admin_stats()

        assert result["success"] is True
        assert "stats" in result
        assert result["stats"]["sources_count"] == 100
        assert result["stats"]["articles_count"] == 50

    def test_stats_exception_handling(self):
        """Test admin_stats exception handling."""
        with patch("valence.mcp.handlers.admin.DatabaseStats.collect", side_effect=Exception("Stats error")):
            result = admin_stats()

        assert result["success"] is False
        assert "Stats error" in result["error"]


class TestAdminMaintenance:
    """Tests for admin_maintenance handler."""

    def test_maintenance_recompute_scores_success(self):
        """Test maintenance with recompute_scores."""
        mock_response = ValenceResponse(success=True, data={})

        with patch("valence.mcp.handlers.admin.run_async", return_value=mock_response):
            result = admin_maintenance(recompute_scores=True)

        assert result["success"] is True
        assert result["maintenance_results"]["recompute_scores"] is True

    def test_maintenance_recompute_scores_failure(self):
        """Test maintenance with recompute_scores failure."""
        mock_response = ValenceResponse(success=False, error="Recompute failed")

        with patch("valence.mcp.handlers.admin.run_async", return_value=mock_response):
            result = admin_maintenance(recompute_scores=True)

        assert result["success"] is True  # Overall success
        assert result["maintenance_results"]["recompute_scores"] is False
        assert "recompute_scores_error" in result["maintenance_results"]

    def test_maintenance_process_queue_success(self):
        """Test maintenance with process_queue."""
        mock_response = ValenceResponse(success=True, data={})

        with patch("valence.mcp.handlers.admin.run_async", return_value=mock_response):
            result = admin_maintenance(process_queue=True)

        assert result["success"] is True
        assert result["maintenance_results"]["process_queue"] is True

    def test_maintenance_process_queue_failure(self):
        """Test maintenance with process_queue failure."""
        mock_response = ValenceResponse(success=False, error="Queue processing failed")

        with patch("valence.mcp.handlers.admin.run_async", return_value=mock_response):
            result = admin_maintenance(process_queue=True)

        assert result["success"] is True
        assert result["maintenance_results"]["process_queue"] is False
        assert "process_queue_error" in result["maintenance_results"]

    def test_maintenance_evict_success(self):
        """Test maintenance with evict_if_over_capacity."""
        mock_response = ValenceResponse(success=True, data={"evicted_count": 5})

        with patch("valence.mcp.handlers.admin.run_async", return_value=mock_response):
            result = admin_maintenance(evict_if_over_capacity=True, evict_count=10)

        assert result["success"] is True
        assert result["maintenance_results"]["evict_if_over_capacity"] is True

    def test_maintenance_evict_failure(self):
        """Test maintenance with eviction failure."""
        mock_response = ValenceResponse(success=False, error="Eviction failed")

        with patch("valence.mcp.handlers.admin.run_async", return_value=mock_response):
            result = admin_maintenance(evict_if_over_capacity=True)

        assert result["success"] is True
        assert result["maintenance_results"]["evict_if_over_capacity"] is False
        assert "evict_error" in result["maintenance_results"]

    def test_maintenance_all_operations(self):
        """Test maintenance with all operations enabled."""
        mock_response = ValenceResponse(success=True, data={})

        with patch("valence.mcp.handlers.admin.run_async", return_value=mock_response):
            result = admin_maintenance(
                recompute_scores=True,
                process_queue=True,
                evict_if_over_capacity=True,
                evict_count=15,
            )

        assert result["success"] is True
        assert result["maintenance_results"]["recompute_scores"] is True
        assert result["maintenance_results"]["process_queue"] is True
        assert result["maintenance_results"]["evict_if_over_capacity"] is True

    def test_maintenance_no_operations(self):
        """Test maintenance with no operations."""
        result = admin_maintenance()

        assert result["success"] is True
        assert result["maintenance_results"] == {}

    def test_maintenance_exception_handling(self):
        """Test admin_maintenance exception handling."""
        with patch("valence.mcp.handlers.admin.run_async", side_effect=Exception("Maintenance error")):
            result = admin_maintenance(recompute_scores=True)

        assert result["success"] is False
        assert "Maintenance error" in result["error"]

    def test_maintenance_custom_evict_count(self):
        """Test maintenance with custom evict_count."""
        mock_response = ValenceResponse(success=True, data={"evicted_count": 25})

        with patch("valence.core.forgetting.evict_lowest"):
            with patch("valence.mcp.handlers.admin.run_async", return_value=mock_response):
                result = admin_maintenance(evict_if_over_capacity=True, evict_count=25)

        assert result["success"] is True
