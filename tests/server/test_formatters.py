"""Tests for formatters.py (text/table output formatting)."""

from __future__ import annotations

from valence.server.formatters import (
    format_conflicts_text,
    format_embeddings_status_text,
    format_maintenance_text,
    format_migration_status_text,
    format_sessions_list_text,
    format_stats_text,
)


class TestFormatStatsText:
    """Tests for format_stats_text."""

    def test_basic_stats_formatting(self):
        data = {
            "stats": {
                "total_articles": 100,
                "active_articles": 80,
                "with_embeddings": 60,
            }
        }
        result = format_stats_text(data)
        assert "Valence Statistics" in result
        assert "Total Articles: 100" in result
        assert "Active Articles: 80" in result
        assert "With Embeddings: 60" in result

    def test_empty_stats(self):
        data = {"stats": {}}
        result = format_stats_text(data)
        assert "Valence Statistics" in result

    def test_handles_missing_stats_key(self):
        data = {}
        result = format_stats_text(data)
        assert isinstance(result, str)


class TestFormatConflictsText:
    """Tests for format_conflicts_text."""

    def test_no_conflicts(self):
        data = {"conflicts": []}
        result = format_conflicts_text(data)
        assert "No potential conflicts detected" in result

    def test_conflicts_formatting(self):
        data = {
            "conflicts": [
                {
                    "id_a": "art1",
                    "id_b": "art2",
                    "content_a": "Python is the best language",
                    "content_b": "Python is not the best language",
                    "similarity": 0.92,
                    "conflict_score": 0.65,
                    "reason": "negation asymmetry",
                }
            ]
        }
        result = format_conflicts_text(data)
        assert "Found 1 potential conflict(s)" in result
        assert "Conflict #1" in result
        assert "92" in result
        assert "0.65" in result
        assert "Reason: negation asymmetry" in result

    def test_limits_to_10_conflicts(self):
        conflicts_data = [
            {
                "id_a": f"art{i}",
                "id_b": f"art{i + 1}",
                "content_a": f"content {i}",
                "content_b": f"content {i + 1}",
                "similarity": 0.9,
                "conflict_score": 0.5,
                "reason": "test",
            }
            for i in range(15)
        ]
        data = {"conflicts": conflicts_data}
        result = format_conflicts_text(data)
        assert "Conflict #10" in result
        assert "Conflict #11" not in result


class TestFormatMaintenanceText:
    """Tests for format_maintenance_text."""

    def test_maintenance_results(self):
        data = {
            "results": [
                {"operation": "vacuum_analyze", "tables_processed": 5},
                {"operation": "refresh_views", "views_refreshed": 3},
            ],
            "dry_run": False,
        }
        result = format_maintenance_text(data)
        assert "Maintenance Report" in result
        assert "vacuum_analyze" in result
        assert "2 operation(s) completed" in result

    def test_dry_run_label(self):
        data = {"results": [], "dry_run": True}
        result = format_maintenance_text(data)
        assert "(dry run)" in result


class TestFormatSessionsListText:
    """Tests for format_sessions_list_text."""

    def test_empty_sessions(self):
        data = {"sessions": []}
        result = format_sessions_list_text(data)
        assert "No sessions found" in result

    def test_sessions_formatting(self):
        data = {
            "sessions": [
                {"id": "sess1", "status": "active", "platform": "telegram"},
                {"id": "sess2", "status": "ended", "platform": "discord"},
            ]
        }
        result = format_sessions_list_text(data)
        assert "Found 2 session(s)" in result
        assert "(active)" in result
        assert "platform=telegram" in result


class TestFormatMigrationStatusText:
    """Tests for format_migration_status_text."""

    def test_empty_migrations(self):
        data = {"migrations": []}
        result = format_migration_status_text(data)
        assert "No migrations found" in result

    def test_migrations_formatting(self):
        data = {
            "migrations": [
                {"name": "001_initial", "status": "applied"},
                {"name": "002_add_embeddings", "status": "pending"},
            ]
        }
        result = format_migration_status_text(data)
        assert "[applied] 001_initial" in result
        assert "[pending] 002_add_embeddings" in result


class TestFormatEmbeddingsStatusText:
    """Tests for format_embeddings_status_text."""

    def test_embeddings_status(self):
        data = {
            "stats": {
                "total_articles": 100,
                "with_embeddings": 80,
                "coverage": "80.0%",
            }
        }
        result = format_embeddings_status_text(data)
        assert "Embedding Status" in result
        assert "Total Articles: 100" in result

    def test_empty_stats(self):
        data = {"stats": {}}
        result = format_embeddings_status_text(data)
        assert "Embedding Status" in result


class TestFormatterEdgeCases:
    """Tests for edge cases across formatters."""

    def test_missing_keys(self):
        assert isinstance(format_stats_text({}), str)
        assert isinstance(format_conflicts_text({}), str)
        assert isinstance(format_maintenance_text({"results": []}), str)
