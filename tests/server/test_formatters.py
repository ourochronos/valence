"""Tests for formatters.py (text/table output formatting)."""

from __future__ import annotations

from valence.server.formatters import (
    format_beliefs_list_text,
    format_conflicts_text,
    format_embeddings_status_text,
    format_entities_list_text,
    format_maintenance_text,
    format_migration_status_text,
    format_patterns_list_text,
    format_sessions_list_text,
    format_stats_text,
    format_tensions_list_text,
)


class TestFormatStatsText:
    """Tests for format_stats_text."""

    def test_basic_stats_formatting(self):
        """Test basic stats formatting."""
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
        """Test formatting empty stats."""
        data = {"stats": {}}
        result = format_stats_text(data)
        assert "Valence Statistics" in result

    def test_handles_missing_stats_key(self):
        """Test handles missing stats key gracefully."""
        data = {}
        result = format_stats_text(data)
        assert isinstance(result, str)


class TestFormatBeliefsListText:
    """Tests for format_beliefs_list_text."""

    def test_empty_beliefs(self):
        """Test formatting empty beliefs list."""
        data = {"beliefs": []}
        result = format_beliefs_list_text(data)
        assert "No beliefs found" in result

    def test_beliefs_with_content(self):
        """Test formatting beliefs with content."""
        data = {
            "beliefs": [
                {
                    "id": "art1",
                    "content": "Test belief content",
                    "confidence": 0.85,
                    "domain_path": ["tech", "ai"],
                }
            ]
        }

        result = format_beliefs_list_text(data)

        assert "Found 1 belief" in result
        assert "Test belief content" in result
        assert "ID:" in result
        assert "Confidence: 85%" in result
        assert "Domains: tech, ai" in result

    def test_multiple_beliefs(self):
        """Test formatting multiple beliefs."""
        data = {
            "beliefs": [
                {"id": "art1", "content": "First", "confidence": 0.9},
                {"id": "art2", "content": "Second", "confidence": 0.7},
            ]
        }

        result = format_beliefs_list_text(data)
        assert "Found 2 belief(s)" in result
        assert "First" in result
        assert "Second" in result

    def test_long_content_truncated(self):
        """Test long content is truncated."""
        long_content = "x" * 100
        data = {"beliefs": [{"id": "art1", "content": long_content}]}

        result = format_beliefs_list_text(data)
        # Should be truncated to 80 chars
        assert len(long_content[:80]) <= 80
        assert "x" * 80 in result

    def test_confidence_as_dict(self):
        """Test handling confidence as dict."""
        data = {"beliefs": [{"id": "art1", "content": "Test", "confidence": {"overall": 0.75}}]}

        result = format_beliefs_list_text(data)
        assert "75%" in result

    def test_missing_fields(self):
        """Test handles missing fields gracefully."""
        data = {"beliefs": [{"id": "art1"}]}
        result = format_beliefs_list_text(data)
        assert isinstance(result, str)


class TestFormatEntitiesListText:
    """Tests for format_entities_list_text."""

    def test_empty_entities(self):
        """Test formatting empty entities list."""
        data = {"entities": []}
        result = format_entities_list_text(data)
        assert "No entities found" in result

    def test_entities_with_data(self):
        """Test formatting entities."""
        data = {
            "entities": [
                {"id": "ent1", "name": "John Doe", "type": "person"},
                {"id": "ent2", "name": "Python", "type": "tool"},
            ]
        }

        result = format_entities_list_text(data)

        assert "Found 2 entity/entities" in result
        assert "[person] John Doe" in result
        assert "[tool] Python" in result

    def test_missing_type(self):
        """Test entity with missing type."""
        data = {"entities": [{"id": "ent1", "name": "Test"}]}
        result = format_entities_list_text(data)
        assert "[unknown] Test" in result


class TestFormatTensionsListText:
    """Tests for format_tensions_list_text."""

    def test_empty_tensions(self):
        """Test formatting empty tensions."""
        data = {"tensions": []}
        result = format_tensions_list_text(data)
        assert "No tensions found" in result

    def test_tensions_with_data(self):
        """Test formatting tensions."""
        data = {
            "tensions": [
                {
                    "id": "con1",
                    "severity": "high",
                    "status": "detected",
                    "description": "Conflicting beliefs about AI safety",
                }
            ]
        }

        result = format_tensions_list_text(data)

        assert "Found 1 tension(s)" in result
        assert "(high/detected)" in result
        assert "Conflicting beliefs" in result

    def test_description_truncation(self):
        """Test long description is truncated."""
        long_desc = "x" * 100
        data = {"tensions": [{"id": "con1", "description": long_desc}]}

        result = format_tensions_list_text(data)
        # Truncated to 60 chars
        assert "x" * 60 in result


class TestFormatConflictsText:
    """Tests for format_conflicts_text."""

    def test_no_conflicts(self):
        """Test no conflicts detected."""
        data = {"conflicts": []}
        result = format_conflicts_text(data)
        assert "No potential conflicts detected" in result

    def test_conflicts_formatting(self):
        """Test conflicts formatting."""
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
        assert "92" in result  # similarity percentage
        assert "0.65" in result  # signal score
        assert "Reason: negation asymmetry" in result
        assert "Python is the best language" in result
        assert "Python is not the best language" in result

    def test_limits_to_10_conflicts(self):
        """Test only shows first 10 conflicts."""
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
        # Should show max 10
        assert "Conflict #10" in result
        assert "Conflict #11" not in result


class TestFormatMaintenanceText:
    """Tests for format_maintenance_text."""

    def test_maintenance_results(self):
        """Test formatting maintenance results."""
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
        assert "refresh_views" in result
        assert "tables_processed=5" in result
        assert "2 operation(s) completed" in result

    def test_dry_run_label(self):
        """Test dry run is labeled."""
        data = {"results": [], "dry_run": True}
        result = format_maintenance_text(data)
        assert "(dry run)" in result

    def test_empty_results(self):
        """Test empty results."""
        data = {"results": [], "dry_run": False}
        result = format_maintenance_text(data)
        assert "0 operation(s) completed" in result


class TestFormatSessionsListText:
    """Tests for format_sessions_list_text."""

    def test_empty_sessions(self):
        """Test empty sessions list."""
        data = {"sessions": []}
        result = format_sessions_list_text(data)
        assert "No sessions found" in result

    def test_sessions_formatting(self):
        """Test sessions formatting."""
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
        assert "(ended)" in result


class TestFormatPatternsListText:
    """Tests for format_patterns_list_text."""

    def test_empty_patterns(self):
        """Test empty patterns list."""
        data = {"patterns": []}
        result = format_patterns_list_text(data)
        assert "No patterns found" in result

    def test_patterns_formatting(self):
        """Test patterns formatting."""
        data = {
            "patterns": [
                {
                    "id": "pat1",
                    "type": "preference",
                    "description": "Prefers Python over JavaScript",
                    "confidence": 0.85,
                }
            ]
        }

        result = format_patterns_list_text(data)

        assert "Found 1 pattern(s)" in result
        assert "(preference, 85%)" in result
        assert "Prefers Python" in result

    def test_confidence_as_string(self):
        """Test handles confidence as string."""
        data = {"patterns": [{"id": "pat1", "confidence": "high"}]}
        result = format_patterns_list_text(data)
        assert "high" in result


class TestFormatMigrationStatusText:
    """Tests for format_migration_status_text."""

    def test_empty_migrations(self):
        """Test empty migrations list."""
        data = {"migrations": []}
        result = format_migration_status_text(data)
        assert "No migrations found" in result

    def test_migrations_formatting(self):
        """Test migrations formatting."""
        data = {
            "migrations": [
                {"name": "001_initial", "status": "applied"},
                {"name": "002_add_embeddings", "status": "pending"},
            ]
        }

        result = format_migration_status_text(data)

        assert "Migration Status" in result
        assert "[applied] 001_initial" in result
        assert "[pending] 002_add_embeddings" in result


class TestFormatEmbeddingsStatusText:
    """Tests for format_embeddings_status_text."""

    def test_embeddings_status(self):
        """Test embeddings status formatting."""
        data = {
            "stats": {
                "total_articles": 100,
                "with_embeddings": 80,
                "missing_embeddings": 20,
                "coverage": "80.0%",
            }
        }

        result = format_embeddings_status_text(data)

        assert "Embedding Status" in result
        assert "Total Articles: 100" in result
        assert "With Embeddings: 80" in result
        assert "Missing Embeddings: 20" in result
        assert "Coverage: 80.0%" in result

    def test_empty_stats(self):
        """Test empty embeddings stats."""
        data = {"stats": {}}
        result = format_embeddings_status_text(data)
        assert "Embedding Status" in result


class TestFormatterEdgeCases:
    """Tests for edge cases across formatters."""

    def test_none_values(self):
        """Test formatters handle None values."""
        data = {"beliefs": [{"id": None, "content": None}]}
        result = format_beliefs_list_text(data)
        assert isinstance(result, str)

    def test_missing_keys(self):
        """Test formatters handle missing keys."""
        data = {}
        assert isinstance(format_stats_text(data), str)
        assert isinstance(format_beliefs_list_text(data), str)
        assert isinstance(format_entities_list_text(data), str)

    def test_unicode_content(self):
        """Test formatters handle unicode content."""
        data = {"beliefs": [{"id": "art1", "content": "Test 你好 🎉"}]}
        result = format_beliefs_list_text(data)
        assert "你好" in result
        assert "🎉" in result
