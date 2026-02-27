"""Tests for valence.core.forgetting module - article removal and organic archival.

Tests cover:
- remove_source() deprecation
- remove_article() deletion
- archive_lowest() / evict_lowest() organic forgetting via archival
- Edge cases (missing article, empty DB, capacity management)
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch


class TestRemoveSource:
    """Test remove_source deprecation."""

    async def test_remove_source_deprecated(self):
        """Test remove_source returns deprecation error."""
        from valence.core.forgetting import remove_source

        result = await remove_source("test-id")

        assert result.success is False
        assert "deprecated" in result.error.lower()
        assert "append-only" in result.error.lower()


class TestRemoveArticle:
    """Test remove_article function."""

    @patch("valence.core.forgetting.get_cursor")
    async def test_remove_article_success(self, mock_get_cursor):
        """Test successful article removal."""
        from valence.core.forgetting import remove_article

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "id": "abc-123",
            "title": "Test Article",
            "usage_score": 5,
        }
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = await remove_article("abc-123")

        assert result.success is True
        assert result.data["article_id"] == "abc-123"
        assert mock_cursor.execute.call_count == 2
        delete_call = mock_cursor.execute.call_args_list[1]
        assert "DELETE FROM articles" in delete_call[0][0]

    @patch("valence.core.forgetting.get_cursor")
    async def test_remove_article_not_found(self, mock_get_cursor):
        """Test error when article doesn't exist."""
        from valence.core.forgetting import remove_article

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = await remove_article("nonexistent")

        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_remove_article_empty_id(self):
        """Test error with empty article ID."""
        from valence.core.forgetting import remove_article

        result = await remove_article("")

        assert result.success is False
        assert "required" in result.error.lower()

    async def test_remove_article_whitespace_id(self):
        """Test error with whitespace-only ID."""
        from valence.core.forgetting import remove_article

        result = await remove_article("   ")

        assert result.success is False
        assert "required" in result.error.lower()


class TestArchiveLowest:
    """Test archive_lowest (and evict_lowest alias) organic forgetting."""

    @patch("valence.core.forgetting.get_cursor")
    async def test_no_overflow(self, mock_get_cursor):
        """Test no archival when under capacity."""
        from valence.core.forgetting import archive_lowest

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 100}'},
            {"cnt": 50},
        ]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = await archive_lowest(count=10)

        assert result.success is True
        assert result.data == []

    @patch("valence.core.forgetting.get_cursor")
    async def test_archives_articles(self, mock_get_cursor):
        """Test articles are archived (not deleted) when over capacity."""
        from valence.core.forgetting import archive_lowest

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 100}'},
            {"cnt": 110},
        ]
        mock_cursor.fetchall.return_value = [
            {"id": "low1"},
            {"id": "low2"},
            {"id": "low3"},
        ]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = await archive_lowest(count=10)

        assert result.success is True
        assert len(result.data) == 3
        # Verify UPDATE (not DELETE) was used
        update_call = mock_cursor.execute.call_args_list[-1]
        sql = update_call[0][0]
        assert "UPDATE articles" in sql
        assert "SET status = 'archived'" in sql
        assert "DELETE" not in sql

    @patch("valence.core.forgetting.get_cursor")
    async def test_respects_count(self, mock_get_cursor):
        """Test archival respects count parameter."""
        from valence.core.forgetting import archive_lowest

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 100}'},
            {"cnt": 105},
        ]
        mock_cursor.fetchall.return_value = [{"id": f"id{i}"} for i in range(3)]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = await archive_lowest(count=3)

        assert result.success is True
        assert len(result.data) == 3
        # Verify LIMIT parameter
        update_call = mock_cursor.execute.call_args_list[-1]
        assert update_call[0][1] == (3,)

    @patch("valence.core.forgetting.get_cursor")
    async def test_skips_pinned(self, mock_get_cursor):
        """Test query excludes pinned articles."""
        from valence.core.forgetting import archive_lowest

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 10}'},
            {"cnt": 15},
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        await archive_lowest(count=5)

        update_call = mock_cursor.execute.call_args_list[-1]
        sql = update_call[0][0]
        assert "pinned = FALSE" in sql

    @patch("valence.core.forgetting.get_cursor")
    async def test_orders_by_usage_score(self, mock_get_cursor):
        """Test query orders by usage_score ascending (lowest first)."""
        from valence.core.forgetting import archive_lowest

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 10}'},
            {"cnt": 15},
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        await archive_lowest(count=5)

        update_call = mock_cursor.execute.call_args_list[-1]
        sql = update_call[0][0]
        assert "ORDER BY usage_score ASC" in sql

    @patch("valence.core.forgetting.get_cursor")
    async def test_default_max_articles(self, mock_get_cursor):
        """Test default max_articles when config missing."""
        from valence.core.forgetting import archive_lowest

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            None,  # No config row
            {"cnt": 10001},
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = await archive_lowest(count=5)

        assert result.success is True
        assert len(result.data) == 1

    @patch("valence.core.forgetting.get_cursor")
    async def test_invalid_config_json(self, mock_get_cursor):
        """Test handles invalid JSON in config."""
        from valence.core.forgetting import archive_lowest

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": "invalid json{"},
            {"cnt": 10001},
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = await archive_lowest(count=5)

        assert result.success is True

    @patch("valence.core.forgetting.get_cursor")
    async def test_minimum_count(self, mock_get_cursor):
        """Test count is clamped to minimum 1."""
        from valence.core.forgetting import archive_lowest

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 10}'},
            {"cnt": 15},
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = await archive_lowest(count=0)

        assert result.success is True
        # Should archive at least 1 (count clamped from 0 to 1)
        assert len(result.data) == 1

    @patch("valence.core.forgetting.get_cursor")
    async def test_only_active_articles(self, mock_get_cursor):
        """Test only active articles are counted and archived."""
        from valence.core.forgetting import archive_lowest

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 10}'},
            {"cnt": 15},
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        await archive_lowest(count=5)

        for call in mock_cursor.execute.call_args_list:
            sql = str(call[0][0])
            if "FROM articles" in sql and "SELECT COUNT" in sql:
                assert "status = 'active'" in sql

    async def test_evict_lowest_is_alias(self):
        """Test evict_lowest delegates to archive_lowest."""
        from valence.core.forgetting import evict_lowest

        with patch("valence.core.forgetting.archive_lowest") as mock_archive:
            from valence.core.response import ok

            mock_archive.return_value = ok(data=[])
            await evict_lowest(count=5)
            mock_archive.assert_called_once_with(5)
