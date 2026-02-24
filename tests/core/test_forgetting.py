"""Tests for valence.core.forgetting module - article removal and organic eviction.

Tests cover:
- remove_source() deprecation
- remove_article() deletion
- evict_lowest() organic forgetting
- Edge cases (missing article, empty DB, capacity management)
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest


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
        # First query: fetch article
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
        # Second call should be DELETE
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

    @patch("valence.core.forgetting.get_cursor")
    async def test_remove_article_with_title(self, mock_get_cursor):
        """Test removal logs article title."""
        from valence.core.forgetting import remove_article

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "id": "test-id",
            "title": "Important Article",
            "usage_score": 10,
        }
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = await remove_article("test-id")

        assert result.success is True
        # Title should be fetched (verified via fetchone call)
        assert mock_cursor.fetchone.called


class TestEvictLowest:
    """Test evict_lowest organic forgetting."""

    @patch("valence.core.forgetting.remove_article")
    @patch("valence.core.forgetting.get_cursor")
    async def test_evict_lowest_no_overflow(self, mock_get_cursor, mock_remove):
        """Test no eviction when under capacity."""
        from valence.core.forgetting import evict_lowest

        mock_cursor = MagicMock()
        # bounded_memory config
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 100}'},  # config query
            {"cnt": 50},  # count query
        ]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = await evict_lowest(count=10)

        assert result.success is True
        assert result.data == []
        mock_remove.assert_not_called()

    @patch("valence.core.forgetting.remove_article")
    @patch("valence.core.forgetting.get_cursor")
    async def test_evict_lowest_evicts_articles(self, mock_get_cursor, mock_remove):
        """Test articles are evicted when over capacity."""
        from valence.core.forgetting import evict_lowest
        from valence.core.response import ok

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 100}'},  # config
            {"cnt": 110},  # count (10 over)
        ]
        mock_cursor.fetchall.return_value = [
            {"id": "low1"},
            {"id": "low2"},
            {"id": "low3"},
        ]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        # Mock successful removals
        mock_remove.return_value = ok(data={"article_id": "test"})

        result = await evict_lowest(count=10)

        assert result.success is True
        # Should evict 3 (min of count=10 and overflow=10, limited by candidates=3)
        assert len(result.data) == 3
        assert mock_remove.call_count == 3

    @patch("valence.core.forgetting.remove_article")
    @patch("valence.core.forgetting.get_cursor")
    async def test_evict_lowest_respects_count(self, mock_get_cursor, mock_remove):
        """Test eviction respects count parameter."""
        from valence.core.forgetting import evict_lowest
        from valence.core.response import ok

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 100}'},
            {"cnt": 105},  # 5 over
        ]
        # Return only 3 candidates (matching min of count and overflow)
        mock_cursor.fetchall.return_value = [{"id": f"id{i}"} for i in range(3)]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        mock_remove.return_value = ok(data={"article_id": "test"})

        result = await evict_lowest(count=3)

        assert result.success is True
        # Should evict min(count=3, overflow=5) = 3
        assert len(result.data) == 3
        assert mock_remove.call_count == 3

    @patch("valence.core.forgetting.remove_article")
    @patch("valence.core.forgetting.get_cursor")
    async def test_evict_lowest_skips_pinned(self, mock_get_cursor, mock_remove):
        """Test query excludes pinned articles."""
        from valence.core.forgetting import evict_lowest
        from valence.core.response import ok

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 10}'},
            {"cnt": 15},
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        mock_remove.return_value = ok(data={"article_id": "test"})

        await evict_lowest(count=5)

        # Check query excludes pinned
        select_call = None
        for call in mock_cursor.execute.call_args_list:
            if "SELECT id" in str(call[0][0]):
                select_call = call[0][0]
                break

        assert select_call is not None
        assert "pinned = FALSE" in select_call

    @patch("valence.core.forgetting.remove_article")
    @patch("valence.core.forgetting.get_cursor")
    async def test_evict_lowest_orders_by_usage_score(self, mock_get_cursor, mock_remove):
        """Test query orders by usage_score ascending."""
        from valence.core.forgetting import evict_lowest
        from valence.core.response import ok

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 10}'},
            {"cnt": 15},
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        mock_remove.return_value = ok(data={"article_id": "test"})

        await evict_lowest(count=5)

        # Check query orders by usage_score ASC
        select_call = None
        for call in mock_cursor.execute.call_args_list:
            if "SELECT id" in str(call[0][0]):
                select_call = call[0][0]
                break

        assert select_call is not None
        assert "ORDER BY usage_score ASC" in select_call

    @patch("valence.core.forgetting.remove_article")
    @patch("valence.core.forgetting.get_cursor")
    async def test_evict_lowest_handles_removal_failure(self, mock_get_cursor, mock_remove):
        """Test eviction continues when some removals fail."""
        from valence.core.forgetting import evict_lowest
        from valence.core.response import err, ok

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 10}'},
            {"cnt": 15},
        ]
        mock_cursor.fetchall.return_value = [
            {"id": "id1"},
            {"id": "id2"},
            {"id": "id3"},
        ]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        # First removal fails, others succeed
        mock_remove.side_effect = [
            err("Failed"),
            ok(data={"article_id": "id2"}),
            ok(data={"article_id": "id3"}),
        ]

        result = await evict_lowest(count=5)

        assert result.success is True
        # Should have 2 successful evictions
        assert len(result.data) == 2
        assert "id2" in result.data
        assert "id3" in result.data

    @patch("valence.core.forgetting.remove_article")
    @patch("valence.core.forgetting.get_cursor")
    async def test_evict_lowest_default_max_articles(self, mock_get_cursor, mock_remove):
        """Test default max_articles when config missing."""
        from valence.core.forgetting import evict_lowest
        from valence.core.response import ok

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            None,  # No config row
            {"cnt": 10001},  # Over default 10000
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        mock_remove.return_value = ok(data={"article_id": "id1"})

        result = await evict_lowest(count=5)

        assert result.success is True
        assert len(result.data) == 1

    @patch("valence.core.forgetting.remove_article")
    @patch("valence.core.forgetting.get_cursor")
    async def test_evict_lowest_invalid_config_json(self, mock_get_cursor, mock_remove):
        """Test handles invalid JSON in config."""
        from valence.core.forgetting import evict_lowest
        from valence.core.response import ok

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": "invalid json{"},  # Bad JSON
            {"cnt": 10001},
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        mock_remove.return_value = ok(data={"article_id": "id1"})

        result = await evict_lowest(count=5)

        # Should fall back to default 10000
        assert result.success is True

    @patch("valence.core.forgetting.remove_article")
    @patch("valence.core.forgetting.get_cursor")
    async def test_evict_lowest_minimum_count(self, mock_get_cursor, mock_remove):
        """Test count is clamped to minimum 1."""
        from valence.core.forgetting import evict_lowest
        from valence.core.response import ok

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 10}'},
            {"cnt": 15},
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        mock_remove.return_value = ok(data={"article_id": "id1"})

        result = await evict_lowest(count=0)

        assert result.success is True
        # Should still try to evict at least 1
        assert mock_remove.called

    @patch("valence.core.forgetting.remove_article")
    @patch("valence.core.forgetting.get_cursor")
    async def test_evict_lowest_active_status_only(self, mock_get_cursor, mock_remove):
        """Test only active articles are counted and evicted."""
        from valence.core.forgetting import evict_lowest
        from valence.core.response import ok

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"value": '{"max_articles": 10}'},
            {"cnt": 15},
        ]
        mock_cursor.fetchall.return_value = [{"id": "id1"}]
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        mock_remove.return_value = ok(data={"article_id": "id1"})

        await evict_lowest(count=5)

        # Check both queries filter by status='active'
        for call in mock_cursor.execute.call_args_list:
            sql = str(call[0][0])
            if "FROM articles" in sql:
                assert "status = 'active'" in sql
