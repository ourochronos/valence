"""Tests for valence.core.forgetting module (WU-10: Forgetting — C10).

Tests cover:
1. remove_source — deletes from sources, creates tombstone, queues recompile for
                   affected articles, no ghost references after removal
2. remove_article — deletes article, creates tombstone, sources untouched
3. evict_lowest  — only fires when over capacity; evicts lowest-score non-pinned;
                   pinned articles are never evicted
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

SOURCE_ID = str(uuid4())
ARTICLE_A_ID = str(uuid4())
ARTICLE_B_ID = str(uuid4())
ARTICLE_C_ID = str(uuid4())


def _make_cursor_mock(
    fetchone: dict | None = None,
    fetchall: list | None = None,
    rowcount: int = 1,
) -> MagicMock:
    """Create a mock cursor that works as a context manager (sync)."""
    cur = MagicMock()
    cur.rowcount = rowcount
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)

    # Allow dynamic override per call
    cur._fetchone_values = fetchone
    cur._fetchall_values = fetchall or []
    cur.fetchone.return_value = fetchone
    cur.fetchall.return_value = fetchall or []

    return cur


@contextmanager
def _patch_cursor(cur: MagicMock):
    """Context manager that patches get_cursor in the forgetting module."""
    with patch("valence.core.forgetting.get_cursor", return_value=cur):
        yield cur


# ---------------------------------------------------------------------------
# remove_source
# ---------------------------------------------------------------------------


class TestRemoveSource:
    """Tests for remove_source()."""

    @pytest.mark.asyncio
    async def test_deletes_source_and_creates_tombstone(self):
        """remove_source should DELETE from sources and insert a tombstone."""
        from valence.core.forgetting import remove_source

        # Sequence of fetchone/fetchall returns (cursor called multiple times)
        cur = _make_cursor_mock()
        call_index = {"n": 0}

        def side_fetchone():
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                # First call: source exists check
                return {"id": SOURCE_ID}
            elif n == 3:
                # Third call: ghost-ref check → 0 ghosts
                return {"cnt": 0}
            return None

        def side_fetchall():
            # article_sources query
            return [{"article_id": ARTICLE_A_ID}, {"article_id": ARTICLE_B_ID}]

        cur.fetchone.side_effect = side_fetchone
        cur.fetchall.side_effect = side_fetchall

        with _patch_cursor(cur):
            result = await remove_source(SOURCE_ID)

        assert result["success"] is True
        assert result["source_id"] == SOURCE_ID
        assert result["tombstone_created"] is True

        # Verify DELETE was called
        all_sql = " ".join(str(c.args[0]) for c in cur.execute.call_args_list if c.args)
        assert "DELETE FROM sources" in all_sql

        # Verify tombstone INSERT was called
        assert "INSERT INTO tombstones" in all_sql

    @pytest.mark.asyncio
    async def test_queues_recompile_for_affected_articles(self):
        """remove_source should queue 'recompile' for each article that referenced it."""
        from valence.core.forgetting import remove_source

        cur = _make_cursor_mock()
        call_index = {"n": 0}

        def side_fetchone():
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                return {"id": SOURCE_ID}
            elif n == 3:
                return {"cnt": 0}
            return None

        def side_fetchall():
            return [
                {"article_id": ARTICLE_A_ID},
                {"article_id": ARTICLE_B_ID},
            ]

        cur.fetchone.side_effect = side_fetchone
        cur.fetchall.side_effect = side_fetchall

        with _patch_cursor(cur):
            result = await remove_source(SOURCE_ID)

        assert result["success"] is True
        assert result["affected_articles"] == 2
        assert result["recompile_queued"] == 2

        # Two mutation_queue inserts should appear
        all_sql = " ".join(str(c.args[0]) for c in cur.execute.call_args_list if c.args)
        assert "INSERT INTO mutation_queue" in all_sql

    @pytest.mark.asyncio
    async def test_no_ghost_references_after_removal(self):
        """After source removal the ghost_references count should be 0."""
        from valence.core.forgetting import remove_source

        cur = _make_cursor_mock()
        call_index = {"n": 0}

        def side_fetchone():
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                return {"id": SOURCE_ID}
            elif n == 3:
                return {"cnt": 0}  # ← 0 ghost references
            return None

        cur.fetchone.side_effect = side_fetchone
        cur.fetchall.return_value = []

        with _patch_cursor(cur):
            result = await remove_source(SOURCE_ID)

        assert result["success"] is True
        assert result["ghost_references"] == 0

    @pytest.mark.asyncio
    async def test_returns_error_for_missing_source(self):
        """remove_source should return an error dict when the source ID doesn't exist."""
        from valence.core.forgetting import remove_source

        cur = _make_cursor_mock(fetchone=None)  # source not found

        with _patch_cursor(cur):
            result = await remove_source(SOURCE_ID)

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_error_for_empty_source_id(self):
        """remove_source should validate input and reject empty source_id."""
        from valence.core.forgetting import remove_source

        cur = _make_cursor_mock()
        with _patch_cursor(cur):
            result = await remove_source("")

        assert result["success"] is False
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_works_when_source_has_no_linked_articles(self):
        """remove_source should succeed even if no articles referenced the source."""
        from valence.core.forgetting import remove_source

        cur = _make_cursor_mock()
        call_index = {"n": 0}

        def side_fetchone():
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                return {"id": SOURCE_ID}
            elif n == 2:
                return {"cnt": 0}
            return None

        cur.fetchone.side_effect = side_fetchone
        cur.fetchall.return_value = []  # no affected articles

        with _patch_cursor(cur):
            result = await remove_source(SOURCE_ID)

        assert result["success"] is True
        assert result["affected_articles"] == 0
        assert result["recompile_queued"] == 0

    @pytest.mark.asyncio
    async def test_tombstone_uses_admin_action_reason(self):
        """Tombstone should use 'admin_action' as the reason."""
        from valence.core.forgetting import remove_source

        cur = _make_cursor_mock()
        call_index = {"n": 0}

        def side_fetchone():
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                return {"id": SOURCE_ID}
            elif n == 3:
                return {"cnt": 0}
            return None

        cur.fetchone.side_effect = side_fetchone
        cur.fetchall.return_value = []

        with _patch_cursor(cur):
            await remove_source(SOURCE_ID)

        tombstone_calls = [
            c for c in cur.execute.call_args_list
            if c.args and "INSERT INTO tombstones" in str(c.args[0])
        ]
        assert len(tombstone_calls) == 1
        # 'admin_action' is hardcoded in the SQL string itself
        tombstone_sql = tombstone_calls[0].args[0]
        assert "admin_action" in tombstone_sql


# ---------------------------------------------------------------------------
# remove_article
# ---------------------------------------------------------------------------


class TestRemoveArticle:
    """Tests for remove_article()."""

    @pytest.mark.asyncio
    async def test_deletes_article_and_creates_tombstone(self):
        """remove_article should DELETE the article and insert a tombstone."""
        from valence.core.forgetting import remove_article

        cur = _make_cursor_mock(
            fetchone={"id": ARTICLE_A_ID, "title": "Test Article", "usage_score": 1.5},
        )

        with _patch_cursor(cur):
            result = await remove_article(ARTICLE_A_ID)

        assert result["success"] is True
        assert result["article_id"] == ARTICLE_A_ID
        assert result["tombstone_created"] is True

        all_sql = " ".join(str(c.args[0]) for c in cur.execute.call_args_list if c.args)
        assert "DELETE FROM articles" in all_sql
        assert "INSERT INTO tombstones" in all_sql

    @pytest.mark.asyncio
    async def test_sources_untouched_after_article_removal(self):
        """remove_article must NOT issue any DELETE against sources."""
        from valence.core.forgetting import remove_article

        cur = _make_cursor_mock(
            fetchone={"id": ARTICLE_A_ID, "title": "Test", "usage_score": 0.0},
        )

        with _patch_cursor(cur):
            await remove_article(ARTICLE_A_ID)

        all_sql = " ".join(str(c.args[0]) for c in cur.execute.call_args_list if c.args)
        assert "DELETE FROM sources" not in all_sql

    @pytest.mark.asyncio
    async def test_returns_error_for_missing_article(self):
        """remove_article should return an error when article ID doesn't exist."""
        from valence.core.forgetting import remove_article

        cur = _make_cursor_mock(fetchone=None)

        with _patch_cursor(cur):
            result = await remove_article(ARTICLE_A_ID)

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_error_for_empty_article_id(self):
        """remove_article should reject an empty article_id."""
        from valence.core.forgetting import remove_article

        cur = _make_cursor_mock()
        with _patch_cursor(cur):
            result = await remove_article("")

        assert result["success"] is False
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_tombstone_records_article_title(self):
        """Tombstone metadata should include the article's title."""
        from valence.core.forgetting import remove_article

        article_title = "My Important Article"
        cur = _make_cursor_mock(
            fetchone={"id": ARTICLE_A_ID, "title": article_title, "usage_score": 2.3},
        )

        with _patch_cursor(cur):
            await remove_article(ARTICLE_A_ID)

        tombstone_calls = [
            c for c in cur.execute.call_args_list
            if c.args and "INSERT INTO tombstones" in str(c.args[0])
        ]
        assert len(tombstone_calls) == 1
        # params: (content_type, content_id, metadata_json) — reason is hardcoded in SQL
        metadata_str = tombstone_calls[0].args[1][2]  # 3rd param = metadata JSON
        metadata = json.loads(metadata_str)
        assert metadata["title"] == article_title

    @pytest.mark.asyncio
    async def test_tombstone_content_type_is_article(self):
        """Tombstone content_type must be 'article' (not 'source')."""
        from valence.core.forgetting import remove_article

        cur = _make_cursor_mock(
            fetchone={"id": ARTICLE_A_ID, "title": None, "usage_score": 0.0},
        )

        with _patch_cursor(cur):
            await remove_article(ARTICLE_A_ID)

        tombstone_calls = [
            c for c in cur.execute.call_args_list
            if c.args and "INSERT INTO tombstones" in str(c.args[0])
        ]
        assert len(tombstone_calls) == 1
        params = tombstone_calls[0].args[1]
        assert params[0] == "article"


# ---------------------------------------------------------------------------
# evict_lowest
# ---------------------------------------------------------------------------


class TestEvictLowest:
    """Tests for evict_lowest()."""

    @pytest.mark.asyncio
    async def test_does_not_evict_when_below_capacity(self):
        """evict_lowest should return [] when article count <= max_articles."""
        from valence.core.forgetting import evict_lowest

        cur = _make_cursor_mock()
        call_index = {"n": 0}

        def side_fetchone():
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                # system_config bounded_memory
                return {"value": json.dumps({"max_articles": 1000})}
            elif n == 2:
                # current article count — below capacity
                return {"cnt": 800}
            return None

        cur.fetchone.side_effect = side_fetchone

        with _patch_cursor(cur):
            result = await evict_lowest(count=10)

        assert result == []

    @pytest.mark.asyncio
    async def test_evicts_when_over_capacity(self):
        """evict_lowest fires when article count > max_articles."""
        from valence.core.forgetting import evict_lowest

        # We'll mock remove_article to track calls
        with patch("valence.core.forgetting.remove_article", new_callable=AsyncMock) as mock_remove:
            mock_remove.return_value = {"success": True}

            cur = _make_cursor_mock()
            call_index = {"n": 0}

            def side_fetchone():
                call_index["n"] += 1
                n = call_index["n"]
                if n == 1:
                    return {"value": json.dumps({"max_articles": 100})}
                elif n == 2:
                    return {"cnt": 105}  # 5 over capacity
                return None

            def side_fetchall():
                return [
                    {"id": ARTICLE_A_ID},
                    {"id": ARTICLE_B_ID},
                    {"id": ARTICLE_C_ID},
                ]

            cur.fetchone.side_effect = side_fetchone
            cur.fetchall.side_effect = side_fetchall

            with _patch_cursor(cur):
                result = await evict_lowest(count=10)

        # Should evict 3 articles (returned by fetchall)
        assert len(result) == 3
        assert mock_remove.call_count == 3

    @pytest.mark.asyncio
    async def test_evicts_only_overflow_amount(self):
        """evict_lowest caps eviction to the overflow, not the requested count."""
        from valence.core.forgetting import evict_lowest

        with patch("valence.core.forgetting.remove_article", new_callable=AsyncMock) as mock_remove:
            mock_remove.return_value = {"success": True}

            cur = _make_cursor_mock()
            call_index = {"n": 0}

            def side_fetchone():
                call_index["n"] += 1
                n = call_index["n"]
                if n == 1:
                    return {"value": json.dumps({"max_articles": 100})}
                elif n == 2:
                    return {"cnt": 102}  # only 2 over capacity
                return None

            def side_fetchall():
                # fetchall returns 2 candidates (capped at overflow=2)
                return [
                    {"id": ARTICLE_A_ID},
                    {"id": ARTICLE_B_ID},
                ]

            cur.fetchone.side_effect = side_fetchone
            cur.fetchall.side_effect = side_fetchall

            with _patch_cursor(cur):
                result = await evict_lowest(count=10)

        # Only 2 should be evicted (overflow=2, not count=10)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_pinned_articles_excluded_from_candidates(self):
        """The SQL passed to get candidates must exclude pinned articles."""
        from valence.core.forgetting import evict_lowest

        cur = _make_cursor_mock()
        call_index = {"n": 0}

        def side_fetchone():
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                return {"value": json.dumps({"max_articles": 100})}
            elif n == 2:
                return {"cnt": 110}
            return None

        # No candidates returned (all might be pinned)
        cur.fetchone.side_effect = side_fetchone
        cur.fetchall.return_value = []

        with _patch_cursor(cur):
            result = await evict_lowest(count=10)

        # Check that the SQL for candidates includes pinned = FALSE
        all_sql = " ".join(str(c.args[0]) for c in cur.execute.call_args_list if c.args)
        assert "pinned" in all_sql.lower()
        assert result == []

    @pytest.mark.asyncio
    async def test_uses_system_config_for_max_articles(self):
        """evict_lowest must read max_articles from system_config, not a hardcoded value."""
        from valence.core.forgetting import evict_lowest

        cur = _make_cursor_mock()
        call_index = {"n": 0}

        def side_fetchone():
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                # Custom max_articles
                return {"value": json.dumps({"max_articles": 50})}
            elif n == 2:
                return {"cnt": 48}  # below 50 → no eviction
            return None

        cur.fetchone.side_effect = side_fetchone
        cur.fetchall.return_value = []

        with _patch_cursor(cur):
            result = await evict_lowest(count=10)

        # Below capacity → no eviction
        assert result == []

        # Verify system_config was queried
        all_sql = " ".join(str(c.args[0]) for c in cur.execute.call_args_list if c.args)
        assert "system_config" in all_sql

    @pytest.mark.asyncio
    async def test_defaults_to_10000_when_no_config(self):
        """evict_lowest defaults to max_articles=10000 if system_config is missing."""
        from valence.core.forgetting import evict_lowest

        cur = _make_cursor_mock()
        call_index = {"n": 0}

        def side_fetchone():
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                return None  # no config row
            elif n == 2:
                return {"cnt": 500}  # well below 10000
            return None

        cur.fetchone.side_effect = side_fetchone
        cur.fetchall.return_value = []

        with _patch_cursor(cur):
            result = await evict_lowest(count=10)

        # 500 < 10000, so no eviction
        assert result == []

    @pytest.mark.asyncio
    async def test_evicted_articles_returned_as_list_of_ids(self):
        """Return value must be a list of evicted article ID strings."""
        from valence.core.forgetting import evict_lowest

        with patch("valence.core.forgetting.remove_article", new_callable=AsyncMock) as mock_remove:
            mock_remove.return_value = {"success": True}

            cur = _make_cursor_mock()
            call_index = {"n": 0}

            def side_fetchone():
                call_index["n"] += 1
                n = call_index["n"]
                if n == 1:
                    return {"value": json.dumps({"max_articles": 10})}
                elif n == 2:
                    return {"cnt": 12}
                return None

            def side_fetchall():
                return [{"id": ARTICLE_A_ID}, {"id": ARTICLE_B_ID}]

            cur.fetchone.side_effect = side_fetchone
            cur.fetchall.side_effect = side_fetchall

            with _patch_cursor(cur):
                result = await evict_lowest(count=5)

        assert isinstance(result, list)
        assert ARTICLE_A_ID in result
        assert ARTICLE_B_ID in result


# ---------------------------------------------------------------------------
# Integration: remove_source → no ghost references
# ---------------------------------------------------------------------------


class TestNoGhostReferences:
    """Verify complete removal — no ghost references after source deletion."""

    @pytest.mark.asyncio
    async def test_article_sources_checked_for_ghost_refs(self):
        """After source deletion, code must query article_sources to verify no ghosts."""
        from valence.core.forgetting import remove_source

        cur = _make_cursor_mock()
        call_index = {"n": 0}

        def side_fetchone():
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                return {"id": SOURCE_ID}
            elif n == 3:
                return {"cnt": 0}
            return None

        cur.fetchone.side_effect = side_fetchone
        cur.fetchall.return_value = []

        with _patch_cursor(cur):
            result = await remove_source(SOURCE_ID)

        # The ghost-check query should appear in executed SQL
        all_sql = " ".join(str(c.args[0]) for c in cur.execute.call_args_list if c.args)
        assert "article_sources" in all_sql
        assert result["ghost_references"] == 0

    @pytest.mark.asyncio
    async def test_remove_source_not_remove_article(self):
        """remove_source deletes from sources, not from articles directly."""
        from valence.core.forgetting import remove_source

        cur = _make_cursor_mock()
        call_index = {"n": 0}

        def side_fetchone():
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                return {"id": SOURCE_ID}
            elif n == 3:
                return {"cnt": 0}
            return None

        cur.fetchone.side_effect = side_fetchone
        cur.fetchall.return_value = []

        with _patch_cursor(cur):
            await remove_source(SOURCE_ID)

        all_sql = " ".join(str(c.args[0]) for c in cur.execute.call_args_list if c.args)
        assert "DELETE FROM sources" in all_sql
        # Articles should NOT be deleted — only the source
        assert "DELETE FROM articles" not in all_sql
