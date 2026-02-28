# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Tests for auto-compilation threshold tuning and recompile_ungrouped_sources (#63)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_compile_similarity_threshold():
    from valence.core.ingest_pipeline import COMPILE_SIMILARITY_THRESHOLD

    assert COMPILE_SIMILARITY_THRESHOLD == 0.65


def test_auto_compile_min_cluster():
    from valence.core.ingest_pipeline import AUTO_COMPILE_MIN_CLUSTER

    assert AUTO_COMPILE_MIN_CLUSTER == 2


@pytest.mark.asyncio
async def test_recompile_ungrouped_finds_and_compiles():
    source_a = "aaaaaaaa-0000-0000-0000-000000000001"
    source_b = "bbbbbbbb-0000-0000-0000-000000000002"

    fake_compile_result = MagicMock(success=True, data={"id": "article-uuid-001"})

    with (
        patch("valence.core.maintenance.get_cursor") as mock_get_cursor,
        patch("valence.core.maintenance.find_similar_ungrouped", new_callable=AsyncMock) as mock_similar,
        patch("valence.core.maintenance.compile_article", new_callable=AsyncMock) as mock_compile,
    ):
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [{"id": source_a}, {"id": source_b}]
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        async def similar_side(sid, threshold):
            return [source_b] if sid == source_a else []

        mock_similar.side_effect = similar_side
        mock_compile.return_value = fake_compile_result

        from valence.core.maintenance import recompile_ungrouped_sources

        result = await recompile_ungrouped_sources(limit=50)

    assert result["checked"] == 2
    assert result["articles_created"] == 1
    assert result["compiled"] == 2
    mock_compile.assert_called_once()


@pytest.mark.asyncio
async def test_recompile_ungrouped_skips_already_processed():
    source_a = "aaaaaaaa-0000-0000-0000-000000000001"
    source_b = "bbbbbbbb-0000-0000-0000-000000000002"

    fake_compile_result = MagicMock(success=True, data={"id": "article-uuid-002"})

    with (
        patch("valence.core.maintenance.get_cursor") as mock_get_cursor,
        patch("valence.core.maintenance.find_similar_ungrouped", new_callable=AsyncMock) as mock_similar,
        patch("valence.core.maintenance.compile_article", new_callable=AsyncMock) as mock_compile,
    ):
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [{"id": source_a}, {"id": source_b}]
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_similar.return_value = [source_b]
        mock_compile.return_value = fake_compile_result

        from valence.core.maintenance import recompile_ungrouped_sources

        result = await recompile_ungrouped_sources(limit=50)

    assert mock_compile.call_count == 1
    assert result["articles_created"] == 1


@pytest.mark.asyncio
async def test_recompile_ungrouped_empty_results():
    with (
        patch("valence.core.maintenance.get_cursor") as mock_get_cursor,
        patch("valence.core.maintenance.find_similar_ungrouped", new_callable=AsyncMock) as mock_similar,
        patch("valence.core.maintenance.compile_article", new_callable=AsyncMock) as mock_compile,
    ):
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        from valence.core.maintenance import recompile_ungrouped_sources

        result = await recompile_ungrouped_sources(limit=50)

    assert result == {"checked": 0, "compiled": 0, "articles_created": 0}
    mock_similar.assert_not_called()
    mock_compile.assert_not_called()


@pytest.mark.asyncio
async def test_recompile_ungrouped_no_similar_found():
    source_a = "aaaaaaaa-0000-0000-0000-000000000001"

    with (
        patch("valence.core.maintenance.get_cursor") as mock_get_cursor,
        patch("valence.core.maintenance.find_similar_ungrouped", new_callable=AsyncMock) as mock_similar,
        patch("valence.core.maintenance.compile_article", new_callable=AsyncMock) as mock_compile,
    ):
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [{"id": source_a}]
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_similar.return_value = []

        from valence.core.maintenance import recompile_ungrouped_sources

        result = await recompile_ungrouped_sources(limit=50)

    assert result["checked"] == 1
    assert result["compiled"] == 0
    assert result["articles_created"] == 0
    mock_compile.assert_not_called()


@pytest.mark.asyncio
async def test_recompile_ungrouped_handles_compile_failure():
    source_a = "aaaaaaaa-0000-0000-0000-000000000001"
    source_b = "bbbbbbbb-0000-0000-0000-000000000002"

    with (
        patch("valence.core.maintenance.get_cursor") as mock_get_cursor,
        patch("valence.core.maintenance.find_similar_ungrouped", new_callable=AsyncMock) as mock_similar,
        patch("valence.core.maintenance.compile_article", new_callable=AsyncMock) as mock_compile,
    ):
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [{"id": source_a}, {"id": source_b}]
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_similar.return_value = [source_b]
        mock_compile.return_value = MagicMock(success=False, error="LLM timeout")

        from valence.core.maintenance import recompile_ungrouped_sources

        result = await recompile_ungrouped_sources(limit=50)

    assert result["articles_created"] == 0


def test_admin_maintenance_recompile_ungrouped_param():
    import inspect
    from valence.mcp.handlers.admin import admin_maintenance

    sig = inspect.signature(admin_maintenance)
    assert "recompile_ungrouped" in sig.parameters
    assert sig.parameters["recompile_ungrouped"].default is False


def test_admin_maintenance_recompile_ungrouped_calls_function():
    fake_stats = {"checked": 10, "compiled": 4, "articles_created": 2}

    with patch("valence.mcp.handlers.admin.run_async", side_effect=lambda coro: fake_stats):
        with patch("valence.core.maintenance.recompile_ungrouped_sources", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = fake_stats
            from valence.mcp.handlers.admin import admin_maintenance

            result = admin_maintenance(recompile_ungrouped=True)

    assert result["success"] is True
    assert "recompile_ungrouped" in result["maintenance_results"]
