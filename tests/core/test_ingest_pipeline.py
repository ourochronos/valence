# SPDX-License-Identifier: MIT
"""Tests for valence.core.ingest_pipeline (issue #565)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from valence.core.ingest_pipeline import (
    _mean_vector,
    _flatten_tree,
    run_source_pipeline,
)


# ---------------------------------------------------------------------------
# Unit tests — pure helpers
# ---------------------------------------------------------------------------


class TestMeanVector:
    def test_single_vector(self):
        result = _mean_vector([[1.0, 2.0, 3.0]])
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_multiple_vectors(self):
        result = _mean_vector([[1.0, 2.0], [3.0, 4.0]])
        assert result == pytest.approx([2.0, 3.0])

    def test_empty(self):
        assert _mean_vector([]) == []

    def test_three_vectors(self):
        result = _mean_vector([[0.0, 6.0], [3.0, 3.0], [6.0, 0.0]])
        assert result == pytest.approx([3.0, 3.0])


# ---------------------------------------------------------------------------
# Integration-ish tests with mocks
# ---------------------------------------------------------------------------


def _make_cursor(source_row=None, section_rows=None):
    cur = MagicMock()
    cur.fetchone.return_value = source_row
    cur.fetchall.return_value = section_rows or []
    return cur


@contextmanager
def _mock_cursor_ctx(cur):
    @contextmanager
    def _inner(dict_cursor=True):
        yield cur

    with patch("valence.core.ingest_pipeline.get_cursor", _inner):
        yield cur


class TestRunSourcePipeline:
    @pytest.mark.asyncio
    async def test_source_not_found(self):
        cur = _make_cursor(source_row=None)
        with _mock_cursor_ctx(cur):
            result = await run_source_pipeline("nonexistent-id")
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_content(self):
        source_row = {
            "content": "   ",
            "title": "Empty",
            "metadata": {},
        }
        cur = _make_cursor(source_row=source_row)
        # Need cursor to be context-manager-able multiple times
        call_count = 0

        @contextmanager
        def _inner(dict_cursor=True):
            nonlocal call_count
            call_count += 1
            yield cur

        with patch("valence.core.ingest_pipeline.get_cursor", _inner):
            # Also need to mock _set_pipeline_status
            with patch("valence.core.ingest_pipeline._set_pipeline_status"):
                result = await run_source_pipeline("some-id")
        assert result.success
        assert result.data["sections_embedded"] == 0

    @pytest.mark.asyncio
    async def test_batch_mode_enqueues(self):
        with patch("valence.core.ingest_pipeline._enqueue_pipeline_task") as mock_enqueue:
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = None
                result = await run_source_pipeline("some-id", batch_mode=True)
        assert result.success
        assert result.data["queued"] is True

    @pytest.mark.asyncio
    async def test_small_source_trivial_tree(self):
        """Small sources get a trivial single-node tree (no LLM call)."""
        source_id = str(uuid4())
        content = "Short content"  # well under SMALL_SOURCE_CHARS

        source_row = {"content": content, "title": "Test", "metadata": {}}
        section_vecs = [[0.1, 0.2, 0.3]]

        call_n = [0]

        @contextmanager
        def _cursor(dict_cursor=True):
            call_n[0] += 1
            c = MagicMock()
            c.fetchone.return_value = source_row if call_n[0] == 1 else None
            c.fetchall.return_value = []
            yield c

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            with patch("valence.core.ingest_pipeline._embed_and_upsert", return_value=1):
                with patch("valence.core.ingest_pipeline._get_section_vectors", return_value=section_vecs):
                    with patch("valence.core.ingest_pipeline._update_source_embedding"):
                        with patch("valence.core.ingest_pipeline._set_pipeline_status"):
                            with patch("valence.core.ingest_pipeline._stage_auto_compile", new_callable=AsyncMock, return_value=False):
                                result = await run_source_pipeline(source_id)

        assert result.success
        assert result.data["sections_embedded"] == 1
        assert result.data["compiled"] is False
