# SPDX-License-Identifier: MIT
"""Tests for valence.core.ingest_pipeline (issue #565)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# flatten_tree lives in section_embeddings; compose_embedding lives in embeddings
from valence.core.embeddings import compose_embedding
from valence.core.ingest_pipeline import run_source_pipeline
from valence.core.section_embeddings import flatten_tree


# ---------------------------------------------------------------------------
# Unit tests — pure helpers
# ---------------------------------------------------------------------------


class TestComposeEmbedding:
    def test_single_vector(self):
        result = compose_embedding([[1.0, 2.0, 3.0]])
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_multiple_vectors(self):
        result = compose_embedding([[1.0, 2.0], [3.0, 4.0]])
        assert result == pytest.approx([2.0, 3.0])

    def test_empty(self):
        assert compose_embedding([]) == []

    def test_three_vectors(self):
        result = compose_embedding([[0.0, 6.0], [3.0, 3.0], [6.0, 0.0]])
        assert result == pytest.approx([3.0, 3.0])


class TestFlattenTree:
    def test_single_node(self):
        nodes = [{"title": "Root", "summary": "", "start_char": 0, "end_char": 100, "children": []}]
        flat = flatten_tree(nodes)
        assert len(flat) == 1
        assert flat[0]["tree_path"] == "0"
        assert flat[0]["depth"] == 0

    def test_nested_nodes(self):
        nodes = [
            {
                "title": "Parent",
                "summary": "",
                "start_char": 0,
                "end_char": 200,
                "children": [
                    {"title": "Child A", "summary": "", "start_char": 0, "end_char": 100, "children": []},
                    {"title": "Child B", "summary": "", "start_char": 100, "end_char": 200, "children": []},
                ],
            }
        ]
        flat = flatten_tree(nodes)
        assert len(flat) == 3
        paths = [n["tree_path"] for n in flat]
        assert "0" in paths
        assert "0.0" in paths
        assert "0.1" in paths

    def test_empty(self):
        assert flatten_tree([]) == []


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
        call_count = 0

        @contextmanager
        def _inner(dict_cursor=True):
            nonlocal call_count
            call_count += 1
            yield cur

        with patch("valence.core.ingest_pipeline.get_cursor", _inner):
            with patch("valence.core.ingest_pipeline._set_pipeline_status"):
                result = await run_source_pipeline("some-id")
        assert result.success
        assert result.data["sections_embedded"] == 0

    @pytest.mark.asyncio
    async def test_batch_mode_enqueues(self):
        with patch("valence.core.ingest_pipeline._enqueue_pipeline_task"):
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = None
                result = await run_source_pipeline("some-id", batch_mode=True)
        assert result.success
        assert result.data["queued"] is True

    @pytest.mark.asyncio
    async def test_small_source_embeds_sections(self):
        """Small sources get section-embedded without LLM tree call."""
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

        embed_ok = MagicMock()
        embed_ok.success = True
        embed_ok.data = {"sections_embedded": 1}

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            with patch(
                "valence.core.ingest_pipeline.embed_source_sections",
                new_callable=AsyncMock,
                return_value=embed_ok,
            ):
                with patch(
                    "valence.core.ingest_pipeline.get_section_vectors",
                    return_value=section_vecs,
                ):
                    with patch("valence.core.ingest_pipeline.store_source_embedding"):
                        with patch("valence.core.ingest_pipeline._set_pipeline_status"):
                            with patch(
                                "valence.core.ingest_pipeline.asyncio.to_thread",
                                new_callable=AsyncMock,
                            ) as mock_to_thread:
                                # First call: get_section_vectors -> section_vecs
                                # Second call: store_source_embedding -> None
                                mock_to_thread.side_effect = [section_vecs, None]
                                with patch(
                                    "valence.core.sources.find_similar_ungrouped",
                                    new_callable=AsyncMock,
                                    return_value=[],
                                ):
                                    result = await run_source_pipeline(source_id)

        assert result.success
        assert result.data["sections_embedded"] == 1
        assert result.data["compiled"] is False
