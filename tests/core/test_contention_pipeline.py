# SPDX-License-Identifier: MIT
"""Tests for Stage 6 contention auto-detection in ingest_pipeline (#67)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from valence.core.ingest_pipeline import (
    CONTENTION_SIMILARITY_THRESHOLD,
    _check_contentions,
    run_source_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source_row(content="Some content about topic X.", title="Test Source"):
    return {"id": str(uuid4()), "content": content, "title": title, "metadata": {}}


def _make_contention(article_id, source_id):
    return {
        "id": str(uuid4()),
        "article_id": article_id,
        "source_id": source_id,
        "type": "contradiction",
        "status": "detected",
        "materiality": 0.7,
    }


@contextmanager
def _mock_cursor_seq(side_effects):
    """Mock get_cursor to return different fetchall/fetchone results in sequence."""
    call_n = [0]

    @contextmanager
    def _inner(dict_cursor=True):
        cur = MagicMock()
        idx = call_n[0]
        call_n[0] += 1
        if idx < len(side_effects):
            effect = side_effects[idx]
            if isinstance(effect, list):
                cur.fetchall.return_value = effect
                cur.fetchone.return_value = effect[0] if effect else None
            else:
                cur.fetchone.return_value = effect
                cur.fetchall.return_value = [effect] if effect else []
        else:
            cur.fetchall.return_value = []
            cur.fetchone.return_value = None
        yield cur

    with patch("valence.core.ingest_pipeline.get_cursor", _inner):
        yield


# ---------------------------------------------------------------------------
# Unit tests for _check_contentions
# ---------------------------------------------------------------------------


class TestCheckContentions:
    @pytest.mark.asyncio
    async def test_calls_detect_contention_for_similar_articles(self):
        """detect_contention is called for each similar unlinked article."""
        source_id = str(uuid4())
        article_id = str(uuid4())

        linked_rows = []  # source not linked to any article
        similar_rows = [{"id": article_id}]

        contention_data = _make_contention(article_id, source_id)
        detect_result = MagicMock(success=True, data=contention_data)

        call_n = [0]

        @contextmanager
        def _cursor(dict_cursor=True):
            cur = MagicMock()
            n = call_n[0]
            call_n[0] += 1
            if n == 0:
                # First call: fetch linked article_ids
                cur.fetchall.return_value = linked_rows
            else:
                # Second call: fetch similar articles
                cur.fetchall.return_value = similar_rows
            yield cur

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            with patch(
                "valence.core.contention.detect_contention",
                new_callable=AsyncMock,
                return_value=detect_result,
            ) as mock_detect:
                # Import and patch at the right location
                with patch(
                    "valence.core.ingest_pipeline._check_contentions",
                    wraps=_check_contentions,
                ):
                    pass

            # Patch detect_contention inside the ingest_pipeline module's import
            with patch(
                "valence.core.contention.detect_contention",
                new_callable=AsyncMock,
                return_value=detect_result,
            ):
                # Re-patch get_cursor for the actual call
                call_n[0] = 0

                @contextmanager
                def _cursor2(dict_cursor=True):
                    cur = MagicMock()
                    n = call_n[0]
                    call_n[0] += 1
                    if n == 0:
                        cur.fetchall.return_value = linked_rows
                    else:
                        cur.fetchall.return_value = similar_rows
                    yield cur

                with patch("valence.core.ingest_pipeline.get_cursor", _cursor2):
                    results = await _check_contentions(source_id)

        assert len(results) == 1
        assert results[0]["article_id"] == article_id

    @pytest.mark.asyncio
    async def test_skips_already_linked_articles(self):
        """Articles already linked to the source via article_sources are skipped."""
        source_id = str(uuid4())
        article_id = str(uuid4())

        # Source IS linked to this article
        linked_rows = [{"article_id": article_id}]
        similar_rows = [{"id": article_id}]

        call_n = [0]

        @contextmanager
        def _cursor(dict_cursor=True):
            cur = MagicMock()
            n = call_n[0]
            call_n[0] += 1
            if n == 0:
                cur.fetchall.return_value = linked_rows
            else:
                cur.fetchall.return_value = similar_rows
            yield cur

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            with patch(
                "valence.core.contention.detect_contention",
                new_callable=AsyncMock,
            ) as mock_detect:
                results = await _check_contentions(source_id)

        # detect_contention should NOT have been called
        mock_detect.assert_not_called()
        assert results == []

    @pytest.mark.asyncio
    async def test_no_similar_articles_returns_empty(self):
        """When no similar articles are found, returns empty list without error."""
        source_id = str(uuid4())

        call_n = [0]

        @contextmanager
        def _cursor(dict_cursor=True):
            cur = MagicMock()
            call_n[0] += 1
            cur.fetchall.return_value = []
            yield cur

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            with patch(
                "valence.core.contention.detect_contention",
                new_callable=AsyncMock,
            ) as mock_detect:
                results = await _check_contentions(source_id)

        mock_detect.assert_not_called()
        assert results == []

    @pytest.mark.asyncio
    async def test_detect_contention_failure_is_non_fatal(self):
        """If detect_contention raises, the loop continues and returns partial results."""
        source_id = str(uuid4())
        article_id_a = str(uuid4())
        article_id_b = str(uuid4())

        similar_rows = [{"id": article_id_a}, {"id": article_id_b}]
        contention_data = _make_contention(article_id_b, source_id)
        good_result = MagicMock(success=True, data=contention_data)

        call_n = [0]

        @contextmanager
        def _cursor(dict_cursor=True):
            cur = MagicMock()
            n = call_n[0]
            call_n[0] += 1
            if n == 0:
                cur.fetchall.return_value = []  # no linked articles
            else:
                cur.fetchall.return_value = similar_rows
            yield cur

        async def _detect_side_effect(article_id, source_id):
            if article_id == article_id_a:
                raise RuntimeError("LLM timeout")
            return good_result

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            with patch(
                "valence.core.contention.detect_contention",
                side_effect=_detect_side_effect,
            ):
                results = await _check_contentions(source_id)

        # Only the second article's contention should be in results
        assert len(results) == 1
        assert results[0]["article_id"] == article_id_b

    @pytest.mark.asyncio
    async def test_db_error_fetching_linked_returns_empty(self):
        """If the first DB call (linked articles) fails, returns empty list."""
        source_id = str(uuid4())

        @contextmanager
        def _cursor(dict_cursor=True):
            cur = MagicMock()
            cur.fetchall.side_effect = Exception("connection error")
            yield cur

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            results = await _check_contentions(source_id)

        assert results == []

    @pytest.mark.asyncio
    async def test_db_error_fetching_similar_returns_empty(self):
        """If the second DB call (similar articles) fails, returns empty list."""
        source_id = str(uuid4())
        call_n = [0]

        @contextmanager
        def _cursor(dict_cursor=True):
            cur = MagicMock()
            n = call_n[0]
            call_n[0] += 1
            if n == 0:
                cur.fetchall.return_value = []
            else:
                cur.fetchall.side_effect = Exception("vector extension missing")
            yield cur

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            results = await _check_contentions(source_id)

        assert results == []

    @pytest.mark.asyncio
    async def test_no_contention_data_not_included(self):
        """detect_contention returning data=None means no contention (skipped)."""
        source_id = str(uuid4())
        article_id = str(uuid4())

        similar_rows = [{"id": article_id}]
        no_contention_result = MagicMock(success=True, data=None)

        call_n = [0]

        @contextmanager
        def _cursor(dict_cursor=True):
            cur = MagicMock()
            n = call_n[0]
            call_n[0] += 1
            if n == 0:
                cur.fetchall.return_value = []
            else:
                cur.fetchall.return_value = similar_rows
            yield cur

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            with patch(
                "valence.core.contention.detect_contention",
                new_callable=AsyncMock,
                return_value=no_contention_result,
            ):
                results = await _check_contentions(source_id)

        assert results == []


# ---------------------------------------------------------------------------
# Integration tests: stage 6 wired into run_source_pipeline
# ---------------------------------------------------------------------------


class TestRunSourcePipelineContentionStage:
    """Test that Stage 6 runs as part of the full pipeline."""

    def _make_full_pipeline_patches(self, source_id, source_content="Hello world"):
        """Return a dict of patches for a successful pipeline run."""
        source_row = {"content": source_content, "title": "T", "metadata": {}}
        embed_ok = MagicMock(success=True, data={"sections_embedded": 1})
        section_vecs = [[0.1, 0.2, 0.3]]

        return source_row, embed_ok, section_vecs

    @pytest.mark.asyncio
    async def test_pipeline_includes_contentions_in_result(self):
        """Pipeline result includes contentions list from stage 6."""
        source_id = str(uuid4())
        article_id = str(uuid4())
        source_row = {"content": "Some content", "title": "T", "metadata": {}}
        embed_ok = MagicMock(success=True, data={"sections_embedded": 1})
        section_vecs = [[0.1, 0.2, 0.3]]

        contention_data = _make_contention(article_id, source_id)

        call_n = [0]

        @contextmanager
        def _cursor(dict_cursor=True):
            cur = MagicMock()
            n = call_n[0]
            call_n[0] += 1
            if n == 0:
                # Stage 1: fetch source
                cur.fetchone.return_value = source_row
            elif n == 1:
                # Stage 5: check already_grouped
                cur.fetchone.return_value = None
            elif n == 2:
                # Stage 6: fetch linked articles
                cur.fetchall.return_value = []
            elif n == 3:
                # Stage 6: fetch similar articles
                cur.fetchall.return_value = [{"id": article_id}]
            else:
                cur.fetchone.return_value = None
                cur.fetchall.return_value = []
            yield cur

        good_result = MagicMock(success=True, data=contention_data)

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            with patch(
                "valence.core.ingest_pipeline.embed_source_sections",
                new_callable=AsyncMock,
                return_value=embed_ok,
            ):
                with patch(
                    "valence.core.ingest_pipeline.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_to_thread:
                    mock_to_thread.side_effect = [section_vecs, None]
                    with patch("valence.core.ingest_pipeline._set_pipeline_status"):
                        with patch(
                            "valence.core.sources.find_similar_ungrouped",
                            new_callable=AsyncMock,
                            return_value=[],
                        ):
                            with patch(
                                "valence.core.contention.detect_contention",
                                new_callable=AsyncMock,
                                return_value=good_result,
                            ):
                                result = await run_source_pipeline(source_id)

        assert result.success
        assert "contentions" in result.data
        assert len(result.data["contentions"]) == 1
        assert result.data["contentions"][0]["article_id"] == article_id

    @pytest.mark.asyncio
    async def test_pipeline_status_complete_even_if_contention_fails(self):
        """pipeline_status ends as 'complete' even when contention stage raises."""
        source_id = str(uuid4())
        source_row = {"content": "Some content", "title": "T", "metadata": {}}
        embed_ok = MagicMock(success=True, data={"sections_embedded": 1})
        section_vecs = [[0.1, 0.2, 0.3]]

        statuses = []

        @contextmanager
        def _cursor(dict_cursor=True):
            cur = MagicMock()
            cur.fetchone.return_value = source_row
            cur.fetchall.return_value = []
            yield cur

        async def _bad_check_contentions(sid):
            raise RuntimeError("unexpected DB failure")

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            with patch(
                "valence.core.ingest_pipeline.embed_source_sections",
                new_callable=AsyncMock,
                return_value=embed_ok,
            ):
                with patch(
                    "valence.core.ingest_pipeline.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_to_thread:
                    mock_to_thread.side_effect = [section_vecs, None]
                    with patch(
                        "valence.core.ingest_pipeline._set_pipeline_status",
                        side_effect=lambda sid, s: statuses.append(s),
                    ):
                        with patch(
                            "valence.core.sources.find_similar_ungrouped",
                            new_callable=AsyncMock,
                            return_value=[],
                        ):
                            with patch(
                                "valence.core.ingest_pipeline._check_contentions",
                                side_effect=_bad_check_contentions,
                            ):
                                result = await run_source_pipeline(source_id)

        assert result.success
        assert statuses[-1] == "complete"
        assert result.data["contentions"] == []

    @pytest.mark.asyncio
    async def test_pipeline_empty_content_returns_empty_contentions(self):
        """Empty-content sources get empty contentions list (no stage 6 run)."""
        source_row = {"content": "   ", "title": "Empty", "metadata": {}}

        @contextmanager
        def _cursor(dict_cursor=True):
            cur = MagicMock()
            cur.fetchone.return_value = source_row
            cur.fetchall.return_value = []
            yield cur

        with patch("valence.core.ingest_pipeline.get_cursor", _cursor):
            with patch("valence.core.ingest_pipeline._set_pipeline_status"):
                result = await run_source_pipeline("some-id")

        assert result.success
        assert result.data["contentions"] == []

    @pytest.mark.asyncio
    async def test_constant_value(self):
        """CONTENTION_SIMILARITY_THRESHOLD is 0.6."""
        assert CONTENTION_SIMILARITY_THRESHOLD == 0.6
