"""Tests for source-grounded article splitting (#502).

split_article() now delegates to compile_article() which recompiles from
linked sources using the v3 multi-article prompt. The LLM decides topic
partitioning. These tests verify the orchestration logic, not the compilation
itself (that's tested in test_compilation.py / test_inference_schemas.py).
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.core.response import ValenceResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARTICLE_ID = str(uuid4())
SOURCE_ID_1 = str(uuid4())
SOURCE_ID_2 = str(uuid4())


def _make_article_row(**kwargs) -> dict:
    now = datetime.now()
    defaults = {
        "id": ARTICLE_ID,
        "content": "First topic about Python.\n\nSecond topic about JavaScript.",
        "title": "Programming Languages",
        "author_type": "system",
        "pinned": False,
        "size_tokens": 12,
        "compiled_at": now,
        "usage_score": 0,
        "confidence": json.dumps({"overall": 0.7}),
        "domain_path": ["programming"],
        "version": 1,
        "content_hash": "abc123",
        "status": "active",
        "created_at": now,
        "modified_at": now,
        "superseded_by_id": None,
    }
    defaults.update(kwargs)
    return defaults


def _make_cursor_mock(fetchone_seq=None, fetchall_seq=None):
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    if fetchone_seq is not None:
        mock_cur.fetchone.side_effect = list(fetchone_seq)
    else:
        mock_cur.fetchone.return_value = None
    if fetchall_seq is not None:
        mock_cur.fetchall.side_effect = list(fetchall_seq)
    else:
        mock_cur.fetchall.return_value = []
    return mock_cur


def _ok(data, degraded=False):
    return ValenceResponse(success=True, data=data, degraded=degraded)


def _err(msg):
    return ValenceResponse(success=False, error=msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSplitArticle:
    async def test_article_not_found(self):
        from valence.core.articles import split_article

        mock_cur = _make_cursor_mock(fetchone_seq=[None])
        with patch("valence.core.articles.get_cursor", return_value=mock_cur):
            result = await split_article(str(uuid4()))
        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_content_too_short(self):
        from valence.core.articles import split_article

        short = _make_article_row(content="Too short")
        mock_cur = _make_cursor_mock(fetchone_seq=[short])
        with patch("valence.core.articles.get_cursor", return_value=mock_cur):
            result = await split_article(ARTICLE_ID)
        assert result.success is False
        assert "too short" in result.error.lower()

    async def test_no_linked_sources(self):
        from valence.core.articles import split_article

        article = _make_article_row()
        mock_cur = _make_cursor_mock(fetchone_seq=[article], fetchall_seq=[[]])
        with patch("valence.core.articles.get_cursor", return_value=mock_cur):
            result = await split_article(ARTICLE_ID)
        assert result.success is False
        assert "no linked sources" in result.error.lower()

    async def test_successful_split_delegates_to_compile(self):
        from valence.core.articles import split_article

        article = _make_article_row()
        source_rows = [{"source_id": SOURCE_ID_1}, {"source_id": SOURCE_ID_2}]

        new_art_1 = {"id": str(uuid4()), "title": "Python Overview", "size_tokens": 50}
        new_art_2 = {"id": str(uuid4()), "title": "JavaScript Fundamentals", "size_tokens": 50}
        compile_result = _ok({"_all_articles": [new_art_1, new_art_2], **new_art_1})

        mock_cur = _make_cursor_mock(
            fetchone_seq=[article],
            fetchall_seq=[source_rows],
        )

        async def mock_compile(source_ids, title_hint=None):
            assert set(source_ids) == {SOURCE_ID_1, SOURCE_ID_2}
            return compile_result

        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.compilation.compile_article", side_effect=mock_compile),
        ):
            result = await split_article(ARTICLE_ID)

        assert result.success is True
        assert "articles" in result.data
        assert len(result.data["articles"]) == 2

        # Verify original was archived
        calls_str = str(mock_cur.execute.call_args_list)
        assert "archived" in calls_str

    async def test_compile_failure_propagates(self):
        from valence.core.articles import split_article

        article = _make_article_row()
        source_rows = [{"source_id": SOURCE_ID_1}]
        mock_cur = _make_cursor_mock(
            fetchone_seq=[article],
            fetchall_seq=[source_rows],
        )

        async def mock_compile(source_ids, title_hint=None):
            return _err("inference unavailable")

        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.compilation.compile_article", side_effect=mock_compile),
        ):
            result = await split_article(ARTICLE_ID)

        assert result.success is False
        assert "failed" in result.error.lower()

    async def test_single_article_result_means_no_split(self):
        from valence.core.articles import split_article

        article = _make_article_row()
        source_rows = [{"source_id": SOURCE_ID_1}]
        single_art = {"id": str(uuid4()), "title": "Coherent Topic", "size_tokens": 100}
        compile_result = _ok(single_art)  # No _all_articles → single article

        mock_cur = _make_cursor_mock(
            fetchone_seq=[article],
            fetchall_seq=[source_rows],
        )

        async def mock_compile(source_ids, title_hint=None):
            return compile_result

        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.compilation.compile_article", side_effect=mock_compile),
        ):
            result = await split_article(ARTICLE_ID)

        assert result.success is False
        assert "coherent" in result.error.lower()
