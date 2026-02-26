"""Tests for article right-sizing (source-grounded split and merge).

Split and merge now delegate to compile_article() which recompiles from
linked sources. These tests verify orchestration, not compilation itself.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

# ---------------------------------------------------------------------------
# Shared IDs & helpers
# ---------------------------------------------------------------------------

ARTICLE_ID = str(uuid4())
ARTICLE_ID_A = str(uuid4())
ARTICLE_ID_B = str(uuid4())
SOURCE_ID_1 = str(uuid4())
SOURCE_ID_2 = str(uuid4())
SOURCE_ID_3 = str(uuid4())

NOW = datetime.now()


def _article_row(
    article_id: str = ARTICLE_ID,
    content: str = "First paragraph about Python.\n\nSecond paragraph about Ruby.",
    title: str | None = "Test Article",
    version: int = 1,
    status: str = "active",
    **kwargs,
) -> dict:
    return {
        "id": article_id,
        "content": content,
        "title": title,
        "author_type": kwargs.get("author_type", "system"),
        "pinned": False,
        "size_tokens": 20,
        "compiled_at": NOW,
        "usage_score": 0,
        "confidence": json.dumps({"overall": 0.7}),
        "domain_path": kwargs.get("domain_path", ["test"]),
        "valid_from": None,
        "valid_until": None,
        "created_at": NOW,
        "modified_at": NOW,
        "source_id": None,
        "extraction_method": None,
        "extraction_metadata": None,
        "supersedes_id": None,
        "superseded_by_id": None,
        "holder_id": None,
        "version": version,
        "content_hash": "abc123",
        "status": status,
    }


def _make_cursor(fetchone_seq=None, fetchall_seq=None):
    cur = MagicMock()
    if fetchone_seq is not None:
        cur.fetchone.side_effect = list(fetchone_seq)
    else:
        cur.fetchone.return_value = None
    if fetchall_seq is not None:
        cur.fetchall.side_effect = list(fetchall_seq)
    else:
        cur.fetchall.return_value = []
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    return cur


# ---------------------------------------------------------------------------
# _split_content_at_midpoint (helper, still exists for fallback)
# ---------------------------------------------------------------------------


class TestSplitContentAtMidpoint:
    async def test_splits_at_paragraph_boundary(self):
        from valence.core.articles import _split_content_at_midpoint

        content = "First paragraph about Python.\n\nSecond paragraph about Ruby."
        first, second = _split_content_at_midpoint(content)
        assert len(first) > 0
        assert len(second) > 0
        combined = first + "\n\n" + second
        assert "Python" in combined
        assert "Ruby" in combined

    def test_falls_back_to_hard_midpoint(self):
        from valence.core.articles import _split_content_at_midpoint

        content = "a b c d e f g h i j k l m n o p q r s t"
        first, second = _split_content_at_midpoint(content)
        assert len(first) > 0
        assert len(second) > 0

    def test_preserves_all_content(self):
        from valence.core.articles import _split_content_at_midpoint

        content = "Alpha.\n\nBeta.\n\nGamma.\n\nDelta."
        first, second = _split_content_at_midpoint(content)
        assert "Alpha" in first or "Alpha" in second
        assert "Delta" in first or "Delta" in second


# ---------------------------------------------------------------------------
# split_article (source-grounded)
# ---------------------------------------------------------------------------


class TestSplitArticle:
    async def test_article_not_found(self):
        from valence.core.articles import split_article

        cur = _make_cursor(fetchone_seq=[None])
        with patch("valence.core.articles.get_cursor", return_value=cur):
            result = await split_article(str(uuid4()))
        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_content_too_short(self):
        from valence.core.articles import split_article

        short = _article_row(content="Hi")
        cur = _make_cursor(fetchone_seq=[short])
        with patch("valence.core.articles.get_cursor", return_value=cur):
            result = await split_article(ARTICLE_ID)
        assert result.success is False

    async def test_no_linked_sources(self):
        from valence.core.articles import split_article

        cur = _make_cursor(fetchone_seq=[_article_row()], fetchall_seq=[[]])
        with patch("valence.core.articles.get_cursor", return_value=cur):
            result = await split_article(ARTICLE_ID)
        assert result.success is False
        assert "no linked sources" in result.error.lower()

    async def test_delegates_to_compile_article(self):
        from valence.core.articles import split_article
        from valence.core.response import ValenceResponse

        cur = _make_cursor(
            fetchone_seq=[_article_row()],
            fetchall_seq=[[{"source_id": SOURCE_ID_1}, {"source_id": SOURCE_ID_2}]],
        )
        art_a = {"id": str(uuid4()), "title": "Python", "size_tokens": 50}
        art_b = {"id": str(uuid4()), "title": "Ruby", "size_tokens": 50}

        async def mock_compile(sids, title_hint=None):
            return ValenceResponse(
                success=True,
                data={"_all_articles": [art_a, art_b], **art_a},
            )

        with (
            patch("valence.core.articles.get_cursor", return_value=cur),
            patch("valence.core.compilation.compile_article", side_effect=mock_compile),
        ):
            result = await split_article(ARTICLE_ID)

        assert result.success is True
        assert len(result.data["articles"]) == 2

    async def test_compile_failure_propagates(self):
        from valence.core.articles import split_article
        from valence.core.response import ValenceResponse

        cur = _make_cursor(
            fetchone_seq=[_article_row()],
            fetchall_seq=[[{"source_id": SOURCE_ID_1}]],
        )

        async def mock_compile(sids, title_hint=None):
            return ValenceResponse(success=False, error="boom")

        with (
            patch("valence.core.articles.get_cursor", return_value=cur),
            patch("valence.core.compilation.compile_article", side_effect=mock_compile),
        ):
            result = await split_article(ARTICLE_ID)
        assert result.success is False

    async def test_single_article_means_no_split(self):
        from valence.core.articles import split_article
        from valence.core.response import ValenceResponse

        cur = _make_cursor(
            fetchone_seq=[_article_row()],
            fetchall_seq=[[{"source_id": SOURCE_ID_1}]],
        )
        single = {"id": str(uuid4()), "title": "Coherent", "size_tokens": 100}

        async def mock_compile(sids, title_hint=None):
            return ValenceResponse(success=True, data=single)

        with (
            patch("valence.core.articles.get_cursor", return_value=cur),
            patch("valence.core.compilation.compile_article", side_effect=mock_compile),
        ):
            result = await split_article(ARTICLE_ID)
        assert result.success is False
        assert "coherent" in result.error.lower()


# ---------------------------------------------------------------------------
# merge_articles (source-grounded)
# ---------------------------------------------------------------------------


class TestMergeArticles:
    async def test_article_a_not_found(self):
        from valence.core.articles import merge_articles

        cur = _make_cursor(fetchone_seq=[None])
        with patch("valence.core.articles.get_cursor", return_value=cur):
            result = await merge_articles(str(uuid4()), ARTICLE_ID_B)
        assert result.success is False

    async def test_article_b_not_found(self):
        from valence.core.articles import merge_articles

        cur = _make_cursor(fetchone_seq=[{"id": ARTICLE_ID_A}, None])
        with patch("valence.core.articles.get_cursor", return_value=cur):
            result = await merge_articles(ARTICLE_ID_A, str(uuid4()))
        assert result.success is False

    async def test_no_sources_errors(self):
        from valence.core.articles import merge_articles

        cur = _make_cursor(
            fetchone_seq=[{"id": ARTICLE_ID_A}, {"id": ARTICLE_ID_B}, {"title": "A"}, {"title": "B"}],
            fetchall_seq=[[]],
        )
        with patch("valence.core.articles.get_cursor", return_value=cur):
            result = await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)
        assert result.success is False

    async def test_delegates_to_compile_article(self):
        from valence.core.articles import merge_articles
        from valence.core.response import ValenceResponse

        merged = {"id": str(uuid4()), "title": "Merged", "size_tokens": 100}

        async def mock_compile(sids, title_hint=None):
            return ValenceResponse(success=True, data=merged)

        cur = _make_cursor(
            fetchone_seq=[
                {"id": ARTICLE_ID_A},
                {"id": ARTICLE_ID_B},
                {"title": "A"},
                {"title": "B"},
            ],
            fetchall_seq=[[{"source_id": SOURCE_ID_1}, {"source_id": SOURCE_ID_2}]],
        )

        with (
            patch("valence.core.articles.get_cursor", return_value=cur),
            patch("valence.core.compilation.compile_article", side_effect=mock_compile),
        ):
            result = await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)

        assert result.success is True

    async def test_compile_failure_propagates(self):
        from valence.core.articles import merge_articles
        from valence.core.response import ValenceResponse

        async def mock_compile(sids, title_hint=None):
            return ValenceResponse(success=False, error="boom")

        cur = _make_cursor(
            fetchone_seq=[
                {"id": ARTICLE_ID_A},
                {"id": ARTICLE_ID_B},
                {"title": "A"},
                {"title": "B"},
            ],
            fetchall_seq=[[{"source_id": SOURCE_ID_1}]],
        )

        with (
            patch("valence.core.articles.get_cursor", return_value=cur),
            patch("valence.core.compilation.compile_article", side_effect=mock_compile),
        ):
            result = await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)
        assert result.success is False


# ---------------------------------------------------------------------------
# Provenance chain integrity
# ---------------------------------------------------------------------------


class TestProvenanceChainIntegrity:
    async def test_split_passes_all_sources(self):
        from valence.core.articles import split_article
        from valence.core.response import ValenceResponse

        collected_ids = []

        async def mock_compile(sids, title_hint=None):
            collected_ids.extend(sids)
            art_a = {"id": str(uuid4()), "title": "A", "size_tokens": 50}
            art_b = {"id": str(uuid4()), "title": "B", "size_tokens": 50}
            return ValenceResponse(success=True, data={"_all_articles": [art_a, art_b], **art_a})

        cur = _make_cursor(
            fetchone_seq=[_article_row()],
            fetchall_seq=[[{"source_id": SOURCE_ID_1}, {"source_id": SOURCE_ID_2}]],
        )
        with (
            patch("valence.core.articles.get_cursor", return_value=cur),
            patch("valence.core.compilation.compile_article", side_effect=mock_compile),
        ):
            await split_article(ARTICLE_ID)

        assert SOURCE_ID_1 in collected_ids
        assert SOURCE_ID_2 in collected_ids

    async def test_merge_collects_all_sources(self):
        from valence.core.articles import merge_articles
        from valence.core.response import ValenceResponse

        collected_ids = []

        async def mock_compile(sids, title_hint=None):
            collected_ids.extend(sids)
            return ValenceResponse(success=True, data={"id": str(uuid4()), "title": "M"})

        cur = _make_cursor(
            fetchone_seq=[
                {"id": ARTICLE_ID_A},
                {"id": ARTICLE_ID_B},
                {"title": "A"},
                {"title": "B"},
            ],
            fetchall_seq=[
                [
                    {"source_id": SOURCE_ID_1},
                    {"source_id": SOURCE_ID_2},
                    {"source_id": SOURCE_ID_3},
                ]
            ],
        )

        with (
            patch("valence.core.articles.get_cursor", return_value=cur),
            patch("valence.core.compilation.compile_article", side_effect=mock_compile),
        ):
            await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)

        assert SOURCE_ID_1 in collected_ids
        assert SOURCE_ID_2 in collected_ids
        assert SOURCE_ID_3 in collected_ids


# ---------------------------------------------------------------------------
# Mutation queue dispatch
# ---------------------------------------------------------------------------


class TestMutationQueueSplitIntegration:
    async def test_queue_dispatches_split(self):
        from valence.core.compilation import process_mutation_queue
        from valence.core.response import ValenceResponse

        cur = _make_cursor(
            fetchall_seq=[
                [
                    {
                        "id": 1,
                        "operation": "split",
                        "article_id": ARTICLE_ID,
                        "payload": {},
                        "priority": 3,
                        "status": "pending",
                    }
                ]
            ],
        )
        mock_split = MagicMock()
        mock_split.return_value = ValenceResponse(success=True, data={})
        mock_status = MagicMock()

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            patch("valence.core.articles.split_article", mock_split),
            patch("valence.core.compilation._set_queue_item_status", mock_status),
        ):
            count = await process_mutation_queue()

        mock_split.assert_called_once_with(ARTICLE_ID)

    async def test_queue_dispatches_merge(self):
        from valence.core.compilation import process_mutation_queue
        from valence.core.response import ValenceResponse

        candidate_id = str(uuid4())
        cur = _make_cursor(
            fetchall_seq=[
                [
                    {
                        "id": 2,
                        "operation": "merge_candidate",
                        "article_id": ARTICLE_ID_A,
                        "payload": {"candidate_article_id": candidate_id},
                        "priority": 3,
                        "status": "pending",
                    }
                ]
            ],
        )
        mock_merge = MagicMock()
        mock_merge.return_value = ValenceResponse(success=True, data={})
        mock_status = MagicMock()

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            patch("valence.core.articles.merge_articles", mock_merge),
            patch("valence.core.compilation._set_queue_item_status", mock_status),
        ):
            count = await process_mutation_queue()

        mock_merge.assert_called_once_with(ARTICLE_ID_A, candidate_id)
        status_call_args = mock_status.call_args[0]
        assert status_call_args[1] == "completed"
        assert count.data == 1
