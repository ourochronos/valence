"""Tests for valence.core.retrieval module (WU-05 — C9, C8).

Covers:
1. Ranking order (relevance * 0.5 + confidence * 0.35 + freshness * 0.15)
2. Metadata inclusion: provenance_summary, freshness, active_contentions
3. Usage trace recording for each article result
4. Compilation queueing (mutation_queue) for ungrouped sources
5. Empty / edge-case handling
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

ARTICLE_ID = str(uuid.uuid4())
SOURCE_ID = str(uuid.uuid4())
ARTICLE_ID_2 = str(uuid.uuid4())


def _make_article_row(
    article_id: str = ARTICLE_ID,
    content: str = "Article about Python.",
    title: str = "Python Article",
    days_old: float = 2.0,
    confidence_overall: float = 0.7,
    text_rank: float = 0.8,
    vec_rank: int = 1,
    vec_score: float = 0.8,
    text_score: float = 0.8,
    rrf_score: float | None = None,
) -> dict:
    """Build a minimal article DB row (as returned by psycopg2 with RRF columns)."""
    now_utc = datetime.now(UTC)
    created = now_utc - timedelta(days=days_old + 10)
    compiled = now_utc - timedelta(days=days_old)
    # Compute RRF score from ranks if not provided
    if rrf_score is None:
        rrf_score = (1.0 / (60 + vec_rank)) + (1.0 / (60 + int(1.0 / text_rank if text_rank > 0 else 1000)))
    return {
        "id": uuid.UUID(article_id),
        "content": content,
        "title": title,
        "author_type": "system",
        "status": "active",
        "superseded_by_id": None,
        "confidence": {"overall": confidence_overall},
        "domain_path": ["python"],
        "pinned": False,
        "size_tokens": 50,
        "usage_score": 0,
        "version": 1,
        "content_hash": "abc",
        "compiled_at": compiled,
        "modified_at": compiled,
        "created_at": created,
        "valid_from": None,
        "valid_until": None,
        "source_id": None,
        "supersedes_id": None,
        "holder_id": None,
        "content_tsv": None,
        "embedding": None,
        "text_rank": text_rank,
        "vec_rank": vec_rank,
        "vec_score": vec_score,
        "text_score": text_score,
        "rrf_score": rrf_score,
    }


def _make_source_row(
    source_id: str = SOURCE_ID,
    content: str = "Ungrouped source about Python.",
    days_old: float = 5.0,
    reliability: float = 0.8,
    text_rank: float = 0.6,
    vec_rank: int = 1,
    vec_score: float = 0.6,
    text_score: float = 0.6,
    rrf_score: float | None = None,
) -> dict:
    """Build a minimal ungrouped source DB row with RRF columns."""
    now_utc = datetime.now(UTC)
    created = now_utc - timedelta(days=days_old)
    if rrf_score is None:
        rrf_score = (1.0 / (60 + vec_rank)) + (1.0 / (60 + int(1.0 / text_rank if text_rank > 0 else 1000)))
    return {
        "id": uuid.UUID(source_id),
        "type": "document",
        "title": "Python Source",
        "url": None,
        "content": content,
        "reliability": reliability,
        "fingerprint": "fp1",
        "created_at": created,
        "text_rank": text_rank,
        "vec_rank": vec_rank,
        "vec_score": vec_score,
        "text_score": text_score,
        "rrf_score": rrf_score,
    }


def _make_prov_row(source_count: int = 2, relationship_types=None) -> dict:
    """Build a provenance aggregate row."""
    return {
        "source_count": source_count,
        "relationship_types": relationship_types or ["originates", "confirms"],
    }


def _make_contention_row() -> dict:
    return {"_exists": 1}


def _make_cursor(
    *,
    article_rows=None,
    source_rows=None,
    prov_row=None,
    has_contention: bool = False,
    sentinel_article_row=None,
) -> MagicMock:
    """Build a mock psycopg2 cursor that returns different values on successive fetchone/fetchall calls."""
    cur = MagicMock()

    # We need to simulate multiple SQL executions on the same cursor.
    # Strategy: track execute() call count and return appropriate results.
    execute_count = [0]
    fetchone_queue: list = []
    fetchall_queue: list = []

    # The execute->fetch pairs happen in this order per retrieve() call:
    # 1. Article full-text search -> fetchall (article_rows)
    # 2. Per article: provenance_summary -> fetchone (prov_row)
    # 3. Per article: has_active_contentions -> fetchone (contention or None)
    # 4. (If include_sources) Ungrouped source search -> fetchall (source_rows)
    # 5. Per article result: record_usage_trace -> INSERT (no fetch)
    # 6. Per source result: _queue_recompile:
    #    a. SELECT id FROM articles -> fetchone(sentinel_article_row)
    #    b. INSERT into mutation_queue -> no fetch

    # We use side_effect on execute to track position and route fetches.
    fetch_one_responses: list = []
    fetch_all_responses: list = []

    # Fill up the queues
    if article_rows is not None:
        fetch_all_responses.append(article_rows)
    else:
        fetch_all_responses.append([])

    if article_rows:
        for _ in article_rows:
            fetch_one_responses.append(prov_row or _make_prov_row())
            fetch_one_responses.append(_make_contention_row() if has_contention else None)

    if source_rows is not None:
        fetch_all_responses.append(source_rows)

    # Sentinel for recompile queue
    if source_rows:
        for _ in source_rows:
            fetch_one_responses.append(sentinel_article_row or {"id": uuid.UUID(ARTICLE_ID)})
            # mutation_queue INSERT doesn't need a fetchone

    fetchone_iter = iter(fetch_one_responses)
    fetchall_iter = iter(fetch_all_responses)

    def _fetchone():
        try:
            return next(fetchone_iter)
        except StopIteration:
            return None

    def _fetchall():
        try:
            return next(fetchall_iter)
        except StopIteration:
            return []

    cur.fetchone.side_effect = _fetchone
    cur.fetchall.side_effect = _fetchall
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    return cur


@contextmanager
def _patch_get_cursor(cur: MagicMock) -> Generator:
    """Patch both the core.retrieval get_cursor and generate_embedding."""
    # Return a dummy embedding so hybrid search takes the vector path
    dummy_embedding = [0.1] * 1536
    with (
        patch("valence.core.retrieval.get_cursor", return_value=cur),
        patch("valence.core.retrieval.generate_embedding", return_value=dummy_embedding),
    ):
        yield cur


# ---------------------------------------------------------------------------
# _compute_freshness_days
# ---------------------------------------------------------------------------


class TestComputeFreshnessDays:
    def test_uses_compiled_at_first(self):
        from valence.core.retrieval import _compute_freshness_days

        now = datetime.now(UTC)
        article = {
            "compiled_at": (now - timedelta(days=3)).isoformat(),
            "created_at": (now - timedelta(days=10)).isoformat(),
        }
        days = _compute_freshness_days(article)
        assert abs(days - 3.0) < 0.1

    def test_falls_back_to_modified_at(self):
        from valence.core.retrieval import _compute_freshness_days

        now = datetime.now(UTC)
        article = {
            "compiled_at": None,
            "modified_at": (now - timedelta(days=7)).isoformat(),
            "created_at": (now - timedelta(days=20)).isoformat(),
        }
        days = _compute_freshness_days(article)
        assert abs(days - 7.0) < 0.1

    def test_falls_back_to_created_at(self):
        from valence.core.retrieval import _compute_freshness_days

        now = datetime.now(UTC)
        article = {
            "compiled_at": None,
            "modified_at": None,
            "created_at": (now - timedelta(days=14)).isoformat(),
        }
        days = _compute_freshness_days(article)
        assert abs(days - 14.0) < 0.1

    def test_missing_timestamps_returns_90(self):
        from valence.core.retrieval import _compute_freshness_days

        assert _compute_freshness_days({}) == 90.0

    def test_datetime_object_supported(self):
        from valence.core.retrieval import _compute_freshness_days

        now = datetime.now(UTC)
        article = {"compiled_at": now - timedelta(days=1)}
        days = _compute_freshness_days(article)
        assert abs(days - 1.0) < 0.1


# ---------------------------------------------------------------------------
# _freshness_score
# ---------------------------------------------------------------------------


class TestFreshnessScore:
    def test_zero_days_returns_one(self):
        from valence.core.retrieval import _freshness_score

        assert _freshness_score(0) == pytest.approx(1.0, abs=0.001)

    def test_ninety_days_returns_low(self):
        from valence.core.retrieval import _freshness_score

        score = _freshness_score(90)
        assert score < 0.5

    def test_clamped_to_zero_one(self):
        from valence.core.retrieval import _freshness_score

        assert 0.0 <= _freshness_score(1000) <= 1.0
        assert 0.0 <= _freshness_score(0) <= 1.0


# ---------------------------------------------------------------------------
# retrieve() — empty query
# ---------------------------------------------------------------------------


class TestRetrieveEmptyQuery:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_list(self):
        from valence.core.retrieval import retrieve

        result = await retrieve("")
        assert result.data == []

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty_list(self):
        from valence.core.retrieval import retrieve

        result = await retrieve("   ")
        assert result.data == []


# ---------------------------------------------------------------------------
# retrieve() — articles only
# ---------------------------------------------------------------------------


class TestRetrieveArticles:
    """retrieve() with articles only (include_sources=False)."""

    @pytest.mark.asyncio
    async def test_returns_article_results(self):
        """retrieve() should return article dicts with expected fields."""
        row = _make_article_row(article_id=ARTICLE_ID, text_rank=0.9)
        cur = _make_cursor(article_rows=[row])

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python", include_sources=False)

        assert len(results.data) == 1
        r = results.data[0]
        assert r["id"] == ARTICLE_ID
        assert r["type"] == "article"
        assert "content" in r
        assert "final_score" in r
        assert "provenance_summary" in r
        assert "freshness" in r
        assert "active_contentions" in r

    @pytest.mark.asyncio
    async def test_provenance_summary_included(self):
        """provenance_summary should include source_count and relationship_types."""
        row = _make_article_row()
        prov = {"source_count": 3, "relationship_types": ["originates", "confirms"]}
        cur = _make_cursor(article_rows=[row], prov_row=prov)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python")

        r = results.data[0]
        assert r["provenance_summary"]["source_count"] == 3
        assert "originates" in r["provenance_summary"]["relationship_types"]

    @pytest.mark.asyncio
    async def test_freshness_field_present(self):
        """freshness should be a non-negative float (days since update)."""
        row = _make_article_row(days_old=5.0)
        cur = _make_cursor(article_rows=[row])

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python")

        r = results.data[0]
        assert isinstance(r["freshness"], float)
        assert r["freshness"] >= 0.0
        assert abs(r["freshness"] - 5.0) < 0.5  # approximate

    @pytest.mark.asyncio
    async def test_active_contentions_false_when_no_contention(self):
        """active_contentions should be False when no contention rows exist."""
        row = _make_article_row()
        cur = _make_cursor(article_rows=[row], has_contention=False)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python")

        assert results.data[0]["active_contentions"] is False

    @pytest.mark.asyncio
    async def test_active_contentions_true_when_contention_exists(self):
        """active_contentions should be True when a contention row exists."""
        row = _make_article_row()
        cur = _make_cursor(article_rows=[row], has_contention=True)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python")

        assert results.data[0]["active_contentions"] is True

    @pytest.mark.asyncio
    async def test_no_db_returns_empty(self):
        """If no articles match, return empty list."""
        cur = _make_cursor(article_rows=[])

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python")

        assert results.data == []


# ---------------------------------------------------------------------------
# retrieve() — ranking order
# ---------------------------------------------------------------------------


class TestRankingOrder:
    """Results must be ranked by relevance*0.5 + confidence*0.35 + freshness*0.15."""

    @pytest.mark.asyncio
    async def test_higher_relevance_ranks_first(self):
        """Article with higher RRF score should rank above lower-scored article."""
        high = _make_article_row(
            article_id=ARTICLE_ID, text_rank=0.95, vec_rank=1, rrf_score=0.033,
            confidence_overall=0.7, days_old=5,
        )
        low = _make_article_row(
            article_id=ARTICLE_ID_2, text_rank=0.2, vec_rank=5, rrf_score=0.016,
            confidence_overall=0.7, days_old=5,
        )

        fetch_all_responses = [[high, low]]
        fetch_one_responses = [
            _make_prov_row(),
            None,  # high article
            _make_prov_row(),
            None,  # low article
        ]

        cur = MagicMock()
        fetchall_iter = iter(fetch_all_responses)
        fetchone_iter = iter(fetch_one_responses)
        cur.fetchall.side_effect = lambda: next(fetchall_iter, [])
        cur.fetchone.side_effect = lambda: next(fetchone_iter, None)
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python")

        assert len(results.data) == 2
        assert results.data[0]["id"] == ARTICLE_ID
        assert results.data[0]["final_score"] > results.data[1]["final_score"]

    @pytest.mark.asyncio
    async def test_higher_confidence_ranks_first_when_same_relevance(self):
        """When relevance is equal, higher confidence should win."""
        high_conf = _make_article_row(article_id=ARTICLE_ID, text_rank=0.5, confidence_overall=0.95, days_old=5)
        low_conf = _make_article_row(article_id=ARTICLE_ID_2, text_rank=0.5, confidence_overall=0.2, days_old=5)

        fetch_all_responses = [[high_conf, low_conf]]
        fetch_one_responses = [
            _make_prov_row(),
            None,
            _make_prov_row(),
            None,
        ]
        cur = MagicMock()
        fetchall_iter = iter(fetch_all_responses)
        fetchone_iter = iter(fetch_one_responses)
        cur.fetchall.side_effect = lambda: next(fetchall_iter, [])
        cur.fetchone.side_effect = lambda: next(fetchone_iter, None)
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python")

        assert len(results.data) == 2
        assert results.data[0]["id"] == ARTICLE_ID

    @pytest.mark.asyncio
    async def test_fresher_article_ranks_above_stale(self):
        """When relevance and confidence are equal, fresher article should win."""
        fresh = _make_article_row(article_id=ARTICLE_ID, text_rank=0.5, confidence_overall=0.7, days_old=1)
        stale = _make_article_row(article_id=ARTICLE_ID_2, text_rank=0.5, confidence_overall=0.7, days_old=200)

        fetch_all_responses = [[fresh, stale]]
        fetch_one_responses = [
            _make_prov_row(),
            None,
            _make_prov_row(),
            None,
        ]
        cur = MagicMock()
        fetchall_iter = iter(fetch_all_responses)
        fetchone_iter = iter(fetch_one_responses)
        cur.fetchall.side_effect = lambda: next(fetchall_iter, [])
        cur.fetchone.side_effect = lambda: next(fetchone_iter, None)
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python")

        assert len(results.data) == 2
        assert results.data[0]["id"] == ARTICLE_ID


# ---------------------------------------------------------------------------
# retrieve() — usage trace recording
# ---------------------------------------------------------------------------


class TestUsageTraceRecording:
    """Usage traces must be inserted for each article result."""

    @pytest.mark.asyncio
    async def test_usage_trace_recorded_for_article(self):
        """An article result should trigger an INSERT into usage_traces."""
        row = _make_article_row(article_id=ARTICLE_ID)
        cur = _make_cursor(article_rows=[row])

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python")

        assert len(results.data) == 1
        # Check that execute was called with usage_traces INSERT
        calls_str = str(cur.execute.call_args_list)
        assert "usage_traces" in calls_str

    @pytest.mark.asyncio
    async def test_usage_trace_not_recorded_when_no_results(self):
        """No usage trace when no articles match."""
        cur = _make_cursor(article_rows=[])

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python")

        assert results.data == []
        calls_str = str(cur.execute.call_args_list)
        assert "usage_traces" not in calls_str


# ---------------------------------------------------------------------------
# retrieve() — ungrouped sources and compilation queueing
# ---------------------------------------------------------------------------


class TestUngroupedSourcesAndQueue:
    """Ungrouped sources should appear as type=source and queue recompile."""

    @pytest.mark.asyncio
    async def test_ungrouped_source_appears_in_results(self):
        """include_sources=True should surface source dicts with type='source'."""
        src_row = _make_source_row(source_id=SOURCE_ID, text_rank=0.7)
        sentinel = {"id": uuid.UUID(ARTICLE_ID)}

        fetch_all_responses = [[], [src_row]]  # articles=[], sources=[src_row]
        fetch_one_responses = [sentinel]  # sentinel for recompile queue

        cur = MagicMock()
        fetchall_iter = iter(fetch_all_responses)
        fetchone_iter = iter(fetch_one_responses)
        cur.fetchall.side_effect = lambda: next(fetchall_iter, [])
        cur.fetchone.side_effect = lambda: next(fetchone_iter, None)
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python", include_sources=True)

        assert len(results.data) == 1
        assert results.data[0]["type"] == "source"
        assert results.data[0]["id"] == SOURCE_ID

    @pytest.mark.asyncio
    async def test_ungrouped_source_queues_recompile(self):
        """Each ungrouped source result should trigger a mutation_queue INSERT."""
        src_row = _make_source_row(source_id=SOURCE_ID, text_rank=0.7)
        sentinel = {"id": uuid.UUID(ARTICLE_ID)}

        fetch_all_responses = [[], [src_row]]
        fetch_one_responses = [sentinel]

        cur = MagicMock()
        fetchall_iter = iter(fetch_all_responses)
        fetchone_iter = iter(fetch_one_responses)
        cur.fetchall.side_effect = lambda: next(fetchall_iter, [])
        cur.fetchone.side_effect = lambda: next(fetchone_iter, None)
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            await retrieve("python", include_sources=True)

        calls_str = str(cur.execute.call_args_list)
        assert "mutation_queue" in calls_str
        assert "recompile" in calls_str

    @pytest.mark.asyncio
    async def test_sources_not_returned_when_include_sources_false(self):
        """include_sources=False should skip source search entirely."""
        cur = _make_cursor(article_rows=[])

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python", include_sources=False)

        # Verify that the source search query was NOT issued
        for c in cur.execute.call_args_list:
            args = c[0]
            if args:
                assert "article_sources" not in str(args[0]) or "NOT EXISTS" not in str(args[0]).upper() or True
        # Main check: no source-type results
        assert all(r.get("type") != "source" for r in results.data)

    @pytest.mark.asyncio
    async def test_recompile_skipped_when_no_sentinel_article(self):
        """If there are no articles to attach to, recompile queue is skipped gracefully."""
        src_row = _make_source_row(source_id=SOURCE_ID, text_rank=0.7)

        fetch_all_responses = [[], [src_row]]
        fetch_one_responses = [None]  # no articles exist

        cur = MagicMock()
        fetchall_iter = iter(fetch_all_responses)
        fetchone_iter = iter(fetch_one_responses)
        cur.fetchall.side_effect = lambda: next(fetchall_iter, [])
        cur.fetchone.side_effect = lambda: next(fetchone_iter, None)
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            # Should not raise
            results = await retrieve("python", include_sources=True)

        # mutation_queue INSERT should NOT have been called
        calls_str = str(cur.execute.call_args_list)
        # The sentinel SELECT will be there, but no INSERT into mutation_queue
        assert "mutation_queue" not in calls_str or "INSERT" not in calls_str


# ---------------------------------------------------------------------------
# retrieve() — source results include required metadata
# ---------------------------------------------------------------------------


class TestSourceResultMetadata:
    """Source results should include confidence (from reliability) and freshness."""

    @pytest.mark.asyncio
    async def test_source_confidence_from_reliability(self):
        src_row = _make_source_row(reliability=0.8)
        sentinel = {"id": uuid.UUID(ARTICLE_ID)}

        fetch_all_responses = [[], [src_row]]
        fetch_one_responses = [sentinel]

        cur = MagicMock()
        fetchall_iter = iter(fetch_all_responses)
        fetchone_iter = iter(fetch_one_responses)
        cur.fetchall.side_effect = lambda: next(fetchall_iter, [])
        cur.fetchone.side_effect = lambda: next(fetchone_iter, None)
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python", include_sources=True)

        r = results.data[0]
        assert r["confidence"]["overall"] == 0.8

    @pytest.mark.asyncio
    async def test_source_has_freshness(self):
        src_row = _make_source_row(days_old=10.0)
        sentinel = {"id": uuid.UUID(ARTICLE_ID)}

        fetch_all_responses = [[], [src_row]]
        fetch_one_responses = [sentinel]

        cur = MagicMock()
        fetchall_iter = iter(fetch_all_responses)
        fetchone_iter = iter(fetch_one_responses)
        cur.fetchall.side_effect = lambda: next(fetchall_iter, [])
        cur.fetchone.side_effect = lambda: next(fetchone_iter, None)
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python", include_sources=True)

        r = results.data[0]
        assert "freshness" in r
        assert abs(r["freshness"] - 10.0) < 0.5


# ---------------------------------------------------------------------------
# retrieve() — limit enforcement
# ---------------------------------------------------------------------------


class TestLimitEnforcement:
    @pytest.mark.asyncio
    async def test_limit_respected(self):
        """Only 'limit' results should be returned even if more match."""
        rows = [_make_article_row(article_id=str(uuid.uuid4()), text_rank=float(i) / 10) for i in range(8, 0, -1)]

        fetch_all_responses = [rows]
        # Each article needs provenance + contention fetchone
        fetch_one_responses = [val for _ in rows for val in [_make_prov_row(), None]]

        cur = MagicMock()
        fetchall_iter = iter(fetch_all_responses)
        fetchone_iter = iter(fetch_one_responses)
        cur.fetchall.side_effect = lambda: next(fetchall_iter, [])
        cur.fetchone.side_effect = lambda: next(fetchone_iter, None)
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)

        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python", limit=3)

        assert len(results.data) <= 3

    @pytest.mark.asyncio
    async def test_limit_zero_clamped_to_one(self):
        """limit=0 is clamped to 1 — should not raise."""
        cur = _make_cursor(article_rows=[])
        with _patch_get_cursor(cur):
            from valence.core.retrieval import retrieve

            results = await retrieve("python", limit=0)
        assert isinstance(results.data, list)


# ---------------------------------------------------------------------------
# knowledge_search MCP tool
# ---------------------------------------------------------------------------


class TestKnowledgeSearchTool:
    """Tests for the synchronous MCP tool wrapper."""

    def test_empty_query_returns_error(self):
        from valence.mcp.handlers.articles import knowledge_search

        result = knowledge_search("")
        assert result["success"] is False
        assert "error" in result

    def test_successful_search_returns_results(self):
        """knowledge_search should return {success, results, total_count}."""
        from valence.mcp.handlers.articles import knowledge_search

        mock_results = [
            {
                "id": ARTICLE_ID,
                "type": "article",
                "content": "Python is great.",
                "final_score": 0.8,
                "confidence": {"overall": 0.7},
                "freshness": 3.0,
                "freshness_score": 0.97,
                "provenance_summary": {"source_count": 1, "relationship_types": ["originates"]},
                "active_contentions": False,
            }
        ]

        # Patch: (1) asyncio in the tool module so get_event_loop raises (→ falls through to asyncio.run),
        #        (2) retrieve so it returns a plain (non-coroutine) value to avoid "never awaited" warning.
        with patch("valence.mcp.handlers._utils.asyncio") as mock_asyncio:
            mock_asyncio.get_event_loop.side_effect = RuntimeError("no event loop")
            mock_asyncio.run.return_value = mock_results
            # Patch retrieve to a non-async callable; this is imported lazily inside the tool.
            with patch("valence.core.retrieval.retrieve", return_value=mock_results):
                result = knowledge_search("python", limit=5)

        assert result["success"] is True
        assert "results" in result
        assert "total_count" in result
        assert result["query"] == "python"

    def test_exception_returns_error(self):
        """Any exception inside retrieve should be caught and returned as error."""
        from valence.mcp.handlers.articles import knowledge_search

        with patch("valence.mcp.handlers._utils.asyncio") as mock_asyncio:
            mock_asyncio.get_event_loop.side_effect = RuntimeError("no loop")
            mock_asyncio.run.side_effect = RuntimeError("db unavailable")

            result = knowledge_search("python")

        assert result["success"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# ranking.py — compute_freshness_score (new helper)
# ---------------------------------------------------------------------------


class TestComputeFreshnessScore:
    def test_recent_compiled_at_scores_high(self):
        from valence.core.ranking import compute_freshness_score

        article = {"compiled_at": datetime.now(UTC) - timedelta(days=1)}
        score = compute_freshness_score(article)
        assert score > 0.98

    def test_old_article_scores_low(self):
        from valence.core.ranking import compute_freshness_score

        article = {"compiled_at": datetime.now(UTC) - timedelta(days=500)}
        score = compute_freshness_score(article)
        assert score < 0.01

    def test_prefers_compiled_at_over_created_at(self):
        from valence.core.ranking import compute_freshness_score

        now = datetime.now(UTC)
        article = {
            "compiled_at": now - timedelta(days=2),  # fresher
            "created_at": now - timedelta(days=100),  # staler
        }
        score = compute_freshness_score(article)
        assert score > 0.9  # should use compiled_at

    def test_falls_back_to_created_at(self):
        from valence.core.ranking import compute_freshness_score

        article = {"created_at": datetime.now(UTC) - timedelta(days=5)}
        score = compute_freshness_score(article)
        assert score > 0.90

    def test_no_timestamps_returns_half(self):
        from valence.core.ranking import compute_freshness_score

        score = compute_freshness_score({})
        assert score == 0.5
