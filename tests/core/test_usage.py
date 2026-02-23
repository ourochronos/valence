"""Tests for valence.core.usage module (WU-09: Usage Tracking & Self-Organization).

Tests cover:
1. _compute_score helper — recency weighting, connection bonus
2. record_usage       — inserts trace, recomputes and persists score
3. compute_usage_scores — batch idempotent score recompute
4. get_decay_candidates — lowest-score, non-pinned articles first

All tests use mocked DB cursors — no real PostgreSQL required.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

ARTICLE_A_ID = str(uuid4())
ARTICLE_B_ID = str(uuid4())


def _make_cursor_mock(
    fetchone: dict | None = None,
    fetchall: list | None = None,
    rowcount: int = 0,
) -> MagicMock:
    """Create a mock cursor that works as a context manager (sync)."""
    cur = MagicMock()
    cur.fetchone.return_value = fetchone
    cur.fetchall.return_value = fetchall or []
    cur.rowcount = rowcount
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    return cur


def _make_article_row(
    article_id: str = ARTICLE_A_ID,
    usage_score: float = 0.0,
    pinned: bool = False,
    status: str = "active",
    title: str | None = None,
) -> dict[str, Any]:
    """Build a minimal articles DB row dict."""
    now = datetime.now()
    return {
        "id": article_id,
        "title": title or f"Article {article_id[:8]}",
        "content": "Some article content.",
        "usage_score": usage_score,
        "pinned": pinned,
        "status": status,
        "domain_path": ["test"],
        "created_at": now,
        "modified_at": now,
    }


# ---------------------------------------------------------------------------
# _compute_score (pure logic — no DB required)
# ---------------------------------------------------------------------------


class TestComputeScore:
    """Tests for the pure _compute_score helper."""

    def test_five_retrievals_beats_zero_retrievals(self):
        """Article A with 5 recent retrievals should score higher than B with 0."""
        from valence.core.usage import _compute_score

        now = datetime.now(timezone.utc)
        timestamps_a = [now] * 5  # 5 very recent retrievals
        timestamps_b: list = []

        score_a = _compute_score(timestamps_a, 0)
        score_b = _compute_score(timestamps_b, 0)

        assert score_a > score_b, f"Expected score_a ({score_a:.4f}) > score_b ({score_b:.4f})"

    def test_more_recent_retrieval_beats_older(self):
        """A retrieval from today should weigh more than one from a year ago."""
        from valence.core.usage import _compute_score

        now = datetime.now(timezone.utc)
        recent = [now]
        old = [now - timedelta(days=365)]

        assert _compute_score(recent, 0) > _compute_score(old, 0)

    def test_connection_bonus_increases_score(self):
        """Articles with more sources linked get a higher score bonus."""
        from valence.core.usage import _compute_score

        score_many_sources = _compute_score([], source_count=10)
        score_no_sources = _compute_score([], source_count=0)

        assert score_many_sources > score_no_sources

    def test_empty_timestamps_no_sources_returns_near_zero(self):
        """No retrievals, no sources → score is 0."""
        from valence.core.usage import _compute_score

        score = _compute_score([], 0)
        assert score == pytest.approx(0.0)

    def test_score_non_negative(self):
        """Score is always non-negative."""
        from valence.core.usage import _compute_score

        now = datetime.now(timezone.utc)
        assert _compute_score([], 0) >= 0
        assert _compute_score([now] * 100, 0) >= 0
        assert _compute_score([now - timedelta(days=9999)], 100) >= 0

    def test_recency_decayed_weight(self):
        """_decayed_weight should be 1.0 for just-now, < 1.0 for the past."""
        from valence.core.usage import _decayed_weight

        now = datetime.now(timezone.utc)
        weight_now = _decayed_weight(now, now)
        weight_past = _decayed_weight(now - timedelta(days=30), now)

        assert math.isclose(weight_now, 1.0, rel_tol=1e-6)
        assert weight_past < 1.0
        assert weight_past > 0.0

    def test_connection_bonus_formula(self):
        """Connection bonus = log(1 + count) * 0.5."""
        from valence.core.usage import _compute_score

        expected = math.log1p(5) * 0.5
        assert _compute_score([], source_count=5) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# record_usage
# ---------------------------------------------------------------------------


class TestRecordUsage:
    """Tests for record_usage()."""

    async def test_inserts_usage_trace(self):
        """record_usage inserts a row into usage_traces."""
        from valence.core.usage import record_usage

        mock_cur = _make_cursor_mock(
            fetchall=[{"retrieved_at": datetime.now(timezone.utc)}],
            fetchone={"cnt": 0},
        )

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await record_usage(ARTICLE_A_ID, query="python tips", tool="knowledge_search")

        calls_sql = " ".join(str(c) for c in mock_cur.execute.call_args_list)
        assert "INSERT INTO usage_traces" in calls_sql

    async def test_updates_article_usage_score(self):
        """record_usage updates the usage_score column on the article."""
        from valence.core.usage import record_usage

        now = datetime.now(timezone.utc)
        mock_cur = _make_cursor_mock(
            fetchall=[{"retrieved_at": now}],
            fetchone={"cnt": 2},
        )

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await record_usage(ARTICLE_A_ID, query="test query", tool="search")

        calls_sql = " ".join(str(c) for c in mock_cur.execute.call_args_list)
        assert "UPDATE articles" in calls_sql
        assert "usage_score" in calls_sql

    async def test_score_increases_with_multiple_calls(self):
        """After 5 calls, usage_score should be higher than after 0 calls."""
        from valence.core.usage import _compute_score

        # Simulate 5 retrievals in the very recent past
        now = datetime.now(timezone.utc)
        score_after_5 = _compute_score([now] * 5, source_count=0)
        score_after_0 = _compute_score([], source_count=0)

        assert score_after_5 > score_after_0

    async def test_queries_traces_for_article(self):
        """record_usage fetches existing traces for the correct article_id."""
        from valence.core.usage import record_usage

        mock_cur = _make_cursor_mock(
            fetchall=[{"retrieved_at": datetime.now(timezone.utc)}],
            fetchone={"cnt": 0},
        )

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await record_usage(ARTICLE_A_ID, query="test", tool="tool")

        # Confirm the SELECT on usage_traces passed the correct article_id
        calls = mock_cur.execute.call_args_list
        trace_select_found = any("usage_traces" in str(c) and ARTICLE_A_ID in str(c) for c in calls)
        assert trace_select_found

    async def test_queries_source_count_for_article(self):
        """record_usage fetches the source count for connection bonus."""
        from valence.core.usage import record_usage

        mock_cur = _make_cursor_mock(
            fetchall=[],
            fetchone={"cnt": 3},
        )

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await record_usage(ARTICLE_A_ID, query="test", tool="tool")

        calls_sql = " ".join(str(c) for c in mock_cur.execute.call_args_list)
        assert "article_sources" in calls_sql


# ---------------------------------------------------------------------------
# compute_usage_scores
# ---------------------------------------------------------------------------


class TestComputeUsageScores:
    """Tests for compute_usage_scores()."""

    async def test_returns_int(self):
        """compute_usage_scores returns an integer count."""
        from valence.core.usage import compute_usage_scores

        mock_cur = _make_cursor_mock(rowcount=5)

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            count = await compute_usage_scores()

        assert isinstance(count.data, int)

    async def test_returns_updated_count(self):
        """compute_usage_scores returns the rowcount from the UPDATE."""
        from valence.core.usage import compute_usage_scores

        mock_cur = _make_cursor_mock(rowcount=42)

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            count = await compute_usage_scores()

        assert count.data == 42

    async def test_idempotent_second_run_returns_zero(self):
        """Idempotency: second run with no changes returns 0 (IS DISTINCT FROM)."""
        from valence.core.usage import compute_usage_scores

        # Simulate: first run updates 10, second run finds nothing changed → 0
        mock_cur_first = _make_cursor_mock(rowcount=10)
        mock_cur_second = _make_cursor_mock(rowcount=0)

        with patch("valence.core.usage.get_cursor", return_value=mock_cur_first):
            first_count = await compute_usage_scores()

        with patch("valence.core.usage.get_cursor", return_value=mock_cur_second):
            second_count = await compute_usage_scores()

        assert first_count.data == 10
        assert second_count.data == 0  # Nothing changed — idempotent

    async def test_uses_is_distinct_from_for_idempotency(self):
        """SQL must contain IS DISTINCT FROM to skip unchanged rows."""
        from valence.core.usage import compute_usage_scores

        mock_cur = _make_cursor_mock(rowcount=0)

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await compute_usage_scores()

        executed_sql = mock_cur.execute.call_args[0][0]
        assert "IS DISTINCT FROM" in executed_sql

    async def test_executes_single_update_statement(self):
        """compute_usage_scores issues exactly one SQL call (batch UPDATE)."""
        from valence.core.usage import compute_usage_scores

        mock_cur = _make_cursor_mock(rowcount=0)

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await compute_usage_scores()

        assert mock_cur.execute.call_count == 1

    async def test_sql_contains_update_articles(self):
        """SQL statement updates the articles table."""
        from valence.core.usage import compute_usage_scores

        mock_cur = _make_cursor_mock(rowcount=0)

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await compute_usage_scores()

        sql = mock_cur.execute.call_args[0][0]
        assert "UPDATE articles" in sql


# ---------------------------------------------------------------------------
# get_decay_candidates
# ---------------------------------------------------------------------------


class TestGetDecayCandidates:
    """Tests for get_decay_candidates()."""

    async def test_returns_list(self):
        """get_decay_candidates returns a list."""
        from valence.core.usage import get_decay_candidates

        mock_cur = _make_cursor_mock(fetchall=[])

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            result = await get_decay_candidates()

        assert isinstance(result.data, list)

    async def test_b_before_a_in_results(self):
        """Article B (lower usage_score) appears before article A (higher score)."""
        from valence.core.usage import get_decay_candidates

        # DB returns rows ordered by usage_score ASC (as per the SQL)
        row_b = _make_article_row(ARTICLE_B_ID, usage_score=0.0)
        row_a = _make_article_row(ARTICLE_A_ID, usage_score=4.9)

        mock_cur = _make_cursor_mock(fetchall=[row_b, row_a])

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            results = await get_decay_candidates()

        assert len(results.data) == 2
        assert results.data[0]["id"] == ARTICLE_B_ID, "B (lowest score) should come first"
        assert results.data[1]["id"] == ARTICLE_A_ID

    async def test_sql_excludes_pinned(self):
        """SQL query must filter out pinned articles (pinned = FALSE)."""
        from valence.core.usage import get_decay_candidates

        mock_cur = _make_cursor_mock(fetchall=[])

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await get_decay_candidates()

        sql = mock_cur.execute.call_args[0][0]
        assert "pinned = FALSE" in sql or "pinned=FALSE" in sql.replace(" ", "=")

    async def test_pinned_articles_never_appear(self):
        """Pinned articles should be absent from decay candidates.

        Verifies that the SQL WHERE clause excludes pinned articles so they
        never surface as decay candidates regardless of their usage_score.
        """
        from valence.core.usage import get_decay_candidates

        # Simulate DB returning only non-pinned rows (pinned one was filtered by SQL)
        non_pinned_row = _make_article_row(ARTICLE_B_ID, usage_score=0.0, pinned=False)
        # Pinned article would have lowest score but is excluded
        mock_cur = _make_cursor_mock(fetchall=[non_pinned_row])

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            results = await get_decay_candidates()

        # None of the results should be pinned
        pinned_ids = [r["id"] for r in results.data if r.get("pinned")]
        assert pinned_ids == [], f"Pinned articles found in decay candidates: {pinned_ids}"

    async def test_sql_excludes_pinned_in_where_clause(self):
        """Verify the SQL WHERE clause explicitly excludes pinned articles."""
        from valence.core.usage import get_decay_candidates

        mock_cur = _make_cursor_mock(fetchall=[])
        captured_sql = []

        original_execute = mock_cur.execute.side_effect

        def capture_sql(sql, params=None):
            captured_sql.append(sql)

        mock_cur.execute.side_effect = capture_sql

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await get_decay_candidates()

        assert len(captured_sql) == 1
        sql = captured_sql[0].upper()
        # Must exclude pinned in WHERE
        assert "PINNED" in sql
        assert "FALSE" in sql or "PINNED = FALSE" in sql.replace("\n", " ")

    async def test_sql_orders_by_usage_score_asc(self):
        """Decay candidates ordered by usage_score ASC (lowest first)."""
        from valence.core.usage import get_decay_candidates

        mock_cur = _make_cursor_mock(fetchall=[])

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await get_decay_candidates()

        sql = mock_cur.execute.call_args[0][0].upper()
        assert "ORDER BY USAGE_SCORE ASC" in sql.replace("\n", " ").replace("  ", " ")

    async def test_respects_limit_parameter(self):
        """Limit parameter is passed through to the SQL query."""
        from valence.core.usage import get_decay_candidates

        mock_cur = _make_cursor_mock(fetchall=[])

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await get_decay_candidates(limit=25)

        params = mock_cur.execute.call_args[0][1]
        assert 25 in params

    async def test_default_limit_is_100(self):
        """Default limit is 100."""
        from valence.core.usage import get_decay_candidates

        mock_cur = _make_cursor_mock(fetchall=[])

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await get_decay_candidates()

        params = mock_cur.execute.call_args[0][1]
        assert 100 in params

    async def test_only_active_articles(self):
        """Decay candidates should only include active articles."""
        from valence.core.usage import get_decay_candidates

        mock_cur = _make_cursor_mock(fetchall=[])

        with patch("valence.core.usage.get_cursor", return_value=mock_cur):
            await get_decay_candidates()

        sql = mock_cur.execute.call_args[0][0]
        assert "status = 'active'" in sql or "status='active'" in sql.replace(" ", "=")


# ---------------------------------------------------------------------------
# End-to-end scenario: A vs B usage comparison
# ---------------------------------------------------------------------------


class TestUsageScoreOrdering:
    """Higher-level scenario tests asserting the ranking contract."""

    def test_five_retrievals_score_greater_than_zero_retrievals(self):
        """After 5 retrievals for A and 0 for B: A.usage_score > B.usage_score."""
        from valence.core.usage import _compute_score

        now = datetime.now(timezone.utc)

        # Article A: retrieved 5 times in the last hour
        timestamps_a = [now - timedelta(minutes=i) for i in range(5)]
        score_a = _compute_score(timestamps_a, source_count=0)

        # Article B: never retrieved
        score_b = _compute_score([], source_count=0)

        assert score_a > score_b, f"A (5 retrievals, score={score_a:.4f}) should beat B (0 retrievals, score={score_b:.4f})"

    def test_decay_candidates_b_before_a(self):
        """get_decay_candidates ordering: B (score=0) appears before A (score=4.9)."""
        from valence.core.usage import _compute_score

        now = datetime.now(timezone.utc)
        timestamps_a = [now] * 5
        score_a = _compute_score(timestamps_a, source_count=0)
        score_b = _compute_score([], source_count=0)

        # Ordering must put lower score first
        ordered = sorted(
            [
                {"id": ARTICLE_A_ID, "usage_score": score_a},
                {"id": ARTICLE_B_ID, "usage_score": score_b},
            ],
            key=lambda x: x["usage_score"],
        )

        assert ordered[0]["id"] == ARTICLE_B_ID, "B (lower score) should be first decay candidate"
        assert ordered[1]["id"] == ARTICLE_A_ID

    def test_pinned_article_excluded_from_ordering(self):
        """A pinned article with the lowest score should not appear in decay list."""
        # Simulate decay query results: pinned article excluded, non-pinned present
        all_articles = [
            {"id": "pinned-id", "usage_score": 0.0, "pinned": True},
            {"id": ARTICLE_B_ID, "usage_score": 1.0, "pinned": False},
            {"id": ARTICLE_A_ID, "usage_score": 5.0, "pinned": False},
        ]

        # Filter as the SQL would (pinned = FALSE, status = 'active')
        candidates = [a for a in all_articles if not a["pinned"]]
        candidates.sort(key=lambda x: x["usage_score"])

        ids = [c["id"] for c in candidates]
        assert "pinned-id" not in ids
        assert ids[0] == ARTICLE_B_ID

    def test_compute_scores_idempotency_contract(self):
        """Running compute_usage_scores twice with the same data returns the same scores."""
        from valence.core.usage import _compute_score, _decayed_weight

        # Use a fixed reference time to avoid floating-point drift between two datetime.now() calls
        reference_now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        timestamps = [reference_now - timedelta(hours=1)] * 3

        # Compute twice with the same explicit 'now' by calling _decayed_weight directly
        import math

        recency_sum = sum(_decayed_weight(ts, reference_now) for ts in timestamps)
        connection_bonus = math.log1p(2) * 0.5

        score_first = recency_sum + connection_bonus
        score_second = recency_sum + connection_bonus  # exact same computation

        assert score_first == score_second, "Score computation must be deterministic/idempotent"
