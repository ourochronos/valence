"""Tests for supersession chain flattening (issue #58).

Covers:
- resolve_supersession_head: simple chain, multi-hop, already at head
- circular chain protection (max_depth guard)
- resolve_supersession_head_sync: synchronous variant
- retrieval de-ranking of superseded sources (is_superseded from SQL)
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.core.sources import resolve_supersession_head, resolve_supersession_head_sync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain_cursor(chain_map):
    """Build a mock cursor whose fetchone resolves forward-supersession lookups.

    chain_map: {source_id: superseding_id_or_None}
    Answers: SELECT id FROM sources WHERE supersedes_id = X
    """
    cur = MagicMock()
    call_args_store = {}

    def _execute(sql, params=None):
        call_args_store["last_params"] = params

    def _fetchone():
        params = call_args_store.get("last_params")
        if not params:
            return None
        queried_id = str(params[0])
        result = chain_map.get(queried_id)
        if result is None:
            return None
        return {"id": result}

    cur.execute.side_effect = _execute
    cur.fetchone.side_effect = _fetchone
    return cur


@contextmanager
def _mock_cursor(cur):
    @contextmanager
    def _get_cursor(**kwargs):
        yield cur

    with patch("valence.core.sources.get_cursor", _get_cursor):
        yield


# ---------------------------------------------------------------------------
# resolve_supersession_head (async)
# ---------------------------------------------------------------------------


class TestResolveSupersessionHead:
    @pytest.mark.asyncio
    async def test_no_supersession_returns_self(self):
        sid = str(uuid4())
        cur = _make_chain_cursor({sid: None})
        with _mock_cursor(cur):
            resp = await resolve_supersession_head(sid)
        assert resp.success
        assert resp.data["head_id"] == sid
        assert resp.data["chain"] == [sid]
        assert resp.data["depth"] == 0

    @pytest.mark.asyncio
    async def test_simple_two_hop_chain(self):
        a = str(uuid4())
        b = str(uuid4())
        cur = _make_chain_cursor({b: a, a: None})
        with _mock_cursor(cur):
            resp = await resolve_supersession_head(b)
        assert resp.success
        assert resp.data["head_id"] == a
        assert resp.data["chain"] == [b, a]
        assert resp.data["depth"] == 1

    @pytest.mark.asyncio
    async def test_three_hop_chain(self):
        a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
        cur = _make_chain_cursor({c: b, b: a, a: None})
        with _mock_cursor(cur):
            resp = await resolve_supersession_head(c)
        assert resp.success
        assert resp.data["head_id"] == a
        assert resp.data["chain"] == [c, b, a]
        assert resp.data["depth"] == 2

    @pytest.mark.asyncio
    async def test_already_at_head(self):
        a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
        cur = _make_chain_cursor({c: b, b: a, a: None})
        with _mock_cursor(cur):
            resp = await resolve_supersession_head(a)
        assert resp.success
        assert resp.data["head_id"] == a
        assert resp.data["depth"] == 0

    @pytest.mark.asyncio
    async def test_circular_chain_protection(self):
        a = str(uuid4())
        b = str(uuid4())
        cur = _make_chain_cursor({a: b, b: a})
        with _mock_cursor(cur):
            resp = await resolve_supersession_head(a, max_depth=10)
        assert resp.success
        assert resp.data["depth"] <= 10


# ---------------------------------------------------------------------------
# resolve_supersession_head_sync
# ---------------------------------------------------------------------------


class TestResolveSupersessionHeadSync:
    def test_simple_chain(self):
        a, b = str(uuid4()), str(uuid4())
        cur = _make_chain_cursor({b: a, a: None})
        head = resolve_supersession_head_sync(b, cur)
        assert head == a

    def test_no_supersessor(self):
        sid = str(uuid4())
        cur = _make_chain_cursor({sid: None})
        head = resolve_supersession_head_sync(sid, cur)
        assert head == sid

    def test_circular_guard(self):
        a, b = str(uuid4()), str(uuid4())
        cur = _make_chain_cursor({a: b, b: a})
        head = resolve_supersession_head_sync(a, cur, max_depth=5)
        assert isinstance(head, str)


# ---------------------------------------------------------------------------
# Retrieval de-ranking of superseded sources
# ---------------------------------------------------------------------------


class TestRetrievalDeRankingSuperseded:
    """Verify that superseded sources get lower similarity/confidence in search results.

    The is_superseded flag comes from the SQL EXISTS subquery (embedded in
    the mock row), so no separate fetchone call is needed per source.
    """

    def _make_row(self, source_id, is_superseded=False):
        return {
            "id": source_id,
            "type": "document",
            "title": "Test",
            "url": None,
            "content": "Some content",
            "reliability": 0.8,
            "fingerprint": "abc",
            "supersedes_id": None,
            "is_superseded": is_superseded,
            "created_at": "2026-01-01T00:00:00",
            "vec_rank": 1,
            "vec_score": 0.9,
            "text_rank": 1,
            "text_score": 0.8,
            "rrf_score": 0.5,
        }

    def test_superseded_source_marked_and_penalized(self):
        from valence.core.retrieval import _search_ungrouped_sources_sync

        old_id = str(uuid4())
        new_id = str(uuid4())

        # chain_cur mock — returns new_id for old_id lookup
        chain_call_store = {}

        def _chain_execute(sql, params=None):
            chain_call_store["last"] = params

        def _chain_fetchone():
            params = chain_call_store.get("last")
            queried = str(params[0]) if params else ""
            return {"id": new_id} if queried == old_id else None

        chain_cur = MagicMock()
        chain_cur.execute.side_effect = _chain_execute
        chain_cur.fetchone.side_effect = _chain_fetchone

        main_cur = MagicMock()
        main_cur.fetchall.return_value = [self._make_row(old_id, is_superseded=True)]

        call_count = [0]

        @contextmanager
        def _get_cursor(**kwargs):
            call_count[0] += 1
            # First call: main search; second call: chain resolution
            yield chain_cur if call_count[0] > 1 else main_cur

        with patch("valence.core.retrieval.get_cursor", _get_cursor), patch("valence.core.retrieval._try_generate_embedding", return_value=None):
            results = _search_ungrouped_sources_sync("test query", limit=10)

        assert len(results) == 1
        assert results[0]["is_superseded"] is True
        assert results[0]["supersession_head_id"] == new_id
        assert results[0]["confidence"]["overall"] < 0.8

    def test_head_source_not_penalized(self):
        from valence.core.retrieval import _search_ungrouped_sources_sync

        head_id = str(uuid4())

        main_cur = MagicMock()
        main_cur.fetchall.return_value = [self._make_row(head_id, is_superseded=False)]

        @contextmanager
        def _get_cursor(**kwargs):
            yield main_cur

        with patch("valence.core.retrieval.get_cursor", _get_cursor), patch("valence.core.retrieval._try_generate_embedding", return_value=None):
            results = _search_ungrouped_sources_sync("test query", limit=10)

        assert len(results) == 1
        assert results[0]["is_superseded"] is False
        assert results[0]["supersession_head_id"] == head_id
        assert results[0]["confidence"]["overall"] == pytest.approx(0.8)
