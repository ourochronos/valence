"""Tests for core.sources module (WU-03, C1 Source Ingestion).

Updated for WU-14: all public functions return ValenceResponse.

Tests cover:
1. ingest_source  — happy path, reliability defaults, fingerprint, dedup rejection
2. get_source     — found, not found
3. search_sources — full-text search, empty query
4. list_sources   — all sources, filtered by type
5. Helper: _compute_fingerprint, _row_to_dict
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.core.sources import (
    RELIABILITY_DEFAULTS,
    VALID_SOURCE_TYPES,
    _compute_fingerprint,
    _row_to_dict,
    get_source,
    ingest_source,
    list_sources,
    search_sources,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cursor():
    """Mock psycopg2 cursor."""
    cur = MagicMock()
    cur.fetchone.return_value = None
    cur.fetchall.return_value = []
    return cur


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Patch ``valence.core.sources.get_cursor`` with a sync context manager."""

    @contextmanager
    def _mock(dict_cursor: bool = True) -> Generator:
        yield mock_cursor

    with patch("valence.core.sources.get_cursor", _mock):
        yield mock_cursor


def _make_source_row(
    content: str = "Test content",
    source_type: str = "document",
    title: str | None = "Test title",
    url: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Build a fake source DB row returned by RETURNING clauses."""
    fp = _compute_fingerprint(content)
    return {
        "id": uuid4(),
        "type": source_type,
        "title": title,
        "url": url,
        "content": content,
        "fingerprint": fp,
        "content_hash": fp,
        "reliability": RELIABILITY_DEFAULTS.get(source_type, 0.5),
        "session_id": None,
        "metadata": metadata or {},
        "created_at": datetime(2026, 2, 21, 12, 0, 0),
        "content_tsv": None,  # should be stripped by _row_to_dict
        "embedding": None,  # should be stripped by _row_to_dict
    }


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


class TestComputeFingerprint:
    def test_returns_sha256_hex(self):
        fp = _compute_fingerprint("hello")
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_deterministic(self):
        assert _compute_fingerprint("same") == _compute_fingerprint("same")

    def test_different_content_different_fingerprint(self):
        assert _compute_fingerprint("a") != _compute_fingerprint("b")


class TestRowToDict:
    def test_strips_tsv_and_embedding(self):
        row = {
            "id": uuid4(),
            "content": "hi",
            "content_tsv": "tsv_blob",
            "embedding": b"\x00\x01",
            "created_at": datetime(2026, 1, 1),
            "metadata": {},
        }
        result = _row_to_dict(row)
        assert "content_tsv" not in result
        assert "embedding" not in result

    def test_id_becomes_string(self):
        uid = uuid4()
        row = {"id": uid, "created_at": datetime(2026, 1, 1), "metadata": {}}
        result = _row_to_dict(row)
        assert result["id"] == str(uid)

    def test_created_at_becomes_isoformat(self):
        dt = datetime(2026, 2, 21, 12, 0, 0)
        row = {"id": uuid4(), "created_at": dt, "metadata": {}}
        result = _row_to_dict(row)
        assert result["created_at"] == dt.isoformat()

    def test_metadata_string_decoded(self):
        row = {
            "id": uuid4(),
            "created_at": datetime(2026, 1, 1),
            "metadata": '{"key": "val"}',
        }
        result = _row_to_dict(row)
        assert result["metadata"] == {"key": "val"}


# ---------------------------------------------------------------------------
# ingest_source
# ---------------------------------------------------------------------------


class TestIngestSource:
    async def test_happy_path_document(self, mock_get_cursor):
        row = _make_source_row("Python 3.12 adds...", "document", title="Changelog")
        mock_get_cursor.fetchone.side_effect = [None, row]  # no dupe, then inserted row

        result = await ingest_source("Python 3.12 adds...", "document", title="Changelog")

        assert result.success is True
        assert result.data["id"] == str(row["id"])
        assert result.data["reliability"] == 0.8
        assert result.data["fingerprint"] == _compute_fingerprint("Python 3.12 adds...")
        assert result.data["type"] == "document"
        assert "created_at" in result.data
        # TSV and embedding stripped
        assert "content_tsv" not in result.data
        assert "embedding" not in result.data

    async def test_reliability_defaults_by_type(self, mock_get_cursor):
        for source_type, expected_reliability in RELIABILITY_DEFAULTS.items():
            row = _make_source_row("content", source_type)
            mock_get_cursor.fetchone.side_effect = [None, row]
            result = await ingest_source("content", source_type)
            assert result.success is True
            assert result.data["reliability"] == expected_reliability, (
                f"Expected {expected_reliability} for {source_type}, got {result.data['reliability']}"
            )

    async def test_rejects_empty_content(self, mock_get_cursor):
        result = await ingest_source("", "document")
        assert result.success is False
        assert "content" in result.error.lower()

    async def test_rejects_whitespace_only_content(self, mock_get_cursor):
        result = await ingest_source("   ", "document")
        assert result.success is False
        assert result.error is not None

    async def test_rejects_invalid_source_type(self, mock_get_cursor):
        result = await ingest_source("some content", "unknown_type")
        assert result.success is False
        assert result.error is not None
        assert "unknown_type" in result.error or "source_type" in result.error.lower() or "invalid" in result.error.lower()

    async def test_rejects_duplicate_fingerprint(self, mock_get_cursor):
        # Simulate existing row returned by SELECT for dedup check
        existing = {"id": uuid4()}
        mock_get_cursor.fetchone.return_value = existing

        result = await ingest_source("duplicate content", "document")
        assert result.success is False
        assert str(existing["id"]) in result.error

    async def test_stores_metadata(self, mock_get_cursor):
        meta = {"source": "test-suite", "version": 1}
        row = _make_source_row("content", "web", metadata=meta)
        mock_get_cursor.fetchone.side_effect = [None, row]

        result = await ingest_source("content", "web", metadata=meta)
        assert result.success is True
        assert result.data["metadata"] == meta

    async def test_url_stored(self, mock_get_cursor):
        row = _make_source_row("content", "web", url="https://example.com")
        row["url"] = "https://example.com"
        mock_get_cursor.fetchone.side_effect = [None, row]

        result = await ingest_source("content", "web", url="https://example.com")
        assert result.success is True
        assert result.data["url"] == "https://example.com"

    async def test_fingerprint_is_sha256_of_content(self, mock_get_cursor):
        content = "unique content for fingerprint test"
        row = _make_source_row(content, "document")
        mock_get_cursor.fetchone.side_effect = [None, row]

        result = await ingest_source(content, "document")
        assert result.success is True
        assert result.data["fingerprint"] == _compute_fingerprint(content)

    async def test_insert_called_with_correct_params(self, mock_get_cursor):
        content = "The quick brown fox"
        row = _make_source_row(content, "code")
        mock_get_cursor.fetchone.side_effect = [None, row]

        await ingest_source(content, "code", title="snippet")

        calls = mock_get_cursor.execute.call_args_list
        # First call: SELECT for dedup; second: INSERT
        assert len(calls) == 2
        insert_sql = calls[1][0][0]
        assert "INSERT INTO sources" in insert_sql
        insert_params = calls[1][0][1]
        assert insert_params[0] == "code"  # type
        assert insert_params[1] == "snippet"  # title
        assert insert_params[3] == content  # content


# ---------------------------------------------------------------------------
# get_source
# ---------------------------------------------------------------------------


class TestGetSource:
    async def test_returns_source_when_found(self, mock_get_cursor):
        row = _make_source_row()
        mock_get_cursor.fetchone.return_value = row

        result = await get_source(str(row["id"]))

        assert result.success is True
        assert result.data["id"] == str(row["id"])
        assert result.data["content"] == row["content"]

    async def test_returns_error_when_missing(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = None

        result = await get_source("nonexistent-id")
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower() or "source" in result.error.lower()

    async def test_queries_by_id(self, mock_get_cursor):
        row = _make_source_row()
        mock_get_cursor.fetchone.return_value = row
        source_id = str(uuid4())

        await get_source(source_id)

        call_args = mock_get_cursor.execute.call_args
        assert "WHERE id = %s" in call_args[0][0]
        assert call_args[0][1] == (source_id,)


# ---------------------------------------------------------------------------
# search_sources
# ---------------------------------------------------------------------------


class TestSearchSources:
    async def test_returns_matches(self, mock_get_cursor):
        row = {**_make_source_row("Python type hints", "document"), "rank": 0.8}
        mock_get_cursor.fetchall.return_value = [row]

        result = await search_sources("python")

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["content"] == "Python type hints"
        assert "rank" in result.data[0]

    async def test_empty_query_returns_empty_list(self, mock_get_cursor):
        result = await search_sources("")
        assert result.success is True
        assert result.data == []
        mock_get_cursor.execute.assert_not_called()

    async def test_whitespace_query_returns_empty_list(self, mock_get_cursor):
        result = await search_sources("   ")
        assert result.success is True
        assert result.data == []
        mock_get_cursor.execute.assert_not_called()

    async def test_passes_limit_to_query(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []

        await search_sources("test", limit=5)

        call_args = mock_get_cursor.execute.call_args
        sql, params = call_args[0]
        assert "LIMIT %s" in sql
        assert 5 in params

    async def test_default_limit_is_20(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []

        await search_sources("test")

        call_args = mock_get_cursor.execute.call_args
        params = call_args[0][1]
        assert 20 in params

    async def test_uses_websearch_to_tsquery(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []

        await search_sources("python type hints")

        sql = mock_get_cursor.execute.call_args[0][0]
        assert "websearch_to_tsquery" in sql
        assert "content_tsv" in sql

    async def test_no_results_returns_empty_list(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []

        result = await search_sources("nonexistent topic xyz")
        assert result.success is True
        assert result.data == []


# ---------------------------------------------------------------------------
# list_sources
# ---------------------------------------------------------------------------


class TestListSources:
    async def test_returns_all_sources(self, mock_get_cursor):
        rows = [_make_source_row(f"content {i}", "document") for i in range(3)]
        mock_get_cursor.fetchall.return_value = rows

        result = await list_sources()
        assert result.success is True
        assert len(result.data) == 3

    async def test_filters_by_type(self, mock_get_cursor):
        row = _make_source_row("code content", "code")
        mock_get_cursor.fetchall.return_value = [row]

        result = await list_sources(source_type="code")

        call_args = mock_get_cursor.execute.call_args
        sql, params = call_args[0]
        assert "WHERE type = %s" in sql
        assert params[0] == "code"
        assert result.success is True
        assert len(result.data) == 1

    async def test_no_type_filter_no_where_clause(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []

        await list_sources()

        call_args = mock_get_cursor.execute.call_args
        sql = call_args[0][0]
        assert "WHERE type" not in sql

    async def test_respects_limit_and_offset(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []

        await list_sources(limit=10, offset=5)

        call_args = mock_get_cursor.execute.call_args
        params = call_args[0][1]
        assert 10 in params
        assert 5 in params

    async def test_default_limit_50(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []

        await list_sources()

        params = mock_get_cursor.execute.call_args[0][1]
        assert 50 in params

    async def test_results_stripped_of_tsv(self, mock_get_cursor):
        row = _make_source_row()
        row["content_tsv"] = "tsvector_data"
        mock_get_cursor.fetchall.return_value = [row]

        result = await list_sources()
        assert result.success is True
        assert "content_tsv" not in result.data[0]


# ---------------------------------------------------------------------------
# All valid source types are covered
# ---------------------------------------------------------------------------


class TestValidSourceTypes:
    def test_all_expected_types_present(self):
        expected = {"document", "conversation", "web", "code", "observation", "tool_output", "user_input"}
        assert VALID_SOURCE_TYPES == expected

    def test_all_types_have_reliability_default(self):
        for t in VALID_SOURCE_TYPES:
            assert t in RELIABILITY_DEFAULTS, f"Missing reliability default for type: {t}"

    def test_reliability_values_in_range(self):
        for t, r in RELIABILITY_DEFAULTS.items():
            assert 0.0 <= r <= 1.0, f"Out-of-range reliability for {t}: {r}"
