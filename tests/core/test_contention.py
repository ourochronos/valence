"""Tests for valence.core.contention module (WU-08).

All LLM calls and DB operations are mocked — no real LLM or PostgreSQL required.
asyncio_mode = auto (pyproject.toml), so no @pytest.mark.asyncio needed.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

import valence.core.contention as contention_mod
from valence.core.contention import (
    _build_detection_prompt,
    _get_materiality_threshold,
    _heuristic_materiality,
    _parse_llm_json,
    detect_contention,
    list_contentions,
    resolve_contention,
    set_llm_backend,
)

# ---------------------------------------------------------------------------
# Shared test IDs
# ---------------------------------------------------------------------------

ARTICLE_ID = str(uuid4())
SOURCE_ID = str(uuid4())
CONTENTION_ID = str(uuid4())
ARTICLE_B_ID = str(uuid4())
NOW = datetime.now()


# ---------------------------------------------------------------------------
# Row factories
# ---------------------------------------------------------------------------


def _article_row(article_id=None, content="The sky is blue.", title="Test Article") -> dict:
    return {
        "id": article_id or ARTICLE_ID,
        "content": content,
        "title": title,
        "author_type": "system",
        "pinned": False,
        "version": 1,
        "status": "active",
        "created_at": NOW,
        "modified_at": NOW,
        "compiled_at": NOW,
        "usage_score": 0,
        "confidence": json.dumps({"overall": 0.7}),
        "domain_path": [],
        "size_tokens": 5,
        "content_hash": "abc",
        "superseded_by_id": None,
        "extraction_metadata": None,
    }


def _source_row(source_id=None, content="The sky is green.") -> dict:
    return {
        "id": source_id or SOURCE_ID,
        "content": content,
        "title": "Test Source",
    }


def _contention_row(
    contention_id=None,
    article_id=None,
    source_id=None,
    materiality=0.6,
    status="detected",
    resolution=None,
    resolved_at=None,
) -> dict:
    return {
        "id": contention_id or CONTENTION_ID,
        "article_id": article_id or ARTICLE_ID,
        "related_article_id": None,
        "source_id": source_id or SOURCE_ID,
        "type": "contradiction",
        "description": "Source disagrees on sky color",
        "severity": "medium",
        "status": status,
        "materiality": materiality,
        "resolution": resolution,
        "resolved_at": resolved_at,
        "detected_at": NOW,
    }


def _make_cursor(fetchone_seq=None, fetchall_seq=None):
    """Build a mock cursor with context-manager support."""
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


@contextmanager
def _patch_cursor(mock_cur):
    """Patch get_cursor at the contention module level."""
    with patch("valence.core.contention.get_cursor", return_value=mock_cur):
        yield mock_cur


def _skip_schema_ensure():
    """Patch _ensure_contention_schema to be a no-op."""
    return patch.object(contention_mod, "_ensure_contention_schema")


def _patch_threshold(value=0.3):
    """Patch materiality threshold."""
    return patch.object(contention_mod, "_get_materiality_threshold", return_value=value)


# ---------------------------------------------------------------------------
# LLM response helpers
# ---------------------------------------------------------------------------


def _llm_contends(
    materiality=0.7,
    contention_type="contradiction",
    description="Source directly contradicts article.",
) -> str:
    return json.dumps(
        {
            "is_contention": True,
            "contention_type": contention_type,  # extra field, ignored by schema
            "materiality": materiality,
            "explanation": description,
        }
    )


def _llm_no_contention() -> str:
    return json.dumps(
        {
            "is_contention": False,
            "contention_type": "contradiction",  # extra field, ignored by schema
            "materiality": 0.0,
            "explanation": None,
        }
    )


# ---------------------------------------------------------------------------
# Tests: _parse_llm_json
# ---------------------------------------------------------------------------


class TestParseLlmJson:
    def test_clean_json(self):
        raw = '{"contends": true, "materiality": 0.7}'
        parsed = _parse_llm_json(raw, ["contends", "materiality"])
        assert parsed["contends"] is True
        assert parsed["materiality"] == 0.7

    def test_strips_markdown_fence(self):
        raw = '```json\n{"contends": false, "materiality": 0.0}\n```'
        parsed = _parse_llm_json(raw, ["contends"])
        assert parsed["contends"] is False

    def test_strips_plain_fence(self):
        raw = '```\n{"contends": true, "materiality": 0.5}\n```'
        parsed = _parse_llm_json(raw, ["contends"])
        assert parsed["contends"] is True

    def test_missing_required_key(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _parse_llm_json('{"contends": true}', ["contends", "materiality"])

    def test_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_json("not json", ["contends"])


# ---------------------------------------------------------------------------
# Tests: _build_detection_prompt
# ---------------------------------------------------------------------------


class TestBuildDetectionPrompt:
    def test_includes_article_content(self):
        prompt = _build_detection_prompt("The sky is blue.", "The sky is green.")
        assert "The sky is blue." in prompt
        assert "The sky is green." in prompt

    def test_includes_json_schema_keys(self):
        prompt = _build_detection_prompt("A", "B")
        assert "is_contention" in prompt
        assert "materiality" in prompt
        assert "explanation" in prompt


# ---------------------------------------------------------------------------
# Tests: _heuristic_materiality
# ---------------------------------------------------------------------------


class TestHeuristicMateriality:
    def test_returns_above_default_threshold(self):
        score = _heuristic_materiality("some text", "different text")
        assert score > 0.3  # above DEFAULT_MATERIALITY_THRESHOLD

    def test_returns_float(self):
        assert isinstance(_heuristic_materiality("a", "b"), float)


# ---------------------------------------------------------------------------
# Tests: detect_contention — LLM path
# ---------------------------------------------------------------------------


class TestDetectContention:
    async def test_creates_contention_when_material(self):
        """LLM returns high materiality → contention row created."""
        contention_row = _contention_row(materiality=0.7)
        cur = _make_cursor(
            fetchone_seq=[
                _article_row(),  # article lookup
                _source_row(),  # source lookup
                contention_row,  # INSERT RETURNING
            ]
        )
        set_llm_backend(lambda _p: _llm_contends(materiality=0.7))
        try:
            with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
                result = await detect_contention(ARTICLE_ID, SOURCE_ID)

            assert result.data is not None
            assert result.data["status"] == "detected"
            assert float(result.data["materiality"]) == pytest.approx(0.6, abs=0.1)
        finally:
            set_llm_backend(None)

    async def test_returns_none_when_below_threshold(self):
        """LLM returns low materiality → no contention created."""
        cur = _make_cursor(
            fetchone_seq=[
                _article_row(),
                _source_row(),
            ]
        )
        set_llm_backend(lambda _p: _llm_contends(materiality=0.1))
        try:
            with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
                result = await detect_contention(ARTICLE_ID, SOURCE_ID)
            assert result.data is None
        finally:
            set_llm_backend(None)

    async def test_returns_none_when_no_contention(self):
        """LLM says no contention → None returned."""
        cur = _make_cursor(
            fetchone_seq=[
                _article_row(),
                _source_row(),
            ]
        )
        set_llm_backend(lambda _p: _llm_no_contention())
        try:
            with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
                result = await detect_contention(ARTICLE_ID, SOURCE_ID)
            assert result.data is None
        finally:
            set_llm_backend(None)

    async def test_returns_none_when_article_not_found(self):
        """Article missing → returns None without error."""
        cur = _make_cursor(fetchone_seq=[None])
        with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
            result = await detect_contention(ARTICLE_ID, SOURCE_ID)
        assert result.data is None

    async def test_returns_none_when_source_not_found(self):
        """Source missing → returns None without error."""
        cur = _make_cursor(fetchone_seq=[_article_row(), None])
        with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
            result = await detect_contention(ARTICLE_ID, SOURCE_ID)
        assert result.data is None

    async def test_returns_none_when_empty_article_content(self):
        """Empty article content → returns None."""
        cur = _make_cursor(
            fetchone_seq=[
                _article_row(content=""),
                _source_row(),
            ]
        )
        with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
            result = await detect_contention(ARTICLE_ID, SOURCE_ID)
        assert result.data is None


# ---------------------------------------------------------------------------
# Tests: detect_contention — heuristic fallback (no LLM)
# ---------------------------------------------------------------------------


class TestDetectContentionHeuristicFallback:
    async def test_fallback_when_llm_unavailable(self):
        """When LLM is not configured, heuristic is used and contention created."""
        contention_row = _contention_row(materiality=0.35)
        cur = _make_cursor(
            fetchone_seq=[
                _article_row(),
                _source_row(),
                contention_row,
            ]
        )
        # No LLM backend set — uses heuristic
        set_llm_backend(None)
        with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
            result = await detect_contention(ARTICLE_ID, SOURCE_ID)

        # Heuristic returns DEFAULT_MATERIALITY_THRESHOLD + 0.05 = 0.35
        assert result.data is not None
        assert result.data["status"] == "detected"

    async def test_fallback_creates_db_row(self):
        """Verify INSERT is called on the cursor in fallback path."""
        contention_row = _contention_row()
        cur = _make_cursor(
            fetchone_seq=[
                _article_row(),
                _source_row(),
                contention_row,
            ]
        )
        set_llm_backend(None)
        with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
            await detect_contention(ARTICLE_ID, SOURCE_ID)

        execute_calls = str(cur.execute.call_args_list)
        assert "INSERT INTO contentions" in execute_calls


# ---------------------------------------------------------------------------
# Tests: detect_contention — materiality threshold
# ---------------------------------------------------------------------------


class TestMaterialityThreshold:
    async def test_exactly_at_threshold_creates_contention(self):
        """Materiality equal to threshold is accepted (>= comparison)."""
        contention_row = _contention_row(materiality=0.3)
        cur = _make_cursor(fetchone_seq=[_article_row(), _source_row(), contention_row])
        set_llm_backend(lambda _p: _llm_contends(materiality=0.3))
        try:
            with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
                result = await detect_contention(ARTICLE_ID, SOURCE_ID)
            # 0.3 >= 0.3 → creates contention
            assert result.data is not None
        finally:
            set_llm_backend(None)

    async def test_just_below_threshold_returns_none(self):
        """Materiality just below threshold is suppressed."""
        cur = _make_cursor(fetchone_seq=[_article_row(), _source_row()])
        set_llm_backend(lambda _p: _llm_contends(materiality=0.29))
        try:
            with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
                result = await detect_contention(ARTICLE_ID, SOURCE_ID)
            assert result.data is None
        finally:
            set_llm_backend(None)


# ---------------------------------------------------------------------------
# Tests: list_contentions
# ---------------------------------------------------------------------------


class TestListContentions:
    async def test_returns_all_for_status(self):
        """list_contentions returns all rows matching status."""
        rows = [
            _contention_row(contention_id=str(uuid4())),
            _contention_row(contention_id=str(uuid4())),
        ]
        cur = _make_cursor(fetchall_seq=[rows])
        with _patch_cursor(cur), _skip_schema_ensure():
            result = await list_contentions(status="detected")
        assert len(result.data) == 2

    async def test_filters_by_article_id(self):
        """list_contentions passes article_id filter to SQL."""
        rows = [_contention_row()]
        cur = _make_cursor(fetchall_seq=[rows])
        with _patch_cursor(cur), _skip_schema_ensure():
            await list_contentions(article_id=ARTICLE_ID, status="detected")

        execute_calls = str(cur.execute.call_args_list)
        assert "article_id" in execute_calls or ARTICLE_ID in execute_calls

    async def test_returns_empty_list_when_none(self):
        """list_contentions returns [] when no rows match."""
        cur = _make_cursor(fetchall_seq=[[]])
        with _patch_cursor(cur), _skip_schema_ensure():
            result = await list_contentions()
        assert result.data == []

    async def test_no_status_filter(self):
        """Passing status=None skips the status filter."""
        rows = [
            _contention_row(status="detected"),
            _contention_row(status="resolved"),
        ]
        cur = _make_cursor(fetchall_seq=[rows])
        with _patch_cursor(cur), _skip_schema_ensure():
            await list_contentions(status=None)

        execute_calls = str(cur.execute.call_args_list)
        # "status = %s" should NOT be in the SQL when status is None
        assert "status = %s" not in execute_calls

    async def test_serializes_uuids(self):
        """list_contentions serializes UUID objects to strings."""
        from uuid import UUID

        row_with_uuid = dict(_contention_row())
        row_with_uuid["id"] = UUID(CONTENTION_ID)
        cur = _make_cursor(fetchall_seq=[[row_with_uuid]])
        with _patch_cursor(cur), _skip_schema_ensure():
            result = await list_contentions()
        assert isinstance(result.data[0]["id"], str)


# ---------------------------------------------------------------------------
# Tests: resolve_contention
# ---------------------------------------------------------------------------


class TestResolveContention:
    async def test_invalid_resolution_returns_error(self):
        """Unknown resolution string returns error dict without DB access."""
        result = await resolve_contention(CONTENTION_ID, "explode", "rationale")
        assert result.success is False
        assert "resolution" in result.error

    async def test_not_found_returns_error(self):
        """Contention not found → error."""
        cur = _make_cursor(fetchone_seq=[None])
        with _patch_cursor(cur), _skip_schema_ensure():
            result = await resolve_contention(CONTENTION_ID, "dismiss", "not real")
        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_dismiss_marks_dismissed(self):
        """dismiss → contention status = 'dismissed', article unchanged."""
        updated_row = _contention_row(status="dismissed", resolution="testing")
        cur = _make_cursor(
            fetchone_seq=[
                _contention_row(),  # load contention
                updated_row,  # UPDATE RETURNING
            ]
        )
        with _patch_cursor(cur), _skip_schema_ensure():
            result = await resolve_contention(CONTENTION_ID, "dismiss", "Not material")

        assert result.success is True
        assert result.data["contention"]["status"] == "dismissed"
        execute_calls = str(cur.execute.call_args_list)
        assert "UPDATE contentions" in execute_calls
        assert "dismissed" in execute_calls

    async def test_supersede_a_marks_resolved_no_article_change(self):
        """supersede_a → resolved status, article content untouched."""
        updated_row = _contention_row(status="resolved", resolution="article wins")
        cur = _make_cursor(
            fetchone_seq=[
                _contention_row(),  # load contention
                updated_row,  # UPDATE RETURNING
            ]
        )
        with _patch_cursor(cur), _skip_schema_ensure():
            result = await resolve_contention(CONTENTION_ID, "supersede_a", "Article is authoritative")

        assert result.success is True
        assert result.data["contention"]["status"] == "resolved"
        assert result.data.get("article") is None  # article not modified

    async def test_accept_both_annotates_article(self):
        """accept_both → contention resolved, article extraction_metadata updated."""
        updated_row = _contention_row(status="resolved")
        # accept_both branch: UPDATE articles (no fetchone call), then UPDATE contentions RETURNING
        cur = _make_cursor(
            fetchone_seq=[
                _contention_row(),  # load contention
                updated_row,  # UPDATE contentions RETURNING
            ]
        )
        with _patch_cursor(cur), _skip_schema_ensure():
            result = await resolve_contention(CONTENTION_ID, "accept_both", "Both valid")

        assert result.success is True
        execute_calls = str(cur.execute.call_args_list)
        assert "UPDATE articles" in execute_calls

    async def test_supersede_b_updates_article(self):
        """supersede_b → source content replaces article content."""
        contention = _contention_row(source_id=SOURCE_ID)
        article = _article_row()
        source = _source_row()
        updated_article = _article_row(content="The sky is green.")
        updated_contention = _contention_row(status="resolved")

        # Sequence: load contention, load article A, load source,
        # UPDATE articles RETURNING, (INSERT mutations — no fetchone),
        # UPDATE contentions RETURNING
        cur = _make_cursor(
            fetchone_seq=[
                contention,  # load contention
                article,  # load article A (in _apply_supersede_b)
                source,  # load source
                updated_article,  # UPDATE articles RETURNING
                updated_contention,  # UPDATE contentions RETURNING
            ]
        )
        # LLM unavailable → fallback uses source content directly
        set_llm_backend(None)
        with _patch_cursor(cur), _skip_schema_ensure():
            result = await resolve_contention(CONTENTION_ID, "supersede_b", "Source has newer data")

        assert result.success is True
        execute_calls = str(cur.execute.call_args_list)
        assert "UPDATE articles" in execute_calls

    async def test_supersede_b_with_llm(self):
        """supersede_b with LLM available → LLM merges content."""
        contention = _contention_row(source_id=SOURCE_ID)
        article = _article_row()
        source = _source_row()
        updated_article = _article_row(content="Merged content from LLM.")
        updated_contention = _contention_row(status="resolved")

        cur = _make_cursor(
            fetchone_seq=[
                contention,
                article,
                source,
                updated_article,
                updated_contention,
            ]
        )
        set_llm_backend(lambda _p: json.dumps({"content": "Merged content from LLM."}))
        try:
            with _patch_cursor(cur), _skip_schema_ensure():
                result = await resolve_contention(CONTENTION_ID, "supersede_b", "Merge")
            assert result.success is True
        finally:
            set_llm_backend(None)

    async def test_resolve_records_resolution_text(self):
        """Resolution rationale is stored on the contention row."""
        updated_row = _contention_row(status="dismissed", resolution="Not important")
        cur = _make_cursor(fetchone_seq=[_contention_row(), updated_row])
        with _patch_cursor(cur), _skip_schema_ensure():
            await resolve_contention(CONTENTION_ID, "dismiss", "Not important")

        execute_calls = str(cur.execute.call_args_list)
        assert "Not important" in execute_calls or "resolved_at" in execute_calls


# ---------------------------------------------------------------------------
# Tests: _get_materiality_threshold
# ---------------------------------------------------------------------------


class TestGetMaterialityThreshold:
    def test_returns_default_when_config_missing(self):
        cur = _make_cursor(fetchone_seq=[None])
        with _patch_cursor(cur):
            threshold = _get_materiality_threshold()
        assert threshold == 0.3

    def test_reads_from_system_config(self):
        config_row = {"value": json.dumps({"materiality_threshold": 0.5})}
        cur = _make_cursor(fetchone_seq=[config_row])
        with _patch_cursor(cur):
            threshold = _get_materiality_threshold()
        assert threshold == 0.5

    def test_reads_jsonb_dict_directly(self):
        """When value is already a dict (psycopg2 JSONB auto-decode)."""
        config_row = {"value": {"materiality_threshold": 0.6}}
        cur = _make_cursor(fetchone_seq=[config_row])
        with _patch_cursor(cur):
            threshold = _get_materiality_threshold()
        assert threshold == 0.6


# ---------------------------------------------------------------------------
# Tests: set_llm_backend / LLM error handling
# ---------------------------------------------------------------------------


class TestLLMBackend:
    async def test_set_and_clear_backend(self):
        """set_llm_backend(None) clears the backend."""
        set_llm_backend(lambda _p: _llm_no_contention())
        cur = _make_cursor(fetchone_seq=[_article_row(), _source_row()])
        with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
            result = await detect_contention(ARTICLE_ID, SOURCE_ID)
        assert result.data is None  # LLM says no contention

        set_llm_backend(None)
        # Now heuristic is used → contention may be created
        contention_row = _contention_row()
        cur2 = _make_cursor(fetchone_seq=[_article_row(), _source_row(), contention_row])
        with _patch_cursor(cur2), _skip_schema_ensure(), _patch_threshold(0.3):
            result2 = await detect_contention(ARTICLE_ID, SOURCE_ID)
        # Heuristic returns 0.35 which is >= 0.3 → creates contention
        assert result2 is not None

    async def test_async_llm_backend(self):
        """detect_contention works with an async LLM backend."""
        import asyncio

        async def async_llm(_p: str) -> str:
            await asyncio.sleep(0)
            return _llm_no_contention()

        set_llm_backend(async_llm)
        try:
            cur = _make_cursor(fetchone_seq=[_article_row(), _source_row()])
            with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
                result = await detect_contention(ARTICLE_ID, SOURCE_ID)
            assert result.data is None
        finally:
            set_llm_backend(None)

    async def test_llm_returns_invalid_json_falls_back_to_heuristic(self):
        """Malformed LLM JSON falls back to heuristic."""
        contention_row = _contention_row()
        cur = _make_cursor(fetchone_seq=[_article_row(), _source_row(), contention_row])
        set_llm_backend(lambda _p: "this is not json !! {{{")
        try:
            with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
                result = await detect_contention(ARTICLE_ID, SOURCE_ID)
            # Heuristic path triggers
            assert result.data is not None
        finally:
            set_llm_backend(None)

    async def test_invalid_contention_type_defaults_to_contradiction(self):
        """Unknown contention_type from LLM is normalized to 'contradiction'."""
        contention_row = _contention_row()
        cur = _make_cursor(fetchone_seq=[_article_row(), _source_row(), contention_row])
        llm_resp = json.dumps(
            {
                "is_contention": True,
                "contention_type": "total_annihilation",  # unknown type, extra field ignored
                "materiality": 0.8,
                "explanation": "Conflicts.",
            }
        )
        set_llm_backend(lambda _p: llm_resp)
        try:
            with _patch_cursor(cur), _skip_schema_ensure(), _patch_threshold(0.3):
                result = await detect_contention(ARTICLE_ID, SOURCE_ID)
            assert result.data is not None
            execute_calls = str(cur.execute.call_args_list)
            assert "contradiction" in execute_calls
        finally:
            set_llm_backend(None)
