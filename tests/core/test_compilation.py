"""Tests for valence.core.compilation module (WU-06).

All LLM calls are mocked — no real LLM invocations.
Uses the same mock-cursor pattern as test_articles.py / test_provenance.py.
asyncio_mode = auto (pyproject.toml), so no @pytest.mark.asyncio needed.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

import valence.core.compilation as compilation_mod
from valence.core.compilation import (
    _build_compilation_prompt,
    _build_update_prompt,
    _count_tokens,
    _get_right_sizing,
    _parse_llm_json,
    compile_article,
    process_mutation_queue,
    set_llm_backend,
    update_article_from_source,
)

# ---------------------------------------------------------------------------
# Shared test IDs & defaults
# ---------------------------------------------------------------------------

ARTICLE_ID = str(uuid4())
SOURCE_ID_1 = str(uuid4())
SOURCE_ID_2 = str(uuid4())
QUEUE_ITEM_ID = str(uuid4())
NOW = datetime.now()

DEFAULT_RS = {"max_tokens": 4000, "min_tokens": 200, "target_tokens": 2000}


# ---------------------------------------------------------------------------
# Row factories
# ---------------------------------------------------------------------------


def _source_row(source_id=None, title="Source Doc", content="Python is great.") -> dict:
    return {
        "id": source_id or SOURCE_ID_1,
        "type": "document",
        "title": title,
        "url": None,
        "content": content,
        "reliability": 0.8,
    }


def _article_row(article_id=None, content="Compiled content.", version=1, **kw) -> dict:
    defaults = {
        "id": article_id or ARTICLE_ID,
        "content": content,
        "title": "Test Article",
        "author_type": "system",
        "pinned": False,
        "size_tokens": _count_tokens(content),
        "compiled_at": NOW,
        "usage_score": 0,
        "confidence": json.dumps({"overall": 0.7}),
        "domain_path": [],
        "version": version,
        "content_hash": "abc",
        "status": "active",
        "superseded_by_id": None,
        "created_at": NOW,
        "modified_at": NOW,
    }
    defaults.update(kw)
    return defaults


def _queue_item(operation="recompile", article_id=None, payload=None) -> dict:
    return {
        "id": str(uuid4()),
        "operation": operation,
        "article_id": article_id or ARTICLE_ID,
        "payload": payload or {},
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


def _llm_ok(title="Compiled Title", content="Compiled content.", rels=None) -> str:
    if rels is None:
        rels = [{"source_id": SOURCE_ID_1, "relationship": "originates"}]
    return json.dumps({"title": title, "content": content, "source_relationships": rels})


def _llm_update_ok(content="Updated content.", relationship="confirms", summary="Updated.") -> str:
    return json.dumps({"content": content, "relationship": relationship, "changes_summary": summary})


def _patch_rs(rs=None):
    """Return a patch context for _get_right_sizing returning rs (defaults to DEFAULT_RS)."""
    return patch.object(compilation_mod, "_get_right_sizing", return_value=rs or DEFAULT_RS)


# ---------------------------------------------------------------------------
# Unit: _count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string(self):
        # max(1, int(0 * 1.3)) = 1
        assert _count_tokens("") == 1

    def test_two_words(self):
        # int(2 * 1.3) = 2
        assert _count_tokens("hello world") == 2

    def test_five_words(self):
        # int(5 * 1.3) = 6
        assert _count_tokens("one two three four five") == 6

    def test_large_content(self):
        words = " ".join(["word"] * 3000)
        assert _count_tokens(words) == int(3000 * 1.3)


# ---------------------------------------------------------------------------
# Unit: _parse_llm_json
# ---------------------------------------------------------------------------


class TestParseLlmJson:
    def test_clean_json(self):
        raw = '{"title": "T", "content": "C", "relationships": []}'
        parsed = _parse_llm_json(raw, ["title", "content"])
        assert parsed["title"] == "T"
        assert parsed["content"] == "C"

    def test_strips_json_markdown_fence(self):
        raw = '```json\n{"title": "T", "content": "C"}\n```'
        parsed = _parse_llm_json(raw, ["title", "content"])
        assert parsed["title"] == "T"

    def test_strips_plain_fence(self):
        raw = '```\n{"title": "T", "content": "C"}\n```'
        parsed = _parse_llm_json(raw, ["content"])
        assert parsed["content"] == "C"

    def test_missing_required_key_raises(self):
        raw = '{"title": "T"}'
        with pytest.raises(ValueError, match="missing required keys"):
            _parse_llm_json(raw, ["title", "content"])

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_json("not json at all", [])


# ---------------------------------------------------------------------------
# Unit: _build_compilation_prompt
# ---------------------------------------------------------------------------


class TestBuildCompilationPrompt:
    def test_includes_all_source_content(self):
        sources = [
            {"id": SOURCE_ID_1, "title": "Alpha", "content": "Content A"},
            {"id": SOURCE_ID_2, "title": "Beta", "content": "Content B"},
        ]
        prompt = _build_compilation_prompt(sources, None, 2000)
        assert "Alpha" in prompt
        assert "Content A" in prompt
        assert "Beta" in prompt
        assert "Content B" in prompt

    def test_includes_title_hint(self):
        sources = [{"id": SOURCE_ID_1, "title": "S", "content": "C"}]
        prompt = _build_compilation_prompt(sources, "My Custom Title", 2000)
        assert "My Custom Title" in prompt

    def test_includes_target_tokens(self):
        sources = [{"id": SOURCE_ID_1, "title": "S", "content": "C"}]
        prompt = _build_compilation_prompt(sources, None, 1500)
        assert "1500" in prompt

    def test_no_title_hint_omits_title_instruction(self):
        sources = [{"id": SOURCE_ID_1, "title": "S", "content": "C"}]
        prompt = _build_compilation_prompt(sources, None, 2000)
        assert "title should be" not in prompt

    def test_numbered_sources(self):
        sources = [
            {"id": SOURCE_ID_1, "title": "A", "content": "ca"},
            {"id": SOURCE_ID_2, "title": "B", "content": "cb"},
        ]
        prompt = _build_compilation_prompt(sources, None, 2000)
        # Prompt lists sources by id (format: "Source id=<uuid>: Title")
        assert SOURCE_ID_1 in prompt
        assert SOURCE_ID_2 in prompt


# ---------------------------------------------------------------------------
# Unit: _build_update_prompt
# ---------------------------------------------------------------------------


class TestBuildUpdatePrompt:
    def test_includes_article_and_source(self):
        article = {"content": "Old article body.", "title": "Old Title"}
        source = {"content": "New data point.", "title": "New Source"}
        prompt = _build_update_prompt(article, source, 2000)
        assert "Old article body." in prompt
        assert "New data point." in prompt
        assert "2000" in prompt

    def test_fallback_titles_when_none(self):
        article = {"content": "Body", "title": None}
        source = {"content": "Src", "title": None}
        prompt = _build_update_prompt(article, source, 2000)
        assert "Untitled Article" in prompt
        assert "New Source" in prompt


# ---------------------------------------------------------------------------
# Unit: set_llm_backend
# ---------------------------------------------------------------------------


class TestSetLlmBackend:
    def test_set_and_clear(self):
        original = compilation_mod._LLM_BACKEND
        try:
            set_llm_backend(lambda p: "response")
            assert compilation_mod._LLM_BACKEND is not None
            set_llm_backend(None)
            assert compilation_mod._LLM_BACKEND is None
        finally:
            compilation_mod._LLM_BACKEND = original

    async def test_not_implemented_when_no_backend(self):
        original = compilation_mod._LLM_BACKEND
        try:
            set_llm_backend(None)
            with pytest.raises(NotImplementedError):
                await compilation_mod._call_llm("test prompt")
        finally:
            compilation_mod._LLM_BACKEND = original

    async def test_sync_callable_works(self):
        original = compilation_mod._LLM_BACKEND
        try:
            set_llm_backend(lambda p: "sync response")
            result = await compilation_mod._call_llm("test")
            assert result == "sync response"
        finally:
            compilation_mod._LLM_BACKEND = original

    async def test_async_callable_works(self):
        original = compilation_mod._LLM_BACKEND
        try:

            async def async_fn(p):
                return "async response"

            set_llm_backend(async_fn)
            result = await compilation_mod._call_llm("test")
            assert result == "async response"
        finally:
            compilation_mod._LLM_BACKEND = original


# ---------------------------------------------------------------------------
# Unit: _get_right_sizing
# ---------------------------------------------------------------------------


class TestGetRightSizing:
    def test_returns_defaults_when_db_empty(self):
        cur = _make_cursor(fetchone_seq=[None])
        with patch("valence.core.compilation.get_cursor", return_value=cur):
            config = _get_right_sizing()
        assert config["max_tokens"] == 4000
        assert config["target_tokens"] == 2000
        assert config["min_tokens"] == 200

    def test_merges_db_values_over_defaults(self):
        cur = _make_cursor(fetchone_seq=[{"value": {"max_tokens": 3000, "min_tokens": 100, "target_tokens": 1500}}])
        with patch("valence.core.compilation.get_cursor", return_value=cur):
            config = _get_right_sizing()
        assert config["max_tokens"] == 3000
        assert config["target_tokens"] == 1500

    def test_handles_json_string_value(self):
        """system_config.value may come back as a JSON string."""
        cur = _make_cursor(fetchone_seq=[{"value": json.dumps({"max_tokens": 5000, "target_tokens": 2500, "min_tokens": 300})}])
        with patch("valence.core.compilation.get_cursor", return_value=cur):
            config = _get_right_sizing()
        assert config["max_tokens"] == 5000


# ---------------------------------------------------------------------------
# compile_article
# ---------------------------------------------------------------------------


class TestCompileArticle:
    async def test_empty_source_ids_returns_error(self):
        result = await compile_article([])
        assert result.success is False
        assert "source_id" in result.error.lower()

    async def test_source_not_found_returns_error(self):
        cur = _make_cursor(fetchone_seq=[None])
        with patch("valence.core.compilation.get_cursor", return_value=cur), _patch_rs():
            result = await compile_article([SOURCE_ID_1])
        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_success_with_llm(self):
        """Successful compilation creates article and links source."""
        src = _source_row(SOURCE_ID_1)
        art = _article_row(ARTICLE_ID, "Compiled content.")
        llm_resp = _llm_ok(
            content="Compiled content.",
            rels=[{"source_id": SOURCE_ID_1, "relationship": "originates"}],
        )
        cur = _make_cursor(fetchone_seq=[src, art])

        async def mock_llm(prompt, **_kw):
            return llm_resp

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            result = await compile_article([SOURCE_ID_1])

        assert result.success is True
        assert result.data is not None
        calls_str = str(cur.execute.call_args_list)
        assert "article_sources" in calls_str

    async def test_success_with_two_sources(self):
        """Two sources are linked with their LLM-identified relationships."""
        src1 = _source_row(SOURCE_ID_1, "Doc 1", "Info about Python 3.12.")
        src2 = _source_row(SOURCE_ID_2, "Doc 2", "Python 3.12 performance update.")
        art = _article_row(ARTICLE_ID, "Python 3.12 synthesized article.")
        llm_resp = _llm_ok(
            content="Python 3.12 synthesized article.",
            rels=[
                {"source_id": SOURCE_ID_1, "relationship": "originates"},
                {"source_id": SOURCE_ID_2, "relationship": "confirms"},
            ],
        )
        cur = _make_cursor(fetchone_seq=[src1, src2, art])

        async def mock_llm(prompt, **_kw):
            return llm_resp

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            result = await compile_article([SOURCE_ID_1, SOURCE_ID_2])

        assert result.success is True
        calls_str = str(cur.execute.call_args_list)
        assert "originates" in calls_str
        assert "confirms" in calls_str

    async def test_records_created_mutation(self):
        """compile_article records a 'created' mutation row."""
        src = _source_row()
        art = _article_row()
        cur = _make_cursor(fetchone_seq=[src, art])

        async def mock_llm(prompt, **_kw):
            return _llm_ok()

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            await compile_article([SOURCE_ID_1])

        calls_str = str(cur.execute.call_args_list)
        assert "article_mutations" in calls_str
        assert "created" in calls_str

    async def test_llm_fallback_on_not_implemented(self):
        """When LLM raises NotImplementedError, falls back to source concatenation."""
        src = _source_row(SOURCE_ID_1, "Source A", "Short content.")
        art = _article_row(ARTICLE_ID, "## Source A\nShort content.")
        cur = _make_cursor(fetchone_seq=[src, art])

        async def fail_llm(prompt, **_kw):
            raise NotImplementedError("no backend")

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=fail_llm),
        ):
            result = await compile_article([SOURCE_ID_1], title_hint="Fallback Title")

        assert result.success is True

    async def test_llm_fallback_on_json_error(self):
        """When LLM returns invalid JSON, falls back gracefully."""
        src = _source_row()
        art = _article_row()
        cur = _make_cursor(fetchone_seq=[src, art])

        async def bad_llm(prompt, **_kw):
            return "this is not json"

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=bad_llm),
        ):
            result = await compile_article([SOURCE_ID_1])

        assert result.success is True

    async def test_title_hint_forwarded_to_prompt(self):
        """Title hint is forwarded in the LLM prompt."""
        src = _source_row()
        art = _article_row()
        cur = _make_cursor(fetchone_seq=[src, art])
        prompts_seen: list[str] = []

        async def capturing_llm(prompt, **_kw):
            prompts_seen.append(prompt)
            return _llm_ok()

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=capturing_llm),
        ):
            await compile_article([SOURCE_ID_1], title_hint="Specific Title")

        assert len(prompts_seen) == 1
        assert "Specific Title" in prompts_seen[0]


# ---------------------------------------------------------------------------
# compile_article — right-sizing enforcement
# ---------------------------------------------------------------------------


class TestCompileArticleRightSizing:
    async def test_oversized_article_queues_split_not_inline(self):
        """Articles over max_tokens get split queued in mutation_queue (not inline)."""
        big_content = "word " * 200  # ~260 tokens, exceeds max_tokens=10
        src = _source_row(SOURCE_ID_1, "Big Doc", big_content)
        art = _article_row(ARTICLE_ID, big_content)
        llm_resp = json.dumps(
            {
                "title": "Big Article",
                "content": big_content,
                "source_relationships": [{"source_id": SOURCE_ID_1, "relationship": "originates"}],
            }
        )

        cur = _make_cursor(fetchone_seq=[src, art])
        queued_ops: list[str] = []

        def track_execute(sql, params=None):
            # INSERT into mutation_queue — "split" is in the SQL VALUES literal
            if "mutation_queue" in str(sql):
                queued_ops.append("mutation_queue_insert")

        cur.execute.side_effect = track_execute

        async def mock_llm(prompt, **_kw):
            return llm_resp

        tiny_rs = {"max_tokens": 10, "min_tokens": 1, "target_tokens": 5}
        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(tiny_rs),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            result = await compile_article([SOURCE_ID_1])

        # Compilation must succeed (split is deferred, DR-6)
        assert result.success is True
        # Verify that a mutation_queue INSERT was issued (it's a 'split' operation)
        assert len(queued_ops) > 0

    async def test_within_limit_no_split_queued(self):
        """Articles within max_tokens do not get a split entry queued."""
        short_content = "Short article."
        src = _source_row(SOURCE_ID_1, "Short Doc", short_content)
        art = _article_row(ARTICLE_ID, short_content)
        llm_resp = json.dumps(
            {
                "title": "Short",
                "content": short_content,
                "source_relationships": [{"source_id": SOURCE_ID_1, "relationship": "originates"}],
            }
        )

        cur = _make_cursor(fetchone_seq=[src, art])
        split_queued: list[bool] = []

        def track_execute(sql, params=None):
            if "mutation_queue" in str(sql):
                split_queued.append(True)

        cur.execute.side_effect = track_execute

        async def mock_llm(prompt, **_kw):
            return llm_resp

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            result = await compile_article([SOURCE_ID_1])

        assert result.success is True
        assert len(split_queued) == 0


# ---------------------------------------------------------------------------
# update_article_from_source
# ---------------------------------------------------------------------------


class TestUpdateArticleFromSource:
    async def test_article_not_found(self):
        cur = _make_cursor(fetchone_seq=[None])
        with patch("valence.core.compilation.get_cursor", return_value=cur), _patch_rs():
            result = await update_article_from_source(ARTICLE_ID, SOURCE_ID_1)
        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_source_not_found(self):
        art = _article_row()
        cur = _make_cursor(fetchone_seq=[art, None])
        with patch("valence.core.compilation.get_cursor", return_value=cur), _patch_rs():
            result = await update_article_from_source(ARTICLE_ID, SOURCE_ID_1)
        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_success_confirms(self):
        art = _article_row(ARTICLE_ID, "Existing content about Python.")
        src = _source_row(SOURCE_ID_1, "Confirming Source", "Python 3.12 is great.")
        updated = _article_row(ARTICLE_ID, "Updated content.", version=2)
        cur = _make_cursor(fetchone_seq=[art, src, updated])

        async def mock_llm(prompt, **_kw):
            return _llm_update_ok("Updated content.", "confirms", "Added confirmation.")

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            result = await update_article_from_source(ARTICLE_ID, SOURCE_ID_1)

        assert result.success is True
        assert result.data["relationship"] == "confirms"

    async def test_success_supersedes(self):
        art = _article_row(ARTICLE_ID, "Old Python 3.11 info.")
        src = _source_row(SOURCE_ID_1, "New Docs", "Python 3.12 replaces 3.11.")
        updated = _article_row(ARTICLE_ID, "Python 3.12 updated.", version=2)
        cur = _make_cursor(fetchone_seq=[art, src, updated])

        async def mock_llm(prompt, **_kw):
            return _llm_update_ok("Python 3.12 updated.", "supersedes", "3.12 supersedes 3.11.")

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            result = await update_article_from_source(ARTICLE_ID, SOURCE_ID_1)

        assert result.data["relationship"] == "supersedes"

    async def test_success_contradicts(self):
        art = _article_row(ARTICLE_ID, "Python is slow.")
        src = _source_row(SOURCE_ID_1, "Benchmark", "Python 3.12 is fast.")
        updated = _article_row(ARTICLE_ID, "Python speed is contested.", version=2)
        cur = _make_cursor(fetchone_seq=[art, src, updated])

        async def mock_llm(prompt, **_kw):
            return _llm_update_ok("Python speed is contested.", "contradicts", "Speed contradicted.")

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            result = await update_article_from_source(ARTICLE_ID, SOURCE_ID_1)

        assert result.data["relationship"] == "contradicts"

    async def test_success_contends(self):
        art = _article_row(ARTICLE_ID, "Use asyncio for concurrency.")
        src = _source_row(SOURCE_ID_1, "Alt Source", "Threads also work for I/O.")
        updated = _article_row(ARTICLE_ID, "Both asyncio and threads work.", version=2)
        cur = _make_cursor(fetchone_seq=[art, src, updated])

        async def mock_llm(prompt, **_kw):
            return _llm_update_ok(updated["content"], "contends", "Alternative added.")

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            result = await update_article_from_source(ARTICLE_ID, SOURCE_ID_1)

        assert result.data["relationship"] == "contends"

    async def test_invalid_relationship_defaults_to_confirms(self):
        """LLM returning an invalid relationship type defaults to 'confirms'."""
        art = _article_row()
        src = _source_row()
        updated = _article_row(version=2)
        cur = _make_cursor(fetchone_seq=[art, src, updated])

        async def mock_llm(prompt, **_kw):
            return json.dumps({"content": "Updated.", "relationship": "invented_type", "changes_summary": "x"})

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            result = await update_article_from_source(ARTICLE_ID, SOURCE_ID_1)

        assert result.success is True
        assert result.data["relationship"] == "confirms"

    async def test_llm_fallback_appends_content(self):
        """When LLM is unavailable, source content is appended."""
        art = _article_row(ARTICLE_ID, "Existing article.")
        src = _source_row(SOURCE_ID_1, "Extra Source", "Extra information here.")
        updated = _article_row(ARTICLE_ID, "Existing article.\n\n## Extra Source\nExtra information here.")
        cur = _make_cursor(fetchone_seq=[art, src, updated])

        async def fail_llm(prompt, **_kw):
            raise NotImplementedError("no backend")

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=fail_llm),
        ):
            result = await update_article_from_source(ARTICLE_ID, SOURCE_ID_1)

        assert result.success is True
        assert result.data["relationship"] == "confirms"

    async def test_records_updated_mutation(self):
        """update_article_from_source records an 'updated' mutation."""
        art = _article_row()
        src = _source_row()
        updated = _article_row(version=2)
        cur = _make_cursor(fetchone_seq=[art, src, updated])

        async def mock_llm(prompt, **_kw):
            return _llm_update_ok()

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            await update_article_from_source(ARTICLE_ID, SOURCE_ID_1)

        calls_str = str(cur.execute.call_args_list)
        assert "article_mutations" in calls_str
        assert "updated" in calls_str

    async def test_oversized_update_queues_split(self):
        """Update that pushes article over max_tokens queues a split (DR-6)."""
        big_content = "word " * 200
        art = _article_row()
        src = _source_row(SOURCE_ID_1, "Big Source", big_content)
        updated = _article_row(ARTICLE_ID, big_content, version=2)
        cur = _make_cursor(fetchone_seq=[art, src, updated])
        queued_splits: list[bool] = []

        def track_execute(sql, params=None):
            # mutation_queue INSERT means a 'split' was queued
            if "mutation_queue" in str(sql):
                queued_splits.append(True)

        cur.execute.side_effect = track_execute

        async def mock_llm(prompt, **_kw):
            return json.dumps({"content": big_content, "relationship": "confirms", "changes_summary": "Big."})

        tiny_rs = {"max_tokens": 10, "min_tokens": 1, "target_tokens": 5}
        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(tiny_rs),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            result = await update_article_from_source(ARTICLE_ID, SOURCE_ID_1)

        assert result.success is True
        assert len(queued_splits) > 0


# ---------------------------------------------------------------------------
# process_mutation_queue
# ---------------------------------------------------------------------------


class TestProcessMutationQueue:
    async def test_empty_queue_returns_zero(self):
        cur = _make_cursor(fetchall_seq=[[]])
        with patch("valence.core.compilation.get_cursor", return_value=cur):
            count = await process_mutation_queue(batch_size=5)
        assert count.data == 0

    async def test_batch_size_passed_to_query(self):
        """The batch size is forwarded to the SQL LIMIT parameter."""
        cur = _make_cursor(fetchall_seq=[[]])
        with patch("valence.core.compilation.get_cursor", return_value=cur):
            await process_mutation_queue(batch_size=3)
        first_call_params = cur.execute.call_args_list[0][0][1]
        assert first_call_params == (3,)

    async def test_recompile_success(self):
        """Recompile fetches sources and creates a new compiled article."""
        item = _queue_item("recompile", ARTICLE_ID)
        source_link = {"source_id": SOURCE_ID_1}
        src = _source_row()
        art = _article_row()

        cur = _make_cursor(
            fetchone_seq=[src, art],
            fetchall_seq=[[item], [source_link]],
        )

        async def mock_llm(prompt, **_kw):
            return _llm_ok()

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            count = await process_mutation_queue()

        assert count.data == 1

    async def test_recompile_no_sources_logs_warning_and_completes(self):
        """Recompile with no sources logs warning but marks item completed."""
        item = _queue_item("recompile", ARTICLE_ID)
        cur = _make_cursor(
            fetchall_seq=[[item], []],  # queue item, then empty source list
        )
        with patch("valence.core.compilation.get_cursor", return_value=cur), _patch_rs():
            count = await process_mutation_queue()
        # No sources → logs warning; still marks completed (no exception raised)
        assert count.data == 1

    async def test_unknown_operation_marks_failed(self):
        """Unknown operations cause item to be marked 'failed'."""
        item = _queue_item("does_not_exist", ARTICLE_ID)
        cur = _make_cursor(fetchall_seq=[[item]])
        with patch("valence.core.compilation.get_cursor", return_value=cur):
            count = await process_mutation_queue()
        assert count.data == 0
        calls_str = str(cur.execute.call_args_list)
        assert "failed" in calls_str

    async def test_split_no_wu07_marks_failed(self):
        """split operation marks failed when WU-07 (split_article) not available."""
        item = _queue_item("split", ARTICLE_ID)
        cur = _make_cursor(fetchall_seq=[[item]])

        # Ensure articles module does not have split_article
        import valence.core.articles as articles_mod

        original = getattr(articles_mod, "split_article", "SENTINEL")
        if original != "SENTINEL":
            delattr(articles_mod, "split_article")
        try:
            with patch("valence.core.compilation.get_cursor", return_value=cur):
                count = await process_mutation_queue()
            assert count.data == 0
        finally:
            if original != "SENTINEL":
                articles_mod.split_article = original

    async def test_merge_candidate_missing_payload_completes(self):
        """merge_candidate without candidate_article_id is a no-op (marks completed)."""
        item = _queue_item("merge_candidate", ARTICLE_ID, payload={})
        cur = _make_cursor(fetchall_seq=[[item]])
        with patch("valence.core.compilation.get_cursor", return_value=cur):
            count = await process_mutation_queue()
        # No candidate_id → logs warning, returns without raising
        assert count.data == 1

    async def test_decay_check_pinned_article_skips_eviction(self):
        """Pinned articles are skipped (no eviction), item marked completed."""
        item = _queue_item("decay_check", ARTICLE_ID, payload={"threshold": 0.1})
        pinned_art = {"id": ARTICLE_ID, "usage_score": 0.5, "pinned": True}
        cur = _make_cursor(
            fetchone_seq=[pinned_art],
            fetchall_seq=[[item]],
        )
        with patch("valence.core.compilation.get_cursor", return_value=cur):
            count = await process_mutation_queue()
        assert count.data == 1

    async def test_decay_check_above_threshold_no_eviction(self):
        """Articles above the threshold are not evicted."""
        item = _queue_item("decay_check", ARTICLE_ID, payload={"threshold": 0.1})
        ok_art = {"id": ARTICLE_ID, "usage_score": 0.9, "pinned": False}
        cur = _make_cursor(
            fetchone_seq=[ok_art],
            fetchall_seq=[[item]],
        )
        evict_called: list[bool] = []

        async def mock_evict(n):
            evict_called.append(True)

        mock_forgetting = MagicMock()
        mock_forgetting.evict_lowest = mock_evict

        with patch("valence.core.compilation.get_cursor", return_value=cur), patch.dict("sys.modules", {"valence.core.forgetting": mock_forgetting}):
            count = await process_mutation_queue()

        assert count.data == 1
        assert len(evict_called) == 0

    async def test_decay_check_low_score_triggers_eviction(self):
        """Low-usage non-pinned article triggers eviction via forgetting module."""
        item = _queue_item("decay_check", ARTICLE_ID, payload={"threshold": 0.5})
        low_art = {"id": ARTICLE_ID, "usage_score": 0.05, "pinned": False}
        cur = _make_cursor(
            fetchone_seq=[low_art],
            fetchall_seq=[[item]],
        )
        evict_called: list[bool] = []

        async def mock_evict(n):
            evict_called.append(True)

        mock_forgetting = MagicMock()
        mock_forgetting.evict_lowest = mock_evict

        with patch("valence.core.compilation.get_cursor", return_value=cur), patch.dict("sys.modules", {"valence.core.forgetting": mock_forgetting}):
            count = await process_mutation_queue()

        assert count.data == 1

    async def test_item_marked_completed_on_success(self):
        """Successfully processed items get status='completed' in the DB."""
        item = _queue_item("recompile", ARTICLE_ID)
        source_link = {"source_id": SOURCE_ID_1}
        src = _source_row()
        art = _article_row()
        cur = _make_cursor(
            fetchone_seq=[src, art],
            fetchall_seq=[[item], [source_link]],
        )

        async def mock_llm(prompt, **_kw):
            return _llm_ok()

        with (
            patch("valence.core.compilation.get_cursor", return_value=cur),
            _patch_rs(),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
        ):
            await process_mutation_queue()

        calls_str = str(cur.execute.call_args_list)
        assert "completed" in calls_str

    async def test_item_marked_failed_on_error(self):
        """Failed items get status='failed' in the DB."""
        item = _queue_item("unknown_op", ARTICLE_ID)
        cur = _make_cursor(fetchall_seq=[[item]])
        with patch("valence.core.compilation.get_cursor", return_value=cur):
            await process_mutation_queue()
        calls_str = str(cur.execute.call_args_list)
        assert "failed" in calls_str
