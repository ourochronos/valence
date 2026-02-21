"""Tests for valence.core.inference — unified inference router (WU-13).

Tests the InferenceProvider, InferenceResult, task-type routing,
per-task overrides, degraded mode, and module-level singleton.

asyncio_mode = auto (pyproject.toml), no @pytest.mark.asyncio needed.
"""

from __future__ import annotations

import asyncio
import pytest

from valence.core.inference import (
    TASK_CLASSIFY,
    TASK_COMPILE,
    TASK_CONTENTION,
    TASK_SPLIT,
    TASK_UPDATE,
    InferenceProvider,
    InferenceResult,
    provider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_provider():
    """Reset the module-level singleton after each test for isolation."""
    yield
    provider.configure(None)


def _sync_backend(response: str):
    """Return a sync callable that returns response."""
    return lambda prompt: response


def _async_backend(response: str):
    """Return an async callable that returns response."""
    async def _fn(prompt: str) -> str:
        return response
    return _fn


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------


class TestInferenceResult:
    def test_success_factory(self):
        result = InferenceResult.success(content="Hello", task_type=TASK_COMPILE)
        assert result.content == "Hello"
        assert result.degraded is False
        assert result.task_type == TASK_COMPILE
        assert result.error is None

    def test_degraded_factory(self):
        result = InferenceResult.degraded_result(task_type=TASK_UPDATE, error="No backend")
        assert result.content == ""
        assert result.degraded is True
        assert result.task_type == TASK_UPDATE
        assert result.error == "No backend"

    def test_dataclass_fields_accessible(self):
        result = InferenceResult(content="x", degraded=False, task_type=TASK_SPLIT, error=None)
        assert result.content == "x"
        assert result.degraded is False
        assert result.task_type == TASK_SPLIT
        assert result.error is None

    def test_degraded_defaults_false(self):
        result = InferenceResult(content="y", task_type=TASK_CLASSIFY)
        assert result.degraded is False

    def test_error_defaults_none(self):
        result = InferenceResult(content="y", task_type=TASK_CLASSIFY)
        assert result.error is None


# ---------------------------------------------------------------------------
# InferenceProvider — configuration
# ---------------------------------------------------------------------------


class TestInferenceProviderConfigure:
    def test_configure_sets_default_backend(self):
        p = InferenceProvider()
        fn = _sync_backend("ok")
        p.configure(fn)
        assert p._default_backend is fn

    def test_configure_none_resets_backend(self):
        p = InferenceProvider()
        p.configure(_sync_backend("ok"))
        p.configure(None)
        assert p._default_backend is None

    def test_configure_none_clears_task_overrides(self):
        p = InferenceProvider()
        p.configure(_sync_backend("ok"), task_overrides={TASK_COMPILE: _sync_backend("compile")})
        p.configure(None)
        assert p._task_overrides == {}

    def test_configure_with_task_overrides(self):
        p = InferenceProvider()
        default_fn = _sync_backend("default")
        compile_fn = _sync_backend("compile-specific")
        p.configure(default_fn, task_overrides={TASK_COMPILE: compile_fn})
        assert p._default_backend is default_fn
        assert p._task_overrides[TASK_COMPILE] is compile_fn

    def test_configure_empty_overrides_clears_existing(self):
        p = InferenceProvider()
        p.configure(_sync_backend("ok"), task_overrides={TASK_COMPILE: _sync_backend("c")})
        p.configure(_sync_backend("new"), task_overrides={})
        assert p._task_overrides == {}

    def test_configure_none_overrides_leaves_existing(self):
        """configure(fn, task_overrides=None) does NOT clear existing overrides."""
        p = InferenceProvider()
        compile_fn = _sync_backend("compile")
        p.configure(_sync_backend("default"), task_overrides={TASK_COMPILE: compile_fn})
        p.configure(_sync_backend("new"))  # task_overrides=None → leave existing
        assert p._task_overrides[TASK_COMPILE] is compile_fn

    def test_set_task_override_adds(self):
        p = InferenceProvider()
        fn = _sync_backend("classify")
        p.set_task_override(TASK_CLASSIFY, fn)
        assert p._task_overrides[TASK_CLASSIFY] is fn

    def test_set_task_override_none_removes(self):
        p = InferenceProvider()
        p.set_task_override(TASK_CLASSIFY, _sync_backend("c"))
        p.set_task_override(TASK_CLASSIFY, None)
        assert TASK_CLASSIFY not in p._task_overrides


# ---------------------------------------------------------------------------
# InferenceProvider — available property
# ---------------------------------------------------------------------------


class TestInferenceProviderAvailable:
    def test_false_when_unconfigured(self):
        p = InferenceProvider()
        assert p.available is False

    def test_true_when_default_backend_set(self):
        p = InferenceProvider()
        p.configure(_sync_backend("ok"))
        assert p.available is True

    def test_true_when_only_task_overrides_set(self):
        p = InferenceProvider()
        p.set_task_override(TASK_COMPILE, _sync_backend("compile"))
        assert p.available is True

    def test_false_after_configure_none(self):
        p = InferenceProvider()
        p.configure(_sync_backend("ok"))
        p.configure(None)
        assert p.available is False

    def test_module_provider_available_after_configure(self):
        provider.configure(_sync_backend("ok"))
        assert provider.available is True

    def test_module_provider_not_available_after_reset(self):
        provider.configure(None)
        assert provider.available is False


# ---------------------------------------------------------------------------
# InferenceProvider — degraded mode (no backend)
# ---------------------------------------------------------------------------


class TestDegradedMode:
    async def test_degraded_when_no_backend(self):
        p = InferenceProvider()
        result = await p.infer(TASK_COMPILE, "compile this")
        assert result.degraded is True
        assert result.content == ""
        assert result.task_type == TASK_COMPILE

    async def test_degraded_error_mentions_task(self):
        p = InferenceProvider()
        result = await p.infer(TASK_COMPILE, "prompt")
        assert "compile" in result.error.lower() or "backend" in result.error.lower()

    async def test_degraded_for_each_task_type(self):
        p = InferenceProvider()
        for task in (TASK_COMPILE, TASK_UPDATE, TASK_CLASSIFY, TASK_CONTENTION, TASK_SPLIT):
            result = await p.infer(task, "prompt")
            assert result.degraded is True, f"Expected degraded for task {task!r}"
            assert result.task_type == task

    async def test_degraded_when_backend_raises(self):
        def failing_backend(prompt: str) -> str:
            raise RuntimeError("Backend exploded")

        p = InferenceProvider()
        p.configure(failing_backend)
        result = await p.infer(TASK_COMPILE, "prompt")
        assert result.degraded is True
        assert "Backend exploded" in (result.error or "")

    async def test_module_provider_degraded_when_unconfigured(self):
        result = await provider.infer(TASK_COMPILE, "test")
        assert result.degraded is True


# ---------------------------------------------------------------------------
# InferenceProvider — default backend routing
# ---------------------------------------------------------------------------


class TestDefaultBackendRouting:
    async def test_sync_backend_works(self):
        p = InferenceProvider()
        p.configure(_sync_backend("sync response"))
        result = await p.infer(TASK_COMPILE, "prompt")
        assert result.degraded is False
        assert result.content == "sync response"
        assert result.task_type == TASK_COMPILE

    async def test_async_backend_works(self):
        p = InferenceProvider()
        p.configure(_async_backend("async response"))
        result = await p.infer(TASK_COMPILE, "prompt")
        assert result.degraded is False
        assert result.content == "async response"

    async def test_default_backend_used_for_all_task_types(self):
        p = InferenceProvider()
        p.configure(_sync_backend("universal"))
        for task in (TASK_COMPILE, TASK_UPDATE, TASK_CLASSIFY, TASK_CONTENTION, TASK_SPLIT):
            result = await p.infer(task, "prompt")
            assert result.content == "universal", f"Failed for task {task!r}"
            assert result.degraded is False

    async def test_prompt_forwarded_to_backend(self):
        received: list[str] = []

        def capturing_backend(prompt: str) -> str:
            received.append(prompt)
            return "response"

        p = InferenceProvider()
        p.configure(capturing_backend)
        await p.infer(TASK_COMPILE, "the exact prompt")
        assert len(received) == 1
        assert received[0] == "the exact prompt"

    async def test_result_task_type_matches_requested(self):
        p = InferenceProvider()
        p.configure(_sync_backend("ok"))
        result = await p.infer(TASK_CONTENTION, "prompt")
        assert result.task_type == TASK_CONTENTION

    async def test_module_provider_routes_correctly(self):
        provider.configure(_sync_backend("module-level"))
        result = await provider.infer(TASK_COMPILE, "prompt")
        assert result.content == "module-level"
        assert not result.degraded


# ---------------------------------------------------------------------------
# InferenceProvider — per-task override routing
# ---------------------------------------------------------------------------


class TestTaskOverrideRouting:
    async def test_override_used_for_specific_task(self):
        p = InferenceProvider()
        p.configure(_sync_backend("default"))
        p.set_task_override(TASK_CLASSIFY, _sync_backend("classifier-response"))

        result = await p.infer(TASK_CLASSIFY, "classify this")
        assert result.content == "classifier-response"

    async def test_default_used_for_other_tasks(self):
        p = InferenceProvider()
        p.configure(_sync_backend("default"))
        p.set_task_override(TASK_CLASSIFY, _sync_backend("classifier-response"))

        result = await p.infer(TASK_COMPILE, "compile this")
        assert result.content == "default"

    async def test_override_takes_priority_over_default(self):
        p = InferenceProvider()
        p.configure(_sync_backend("default"))
        p.configure(_sync_backend("still-default"), task_overrides={TASK_COMPILE: _sync_backend("compile-override")})

        result = await p.infer(TASK_COMPILE, "prompt")
        assert result.content == "compile-override"

    async def test_multiple_overrides_each_route_correctly(self):
        p = InferenceProvider()
        p.configure(
            _sync_backend("default"),
            task_overrides={
                TASK_COMPILE: _sync_backend("compile-model"),
                TASK_CLASSIFY: _sync_backend("classify-model"),
            },
        )

        compile_result = await p.infer(TASK_COMPILE, "p")
        classify_result = await p.infer(TASK_CLASSIFY, "p")
        update_result = await p.infer(TASK_UPDATE, "p")

        assert compile_result.content == "compile-model"
        assert classify_result.content == "classify-model"
        assert update_result.content == "default"

    async def test_override_without_default_works(self):
        """Task overrides work even when no default backend is set."""
        p = InferenceProvider()
        p.set_task_override(TASK_COMPILE, _sync_backend("compile-only"))

        compile_result = await p.infer(TASK_COMPILE, "p")
        update_result = await p.infer(TASK_UPDATE, "p")  # no override, no default

        assert compile_result.content == "compile-only"
        assert update_result.degraded is True  # no backend for this task

    async def test_clearing_override_falls_back_to_default(self):
        p = InferenceProvider()
        p.configure(_sync_backend("default"))
        p.set_task_override(TASK_CLASSIFY, _sync_backend("classifier"))
        p.set_task_override(TASK_CLASSIFY, None)  # clear override

        result = await p.infer(TASK_CLASSIFY, "p")
        assert result.content == "default"

    async def test_configure_with_overrides_via_configure_method(self):
        p = InferenceProvider()
        p.configure(
            _sync_backend("default"),
            task_overrides={TASK_CONTENTION: _sync_backend("contention-model")},
        )
        result = await p.infer(TASK_CONTENTION, "p")
        assert result.content == "contention-model"


# ---------------------------------------------------------------------------
# InferenceProvider — async backend edge cases
# ---------------------------------------------------------------------------


class TestAsyncBackendEdgeCases:
    async def test_async_backend_prompt_forwarded(self):
        received: list[str] = []

        async def async_fn(prompt: str) -> str:
            received.append(prompt)
            return "async ok"

        p = InferenceProvider()
        p.configure(async_fn)
        await p.infer(TASK_COMPILE, "my async prompt")
        assert received[0] == "my async prompt"

    async def test_async_backend_exception_yields_degraded(self):
        async def failing_async(prompt: str) -> str:
            raise ValueError("async failure")

        p = InferenceProvider()
        p.configure(failing_async)
        result = await p.infer(TASK_COMPILE, "p")
        assert result.degraded is True
        assert "async failure" in (result.error or "")

    async def test_async_override_works(self):
        p = InferenceProvider()
        p.configure(_sync_backend("default"))
        p.set_task_override(TASK_SPLIT, _async_backend("async-split"))
        result = await p.infer(TASK_SPLIT, "p")
        assert result.content == "async-split"
        assert not result.degraded


# ---------------------------------------------------------------------------
# Task type constants
# ---------------------------------------------------------------------------


class TestTaskConstants:
    def test_all_task_types_are_strings(self):
        for task in (TASK_COMPILE, TASK_UPDATE, TASK_CLASSIFY, TASK_CONTENTION, TASK_SPLIT):
            assert isinstance(task, str)

    def test_task_types_are_unique(self):
        tasks = [TASK_COMPILE, TASK_UPDATE, TASK_CLASSIFY, TASK_CONTENTION, TASK_SPLIT]
        assert len(set(tasks)) == len(tasks)

    def test_compile_is_compile(self):
        assert TASK_COMPILE == "compile"

    def test_contention_is_contention(self):
        assert TASK_CONTENTION == "contention"
