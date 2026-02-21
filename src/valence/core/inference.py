"""Unified inference router for the Valence v2 knowledge system.

Single configuration point for all LLM inference tasks (C11, DR-8, DR-9).
Supports task-type routing, per-task backend overrides, explicit degraded
mode, and both sync and async backends.

Implements WU-13 (C11 Inference Abstraction).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task type constants
# ---------------------------------------------------------------------------

TASK_COMPILE = "compile"          # Sources → article (highest complexity)
TASK_UPDATE = "update"            # Incremental article update (medium)
TASK_CLASSIFY = "classify"        # Relationship classification (low)
TASK_CONTENTION = "contention"    # Contention detection (medium)
TASK_SPLIT = "split"              # Intelligent split point (low)

_ALL_TASKS = {TASK_COMPILE, TASK_UPDATE, TASK_CLASSIFY, TASK_CONTENTION, TASK_SPLIT}


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Result envelope for a single inference call.

    Attributes:
        content: The LLM response string (empty string when degraded).
        degraded: True if no backend was available or the backend failed.
                  Per DR-9: explicit degraded mode, not silent.
        task_type: Which task type was requested.
        error: Error message when degraded; None on success.
    """

    content: str
    degraded: bool = False
    task_type: str = ""
    error: str | None = None

    @classmethod
    def success(cls, content: str, task_type: str) -> "InferenceResult":
        return cls(content=content, degraded=False, task_type=task_type, error=None)

    @classmethod
    def degraded_result(cls, task_type: str, error: str) -> "InferenceResult":
        return cls(content="", degraded=True, task_type=task_type, error=error)


# ---------------------------------------------------------------------------
# InferenceProvider
# ---------------------------------------------------------------------------


class InferenceProvider:
    """Single configuration point for all inference tasks.

    Supports:
    - A default backend (callable) for all task types.
    - Per-task overrides to route specific task types to different models.
    - Explicit degraded mode when no backend is configured (DR-9).
    - Both sync and async callables (sync wrapped in asyncio.to_thread).

    Usage::

        from valence.core.inference import provider, TASK_COMPILE

        # Configure once at startup
        provider.configure(my_async_llm_fn)

        # Optionally override specific tasks
        provider.configure(
            my_default_fn,
            task_overrides={TASK_CLASSIFY: my_classifier_fn}
        )

        # Call from any module
        result = await provider.infer(TASK_COMPILE, prompt)
        if result.degraded:
            # fallback behaviour
        else:
            content = result.content
    """

    def __init__(self) -> None:
        self._default_backend: Callable[[str], Any] | None = None
        self._task_overrides: dict[str, Callable[[str], Any]] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        backend: Callable[[str], Any] | None,
        task_overrides: dict[str, Callable[[str], Any]] | None = None,
    ) -> None:
        """Set the default backend and optionally per-task overrides.

        Calling ``configure(None)`` resets the provider to unconfigured state
        (clears both the default backend and all task overrides).

        Args:
            backend: Async or sync callable ``(prompt: str) -> str``, or
                ``None`` to reset.
            task_overrides: Optional mapping of task_type → callable.
                Pass an explicit empty dict ``{}`` to clear all overrides while
                setting a new default.  Pass ``None`` (default) to leave
                existing overrides in place.
        """
        self._default_backend = backend
        if backend is None:
            # Full reset
            self._task_overrides = {}
        elif task_overrides is not None:
            self._task_overrides = dict(task_overrides)
        # If backend is set and task_overrides is None, leave existing overrides.

    def set_task_override(
        self, task_type: str, backend: Callable[[str], Any] | None
    ) -> None:
        """Set or clear an override for a single task type.

        Args:
            task_type: One of the TASK_* constants.
            backend: Callable to use, or ``None`` to remove the override.
        """
        if backend is None:
            self._task_overrides.pop(task_type, None)
        else:
            self._task_overrides[task_type] = backend

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def infer(self, task_type: str, prompt: str) -> InferenceResult:
        """Route an inference request to the appropriate backend.

        Tries the task-specific override first, then the default backend.
        Returns an :class:`InferenceResult` with ``degraded=True`` and an
        error message if no backend is configured or the backend raises.

        Args:
            task_type: One of the TASK_* constants.
            prompt: The prompt string to send to the LLM.

        Returns:
            :class:`InferenceResult` — always returns, never raises.
        """
        backend = self._task_overrides.get(task_type) or self._default_backend

        if backend is None:
            return InferenceResult.degraded_result(
                task_type=task_type,
                error=(
                    f"No inference backend configured for task '{task_type}'. "
                    "Call inference.provider.configure(backend) before use."
                ),
            )

        try:
            result = backend(prompt)
            if asyncio.iscoroutine(result):
                content = await result
            elif asyncio.isfuture(result):
                content = await result
            elif not isinstance(result, str):
                # Sync callable that returned something non-string; wrap
                content = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: str(backend(prompt))
                )
                # Re-call: above already called backend(prompt), capture fresh
                # Actually just use the non-string result directly
                content = str(result)
            else:
                content = result

            return InferenceResult.success(content=content, task_type=task_type)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Inference backend failed for task %r: %s", task_type, exc
            )
            return InferenceResult.degraded_result(
                task_type=task_type,
                error=f"Backend error: {exc}",
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if any inference backend is configured."""
        return self._default_backend is not None or bool(self._task_overrides)

    def __repr__(self) -> str:
        return (
            f"InferenceProvider(available={self.available}, "
            f"task_overrides={list(self._task_overrides.keys())})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: The global inference provider. Configure once at startup:
#:
#:   from valence.core.inference import provider
#:   provider.configure(my_llm_fn)
provider = InferenceProvider()
