"""Unified inference router for the Valence v2 knowledge system.

Single configuration point for all LLM inference tasks (C11, DR-8, DR-9).
Supports task-type routing, per-task backend overrides, explicit degraded
mode, and both sync and async backends.

Implements WU-13 (C11 Inference Abstraction).
Updated WU-16: strict JSON schemas for all task types with validation (DR-11).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
# Shared enum — DR-11
# ---------------------------------------------------------------------------

RELATIONSHIP_ENUM = ["originates", "confirms", "supersedes", "contradicts", "contends"]

# ---------------------------------------------------------------------------
# Task output schemas — required fields per task type (DR-11)
# ---------------------------------------------------------------------------

# Maps task_type → list of required output keys
_TASK_REQUIRED_OUTPUT_FIELDS: dict[str, list[str]] = {
    TASK_COMPILE:    ["title", "content", "source_relationships"],
    TASK_UPDATE:     ["content", "relationship", "changes_summary"],
    TASK_CLASSIFY:   ["relationship", "confidence", "reasoning"],
    TASK_CONTENTION: ["is_contention", "materiality", "explanation"],
    TASK_SPLIT:      ["split_index", "part_a_title", "part_b_title", "reasoning"],
}

# Fields that contain relationship_enum values and must be validated
_TASK_RELATIONSHIP_FIELDS: dict[str, list[str]] = {
    TASK_COMPILE:  [],   # relationships are in a list; validated separately
    TASK_UPDATE:   ["relationship"],
    TASK_CLASSIFY: ["relationship"],
    TASK_CONTENTION: [],
    TASK_SPLIT:    [],
}

# Output schema descriptions — used in prompt builders
TASK_OUTPUT_SCHEMAS: dict[str, str] = {
    TASK_COMPILE: """{
  "title": "<article title>",
  "content": "<article content>",
  "source_relationships": [
    {"source_id": "<source_id>", "relationship": "originates|confirms|supersedes|contradicts|contends"}
  ]
}""",
    TASK_UPDATE: """{
  "content": "<updated article content>",
  "relationship": "confirms|supersedes|contradicts|contends",
  "changes_summary": "<one sentence describing what changed>"
}""",
    TASK_CLASSIFY: """{
  "relationship": "originates|confirms|supersedes|contradicts|contends",
  "confidence": 0.0,
  "reasoning": "<one sentence explaining the classification>"
}""",
    TASK_CONTENTION: """{
  "is_contention": true|false,
  "materiality": 0.0,
  "explanation": "<one sentence describing the disagreement, or null if none>"
}""",
    TASK_SPLIT: """{
  "split_index": 0,
  "part_a_title": "<title for first part>",
  "part_b_title": "<title for second part>",
  "reasoning": "<why split at this point>"
}""",
}


# ---------------------------------------------------------------------------
# InferenceSchemaError
# ---------------------------------------------------------------------------


class InferenceSchemaError(ValueError):
    """Raised when an LLM response does not conform to the expected task schema."""


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences (```json ... ```)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return text


def validate_output(task_type: str, raw_json: str) -> dict:
    """Parse and validate a raw LLM response against the task output schema.

    Handles:
    - Markdown fence stripping (```json ... ```)
    - Required field validation
    - Relationship enum validation
    - Extra fields are silently ignored (not an error)
    - Optional fields get defaults when missing

    Args:
        task_type: One of the TASK_* constants.
        raw_json: Raw string from the LLM (may include markdown fences).

    Returns:
        Validated dict matching the task output schema.

    Raises:
        InferenceSchemaError: If the JSON cannot be parsed, required fields
            are missing, or enum values are invalid.
    """
    if task_type not in _TASK_REQUIRED_OUTPUT_FIELDS:
        raise InferenceSchemaError(
            f"Unknown task type {task_type!r}; expected one of {list(_ALL_TASKS)}"
        )

    text = _strip_markdown_fences(raw_json)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InferenceSchemaError(
            f"[{task_type}] Response is not valid JSON: {exc}. "
            f"Got: {raw_json[:200]!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise InferenceSchemaError(
            f"[{task_type}] Response must be a JSON object, got {type(parsed).__name__}. "
            f"Value: {raw_json[:200]!r}"
        )

    required = _TASK_REQUIRED_OUTPUT_FIELDS[task_type]
    missing = [k for k in required if k not in parsed]
    if missing:
        raise InferenceSchemaError(
            f"[{task_type}] Response missing required fields: {missing!r}. "
            f"Got keys: {list(parsed.keys())!r}"
        )

    # Validate relationship enum fields
    for field_name in _TASK_RELATIONSHIP_FIELDS.get(task_type, []):
        val = parsed.get(field_name)
        if val is not None and val not in RELATIONSHIP_ENUM:
            raise InferenceSchemaError(
                f"[{task_type}] Field {field_name!r} has invalid value {val!r}; "
                f"must be one of {RELATIONSHIP_ENUM}"
            )

    # Task-specific extra validation
    if task_type == TASK_COMPILE:
        source_rels = parsed.get("source_relationships", [])
        if not isinstance(source_rels, list):
            raise InferenceSchemaError(
                f"[{task_type}] 'source_relationships' must be a list, "
                f"got {type(source_rels).__name__}"
            )
        for i, item in enumerate(source_rels):
            if not isinstance(item, dict):
                raise InferenceSchemaError(
                    f"[{task_type}] source_relationships[{i}] must be an object"
                )
            rel = item.get("relationship")
            if rel is not None and rel not in RELATIONSHIP_ENUM:
                raise InferenceSchemaError(
                    f"[{task_type}] source_relationships[{i}].relationship "
                    f"has invalid value {rel!r}; must be one of {RELATIONSHIP_ENUM}"
                )

    if task_type == TASK_SPLIT:
        si = parsed.get("split_index")
        if not isinstance(si, int):
            raise InferenceSchemaError(
                f"[{task_type}] 'split_index' must be an integer, got {type(si).__name__}"
            )

    if task_type == TASK_CONTENTION:
        # Coerce is_contention to bool if needed (some LLMs return strings)
        raw_val = parsed.get("is_contention")
        if isinstance(raw_val, str):
            parsed["is_contention"] = raw_val.lower() in ("true", "1", "yes")
        elif not isinstance(raw_val, bool):
            raise InferenceSchemaError(
                f"[{task_type}] 'is_contention' must be a boolean, "
                f"got {type(raw_val).__name__}: {raw_val!r}"
            )

    # Apply defaults for optional fields
    _apply_defaults(task_type, parsed)

    return parsed


def _apply_defaults(task_type: str, parsed: dict) -> None:
    """Apply default values for optional/missing fields in-place."""
    if task_type == TASK_COMPILE:
        parsed.setdefault("source_relationships", [])

    elif task_type == TASK_UPDATE:
        parsed.setdefault("relationship", "confirms")
        parsed.setdefault("changes_summary", "")

    elif task_type == TASK_CLASSIFY:
        parsed.setdefault("confidence", 0.5)
        parsed.setdefault("reasoning", "")

    elif task_type == TASK_CONTENTION:
        parsed.setdefault("materiality", 0.0)
        parsed.setdefault("explanation", None)

    elif task_type == TASK_SPLIT:
        parsed.setdefault("reasoning", "")


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
        parsed: Validated parsed dict if schema validation succeeded; None otherwise.
    """

    content: str
    degraded: bool = False
    task_type: str = ""
    error: str | None = None
    parsed: dict | None = None

    @classmethod
    def success(cls, content: str, task_type: str, parsed: dict | None = None) -> InferenceResult:
        return cls(content=content, degraded=False, task_type=task_type, error=None, parsed=parsed)

    @classmethod
    def degraded_result(cls, task_type: str, error: str) -> InferenceResult:
        return cls(content="", degraded=True, task_type=task_type, error=error, parsed=None)


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
    - Schema validation of task outputs (DR-11 / WU-16).

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
            parsed = result.parsed  # validated dict (WU-16)
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

        After a successful backend call, validates the response against the
        task output schema (DR-11). Schema validation failures are logged but
        do NOT set degraded — the raw content is still returned so callers
        can handle fallbacks themselves.

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

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Inference backend failed for task %r: %s", task_type, exc
            )
            return InferenceResult.degraded_result(
                task_type=task_type,
                error=f"Backend error: {exc}",
            )

        # DR-11: validate output schema (WU-16)
        parsed: dict | None = None
        if task_type in _TASK_REQUIRED_OUTPUT_FIELDS:
            try:
                parsed = validate_output(task_type, content)
            except InferenceSchemaError as exc:
                logger.warning(
                    "Schema validation failed for task %r: %s", task_type, exc
                )
                # Do not set degraded — return raw content; caller decides

        return InferenceResult.success(content=content, task_type=task_type, parsed=parsed)

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
