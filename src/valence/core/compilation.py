"""Compilation pipeline for Valence v2 knowledge articles.

Provides LLM-based article compilation from sources, incremental updates,
and a mutation queue processor (DR-6 deferred follow-ups).

Implements WU-06 (C2 compilation pipeline, C4 self-organization).
Updated WU-13: routes through unified InferenceProvider (C11, DR-8, DR-9).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID

from valence.core.db import get_cursor
from valence.core.inference import (
    RELATIONSHIP_ENUM,
    TASK_COMPILE,
    TASK_OUTPUT_SCHEMAS,
    TASK_UPDATE,
    InferenceSchemaError,
    validate_output,
)
from valence.core.inference import (
    provider as _inference_provider,
)
from valence.core.response import ValenceResponse, err, ok

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Backend — thin wrappers around unified inference provider (WU-13).
#
# set_llm_backend / _LLM_BACKEND / _call_llm are preserved for backward
# compatibility.  Internally they delegate to InferenceProvider.
# ---------------------------------------------------------------------------

_LLM_BACKEND: Callable[[str], Any] | None = None


def set_llm_backend(fn: Callable[[str], Any] | None) -> None:
    """Configure the LLM callable used for compilation.

    Delegates to the unified inference provider (WU-13).

    Args:
        fn: A sync or async callable ``(prompt: str) -> str``, or ``None`` to reset.

    Example::

        # In tests:
        set_llm_backend(lambda p: json.dumps({"title": "T", "content": "C", "relationships": []}))
    """
    global _LLM_BACKEND
    _LLM_BACKEND = fn
    _inference_provider.configure(fn)


async def _call_llm(prompt: str, task_type: str = TASK_COMPILE) -> str:
    """Invoke the configured LLM backend via the unified inference provider.

    Args:
        prompt: Prompt string.
        task_type: Which task type to route through (defaults to TASK_COMPILE).

    Returns:
        LLM response string.

    Raises:
        NotImplementedError: If no backend is configured.
    """
    result = await _inference_provider.infer(task_type, prompt)
    if result.degraded:
        raise NotImplementedError(result.error or "No LLM backend configured for compilation. Call valence.core.compilation.set_llm_backend(fn).")
    return result.content


# ---------------------------------------------------------------------------
# Schema helpers — degraded column (WU-13 / DR-9)
# ---------------------------------------------------------------------------

_DEGRADED_SCHEMA_ENSURED = False


def _ensure_degraded_column() -> None:
    """Add ``degraded`` column to articles if not present. Runs once per process."""
    global _DEGRADED_SCHEMA_ENSURED
    if _DEGRADED_SCHEMA_ENSURED:
        return
    try:
        with get_cursor() as cur:
            cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS degraded BOOLEAN DEFAULT FALSE")
    except Exception:
        pass
    _DEGRADED_SCHEMA_ENSURED = True


# ---------------------------------------------------------------------------
# Right-sizing config
# ---------------------------------------------------------------------------

DEFAULT_RIGHT_SIZING: dict[str, int] = {
    "max_tokens": 4000,
    "min_tokens": 200,
    "target_tokens": 2000,
}


def _get_right_sizing() -> dict[str, int]:
    """Read ``right_sizing`` from ``system_config``, falling back to defaults."""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT value FROM system_config WHERE key = 'right_sizing' LIMIT 1")
            row = cur.fetchone()
            if row:
                val = row["value"]
                if isinstance(val, str):
                    val = json.loads(val)
                if isinstance(val, dict):
                    return {**DEFAULT_RIGHT_SIZING, **val}
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read right_sizing config: %s", exc)
    return DEFAULT_RIGHT_SIZING.copy()


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def _count_tokens(content: str) -> int:
    """Approximate token count: word_count × 1.3 (mirrors articles.py)."""
    return max(1, int(len(content.split()) * 1.3))


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_compilation_prompt(sources: list[dict], title_hint: str | None, target_tokens: int) -> str:
    """Build the LLM prompt for compiling multiple sources into a single article."""
    source_blocks = []
    for src in sources:
        sid = src.get("id") or "?"
        title = src.get("title") or "Untitled"
        content = (src.get("content") or "").strip()
        source_blocks.append(f"## Source id={sid}: {title}\n{content}")
    sources_text = "\n\n".join(source_blocks)

    title_line = f'\nThe article title should be: "{title_hint}"' if title_hint else ""

    schema = TASK_OUTPUT_SCHEMAS[TASK_COMPILE]

    return f"""You are compiling a knowledge article from the following sources.{title_line}

Write a coherent, concise article that synthesizes the key information.
Target approximately {target_tokens} tokens (words × 1.3). Stay under {target_tokens + 500} tokens.

For each source, identify its relationship to the compiled article content:
- "originates": source introduced the article's core information
- "confirms": independent source corroborating existing information
- "supersedes": newer source authoritatively replacing older information
- "contradicts": source directly disagreeing with article content
- "contends": source offering a valid alternative viewpoint

SOURCES:
{sources_text}

Respond ONLY with valid JSON matching this exact schema (no markdown fences):
{schema}

Use the actual source id values (shown above) in source_relationships."""


def _build_update_prompt(article: dict, source: dict, target_tokens: int) -> str:
    """Build the LLM prompt for incrementally updating an article with a new source."""
    article_title = article.get("title") or "Untitled Article"
    article_content = (article.get("content") or "").strip()
    source_title = source.get("title") or "New Source"
    source_content = (source.get("content") or "").strip()

    schema = TASK_OUTPUT_SCHEMAS[TASK_UPDATE]

    return f"""You are updating a knowledge article with new source material.

EXISTING ARTICLE: {article_title}
{article_content}

NEW SOURCE: {source_title}
{source_content}

Update the article to incorporate the new information.
Target approximately {target_tokens} tokens.

Identify the relationship of the new source to the article:
- "confirms": source corroborates existing information
- "supersedes": source authoritatively replaces older information
- "contradicts": source directly disagrees with article content
- "contends": source offers a valid alternative viewpoint

Respond ONLY with valid JSON matching this exact schema (no markdown fences):
{schema}"""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_llm_json(response: str, required_keys: list[str]) -> dict:
    """Parse JSON from LLM response, stripping markdown code fences if present.

    Args:
        response: Raw LLM response string.
        required_keys: Keys that must be present in the parsed dict.

    Returns:
        Parsed dict.

    Raises:
        json.JSONDecodeError: If response is not valid JSON.
        ValueError: If required_keys are missing.
    """
    text = response.strip()
    # Strip markdown code fences (```json ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```json or ```) and last ``` if present
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()

    parsed = json.loads(text)
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        raise ValueError(f"LLM response missing required keys: {missing!r}. Got: {list(parsed.keys())!r}")
    return parsed


# ---------------------------------------------------------------------------
# DB row serialization
# ---------------------------------------------------------------------------


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a DB row to a JSON-safe plain dict."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, UUID):
            out[k] = str(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def compile_article(
    source_ids: list[str],
    title_hint: str | None = None,
) -> ValenceResponse:
    """Compile sources into a new article using LLM summarization.

    Fetches source content, invokes the LLM to synthesize an article,
    creates the article row, links all sources with identified relationships,
    and records a 'created' mutation.

    If the LLM is unavailable, falls back to concatenating source content.

    Articles exceeding ``max_tokens`` from system_config.right_sizing get a
    'split' entry queued in mutation_queue (not split inline — DR-6).

    Args:
        source_ids: UUIDs of sources to compile.
        title_hint: Optional title suggestion forwarded to the LLM.

    Returns:
        Dict with ``success``, ``article``, and optionally ``error``.
    """
    if not source_ids:
        return err("At least one source_id is required")

    right_sizing = _get_right_sizing()
    target_tokens = right_sizing["target_tokens"]
    max_tokens = right_sizing["max_tokens"]

    # ---- Fetch all sources ----
    sources: list[dict] = []
    with get_cursor() as cur:
        for sid in source_ids:
            cur.execute(
                "SELECT id, type, title, url, content, reliability FROM sources WHERE id = %s",
                (sid,),
            )
            row = cur.fetchone()
            if row is None:
                return err(f"Source not found: {sid}")
            src = dict(row)
            src["id"] = str(src["id"])
            sources.append(src)

    # ---- LLM compilation ----
    prompt = _build_compilation_prompt(sources, title_hint, target_tokens)
    is_degraded = False
    try:
        llm_response = await _call_llm(prompt, task_type=TASK_COMPILE)
        parsed = validate_output(TASK_COMPILE, llm_response)
    except (NotImplementedError, InferenceSchemaError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("LLM compilation unavailable (%s), using fallback concatenation", exc)
        is_degraded = True
        content_parts = [f"## {s.get('title') or 'Source'}\n{s.get('content', '')}" for s in sources]
        parsed = {
            "title": title_hint or (sources[0].get("title") if sources else "Compiled Article"),
            "content": "\n\n".join(content_parts),
            "source_relationships": [{"source_id": s["id"], "relationship": "originates"} for s in sources],
        }

    article_content: str = parsed.get("content") or ""
    article_title: str | None = parsed.get("title") or title_hint

    # Map source_id → relationship from source_relationships list
    relationships_by_source_id: dict[str, str] = {}
    for rel_item in parsed.get("source_relationships", []):
        sid = rel_item.get("source_id")
        rel = rel_item.get("relationship", "originates")
        if rel not in RELATIONSHIP_ENUM:
            rel = "originates"
        if sid:
            relationships_by_source_id[str(sid)] = rel

    token_count = _count_tokens(article_content)

    # ---- Create article, link sources, record mutation ----
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO articles
                (content, title, author_type, domain_path, size_tokens, confidence,
                 content_hash, compiled_at)
            VALUES (%s, %s, 'system', '{}', %s, %s::jsonb, md5(%s), NOW())
            RETURNING *
            """,
            (
                article_content,
                article_title,
                token_count,
                json.dumps({"overall": 0.7}),
                article_content,
            ),
        )
        row = cur.fetchone()
        article = _serialize_row(dict(row))
        article_id = article["id"]

        # Link each source with its identified relationship
        for src in sources:
            rel = relationships_by_source_id.get(src["id"], "originates")
            try:
                cur.execute(
                    """
                    INSERT INTO article_sources (article_id, source_id, relationship)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (article_id, source_id, relationship) DO NOTHING
                    """,
                    (article_id, src["id"], rel),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to link source %s to article %s: %s", src["id"], article_id, exc)

        # Record 'created' mutation
        cur.execute(
            """
            INSERT INTO article_mutations (mutation_type, article_id, summary)
            VALUES ('created', %s, %s)
            """,
            (article_id, f"Compiled from {len(sources)} source(s) via LLM"),
        )

        # Queue split if article exceeds max_tokens (DR-6: never split inline)
        if token_count > max_tokens:
            cur.execute(
                """
                INSERT INTO mutation_queue (operation, article_id, priority, payload)
                VALUES ('split', %s, 3, %s::jsonb)
                """,
                (
                    article_id,
                    json.dumps({"reason": "exceeds_max_tokens", "token_count": token_count}),
                ),
            )
            logger.info("Article %s queued for split (tokens=%d > max=%d)", article_id, token_count, max_tokens)

        # DR-9: Mark degraded articles and queue for reprocessing (WU-13).
        if is_degraded:
            try:
                cur.execute(
                    "UPDATE articles SET degraded = TRUE WHERE id = %s",
                    (article_id,),
                )
                cur.execute(
                    """
                    INSERT INTO mutation_queue (operation, article_id, priority, payload)
                    VALUES ('recompile_degraded', %s, 5, %s::jsonb)
                    """,
                    (
                        article_id,
                        json.dumps({"reason": "inference_unavailable_at_compile_time"}),
                    ),
                )
                logger.info(
                    "Article %s marked degraded; queued for recompile when inference available",
                    article_id,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to mark article %s as degraded: %s", article_id, exc)

    return ok(data=article, degraded=is_degraded)


async def update_article_from_source(
    article_id: str,
    source_id: str,
) -> ValenceResponse:
    """Incrementally update an existing article with new source material.

    Fetches the article and source, invokes the LLM to produce updated content
    and identify the relationship type, then records an 'updated' mutation.

    If the LLM is unavailable, falls back to appending source content.

    Args:
        article_id: UUID of the article to update.
        source_id: UUID of the new source to incorporate.

    Returns:
        Dict with ``success``, ``article``, ``relationship``, and optionally ``error``.
    """
    right_sizing = _get_right_sizing()
    target_tokens = right_sizing["target_tokens"]
    max_tokens = right_sizing["max_tokens"]

    # ---- Fetch article and source ----
    with get_cursor() as cur:
        cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
        row = cur.fetchone()
        if not row:
            return err(f"Article not found: {article_id}")
        article = _serialize_row(dict(row))

        cur.execute(
            "SELECT id, type, title, url, content, reliability FROM sources WHERE id = %s",
            (source_id,),
        )
        src_row = cur.fetchone()
        if not src_row:
            return err(f"Source not found: {source_id}")
        source = dict(src_row)
        source["id"] = str(source["id"])

    # ---- LLM update ----
    prompt = _build_update_prompt(article, source, target_tokens)
    is_degraded = False
    try:
        llm_response = await _call_llm(prompt, task_type=TASK_UPDATE)
        parsed = validate_output(TASK_UPDATE, llm_response)
        relationship: str = parsed.get("relationship", "confirms")
    except (NotImplementedError, InferenceSchemaError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("LLM update unavailable (%s), using fallback append", exc)
        is_degraded = True
        existing_content = article.get("content") or ""
        src_title = source.get("title") or "New Source"
        src_content = (source.get("content") or "").strip()
        parsed = {
            "content": f"{existing_content}\n\n## {src_title}\n{src_content}",
            "relationship": "confirms",
            "changes_summary": "Source appended (LLM unavailable)",
        }
        relationship = "confirms"

    new_content: str = parsed.get("content") or ""
    summary: str = parsed.get("changes_summary") or "Article updated with new source"
    token_count = _count_tokens(new_content)

    # ---- Update article, link source, record mutation ----
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE articles
            SET content      = %s,
                size_tokens  = %s,
                content_hash = md5(%s),
                version      = version + 1,
                modified_at  = NOW(),
                compiled_at  = NOW()
            WHERE id = %s
            RETURNING *
            """,
            (new_content, token_count, new_content, article_id),
        )
        row = cur.fetchone()
        if not row:
            return err(f"Article not found after update: {article_id}")
        updated_article = _serialize_row(dict(row))

        # Link source with identified relationship
        try:
            cur.execute(
                """
                INSERT INTO article_sources (article_id, source_id, relationship)
                VALUES (%s, %s, %s)
                ON CONFLICT (article_id, source_id, relationship) DO NOTHING
                """,
                (article_id, source_id, relationship),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to link source %s to article %s: %s", source_id, article_id, exc)

        # Record 'updated' mutation
        cur.execute(
            """
            INSERT INTO article_mutations (mutation_type, article_id, trigger_source_id, summary)
            VALUES ('updated', %s, %s, %s)
            """,
            (article_id, source_id, summary),
        )

        # Queue split if now over max_tokens
        if token_count > max_tokens:
            cur.execute(
                """
                INSERT INTO mutation_queue (operation, article_id, priority, payload)
                VALUES ('split', %s, 3, %s::jsonb)
                """,
                (
                    article_id,
                    json.dumps({"reason": "exceeds_max_tokens", "token_count": token_count}),
                ),
            )
            logger.info(
                "Article %s queued for split after update (tokens=%d > max=%d)",
                article_id,
                token_count,
                max_tokens,
            )

        # DR-9: Mark degraded articles and queue for reprocessing (WU-13).
        if is_degraded:
            try:
                cur.execute(
                    "UPDATE articles SET degraded = TRUE WHERE id = %s",
                    (article_id,),
                )
                cur.execute(
                    """
                    INSERT INTO mutation_queue (operation, article_id, priority, payload)
                    VALUES ('recompile_degraded', %s, 5, %s::jsonb)
                    """,
                    (
                        article_id,
                        json.dumps({"reason": "inference_unavailable_at_update_time"}),
                    ),
                )
                logger.info("Article %s update marked degraded; queued for recompile", article_id)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to mark article %s degraded after update: %s", article_id, exc)

    return ok(data={"article": updated_article, "relationship": relationship}, degraded=is_degraded)


async def process_mutation_queue(batch_size: int = 10) -> ValenceResponse:
    """Process pending mutations from the queue.

    Claims a batch of 'pending' items by setting status to 'processing',
    processes each one atomically (DR-6: one-level atomic), then marks
    'completed' or 'failed'.

    Handles:
      - ``recompile``: Re-run compilation from the article's existing sources.
      - ``split``: Delegates to WU-07 ``articles.split_article`` (raises if unavailable).
      - ``merge_candidate``: Delegates to WU-07 ``articles.merge_articles``.
      - ``decay_check``: Checks usage_score; triggers eviction via forgetting module.

    Args:
        batch_size: Maximum items to claim and process per call.

    Returns:
        Count of items successfully processed (marked 'completed').
    """
    # ---- Claim a batch ----
    with get_cursor() as cur:
        cur.execute(
            """
            WITH claimed AS (
                SELECT id FROM mutation_queue
                WHERE status = 'pending'
                ORDER BY priority ASC, created_at ASC
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            )
            UPDATE mutation_queue mq
            SET status = 'processing'
            FROM claimed
            WHERE mq.id = claimed.id
            RETURNING mq.id, mq.operation, mq.article_id, mq.payload
            """,
            (batch_size,),
        )
        items = [dict(row) for row in cur.fetchall()]

    processed = 0
    for item in items:
        item_id = str(item["id"])
        operation = item["operation"]
        article_id = str(item["article_id"])
        payload = item.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}

        try:
            await _process_mutation_item(operation, article_id, payload)
            _set_queue_item_status(item_id, "completed")
            processed += 1
        except Exception as exc:
            logger.error(
                "Mutation queue item %s (op=%s, article=%s) failed: %s",
                item_id,
                operation,
                article_id,
                exc,
            )
            _set_queue_item_status(item_id, "failed", error=str(exc))

    return ok(data=processed)


async def _process_mutation_item(operation: str, article_id: str, payload: dict) -> None:
    """Execute a single mutation queue item. Raises on failure."""

    if operation in ("recompile", "recompile_degraded"):
        # For recompile_degraded: skip if inference still not available (DR-9).
        if operation == "recompile_degraded" and not _inference_provider.available:
            logger.info(
                "recompile_degraded: inference still unavailable for article %s; skipping",
                article_id,
            )
            return  # Leave pending; retried on next queue run

        with get_cursor() as cur:
            cur.execute(
                "SELECT source_id FROM article_sources WHERE article_id = %s ORDER BY added_at",
                (article_id,),
            )
            rows = cur.fetchall()
        source_ids = [str(r["source_id"]) for r in rows]
        if source_ids:
            result = await compile_article(source_ids)
            if not result.success:
                raise RuntimeError(f"recompile failed: {result.error}")
            # Clear degraded flag if this recompile succeeded without degradation
            if not result.degraded:
                with get_cursor() as cur:
                    cur.execute(
                        "UPDATE articles SET degraded = FALSE WHERE id = %s",
                        (article_id,),
                    )
            _article = result.data
            logger.info(
                "Recompile queued for article %s → new article %s",
                article_id,
                (_article or {}).get("id", "?"),
            )
        else:
            logger.warning("recompile: no sources found for article %s", article_id)

    elif operation == "split":
        # WU-07 implements the actual split; try to call it if available
        try:
            from valence.core import articles as _articles  # noqa: PLC0415

            split_fn = getattr(_articles, "split_article", None)
            if split_fn is None:
                raise NotImplementedError("split_article not yet implemented (WU-07 pending)")
            result = split_fn(article_id)
            if asyncio.iscoroutine(result):
                await result
            logger.info("Split article %s via WU-07", article_id)
        except ImportError as exc:
            raise NotImplementedError(f"articles module unavailable: {exc}") from exc

    elif operation == "merge_candidate":
        candidate_id = payload.get("candidate_article_id")
        if not candidate_id:
            logger.warning("merge_candidate: missing candidate_article_id in payload for %s", article_id)
            return  # Not a failure — just a no-op
        try:
            from valence.core import articles as _articles  # noqa: PLC0415

            merge_fn = getattr(_articles, "merge_articles", None)
            if merge_fn is None:
                raise NotImplementedError("merge_articles not yet implemented (WU-07 pending)")
            result = merge_fn(article_id, candidate_id)
            if asyncio.iscoroutine(result):
                await result
            logger.info("Merged articles %s + %s via WU-07", article_id, candidate_id)
        except ImportError as exc:
            raise NotImplementedError(f"articles module unavailable: {exc}") from exc

    elif operation == "decay_check":
        with get_cursor() as cur:
            cur.execute(
                "SELECT id, usage_score, pinned FROM articles WHERE id = %s",
                (article_id,),
            )
            row = cur.fetchone()
        if not row:
            logger.warning("decay_check: article %s not found", article_id)
            return
        article_row = dict(row)
        if article_row.get("pinned"):
            logger.debug("decay_check: article %s is pinned, skipping eviction", article_id)
            return
        usage_score = float(article_row.get("usage_score") or 0)
        threshold = float(payload.get("threshold", 0.1))
        if usage_score < threshold:
            try:
                from valence.core import forgetting as _forgetting  # noqa: PLC0415

                evict_fn = getattr(_forgetting, "evict_lowest", None)
                if evict_fn:
                    result = evict_fn(1)
                    if asyncio.iscoroutine(result):
                        await result
                    logger.info(
                        "Evicted low-usage article %s (score=%.3f < threshold=%.3f)",
                        article_id,
                        usage_score,
                        threshold,
                    )
            except ImportError:
                logger.warning("forgetting module not available for decay_check on %s", article_id)
        else:
            logger.debug(
                "decay_check: article %s usage_score=%.3f >= threshold=%.3f, no eviction",
                article_id,
                usage_score,
                threshold,
            )

    else:
        raise ValueError(f"Unknown mutation operation: {operation!r}")


def _set_queue_item_status(item_id: str, status: str, error: str | None = None) -> None:
    """Persist the final status of a processed mutation queue item."""
    with get_cursor() as cur:
        if error:
            cur.execute(
                """
                UPDATE mutation_queue
                SET status = %s,
                    processed_at = NOW(),
                    payload = payload || %s::jsonb
                WHERE id = %s
                """,
                (status, json.dumps({"error": error[:500]}), item_id),
            )
        else:
            cur.execute(
                "UPDATE mutation_queue SET status = %s, processed_at = NOW() WHERE id = %s",
                (status, item_id),
            )
