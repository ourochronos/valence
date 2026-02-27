# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

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
from typing import Any

from valence.core.confidence import ConfidenceResult, compute_confidence
from valence.core.db import get_cursor, serialize_row
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
    "max_tokens": 800,
    "min_tokens": 300,
    "target_tokens": 550,
}

DEFAULT_PROMPT_LIMITS: dict[str, int] = {
    "max_total_chars": 100_000,  # Maximum total characters in prompt
    "max_source_chars": 50_000,  # Maximum characters per individual source
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


def _get_prompt_limits() -> dict[str, int]:
    """Read ``prompt_limits`` from ``system_config``, falling back to defaults."""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT value FROM system_config WHERE key = 'prompt_limits' LIMIT 1")
            row = cur.fetchone()
            if row:
                val = row["value"]
                if isinstance(val, str):
                    val = json.loads(val)
                if isinstance(val, dict):
                    return {**DEFAULT_PROMPT_LIMITS, **val}
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read prompt_limits config: %s", exc)
    return DEFAULT_PROMPT_LIMITS.copy()


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
    """Build the LLM prompt for compiling multiple sources into a single article.

    Returns structured prompt for LLM compilation task.
    """
    source_blocks = []

    for src in sources:
        sid = src.get("id") or "?"
        title = src.get("title") or "Untitled"
        content = (src.get("content") or "").strip()

        block = f"## Source id={sid}: {title}\n{content}"
        source_blocks.append(block)

    sources_text = "\n\n".join(source_blocks)

    title_line = f'\nThe article title should be: "{title_hint}"' if title_hint else ""

    schema = TASK_OUTPUT_SCHEMAS[TASK_COMPILE]

    return f"""You are compiling knowledge articles from the following sources.{title_line}

Produce ONE OR MORE coherent, concise articles that synthesize the key information.
Each article should target approximately {target_tokens} tokens (words × 1.3).
If the source material covers multiple distinct topics, split into separate articles — one per topic.
If the material is focused on a single topic, produce exactly one article.

IMPORTANT:
- Each article should cover ONE coherent topic.
- Use descriptive titles that reflect each article's actual content.
- Never use generic titles like "part 1", "part 2", or sequential numbering.
- Every source_id must appear in at least one article's source_relationships.
- Classify each article's epistemic_type:
  - "episodic": temporal/event-based — what happened, session logs, meeting notes (decays over time)
  - "semantic": factual/persistent — what is true, config details, architecture decisions (persists until superseded)
  - "procedural": instructional — how to do something, runbooks, setup guides (pinned, updated explicitly)

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


# ---------------------------------------------------------------------------
# Shared helpers for compile / recompile
# ---------------------------------------------------------------------------


def _fetch_sources_by_ids(source_ids: list[str]) -> tuple[list[dict], list[str]]:
    """Fetch sources by ID, preserving input order.

    Returns (sources, missing_ids).
    """
    sources: list[dict] = []
    missing_ids: list[str] = []
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, type, title, url, content, reliability FROM sources WHERE id = ANY(%s::uuid[])",
            (source_ids,),
        )
        rows = cur.fetchall()
        fetched_by_id = {str(row["id"]): dict(row) for row in rows}
        for sid in source_ids:
            if sid in fetched_by_id:
                src = fetched_by_id[sid]
                src["id"] = str(src["id"])
                sources.append(src)
            else:
                missing_ids.append(sid)
    return sources, missing_ids


def _extract_relationships(article_data: dict) -> dict[str, str]:
    """Extract source_id → relationship mapping from LLM article output."""
    rels: dict[str, str] = {}
    for rel_item in article_data.get("source_relationships", []):
        sid = rel_item.get("source_id")
        rel = rel_item.get("relationship", "originates")
        if rel not in RELATIONSHIP_ENUM:
            rel = "originates"
        if sid:
            rels[str(sid)] = rel
    return rels


def _normalize_llm_articles(parsed: dict) -> list[dict]:
    """Normalize LLM output to a list of article dicts.

    Supports both v2 (single article) and v3 (articles array) formats.
    """
    if "articles" in parsed:
        return parsed["articles"]
    return [parsed]


def _create_article_row(
    cur: Any,
    content: str,
    title: str | None,
    sources: list[dict],
    conf: ConfidenceResult,
    epistemic_type: str = "semantic",
) -> dict:
    """INSERT a new article row and return the serialized dict."""
    token_count = _count_tokens(content)
    cur.execute(
        """
        INSERT INTO articles
            (content, title, author_type, domain_path, size_tokens, confidence,
             content_hash, epistemic_type, compiled_at,
             confidence_source, corroboration_count)
        VALUES (%s, %s, 'system', '{}', %s, %s::jsonb, md5(%s), %s, NOW(),
                %s, %s)
        RETURNING *
        """,
        (
            content,
            title,
            token_count,
            json.dumps(conf.to_jsonb()),
            content,
            epistemic_type,
            round(conf.avg_reliability, 3),
            conf.corroboration_count,
        ),
    )
    row = cur.fetchone()
    return serialize_row(dict(row))


def _link_sources(
    cur: Any,
    article_id: str,
    sources: list[dict],
    relationships: dict[str, str],
    *,
    clear_existing: bool = False,
) -> None:
    """Link sources to an article via article_sources."""
    if clear_existing:
        cur.execute("DELETE FROM article_sources WHERE article_id = %s", (article_id,))
    for src in sources:
        rel = relationships.get(src["id"], "originates")
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


def _maybe_queue_split(cur: Any, article_id: str, token_count: int, max_tokens: int) -> None:
    """Queue a split mutation if article exceeds max_tokens."""
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


def _create_extra_articles(
    articles_data: list[dict],
    sources: list[dict],
    conf: ConfidenceResult,
    max_tokens: int,
    mutation_summary_prefix: str = "Created",
) -> list[dict]:
    """Create additional articles from multi-article LLM output (index 1+)."""
    extras = []
    for extra in articles_data[1:]:
        extra_content = extra.get("content") or ""
        extra_title = extra.get("title") or "Untitled"
        extra_rels = _extract_relationships(extra)

        ep_type = extra.get("epistemic_type", "semantic")
        if ep_type not in ("episodic", "semantic", "procedural"):
            ep_type = "semantic"

        with get_cursor() as cur:
            article = _create_article_row(cur, extra_content, extra_title, sources, conf, ep_type)
            _link_sources(cur, article["id"], sources, extra_rels)
            cur.execute(
                "INSERT INTO article_mutations (mutation_type, article_id, summary) VALUES ('created', %s, %s)",
                (article["id"], f"{mutation_summary_prefix} ({len(sources)} source(s))"),
            )
            _maybe_queue_split(cur, article["id"], _count_tokens(extra_content), max_tokens)
            extras.append(article)
    return extras


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
    sources, missing_ids = _fetch_sources_by_ids(source_ids)
    if missing_ids:
        return err(f"Source(s) not found: {', '.join(missing_ids)}")

    # ---- LLM compilation ----
    try:
        prompt = _build_compilation_prompt(sources, title_hint, target_tokens)
    except ValueError as exc:
        return err(str(exc))

    # #491: Compilation guard — if inference unavailable, queue instead of creating degraded article
    if not _inference_provider.available:
        logger.info("Inference unavailable, queueing compilation for later")
        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO compilation_queue (source_ids, title_hint, status)
                VALUES (%s, %s, 'pending')
                RETURNING id
                """,
                ([str(sid) for sid in source_ids], title_hint),
            )
            queue_row = cur.fetchone()
            queue_id = str(queue_row["id"]) if queue_row else "unknown"
        return ValenceResponse(
            success=False,
            error=f"Inference unavailable, queued for later compilation (queue_id={queue_id})",
            data={"queued": True, "queue_id": queue_id},
        )

    is_degraded = False
    try:
        llm_response = await _call_llm(prompt, task_type=TASK_COMPILE)
        parsed = validate_output(TASK_COMPILE, llm_response)
    except (NotImplementedError, InferenceSchemaError, json.JSONDecodeError) as exc:
        logger.warning("LLM compilation failed (%s), using fallback concatenation", exc)
        is_degraded = True
        content_parts = [f"## {s.get('title') or 'Source'}\n{s.get('content', '')}" for s in sources]
        parsed = {
            "articles": [
                {
                    "title": title_hint or (sources[0].get("title") if sources else "Compiled Article"),
                    "content": "\n\n".join(content_parts),
                    "source_relationships": [{"source_id": s["id"], "relationship": "originates"} for s in sources],
                }
            ],
        }

    articles_data = _normalize_llm_articles(parsed)
    if not articles_data:
        return err("LLM returned empty articles array")

    conf = compute_confidence(sources)
    created_articles = []

    # Create first (or only) article
    first_data = articles_data[0]
    article_content: str = first_data.get("content") or ""
    article_title: str | None = first_data.get("title") or title_hint
    relationships = _extract_relationships(first_data)

    epistemic_type = first_data.get("epistemic_type", "semantic")
    if epistemic_type not in ("episodic", "semantic", "procedural"):
        epistemic_type = "semantic"

    token_count = _count_tokens(article_content)

    with get_cursor() as cur:
        article = _create_article_row(cur, article_content, article_title, sources, conf, epistemic_type)
        article_id = article["id"]
        _link_sources(cur, article_id, sources, relationships)

        cur.execute(
            "INSERT INTO article_mutations (mutation_type, article_id, summary) VALUES ('created', %s, %s)",
            (article_id, f"Compiled from {len(sources)} source(s) via LLM"),
        )
        _maybe_queue_split(cur, article_id, token_count, max_tokens)

        # DR-9: Mark degraded articles and queue for reprocessing
        if is_degraded:
            try:
                cur.execute("UPDATE articles SET degraded = TRUE WHERE id = %s", (article_id,))
                cur.execute(
                    """
                    INSERT INTO mutation_queue (operation, article_id, priority, payload)
                    VALUES ('recompile_degraded', %s, 5, %s::jsonb)
                    """,
                    (article_id, json.dumps({"reason": "inference_unavailable_at_compile_time"})),
                )
                logger.info("Article %s marked degraded; queued for recompile", article_id)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to mark article %s as degraded: %s", article_id, exc)

        created_articles.append(article)

    # Create additional articles if LLM produced multiple
    extras = _create_extra_articles(articles_data, sources, conf, max_tokens, "Compiled")
    created_articles.extend(extras)

    first_article = created_articles[0]
    if len(created_articles) > 1:
        first_article["_all_articles"] = created_articles
        logger.info("Compiled %d articles from %d sources", len(created_articles), len(sources))

    return ok(data=first_article, degraded=is_degraded)


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
        article = serialize_row(dict(row))

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
        updated_article = serialize_row(dict(row))

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


# ---------------------------------------------------------------------------
# Recompilation (#490)
# ---------------------------------------------------------------------------


async def recompile_article(article_id: str) -> ValenceResponse:
    """Recompile an existing article from its linked sources.

    Fetches the article's linked sources from the article_sources provenance table,
    checks if inference is available, recompiles the article in-place, and clears
    the degraded flag on success.

    Args:
        article_id: UUID of the article to recompile.

    Returns:
        ValenceResponse with success status and updated article data.
    """
    if not _inference_provider.available:
        return err("Inference unavailable, cannot recompile")

    # Fetch article and its linked source IDs
    with get_cursor() as cur:
        cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
        row = cur.fetchone()
        if not row:
            return err(f"Article not found: {article_id}")
        article = dict(row)

        cur.execute(
            "SELECT source_id FROM article_sources WHERE article_id = %s ORDER BY added_at",
            (article_id,),
        )
        source_rows = cur.fetchall()

    if not source_rows:
        return err(f"No sources linked to article {article_id}")

    source_ids = [str(r["source_id"]) for r in source_rows]

    right_sizing = _get_right_sizing()
    target_tokens = right_sizing["target_tokens"]
    max_tokens = right_sizing["max_tokens"]

    sources, missing_ids = _fetch_sources_by_ids(source_ids)
    if missing_ids:
        return err(f"Source(s) not found during recompile: {', '.join(missing_ids)}")

    # Build prompt and call LLM
    try:
        prompt = _build_compilation_prompt(sources, article.get("title"), target_tokens)
        llm_response = await _call_llm(prompt, task_type=TASK_COMPILE)
        parsed = validate_output(TASK_COMPILE, llm_response)
    except (NotImplementedError, InferenceSchemaError, json.JSONDecodeError) as exc:
        logger.warning("LLM recompilation failed (%s), marking degraded", exc)
        return err(f"Recompilation failed: {exc}")

    articles_data = _normalize_llm_articles(parsed)
    if not articles_data:
        return err("LLM returned empty articles array during recompile")

    # First article updates the existing article in-place
    first = articles_data[0]
    article_content: str = first.get("content") or ""
    article_title: str | None = first.get("title") or article.get("title")
    relationships = _extract_relationships(first)
    token_count = _count_tokens(article_content)
    conf = compute_confidence(sources)

    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE articles
            SET content      = %s,
                title        = %s,
                size_tokens  = %s,
                content_hash = md5(%s),
                version      = version + 1,
                modified_at  = NOW(),
                compiled_at  = NOW(),
                degraded     = FALSE,
                confidence   = %s::jsonb,
                confidence_source = %s,
                corroboration_count = %s
            WHERE id = %s
            RETURNING *
            """,
            (
                article_content,
                article_title,
                token_count,
                article_content,
                json.dumps(conf.to_jsonb()),
                round(conf.avg_reliability, 3),
                conf.corroboration_count,
                article_id,
            ),
        )
        row = cur.fetchone()
        if not row:
            return err(f"Article not found after recompile: {article_id}")
        updated_article = serialize_row(dict(row))

        _link_sources(cur, article_id, sources, relationships, clear_existing=True)

        cur.execute(
            "INSERT INTO article_mutations (mutation_type, article_id, summary) VALUES ('recompiled', %s, %s)",
            (article_id, f"Recompiled from {len(sources)} source(s) via LLM"),
        )
        _maybe_queue_split(cur, article_id, token_count, max_tokens)

    # Create additional articles if LLM produced multiple
    extras = _create_extra_articles(
        articles_data,
        sources,
        conf,
        max_tokens,
        f"Created during recompile of {article_id}",
    )
    if extras:
        updated_article["_extra_articles"] = extras
        logger.info("Recompile of %s produced %d additional articles", article_id, len(extras))

    return ok(data=updated_article)


async def recompile_degraded_articles(limit: int = 10) -> ValenceResponse:
    """Recompile all degraded articles.

    Args:
        limit: Maximum number of articles to recompile (default 10).

    Returns:
        Dict with counts of recompiled and remaining degraded articles.
    """
    # Check inference availability first
    if not _inference_provider.available:
        return err("Inference unavailable, skipping recompilation")

    # Get degraded articles
    with get_cursor() as cur:
        cur.execute(
            "SELECT id FROM articles WHERE degraded = true AND status = 'active' ORDER BY created_at LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()

    article_ids = [str(r["id"]) for r in rows]

    recompiled = 0
    failed = 0

    for article_id in article_ids:
        result = await recompile_article(article_id)
        if result.success:
            recompiled += 1
            logger.info("Recompiled degraded article %s", article_id)
        else:
            failed += 1
            logger.warning("Failed to recompile degraded article %s: %s", article_id, result.error)

    # Get remaining count
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) as count FROM articles WHERE degraded = true AND status = 'active'")
        row = cur.fetchone()
        remaining = row["count"] if row else 0

    return ok(
        data={
            "recompiled": recompiled,
            "failed": failed,
            "remaining": remaining,
        }
    )


# ---------------------------------------------------------------------------
# Compilation queue (#491)
# ---------------------------------------------------------------------------


async def drain_compilation_queue(limit: int = 10) -> ValenceResponse:
    """Process pending items from the compilation queue.

    Args:
        limit: Maximum number of items to process (default 10).

    Returns:
        Dict with counts of processed, failed, and remaining items.
    """
    # Check inference availability first
    if not _inference_provider.available:
        logger.info("Inference unavailable, skipping compilation queue drain")
        return ok(data={"processed": 0, "failed": 0, "remaining": 0, "skipped": "inference_unavailable"})

    # Get pending items
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, source_ids, title_hint, attempts FROM compilation_queue WHERE status = 'pending' ORDER BY queued_at LIMIT %s FOR UPDATE SKIP LOCKED",
            (limit,),
        )
        rows = cur.fetchall()

    if not rows:
        return ok(data={"processed": 0, "failed": 0, "remaining": 0})

    processed = 0
    failed = 0

    for row in rows:
        item_id = str(row["id"])
        source_ids = row["source_ids"]
        title_hint = row.get("title_hint")
        attempts = row.get("attempts", 0)

        try:
            # Compile the article
            result = await compile_article(source_ids, title_hint=title_hint)

            if result.success and not result.degraded:
                # Success - remove from queue
                with get_cursor() as cur:
                    cur.execute("DELETE FROM compilation_queue WHERE id = %s", (item_id,))
                processed += 1
                logger.info("Compilation queue: processed item %s → article %s", item_id, result.data.get("id"))
            else:
                # Failed or degraded - increment attempts
                with get_cursor() as cur:
                    cur.execute(
                        """
                        UPDATE compilation_queue
                        SET attempts = attempts + 1,
                            last_attempt = NOW(),
                            status = CASE WHEN attempts + 1 >= 5 THEN 'failed' ELSE 'pending' END
                        WHERE id = %s
                        """,
                        (item_id,),
                    )
                failed += 1
                logger.warning("Compilation queue: item %s failed (attempts=%d): %s", item_id, attempts + 1, result.error)

        except Exception as exc:
            logger.error("Compilation queue: error processing item %s: %s", item_id, exc)
            with get_cursor() as cur:
                cur.execute(
                    """
                    UPDATE compilation_queue
                    SET attempts = attempts + 1,
                        last_attempt = NOW(),
                        status = CASE WHEN attempts + 1 >= 5 THEN 'failed' ELSE 'pending' END
                    WHERE id = %s
                    """,
                    (item_id,),
                )
            failed += 1

    # Get remaining count
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) as count FROM compilation_queue WHERE status = 'pending'")
        row = cur.fetchone()
        remaining = row["count"] if row else 0

    return ok(
        data={
            "processed": processed,
            "failed": failed,
            "remaining": remaining,
        }
    )
