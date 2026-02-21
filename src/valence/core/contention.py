"""Contention detection and surfacing for the Valence v2 knowledge system.

A contention records a discovered disagreement between an article and a source
(or two articles). Detection uses LLM analysis when available, falling back to
text-overlap heuristics. Only contentions above the ``materiality_threshold``
(from ``system_config``) are persisted.

Implements WU-08 (C7 — contention detection + surfacing).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable
from uuid import UUID

from our_db import get_cursor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Backend — same injectable pattern as compilation.py
# ---------------------------------------------------------------------------

_LLM_BACKEND: Callable[[str], Any] | None = None


def set_llm_backend(fn: Callable[[str], Any] | None) -> None:
    """Configure the LLM callable used for contention detection.

    Args:
        fn: A sync or async callable ``(prompt: str) -> str``, or ``None`` to reset.

    Example::

        set_llm_backend(lambda p: json.dumps({"contends": True, "materiality": 0.7,
                                              "description": "Source disagrees on X"}))
    """
    global _LLM_BACKEND
    _LLM_BACKEND = fn


async def _call_llm(prompt: str) -> str:
    """Invoke the configured LLM backend.

    Raises:
        NotImplementedError: If no backend is configured.
    """
    if _LLM_BACKEND is None:
        raise NotImplementedError(
            "No LLM backend configured for contention detection. "
            "Call valence.core.contention.set_llm_backend(fn) with a "
            "sync or async callable (str) -> str."
        )
    result = _LLM_BACKEND(prompt)
    if asyncio.iscoroutine(result):
        return await result
    return result


# ---------------------------------------------------------------------------
# Schema helpers — ensure contention table can store source references
# ---------------------------------------------------------------------------

_SCHEMA_ENSURED = False


def _ensure_contention_schema() -> None:
    """Make belief_b_id nullable and add source_id to contentions table.

    The original tensions table required two article FKs; for source-triggered
    contentions we need only one article + a source reference.  This runs once
    per process.
    """
    global _SCHEMA_ENSURED
    if _SCHEMA_ENSURED:
        return
    try:
        with get_cursor() as cur:
            cur.execute(
                """
                ALTER TABLE contentions
                    ALTER COLUMN belief_b_id DROP NOT NULL
                """
            )
    except Exception:
        pass  # Already nullable or table not yet created; ignore.

    try:
        with get_cursor() as cur:
            cur.execute(
                """
                ALTER TABLE contentions
                    ADD COLUMN IF NOT EXISTS source_id UUID
                        REFERENCES sources(id) ON DELETE SET NULL
                """
            )
    except Exception:
        pass  # Column already exists; ignore.

    _SCHEMA_ENSURED = True


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

DEFAULT_MATERIALITY_THRESHOLD = 0.3


def _get_materiality_threshold() -> float:
    """Read ``contention.materiality_threshold`` from system_config."""
    try:
        with get_cursor() as cur:
            cur.execute(
                "SELECT value FROM system_config WHERE key = 'contention' LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                val = row["value"]
                if isinstance(val, str):
                    val = json.loads(val)
                if isinstance(val, dict):
                    return float(val.get("materiality_threshold", DEFAULT_MATERIALITY_THRESHOLD))
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read contention config: %s", exc)
    return DEFAULT_MATERIALITY_THRESHOLD


# ---------------------------------------------------------------------------
# Row serialization
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
# Prompt builders
# ---------------------------------------------------------------------------


def _build_detection_prompt(article_content: str, source_content: str) -> str:
    """Build LLM prompt asking whether the source contends with the article."""
    return f"""You are a knowledge analyst. Assess whether a new source contradicts or contends with an existing knowledge article.

EXISTING ARTICLE:
{article_content}

NEW SOURCE:
{source_content}

Assess:
1. Does the source CONTRADICT the article (direct factual disagreement)?
2. Does the source CONTEND with the article (alternative viewpoint, partial disagreement)?
3. How MATERIAL is the disagreement? (0.0 = trivial/irrelevant, 1.0 = fundamentally incompatible)

Respond ONLY with valid JSON (no markdown fences):
{{
  "contends": true|false,
  "contention_type": "contradiction|scope_conflict|temporal_conflict|partial_overlap",
  "materiality": 0.0-1.0,
  "description": "<one sentence describing the disagreement, or null if no disagreement>"
}}

If there is no disagreement, set "contends": false and "materiality": 0.0."""


# ---------------------------------------------------------------------------
# Text-overlap heuristic fallback
# ---------------------------------------------------------------------------


def _heuristic_materiality(article_content: str, source_content: str) -> float:
    """Simple word-overlap heuristic when LLM is unavailable.

    Returns a score in [0, 1]. Shared vocabulary between two short texts is
    *not* a contention signal; we use dissimilarity as a weak proxy by
    flagging for review at a fixed low score.
    """
    # No meaningful heuristic — flag all for review at just-above threshold.
    # This ensures human review rather than silent suppression.
    return DEFAULT_MATERIALITY_THRESHOLD + 0.05


# ---------------------------------------------------------------------------
# JSON parsing (same pattern as compilation.py)
# ---------------------------------------------------------------------------


def _parse_llm_json(response: str, required_keys: list[str]) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()

    parsed = json.loads(text)
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        raise ValueError(
            f"LLM response missing required keys: {missing!r}. Got: {list(parsed.keys())!r}"
        )
    return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def detect_contention(article_id: str, source_id: str) -> dict[str, Any] | None:
    """Check whether a source contradicts or contends with an article.

    Invokes the LLM to assess materiality if a backend is configured; falls
    back to a heuristic if not.  Creates a ``contentions`` row only when
    ``materiality >= materiality_threshold`` from ``system_config``.

    Args:
        article_id: UUID of the existing article.
        source_id: UUID of the incoming source.

    Returns:
        The newly created contention dict, or ``None`` if materiality is below
        the threshold (or if the article/source is not found).
    """
    _ensure_contention_schema()
    threshold = _get_materiality_threshold()

    # ---- Fetch article and source ----
    with get_cursor() as cur:
        cur.execute("SELECT id, content, title FROM articles WHERE id = %s", (article_id,))
        article_row = cur.fetchone()
        if article_row is None:
            logger.warning("detect_contention: article %s not found", article_id)
            return None
        article = dict(article_row)
        article["id"] = str(article["id"])

        cur.execute("SELECT id, content, title FROM sources WHERE id = %s", (source_id,))
        source_row = cur.fetchone()
        if source_row is None:
            logger.warning("detect_contention: source %s not found", source_id)
            return None
        source = dict(source_row)
        source["id"] = str(source["id"])

    article_content = (article.get("content") or "").strip()
    source_content = (source.get("content") or "").strip()

    if not article_content or not source_content:
        logger.debug(
            "detect_contention: empty content for article %s or source %s, skipping",
            article_id, source_id,
        )
        return None

    # ---- LLM detection ----
    prompt = _build_detection_prompt(article_content, source_content)
    try:
        llm_response = await _call_llm(prompt)
        parsed = _parse_llm_json(llm_response, ["contends", "materiality"])
        contends: bool = bool(parsed.get("contends", False))
        materiality: float = float(parsed.get("materiality", 0.0))
        contention_type: str = parsed.get("contention_type", "contradiction")
        description: str | None = parsed.get("description")
    except (NotImplementedError, ValueError, json.JSONDecodeError) as exc:
        logger.info(
            "LLM contention detection unavailable (%s); using heuristic for article=%s source=%s",
            exc, article_id, source_id,
        )
        materiality = _heuristic_materiality(article_content, source_content)
        contends = materiality > 0
        contention_type = "contradiction"
        description = "Potential contention flagged for review (LLM unavailable)"

    # ---- Threshold check ----
    if not contends or materiality < threshold:
        logger.debug(
            "detect_contention: materiality=%.2f below threshold=%.2f for article=%s source=%s; skipping",
            materiality, threshold, article_id, source_id,
        )
        return None

    # ---- Clamp contention_type ----
    valid_types = {"contradiction", "temporal_conflict", "scope_conflict", "partial_overlap"}
    if contention_type not in valid_types:
        contention_type = "contradiction"

    # ---- Create contention record ----
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO contentions
                (belief_a_id, source_id, type, description, severity, status, materiality)
            VALUES (%s, %s, %s, %s,
                    CASE WHEN %s::numeric >= 0.7 THEN 'high'
                         WHEN %s::numeric >= 0.4 THEN 'medium'
                         ELSE 'low' END,
                    'detected', %s)
            RETURNING *
            """,
            (
                article_id, source_id,
                contention_type,
                description,
                materiality, materiality,  # two references for the CASE
                materiality,
            ),
        )
        row = cur.fetchone()
        if row is None:
            logger.error("detect_contention: INSERT returned no row")
            return None
        contention = _serialize_row(dict(row))

    logger.info(
        "Contention created: id=%s article=%s source=%s materiality=%.2f type=%s",
        contention["id"], article_id, source_id, materiality, contention_type,
    )
    return contention


async def list_contentions(
    article_id: str | None = None,
    status: str = "detected",
) -> list[dict[str, Any]]:
    """List contentions, optionally filtered by article and status.

    Args:
        article_id: If given, only return contentions involving this article.
        status: Filter by contention status (``detected``, ``resolved``,
            ``dismissed``).  Pass ``None`` to return all statuses.

    Returns:
        List of contention dicts, newest first.
    """
    _ensure_contention_schema()

    with get_cursor() as cur:
        sql = "SELECT * FROM contentions WHERE 1=1"
        params: list[Any] = []

        if status is not None:
            sql += " AND status = %s"
            params.append(status)

        if article_id is not None:
            sql += " AND (belief_a_id = %s OR belief_b_id = %s)"
            params.extend([article_id, article_id])

        sql += " ORDER BY detected_at DESC"

        cur.execute(sql, params)
        rows = cur.fetchall()

    return [_serialize_row(dict(r)) for r in rows]


async def resolve_contention(
    contention_id: str,
    resolution: str,
    rationale: str,
) -> dict[str, Any]:
    """Resolve a contention.

    Resolution types:
    - ``supersede_a``: Article's position wins; source is noted but article
      is unchanged.
    - ``supersede_b``: Source's position replaces article content.
    - ``accept_both``: Both perspectives are valid; article is annotated.
    - ``dismiss``: Contention is not material; dismissed without change.

    Args:
        contention_id: UUID of the contention to resolve.
        resolution: One of ``supersede_a``, ``supersede_b``, ``accept_both``,
            ``dismiss``.
        rationale: Free-text reason for the resolution.

    Returns:
        Dict with ``success``, the updated ``contention``, and optionally
        ``article`` (when the article was modified).
    """
    _ensure_contention_schema()

    valid_resolutions = {"supersede_a", "supersede_b", "accept_both", "dismiss"}
    if resolution not in valid_resolutions:
        return {
            "success": False,
            "error": f"resolution must be one of {sorted(valid_resolutions)}",
        }

    # ---- Load contention ----
    with get_cursor() as cur:
        cur.execute("SELECT * FROM contentions WHERE id = %s", (contention_id,))
        row = cur.fetchone()
    if row is None:
        return {"success": False, "error": f"Contention not found: {contention_id}"}

    contention = dict(row)
    article_id = str(contention["belief_a_id"])
    source_id = str(contention.get("source_id") or "")
    article_b_id = str(contention["belief_b_id"]) if contention.get("belief_b_id") else None

    updated_article: dict[str, Any] | None = None

    # ---- Act on resolution ----
    if resolution == "supersede_b" and (source_id or article_b_id):
        # Source / article B's position replaces article A.
        updated_article = await _apply_supersede_b(
            article_id=article_id,
            source_id=source_id or None,
            article_b_id=article_b_id,
            rationale=rationale,
        )

    elif resolution == "accept_both":
        # Annotate the article to surface the contention without overwriting.
        with get_cursor() as cur:
            cur.execute(
                """
                UPDATE articles
                SET extraction_metadata =
                    COALESCE(extraction_metadata, '{}'::jsonb)
                    || %s::jsonb,
                    modified_at = NOW()
                WHERE id = %s
                RETURNING id
                """,
                (
                    json.dumps({
                        "contention_note": rationale,
                        "contention_id": contention_id,
                    }),
                    article_id,
                ),
            )

    # ---- Mark contention resolved / dismissed ----
    final_status = "dismissed" if resolution == "dismiss" else "resolved"
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE contentions
            SET status = %s,
                resolution = %s,
                resolved_at = NOW()
            WHERE id = %s
            RETURNING *
            """,
            (final_status, rationale, contention_id),
        )
        updated_row = cur.fetchone()

    if updated_row is None:
        return {"success": False, "error": "Failed to update contention row"}

    result: dict[str, Any] = {
        "success": True,
        "contention": _serialize_row(dict(updated_row)),
    }
    if updated_article:
        result["article"] = updated_article
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _apply_supersede_b(
    article_id: str,
    source_id: str | None,
    article_b_id: str | None,
    rationale: str,
) -> dict[str, Any] | None:
    """Apply a 'supersede_b' resolution: update article A with source/B's content.

    Uses the LLM to produce a merged article when available; otherwise
    appends the source content directly.

    Returns the updated article dict, or None on failure.
    """
    # Fetch article A
    with get_cursor() as cur:
        cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
        art_row = cur.fetchone()
    if art_row is None:
        logger.warning("_apply_supersede_b: article %s not found", article_id)
        return None

    article = dict(art_row)
    article_content = (article.get("content") or "").strip()

    # Obtain superseding content
    if source_id:
        with get_cursor() as cur:
            cur.execute("SELECT content, title FROM sources WHERE id = %s", (source_id,))
            src_row = cur.fetchone()
        new_content = (dict(src_row).get("content") or "").strip() if src_row else ""
        trigger_source_id: str | None = source_id
    elif article_b_id:
        with get_cursor() as cur:
            cur.execute("SELECT content FROM articles WHERE id = %s", (article_b_id,))
            b_row = cur.fetchone()
        new_content = (dict(b_row).get("content") or "").strip() if b_row else ""
        trigger_source_id = None
    else:
        return None

    if not new_content:
        return None

    # Try LLM-assisted merge; fall back to replacement
    try:
        merge_prompt = (
            f"Rewrite the following article incorporating the correction described by "
            f"the rationale '{rationale}'. "
            f"Use the new source's perspective as the authoritative position.\n\n"
            f"CURRENT ARTICLE:\n{article_content}\n\n"
            f"SUPERSEDING CONTENT:\n{new_content}\n\n"
            f"Respond ONLY with valid JSON (no markdown):\n"
            f'{{ "content": "<revised article content>" }}'
        )
        llm_response = await _call_llm(merge_prompt)
        parsed = _parse_llm_json(llm_response, ["content"])
        final_content: str = parsed["content"]
    except (NotImplementedError, ValueError, json.JSONDecodeError) as exc:
        logger.info("LLM supersede merge unavailable (%s); using source content directly", exc)
        final_content = new_content

    # Update article
    import hashlib

    content_hash = hashlib.sha256(final_content.strip().lower().encode()).hexdigest()
    token_count = max(1, int(len(final_content.split()) * 1.3))

    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE articles
            SET content      = %s,
                size_tokens  = %s,
                content_hash = %s,
                version      = version + 1,
                modified_at  = NOW(),
                compiled_at  = NOW()
            WHERE id = %s
            RETURNING *
            """,
            (final_content, token_count, content_hash, article_id),
        )
        updated_row = cur.fetchone()
        if not updated_row:
            return None

        # Record mutation
        cur.execute(
            """
            INSERT INTO article_mutations
                (mutation_type, article_id, trigger_source_id, summary)
            VALUES ('updated', %s, %s, %s)
            """,
            (
                article_id,
                trigger_source_id,
                f"Superseded by contention resolution: {rationale}",
            ),
        )

    updated_article = _serialize_row(dict(updated_row))
    logger.info("Article %s updated via supersede_b resolution", article_id)
    return updated_article
