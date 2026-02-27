# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Memory tool handlers for agent-friendly memory operations.

Provides thin wrappers around source/article primitives optimized for AI agents:
- memory_store: Store a memory (wraps source_ingest with agent metadata)
- memory_recall: Search memories (wraps knowledge_search with agent-friendly defaults)
- memory_status: Get memory system stats
- memory_forget: Soft-delete a memory
"""

from __future__ import annotations

import json
import logging
from typing import Any

from valence.core.db import get_cursor

from .sources import source_ingest

logger = logging.getLogger(__name__)

# Constants for memory recall
RECALL_OVERFETCH_MULTIPLIER = 3  # Over-fetch to account for filtering
SNIPPET_TRUNCATE_LENGTH = 200  # Max length before truncation


def memory_store(
    content: str,
    context: str | None = None,
    importance: float = 0.5,
    tags: list[str] | None = None,
    supersedes_id: str | None = None,
) -> dict[str, Any]:
    """Store a memory as an observation source.

    Args:
        content: The memory content (required)
        context: Where this memory came from (session, conversation, observation)
        importance: How important this memory is (0.0-1.0, default 0.5)
        tags: Categorization tags
        supersedes_id: UUID of a previous memory this replaces

    Returns:
        Success dict with source ID, or error dict
    """
    if not content or not content.strip():
        return {"success": False, "error": "content must be non-empty"}

    # Validate importance
    importance = max(0.0, min(1.0, float(importance)))

    # Build metadata with agent-specific fields
    metadata: dict[str, Any] = {
        "memory": True,
        "importance": importance,
    }
    if context:
        metadata["context"] = context
    if tags:
        metadata["tags"] = tags

    # Generate title from first line or context
    title = None
    if content:
        first_line = content.split("\n")[0].strip()
        if first_line and len(first_line) < 100:
            title = first_line
        elif context:
            title = f"Memory from {context}"
        else:
            title = "Memory"

    # Map importance to reliability for observation type
    # observation base = 0.4, scale by importance: 0.4 + (importance * 0.4) → [0.4, 0.8]
    # We'll let source_ingest use the default reliability (0.4) and store importance in metadata

    # Call source_ingest
    result = source_ingest(
        content=content,
        source_type="observation",
        title=title,
        url=None,
        metadata=metadata,
        supersedes=supersedes_id,
    )

    if not result.get("success"):
        return result

    # Return agent-friendly format
    source = result.get("source", {})
    return {
        "success": True,
        "memory_id": source.get("id"),
        "title": source.get("title"),
        "importance": importance,
        "tags": tags or [],
        "created_at": source.get("created_at"),
    }


def memory_recall(
    query: str,
    limit: int = 5,
    min_confidence: float | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Search memories using knowledge_search with agent-friendly defaults.

    Args:
        query: What to recall
        limit: Maximum results (default 5)
        min_confidence: Optional minimum confidence threshold (0.0-1.0)
        tags: Optional tag filter

    Returns:
        Success dict with memories list, or error dict
    """
    if not query or not query.strip():
        return {"success": False, "error": "query must be non-empty"}

    limit = max(1, min(int(limit), 50))

    # Use knowledge_search but filter for observation sources
    from .articles import knowledge_search

    # For now, call knowledge_search and post-filter
    # Future: could add metadata filter to knowledge_search
    result = knowledge_search(
        query=query,
        limit=limit * RECALL_OVERFETCH_MULTIPLIER,
        include_sources=True,
        session_id=None,
    )

    if not result.get("success"):
        return result

    results = result.get("results", [])

    # Collect source IDs to batch fetch metadata
    source_ids = []
    source_items = {}
    for item in results:
        if item.get("type") == "source":
            source_id = item.get("id")
            if source_id:
                source_ids.append(source_id)
                source_items[source_id] = item

    # Batch fetch all source metadata to avoid N+1 queries
    source_metadata = {}
    if source_ids:
        with get_cursor() as cur:
            cur.execute(
                "SELECT id, type, metadata FROM sources WHERE id = ANY(%s::uuid[])",
                (source_ids,),
            )
            for row in cur.fetchall():
                metadata = row.get("metadata", {})
                # JSONB should return dict directly from psycopg2; this isinstance check
                # handles legacy data or non-JSONB columns that might return serialized strings
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                source_metadata[row["id"]] = {
                    "type": row.get("type"),
                    "metadata": metadata,
                }

    # Filter for memories (observation sources with memory metadata)
    memories = []
    for source_id, item in source_items.items():
        src_data = source_metadata.get(source_id)
        if src_data and src_data["type"] == "observation":
            metadata = src_data["metadata"]
            if metadata.get("memory"):
                # Filter by tags if provided
                if tags:
                    item_tags = metadata.get("tags", [])
                    if not any(tag in item_tags for tag in tags):
                        continue

                # Filter by confidence if provided
                if min_confidence is not None:
                    confidence = item.get("confidence", {})
                    if isinstance(confidence, dict):
                        overall = confidence.get("overall", 0)
                    else:
                        overall = float(item.get("reliability", 0))
                    if overall < min_confidence:
                        continue

                # Build agent-friendly memory result
                memory = {
                    "memory_id": str(source_id),
                    "content": item.get("content", ""),
                    "title": item.get("title"),
                    "importance": metadata.get("importance", 0.5),
                    "context": metadata.get("context"),
                    "tags": metadata.get("tags", []),
                    "confidence": item.get("confidence", {}),
                    "age_days": item.get("freshness", 0),
                    "created_at": item.get("created_at"),
                    "score": item.get("final_score", 0),
                }
                memories.append(memory)

                if len(memories) >= limit:
                    break

    return {
        "success": True,
        "memories": memories,
        "count": len(memories),
        "query": query,
    }


def memory_status() -> dict[str, Any]:
    """Get memory system statistics.

    Returns count of memories, articles compiled from them, and last memory timestamp.
    """
    with get_cursor() as cur:
        # Count observation sources with memory metadata
        cur.execute(
            """
            SELECT COUNT(*) as count
            FROM sources
            WHERE type = 'observation'
              AND metadata->>'memory' = 'true'
            """
        )
        row = cur.fetchone()
        memory_count = row.get("count", 0) if row else 0

        # Get last memory timestamp
        cur.execute(
            """
            SELECT MAX(created_at) as last_memory
            FROM sources
            WHERE type = 'observation'
              AND metadata->>'memory' = 'true'
            """
        )
        row = cur.fetchone()
        last_memory = row.get("last_memory") if row else None
        if last_memory:
            last_memory = last_memory.isoformat()

        # Count articles compiled from memory sources
        cur.execute(
            """
            SELECT COUNT(DISTINCT a.id) as count
            FROM articles a
            JOIN article_sources asrc ON a.id = asrc.article_id
            JOIN sources s ON asrc.source_id = s.id
            WHERE s.type = 'observation'
              AND s.metadata->>'memory' = 'true'
              AND a.status = 'active'
            """
        )
        row = cur.fetchone()
        compiled_article_count = row.get("count", 0) if row else 0

        # Get tag distribution
        cur.execute(
            """
            SELECT jsonb_array_elements_text(metadata->'tags') as tag, COUNT(*) as count
            FROM sources
            WHERE type = 'observation'
              AND metadata->>'memory' = 'true'
              AND metadata->'tags' IS NOT NULL
            GROUP BY tag
            ORDER BY count DESC
            LIMIT 10
            """
        )
        tag_dist = {row["tag"]: row["count"] for row in cur.fetchall()}

    return {
        "success": True,
        "memory_count": memory_count,
        "compiled_articles": compiled_article_count,
        "last_memory_at": last_memory,
        "top_tags": tag_dist,
    }


def memory_forget(
    memory_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Mark a memory as forgotten (soft delete).

    Sets metadata to include forgotten flag and optional reason.
    Does not actually delete the source.

    Args:
        memory_id: UUID of the source to forget
        reason: Optional reason for forgetting

    Returns:
        Success dict or error dict
    """
    if not memory_id:
        return {"success": False, "error": "memory_id is required"}

    with get_cursor() as cur:
        # Verify it's a memory source
        cur.execute(
            """
            SELECT id, type, metadata
            FROM sources
            WHERE id = %s
            """,
            (memory_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Memory not found: {memory_id}"}

        if row.get("type") != "observation":
            return {"success": False, "error": f"Source {memory_id} is not a memory (type={row.get('type')})"}

        metadata = row.get("metadata", {})
        # JSONB should return dict directly from psycopg2; this isinstance check
        # handles legacy data or non-JSONB columns that might return serialized strings
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        if not metadata.get("memory"):
            return {"success": False, "error": f"Source {memory_id} is not marked as a memory"}

        # Update metadata to mark as forgotten
        metadata["forgotten"] = True
        if reason:
            metadata["forget_reason"] = reason

        cur.execute(
            """
            UPDATE sources
            SET metadata = %s::jsonb
            WHERE id = %s
            """,
            (json.dumps(metadata), memory_id),
        )

    return {
        "success": True,
        "memory_id": memory_id,
        "forgotten": True,
        "reason": reason,
    }
