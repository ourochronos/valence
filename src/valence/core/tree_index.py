# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Tree index builder for large sources (C1, #516).

Builds a hierarchical tree index over a source's content, stored as JSONB
metadata on the source. Each tree node references a character range in the
original text — no copies, no rewriting.

For sources that fit in context: single LLM call.
For sources that exceed context: sliding window with 20% overlap + merge pass.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC

from valence.core.compilation import _call_llm
from valence.core.db import get_cursor
from valence.core.inference import TASK_TREE

from .response import ValenceResponse, err, ok

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sources below this token estimate won't be indexed
DEFAULT_INDEX_THRESHOLD = 2000

# Maximum tokens per window for large source indexing
DEFAULT_WINDOW_TOKENS = 80000

# Overlap between windows (fraction)
WINDOW_OVERLAP = 0.20

# Rough chars-per-token estimate
CHARS_PER_TOKEN = 3.5

# Task type for inference provider routing

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SINGLE_WINDOW_PROMPT = """You are building a tree index (table of contents) for a source document.

Your job is to identify the natural topic structure and return a tree where each
node references a region of the original text by character offset.

Rules:
- Identify major topics (depth 1) and subtopics (depth 2+) where they naturally exist
- Each leaf node MUST have start_char and end_char pointing to exact character positions
  in the original text
- Parent nodes should have start_char and end_char spanning their children's full range
- Nodes must cover the entire document without gaps
- Minimal overlap between sibling nodes (only at natural boundaries)
- Do NOT split mid-sentence or mid-thought
- Keep titles concise (5-10 words)
- Add a one-sentence summary for each node (for navigation, not replacement)

Return ONLY valid JSON:
{{
  "nodes": [
    {{
      "title": "Topic Title",
      "summary": "One sentence summary for navigation",
      "start_char": 0,
      "end_char": 500,
      "children": [
        {{
          "title": "Subtopic",
          "summary": "One sentence summary",
          "start_char": 0,
          "end_char": 250
        }}
      ]
    }}
  ]
}}

Omit "children" for leaf nodes. The source is {source_chars} characters long.

--- SOURCE TEXT ---
{source_text}
--- END SOURCE TEXT ---"""


WINDOW_PROMPT = """You are building a tree index (table of contents) for a SECTION of a larger document.

This section starts at character offset {global_offset} in the full document.
{context_note}

Identify the natural topic structure within this section. Each node's start_char and
end_char must be GLOBAL character offsets (relative to the full document, not this section).

Rules:
- This section covers global characters {global_offset} to {global_end}
- All start_char/end_char values must be within this range
- Do NOT split mid-sentence or mid-thought
- Keep titles concise (5-10 words)
- Add a one-sentence summary for each node

Return ONLY valid JSON:
{{
  "nodes": [
    {{
      "title": "Topic Title",
      "summary": "One sentence summary",
      "start_char": {global_offset},
      "end_char": <offset>,
      "children": [...]
    }}
  ]
}}

Omit "children" for leaf nodes.

--- SECTION TEXT ---
{window_text}
--- END SECTION TEXT ---"""


MERGE_PROMPT = """You are merging local tree indexes into a coherent top-level tree structure.

Below are tree indexes built from sequential sections of a large document.
Merge them into a single coherent hierarchy:
- Combine nodes that clearly belong to the same topic across section boundaries
- Create parent nodes to group related sections
- Preserve all leaf-level start_char/end_char offsets exactly as given
- Keep titles concise (5-10 words)
- Add a one-sentence summary for each new parent node

Return ONLY valid JSON with the same node schema:
{{
  "nodes": [...]
}}

--- LOCAL TREES ---
{local_trees_json}
--- END LOCAL TREES ---"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    return int(len(text) / CHARS_PER_TOKEN)


def _validate_tree(tree: dict, source_len: int) -> list[str]:
    """Validate tree node offsets against source length. Returns list of issues."""
    issues = []

    def walk(nodes: list[dict], depth: int = 0) -> None:
        for node in nodes:
            start = node.get("start_char", -1)
            end = node.get("end_char", -1)
            title = node.get("title", "<untitled>")

            if start < 0:
                issues.append(f"Missing start_char: {title}")
            if end < 0:
                issues.append(f"Missing end_char: {title}")
            if start >= end:
                issues.append(f"Invalid range [{start}:{end}]: {title}")
            if end > source_len:
                # Clamp rather than reject — LLM may overshoot by a few chars
                node["end_char"] = source_len
            if start > source_len:
                issues.append(f"start_char beyond source: {title} [{start}]")

            children = node.get("children", [])
            if children:
                walk(children, depth + 1)

    walk(tree.get("nodes", []))
    return issues


async def _build_tree_single(source_text: str) -> dict:
    """Build tree index in a single LLM call (source fits in context)."""
    prompt = SINGLE_WINDOW_PROMPT.format(
        source_chars=len(source_text),
        source_text=source_text,
    )

    response = await _call_llm(prompt, task_type=TASK_TREE)
    tree = _extract_json(response)
    return tree


async def _build_tree_windowed(source_text: str, window_tokens: int = DEFAULT_WINDOW_TOKENS) -> dict:
    """Build tree index using sliding windows with overlap for large sources."""
    window_chars = int(window_tokens * CHARS_PER_TOKEN)
    overlap_chars = int(window_chars * WINDOW_OVERLAP)
    step = window_chars - overlap_chars

    local_trees: list[dict] = []
    offset = 0

    while offset < len(source_text):
        end = min(offset + window_chars, len(source_text))
        window_text = source_text[offset:end]

        context_note = ""
        if local_trees:
            prev_title = local_trees[-1].get("nodes", [{}])[-1].get("title", "unknown")
            context_note = f"The previous section ended with topic: '{prev_title}'"

        prompt = WINDOW_PROMPT.format(
            global_offset=offset,
            global_end=end,
            context_note=context_note,
            window_text=window_text,
        )

        response = await _call_llm(prompt, task_type=TASK_TREE)
        local_tree = _extract_json(response)
        local_trees.append(local_tree)

        logger.info(
            "Window %d: chars [%d:%d], %d nodes",
            len(local_trees),
            offset,
            end,
            len(local_tree.get("nodes", [])),
        )

        if end >= len(source_text):
            break
        offset += step

    # Merge pass
    if len(local_trees) == 1:
        return local_trees[0]

    local_trees_json = json.dumps(local_trees, indent=2)

    # If merge input fits in context, single merge call
    if _estimate_tokens(local_trees_json) < window_tokens:
        prompt = MERGE_PROMPT.format(local_trees_json=local_trees_json)
        response = await _call_llm(prompt, task_type=TASK_TREE)
        return _extract_json(response)

    # Recursive merge for very large sources
    # Group local trees and merge in batches
    group_size = max(2, window_tokens // (_estimate_tokens(json.dumps(local_trees[0])) + 100))
    merged_groups = []
    for i in range(0, len(local_trees), group_size):
        group = local_trees[i : i + group_size]
        group_json = json.dumps(group, indent=2)
        prompt = MERGE_PROMPT.format(local_trees_json=group_json)
        response = await _call_llm(prompt, task_type=TASK_TREE)
        merged_groups.append(_extract_json(response))

    # Final merge of merged groups
    final_json = json.dumps(merged_groups, indent=2)
    prompt = MERGE_PROMPT.format(local_trees_json=final_json)
    response = await _call_llm(prompt, task_type=TASK_TREE)
    return _extract_json(response)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def build_tree_index(
    source_id: str,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
    force: bool = False,
) -> ValenceResponse:
    """Build a tree index for a source and store it in source metadata.

    Args:
        source_id: UUID of the source to index.
        window_tokens: Max tokens per window for windowed indexing.
        force: Rebuild even if tree_index already exists in metadata.

    Returns:
        ValenceResponse with data = {source_id, tree, node_count, method}.
    """
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, content, metadata FROM sources WHERE id = %s::uuid",
            (source_id,),
        )
        row = cur.fetchone()

    if not row:
        return err(f"Source not found: {source_id}")

    content = row["content"]
    if not content:
        return err(f"Source has no content: {source_id}")

    metadata = row["metadata"] or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    if not force and "tree_index" in metadata:
        return err("Source already has tree_index. Use force=True to rebuild.")

    # Skip sources too small for meaningful tree indexing
    min_tree_chars = 200
    if len(content) < min_tree_chars:
        return err(f"Source too small for tree indexing ({len(content)} chars, minimum {min_tree_chars}). Small sources are served whole.")

    token_estimate = _estimate_tokens(content)
    logger.info(
        "Building tree index for source %s (~%d tokens, %d chars)",
        source_id,
        token_estimate,
        len(content),
    )

    # Choose method based on size
    if token_estimate <= window_tokens:
        tree = await _build_tree_single(content)
        method = "single"
    else:
        tree = await _build_tree_windowed(content, window_tokens=window_tokens)
        method = "windowed"

    # Validate
    issues = _validate_tree(tree, len(content))
    if issues:
        logger.warning("Tree validation issues for source %s: %s", source_id, issues)

    # Count nodes
    def count_nodes(nodes: list[dict]) -> int:
        total = len(nodes)
        for n in nodes:
            total += count_nodes(n.get("children", []))
        return total

    node_count = count_nodes(tree.get("nodes", []))

    # Store in metadata
    metadata["tree_index"] = tree
    metadata["tree_indexed_at"] = "now()"  # Will be replaced by actual timestamp

    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE sources
            SET metadata = %s::jsonb
            WHERE id = %s::uuid
            RETURNING id
            """,
            (json.dumps(metadata), source_id),
        )
        updated = cur.fetchone()

    if not updated:
        return err(f"Failed to update source metadata: {source_id}")

    # Fix timestamp — do it properly
    from datetime import datetime

    metadata["tree_indexed_at"] = datetime.now(UTC).isoformat()
    with get_cursor() as cur:
        cur.execute(
            "UPDATE sources SET metadata = %s::jsonb WHERE id = %s::uuid",
            (json.dumps(metadata), source_id),
        )

    logger.info(
        "Tree index built for source %s: %d nodes, method=%s",
        source_id,
        node_count,
        method,
    )

    return ok(
        data={
            "source_id": source_id,
            "tree": tree,
            "node_count": node_count,
            "method": method,
            "token_estimate": token_estimate,
            "issues": issues,
        }
    )


async def get_tree_index(source_id: str) -> ValenceResponse:
    """Retrieve the tree index for a source.

    Returns:
        ValenceResponse with data = tree dict, or error if no index exists.
    """
    with get_cursor() as cur:
        cur.execute(
            "SELECT metadata FROM sources WHERE id = %s::uuid",
            (source_id,),
        )
        row = cur.fetchone()

    if not row:
        return err(f"Source not found: {source_id}")

    metadata = row["metadata"] or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    tree = metadata.get("tree_index")
    if not tree:
        return err(f"Source has no tree index: {source_id}")

    return ok(data=tree)


async def get_tree_region(source_id: str, start_char: int, end_char: int) -> ValenceResponse:
    """Extract a region of the original source text by character offsets.

    This is the key function — retrieval and compilation use tree node offsets
    to pull exact text from the original source. No copies of copies.

    Returns:
        ValenceResponse with data = {text, start_char, end_char, token_estimate}.
    """
    with get_cursor() as cur:
        cur.execute(
            "SELECT content FROM sources WHERE id = %s::uuid",
            (source_id,),
        )
        row = cur.fetchone()

    if not row:
        return err(f"Source not found: {source_id}")

    content = row["content"]
    if start_char < 0 or end_char > len(content) or start_char >= end_char:
        return err(f"Invalid range [{start_char}:{end_char}] for source of length {len(content)}")

    text = content[start_char:end_char]

    return ok(
        data={
            "text": text,
            "start_char": start_char,
            "end_char": end_char,
            "token_estimate": _estimate_tokens(text),
        }
    )
