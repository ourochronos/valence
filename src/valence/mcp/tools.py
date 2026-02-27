# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Substrate tool definitions — Valence v2 knowledge system (WU-11).

Contains SUBSTRATE_TOOLS — the MCP tool definitions for the v2
knowledge surface as specified in IMPL-SPEC.md §3.WU-11.

Tool list:
    source_ingest      C1 — Ingest a new source
    source_get         C1 — Get source by ID
    source_search      C1 — Full-text search over sources
    source_list        C1 — List sources with filters
    knowledge_search   C9 — Unified ranked retrieval (articles + sources)
    article_get        C2 — Get article with optional provenance
    article_create     C2 — Manually create an article
    article_compile    C2 — Compile sources into an article via LLM
    article_update     C2 — Update article with new content / source
    article_split      C3 — Split an oversized article
    article_merge      C3 — Merge two related articles
    article_search     C2 — Search articles by query/domain
    provenance_trace   C5 — Trace a claim to its contributing sources
    provenance_get     C5 — Get provenance links for an article
    provenance_link    C5 — Link a source to an article
    contention_list    C7 — List active contentions
    contention_resolve C7 — Resolve a contention
    contention_detect  C7 — Detect contradictions via semantic similarity
    admin_forget       C10 — Remove a source or article
    admin_stats        —   Health and capacity statistics
    admin_maintenance  —   Trigger maintenance operations
    memory_store       —   Store a memory (agent-friendly wrapper)
    memory_recall      —   Search memories (agent-friendly wrapper)
    memory_status      —   Get memory system stats
    memory_forget      —   Mark a memory as forgotten (soft delete)
    session_start      —   Upsert a session (insert-if-new)
    session_append     —   Append message(s) to session buffer
    session_flush      —   Flush unflushed messages to source
    session_finalize   —   Flush + complete + compile
    session_search     —   Search conversation sources
    session_list       —   List sessions with filters
    session_get        —   Get session + optional messages
    session_compile    —   Compile session sources into article
    session_flush_stale —  Flush all stale sessions
"""

from __future__ import annotations

from mcp.types import Tool

SUBSTRATE_TOOLS = [
    # =========================================================================
    # Source tools (C1)
    # =========================================================================
    Tool(
        name="source_ingest",
        description=(
            "Ingest a new source into the knowledge substrate.\n\n"
            "Sources are the raw, immutable input material from which articles are compiled. "
            "Call this whenever new information arrives — documents, conversation transcripts, "
            "web pages, code snippets, observations, or tool outputs.\n\n"
            "A SHA-256 fingerprint is generated for deduplication; ingesting the same content "
            "twice returns the existing ID instead of creating a duplicate. "
            "Embedding computation is deferred to first retrieval."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Raw text content of the source (required)",
                },
                "source_type": {
                    "type": "string",
                    "enum": [
                        "document",
                        "conversation",
                        "web",
                        "code",
                        "observation",
                        "tool_output",
                        "user_input",
                    ],
                    "description": (
                        "Source type determines initial reliability score: "
                        "document/code=0.8, web=0.6, conversation=0.5, "
                        "observation=0.4, tool_output=0.7, user_input=0.75"
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Optional human-readable title",
                },
                "url": {
                    "type": "string",
                    "description": "Optional canonical URL for web sources",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional arbitrary metadata (JSON object)",
                },
            },
            "required": ["content", "source_type"],
        },
    ),
    Tool(
        name="source_get",
        description="Get a source by ID with full details including content and metadata.",
        inputSchema={
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "UUID of the source",
                },
            },
            "required": ["source_id"],
        },
    ),
    Tool(
        name="source_search",
        description=(
            "Full-text search over source content.\n\n"
            "Uses PostgreSQL ``websearch_to_tsquery`` over the GIN-indexed ``content_tsv`` column. "
            "Results ordered by relevance descending."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms (natural language or keyword phrase)",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum results (default 20, max 200)",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="source_list",
        description="List sources with optional filters. Returns source metadata without full content.",
        inputSchema={
            "type": "object",
            "properties": {
                "source_type": {
                    "type": "string",
                    "description": "Filter by source type (document, conversation, web, code, observation, tool_output, user_input)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 50, max 200)",
                    "default": 50,
                },
            },
        },
    ),
    # =========================================================================
    # Retrieval (C9)
    # =========================================================================
    Tool(
        name="knowledge_search",
        description=(
            "Unified knowledge retrieval — search articles and optionally raw sources.\n\n"
            "CRITICAL: Call this BEFORE answering questions about any topic that may have "
            "been discussed, documented, or learned previously. This ensures responses are "
            "grounded in accumulated knowledge.\n\n"
            "Results are ranked by: relevance × 0.5 + confidence × 0.35 + freshness × 0.15.\n\n"
            "Ungrouped raw sources matching the query are surfaced and automatically queued "
            "for compilation into articles. Usage is recorded for self-organisation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum results to return (default 10, max 200)",
                },
                "include_sources": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include ungrouped raw sources alongside compiled articles",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID for usage trace attribution",
                },
                "epistemic_type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural"],
                    "description": "Filter results by epistemic type",
                },
            },
            "required": ["query"],
        },
    ),
    # =========================================================================
    # Article tools (C2)
    # =========================================================================
    Tool(
        name="article_get",
        description=(
            "Get an article by ID, optionally with its full provenance list.\n\n"
            "Set ``include_provenance=true`` to see all linked sources and their "
            "relationship types (originates, confirms, supersedes, contradicts, contends)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "string",
                    "description": "UUID of the article",
                },
                "include_provenance": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include linked source provenance in the response",
                },
            },
            "required": ["article_id"],
        },
    ),
    Tool(
        name="article_create",
        description=(
            "Manually create a new knowledge article.\n\n"
            "Use this when you want to create an article directly without LLM compilation. "
            "For compilation from sources, use ``article_compile`` instead.\n\n"
            "Optionally link originating source UUIDs — they will be linked with "
            "relationship='originates'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Article body text (required)",
                },
                "title": {
                    "type": "string",
                    "description": "Optional human-readable title",
                },
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "UUIDs of source documents this article originates from",
                },
                "author_type": {
                    "type": "string",
                    "enum": ["system", "operator", "agent"],
                    "default": "system",
                    "description": "Who authored this article",
                },
                "domain_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hierarchical domain tags (e.g. ['python', 'stdlib'])",
                },
                "epistemic_type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural"],
                    "default": "semantic",
                    "description": "Knowledge type: episodic (decays), semantic (persists), procedural (pinned)",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="article_compile",
        description=(
            "Compile one or more sources into a new knowledge article using LLM summarization.\n\n"
            "The LLM produces a coherent, right-sized article from the given source documents. "
            "All sources are linked to the resulting article with appropriate provenance "
            "relationship types (originates, confirms, supersedes, contradicts, contends).\n\n"
            "The compiled article respects right-sizing bounds from system_config "
            "(default: 300–800 tokens, target 550)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "UUIDs of source documents to compile (required, non-empty)",
                },
                "title_hint": {
                    "type": "string",
                    "description": "Optional hint for the article title",
                },
            },
            "required": ["source_ids"],
        },
    ),
    Tool(
        name="article_update",
        description=(
            "Update an article's content with new material.\n\n"
            "Increments the article version, records an 'updated' mutation, and "
            "optionally links the triggering source. The source is linked with a "
            "relationship type inferred from content (typically 'confirms' or 'supersedes')."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "string",
                    "description": "UUID of the article to update",
                },
                "content": {
                    "type": "string",
                    "description": "New article body text",
                },
                "source_id": {
                    "type": "string",
                    "description": "Optional UUID of the source that triggered this update",
                },
                "epistemic_type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural"],
                    "description": "Optional new epistemic type classification",
                },
            },
            "required": ["article_id", "content"],
        },
    ),
    # =========================================================================
    # Right-sizing tools (C3)
    # =========================================================================
    Tool(
        name="article_split",
        description=(
            "Split an oversized article into two smaller articles.\n\n"
            "The original article retains its ID and the first half of the content. "
            "A new article is created for the remainder. Both inherit all provenance "
            "sources, and mutation records of type 'split' are written for both."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "string",
                    "description": "UUID of the article to split",
                },
            },
            "required": ["article_id"],
        },
    ),
    Tool(
        name="article_merge",
        description=(
            "Merge two related articles into one.\n\n"
            "A new article is created with combined content. Both originals are archived. "
            "The merged article inherits the union of provenance sources from both. "
            "Mutation records of type 'merged' are written."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "article_id_a": {
                    "type": "string",
                    "description": "UUID of the first article",
                },
                "article_id_b": {
                    "type": "string",
                    "description": "UUID of the second article",
                },
            },
            "required": ["article_id_a", "article_id_b"],
        },
    ),
    Tool(
        name="article_search",
        description=(
            "Search articles by query, domain, or filters. Returns ranked article results. "
            "For unified search across articles AND sources, use knowledge_search instead."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text",
                },
                "domain": {
                    "type": "string",
                    "description": "Filter by domain path prefix",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 10, max 200)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    # =========================================================================
    # Provenance (C5)
    # =========================================================================
    Tool(
        name="provenance_trace",
        description=(
            "Trace which sources likely contributed a specific claim in an article.\n\n"
            "Uses text-similarity (TF-IDF) to rank the article's linked sources by "
            "how much their content overlaps with the given claim text. "
            "Useful for attribution and fact-checking."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "string",
                    "description": "UUID of the article",
                },
                "claim_text": {
                    "type": "string",
                    "description": "The specific claim or sentence to trace back to sources",
                },
            },
            "required": ["article_id", "claim_text"],
        },
    ),
    Tool(
        name="provenance_get",
        description="Get all provenance links for an article — which sources contributed and how.",
        inputSchema={
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "string",
                    "description": "UUID of the article",
                },
            },
            "required": ["article_id"],
        },
    ),
    Tool(
        name="provenance_link",
        description=("Link a source to an article with a relationship type. Types: originates, confirms, supersedes, contradicts, contends."),
        inputSchema={
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "string",
                    "description": "UUID of the article",
                },
                "source_id": {
                    "type": "string",
                    "description": "UUID of the source to link",
                },
                "relationship": {
                    "type": "string",
                    "enum": ["originates", "confirms", "supersedes", "contradicts", "contends"],
                    "description": "Relationship type between source and article",
                },
            },
            "required": ["article_id", "source_id", "relationship"],
        },
    ),
    # =========================================================================
    # Contention tools (C7)
    # =========================================================================
    Tool(
        name="contention_list",
        description=(
            "List active contentions (contradictions or disagreements) in the knowledge base.\n\n"
            "Contentions arise when a source contradicts or contends with an existing article. "
            "Review contentions to identify knowledge that needs reconciliation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "string",
                    "description": "Optional UUID — return only contentions for this article",
                },
                "status": {
                    "type": "string",
                    "enum": ["detected", "resolved", "dismissed"],
                    "description": "Filter by status (omit to return all)",
                },
            },
        },
    ),
    Tool(
        name="contention_resolve",
        description=(
            "Resolve a contention between an article and a source.\n\n"
            "Resolution types:\n"
            "- ``supersede_a``: Article wins; source is noted but article unchanged.\n"
            "- ``supersede_b``: Source wins; article content is replaced.\n"
            "- ``accept_both``: Both perspectives are valid; article is annotated.\n"
            "- ``dismiss``: Not material; dismissed without change."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "contention_id": {
                    "type": "string",
                    "description": "UUID of the contention to resolve",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["supersede_a", "supersede_b", "accept_both", "dismiss"],
                    "description": "Resolution type",
                },
                "rationale": {
                    "type": "string",
                    "description": "Free-text rationale recorded on the contention",
                },
            },
            "required": ["contention_id", "resolution", "rationale"],
        },
    ),
    Tool(
        name="contention_detect",
        description=(
            "Detect potential contradictions between articles using semantic similarity. "
            "Optionally auto-record detected contradictions as contentions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Minimum similarity threshold (default 0.85)",
                    "default": 0.85,
                },
                "auto_record": {
                    "type": "boolean",
                    "description": "Automatically record detected contradictions as contentions",
                    "default": False,
                },
            },
        },
    ),
    # =========================================================================
    # Admin tools (C10, health)
    # =========================================================================
    Tool(
        name="admin_forget",
        description=(
            "Permanently remove a source or article from the knowledge system (C10).\n\n"
            "For sources: deletes the source, cascades to article_sources, queues "
            "affected articles for recompilation, creates a tombstone.\n\n"
            "For articles: deletes the article and provenance links; sources are unaffected; "
            "a tombstone is created.\n\n"
            "This operation is IRREVERSIBLE."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "target_type": {
                    "type": "string",
                    "enum": ["source", "article"],
                    "description": "Whether to delete a source or an article",
                },
                "target_id": {
                    "type": "string",
                    "description": "UUID of the record to delete",
                },
            },
            "required": ["target_type", "target_id"],
        },
    ),
    Tool(
        name="admin_stats",
        description=(
            "Return health and capacity statistics for the knowledge system.\n\n"
            "Includes: article counts (total/active/pinned), source count, "
            "pending mutation queue depth, tombstones (last 30 days), "
            "and bounded-memory capacity utilization."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="admin_maintenance",
        description=(
            "Trigger maintenance operations for the knowledge system.\n\n"
            "Available operations (pass true to enable):\n"
            "- ``recompute_scores``: Batch-recompute usage_score for all articles.\n"
            "- ``process_queue``: Process pending entries in mutation_queue "
            "(recompile, split, merge_candidate, decay_check).\n"
            "- ``evict_if_over_capacity``: Run organic forgetting if article count "
            "exceeds the configured maximum (from system_config.bounded_memory)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "recompute_scores": {
                    "type": "boolean",
                    "default": False,
                    "description": "Batch-recompute usage scores for all articles",
                },
                "process_queue": {
                    "type": "boolean",
                    "default": False,
                    "description": "Process pending entries in the mutation queue",
                },
                "evict_if_over_capacity": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run organic eviction if over capacity",
                },
                "evict_count": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum articles to evict per run (default 10)",
                },
            },
        },
    ),
    # =========================================================================
    # Memory tools (agent-friendly interface)
    # =========================================================================
    Tool(
        name="memory_store",
        description=(
            "Store a memory for later recall (agent-friendly wrapper).\n\n"
            "Memories are stored as observation sources with special metadata "
            "that makes them easy for agents to search and manage. Use this to "
            "remember important facts, learnings, decisions, or observations.\n\n"
            "Memories can supersede previous memories and are tagged with importance "
            "and optional context tags for better retrieval."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content (required)",
                },
                "context": {
                    "type": "string",
                    "description": "Where this memory came from (e.g., 'session:main', 'conversation:user', 'observation:system')",
                },
                "importance": {
                    "type": "number",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "How important this memory is (0.0-1.0, default 0.5)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional categorization tags (e.g., ['infrastructure', 'decision'])",
                },
                "supersedes_id": {
                    "type": "string",
                    "description": "UUID of a previous memory this replaces",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="memory_recall",
        description=(
            "Search and recall memories (agent-friendly wrapper).\n\n"
            "Returns memories ranked by relevance, confidence, and freshness. "
            "Results are filtered to only include observation sources marked as memories. "
            "Optionally filter by tags or minimum confidence threshold.\n\n"
            "Use this to retrieve relevant past knowledge before making decisions "
            "or answering questions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to recall (natural language query)",
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum results to return (default 5, max 50)",
                },
                "min_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Optional minimum confidence threshold (0.0-1.0)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tag filter — only return memories with at least one matching tag",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_status",
        description=(
            "Get statistics about the memory system.\n\n"
            "Returns count of stored memories, articles compiled from them, "
            "last memory timestamp, and top tags. Use this to understand "
            "the current state of the memory system."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="memory_forget",
        description=(
            "Mark a memory as forgotten (soft delete).\n\n"
            "Sets the memory's metadata to include a 'forgotten' flag and optional reason. "
            "The memory is not actually deleted from the database, but will be filtered "
            "out of future recall results.\n\n"
            "Use this to mark outdated or incorrect memories without losing the audit trail."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "UUID of the memory (source) to forget",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional reason why this memory is being forgotten",
                },
            },
            "required": ["memory_id"],
        },
    ),
    # =========================================================================
    # Session tools (conversation ingestion)
    # =========================================================================
    Tool(
        name="session_start",
        description=(
            "Upsert a session (insert-if-new or update last_activity_at).\n\n"
            "Sessions are first-class sources that buffer conversation messages in the database. "
            "Call this when a session begins or resumes. If the session already exists, "
            "its last_activity_at timestamp is updated."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Unique session identifier (platform-provided, required)",
                },
                "platform": {
                    "type": "string",
                    "description": "Platform name (e.g., 'openclaw', 'claude-code', required)",
                },
                "channel": {
                    "type": "string",
                    "description": "Optional channel (e.g., 'discord', 'telegram', 'cli')",
                },
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of participant names",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional JSON metadata",
                },
                "parent_session_id": {
                    "type": "string",
                    "description": "Parent session ID for subagents",
                },
                "subagent_label": {
                    "type": "string",
                    "description": "Label for subagent sessions",
                },
                "subagent_model": {
                    "type": "string",
                    "description": "Model used for subagent",
                },
                "subagent_task": {
                    "type": "string",
                    "description": "Task description for subagent",
                },
            },
            "required": ["session_id", "platform"],
        },
    ),
    Tool(
        name="session_append",
        description=(
            "Append message(s) to a session buffer.\n\n"
            "Supports two modes:\n"
            "1. Batch mode: pass 'messages' as a list of dicts with keys: speaker, role, content, metadata?\n"
            "2. Single mode: pass speaker, role, content directly\n\n"
            "Updates the session's last_activity_at timestamp."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier (required)",
                },
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string"},
                            "role": {"type": "string", "enum": ["user", "assistant", "system", "tool"]},
                            "content": {"type": "string"},
                            "metadata": {"type": "object"},
                        },
                        "required": ["speaker", "role", "content"],
                    },
                    "description": "List of message dicts (batch mode)",
                },
                "speaker": {
                    "type": "string",
                    "description": "Speaker name (single mode)",
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant", "system", "tool"],
                    "description": "Message role (single mode)",
                },
                "content": {
                    "type": "string",
                    "description": "Message content (single mode)",
                },
                "message_metadata": {
                    "type": "object",
                    "description": "Optional message-specific metadata (single mode)",
                },
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="session_flush",
        description=(
            "Flush unflushed messages to a conversation source.\n\n"
            "Serializes buffered messages to markdown transcript format, "
            "ingests as a conversation source, marks messages as flushed, "
            "and increments the session's chunk_index.\n\n"
            "Optionally triggers compilation into an article."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier (required)",
                },
                "compile": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to trigger compilation after flush",
                },
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="session_finalize",
        description=(
            "Flush final messages and mark session as completed.\n\n"
            "Flushes any remaining unflushed messages, marks the session status "
            "as 'completed', sets ended_at timestamp, and triggers compilation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier (required)",
                },
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="session_search",
        description=(
            "Semantic search over conversation sources.\n\n"
            "Searches for conversation-type sources matching the query. "
            "Returns source records with session metadata."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (required)",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum results to return (default 20)",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="session_list",
        description=(
            "List sessions with optional filters.\n\nReturns session records matching the specified filters, ordered by last_activity_at descending."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "stale", "completed"],
                    "description": "Filter by status",
                },
                "platform": {
                    "type": "string",
                    "description": "Filter by platform",
                },
                "since": {
                    "type": "string",
                    "description": "Filter by started_at >= since (ISO timestamp)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                },
            },
        },
    ),
    Tool(
        name="session_get",
        description=("Get session details with optional messages.\n\nReturns the session record and optionally all messages in the buffer."),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier (required)",
                },
                "include_messages": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include messages in the response",
                },
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="session_compile",
        description=(
            "Compile session sources into an article.\n\n"
            "Finds all conversation sources for the given session and compiles "
            "them into a coherent knowledge article using LLM summarization."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier (required)",
                },
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="session_flush_stale",
        description=(
            "Flush all stale sessions (no activity for stale_minutes).\n\n"
            "Finds active sessions with no activity for the specified duration, "
            "flushes them to sources, marks them as stale, and triggers compilation.\n\n"
            "This is typically called periodically by a cron job or timer."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "stale_minutes": {
                    "type": "integer",
                    "default": 30,
                    "description": "Inactivity threshold in minutes (default 30)",
                },
            },
        },
    ),
]
