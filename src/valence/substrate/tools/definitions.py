"""Substrate tool definitions with behavioral conditioning.

Contains SUBSTRATE_TOOLS -- the list of all Tool() definitions for the substrate.
"""

from __future__ import annotations

from mcp.types import Tool

SUBSTRATE_TOOLS = [
    Tool(
        name="belief_query",
        description=(
            "Search beliefs by content, domain, or entity. Uses hybrid search (keyword + semantic).\n\n"
            "CRITICAL: You MUST call this BEFORE answering questions about:\n"
            "- Past decisions or discussions\n"
            "- User preferences or values\n"
            "- Technical approaches previously explored\n"
            "- Any topic that may have been discussed before\n\n"
            "Query first, then respond with grounded information. This ensures your "
            "responses are consistent with what has been learned and decided previously.\n\n"
            "Note: Beliefs with revoked consent chains are filtered out by default for privacy."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "domain_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by domain path (e.g., ['tech', 'architecture'])",
                },
                "entity_id": {
                    "type": "string",
                    "description": "Filter by related entity UUID",
                },
                "include_superseded": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include superseded beliefs",
                },
                "include_revoked": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include beliefs with revoked consent chains (requires audit logging)",
                },
                "include_archived": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include archived beliefs in results. Archived beliefs are kept for provenance and reference but excluded from active queries by default.",
                },
                "include_expired": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include beliefs outside their temporal validity window (valid_from/valid_until)",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum results",
                },
                "ranking": {
                    "type": "object",
                    "description": "Configure result ranking weights",
                    "properties": {
                        "semantic_weight": {"type": "number", "default": 0.50, "description": "Weight for semantic relevance (0-1)"},
                        "confidence_weight": {"type": "number", "default": 0.35, "description": "Weight for belief confidence (0-1)"},
                        "recency_weight": {"type": "number", "default": 0.15, "description": "Weight for recency (0-1)"},
                        "explain": {"type": "boolean", "default": False, "description": "Include score breakdown in results"},
                    },
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="belief_create",
        description=(
            "Create a new belief with optional entity links.\n\n"
            "Use PROACTIVELY when:\n"
            "- A decision is made with clear rationale\n"
            "- User expresses a preference or value\n"
            "- A problem is solved with a novel approach\n"
            "- Important factual information is shared\n"
            "- Architectural or design choices are finalized\n\n"
            "Capturing beliefs ensures future conversations have access to this knowledge."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The belief content - should be a clear, factual statement",
                },
                "confidence": {
                    "type": "object",
                    "description": "Confidence dimensions (or single 'overall' value)",
                    "default": {"overall": 0.7},
                },
                "domain_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Domain classification (e.g., ['tech', 'python', 'testing'])",
                },
                "source_type": {
                    "type": "string",
                    "enum": [
                        "document",
                        "conversation",
                        "inference",
                        "observation",
                        "user_input",
                    ],
                    "description": "Type of source",
                },
                "source_ref": {
                    "type": "string",
                    "description": "Reference to source (URL, session_id, etc.)",
                },
                "opt_out_federation": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, belief will not be shared via federation (privacy opt-out)",
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "role": {
                                "type": "string",
                                "enum": ["subject", "object", "context"],
                            },
                        },
                        "required": ["name"],
                    },
                    "description": "Entities to link (will be created if not exist)",
                },
                "visibility": {
                    "type": "string",
                    "enum": ["private", "federated", "public"],
                    "default": "private",
                    "description": "Visibility level for the belief",
                },
                "sharing_intent": {
                    "type": "string",
                    "enum": ["know_me", "work_with_me", "learn_from_me", "use_this"],
                    "description": "Optional sharing intent — generates a SharePolicy stored with the belief",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="belief_supersede",
        description=(
            "Replace an old belief with a new one, maintaining history.\n\n"
            "Use when:\n"
            "- Information needs to be updated or corrected\n"
            "- A previous decision has been revised\n"
            "- More accurate information is now available\n\n"
            "This maintains the full history chain so we can understand how knowledge evolved."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "old_belief_id": {
                    "type": "string",
                    "description": "UUID of belief to supersede",
                },
                "new_content": {
                    "type": "string",
                    "description": "Updated belief content",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this belief is being superseded",
                },
                "confidence": {
                    "type": "object",
                    "description": "Confidence for new belief",
                },
            },
            "required": ["old_belief_id", "new_content", "reason"],
        },
    ),
    Tool(
        name="belief_get",
        description=(
            "Get a single belief by ID with full details.\n\n"
            "Use to examine a specific belief's content, history, and related tensions "
            "when you need more context than what belief_query provides."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "belief_id": {
                    "type": "string",
                    "description": "UUID of the belief",
                },
                "include_history": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include supersession chain",
                },
                "include_tensions": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include related tensions",
                },
            },
            "required": ["belief_id"],
        },
    ),
    Tool(
        name="entity_get",
        description=(
            "Get entity details with optional beliefs.\n\n"
            "Use when you need comprehensive information about a person, tool, "
            "concept, or organization that has been discussed before."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "UUID of the entity",
                },
                "include_beliefs": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include related beliefs",
                },
                "belief_limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max beliefs to include",
                },
            },
            "required": ["entity_id"],
        },
    ),
    Tool(
        name="entity_search",
        description=(
            "Find entities by name or type.\n\n"
            "Use to discover what's known about specific people, tools, projects, "
            "or concepts before making statements about them."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (matches name and aliases)",
                },
                "type": {
                    "type": "string",
                    "enum": [
                        "person",
                        "organization",
                        "tool",
                        "concept",
                        "project",
                        "location",
                        "service",
                    ],
                    "description": "Filter by entity type",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="tension_list",
        description=(
            "List contradictions/tensions between beliefs.\n\n"
            "Review tensions periodically to identify knowledge that needs "
            "reconciliation or clarification."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["detected", "investigating", "resolved", "accepted"],
                    "description": "Filter by status",
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Minimum severity",
                },
                "entity_id": {
                    "type": "string",
                    "description": "Tensions involving this entity",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                },
            },
        },
    ),
    Tool(
        name="tension_resolve",
        description=("Mark a tension as resolved with explanation.\n\nUse when you've determined how to reconcile conflicting beliefs."),
        inputSchema={
            "type": "object",
            "properties": {
                "tension_id": {
                    "type": "string",
                    "description": "UUID of the tension",
                },
                "resolution": {
                    "type": "string",
                    "description": "How the tension was resolved",
                },
                "action": {
                    "type": "string",
                    "enum": ["supersede_a", "supersede_b", "keep_both", "archive_both"],
                    "description": "What to do with the beliefs",
                },
            },
            "required": ["tension_id", "resolution", "action"],
        },
    ),
    Tool(
        name="belief_search",
        description=(
            "Semantic search for beliefs using vector embeddings.\n\n"
            "Best for finding conceptually related beliefs even with different wording. "
            "Use this instead of belief_query when:\n"
            "- The exact keywords may not match but the concept is the same\n"
            "- You want to find beliefs that are semantically similar\n"
            "- You need to discover related knowledge that uses different terminology\n\n"
            "Requires embeddings to be enabled (OPENAI_API_KEY)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to find semantically similar beliefs",
                },
                "min_similarity": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Minimum similarity threshold (0-1)",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Filter by minimum overall confidence",
                },
                "domain_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by domain path",
                },
                "include_archived": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include archived beliefs in results. Archived beliefs are kept for provenance and reference but excluded from active queries by default.",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum results",
                },
                "ranking": {
                    "type": "object",
                    "description": "Configure result ranking weights",
                    "properties": {
                        "semantic_weight": {"type": "number", "default": 0.50, "description": "Weight for semantic similarity (0-1)"},
                        "confidence_weight": {"type": "number", "default": 0.35, "description": "Weight for belief confidence (0-1)"},
                        "recency_weight": {"type": "number", "default": 0.15, "description": "Weight for recency (0-1)"},
                        "explain": {"type": "boolean", "default": False, "description": "Include score breakdown in results"},
                    },
                },
            },
            "required": ["query"],
        },
    ),
    # -------------------------------------------------------------------------
    # Article tools (WU-04 — replaces belief tools)
    # -------------------------------------------------------------------------
    Tool(
        name="article_create",
        description=(
            "Create a new knowledge article compiled from one or more sources.\n\n"
            "Use this to record synthesised, compiled knowledge that may draw on "
            "multiple source documents or conversations."
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
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="article_get",
        description=(
            "Get an article by ID, optionally with its full provenance list."
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
        name="article_update",
        description=(
            "Update an article's content. Increments version and records an 'updated' mutation."
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
            },
            "required": ["article_id", "content"],
        },
    ),
    Tool(
        name="article_search",
        description=(
            "Search articles via full-text and semantic search.\n\n"
            "Returns articles ordered by relevance. Uses both keyword matching and "
            "vector similarity when embeddings are available."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results (max 50)",
                },
                "domain_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional domain path segments to restrict results",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="provenance_link",
        description=(
            "Link a source to an article with a provenance relationship.\n\n"
            "Relationship types: originates, confirms, supersedes, contradicts, contends"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "article_id": {"type": "string", "description": "UUID of the article"},
                "source_id": {"type": "string", "description": "UUID of the source"},
                "relationship": {
                    "type": "string",
                    "enum": ["originates", "confirms", "supersedes", "contradicts", "contends"],
                    "description": "Relationship type",
                },
                "notes": {"type": "string", "description": "Optional notes about the relationship"},
            },
            "required": ["article_id", "source_id", "relationship"],
        },
    ),
    Tool(
        name="provenance_get",
        description="Get the full provenance list for an article.",
        inputSchema={
            "type": "object",
            "properties": {
                "article_id": {"type": "string", "description": "UUID of the article"},
            },
            "required": ["article_id"],
        },
    ),
    Tool(
        name="provenance_trace",
        description=(
            "Trace which sources likely contributed a specific claim in an article.\n\n"
            "Uses text similarity to rank sources by relevance to the claim."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "article_id": {"type": "string", "description": "UUID of the article"},
                "claim_text": {"type": "string", "description": "The specific claim to trace"},
            },
            "required": ["article_id", "claim_text"],
        },
    ),
]
