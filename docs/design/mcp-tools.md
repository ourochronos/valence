# Valence MCP Tool Design

## Overview

Valence exposes a **unified MCP server** that provides all tools in a single process. The `VALENCE_MODE` environment variable controls which tools are exposed:

- **`personal`** (16 core tools) — For personal knowledge management
- **`connected`** (+ trust/federation tools) — For connected deployments
- **`full`** (all 25 tools, default) — Everything

Legacy: The individual `valence-substrate` and `valence-vkb` stdio servers are still available for backward compatibility but the unified server is the recommended approach.

## Design Principles

### Tool Naming
- Use `snake_case` for tool names
- Prefix with noun (e.g., `belief_`, `session_`)
- Verb at end (e.g., `_query`, `_create`, `_list`)

### Response Format
- Always return JSON
- Include `success: boolean` in all responses
- Include meaningful error messages
- Return created/updated objects in full

### Error Handling
- Return structured error objects, don't throw
- Include `error_code` for programmatic handling
- Include `error_message` for human reading

## Knowledge Substrate Tools

### belief_query
Search beliefs by content, domain, or entity. Uses hybrid search (keyword + semantic).
Supports `ranking` param with configurable weights (`semantic_weight`, `confidence_weight`, `recency_weight`) and `explain` mode.

### belief_create
Create a new belief with optional entity links. **Deduplicates automatically**: if an exact content hash or high-similarity embedding match is found, reinforces the existing belief instead of creating a duplicate.

### belief_supersede
Replace an old belief with a new one, maintaining the full supersession history chain.

### belief_get
Get a single belief by ID with full details. Supports `include_history` (supersession chain) and `include_tensions`.

### belief_search
Semantic search for beliefs using vector embeddings. Best for finding conceptually related beliefs even with different wording. Supports `ranking` param.

### entity_get
Get entity details with optional related beliefs.

### entity_search
Find entities by name or type.

### tension_list
List contradictions/tensions between beliefs. Filter by status, severity, or entity.

### tension_resolve
Mark a tension as resolved with explanation. Actions: supersede_a, supersede_b, keep_both, archive_both.

### belief_corroboration
Get corroboration details for a belief — how many independent sources confirm it. Returns count, confidence from corroboration, and source details.

### trust_check
Check trust levels for entities or federation nodes on a specific topic/domain. Returns entities with high-confidence beliefs and trusted federation nodes.

### confidence_explain
Explain why a belief has a particular confidence score. Shows all contributing dimensions with weights and recommendations.

## Conversation Tracking Tools

### session_start / session_end / session_get / session_list
Manage conversation sessions. Sessions track platform, project context, summary, and themes.

### session_find_by_room
Find active session by external room ID (for chat platform integrations).

### exchange_add / exchange_list
Record and retrieve individual conversation turns within a session.

### pattern_record / pattern_reinforce / pattern_list / pattern_search
Track behavioral patterns across sessions. Patterns emerge from repeated topics, preferences, and working styles.

### insight_extract / insight_list
Extract insights from sessions and create beliefs in the knowledge base. **Deduplicates automatically** via content hash.

## Tool Invocation Patterns

### Query First Pattern
For questions that might be in the KB:
1. `belief_query("user's preferred deployment strategy")`
2. If results found, use them in response
3. If not, answer from training and optionally capture new belief

### Capture on Insight Pattern
When user shares important information:
1. Recognize valuable information in user message
2. `insight_extract(session_id, content, domain_path)`
3. Dedup handles the case where this was already captured

### Tension Detection Pattern
After creating beliefs:
1. `belief_create(...)` — may deduplicate
2. `tension_list(entity_id=related_entity)`
3. If tensions found, alert user or auto-resolve
