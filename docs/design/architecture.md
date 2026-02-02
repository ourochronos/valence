# Valence Architecture Design

## Overview

Valence is a personal knowledge substrate with Claude Code at its core. Rather than building a separate bot that calls Claude, we make Claude Code itself the agent through:

1. **Plugin** - Behavioral conditioning and skills
2. **MCP Servers** - Knowledge substrate access
3. **Hooks** - Conversation capture and context injection
4. **Sessions** - Continuity via resume

## Core Insight: Claude Code AS the Agent

The key architectural insight is that Claude Code's session resumption, plugins, and MCP integration allow us to turn it into a persistent, context-aware agent without building wrapper infrastructure.

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INTERFACES                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Claude Code │  │   Matrix    │  │   Future    │         │
│  │  (Direct)   │  │    Bot      │  │  Clients    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                   VALENCE PLUGIN                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SessionStart Hook: Inject context + conditioning    │    │
│  │ - Load relevant beliefs from substrate              │    │
│  │ - Set behavioral rules (query before answering)     │    │
│  │ - Establish session in VKB                          │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Skills                                              │    │
│  │ - /valence:query-knowledge                         │    │
│  │ - /valence:capture-insight                         │    │
│  │ - /valence:ingest-document                         │    │
│  │ - /valence:review-tensions                         │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ PostToolUse Hook: Capture exchanges                 │    │
│  │ SessionEnd Hook: Close session, extract patterns    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                   MCP SERVERS                                │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ valence-substrate │  │ valence-vkb      │                │
│  │ (Knowledge Base)  │  │ (Conversations)  │                │
│  │                   │  │                  │                │
│  │ - belief_query    │  │ - session_*      │                │
│  │ - belief_create   │  │ - exchange_*     │                │
│  │ - entity_*        │  │ - pattern_*      │                │
│  │ - tension_*       │  │ - insight_*      │                │
│  └────────┬──────────┘  └────────┬─────────┘                │
└───────────┼──────────────────────┼──────────────────────────┘
            │                      │
            ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   POSTGRESQL + pgvector                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Unified Schema                                        │   │
│  │ - beliefs (with dimensional confidence)              │   │
│  │ - entities (with aliases, relationships)             │   │
│  │ - sources (provenance tracking)                      │   │
│  │ - sessions, exchanges, patterns (VKB)               │   │
│  │ - embeddings (pgvector for similarity search)        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. User Sovereignty
- All data stored locally or on user-controlled infrastructure
- No external services required except embedding APIs
- Federation support designed in but optional

### 2. Context Efficiency
- Hooks inject minimal relevant context at session start
- MCP tools allow on-demand knowledge access
- Session resumption maintains conversation history

### 3. Gradual Adoption
- Plugin can be enabled/disabled without losing data
- MCP servers work standalone
- Skills provide guided workflows for common tasks

### 4. Dimensional Confidence
Instead of a single confidence score, beliefs have multiple dimensions:
- **Source Reliability**: How trustworthy is the source?
- **Method Quality**: How rigorous was the extraction?
- **Internal Consistency**: Does it align with other beliefs?
- **Temporal Freshness**: How recent is this information?
- **Corroboration**: Is it supported by multiple sources?
- **Domain Applicability**: How relevant is it to current context?

### 5. Temporal Validity
Beliefs can have:
- `valid_from`: When the belief became true
- `valid_until`: When it stopped being true (null = still valid)
- `supersedes_id`: What belief this replaced

## Data Flow

### Session Start
1. Hook fires on session start
2. Query substrate for relevant beliefs (based on project context)
3. Query VKB for established patterns
4. Inject context into Claude's prompt
5. Create new VKB session

### During Session
1. User sends message
2. Claude responds (may use MCP tools)
3. PostToolUse hook captures exchanges
4. Embeddings generated asynchronously

### Session End
1. Hook fires on session end
2. Summarize session themes
3. Extract patterns if detected
4. Mark session as completed

## MCP Server Design

### valence-substrate (Knowledge)
Primary tools for knowledge management:

| Tool | Purpose |
|------|---------|
| `belief_query` | Search beliefs by content, domain, entity |
| `belief_create` | Store new belief with confidence and source |
| `belief_supersede` | Replace old belief with new (maintains history) |
| `entity_get` | Get entity details and related beliefs |
| `entity_search` | Find entities by name or type |
| `tension_list` | List contradictions/tensions in beliefs |
| `tension_resolve` | Mark tension as resolved with explanation |

### valence-vkb (Conversations)
Primary tools for conversation tracking:

| Tool | Purpose |
|------|---------|
| `session_start` | Begin a new session |
| `session_end` | Close session with summary |
| `session_get` | Get session details |
| `exchange_add` | Record a conversation turn |
| `exchange_list` | Get session exchanges |
| `pattern_record` | Record a new behavioral pattern |
| `pattern_reinforce` | Strengthen existing pattern |
| `insight_extract` | Extract insight from session to KB |

## Plugin Structure

```
plugin/
├── .claude-plugin/
│   └── plugin.json          # Plugin manifest
├── hooks/
│   ├── hooks.json           # Hook configuration
│   ├── session-start.py     # Context injection
│   └── session-end.py       # Cleanup
├── skills/
│   ├── using-valence/SKILL.md
│   ├── query-knowledge/SKILL.md
│   ├── capture-insight/SKILL.md
│   ├── ingest-document/SKILL.md
│   └── review-tensions/SKILL.md
└── .mcp.json                # Plugin-bundled MCP servers
```

## Session Management Strategy

### Direct Claude Code Usage
- Session state maintained by Claude Code itself
- Hooks provide context injection at start
- `--resume` flag continues sessions

### Matrix Bot Integration
```python
# Store session IDs per room
room_sessions: dict[str, str] = {}  # room_id -> claude_session_id

# First message in room
result = subprocess.run([
    "claude", "-p", message,
    "--plugin-dir", "/opt/valence/plugin",
    "--output-format", "json"
])
session_id = json.loads(result.stdout)["session_id"]
room_sessions[room_id] = session_id

# Subsequent messages
result = subprocess.run([
    "claude", "-p", message,
    "--resume", room_sessions[room_id],
    "--plugin-dir", "/opt/valence/plugin"
])
```

## Future Considerations

### Federation
- Design for single-user first
- Schema supports scope/visibility fields
- Entity resolution handles cross-source identity

### L0 Storage
- Raw transcripts stored separately from curated beliefs
- Enables reprocessing with improved extraction
- Storage in separate PostgreSQL table or file system

### Multi-Model Embeddings
- Embedding registry supports multiple models
- Lazy backfilling when new models added
- Fallback hierarchy for searches
