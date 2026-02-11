# Valence Architecture Design

## Overview

Valence is a personal knowledge substrate with Claude Code at its core. Rather than building a separate bot that calls Claude, we make Claude Code itself the agent through:

1. **Plugin** - Behavioral conditioning and skills
2. **MCP Server** - Unified knowledge substrate access
3. **Hooks** - Conversation capture and context injection
4. **Sessions** - Continuity via resume

## Core Insight: Claude Code AS the Agent

The key architectural insight is that Claude Code's session resumption, plugins, and MCP integration allow us to turn it into a persistent, context-aware agent without building wrapper infrastructure.

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INTERFACES                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Claude Code │  │  Telegram   │  │   Future    │         │
│  │  (Direct)   │  │    Bot      │  │  Clients    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                   VALENCE PLUGIN                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SessionStart Hook: Inject context + conditioning    │    │
│  │ SessionEnd Hook: Close session, auto-capture beliefs│    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Skills                                              │    │
│  │ - /valence:using-valence                            │    │
│  │ - /valence:query-knowledge                          │    │
│  │ - /valence:capture-insight                          │    │
│  │ - /valence:review-tensions                          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│               UNIFIED MCP SERVER (valence)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Knowledge Substrate        │ Conversation Tracking   │   │
│  │ - belief_query/create/get  │ - session_*             │   │
│  │ - belief_search/supersede  │ - exchange_*            │   │
│  │ - entity_*                 │ - pattern_*             │   │
│  │ - tension_*                │ - insight_*             │   │
│  │ - trust_check              │                         │   │
│  │ - confidence_explain       │                         │   │
│  │ - belief_corroboration     │                         │   │
│  └──────────────────────────────────────────────────────┘   │
│  VALENCE_MODE: personal | connected | full                   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                   POSTGRESQL + pgvector                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Unified Schema                                        │   │
│  │ - beliefs (dimensional confidence, 384-dim embeddings)│   │
│  │ - entities (with aliases, relationships)              │   │
│  │ - sources (provenance tracking)                       │   │
│  │ - sessions, exchanges, patterns (VKB)                 │   │
│  │ - belief_corroborations (dedup/reinforcement)         │   │
│  │ - federation tables (nodes, trust, sync)              │   │
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
- MCP server works standalone
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

### 6. Belief Deduplication
- Content-hash based exact duplicate detection
- Embedding-based fuzzy duplicate detection (cosine > 0.90)
- Duplicates reinforce existing beliefs instead of creating new ones
- Corroboration count drives confidence escalation (tentative → emerging → established)

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
3. Belief creation auto-deduplicates

### Session End
1. Hook fires on session end
2. Auto-capture summary and themes as beliefs (with dedup)
3. Generate embeddings for captured beliefs
4. Mark session as completed

## MCP Server Design

All tools are served by the unified `valence` MCP server. Legacy `valence-substrate` and `valence-vkb` stdio servers remain available for backward compatibility.

| Category | Tools |
|----------|-------|
| Belief Management | `belief_query`, `belief_create`, `belief_supersede`, `belief_get`, `belief_search` |
| Entity Management | `entity_get`, `entity_search` |
| Tensions | `tension_list`, `tension_resolve` |
| Trust & Confidence | `trust_check`, `confidence_explain`, `belief_corroboration` |
| Sessions | `session_start`, `session_end`, `session_get`, `session_list`, `session_find_by_room` |
| Exchanges | `exchange_add`, `exchange_list` |
| Patterns | `pattern_record`, `pattern_reinforce`, `pattern_list`, `pattern_search` |
| Insights | `insight_extract`, `insight_list` |

## Plugin Structure

```
plugin/
├── hooks/
│   ├── hooks.json           # Hook configuration
│   ├── session-start.py     # Context injection
│   └── session-end.py       # Auto-capture and session closing
├── skills/
│   ├── using-valence/SKILL.md
│   ├── query-knowledge/SKILL.md
│   ├── capture-insight/SKILL.md
│   └── review-tensions/SKILL.md
└── .mcp.json                # MCP server configuration
```
