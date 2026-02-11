# Vision: Agent as Curation Layer

**Status:** Deferred

## Summary

A background agent that actively curates the Valence knowledge base -- reviewing beliefs, detecting staleness, resolving tensions, and proposing compressions. Third step in an adoption ladder from manual tools to passive capture to active curation. Deferred because passive capture plus dedup covers current needs.

## Adoption Ladder

1. **MCP Tools (Manual) -- CURRENT** -- Claude uses Valence tools on demand. Capture depends on someone remembering to do it.
2. **Plugin Hooks (Passive) -- CURRENT** -- Session hooks automatically capture summaries, patterns, and insights as a side effect of normal conversation.
3. **Agent (Active) -- FUTURE** -- Background process that independently reviews and curates the KB without waiting for conversations.

## Current State

Steps 1 and 2 are working. Explicit MCP usage plus automatic hook capture produces a steady flow of beliefs. Dedup (#329) and corroboration (#332) keep the volume manageable.

The gap: no one reviews old beliefs. Stale beliefs persist. Tensions accumulate without resolution. The KB grows but is never pruned.

## Agent Responsibilities

- **Periodic KB review** -- Scan beliefs older than a threshold (e.g., 90 days), flag dormant ones for archival
- **Stale belief detection** -- Identify beliefs whose temporal validity may have expired, propose supersessions
- **Proactive corroboration** -- When a new belief is created, search for existing beliefs it corroborates
- **Tension detection** -- Compare new beliefs against existing ones, escalate high-severity contradictions
- **Semantic compression** -- Group similar beliefs and propose distilled versions (see `semantic-compression.md`)

## Architecture

```
Curation Agent (cron schedule)
  │
  ├── Task queue: review_old, detect_stale, compress, tensions
  │
  └── MCP Client (same tools as Claude)
        │
        ▼
      Valence MCP Server (HTTP or stdio)
```

### Key Design Decisions

- **Same MCP tools as Claude** -- no special DB access. The agent is a client like any other.
- **Proposes, does not apply** -- all actions surface as proposals for human review. No silent KB mutations.
- **Runs on a schedule** -- daily or weekly cron, not continuously.
- **Stateless between runs** -- reads KB, produces proposals, exits. No persistent agent state.

## Prerequisites

- Stable MCP HTTP server for remote agent access (done)
- Token auth for agent client identity (done)
- Belief volume large enough that curation adds value (not yet)
- LLM access for compression and tension resolution proposals
- Review UX -- could be a daily summary belief or Telegram message via valence-agent

## Why Deferred

Passive capture plus dedup and corroboration handle current needs. The KB is small enough that manual review during sessions is sufficient. Active curation becomes valuable when:
- Belief count exceeds what can be casually reviewed (~500+)
- Stale beliefs start appearing in retrieval results
- Tensions accumulate faster than they are resolved in conversation

Building the agent now would be premature infrastructure for a problem not yet at a painful scale.
