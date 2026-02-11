# Valence Plugin Design

## Overview

The Valence plugin transforms Claude Code into a context-aware, persistent agent by:
1. Injecting relevant knowledge at session start
2. Capturing exchanges during the session
3. Extracting patterns at session end
4. Providing skills for common knowledge operations

## Plugin Structure

```
plugin/
├── .claude-plugin/
│   └── plugin.json          # Plugin manifest
├── hooks/
│   ├── hooks.json           # Hook configuration
│   ├── session-start.py     # Context injection (Python)
│   └── session-end.py       # Cleanup and session closing (Python)
├── skills/
│   ├── using-valence/
│   │   └── SKILL.md         # Onboarding skill
│   ├── query-knowledge/
│   │   └── SKILL.md         # Knowledge query skill
│   ├── capture-insight/
│   │   └── SKILL.md         # Insight capture skill
│   ├── ingest-document/
│   │   └── SKILL.md         # Document ingestion skill
│   └── review-tensions/
│       └── SKILL.md         # Tension review skill
└── .mcp.json                # MCP server configuration
```

## Plugin Manifest

**plugin.json:**
```json
{
  "name": "valence",
  "version": "1.0.0",
  "description": "Personal knowledge substrate for Claude Code",
  "author": "zonk1024",
  "homepage": "https://github.com/zonk1024/valence",
  "license": "MIT",
  "claude_code_version": ">=1.0.0",
  "capabilities": {
    "hooks": true,
    "skills": true,
    "mcp_servers": true
  }
}
```

## Hooks

### hooks.json
```json
{
  "hooks": [
    {
      "event": "SessionStart",
      "command": "python3 ./session-start.py",
      "description": "Inject relevant context and behavioral conditioning at session start"
    },
    {
      "event": "Stop",
      "command": "python3 ./session-end.py",
      "description": "Close VKB session and extract patterns when Claude session ends"
    }
  ]
}
```

### session-start.py

Fires at session start to inject context and behavioral conditioning. This Python script:

1. Queries the substrate for recent active beliefs
2. Queries for established behavioral patterns
3. Creates a new VKB session
4. Returns context injection and environment variables

Key features:
- Uses parameterized SQL queries to prevent injection
- Gracefully handles missing psycopg2 dependency
- Falls back to defaults when database is unavailable

See `plugin/hooks/session-start.py` for the full implementation.

### session-end.py

Fires when the Claude session ends to close the VKB session. This Python script:

1. Retrieves the session ID from environment variables
2. Updates the session status to "completed"
3. Sets the ended_at timestamp

Key features:
- Uses parameterized SQL queries to prevent injection
- Handles missing session IDs gracefully

See `plugin/hooks/session-end.py` for the full implementation.

## Skills

### using-valence/SKILL.md

Onboarding skill explaining Valence capabilities.

```markdown
---
name: using-valence
description: Learn how to use the Valence knowledge substrate
user_invocable: true
---

# Using Valence

Valence is your personal knowledge substrate. It stores beliefs, tracks conversations, and learns your patterns over time.

## Key Concepts

**Beliefs**: Facts, decisions, preferences stored with confidence levels and provenance.

**Sessions**: Conversation tracking at multiple scales (micro: turns, meso: sessions, macro: patterns).

**Patterns**: Behavioral patterns that emerge across multiple sessions.

**Tensions**: Contradictions between beliefs that need resolution.

## Available Skills

- `/valence:query-knowledge` - Search your knowledge base
- `/valence:capture-insight` - Store something important
- `/valence:ingest-document` - Add a document to the substrate
- `/valence:review-tensions` - Review contradictions

## MCP Tools

You can also use the raw MCP tools directly:

### Knowledge (valence-substrate)
- `belief_query` - Search beliefs
- `belief_create` - Store new belief
- `entity_get` - Get entity details
- `tension_list` - List contradictions

### Conversations (valence-vkb)
- `session_start/end` - Manage sessions
- `exchange_add` - Record turns
- `pattern_record/reinforce` - Track patterns
- `insight_extract` - Extract to KB

## Best Practices

1. **Query first**: Before answering questions about past decisions or preferences, query the KB
2. **Capture insights**: When you learn something important, capture it
3. **Review tensions**: Periodically review and resolve contradictions
4. **Use patterns**: Pay attention to established patterns
```

### query-knowledge/SKILL.md

```markdown
---
name: query-knowledge
description: Search the Valence knowledge base for relevant information
user_invocable: true
args:
  - name: query
    description: What to search for
    required: true
---

# Query Knowledge

Search the Valence knowledge substrate for relevant beliefs.

## Instructions

1. Use `mcp__valence_substrate__belief_query` with the provided query
2. Present results in a clear, organized format
3. Highlight confidence levels and sources
4. Note any related tensions if present

## Query: {{ query }}

Search for beliefs related to this query. Include:
- Direct matches
- Related entities
- Recent vs historical beliefs
- Any tensions or contradictions
```

### capture-insight/SKILL.md

```markdown
---
name: capture-insight
description: Capture an important insight or decision to the knowledge base
user_invocable: true
args:
  - name: content
    description: The insight to capture
    required: true
  - name: domain
    description: Domain classification (optional)
    required: false
---

# Capture Insight

Store an important piece of information in the Valence knowledge base.

## Instructions

1. First, search for similar existing beliefs using `belief_query`
2. If a similar belief exists, consider using `belief_supersede` instead
3. Otherwise, use `belief_create` with:
   - The content as provided
   - Appropriate confidence level
   - Domain path if provided
   - Link to current session as source

## Content: {{ content }}
## Domain: {{ domain | default: "general" }}

Capture this insight, checking for duplicates first.
```

### ingest-document/SKILL.md

```markdown
---
name: ingest-document
description: Ingest a document into the knowledge substrate
user_invocable: true
args:
  - name: path
    description: Path to the document
    required: true
---

# Ingest Document

Process a document and extract beliefs from it.

## Instructions

1. Read the document at the given path
2. Extract key facts, decisions, and insights
3. For each extracted item:
   - Create a belief with appropriate confidence
   - Link to entities mentioned
   - Set document as source
4. Report what was extracted

## Document: {{ path }}

Read and process this document, extracting beliefs.
```

### review-tensions/SKILL.md

```markdown
---
name: review-tensions
description: Review and resolve contradictions in the knowledge base
user_invocable: true
---

# Review Tensions

Review unresolved contradictions between beliefs.

## Instructions

1. Use `tension_list` to get unresolved tensions
2. For each tension, present:
   - The two conflicting beliefs
   - Their confidence levels and sources
   - When each was created
3. Ask user how to resolve, or suggest resolution:
   - Supersede one with the other
   - Keep both (mark as accepted difference)
   - Archive both (outdated)
4. Use `tension_resolve` to record resolution

Present tensions for review and help resolve them.
```

## MCP Configuration

**.mcp.json** — uses the unified MCP server:
```json
{
  "mcpServers": {
    "valence": {
      "command": "python",
      "args": ["-m", "valence.mcp_server"],
      "env": {
        "VKB_DB_HOST": "localhost",
        "VKB_DB_NAME": "valence",
        "VKB_DB_USER": "valence",
        "VALENCE_MODE": "full"
      }
    }
  }
}
```

The legacy split servers (`valence-substrate`, `valence-vkb`) are still supported but the unified server is recommended.

## Installation

1. Clone/copy plugin to `~/.claude/plugins/valence/`
2. Ensure PostgreSQL is running with valence database
3. Configure environment variables for DB connection
4. Enable plugin: `claude --plugin valence`

Or for development:
```bash
claude --plugin-dir /path/to/valence/plugin
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VKB_DB_HOST` | PostgreSQL host | localhost |
| `VKB_DB_NAME` | Database name | valence |
| `VKB_DB_USER` | Database user | valence |
| `VKB_DB_PASSWORD` | Database password | (none) |
| `OPENAI_API_KEY` | For embeddings | (required) |
