# Valence MCP Tool Design

## Overview

Valence exposes two MCP servers, each with a focused set of tools:

1. **valence-substrate** - Knowledge management (beliefs, entities, tensions)
2. **valence-vkb** - Conversation tracking (sessions, exchanges, patterns)

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

## valence-substrate Tools

### belief_query

Search beliefs by content, domain, or entity.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language search query"
    },
    "domain_filter": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Filter by domain path (e.g., ['tech', 'architecture'])"
    },
    "entity_id": {
      "type": "string",
      "description": "Filter by related entity UUID"
    },
    "include_superseded": {
      "type": "boolean",
      "default": false,
      "description": "Include superseded beliefs"
    },
    "limit": {
      "type": "integer",
      "default": 20,
      "description": "Maximum results"
    }
  },
  "required": ["query"]
}
```

**Response:**
```json
{
  "success": true,
  "beliefs": [
    {
      "id": "uuid",
      "content": "...",
      "confidence": {"overall": 0.8, "source_reliability": 0.9},
      "domain_path": ["tech", "architecture"],
      "created_at": "2025-01-15T10:30:00Z",
      "source": {"type": "document", "title": "..."},
      "relevance_score": 0.87
    }
  ],
  "total_count": 42
}
```

### belief_create

Create a new belief with optional entity links.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "content": {
      "type": "string",
      "description": "The belief content"
    },
    "confidence": {
      "type": "object",
      "description": "Confidence dimensions (or single 'overall' value)"
    },
    "domain_path": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Domain classification"
    },
    "source_type": {
      "type": "string",
      "enum": ["document", "conversation", "inference", "observation"],
      "description": "Type of source"
    },
    "source_ref": {
      "type": "string",
      "description": "Reference to source (URL, session_id, etc.)"
    },
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string"},
          "role": {"type": "string", "enum": ["subject", "object", "context"]}
        }
      },
      "description": "Entities to link (will be created if not exist)"
    },
    "valid_from": {
      "type": "string",
      "format": "date-time",
      "description": "When the belief became true"
    }
  },
  "required": ["content"]
}
```

### belief_supersede

Replace an old belief with a new one, maintaining history.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "old_belief_id": {
      "type": "string",
      "description": "UUID of belief to supersede"
    },
    "new_content": {
      "type": "string",
      "description": "Updated belief content"
    },
    "reason": {
      "type": "string",
      "description": "Why this belief is being superseded"
    },
    "confidence": {
      "type": "object",
      "description": "Confidence for new belief"
    }
  },
  "required": ["old_belief_id", "new_content", "reason"]
}
```

### belief_get

Get a single belief by ID with full details.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "belief_id": {
      "type": "string",
      "description": "UUID of the belief"
    },
    "include_history": {
      "type": "boolean",
      "default": false,
      "description": "Include supersession chain"
    },
    "include_tensions": {
      "type": "boolean",
      "default": false,
      "description": "Include related tensions"
    }
  },
  "required": ["belief_id"]
}
```

### entity_get

Get entity details with optional beliefs.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "entity_id": {
      "type": "string",
      "description": "UUID of the entity"
    },
    "include_beliefs": {
      "type": "boolean",
      "default": false,
      "description": "Include related beliefs"
    },
    "belief_limit": {
      "type": "integer",
      "default": 10,
      "description": "Max beliefs to include"
    }
  },
  "required": ["entity_id"]
}
```

### entity_search

Find entities by name or type.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query (matches name and aliases)"
    },
    "type": {
      "type": "string",
      "enum": ["person", "organization", "tool", "concept", "project", "location", "service"],
      "description": "Filter by entity type"
    },
    "limit": {
      "type": "integer",
      "default": 20
    }
  },
  "required": ["query"]
}
```

### tension_list

List contradictions/tensions between beliefs.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["detected", "investigating", "resolved", "accepted"],
      "description": "Filter by status"
    },
    "severity": {
      "type": "string",
      "enum": ["low", "medium", "high", "critical"],
      "description": "Minimum severity"
    },
    "entity_id": {
      "type": "string",
      "description": "Tensions involving this entity"
    },
    "limit": {
      "type": "integer",
      "default": 20
    }
  }
}
```

### tension_resolve

Mark a tension as resolved with explanation.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "tension_id": {
      "type": "string",
      "description": "UUID of the tension"
    },
    "resolution": {
      "type": "string",
      "description": "How the tension was resolved"
    },
    "action": {
      "type": "string",
      "enum": ["supersede_a", "supersede_b", "keep_both", "archive_both"],
      "description": "What to do with the beliefs"
    }
  },
  "required": ["tension_id", "resolution", "action"]
}
```

## valence-vkb Tools

### session_start

Begin a new conversation session.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "platform": {
      "type": "string",
      "enum": ["claude-code", "matrix", "api", "slack"],
      "description": "Platform this session is on"
    },
    "project_context": {
      "type": "string",
      "description": "Project or topic context"
    },
    "external_room_id": {
      "type": "string",
      "description": "Room/channel ID for chat platforms"
    },
    "metadata": {
      "type": "object",
      "description": "Additional session metadata"
    }
  },
  "required": ["platform"]
}
```

### session_end

Close a session with summary and themes.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "session_id": {
      "type": "string",
      "description": "UUID of the session"
    },
    "summary": {
      "type": "string",
      "description": "Session summary"
    },
    "themes": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Key themes from session"
    },
    "status": {
      "type": "string",
      "enum": ["completed", "abandoned"],
      "default": "completed"
    }
  },
  "required": ["session_id"]
}
```

### session_get

Get session details.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "session_id": {
      "type": "string",
      "description": "UUID of the session"
    },
    "include_exchanges": {
      "type": "boolean",
      "default": false,
      "description": "Include recent exchanges"
    },
    "exchange_limit": {
      "type": "integer",
      "default": 10
    }
  },
  "required": ["session_id"]
}
```

### session_list

List sessions with filters.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "platform": {"type": "string"},
    "project_context": {"type": "string"},
    "status": {"type": "string"},
    "limit": {"type": "integer", "default": 20}
  }
}
```

### session_find_by_room

Find a session by external room ID (for chat platform integrations).

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "external_room_id": {
      "type": "string",
      "description": "Room/channel ID from the chat platform"
    },
    "platform": {
      "type": "string",
      "description": "Platform name (matrix, slack, etc.)"
    },
    "status": {
      "type": "string",
      "enum": ["active", "completed", "abandoned"],
      "description": "Filter by session status"
    }
  },
  "required": ["external_room_id"]
}
```

**Response:**
```json
{
  "success": true,
  "session": {
    "id": "uuid",
    "platform": "matrix",
    "external_room_id": "!room:example.com",
    "status": "active",
    "created_at": "2025-01-15T10:30:00Z"
  }
}
```

### exchange_add

Record a conversation turn.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "session_id": {
      "type": "string",
      "description": "UUID of the session"
    },
    "role": {
      "type": "string",
      "enum": ["user", "assistant", "system"]
    },
    "content": {
      "type": "string",
      "description": "Message content"
    },
    "tool_uses": {
      "type": "array",
      "description": "Tools used in this turn"
    }
  },
  "required": ["session_id", "role", "content"]
}
```

### exchange_list

Get exchanges from a session.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "session_id": {"type": "string"},
    "limit": {"type": "integer"},
    "offset": {"type": "integer", "default": 0}
  },
  "required": ["session_id"]
}
```

### pattern_record

Record a new behavioral pattern.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "type": {
      "type": "string",
      "description": "Pattern type (topic_recurrence, preference, etc.)"
    },
    "description": {
      "type": "string",
      "description": "What the pattern is"
    },
    "evidence": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Session IDs as evidence"
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 0.5
    }
  },
  "required": ["type", "description"]
}
```

### pattern_reinforce

Strengthen an existing pattern.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "pattern_id": {
      "type": "string",
      "description": "UUID of the pattern"
    },
    "session_id": {
      "type": "string",
      "description": "Session that supports this pattern"
    }
  },
  "required": ["pattern_id"]
}
```

### pattern_list

List patterns with filters.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "type": {"type": "string"},
    "status": {"type": "string", "enum": ["emerging", "established", "fading"]},
    "min_confidence": {"type": "number", "default": 0},
    "limit": {"type": "integer", "default": 20}
  }
}
```

### insight_extract

Extract an insight from a session to create a belief.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "session_id": {
      "type": "string",
      "description": "Source session"
    },
    "content": {
      "type": "string",
      "description": "The insight/belief content"
    },
    "domain_path": {
      "type": "array",
      "items": {"type": "string"}
    },
    "confidence": {
      "type": "number",
      "default": 0.8
    },
    "entities": {
      "type": "array",
      "description": "Entities to link"
    }
  },
  "required": ["session_id", "content"]
}
```

## Tool Invocation Patterns

### Query First Pattern
For questions that might be in the KB:
```
1. belief_query("user's preferred deployment strategy")
2. If results found, use them in response
3. If not, answer from training and optionally capture new belief
```

### Capture on Insight Pattern
When user shares important information:
```
1. Recognize valuable information in user message
2. insight_extract(session_id, content, domain_path)
3. Confirm capture to user (optional)
```

### Tension Detection Pattern
After creating beliefs:
```
1. belief_create(...)
2. tension_list(entity_id=related_entity)
3. If tensions found, alert user or auto-resolve
```
