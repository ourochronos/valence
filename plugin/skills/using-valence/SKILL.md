---
name: using-valence
description: Learn how to use the Valence knowledge substrate
user_invocable: true
---

# Using Valence

Valence is your personal knowledge substrate. It stores beliefs, tracks conversations, and learns your patterns over time.

## Key Concepts

**Beliefs**: Facts, decisions, preferences stored with confidence levels and provenance. Each belief tracks:
- Content (what is believed)
- Confidence (how certain, with multiple dimensions)
- Domain path (categorization)
- Source (where it came from)
- Temporal validity (when true)

**Sessions**: Conversation tracking at multiple scales:
- Micro: Individual exchanges (turns)
- Meso: Sessions (one conversation)
- Macro: Patterns (across sessions)

**Patterns**: Behavioral patterns that emerge across multiple sessions, like:
- Topic recurrence (what you keep coming back to)
- Preferences (how you like things done)
- Working style (when you're productive, how you approach problems)

**Tensions**: Contradictions between beliefs that need resolution. Valence detects when beliefs conflict.

## Available Skills

- `/valence:query-knowledge` - Search your knowledge base
- `/valence:capture-insight` - Store something important you've learned
- `/valence:ingest-document` - Add a document to the substrate
- `/valence:review-tensions` - Review and resolve contradictions

## MCP Tools

All tools are served by the unified `valence` MCP server:

### Knowledge Substrate
- `belief_query` - Search beliefs by content, domain, or entity (supports ranking weights)
- `belief_create` - Store a new belief (auto-deduplicates via content hash)
- `belief_supersede` - Update a belief while maintaining history
- `belief_get` - Get a belief with full details
- `belief_search` - Semantic search via embeddings
- `belief_corroboration` - Check how many sources confirm a belief
- `entity_get` - Get entity details and related beliefs
- `entity_search` - Find entities by name
- `tension_list` - List contradictions
- `tension_resolve` - Resolve a contradiction
- `trust_check` - Check trust levels for entities/topics
- `confidence_explain` - Explain confidence score dimensions

### Conversation Tracking
- `session_start/end/get/list` - Manage sessions
- `session_find_by_room` - Find session by room ID
- `exchange_add/list` - Record conversation turns
- `pattern_record/reinforce/list/search` - Track patterns
- `insight_extract/list` - Extract insights to KB (auto-deduplicates)

## Best Practices

1. **Query first**: Before answering questions about past decisions or preferences, query the KB
2. **Capture insights**: When you learn something important about the user, capture it
3. **Link entities**: When creating beliefs, link them to relevant entities (people, tools, projects)
4. **Review tensions**: Periodically check for and resolve contradictions
5. **Note patterns**: When you observe recurring behaviors, record them as patterns

## Example Workflows

### Learning a Preference
```
User: "I prefer tabs over spaces"
Assistant:
1. Check if there's an existing preference: belief_query("tabs spaces preference")
2. If new, capture: belief_create("User prefers tabs over spaces for indentation",
   confidence={"overall": 0.9}, domain_path=["preferences", "coding"],
   entities=[{"name": "coding style", "type": "concept"}])
```

### Answering from Memory
```
User: "What did we decide about the database?"
Assistant:
1. Query: belief_query("database decision")
2. Review results for relevant beliefs
3. Answer grounded in the beliefs found
4. Cite sources/sessions where relevant
```

### Recording a Pattern
```
After noticing the user often asks about architecture in the morning:
pattern_record(
  type="working_style",
  description="User tends to work on architecture decisions in morning sessions",
  confidence=0.6
)
```
