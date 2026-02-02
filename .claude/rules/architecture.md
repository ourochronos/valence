# Valence Architecture Rules

## Core Principles

1. **User Sovereignty**: All data stored locally or on user-controlled infrastructure
2. **Context Efficiency**: Inject minimal relevant context, use MCP for on-demand access
3. **Gradual Adoption**: Plugin can be enabled/disabled without data loss
4. **Dimensional Confidence**: Beliefs have multiple confidence dimensions, not just one score

## Database Conventions

- Use PostgreSQL with pgvector for similarity search
- All tables use UUID primary keys
- Timestamps are always TIMESTAMPTZ
- JSON fields use JSONB for indexing
- Status fields use CHECK constraints with enum values

## MCP Tool Patterns

- Tools return `{success: true/false, ...data}` format
- Always include meaningful error messages
- Use snake_case for tool names
- Prefix with noun (belief_, session_, pattern_)

## Belief Model

Beliefs have:
- Content (the actual claim)
- Confidence (dimensional: overall, source_reliability, method_quality, etc.)
- Domain path (hierarchical classification)
- Temporal validity (valid_from, valid_until)
- Source (provenance tracking)
- Supersession chain (for updates)

## Session Model

Sessions track conversations at three scales:
- Micro: Individual exchanges (turns)
- Meso: Sessions (one conversation)
- Macro: Patterns (across sessions)

## Code Style

- Use type hints throughout
- Prefer dataclasses for data models
- Use context managers for database connections
- Handle errors gracefully with structured responses
