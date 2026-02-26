# Valence MCP Integration Guide

This guide explains how to connect Valence to Claude Desktop (or other MCP-compatible clients) for seamless LLM integration with your personal knowledge base.

## Quick Start

### 1. Prerequisites

- Valence installed and database running
- Claude Desktop (or another MCP client)
- Python 3.11+ with Valence in your PATH

### 2. Configure Claude Desktop

Add Valence to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

**Recommended: Unified server** (all tools in one process):
```json
{
  "mcpServers": {
    "valence": {
      "command": "python",
      "args": ["-m", "valence.mcp_server"],
      "env": {
        "VALENCE_DB_HOST": "localhost",
        "VALENCE_DB_PORT": "5432",
        "VALENCE_DB_NAME": "valence",
        "VALENCE_DB_USER": "valence",
        "VALENCE_DB_PASSWORD": "your-password",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

**Legacy: Split servers** (still supported for backward compatibility):
```json
{
  "mcpServers": {
    "valence-substrate": {
      "command": "python",
      "args": ["-m", "valence.substrate.mcp_server"],
      "env": {
        "VALENCE_DB_HOST": "localhost",
        "VALENCE_DB_NAME": "valence",
        "VALENCE_DB_USER": "valence",
        "VALENCE_DB_PASSWORD": "your-password",
        "OPENAI_API_KEY": "sk-..."
      }
    },
    "valence-vkb": {
      "command": "python",
      "args": ["-m", "valence.vkb.mcp_server"],
      "env": {
        "VALENCE_DB_HOST": "localhost",
        "VALENCE_DB_NAME": "valence",
        "VALENCE_DB_USER": "valence",
        "VALENCE_DB_PASSWORD": "your-password"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

After saving the configuration, restart Claude Desktop to load the MCP servers.

---

## Available Tools

### Unified Server Tools

All tools are available through the unified `valence` MCP server. `VALENCE_MODE` controls which tools are exposed (`personal`, `connected`, or `full`).

#### Belief Management

| Tool | Description |
|------|-------------|
| `belief_query` | Full-text search using keywords (supports `ranking` with configurable weights) |
| `belief_search` | **Semantic search** using embeddings (finds conceptually similar beliefs) |
| `belief_create` | Create a new belief — **auto-deduplicates** via content hash and embedding similarity |
| `belief_get` | Get full details of a belief including history and tensions |
| `belief_supersede` | Update a belief while preserving history |
| `belief_corroboration` | Check how many independent sources confirm a belief |

#### Entity Management

| Tool | Description |
|------|-------------|
| `entity_search` | Find entities by name or type |
| `entity_get` | Get entity details with related beliefs |

#### Trust & Confidence

| Tool | Description |
|------|-------------|
| `trust_check` | **Who do I trust on this topic?** Shows entities/nodes with authority |
| `confidence_explain` | **Why this confidence?** Explains all contributing dimensions with weights |

#### Tension Resolution

| Tool | Description |
|------|-------------|
| `tension_list` | List contradictions between beliefs |
| `tension_resolve` | Resolve contradictions with explanation |

#### Conversation Tracking

| Tool | Description |
|------|-------------|
| `session_start/end/get/list` | Manage conversation sessions |
| `session_find_by_room` | Find session by external room ID |
| `exchange_add/list` | Record and retrieve conversation turns |
| `pattern_record/reinforce/list/search` | Track behavioral patterns |
| `insight_extract/list` | Extract knowledge from conversations (**auto-deduplicates**) |

---

## Resources

Access these via the resources panel in Claude Desktop:

| Resource URI | Description |
|--------------|-------------|
| `valence://beliefs/recent` | Most recently modified beliefs |
| `valence://trust/graph` | Trust relationships visualization data |
| `valence://stats` | Database statistics and health |

---

## Example Prompts

### Knowledge Retrieval

```
What do I know about distributed systems architecture?
```
*Claude will use `belief_search` to find semantically related beliefs*

```
Search my knowledge base for anything related to "microservices vs monoliths"
```
*Uses semantic embeddings to find conceptually similar content*

### Trust & Authority

```
Who do I trust for advice about Kubernetes?
```
*Uses `trust_check` to find entities with high-confidence beliefs about Kubernetes*

```
Why do I trust the information about PostgreSQL performance?
Show me the confidence breakdown.
```
*Uses `confidence_explain` to show all confidence dimensions*

### Knowledge Creation

```
Remember this: "PostgreSQL's JSONB indexes are slower to update but faster 
to query than regular JSON columns" - I learned this from the official docs.
```
*Claude will use `belief_create` with appropriate confidence and source*

```
I need to update my understanding about React hooks.
The old belief was that useEffect runs synchronously, but actually it runs 
after paint. Please supersede the old belief.
```
*Uses `belief_supersede` to update while preserving history*

### Contradiction Detection

```
Are there any contradictions in my knowledge about API design?
```
*Uses `tension_list` to find unresolved conflicts*

```
I see beliefs that contradict each other about REST vs GraphQL.
Let's resolve this - GraphQL is better for complex queries, REST for simple CRUD.
```
*Uses `tension_resolve` to document the resolution*

### Conversation Patterns

```
What patterns have emerged from my recent conversations?
What topics do I keep returning to?
```
*Uses `pattern_list` from VKB server*

### Stats & Overview

```
Give me an overview of my knowledge base - how many beliefs, what domains, 
what's the confidence distribution?
```
*Accesses `valence://stats` resource*

---

## Advanced Usage

### Combining Tools

Claude can combine multiple tools in a single interaction:

```
Find what I know about caching strategies, check who I trust on this topic,
and explain the confidence for the highest-confidence belief.
```

### Domain Filtering

```
Search only in the 'architecture' domain for beliefs about event sourcing.
```
*Uses `belief_search` with domain_filter parameter*

### Confidence Thresholds

```
Only show me high-confidence beliefs (>0.8) about database design.
```
*Uses `belief_search` with min_confidence parameter*

---

## Troubleshooting

### Server Not Connecting

1. Check that the database is running:
   ```bash
   docker ps | grep valence
   ```

2. Verify environment variables are correct

3. Check Claude Desktop logs:
   - macOS: `~/Library/Logs/Claude/`
   - Windows: `%LOCALAPPDATA%\Claude\logs\`

### Semantic Search Not Working

Semantic search requires:
1. `OPENAI_API_KEY` environment variable set
2. Beliefs to have embeddings generated

Backfill embeddings:
```bash
valence embeddings backfill --content-type belief
```

### No Results Found

- Try broader search terms
- Lower the `min_similarity` threshold (default 0.5)
- Check that beliefs exist in the domain you're searching

---

## Security Considerations

1. **API Keys**: Keep your `OPENAI_API_KEY` secure. It's used only for generating embeddings locally.

2. **Database Credentials**: The MCP server connects to your local database. Keep credentials secure.

3. **Privacy**: When using OpenAI embeddings, belief content is sent to OpenAI's API. For sensitive data, consider using local embeddings (see `VALENCE_EMBEDDING_PROVIDER=local`).

4. **Federation**: If using federation features, review trust settings before sharing beliefs with external nodes.

---

## What's Next?

- [Trust Model Documentation](./TRUST_MODEL.md) - Understanding how trust works
- [Federation Protocol](./FEDERATION_PROTOCOL.md) - Connecting with other Valence nodes
- [Privacy Guarantees](./PRIVACY_GUARANTEES.md) - How your data is protected
