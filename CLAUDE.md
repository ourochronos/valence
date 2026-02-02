# Valence - Personal Knowledge Substrate

Valence is a personal knowledge substrate with Claude Code at its core. It stores beliefs, tracks conversations, and learns patterns over time.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Initialize database
psql -c "CREATE DATABASE valence;"
psql -d valence -f src/valence/substrate/schema.sql
psql -d valence -f src/valence/substrate/procedures.sql

# Run MCP servers (for development)
python -m valence.substrate.mcp_server  # Knowledge substrate
python -m valence.vkb.mcp_server        # Conversation tracking

# Run Matrix bot
MATRIX_PASSWORD=xxx python -m valence.agents.matrix_bot
```

## Architecture

### Components

1. **Knowledge Substrate** (`src/valence/substrate/`)
   - Beliefs with dimensional confidence
   - Entities with aliases and relationships
   - Tensions (contradictions) with resolution

2. **Conversation Tracking** (`src/valence/vkb/`)
   - Sessions (meso-scale)
   - Exchanges (micro-scale)
   - Patterns (macro-scale)

3. **Core Library** (`src/valence/core/`)
   - Data models
   - Dimensional confidence
   - Temporal validity
   - Database utilities

4. **Claude Code Plugin** (`plugin/`)
   - SessionStart hook for context injection
   - Skills for knowledge operations
   - MCP server configuration

5. **Agents** (`src/valence/agents/`)
   - Matrix bot with session resumption

### MCP Tools

**valence-substrate:**
- `belief_query` - Search beliefs
- `belief_create` - Store new belief
- `belief_supersede` - Update with history
- `belief_get` - Get belief details
- `entity_get` - Get entity with beliefs
- `entity_search` - Find entities
- `tension_list` - List contradictions
- `tension_resolve` - Resolve contradiction

**valence-vkb:**
- `session_start/end/get/list` - Manage sessions
- `session_find_by_room` - Find session by external room ID
- `exchange_add/list` - Record turns
- `pattern_record/reinforce/list/search` - Track patterns
- `insight_extract/list` - Extract to KB

## Database

PostgreSQL with pgvector. See `src/valence/substrate/schema.sql`.

Key tables:
- `beliefs` - Knowledge claims with confidence
- `entities` - People, tools, concepts
- `sessions` - Conversation sessions
- `exchanges` - Individual turns
- `patterns` - Behavioral patterns
- `tensions` - Contradictions

## Environment Variables

```bash
VKB_DB_HOST=localhost
VKB_DB_NAME=valence
VKB_DB_USER=valence
VKB_DB_PASSWORD=

MATRIX_HOMESERVER=https://matrix.example.com
MATRIX_USER=@bot:example.com
MATRIX_PASSWORD=xxx

OPENAI_API_KEY=xxx  # For embeddings
```

## Development

```bash
# Run tests
pytest

# Type check
mypy src/valence

# Format
black src/valence
```

## Plugin Usage

```bash
# Use with Claude Code
claude --plugin-dir /path/to/valence/plugin

# Or install to default location
cp -r plugin ~/.claude/plugins/valence
```

## Pod Deployment

Deploy Valence to a Digital Ocean droplet with the full stack (Matrix, PostgreSQL, VKB).

```bash
cd infra

# Configure environment
cp .env.example .env.pod
# Edit .env.pod with your values
source .env.pod

# Validate configuration
./scripts/validate-env.sh

# Deploy (preview first)
./deploy.sh --check  # Dry run
./deploy.sh          # Full deployment

# Verify
./scripts/verify-deployment.sh
```

### Deployment Architecture

- **Security**: UFW firewall, SSH key auth, fail2ban, auto-updates
- **Database**: PostgreSQL 16 + pgvector for embeddings
- **Matrix**: Synapse homeserver with nginx + SSL
- **VKB**: MCP server running as systemd service

### Idempotent Re-deployment

The deployment is idempotent - run it multiple times safely:

```bash
./deploy.sh  # First run - installs everything
./deploy.sh  # Second run - no changes if already configured
```

See `infra/README.md` for detailed documentation.

## Key Design Decisions

1. **Claude Code AS the agent**: Use session resumption and plugins instead of building wrapper infrastructure
2. **Dimensional confidence**: Multiple confidence dimensions, not just one score
3. **Temporal validity**: Beliefs can have time-bounded validity and supersession chains
4. **User sovereignty**: All data stored locally or on user-controlled infrastructure
