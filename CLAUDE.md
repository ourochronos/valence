# Valence - Personal Knowledge Substrate

Valence is a personal knowledge substrate with Claude Code at its core. It stores beliefs, tracks conversations, and learns patterns over time.

## Ourochronos Conventions

Valence is a **composed project** in the ourochronos ecosystem. Conventions and tooling standards are defined in [`our-infra`](https://github.com/ourochronos/our-infra). Valence uses line-length 150 (configured in pyproject.toml). New ourochronos bricks use 120, but valence's existing codebase uses 150 consistently.

Key references:
- Naming: `our-infra/standards/naming.md`
- Versioning: `our-infra/standards/versioning.md`
- API contracts: `our-infra/standards/api-contracts.md`
- Testing: `our-infra/standards/testing.md`
- State ownership: `our-infra/standards/state-ownership.md`

## Quick Start

```bash
# Install dependencies
pip install -e .

# Initialize database
psql -c "CREATE DATABASE valence;"
psql -d valence -f src/valence/substrate/schema.sql
psql -d valence -f src/valence/substrate/procedures.sql

# Run HTTP MCP server (recommended for remote access)
valence-server  # Starts on http://127.0.0.1:8420

# Or run unified stdio MCP server (for local Claude Code)
python -m valence.mcp_server            # All tools in one server

# Or run individual stdio servers (legacy, still supported)
python -m valence.substrate.mcp_server  # Knowledge substrate only
python -m valence.vkb.mcp_server        # Conversation tracking only

```

## HTTP MCP Server

The unified HTTP MCP server provides remote access to all Valence tools from any Claude client.

### Token Management

```bash
# Create a new token
valence-token create --client-id "claude-code-laptop" --description "My laptop"

# List tokens
valence-token list

# Revoke a token
valence-token revoke --client-id "claude-code-laptop"

# Verify a token
valence-token verify vt_xxxxxxxxxxxx
```

### Client Configuration

**Claude Code (native HTTP support):**
```bash
claude mcp add --transport http valence https://your-domain.com/api/v1/mcp \
  --header "Authorization: Bearer vt_xxxxxxxxxxxx"
```

**Claude Desktop (requires mcp-remote bridge):**

Claude Desktop only supports stdio transport. Use `mcp-remote` to bridge:

```json
{
  "mcpServers": {
    "valence": {
      "command": "npx",
      "args": [
        "mcp-remote@latest",
        "--http",
        "https://your-domain.com/api/v1/mcp",
        "--header",
        "Authorization: Bearer vt_xxxxxxxxxxxx"
      ]
    }
  }
}
```

Config location:
- macOS/Linux: `~/.config/claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

### Security

**Two authentication methods** are supported:

1. **Bearer Tokens** (for Claude Code): Simple token-based auth
   - Tokens are 256-bit random (unguessable)
   - Stored as SHA-256 hashes (safe at rest)

2. **OAuth 2.1 with PKCE** (for Claude mobile/web): Standards-compliant OAuth
   - Dynamic Client Registration (RFC 7591)
   - Authorization Code flow with PKCE
   - JWT access tokens

**Optional IP allowlist** for additional security:

In your Ansible inventory or group_vars:
```yaml
valence_allowed_ips:
  - 1.2.3.4        # Your home IP
  - 5.6.7.8/24     # Your office network
```

This restricts access at the nginx level before token verification.

### OAuth 2.1 Setup (for Claude Mobile/Web)

OAuth allows Claude mobile app and claude.ai to authenticate with your Valence server.

**1. Set a password:**
```bash
export VALENCE_OAUTH_PASSWORD="your-secure-password"
```

**2. Optionally configure:**
```bash
export VALENCE_OAUTH_USERNAME="admin"  # Default: admin
export VALENCE_OAUTH_JWT_SECRET="..."  # Auto-generated if not set
export VALENCE_EXTERNAL_URL="https://your-domain.com"  # Required for OAuth
```

**3. Start the server:**
```bash
valence-server
```

**4. Discovery endpoints:**
- `/.well-known/oauth-protected-resource` - RFC 9728 metadata
- `/.well-known/oauth-authorization-server` - RFC 8414 metadata

**5. Configure Claude:**
Claude clients that support OAuth will automatically discover and use these endpoints.
The server URL is: `https://your-domain.com/api/v1/mcp`

### Local Development

```bash
# Start server locally
VALENCE_TOKEN_FILE=./tokens.json valence-server

# Create a test token
valence-token --token-file ./tokens.json create -c test

# Test with curl
curl -X POST http://localhost:8420/api/v1/mcp \
  -H "Authorization: Bearer vt_xxxx" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}'
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

### MCP Tools

All tools are served by the unified `valence` MCP server. All tools are always available.

**Knowledge Substrate:**
- `belief_query` - Search beliefs (supports `ranking` param with configurable weights + `explain` mode)
- `belief_create` - Store new belief
- `belief_supersede` - Update with history
- `belief_get` - Get belief details
- `belief_search` - Semantic search via embeddings (supports `ranking` param)
- `entity_get` - Get entity with beliefs
- `entity_search` - Find entities
- `tension_list` - List contradictions
- `tension_resolve` - Resolve contradiction
- `confidence_explain` - Explain confidence dimensions
- `belief_corroboration` - Check corroboration sources
- `trust_check` - Check trust levels for entities/topics

**Conversation Tracking:**
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
- `belief_entities` - Many-to-many links
- `tensions` - Contradictions between beliefs
- `vkb_sessions` - Conversation sessions
- `vkb_exchanges` - Individual turns
- `vkb_patterns` - Behavioral patterns
- `vkb_session_insights` - Links sessions to extracted beliefs
- `belief_retrievals` - Tracks which beliefs are retrieved (feedback loop)

## Environment Variables

```bash
# Database
VALENCE_DB_HOST=localhost
VALENCE_DB_NAME=valence
VALENCE_DB_USER=valence
VALENCE_DB_PASSWORD=

# HTTP Server
VALENCE_HOST=127.0.0.1
VALENCE_PORT=8420
VALENCE_EXTERNAL_URL=https://your-domain.com  # For OAuth redirects

# OAuth 2.1 (for Claude mobile/web)
VALENCE_OAUTH_PASSWORD=xxx        # REQUIRED for OAuth
VALENCE_OAUTH_USERNAME=admin      # Default: admin
VALENCE_OAUTH_JWT_SECRET=xxx      # Auto-generated if not set
VALENCE_OAUTH_ENABLED=true        # Default: true

# OpenAI
OPENAI_API_KEY=xxx  # For embeddings
```

## Development

```bash
# Run tests
pytest

# Type check
mypy src/valence

# Format
ruff format src/valence tests/
ruff check src/valence tests/ --fix
```

## Plugin Usage

```bash
# Use with Claude Code
claude --plugin-dir /path/to/valence/plugin

# Or install to default location
cp -r plugin ~/.claude/plugins/valence
```

## Pod Deployment

Deploy Valence to a Digital Ocean droplet with PostgreSQL and the VKB server.

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
- **Valence HTTP MCP**: Remote MCP server at /api/v1/mcp endpoint with token auth
- **MCP Transport**: HTTP is primary for remote access; stdio for local Claude Code development

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
