# Valence Deployment Guide

## Prerequisites

- Python 3.11+
- PostgreSQL 16 with pgvector extension
- An embedding provider (OpenAI API key for `text-embedding-3-small`)

## First-Time Setup

### 1. Database

```bash
# Using Docker (recommended):
docker run -d --name valence-pg \
  -p 5433:5432 \
  -e POSTGRES_USER=valence \
  -e POSTGRES_PASSWORD=valence \
  -e POSTGRES_DB=valence \
  pgvector/pgvector:pg16

# Or configure an existing PostgreSQL instance with pgvector.
```

### 2. Install Valence

```bash
pip install -e .
# or: pip install valence
```

### 3. Database Migrations

```bash
valence migrate up
```

### 4. Create an Auth Token

```bash
valence auth create-token -c my-client -d "My first token"
# Token is saved to ~/.valence/tokens/<client>_<timestamp>.token
cat ~/.valence/tokens/my-client_*.token
```

### 5. Configure the CLI

```bash
valence config set-url http://127.0.0.1:8420
valence config set-token <your-token>
```

### 6. Start the Server

```bash
# Foreground:
valence server start

# Or via launchd (macOS):
cp docs/com.ourochronos.valence-server.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ourochronos.valence-server.plist
```

## Daily Workflow

### After Code Changes

```bash
cd ~/projects/valence
git pull
pip install -e .
valence server restart
```

### Check Server Status

```bash
valence server status
# Shows: PID, uptime, port, health
```

### View Logs

```bash
valence server logs
# Or: tail -f /tmp/valence-server.log
```

## Configuration Reference

### File Locations

| File | Purpose |
|------|---------|
| `~/.valence/cli.toml` | CLI config (server URL, token) |
| `~/.valence/tokens.json` | Token store |
| `~/.valence/tokens/*.token` | Raw token files |

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `VALENCE_DB_HOST` | `127.0.0.1` | PostgreSQL host |
| `VALENCE_DB_PORT` | `5433` | PostgreSQL port |
| `VALENCE_DB_NAME` | `valence` | Database name |
| `VALENCE_DB_USER` | `valence` | Database user |
| `VALENCE_DB_PASSWORD` | `valence` | Database password |
| `VALENCE_PORT` | `8420` | Server listen port |
| `OPENAI_API_KEY` | — | For embeddings (`text-embedding-3-small`) |

### Inference Backend

Inference is configured in the `system_config` database table, not via environment variables:

```sql
-- Callback-based (recommended for OpenClaw):
INSERT INTO system_config (key, value) VALUES (
  'inference_backend',
  '{"provider": "callback", "callback_url": "http://127.0.0.1:18789/valence/inference"}'
);
```

The server loads this at startup. Change it via SQL and restart.

## Auth Token Management

```bash
# Create a token
valence auth create-token -c <client-id> [-d "description"] [-e 90]

# List all tokens
valence auth list-tokens

# Revoke by client ID
valence auth revoke-token -c <client-id>

# Verify a token
valence auth verify <token-string>
```

Tokens are passed as `Authorization: Bearer <token>` headers.

## Troubleshooting

### Server Won't Start

1. Check if port is in use: `lsof -i :8420`
2. Check database connectivity: `psql -h 127.0.0.1 -p 5433 -U valence -d valence`
3. Check migrations: `valence migrate status`

### "No inference backend configured"

The server will log this warning at startup. Configure via `system_config` table (see above).
Tree indexing and compilation will be unavailable until configured.

### Token Issues

```bash
# Verify your CLI token works:
valence status
# If it fails, check: valence config show
```

### Memory Pressure

Valence runs well within 256MB RSS. If the host is under memory pressure,
check for PostgreSQL shared buffers and consider reducing `work_mem`.
