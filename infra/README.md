# Valence Pod Infrastructure

Ansible playbooks for deploying the Valence pod to a Digital Ocean droplet.

## Architecture

The pod runs:
- **Synapse** - Matrix homeserver for chat interface
- **PostgreSQL 16 + pgvector** - Database with vector similarity search
- **VKB Schema** - Valence Knowledge Base database schema (MCP server is spawned on-demand by Claude Code)
- **Nginx** - Reverse proxy with SSL termination

## Prerequisites

1. A Digital Ocean account with API token
2. A domain pointed to your droplet's IP
3. SSH key pair for droplet access

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env.pod
# Edit .env.pod with your values

# 2. Source environment
source .env.pod

# 3. Validate configuration
./scripts/validate-env.sh

# 4. Deploy (or preview with --check)
./deploy.sh --check  # Preview changes
./deploy.sh          # Full deployment

# 5. Verify deployment
./scripts/verify-deployment.sh
```

## Required Environment Variables

```bash
# Infrastructure
export VALENCE_POD_IP="your.droplet.ip"
export VALENCE_DOMAIN="your.domain.com"
export LETSENCRYPT_EMAIL="your@email.com"

# SSH Key (choose ONE of these options)
export VALENCE_SSH_PUBKEY_FILE="~/.ssh/valence_pod.pub"  # Path to public key file
# OR
export VALENCE_SSH_PUBKEY="ssh-ed25519 AAAA... user@host"  # Inline key content

# Database password (generate a strong one)
export VALENCE_DB_PASSWORD="$(openssl rand -base64 32)"

# Matrix/Synapse secrets (generate strong ones)
export VALENCE_BOT_PASSWORD="$(openssl rand -base64 32)"
export SYNAPSE_ADMIN_PASSWORD="$(openssl rand -base64 32)"
export SYNAPSE_REGISTRATION_SECRET="$(openssl rand -base64 32)"
export SYNAPSE_MACAROON_SECRET="$(openssl rand -base64 32)"
export SYNAPSE_FORM_SECRET="$(openssl rand -base64 32)"

# API keys for VKB service
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."  # Optional
```

## Deployment Scripts

| Script | Purpose |
|--------|---------|
| `deploy.sh` | Main deployment wrapper with validation and verification |
| `scripts/validate-env.sh` | Validates all required environment variables |
| `scripts/verify-deployment.sh` | Post-deployment health checks |

### deploy.sh Options

```bash
./deploy.sh [OPTIONS]

Options:
  -c, --check           Dry-run mode (show what would change)
  -v, --verbose         Verbose Ansible output
  -t, --tags TAGS       Only run specific tags (e.g., -t vkb)
  --skip-validation     Skip environment validation
  --skip-verification   Skip post-deployment verification
  -h, --help            Show help
```

### validate-env.sh Options

```bash
./scripts/validate-env.sh [OPTIONS]

Options:
  --generate    Generate missing secrets (passwords, tokens)
  --export      Output export commands for sourcing
```

## Deployment Steps (Manual)

### 1. Create the Droplet

```bash
# Install doctl if needed
# brew install doctl  # macOS
# snap install doctl  # Linux

# Authenticate
doctl auth init

# Create droplet
doctl compute droplet create valence-pod \
  --image ubuntu-24-04-x64 \
  --size s-2vcpu-4gb \
  --region nyc1 \
  --ssh-keys $(doctl compute ssh-key list --format ID --no-header | head -1) \
  --wait

# Get the IP
export VALENCE_POD_IP=$(doctl compute droplet get valence-pod --format PublicIPv4 --no-header)
```

### 2. Configure DNS

Point your domain to the droplet IP. Required records:
- `A` record: `your.domain.com` -> `VALENCE_POD_IP`
- `A` record: `matrix.your.domain.com` -> `VALENCE_POD_IP` (optional)

Wait for DNS propagation before proceeding.

### 3. Generate SSH Key (if needed)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/valence_pod -N ""
export VALENCE_SSH_PUBKEY_FILE="$HOME/.ssh/valence_pod.pub"

# Add public key to droplet (if not done during creation)
cat ~/.ssh/valence_pod.pub | ssh root@$VALENCE_POD_IP "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### 4. Run the Playbook

```bash
cd infra

# Install Ansible if needed
pip install ansible

# Run deployment
ansible-playbook -i inventory.yml site.yml
```

### 5. Authenticate Claude Code (Manual Step)

```bash
ssh -i ~/.ssh/valence_pod valence@$VALENCE_POD_IP
claude  # Follow OAuth prompts
```

### 6. Connect Element Client

1. Download Element: https://element.io/download
2. Choose "Sign in"
3. Click "Edit" homeserver
4. Enter: `https://your.domain.com`
5. Sign in with admin credentials

## Roles

| Role | Purpose |
|------|---------|
| security | UFW, SSH hardening, fail2ban, valence user |
| common | Base packages, Valence repo clone, Claude Code |
| postgresql | PostgreSQL 16, pgvector extension, database |
| synapse | Matrix homeserver, nginx, SSL certificates |
| vkb | VKB database schema, MCP config, Matrix bot user |
| verify | Post-deployment health checks |

## Idempotency

The deployment is designed to be idempotent - running it multiple times should produce the same result:

```bash
# First run - installs everything
./deploy.sh

# Second run - should show mostly "ok" (no changes)
./deploy.sh

# Verify idempotency
# If second run shows significant "changed" tasks, that's a bug
```

Key idempotency features:
- Schema uses `IF NOT EXISTS` patterns
- Marker files track initialization state
- Service restarts only on actual changes
- Configuration files only update when different

## Testing

### Integration Tests

```bash
# Test local database
pytest tests/integration/test_deployment.py -v

# Test remote pod
VALENCE_POD_IP=x.x.x.x VALENCE_DOMAIN=pod.example.com \
    pytest tests/integration/test_deployment.py -v

# Skip slow network tests
pytest tests/integration/test_deployment.py -v -m "not slow"
```

### Manual Verification

```bash
# Quick health check
./scripts/verify-deployment.sh

# Or run Ansible verify role
ansible-playbook -i inventory.yml site.yml -t verify
```

## Security

The playbook configures:
- UFW firewall (only SSH, HTTP, HTTPS, Matrix federation)
- SSH key-only authentication (password auth disabled)
- fail2ban for SSH protection (3 attempts, 1 hour ban)
- Automatic security updates via unattended-upgrades
- SSL via Let's Encrypt with auto-renewal
- Systemd security hardening for services

## Troubleshooting

### Check service status

```bash
sudo systemctl status postgresql matrix-synapse vkb nginx
```

### View logs

```bash
# VKB service
sudo journalctl -u vkb -f

# Matrix Synapse
sudo journalctl -u matrix-synapse -f

# Recent errors across all services
sudo journalctl --since "10 minutes ago" | grep -i error
```

### Test endpoints

```bash
# Matrix federation
curl https://your.domain.com/.well-known/matrix/server
curl https://your.domain.com/.well-known/matrix/client
curl https://your.domain.com/_matrix/client/versions

# SSL certificate
openssl s_client -connect your.domain.com:443 -servername your.domain.com < /dev/null 2>/dev/null | openssl x509 -noout -dates
```

### Database connectivity

```bash
# Connect as valence user
sudo -u postgres psql -d valence

# Check tables
\dt

# Check pgvector
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Common Issues

**SSH connection refused:**
- Check UFW allows port 22: `sudo ufw status`
- Verify SSH key is correct: check `VALENCE_SSH_PUBKEY` or `VALENCE_SSH_PUBKEY_FILE`

**SSL certificate fails:**
- DNS must propagate before Let's Encrypt can verify
- Check nginx is running: `systemctl status nginx`
- Check certbot logs: `journalctl -u certbot`

**VKB not responding:**
- Note: VKB MCP server uses stdio mode and is designed to be spawned by Claude Code, not run as a standalone service
- The vkb.service is disabled by default; the database schema is set up during deployment
- Check environment file: `cat /opt/valence/config/vkb.env`
- Check database connectivity: `sudo -u valence psql -h localhost -d valence`
- Test MCP server manually: `source /opt/valence/config/vkb.env && /opt/valence/venv/bin/python -m valence.substrate.mcp_server --health-check`

**Matrix federation issues:**
- Verify `.well-known` files are accessible
- Check https://federationtester.matrix.org/
- Review Synapse logs: `journalctl -u matrix-synapse`
