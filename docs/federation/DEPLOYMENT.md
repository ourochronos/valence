# Valence Federation Deployment

> Deploy a federated Valence node with privacy-first defaults.

---

## Prerequisites

### Required

- **Server**: Linux VPS (Ubuntu 22.04+ recommended)
  - Minimum: 2 vCPU, 4GB RAM, 40GB SSD
  - Recommended: 4 vCPU, 8GB RAM, 80GB SSD
- **Domain**: A domain name pointed to your server's IP
- **Python**: 3.11+
- **PostgreSQL**: 16+ with pgvector extension
- **SSL**: Let's Encrypt (automatic) or your own certificate

### Optional

- **OpenAI API Key**: For embedding generation (semantic search)
- **Ansible**: For automated deployment (recommended)

---

## Quick Start (Ansible)

The recommended deployment uses Ansible for reproducibility:

```bash
# 1. Clone the repository
git clone https://github.com/ourochronos/valence.git
cd valence/infra

# 2. Configure environment
cp .env.example .env.pod
# Edit .env.pod with your values (see below)

# 3. Source environment
source .env.pod

# 4. Validate configuration
./scripts/validate-env.sh

# 5. Deploy
./deploy.sh --check  # Preview changes
./deploy.sh          # Full deployment

# 6. Verify
./scripts/verify-deployment.sh
```

---

## Environment Variables

Create `.env.pod` with these variables:

### Required

```bash
# =============================================================================
# INFRASTRUCTURE
# =============================================================================

# Your server's IP address
export VALENCE_POD_IP="203.0.113.10"

# Your domain (must have DNS pointing to VALENCE_POD_IP)
export VALENCE_DOMAIN="valence.example.com"

# Email for Let's Encrypt SSL notifications
export LETSENCRYPT_EMAIL="admin@example.com"

# =============================================================================
# SSH
# =============================================================================

# Path to your SSH public key
export VALENCE_SSH_PUBKEY_FILE="$HOME/.ssh/valence_pod.pub"

# =============================================================================
# DATABASE
# =============================================================================

# Generate a strong password
export VALENCE_DB_PASSWORD="$(openssl rand -base64 32)"

# =============================================================================
# API KEYS
# =============================================================================

# Required for embeddings
export OPENAI_API_KEY="sk-..."
```

### Optional (Matrix Integration)

```bash
# Matrix/Synapse secrets (if using chat interface)
export VALENCE_BOT_PASSWORD="$(openssl rand -base64 32)"
export SYNAPSE_ADMIN_PASSWORD="$(openssl rand -base64 32)"
export SYNAPSE_REGISTRATION_SECRET="$(openssl rand -base64 32)"
export SYNAPSE_MACAROON_SECRET="$(openssl rand -base64 32)"
export SYNAPSE_FORM_SECRET="$(openssl rand -base64 32)"
```

---

## DNS Requirements

Before deploying, configure DNS:

| Record Type | Name | Value | TTL |
|-------------|------|-------|-----|
| A | `valence.example.com` | `203.0.113.10` | 300 |
| A | `matrix.example.com` | `203.0.113.10` | 300 (optional) |

**Wait for DNS propagation before deploying** (usually 5-30 minutes).

Verify:
```bash
dig +short valence.example.com
# Should return your server IP
```

---

## SSL/TLS

### Automatic (Let's Encrypt)

The Ansible deployment handles SSL automatically:
- Requests certificate from Let's Encrypt
- Configures Nginx with TLS 1.3
- Sets up auto-renewal (certbot timer)

### Manual (Existing Certificate)

If you have your own certificate:

```bash
# Place your certificates
sudo cp fullchain.pem /etc/ssl/certs/valence.crt
sudo cp privkey.pem /etc/ssl/private/valence.key
sudo chmod 600 /etc/ssl/private/valence.key

# Update nginx config (skip certbot tasks in Ansible)
./deploy.sh --skip-tags certbot
```

---

## Manual Deployment

If not using Ansible, here's the step-by-step process:

### 1. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
  python3.11 python3.11-venv python3-pip \
  postgresql-16 postgresql-16-pgvector \
  nginx certbot python3-certbot-nginx \
  git curl jq

# Verify PostgreSQL version
psql --version  # Should be 16+
```

### 2. Configure PostgreSQL

```bash
# Enable pgvector extension
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Create database and user
sudo -u postgres psql <<EOF
CREATE USER valence WITH PASSWORD 'your-secure-password';
CREATE DATABASE valence OWNER valence;
GRANT ALL PRIVILEGES ON DATABASE valence TO valence;
\c valence
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
EOF
```

### 3. Set Up Valence

```bash
# Create system user
sudo useradd -r -m -d /opt/valence -s /bin/bash valence

# Clone and set up
sudo -u valence git clone https://github.com/ourochronos/valence.git /opt/valence/repo
cd /opt/valence/repo

# Create virtual environment
sudo -u valence python3.11 -m venv /opt/valence/.venv
source /opt/valence/.venv/bin/activate

# Install
pip install -e ".[dev]"

# Initialize database
valence init
```

### 4. Configure Environment

Create `/opt/valence/config/valence.env`:

```bash
# Database
VALENCE_DB_HOST=localhost
VALENCE_DB_PORT=5432
VALENCE_DB_NAME=valence
VALENCE_DB_USER=valence
VALENCE_DB_PASSWORD=your-secure-password

# API Keys
OPENAI_API_KEY=sk-...

# Federation
FEDERATION_ENABLED=true
FEDERATION_DOMAIN=valence.example.com
FEDERATION_PORT=8443

# Node Identity (auto-generated on first run)
# NODE_DID=did:vkb:web:valence.example.com
```

### 5. Configure Nginx

Create `/etc/nginx/sites-available/valence`:

```nginx
server {
    listen 80;
    server_name valence.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name valence.example.com;

    ssl_certificate /etc/letsencrypt/live/valence.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/valence.example.com/privkey.pem;
    ssl_protocols TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # Well-known endpoints for federation discovery
    location /.well-known/vfp-node-metadata {
        proxy_pass http://127.0.0.1:8000/federation/metadata;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /.well-known/vfp-trust-anchors {
        proxy_pass http://127.0.0.1:8000/federation/trust-anchors;
        proxy_set_header Host $host;
    }

    # Federation API
    location /federation/ {
        proxy_pass http://127.0.0.1:8000/federation/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # MCP endpoint (authenticated)
    location /mcp {
        proxy_pass http://127.0.0.1:8000/mcp;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Health check
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/valence /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 6. Get SSL Certificate

```bash
sudo certbot --nginx -d valence.example.com --email admin@example.com --agree-tos --non-interactive
```

### 7. Create Systemd Service

Create `/etc/systemd/system/valence.service`:

```ini
[Unit]
Description=Valence Knowledge Base Server
After=network.target postgresql.service

[Service]
Type=simple
User=valence
Group=valence
WorkingDirectory=/opt/valence/repo
EnvironmentFile=/opt/valence/config/valence.env
ExecStart=/opt/valence/.venv/bin/uvicorn valence.server:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths=/opt/valence

[Install]
WantedBy=multi-user.target
```

Start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable valence
sudo systemctl start valence
```

---

## Federation Configuration

### Enable Federation

In your environment file:

```bash
# Enable federation protocol
FEDERATION_ENABLED=true

# Your node's public domain
FEDERATION_DOMAIN=valence.example.com

# Federation server port (internal)
FEDERATION_PORT=8443
```

### Node Identity

On first run, your node generates an Ed25519 keypair and derives a DID:

```
did:vkb:web:valence.example.com
```

The private key is stored securely at `/opt/valence/config/node_key.pem`.

**⚠️ BACKUP YOUR NODE KEY** — If lost, you'll need to re-establish trust with all peers.

### Trust Configuration

Create `/opt/valence/config/trust.yml`:

```yaml
# Nodes you explicitly trust
trust_anchors:
  - did: did:vkb:web:valence2.zonk1024.net
    trust_level: 0.8
    domains: [tech, philosophy]
    notes: "Secondary personal node"

# Default trust for new peers
default_trust: 0.0

# Minimum trust required to accept beliefs
min_import_trust: 0.3

# Auto-adjust trust based on belief quality
trust_decay_enabled: true
trust_decay_rate: 0.01  # Per day of inactivity
```

---

## Verification

### Check Services

```bash
# All services running?
sudo systemctl status postgresql nginx valence

# Check logs
sudo journalctl -u valence -f
```

### Test Endpoints

```bash
# Health check
curl https://valence.example.com/health

# Federation metadata
curl https://valence.example.com/.well-known/vfp-node-metadata

# Should return your DID document
```

### Verify SSL

```bash
# Check certificate
openssl s_client -connect valence.example.com:443 -servername valence.example.com < /dev/null 2>/dev/null | openssl x509 -noout -dates
```

### Test Federation

```bash
# From the server
source /opt/valence/config/valence.env
cd /opt/valence/repo
python scripts/federation_demo.py
```

---

## Multi-Node Setup

For testing federation between two nodes (e.g., valence.zonk1024.net and valence2.zonk1024.net):

### Node 1 Configuration

```bash
# .env.pod for node 1
export VALENCE_DOMAIN="valence.zonk1024.net"
```

### Node 2 Configuration

```bash
# .env.pod for node 2
export VALENCE_DOMAIN="valence2.zonk1024.net"
```

### Connect Nodes

```bash
# On node 1: Add node 2 as peer
valence peer add did:vkb:web:valence2.zonk1024.net --trust 0.8 --name "Node 2"

# On node 2: Add node 1 as peer
valence peer add did:vkb:web:valence.zonk1024.net --trust 0.8 --name "Node 1"

# Test connectivity
valence query "test" --scope federated
```

---

## Firewall Configuration

Required ports:

| Port | Protocol | Purpose |
|------|----------|---------|
| 22 | TCP | SSH |
| 80 | TCP | HTTP (redirect to HTTPS) |
| 443 | TCP | HTTPS (federation + API) |

```bash
# UFW example
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

---

## Backup Strategy

### What to Backup

| Path | Content | Frequency |
|------|---------|-----------|
| `/opt/valence/config/node_key.pem` | Node identity | Once (critical!) |
| `/opt/valence/config/tokens.json` | API tokens | On change |
| `/opt/valence/config/trust.yml` | Trust configuration | On change |
| PostgreSQL database | All beliefs | Daily |

### Database Backup

```bash
# Backup
sudo -u postgres pg_dump valence | gzip > valence_$(date +%Y%m%d).sql.gz

# Restore
gunzip < valence_20260204.sql.gz | sudo -u postgres psql valence
```

---

## Troubleshooting

### Common Issues

**"Connection refused" on federation endpoints:**
- Check nginx is running: `sudo systemctl status nginx`
- Check valence service: `sudo systemctl status valence`
- Check firewall: `sudo ufw status`

**SSL certificate errors:**
- Ensure DNS is propagated: `dig +short your.domain.com`
- Check certbot: `sudo certbot certificates`
- Renew manually: `sudo certbot renew --force-renewal`

**Database connection issues:**
- Check PostgreSQL: `sudo systemctl status postgresql`
- Test connection: `psql -h localhost -U valence -d valence`
- Check pg_hba.conf for local connections

**Node identity issues:**
- Verify key exists: `ls -la /opt/valence/config/node_key.pem`
- Check permissions: `chmod 600 /opt/valence/config/node_key.pem`
- Regenerate if needed (will require re-establishing trust)

---

## Next Steps

1. **Add trusted peers** → See [OPERATIONS.md](./OPERATIONS.md)
2. **Configure privacy settings** → Review belief visibility
3. **Set up monitoring** → Add health checks to your monitoring system
4. **Enable backups** → Automate database and key backups

---

*"Deploy once, federate everywhere."*
