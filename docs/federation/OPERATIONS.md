# Valence Federation Operations

> Day-to-day operations, monitoring, and troubleshooting for federated nodes.

---

## CLI Quick Reference

### Peer Management

```bash
# Add a trusted peer
valence peer add did:vkb:web:peer.example.com --trust 0.8 --name "Friendly Node"

# List all peers
valence peer list

# Remove a peer
valence peer remove did:vkb:web:peer.example.com

# Update trust level
valence peer add did:vkb:web:peer.example.com --trust 0.6  # Updates existing
```

### Belief Sharing

```bash
# Export beliefs for a peer
valence export --to did:vkb:web:peer.example.com \
  --domain tech \
  --min-confidence 0.7 \
  --output beliefs.json

# Import beliefs from a peer
valence import beliefs.json --from did:vkb:web:peer.example.com

# Import with trust override (for unknown peers)
valence import beliefs.json --from did:vkb:web:new-peer.example.com --trust 0.5
```

### Federated Queries

```bash
# Query local only (default)
valence query "PostgreSQL optimization"

# Query including federated beliefs
valence query "PostgreSQL optimization" --scope federated

# Filter by domain
valence query "performance" --scope federated --domain tech

# See belief sources in results
# Results show: LOCAL or did:vkb:web:peer.example.com (trust: 80%)
```

### Database Operations

```bash
# Initialize/check database
valence init

# Show statistics
valence stats

# List recent beliefs
valence list -n 20

# Detect conflicts
valence conflicts
```

---

## Common Operations

### Adding a New Peer

1. **Verify the peer's node metadata:**
   ```bash
   curl https://peer.example.com/.well-known/vfp-node-metadata | jq .
   ```

2. **Add to your trust registry:**
   ```bash
   valence peer add did:vkb:web:peer.example.com \
     --trust 0.5 \
     --name "New Research Partner"
   ```

3. **Share some beliefs (optional):**
   ```bash
   valence export --to did:vkb:web:peer.example.com \
     --domain research \
     --output to_share.json
   # Then send to_share.json to the peer via secure channel
   ```

### Sharing Beliefs with a Peer

```bash
# 1. Export beliefs
valence export \
  --to did:vkb:web:peer.example.com \
  --domain tech \
  --min-confidence 0.7 \
  --limit 100 \
  --output export.json

# 2. Review what you're sharing
cat export.json | jq '.beliefs | length'  # Count
cat export.json | jq '.beliefs[0]'         # First belief

# 3. Send to peer (out of band - email, secure transfer, etc.)
# The peer then imports with:
# valence import export.json --from did:vkb:web:your-node.example.com
```

### Importing Beliefs

```bash
# 1. Receive the export file from peer

# 2. Check the contents
cat received.json | jq '.exporter_did'     # Who sent this?
cat received.json | jq '.beliefs | length'  # How many beliefs?

# 3. Import (peer must be in trust registry)
valence import received.json --from did:vkb:web:sender.example.com

# 4. Check import results
valence list -n 10  # See recent beliefs
valence stats       # Check counts
```

### Adjusting Trust Levels

Trust affects how peer beliefs are weighted in queries:

```bash
# View current trust
valence peer list

# Increase trust after good experience
valence peer add did:vkb:web:peer.example.com --trust 0.9

# Decrease trust if quality drops
valence peer add did:vkb:web:peer.example.com --trust 0.4

# Remove completely (beliefs remain but marked)
valence peer remove did:vkb:web:peer.example.com
```

---

## Monitoring

### Health Checks

```bash
# Quick health check
curl -s https://valence.example.com/health | jq .

# Expected response:
# {
#   "status": "healthy",
#   "database": "connected",
#   "federation": "enabled",
#   "version": "0.1.0"
# }
```

### Service Status

```bash
# Check all services
sudo systemctl status valence postgresql nginx

# Check federation specifically
curl -s https://valence.example.com/.well-known/vfp-node-metadata | jq '.id'

# Should return your DID
```

### Log Monitoring

```bash
# Follow valence logs
sudo journalctl -u valence -f

# Recent errors
sudo journalctl -u valence --since "1 hour ago" | grep -i error

# Federation-specific logs
sudo journalctl -u valence | grep -i federation

# Nginx access logs
sudo tail -f /var/log/nginx/access.log | grep federation
```

### Database Health

```bash
# Connect to database
sudo -u postgres psql -d valence

# Check table sizes
\dt+

# Count beliefs by source
SELECT 
  CASE WHEN is_local THEN 'local' ELSE 'federated' END as source,
  COUNT(*) 
FROM beliefs 
GROUP BY is_local;

# Check embedding coverage
SELECT 
  COUNT(*) as total,
  COUNT(embedding) as with_embedding,
  ROUND(100.0 * COUNT(embedding) / COUNT(*), 1) as pct
FROM beliefs;

# Recent activity
SELECT created_at, content 
FROM beliefs 
ORDER BY created_at DESC 
LIMIT 10;
```

### Metrics to Watch

| Metric | Warning | Critical |
|--------|---------|----------|
| Disk usage | >70% | >90% |
| Database connections | >50 | >100 |
| Response time | >500ms | >2s |
| SSL cert expiry | <30 days | <7 days |
| Failed auth attempts | >10/hour | >50/hour |

---

## Troubleshooting

### Federation Issues

**Peer connection fails:**
```bash
# 1. Check DNS resolution
dig +short peer.example.com

# 2. Check SSL certificate
openssl s_client -connect peer.example.com:443 -servername peer.example.com

# 3. Test metadata endpoint
curl -v https://peer.example.com/.well-known/vfp-node-metadata

# 4. Check your outbound connectivity
curl -I https://peer.example.com/health
```

**Beliefs not syncing:**
```bash
# 1. Verify peer is in trust registry
valence peer list

# 2. Check trust level (must be > min_import_trust)
valence peer list | grep peer.example.com

# 3. Verify belief visibility (must be 'federated' or 'trusted')
valence query "topic" --verbose

# 4. Check for import errors
sudo journalctl -u valence | grep -i import
```

**Trust not updating:**
```bash
# 1. Trust is stored in trust.yml
cat /opt/valence/config/trust.yml

# 2. Check peer registry
valence peer list

# 3. Manual trust update
valence peer add did:vkb:web:peer.example.com --trust 0.7
```

### Database Issues

**Connection refused:**
```bash
# 1. Check PostgreSQL is running
sudo systemctl status postgresql

# 2. Check it's listening
sudo ss -tlnp | grep 5432

# 3. Check pg_hba.conf allows connections
sudo cat /etc/postgresql/16/main/pg_hba.conf | grep valence

# 4. Test connection
psql -h localhost -U valence -d valence -c "SELECT 1"
```

**Slow queries:**
```bash
# 1. Check for missing indexes
sudo -u postgres psql -d valence -c "\di"

# 2. Analyze tables
sudo -u postgres psql -d valence -c "ANALYZE beliefs;"

# 3. Check query plan
sudo -u postgres psql -d valence -c "EXPLAIN ANALYZE SELECT * FROM beliefs WHERE embedding <=> '[...]' LIMIT 10;"

# 4. Rebuild indexes if needed
sudo -u postgres psql -d valence -c "REINDEX INDEX idx_beliefs_embedding;"
```

**Disk space issues:**
```bash
# 1. Check database size
sudo -u postgres psql -c "SELECT pg_size_pretty(pg_database_size('valence'));"

# 2. Find large tables
sudo -u postgres psql -d valence -c "SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) FROM pg_catalog.pg_statio_user_tables ORDER BY pg_total_relation_size(relid) DESC LIMIT 5;"

# 3. Vacuum to reclaim space
sudo -u postgres psql -d valence -c "VACUUM FULL ANALYZE;"
```

### SSL/TLS Issues

**Certificate expired:**
```bash
# 1. Check expiry
sudo certbot certificates

# 2. Renew
sudo certbot renew

# 3. If renewal fails, force
sudo certbot certonly --nginx -d valence.example.com --force-renewal

# 4. Reload nginx
sudo systemctl reload nginx
```

**Certificate verification fails:**
```bash
# 1. Check certificate chain
openssl s_client -connect valence.example.com:443 -servername valence.example.com -showcerts

# 2. Verify against CA
curl -v https://valence.example.com/health 2>&1 | grep -i ssl

# 3. Check nginx config
sudo nginx -t
```

### Authentication Issues

**Token rejected:**
```bash
# 1. List valid tokens
sudo valence-token list

# 2. Verify specific token
sudo valence-token verify "your-token-here"

# 3. Check token hasn't expired
sudo valence-token list | grep -A2 "client-id"

# 4. Create new token if needed
sudo valence-token create --client-id myapp --description "New token"
```

**DID verification fails:**
```bash
# 1. Check your node key exists
ls -la /opt/valence/config/node_key.pem

# 2. Verify key permissions
stat /opt/valence/config/node_key.pem
# Should be 600 (rw-------)

# 3. Regenerate if corrupted (WARNING: breaks trust!)
# Backup first, then:
# openssl genpkey -algorithm ED25519 -out /opt/valence/config/node_key.pem
```

---

## Security Considerations

### Access Control

```bash
# View active tokens
sudo valence-token list

# Revoke compromised token
sudo valence-token revoke --client-id compromised-client

# Rotate tokens periodically
# Create new → update clients → revoke old
```

### Firewall Rules

```bash
# View current rules
sudo ufw status numbered

# Only allow necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP redirect
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 5432/tcp   # PostgreSQL (local only)
```

### Key Security

```bash
# Node key should be:
# - Owned by valence user
# - Mode 600
# - Backed up securely
# - Never transmitted

# Check
stat /opt/valence/config/node_key.pem

# Fix if needed
sudo chown valence:valence /opt/valence/config/node_key.pem
sudo chmod 600 /opt/valence/config/node_key.pem
```

### Audit Logging

```bash
# Federation events are logged to journal
sudo journalctl -u valence | grep -E "(peer|trust|import|export)"

# Authentication attempts
sudo journalctl -u valence | grep -i auth

# Failed operations
sudo journalctl -u valence | grep -i "error\|fail"
```

---

## Maintenance Tasks

### Daily

- [ ] Check service health: `curl https://your.domain/health`
- [ ] Review error logs: `journalctl -u valence --since yesterday | grep error`

### Weekly

- [ ] Review trust levels: `valence peer list`
- [ ] Check disk usage: `df -h`
- [ ] Review slow queries (if monitoring enabled)

### Monthly

- [ ] Database vacuum: `sudo -u postgres vacuumdb --analyze valence`
- [ ] SSL certificate check: `sudo certbot certificates`
- [ ] Rotate access tokens
- [ ] Review and update peer trust levels
- [ ] Backup verification

### As Needed

- [ ] Apply security updates: `sudo apt update && sudo apt upgrade`
- [ ] Update Valence: `cd /opt/valence/repo && git pull && pip install -e .`
- [ ] Re-index embeddings after schema changes

---

## Emergency Procedures

### Node Compromise

1. **Immediately revoke all tokens:**
   ```bash
   sudo valence-token list
   sudo valence-token revoke --client-id <each-client>
   ```

2. **Disable federation:**
   ```bash
   sudo systemctl stop valence
   # Edit /opt/valence/config/valence.env
   # Set FEDERATION_ENABLED=false
   ```

3. **Notify peers** (out of band) that your node DID should not be trusted

4. **Investigate and remediate**

5. **Rotate node key if needed** (regenerates DID - requires re-establishing all trust)

### Database Recovery

```bash
# 1. Stop valence
sudo systemctl stop valence

# 2. Restore from backup
gunzip < valence_backup.sql.gz | sudo -u postgres psql valence

# 3. Verify
sudo -u postgres psql -d valence -c "SELECT COUNT(*) FROM beliefs;"

# 4. Restart
sudo systemctl start valence
```

### Full Node Recovery

1. Deploy fresh server using [DEPLOYMENT.md](./DEPLOYMENT.md)
2. Restore node key from backup (critical!)
3. Restore database from backup
4. Restore trust configuration
5. Verify endpoints respond correctly
6. Re-verify with peers

---

## Performance Tuning

### PostgreSQL

```sql
-- For nodes with many beliefs (>100k)
-- Add to postgresql.conf:
shared_buffers = 1GB
effective_cache_size = 3GB
maintenance_work_mem = 256MB
work_mem = 16MB

-- For embedding queries
-- Increase lists parameter for ivfflat index
-- (requires reindex)
```

### Application

```bash
# Increase workers for concurrent requests
# Edit valence.service:
ExecStart=/opt/valence/.venv/bin/uvicorn valence.server:app \
  --host 127.0.0.1 --port 8000 \
  --workers 4
```

### Nginx

```nginx
# Add caching for federation metadata
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=federation:10m max_size=100m;

location /.well-known/vfp-node-metadata {
    proxy_cache federation;
    proxy_cache_valid 200 5m;
    proxy_pass http://127.0.0.1:8000/federation/metadata;
}
```

---

## Getting Help

- **Documentation**: [Valence docs](/docs/)
- **Protocol spec**: [FEDERATION_PROTOCOL.md](../FEDERATION_PROTOCOL.md)
- **Issues**: [GitHub Issues](https://github.com/orobobos/valence/issues)

---

*"Operate with sovereignty, troubleshoot with precision."*
