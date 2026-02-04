# Business Model

*How Valence sustains itself without betraying its principles.*

---

## Core Tension

Valence's principles demand user sovereignty, structural integrity, and resistance to capture. Most business models conflict with these — monetizing users, creating lock-in, or centralizing control.

This document describes how we generate revenue while maintaining 10/10 principle alignment.

---

## Revenue Streams

### 1. Sovereign Hosted Service

**The model**: We run infrastructure; users own identity.

| Component | Who Owns It |
|-----------|-------------|
| Domain | User (`knowledge.acme.com`) |
| DID/Identity | User (anchored to their domain) |
| Trust relationships | User (reference their domain) |
| Data | User (full export anytime) |
| Compute/storage | Valence (rented infrastructure) |

**How it works**:

1. User registers, points their subdomain to our infrastructure
   ```
   knowledge.acme.com  CNAME  hosted.valence.network
   ```

2. We provision TLS (Let's Encrypt) for their domain

3. Their DID is `did:web:knowledge.acme.com` — anchored to *their* domain, not ours

4. All federation, trust relationships, and belief provenance reference their domain

5. If they migrate: DNS points elsewhere, trust graph follows seamlessly

**Why this achieves 10/10 alignment**:

- **User sovereignty**: They own the trust anchor (domain). We're infrastructure, not identity.
- **Structurally incapable of betrayal**: Lock-in is architecturally impossible. Migration = DNS change + data export.
- **Security through structure**: The design makes capture impossible, not policy promises.

**Pricing** (indicative):

| Tier | Storage | Beliefs | Price |
|------|---------|---------|-------|
| Personal | 1 GB | 10K | Free |
| Pro | 10 GB | 100K | $10/mo |
| Team | 100 GB | 1M | $50/mo |
| Enterprise | Custom | Custom | Contact |

---

### 2. Enterprise Federation

Large organizations running private federation networks.

**What they get**:
- Private federation namespace (beliefs stay internal)
- Advanced compliance features (audit logs, retention policies, cryptographic deletion)
- Priority support and SLAs
- Custom integrations

**What we get**:
- Annual contracts
- Higher margins
- Case studies and credibility

**Pricing**: $10K-100K/year depending on scale and requirements.

---

### 3. Protocol Services

Optional paid services that enhance the network without creating dependency.

| Service | Description | Model |
|---------|-------------|-------|
| **Hosted Embeddings** | We compute embeddings so you don't need OpenAI key | Per-request or subscription |
| **Verification Staking** | We stake verification bounties on your behalf | Commission on rewards |
| **Trust Bridge** | Cross-federation trust attestation | Per-attestation fee |
| **Compliance Reports** | Automated regulatory compliance documentation | Per-report or subscription |

All services are optional. Self-hosters can run everything themselves.

---

## Principle Alignment Checklist

| Principle | How Business Model Aligns |
|-----------|---------------------------|
| **Users own their data** | Data export is always available. No lock-in. |
| **Sovereignty** | DNS-anchored identity means users own their trust anchor |
| **Structurally incapable of betrayal** | Migration is DNS change + export. Architecture prevents capture. |
| **Security through structure** | DID:web on user domain = structural, not policy |
| **Transparency** | Pricing is public. No hidden costs or gotchas. |
| **Designed to survive being stolen** | Protocol is open. Self-hosting always possible. |
| **Mission permanence** | PBC structure. Cannot be sold or pivoted. |

**Score: 10/10** — No principle compromises.

---

## What We Don't Do

- **Sell user data** — Never. Not anonymized, not aggregated, not ever.
- **Advertising** — No. Users are customers, not products.
- **Mandatory fees for core protocol** — Self-hosting is always free.
- **Artificial lock-in** — Export is full and usable. No "you can leave but good luck."
- **Tiered access to security** — Security features are not premium. Everyone gets the same protection.

---

## Sustainability Model

**Phase 1 (Now → Product-Market Fit)**:
- Bootstrap on savings/grants
- Focus on adoption, not revenue
- Hosted free tier acquires users

**Phase 2 (PMF → Growth)**:
- Pro/Team tiers convert power users
- Enterprise pilots generate revenue
- Protocol services launch

**Phase 3 (Growth → Sustainability)**:
- Enterprise becomes primary revenue
- Network effects reduce CAC
- Protocol services scale with network

---

## Competitive Moat

Not lock-in. The moat is:

1. **Network effects** — More nodes = more valuable federation
2. **Trust accumulation** — Reputation compounds over time, hard to restart elsewhere
3. **Switching cost is legitimately high** — Not artificial lock-in, but real value accumulation
4. **Open protocol paradox** — Competitors using the protocol grow the network

If someone forks Valence and does it better, that's success — the pattern exists in the world.

---

## FAQ

**Q: If self-hosting is always possible, why would anyone pay?**

A: Same reason people pay for email hosting. Convenience, reliability, support. Running infrastructure is work. Most users and companies prefer to pay for that.

**Q: What if a big cloud provider offers Valence hosting?**

A: Good. They grow the network. We compete on trust, mission alignment, and being the canonical implementation. Some users will choose AWS; sovereignty-conscious users will choose us.

**Q: How do you prevent enshittification?**

A: PBC structure with constitutional principles. The entity cannot be sold. Principles are load-bearing. Any change that violates them isn't evolution — it's corruption, and the structure prevents it.

---

*Revenue enables mission. Mission constrains revenue. This is the balance.*
