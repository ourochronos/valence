# Trust Graph Specification

*Component 4 of Valence: Distributed Epistemic Infrastructure*

---

## Overview

The Trust Graph is each agent's **personal map of who they trust, how much, and for what**. Unlike centralized reputation systems, trust is:
- **Subjective**: My trust in you is mine alone
- **Directional**: A→B trust differs from B→A
- **Domain-specific**: Trust for code ≠ trust for medical advice
- **Decayable**: Unused relationships fade over time
- **Composable**: Transitive trust enables discovery beyond direct connections

---

## Core Data Structures

### TrustEdge

The fundamental unit: a directed, weighted, domain-scoped trust relationship.

```
TrustEdge {
  id: UUID
  from: AgentIdentity.id          # The truster (always "me" in my graph)
  to: AgentIdentity.id            # The trustee
  
  # Trust levels
  trust_level: float              # 0.0-1.0, overall trust
  domain_trust: Map<domain, float> # Domain-specific overrides
  
  # Provenance
  basis: TrustBasis[]             # Why do I trust them?
  
  # Temporal
  created: timestamp
  updated: timestamp
  last_used: timestamp            # For decay calculation
  
  # Metadata
  notes: string | null            # Personal annotation
  tags: string[]                  # User-defined categorization
}
```

### TrustBasis

Why do you trust this agent? Evidence matters.

```
TrustBasis {
  type: enum {
    DIRECT_INTERACTION    # I've worked with them
    VERIFICATION_HISTORY  # They verified claims I made
    FEDERATION_MEMBER     # We share a trusted group
    TRANSITIVE            # Someone I trust trusts them
    EXTERNAL_REPUTATION   # Imported from another system
    MANUAL                # I decided to trust them
  }
  confidence: float         # How confident in this basis?
  evidence: Evidence[]      # Supporting data
  timestamp: timestamp
}

Evidence {
  type: string              # belief_verified, discrepancy_found, etc.
  reference_id: UUID        # The belief/verification/event
  weight: float             # Contribution to trust
}
```

### TrustGraph

The per-agent container for all trust relationships.

```
TrustGraph {
  owner: AgentIdentity.id
  edges: Map<AgentIdentity.id, TrustEdge>
  
  # Cached computations
  transitive_cache: Map<(target_id, domain, max_hops), CachedTrust>
  
  # Configuration
  config: TrustConfig {
    default_trust: float                 # Trust for unknown agents (default: 0.1)
    decay_rate: float                    # Trust decay per day (default: 0.001)
    decay_floor: float                   # Minimum trust after decay (default: 0.0)
    transitive_damping: float            # Multiplier per hop (default: 0.7)
    max_transitive_hops: int             # Limit for path search (default: 3)
    sybil_threshold: int                 # Min distinct trust sources (default: 2)
  }
  
  # Stats
  stats: {
    edge_count: int
    last_pruned: timestamp
    avg_trust: float
    domain_coverage: Map<domain, int>
  }
}
```

---

## Trust Levels

### Continuous Model (0.0 - 1.0)

Trust is a **continuous float**, not discrete tiers. However, semantic thresholds help interpretation:

| Range | Interpretation | Behavior |
|-------|----------------|----------|
| 0.0 | **Distrust** | Actively discount their claims |
| 0.01-0.2 | **Skeptical** | Treat claims as unverified |
| 0.2-0.4 | **Neutral** | Default stranger treatment |
| 0.4-0.6 | **Cautious trust** | Consider but verify |
| 0.6-0.8 | **Trust** | Generally accept claims |
| 0.8-0.95 | **High trust** | Accept without immediate verification |
| 0.95-1.0 | **Full trust** | Rarely warranted; self-only |

**Why continuous?**
- Discrete tiers lose precision
- Math works better (damping, decay, aggregation)
- Agents can set their own semantic thresholds
- Avoids "tier boundary gaming"

**Special values:**
- `0.0` = Active distrust (may want to explicitly ignore)
- `null` / missing = No relationship (use `default_trust`)
- `1.0` = Reserved for self-trust only; external 1.0 is discouraged

---

## Domain-Specific Trust

Trust is not monolithic. I might trust you for:
- `code:rust` but not `code:javascript`
- `medical:general` but not `medical:surgery`
- `politics:analysis` but not `politics:predictions`

### Domain Hierarchy

Domains use hierarchical dot-notation:
```
code
code:rust
code:rust:async
```

Trust lookup walks up the hierarchy:
1. Check `code:rust:async` → if exists, use it
2. Check `code:rust` → if exists, use it
3. Check `code` → if exists, use it
4. Fall back to `trust_level` (overall)

### Domain Examples

```
domains = {
  "code": 0.8,
  "code:security": 0.9,     # I really trust their security opinions
  "code:frontend": 0.4,     # But less so for frontend
  "finance": 0.3,           # Not their area
  "science:physics": 0.85,
}
```

### Domain Inheritance Rules

1. **Explicit overrides win**: If `code:rust` is set, use it
2. **Walk up hierarchy**: Unset children inherit from parent
3. **Overall as fallback**: `trust_level` is the root
4. **No negative inheritance**: Child cannot be auto-lower than parent

---

## Transitive Trust

I trust Alice. Alice trusts Bob. Should I trust Bob?

### The Model

Transitive trust is **computed, not stored**. Formula:

```
transitive_trust(me, target, domain) = max over all paths P {
  Π (edge.trust(domain) × damping^hop)  for each edge in P
}
```

**Damping factor** (default 0.7): Each hop reduces trust.
- 1 hop: 0.8 × 0.7 = 0.56
- 2 hops: 0.8 × 0.7 × 0.8 × 0.7 = 0.31
- 3 hops: diminishing returns

**Max hops** (default 3): Limits computational cost and prevents trust dilution.

### Why Max, Not Sum/Average?

- **Sum**: Would inflate trust with many weak paths
- **Average**: Would undervalue strong paths
- **Max**: Best path wins; additional paths can only help (Sybil mitigation uses min distinct sources)

### Transitive Trust Bounds

Transitive trust can never exceed direct trust to intermediate agents:
```
transitive_trust(A→C via B) ≤ min(trust(A→B), trust(B→C))
```

This prevents trust laundering.

---

## Trust Decay

Relationships fade without maintenance.

### Decay Formula

```
effective_trust(edge, now) = max(
  edge.trust_level × decay_factor(edge.last_used, now),
  config.decay_floor
)

decay_factor(last_used, now) = 
  exp(-config.decay_rate × days_since(last_used))
```

Default: 0.1% decay per day → ~30% reduction after 1 year of no interaction.

### What Refreshes Trust?

The `last_used` timestamp updates when:
- You query beliefs from this agent
- You verify one of their claims
- You explicitly update the trust edge
- They verify one of your claims

### Why Decay?

1. **Staleness**: Trust built 5 years ago may not reflect current capabilities
2. **Network hygiene**: Prevents accumulation of dead relationships
3. **Active participation incentive**: Rewards ongoing engagement
4. **Sybil resistance**: Can't pre-build massive trust networks and sit on them

### Decay Exemptions

Some trust sources resist decay:
- `MANUAL` basis decays slower (0.01% per day)
- Trust above 0.9 decays at half rate
- Agents can mark edges as "persistent" (no decay, but flagged for review)

---

## Sybil Resistance

The attack: Create 1000 fake agents who all trust each other, inject false beliefs.

### Defense Layers

**1. Distinct Source Requirement**
For transitive trust to count, it must come from `sybil_threshold` (default 2) distinct direct trust sources.

```
If I trust Alice and Bob, and both trust Mallory:
  → Mallory gets transitive trust (2 sources)

If I only trust Alice, and Alice trusts 100 Sybils:
  → Sybils get minimal transitive trust (1 source)
```

**2. Path Independence**
Only count paths through different first-hop nodes. Mallory creating 1000 agents who all trust her doesn't help if they all route through the same first hop.

**3. New Agent Skepticism**
- Trust to agents with `age < 30 days` is capped at 0.3
- Trust from agents with `verification_count < 5` has reduced weight
- Transitive trust through new agents applies 0.5× multiplier

**4. Velocity Limits**
- Maximum trust gain per day: 0.1
- Maximum new edges per day: 20
- Maximum trust propagation depth: 3 hops

**5. Verification Requirement**
High trust (>0.7) requires at least one of:
- Direct interaction basis
- Mutual verification history
- Federation membership

Purely transitive trust caps at 0.6.

---

## Graph Invariants

These must always hold:

1. **Self-trust is 1.0**: `trust(me, me) = 1.0` always
2. **Trust is bounded**: `0.0 ≤ trust ≤ 1.0`
3. **No orphan edges**: Every edge points to a known AgentIdentity
4. **Timestamps monotonic**: `created ≤ updated ≤ now`
5. **Domain consistency**: Domain overrides ≤ hierarchy constraints
6. **Decay never increases**: `effective_trust(t2) ≤ effective_trust(t1)` for t2 > t1

---

## Storage Requirements

### Per-Agent Storage

Each agent stores:
- Own TrustGraph (edges to others)
- Cached transitive trust computations
- Trust config

**Not stored:**
- Other agents' trust graphs (privacy)
- Global trust scores (no central authority)

### Estimate

- Average edges per agent: ~100-1000
- Edge size: ~500 bytes
- Per-agent storage: 50KB - 500KB
- Index overhead: ~20%

### Indices Required

```sql
-- Primary lookups
CREATE INDEX ON trust_edges (owner_id);
CREATE INDEX ON trust_edges (owner_id, to_id);
CREATE INDEX ON trust_edges (owner_id, domain) WHERE domain IS NOT NULL;

-- Decay management
CREATE INDEX ON trust_edges (last_used);

-- Trust level queries
CREATE INDEX ON trust_edges (owner_id, trust_level) WHERE trust_level > 0.5;
```

---

## Privacy Model

### What's Private

- My trust graph is **mine alone**
- Others cannot query who I trust
- Trust levels are not shared by default

### What's Shareable

- I can **choose** to expose:
  - That I trust someone (boolean)
  - My trust level for them (float)
  - My domain-specific trust (map)
- Exposure is opt-in per edge

### Federation Exception

Federation members may share:
- Aggregate trust scores (not individual graphs)
- "N members trust agent X with avg score Y"
- Individual trust only with explicit consent

---

## Edge Cases

### Self-Loops
`trust(A, A)` = 1.0, implicit, never stored.

### Missing Agents
Trust edge to unknown AgentIdentity → resolve or mark "unverified"

### Conflicting Bases
Multiple TrustBasis entries may disagree → weighted average by confidence

### Trust Oscillation
Agent rapidly changes trust up/down → velocity limits prevent gaming

### Negative Trust
`trust = 0.0` means distrust. Beliefs from distrusted agents should be actively discounted or flagged, not just ignored.

---

## Relationship to Other Components

- **Identity**: TrustEdge.to points to AgentIdentity
- **Beliefs**: Trust affects belief confidence weighting
- **Verification**: Successful verifications build trust
- **Federation**: Shared trust enables group knowledge
- **Query Protocol**: Trust weights rank query results
- **Reputation**: Network reputation informs (but doesn't override) personal trust

---

*"Trust is the currency of knowledge. Guard it carefully."*
