# Valence Federation Schema

*Database schema extensions for federation support*

This document describes the database schema additions required for Valence federation. All new tables are designed to work alongside the existing schema in `migrations/schema.sql`.

---

## 1. Overview

### 1.1 New Tables

| Table | Purpose |
|-------|---------|
| `federation_nodes` | Node identity, capabilities, and status |
| `belief_provenance` | Federation path and origin tracking |
| `node_trust` | Node-to-node trust relationships |
| `user_node_trust` | User preferences for node trust |
| `belief_trust_annotations` | Per-belief trust adjustments |
| `aggregated_beliefs` | Privacy-preserving aggregates |
| `aggregation_sources` | Links aggregates to contributing sources |
| `tension_resolutions` | Cross-node conflict resolution |
| `consensus_votes` | Voting on resolutions |
| `sync_state` | Sync cursors and vector clocks |
| `sync_events` | Audit log for sync operations |
| `sync_outbound_queue` | Pending outbound sync operations |

### 1.2 Modified Tables

| Table | Changes |
|-------|---------|
| `beliefs` | Add `origin_node_id`, `is_local`, `federation_id`, `visibility`, `share_level` |

---

## 2. Identity Tables

### 2.1 federation_nodes

Stores identity and metadata for federation nodes.

```sql
CREATE TABLE IF NOT EXISTS federation_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- DID identifier (unique)
    did TEXT NOT NULL UNIQUE,
    -- Format: did:vkb:web:domain or did:vkb:key:z6Mk...

    -- Connection information
    federation_endpoint TEXT,
    mcp_endpoint TEXT,

    -- Cryptographic identity
    public_key_multibase TEXT NOT NULL,
    -- Ed25519 public key in multibase format

    -- Node profile (optional)
    name TEXT,
    domains TEXT[] DEFAULT '{}',

    -- Capabilities
    capabilities TEXT[] NOT NULL DEFAULT '{}',
    -- Values: belief_sync, aggregation_participate, aggregation_publish

    -- Status
    status TEXT NOT NULL DEFAULT 'discovered',
    -- Values: discovered, connecting, active, suspended, unreachable

    -- Trust phase (per TRUST_MODEL.md)
    trust_phase TEXT NOT NULL DEFAULT 'observer',
    -- Values: observer, contributor, participant, anchor

    phase_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Protocol version
    protocol_version TEXT NOT NULL DEFAULT '1.0',

    -- Timestamps
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ,
    last_sync_at TIMESTAMPTZ,

    -- Metadata
    metadata JSONB DEFAULT '{}',

    CONSTRAINT federation_nodes_valid_status CHECK (
        status IN ('discovered', 'connecting', 'active', 'suspended', 'unreachable')
    ),
    CONSTRAINT federation_nodes_valid_phase CHECK (
        trust_phase IN ('observer', 'contributor', 'participant', 'anchor')
    )
);

CREATE INDEX IF NOT EXISTS idx_federation_nodes_did ON federation_nodes(did);
CREATE INDEX IF NOT EXISTS idx_federation_nodes_status ON federation_nodes(status);
CREATE INDEX IF NOT EXISTS idx_federation_nodes_phase ON federation_nodes(trust_phase);
CREATE INDEX IF NOT EXISTS idx_federation_nodes_domains ON federation_nodes USING GIN (domains);
```

---

## 3. Belief Provenance

### 3.1 belief_provenance

Tracks the federation path of beliefs received from other nodes.

```sql
CREATE TABLE IF NOT EXISTS belief_provenance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to local belief
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,

    -- Federation identity
    federation_id UUID NOT NULL,
    -- Stable ID across federation (same belief on different nodes)

    -- Origin information
    origin_node_id UUID NOT NULL REFERENCES federation_nodes(id),
    origin_belief_id UUID NOT NULL,
    -- Original belief ID on origin node

    -- Cryptographic proof
    origin_signature TEXT NOT NULL,
    signed_at TIMESTAMPTZ NOT NULL,
    signature_verified BOOLEAN DEFAULT FALSE,

    -- Federation path
    hop_count INTEGER NOT NULL DEFAULT 1,
    federation_path TEXT[] NOT NULL DEFAULT '{}',
    -- Array of DIDs showing path from origin

    -- Share level at which belief was shared
    share_level TEXT NOT NULL DEFAULT 'belief_only',
    -- Values: belief_only, with_provenance, full

    -- Reception timestamps
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT belief_provenance_valid_share_level CHECK (
        share_level IN ('belief_only', 'with_provenance', 'full')
    )
);

CREATE INDEX IF NOT EXISTS idx_belief_provenance_belief ON belief_provenance(belief_id);
CREATE INDEX IF NOT EXISTS idx_belief_provenance_federation ON belief_provenance(federation_id);
CREATE INDEX IF NOT EXISTS idx_belief_provenance_origin ON belief_provenance(origin_node_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_belief_provenance_unique
    ON belief_provenance(origin_node_id, origin_belief_id);
```

### 3.2 Beliefs Table Extensions

Add federation columns to existing beliefs table:

```sql
-- Add federation columns to beliefs (backward compatible)
ALTER TABLE beliefs
    ADD COLUMN IF NOT EXISTS origin_node_id UUID REFERENCES federation_nodes(id),
    ADD COLUMN IF NOT EXISTS is_local BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS federation_id UUID,
    ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'private',
    ADD COLUMN IF NOT EXISTS share_level TEXT NOT NULL DEFAULT 'belief_only';

-- Add constraints
ALTER TABLE beliefs
    ADD CONSTRAINT beliefs_valid_visibility CHECK (
        visibility IN ('private', 'trusted', 'federated', 'public')
    ),
    ADD CONSTRAINT beliefs_valid_share_level CHECK (
        share_level IN ('belief_only', 'with_provenance', 'full')
    );

-- Add indexes for federation queries
CREATE INDEX IF NOT EXISTS idx_beliefs_origin_node ON beliefs(origin_node_id)
    WHERE origin_node_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_beliefs_local ON beliefs(is_local);
CREATE INDEX IF NOT EXISTS idx_beliefs_visibility ON beliefs(visibility);
CREATE INDEX IF NOT EXISTS idx_beliefs_federation ON beliefs(federation_id)
    WHERE federation_id IS NOT NULL;
```

---

## 4. Trust Tables

### 4.1 node_trust

Stores trust relationships between this node and other federation nodes.

```sql
CREATE TABLE IF NOT EXISTS node_trust (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- The node being trusted
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,

    -- Trust dimensions (mirrors DimensionalConfidence)
    trust JSONB NOT NULL DEFAULT '{"overall": 0.1}',
    -- Dimensions:
    --   overall: Combined score (0-1)
    --   belief_accuracy: How often their beliefs are corroborated
    --   extraction_quality: Quality of their knowledge extraction
    --   curation_accuracy: How well they handle contradictions
    --   uptime_reliability: Consistent availability
    --   contribution_consistency: Regular, quality participation
    --   endorsement_strength: Trust from others we trust
    --   domain_expertise: Per-domain scores (JSONB nested)

    -- Trust factors
    beliefs_received INTEGER DEFAULT 0,
    beliefs_corroborated INTEGER DEFAULT 0,
    beliefs_disputed INTEGER DEFAULT 0,
    sync_requests_served INTEGER DEFAULT 0,
    aggregation_participations INTEGER DEFAULT 0,

    -- Social trust
    endorsements_received INTEGER DEFAULT 0,
    endorsements_given INTEGER DEFAULT 0,

    -- Relationship timeline
    relationship_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_interaction_at TIMESTAMPTZ,

    -- Manual adjustments (user override)
    manual_trust_adjustment NUMERIC(3,2) DEFAULT 0,
    adjustment_reason TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(node_id)
);

CREATE INDEX IF NOT EXISTS idx_node_trust_node ON node_trust(node_id);
CREATE INDEX IF NOT EXISTS idx_node_trust_overall ON node_trust(
    ((trust->>'overall')::numeric) DESC
);
```

### 4.2 user_node_trust

Allows users to override automatic trust calculations.

```sql
CREATE TABLE IF NOT EXISTS user_node_trust (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- The node
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,

    -- User preference
    trust_preference TEXT NOT NULL DEFAULT 'automatic',
    -- Values: blocked, reduced, automatic, elevated, anchor

    -- Manual trust score (if not automatic)
    manual_trust_score NUMERIC(3,2),

    -- Reason for override
    reason TEXT,

    -- Domain-specific overrides
    domain_overrides JSONB DEFAULT '{}',
    -- Format: { "tech": "elevated", "politics": "blocked" }

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(node_id),
    CONSTRAINT user_node_trust_valid_preference CHECK (
        trust_preference IN ('blocked', 'reduced', 'automatic', 'elevated', 'anchor')
    )
);

CREATE INDEX IF NOT EXISTS idx_user_node_trust_node ON user_node_trust(node_id);
CREATE INDEX IF NOT EXISTS idx_user_node_trust_pref ON user_node_trust(trust_preference);
```

### 4.3 belief_trust_annotations

Per-belief trust adjustments from federation context.

```sql
CREATE TABLE IF NOT EXISTS belief_trust_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,

    -- Annotation type
    type TEXT NOT NULL,
    -- Values: corroboration, dispute, endorsement, flag

    -- Source of annotation
    source_node_id UUID REFERENCES federation_nodes(id) ON DELETE SET NULL,

    -- Corroboration details
    corroboration_attestation JSONB,
    -- Contains: claim_hash, corroboration_level, confidence_boost, etc.

    -- Confidence adjustment
    confidence_delta NUMERIC(3,2) DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,

    CONSTRAINT belief_trust_annotations_valid_type CHECK (
        type IN ('corroboration', 'dispute', 'endorsement', 'flag')
    )
);

CREATE INDEX IF NOT EXISTS idx_belief_trust_annotations_belief ON belief_trust_annotations(belief_id);
CREATE INDEX IF NOT EXISTS idx_belief_trust_annotations_type ON belief_trust_annotations(type);
CREATE INDEX IF NOT EXISTS idx_belief_trust_annotations_source ON belief_trust_annotations(source_node_id);
```

---

## 5. Aggregation Tables

### 5.1 aggregated_beliefs

Stores privacy-preserving aggregates from federation queries.

```sql
CREATE TABLE IF NOT EXISTS aggregated_beliefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Query that produced this aggregate
    query_hash TEXT NOT NULL,
    -- SHA256 of normalized query for deduplication

    query_domain TEXT[] NOT NULL DEFAULT '{}',
    query_semantic TEXT,

    -- Aggregate results
    collective_confidence NUMERIC(3,2) NOT NULL,
    agreement_score NUMERIC(3,2),

    -- Participation stats (not which nodes, just counts)
    contributor_count INTEGER NOT NULL,
    node_count INTEGER NOT NULL,
    total_belief_count INTEGER,

    -- Stance summary (AI-generated)
    stance_summary TEXT,
    key_factors TEXT[],

    -- Privacy guarantees
    privacy_epsilon NUMERIC(5,4) NOT NULL,
    privacy_delta NUMERIC(10,9) NOT NULL,
    privacy_mechanism TEXT NOT NULL DEFAULT 'laplace',

    -- Aggregator info
    aggregator_node_id UUID REFERENCES federation_nodes(id),

    -- Validity
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_aggregated_beliefs_query ON aggregated_beliefs(query_hash);
CREATE INDEX IF NOT EXISTS idx_aggregated_beliefs_domain ON aggregated_beliefs USING GIN (query_domain);
CREATE INDEX IF NOT EXISTS idx_aggregated_beliefs_computed ON aggregated_beliefs(computed_at DESC);
```

### 5.2 aggregation_sources

Links aggregated beliefs to their (anonymized) sources.

```sql
CREATE TABLE IF NOT EXISTS aggregation_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    aggregated_belief_id UUID NOT NULL REFERENCES aggregated_beliefs(id) ON DELETE CASCADE,

    -- Anonymous source tracking (hashed for privacy)
    source_hash TEXT NOT NULL,
    -- SHA256(node_did + salt), not reversible

    -- Contribution metadata (no identifying info)
    contribution_weight NUMERIC(3,2) NOT NULL,
    local_confidence NUMERIC(3,2),
    local_belief_count INTEGER,

    -- Timestamps
    contributed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_aggregation_sources_aggregate ON aggregation_sources(aggregated_belief_id);
```

---

## 6. Tension Resolution Tables

### 6.1 tension_resolutions

Extended tension resolution for cross-node contradictions.

```sql
CREATE TABLE IF NOT EXISTS tension_resolutions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to local tension (if exists)
    tension_id UUID REFERENCES tensions(id) ON DELETE SET NULL,

    -- Cross-node tension identification
    is_cross_node BOOLEAN NOT NULL DEFAULT FALSE,

    -- Participating nodes
    node_a_id UUID REFERENCES federation_nodes(id),
    node_b_id UUID REFERENCES federation_nodes(id),

    -- Resolution proposal
    proposed_resolution TEXT NOT NULL,
    -- Values: supersede_a, supersede_b, accept_both, merge, refer_to_authority

    resolution_rationale TEXT,

    -- Proposer
    proposed_by_node_id UUID REFERENCES federation_nodes(id),
    proposed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Consensus tracking
    consensus_method TEXT NOT NULL DEFAULT 'trust_weighted',
    -- Values: trust_weighted, unanimous, majority, authority

    consensus_threshold NUMERIC(3,2) DEFAULT 0.6,
    current_support NUMERIC(3,2) DEFAULT 0,

    -- Status
    status TEXT NOT NULL DEFAULT 'proposed',
    -- Values: proposed, voting, accepted, rejected, implemented

    resolved_at TIMESTAMPTZ,

    -- Result
    winning_belief_id UUID REFERENCES beliefs(id),
    superseded_belief_id UUID REFERENCES beliefs(id),

    CONSTRAINT tension_resolutions_valid_proposal CHECK (
        proposed_resolution IN ('supersede_a', 'supersede_b', 'accept_both', 'merge', 'refer_to_authority')
    ),
    CONSTRAINT tension_resolutions_valid_status CHECK (
        status IN ('proposed', 'voting', 'accepted', 'rejected', 'implemented')
    )
);

CREATE INDEX IF NOT EXISTS idx_tension_resolutions_tension ON tension_resolutions(tension_id);
CREATE INDEX IF NOT EXISTS idx_tension_resolutions_status ON tension_resolutions(status);
CREATE INDEX IF NOT EXISTS idx_tension_resolutions_nodes ON tension_resolutions(node_a_id, node_b_id);
```

### 6.2 consensus_votes

Individual votes on tension resolutions.

```sql
CREATE TABLE IF NOT EXISTS consensus_votes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    resolution_id UUID NOT NULL REFERENCES tension_resolutions(id) ON DELETE CASCADE,

    -- Voter
    voter_node_id UUID NOT NULL REFERENCES federation_nodes(id),

    -- Vote
    vote TEXT NOT NULL,
    -- Values: support, oppose, abstain

    -- Weight (based on trust at vote time)
    vote_weight NUMERIC(3,2) NOT NULL,

    -- Rationale (optional)
    rationale TEXT,

    -- Signature
    signature TEXT NOT NULL,

    voted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(resolution_id, voter_node_id),
    CONSTRAINT consensus_votes_valid_vote CHECK (
        vote IN ('support', 'oppose', 'abstain')
    )
);

CREATE INDEX IF NOT EXISTS idx_consensus_votes_resolution ON consensus_votes(resolution_id);
CREATE INDEX IF NOT EXISTS idx_consensus_votes_voter ON consensus_votes(voter_node_id);
```

---

## 7. Sync Tables

### 7.1 sync_state

Tracks synchronization state with each peer node.

```sql
CREATE TABLE IF NOT EXISTS sync_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Peer node
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,

    -- Sync cursors
    last_received_cursor TEXT,
    last_sent_cursor TEXT,

    -- Vector clock (for conflict resolution)
    vector_clock JSONB NOT NULL DEFAULT '{}',
    -- Format: { "node_did": sequence_number, ... }

    -- Sync status
    status TEXT NOT NULL DEFAULT 'idle',
    -- Values: idle, syncing, error, paused

    -- Statistics
    beliefs_sent INTEGER DEFAULT 0,
    beliefs_received INTEGER DEFAULT 0,
    last_sync_duration_ms INTEGER,

    -- Error tracking
    last_error TEXT,
    error_count INTEGER DEFAULT 0,

    -- Timestamps
    last_sync_at TIMESTAMPTZ,
    next_sync_scheduled TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(node_id),
    CONSTRAINT sync_state_valid_status CHECK (
        status IN ('idle', 'syncing', 'error', 'paused')
    )
);

CREATE INDEX IF NOT EXISTS idx_sync_state_node ON sync_state(node_id);
CREATE INDEX IF NOT EXISTS idx_sync_state_status ON sync_state(status);
CREATE INDEX IF NOT EXISTS idx_sync_state_next ON sync_state(next_sync_scheduled);
```

### 7.2 sync_events

Audit log for sync operations.

```sql
CREATE TABLE IF NOT EXISTS sync_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Peer node
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,

    -- Event type
    event_type TEXT NOT NULL,
    -- Values: sync_started, sync_completed, sync_failed, belief_sent, belief_received,
    --         cursor_updated, conflict_detected, conflict_resolved

    -- Event details
    details JSONB DEFAULT '{}',

    -- Direction
    direction TEXT NOT NULL DEFAULT 'inbound',
    -- Values: inbound, outbound

    -- Related belief (if applicable)
    belief_id UUID REFERENCES beliefs(id) ON DELETE SET NULL,

    -- Timestamp
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sync_events_node ON sync_events(node_id);
CREATE INDEX IF NOT EXISTS idx_sync_events_type ON sync_events(event_type);
CREATE INDEX IF NOT EXISTS idx_sync_events_time ON sync_events(occurred_at DESC);

-- Partition by time for efficient cleanup
-- (Implementation depends on PostgreSQL version and operational needs)
```

### 7.3 sync_outbound_queue

Pending outbound sync operations.

```sql
CREATE TABLE IF NOT EXISTS sync_outbound_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Target node (NULL = broadcast to all syncing nodes)
    target_node_id UUID REFERENCES federation_nodes(id) ON DELETE CASCADE,

    -- Operation type
    operation TEXT NOT NULL,
    -- Values: share_belief, update_belief, supersede_belief, share_tension

    -- Payload
    belief_id UUID REFERENCES beliefs(id) ON DELETE CASCADE,
    payload JSONB,

    -- Priority
    priority INTEGER NOT NULL DEFAULT 5,
    -- 1 = highest, 10 = lowest

    -- Status
    status TEXT NOT NULL DEFAULT 'pending',
    -- Values: pending, processing, sent, failed, cancelled

    -- Retry tracking
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    last_attempt_at TIMESTAMPTZ,
    last_error TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scheduled_for TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT sync_outbound_queue_valid_status CHECK (
        status IN ('pending', 'processing', 'sent', 'failed', 'cancelled')
    )
);

CREATE INDEX IF NOT EXISTS idx_sync_outbound_queue_target ON sync_outbound_queue(target_node_id);
CREATE INDEX IF NOT EXISTS idx_sync_outbound_queue_status ON sync_outbound_queue(status);
CREATE INDEX IF NOT EXISTS idx_sync_outbound_queue_scheduled ON sync_outbound_queue(scheduled_for)
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_sync_outbound_queue_belief ON sync_outbound_queue(belief_id);
```

---

## 8. Views

### 8.1 Federation Views

```sql
-- Active federation nodes with trust
CREATE OR REPLACE VIEW federation_nodes_with_trust AS
SELECT
    fn.*,
    nt.trust,
    (nt.trust->>'overall')::numeric as trust_overall,
    nt.beliefs_received,
    nt.beliefs_corroborated,
    unt.trust_preference as user_preference
FROM federation_nodes fn
LEFT JOIN node_trust nt ON fn.id = nt.node_id
LEFT JOIN user_node_trust unt ON fn.id = unt.node_id;

-- Federated beliefs with provenance
CREATE OR REPLACE VIEW federated_beliefs AS
SELECT
    b.*,
    bp.federation_id,
    bp.origin_node_id,
    bp.hop_count,
    bp.federation_path,
    bp.share_level as received_share_level,
    fn.did as origin_did,
    fn.name as origin_name
FROM beliefs b
JOIN belief_provenance bp ON b.id = bp.belief_id
JOIN federation_nodes fn ON bp.origin_node_id = fn.id
WHERE b.is_local = FALSE;

-- Sync status overview
CREATE OR REPLACE VIEW sync_status_overview AS
SELECT
    fn.id as node_id,
    fn.did,
    fn.name,
    fn.status as node_status,
    ss.status as sync_status,
    ss.last_sync_at,
    ss.beliefs_sent,
    ss.beliefs_received,
    ss.error_count
FROM federation_nodes fn
LEFT JOIN sync_state ss ON fn.id = ss.node_id
WHERE fn.status = 'active';
```

---

## 9. Migration Strategy

### 9.1 Migration Script

```sql
-- Migration: Add federation support
-- Version: 001_add_federation_tables

BEGIN;

-- 1. Create federation_nodes table
-- (see section 2.1)

-- 2. Add columns to beliefs
-- (see section 3.2)

-- 3. Create remaining tables
-- (in dependency order)

-- 4. Update existing beliefs to be local
UPDATE beliefs SET is_local = TRUE, visibility = 'private'
WHERE is_local IS NULL;

-- 5. Create views
-- (see section 8)

COMMIT;
```

### 9.2 Rollback

```sql
-- Rollback: Remove federation support
-- Use with caution - will lose all federation data

BEGIN;

DROP VIEW IF EXISTS sync_status_overview;
DROP VIEW IF EXISTS federated_beliefs;
DROP VIEW IF EXISTS federation_nodes_with_trust;

DROP TABLE IF EXISTS sync_outbound_queue;
DROP TABLE IF EXISTS sync_events;
DROP TABLE IF EXISTS sync_state;
DROP TABLE IF EXISTS consensus_votes;
DROP TABLE IF EXISTS tension_resolutions;
DROP TABLE IF EXISTS aggregation_sources;
DROP TABLE IF EXISTS aggregated_beliefs;
DROP TABLE IF EXISTS belief_trust_annotations;
DROP TABLE IF EXISTS user_node_trust;
DROP TABLE IF EXISTS node_trust;
DROP TABLE IF EXISTS belief_provenance;
DROP TABLE IF EXISTS federation_nodes;

ALTER TABLE beliefs
    DROP COLUMN IF EXISTS origin_node_id,
    DROP COLUMN IF EXISTS is_local,
    DROP COLUMN IF EXISTS federation_id,
    DROP COLUMN IF EXISTS visibility,
    DROP COLUMN IF EXISTS share_level;

COMMIT;
```

---

## 10. Index Strategy

### 10.1 Query Patterns

| Query Pattern | Index Used |
|---------------|------------|
| Find beliefs from specific node | `idx_beliefs_origin_node` |
| Find federated beliefs only | `idx_beliefs_local` |
| Find beliefs by visibility | `idx_beliefs_visibility` |
| Find nodes by trust level | `idx_node_trust_overall` |
| Find pending sync operations | `idx_sync_outbound_queue_scheduled` |
| Query aggregates by domain | `idx_aggregated_beliefs_domain` |

### 10.2 Performance Considerations

- Vector clock JSONB field uses GIN index for containment queries
- Sync events table should be partitioned by time in production
- Consider partial indexes for common query patterns (e.g., active nodes only)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0-draft | 2025-01-20 | Initial schema design |
