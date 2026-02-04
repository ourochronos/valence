# Valence SDK Design

*Client libraries for TypeScript/JavaScript and Python*

---

## Design Principles

1. **Progressive Complexity** — Simple things simple, complex things possible
2. **Offline-First** — Work without network, sync when available
3. **Type-Safe** — Strong typing with IDE support
4. **Async-Native** — Embrace modern async patterns
5. **Batteries Included** — Auth, caching, retry, rate-limiting built in

---

## Core Abstractions

Both SDKs share these core abstractions:

```
┌─────────────────────────────────────────────────────────────┐
│                      ValenceClient                          │
│  - Authentication                                           │
│  - Connection management                                    │
│  - Rate limiting & retry                                    │
│  - Offline queue                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ BeliefStore  │  │  TrustGraph  │  │  Federation  │      │
│  │              │  │              │  │              │      │
│  │ - create     │  │ - setTrust   │  │ - join       │      │
│  │ - query      │  │ - getTrust   │  │ - share      │      │
│  │ - update     │  │ - compute    │  │ - query      │      │
│  │ - verify     │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Identity    │  │ Verification │  │  SyncEngine  │      │
│  │              │  │              │  │              │      │
│  │ - create     │  │ - submit     │  │ - push       │      │
│  │ - prove      │  │ - dispute    │  │ - pull       │      │
│  │ - rotate     │  │ - resolve    │  │ - conflicts  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## TypeScript/JavaScript SDK

### Installation

```bash
npm install @valence/sdk
# or
yarn add @valence/sdk
# or
pnpm add @valence/sdk
```

### Quick Start

```typescript
import { ValenceClient } from '@valence/sdk';

// Initialize with API key (simplest)
const client = new ValenceClient({
  apiKey: 'val_sk_...'
});

// Or with DID authentication (recommended for agents)
const client = new ValenceClient({
  identity: {
    did: 'did:valence:z6Mk...',
    privateKey: privateKeyBytes
  }
});

// Query beliefs
const results = await client.beliefs.query('machine learning best practices', {
  minConfidence: 0.7,
  domains: ['tech/ai'],
  limit: 10
});

for (const result of results) {
  console.log(`[${result.score.final.toFixed(2)}] ${result.belief.content}`);
}

// Store a belief
const belief = await client.beliefs.create({
  content: 'TypeScript 5.0 introduced decorators as a stable feature',
  confidence: { sourceReliability: 0.95, methodQuality: 0.9 },
  domains: ['tech/typescript'],
  derivation: {
    type: 'observation',
    sources: [{ externalRef: 'https://devblogs.microsoft.com/typescript/' }]
  }
});
```

### Client Configuration

```typescript
interface ValenceClientOptions {
  // Authentication (one required)
  apiKey?: string;
  identity?: {
    did: string;
    privateKey: Uint8Array | string;  // Raw bytes or base64
  };
  oauth?: {
    clientId: string;
    clientSecret: string;
    refreshToken: string;
  };
  
  // Connection
  baseUrl?: string;            // Default: 'https://api.valence.network'
  apiVersion?: string;         // Default: '2024-01-15'
  timeout?: number;            // Default: 30000ms
  
  // Retry & Rate Limiting
  retry?: {
    maxRetries?: number;       // Default: 3
    backoffFactor?: number;    // Default: 2
    retryableErrors?: string[];
  };
  rateLimiting?: {
    enabled?: boolean;         // Default: true
    maxRequestsPerMinute?: number;
  };
  
  // Offline Support
  offline?: {
    enabled?: boolean;         // Default: false
    storage?: 'indexeddb' | 'localstorage' | 'memory';
    syncInterval?: number;     // Default: 60000ms
  };
  
  // Caching
  cache?: {
    enabled?: boolean;         // Default: true
    ttl?: number;              // Default: 60000ms
    maxSize?: number;          // Default: 1000 entries
  };
  
  // Logging
  logger?: Logger;             // Custom logger
  logLevel?: 'debug' | 'info' | 'warn' | 'error';
}
```

### BeliefStore

```typescript
interface BeliefStore {
  // Create
  create(input: CreateBeliefInput): Promise<Belief>;
  createMany(inputs: CreateBeliefInput[]): Promise<Belief[]>;
  
  // Read
  get(id: string, options?: GetBeliefOptions): Promise<Belief | null>;
  getMany(ids: string[]): Promise<Belief[]>;
  
  // Query
  query(semantic: string, options?: QueryOptions): Promise<QueryResult>;
  queryStream(semantic: string, options?: QueryOptions): AsyncIterable<RankedBelief>;
  list(filters?: QueryFilters, options?: ListOptions): Promise<ListResult>;
  
  // Update
  update(id: string, changes: UpdateBeliefInput): Promise<Belief>;
  supersede(id: string, newBelief: CreateBeliefInput): Promise<Belief>;
  
  // Delete
  delete(id: string, reason?: string): Promise<void>;
  
  // Relationships
  findSimilar(id: string, options?: SimilarOptions): Promise<RankedBelief[]>;
  findContradictions(id: string, options?: ContradictionOptions): Promise<ContradictingBelief[]>;
  getDerivationChain(id: string, maxDepth?: number): Promise<DerivationChain>;
  getVersionHistory(id: string): Promise<Belief[]>;
  
  // Subscriptions
  subscribe(query: string, options?: SubscribeOptions): Subscription<BeliefEvent>;
  
  // Verification shortcuts
  verify(id: string): Promise<VerificationStatus>;
  submitVerification(id: string, verification: VerificationInput): Promise<Verification>;
}

// Usage examples
const beliefs = client.beliefs;

// Create with full options
const belief = await beliefs.create({
  content: 'React 18 introduced concurrent features',
  confidence: {
    sourceReliability: 0.95,
    methodQuality: 0.9,
    temporalFreshness: 0.85
  },
  domains: ['tech/react', 'tech/javascript'],
  visibility: 'public',
  derivation: {
    type: 'observation',
    sources: [
      { externalRef: 'https://react.dev/blog/2022/03/29/react-v18' }
    ],
    methodDescription: 'Read from official React blog'
  },
  validUntil: new Date('2025-12-31')
});

// Query with streaming for large results
for await (const result of beliefs.queryStream('frontend frameworks', { limit: 100 })) {
  console.log(result.belief.content);
}

// Subscribe to updates
const subscription = beliefs.subscribe('AI safety research', {
  minConfidence: 0.7,
  scope: 'federated',
  debounceMs: 5000
});

subscription.on('added', (belief) => {
  console.log('New belief:', belief.content);
});

subscription.on('updated', (belief, changes) => {
  console.log('Belief updated:', changes);
});

// Later: cleanup
subscription.unsubscribe();
```

### TrustGraph

```typescript
interface TrustGraph {
  // Direct trust
  set(targetDid: string, trust: TrustInput): Promise<TrustEdge>;
  get(targetDid: string, domain?: string): Promise<TrustResult>;
  remove(targetDid: string): Promise<void>;
  
  // Querying
  getTrusted(options?: GetTrustedOptions): Promise<TrustedAgent[]>;
  getStats(): Promise<TrustGraphStats>;
  
  // Transitive
  computeTransitive(targetDid: string, options?: TransitiveOptions): Promise<TransitiveTrustResult>;
  getPaths(targetDid: string, options?: PathOptions): Promise<TrustPath[]>;
  
  // Configuration
  getConfig(): Promise<TrustConfig>;
  updateConfig(changes: Partial<TrustConfig>): Promise<TrustConfig>;
  
  // Maintenance
  pruneDecayed(threshold?: number, dryRun?: boolean): Promise<PruneResult>;
  refresh(targetDid: string): Promise<TrustEdge>;
  
  // Events
  on(event: 'changed', callback: (change: TrustChange) => void): void;
}

// Usage
const trust = client.trust;

// Set trust with domain-specific overrides
await trust.set('did:valence:z6MkABC...', {
  level: 0.75,
  domains: {
    'tech/ai': 0.9,
    'finance': 0.3
  },
  basis: [
    { type: 'directInteraction', confidence: 0.8 }
  ],
  notes: 'Met at conference, great AI researcher'
});

// Get trusted agents in a domain
const aiExperts = await trust.getTrusted({
  minLevel: 0.7,
  domain: 'tech/ai',
  includeTransitive: true,
  maxHops: 2
});

// Compute transitive trust with explanation
const transitive = await trust.computeTransitive('did:valence:z6MkXYZ...', {
  domain: 'tech/ai',
  maxHops: 3,
  explain: true
});

console.log(`Trust: ${transitive.level}`);
console.log(`Path: ${transitive.bestPath.join(' → ')}`);
```

### Federation

```typescript
interface Federation {
  // Discovery
  list(options?: ListFederationsOptions): Promise<FederationInfo[]>;
  get(id: string, options?: GetFederationOptions): Promise<FederationDetails>;
  search(query: string, options?: SearchOptions): Promise<FederationInfo[]>;
  
  // Membership
  join(id: string, options?: JoinOptions): Promise<JoinResult>;
  leave(id: string, options?: LeaveOptions): Promise<void>;
  getMyMemberships(): Promise<Membership[]>;
  
  // Creation & Management (if authorized)
  create(config: CreateFederationConfig): Promise<Federation>;
  update(id: string, changes: UpdateFederationConfig): Promise<Federation>;
  invite(id: string, options: InviteOptions): Promise<Invite>;
  
  // Sharing
  share(federationId: string, beliefId: string, options?: ShareOptions): Promise<SharedBelief>;
  unshare(federationId: string, shareId: string): Promise<void>;
  
  // Querying
  query(federationId: string, query: string, options?: FederationQueryOptions): Promise<AggregatedBelief[]>;
  getAggregates(federationId: string, topic: string): Promise<AggregatedBeliefResult>;
  
  // Governance
  createProposal(federationId: string, proposal: ProposalConfig): Promise<Proposal>;
  vote(federationId: string, proposalId: string, vote: Vote): Promise<VoteResult>;
  getProposals(federationId: string, options?: ProposalFilter): Promise<Proposal[]>;
  
  // Events
  subscribe(federationId: string): Subscription<FederationEvent>;
}

// Usage
const federations = client.federations;

// Find and join a federation
const aiSafetyFeds = await federations.search('AI safety research');
const joinResult = await federations.join(aiSafetyFeds[0].id, {
  applicationMessage: 'I research interpretability at Anthropic'
});

if (joinResult.status === 'pending_approval') {
  console.log(`Application pending. ${joinResult.pendingInfo.requiredApprovals} approvals needed.`);
}

// Share a belief anonymously
await federations.share(federationId, beliefId, {
  anonymous: true,
  context: 'Based on direct experimental results'
});

// Query aggregated federation knowledge
const knowledge = await federations.query(federationId, 'RLHF limitations', {
  minAgreement: 0.7,
  minContributors: 5
});
```

### Identity

```typescript
interface Identity {
  // Generation
  generate(options?: GenerateOptions): Promise<IdentityBundle>;
  fromMnemonic(mnemonic: string, passphrase?: string): Promise<IdentityBundle>;
  
  // Current identity
  getCurrent(): AgentIdentity;
  getReputation(domain?: string): Promise<ReputationScore>;
  updateMetadata(metadata: IdentityMetadata): Promise<AgentIdentity>;
  
  // Key management
  rotateKeys(reason?: string): Promise<KeyRotationResult>;
  exportIdentity(password: string): Promise<IdentityExport>;
  importIdentity(exported: IdentityExport, password: string): Promise<IdentityBundle>;
  
  // Resolution
  resolve(did: string): Promise<AgentIdentity | null>;
  
  // Proofs
  createProof(challenge: Uint8Array): Promise<IdentityProof>;
  verifyProof(proof: IdentityProof, publicKey: Uint8Array): Promise<boolean>;
}

// Usage
const identity = client.identity;

// Generate new identity
const { identity: newId, privateKeys, mnemonic } = await client.identity.generate({
  generateMnemonic: true,
  metadata: { displayName: 'My Research Agent' }
});

console.log('Backup mnemonic:', mnemonic);

// Export for backup
const exported = await identity.exportIdentity('secure-password-123');
// Store exported safely...

// Check reputation
const rep = await identity.getReputation('tech/ai');
console.log(`AI reputation: ${rep.byDomain['tech/ai']}`);
```

### Verification

```typescript
interface Verification {
  // Submit
  submit(beliefId: string, input: VerificationInput): Promise<VerificationSubmission>;
  submitBatch(verifications: BatchVerificationInput[]): Promise<BatchVerificationResult>;
  
  // Query
  get(verificationId: string): Promise<Verification>;
  forBelief(beliefId: string, options?: VerificationFilter): Promise<VerificationList>;
  myHistory(options?: HistoryOptions): Promise<VerifierHistory>;
  
  // Disputes
  dispute(verificationId: string, input: DisputeInput): Promise<DisputeSubmission>;
  getDispute(disputeId: string): Promise<Dispute>;
  
  // Stakes
  getStakeBalance(): Promise<StakeBalance>;
  withdrawStake(verificationId: string): Promise<StakeWithdrawal>;
  
  // Events
  subscribe(options: VerificationSubscribeOptions): Subscription<VerificationEvent>;
}

// Usage
const verification = client.verification;

// Submit verification with evidence
const submission = await verification.submit(beliefId, {
  result: 'confirmed',
  evidence: [
    {
      type: 'external',
      externalSource: {
        url: 'https://arxiv.org/abs/2106.09685',
        archiveHash: 'sha256:...'
      },
      relevance: 0.95,
      contribution: 'supports'
    }
  ],
  stake: 0.05,
  reasoning: 'Verified against original paper'
});

// Check stake balance
const balance = await verification.getStakeBalance();
console.log(`Available to stake: ${balance.availableReputation}`);
console.log(`Currently staked: ${balance.totalStaked}`);
```

### Offline Support & Sync

```typescript
interface SyncEngine {
  // Status
  getStatus(): SyncStatus;
  isOnline(): boolean;
  
  // Manual sync
  syncNow(): Promise<SyncResult>;
  
  // Queue management
  getQueue(): QueuedOperation[];
  clearQueue(): void;
  retryFailed(): Promise<void>;
  
  // Conflict resolution
  getConflicts(): Conflict[];
  resolveConflict(conflictId: string, resolution: ConflictResolution): Promise<void>;
  
  // Events
  on(event: 'statusChange', callback: (status: SyncStatus) => void): void;
  on(event: 'syncComplete', callback: (result: SyncResult) => void): void;
  on(event: 'conflict', callback: (conflict: Conflict) => void): void;
}

// Initialize with offline support
const client = new ValenceClient({
  apiKey: 'val_sk_...',
  offline: {
    enabled: true,
    storage: 'indexeddb',
    syncInterval: 60000  // Sync every minute
  }
});

// Operations work offline, queued for sync
await client.beliefs.create({
  content: 'This works even without internet',
  domains: ['testing']
});

// Check sync status
const status = client.sync.getStatus();
console.log(`Online: ${status.isOnline}`);
console.log(`Queued operations: ${status.queueLength}`);
console.log(`Last sync: ${status.lastSyncAt}`);

// Handle conflicts
client.sync.on('conflict', async (conflict) => {
  console.log('Conflict detected:', conflict);
  
  // Auto-resolve: server wins
  await client.sync.resolveConflict(conflict.id, {
    strategy: 'server_wins'
  });
  
  // Or: custom resolution
  await client.sync.resolveConflict(conflict.id, {
    strategy: 'custom',
    mergedData: { /* custom merge */ }
  });
});
```

### Type Definitions

```typescript
// Core types
interface Belief {
  id: string;
  version: number;
  content: string;
  contentHash: string;
  confidence: ConfidenceVector;
  confidenceOverall: number;
  validFrom: Date;
  validUntil: Date | null;
  derivation: Derivation;
  domains: string[];
  visibility: 'private' | 'federated' | 'public';
  holderId: string;
  createdAt: Date;
  supersedes: string | null;
  supersededBy: string | null;
}

interface ConfidenceVector {
  sourceReliability: number | null;
  methodQuality: number | null;
  internalConsistency: number | null;
  temporalFreshness: number | null;
  corroboration: number | null;
  domainApplicability: number | null;
}

interface RankedBelief {
  belief: Belief;
  score: {
    final: number;
    components: {
      semanticSimilarity: number;
      confidenceScore: number;
      trustScore: number;
      recencyScore: number;
      diversityPenalty: number;
    };
  };
  explanation?: RankingExplanation;
}

interface QueryResult {
  beliefs: RankedBelief[];
  totalCount: number;
  nextCursor?: string;
  queryTimeMs: number;
  partial: boolean;
  coverage: QueryCoverage;
}

// Input types
interface CreateBeliefInput {
  content: string;
  confidence?: Partial<ConfidenceVector>;
  domains?: string[];
  visibility?: 'private' | 'federated' | 'public';
  validFrom?: Date;
  validUntil?: Date;
  derivation?: DerivationInput;
}

interface QueryOptions {
  minConfidence?: number | Partial<ConfidenceVector>;
  domains?: string[];
  domainsExclude?: string[];
  minTrust?: number;
  validAt?: Date;
  scope?: 'local' | 'federated' | 'network';
  limit?: number;
  cursor?: string;
  includeExplanations?: boolean;
  diversity?: DiversityConfig;
}

// Event types
type BeliefEvent = 
  | { type: 'created'; belief: Belief }
  | { type: 'updated'; belief: Belief; changes: string[] }
  | { type: 'superseded'; oldBelief: Belief; newBelief: Belief }
  | { type: 'deleted'; beliefId: string }
  | { type: 'verified'; belief: Belief; verification: Verification };
```

---

## Python SDK

### Installation

```bash
pip install valence-sdk
# or
poetry add valence-sdk
```

### Quick Start

```python
from valence import ValenceClient, Confidence, Derivation

# Initialize
client = ValenceClient(api_key="val_sk_...")

# Or with identity
client = ValenceClient(
    identity={
        "did": "did:valence:z6Mk...",
        "private_key": private_key_bytes
    }
)

# Query beliefs
results = await client.beliefs.query(
    "machine learning best practices",
    min_confidence=0.7,
    domains=["tech/ai"],
    limit=10
)

for result in results:
    print(f"[{result.score.final:.2f}] {result.belief.content}")

# Store a belief
belief = await client.beliefs.create(
    content="Python 3.12 introduced the new 'type' statement",
    confidence=Confidence(source_reliability=0.95, method_quality=0.9),
    domains=["tech/python"],
    derivation=Derivation(
        type="observation",
        sources=[{"external_ref": "https://docs.python.org/3.12/whatsnew/"}]
    )
)
```

### Async & Sync Interfaces

```python
# Async (recommended)
from valence import ValenceClient

async def main():
    client = ValenceClient(api_key="val_sk_...")
    results = await client.beliefs.query("AI safety")
    
# Sync wrapper for simple scripts
from valence import SyncValenceClient

client = SyncValenceClient(api_key="val_sk_...")
results = client.beliefs.query("AI safety")  # Blocking call
```

### BeliefStore

```python
from valence import ValenceClient, Confidence, Derivation, Visibility
from datetime import datetime, timedelta

client = ValenceClient(api_key="val_sk_...")
beliefs = client.beliefs

# Create with full options
belief = await beliefs.create(
    content="FastAPI is built on Starlette and Pydantic",
    confidence=Confidence(
        source_reliability=0.95,
        method_quality=0.9,
        temporal_freshness=0.85
    ),
    domains=["tech/python", "tech/web"],
    visibility=Visibility.PUBLIC,
    derivation=Derivation(
        type="observation",
        sources=[
            {"external_ref": "https://fastapi.tiangolo.com/"}
        ],
        method_description="Read from official docs"
    ),
    valid_until=datetime.now() + timedelta(days=365)
)

# Query with filters
results = await beliefs.query(
    "Python web frameworks",
    min_confidence=0.6,
    domains=["tech/python"],
    scope="federated",
    limit=20
)

# Stream large results
async for result in beliefs.query_stream("frontend frameworks", limit=100):
    print(result.belief.content)

# Find contradictions
contradictions = await beliefs.find_contradictions(belief_id)
for c in contradictions:
    print(f"[{c.contradiction_type}] {c.explanation}")

# Subscribe to updates
async with beliefs.subscribe("AI safety research") as subscription:
    async for event in subscription:
        if event.type == "added":
            print(f"New belief: {event.belief.content}")
```

### TrustGraph

```python
trust = client.trust

# Set trust with domain overrides
await trust.set(
    target_did="did:valence:z6MkABC...",
    level=0.75,
    domains={
        "tech/ai": 0.9,
        "finance": 0.3
    },
    basis=[
        {"type": "direct_interaction", "confidence": 0.8}
    ],
    notes="Great AI researcher"
)

# Get trusted agents
ai_experts = await trust.get_trusted(
    min_level=0.7,
    domain="tech/ai",
    include_transitive=True,
    max_hops=2
)

# Compute transitive trust with explanation
result = await trust.compute_transitive(
    target_did="did:valence:z6MkXYZ...",
    domain="tech/ai",
    max_hops=3,
    explain=True
)

print(f"Trust: {result.level}")
print(f"Path: {' → '.join(result.best_path)}")
```

### Federation

```python
federations = client.federations

# Search and join
ai_safety_feds = await federations.search("AI safety research")
result = await federations.join(
    ai_safety_feds[0].id,
    application_message="I research interpretability"
)

if result.status == "pending_approval":
    print(f"Pending. Need {result.pending_info.required_approvals} approvals.")

# Share anonymously
await federations.share(
    federation_id,
    belief_id,
    anonymous=True,
    context="Based on experiments"
)

# Query aggregated knowledge
knowledge = await federations.query(
    federation_id,
    "RLHF limitations",
    min_agreement=0.7,
    min_contributors=5
)

for k in knowledge:
    print(f"[{k.agreement_score:.2f}] {k.content_summary}")
```

### Context Managers & Cleanup

```python
# Recommended: use as context manager
async with ValenceClient(api_key="val_sk_...") as client:
    results = await client.beliefs.query("test")
    # Client automatically cleaned up

# Or manual cleanup
client = ValenceClient(api_key="val_sk_...")
try:
    results = await client.beliefs.query("test")
finally:
    await client.close()
```

### Type Hints

```python
from valence.types import (
    Belief,
    ConfidenceVector,
    RankedBelief,
    QueryResult,
    TrustEdge,
    AgentIdentity,
    Verification,
    Federation,
    AggregatedBelief
)

async def process_beliefs(results: QueryResult) -> list[str]:
    return [r.belief.content for r in results.beliefs if r.score.final > 0.8]
```

### Offline Support

```python
from valence import ValenceClient
from valence.storage import IndexedDBStorage, SQLiteStorage

# Enable offline with SQLite (good for desktop apps)
client = ValenceClient(
    api_key="val_sk_...",
    offline=True,
    storage=SQLiteStorage("~/.valence/cache.db"),
    sync_interval=60
)

# Operations work offline
await client.beliefs.create(content="This works offline too")

# Check sync status
status = client.sync.get_status()
print(f"Online: {status.is_online}")
print(f"Queued: {status.queue_length}")

# Manual sync
result = await client.sync.sync_now()
print(f"Synced: {result.uploaded} up, {result.downloaded} down")

# Handle conflicts
@client.sync.on_conflict
async def handle_conflict(conflict):
    print(f"Conflict: {conflict}")
    await client.sync.resolve_conflict(
        conflict.id,
        strategy="server_wins"
    )
```

### Error Handling

```python
from valence.exceptions import (
    ValenceError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    InsufficientReputationError
)

try:
    await client.beliefs.create(content="x" * 100000)  # Too long
except ValidationError as e:
    print(f"Validation failed: {e.field} - {e.message}")
    print(f"Details: {e.details}")

try:
    await client.verification.submit(belief_id, stake=0.9)  # Too high
except InsufficientReputationError as e:
    print(f"Need {e.required} reputation, have {e.available}")

try:
    await client.beliefs.query("test")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
    await asyncio.sleep(e.retry_after)
```

---

## Authentication Patterns

### API Keys

Simplest approach for server-to-server and personal use.

```typescript
// TypeScript
const client = new ValenceClient({ apiKey: 'val_sk_...' });

// Python
client = ValenceClient(api_key="val_sk_...")
```

**Key Management:**
```typescript
// Create new key
const key = await client.auth.createKey({
  name: 'Production API Key',
  scopes: ['beliefs:read', 'beliefs:write', 'trust:read']
});

// List keys
const keys = await client.auth.listKeys();

// Revoke key
await client.auth.revokeKey(keyId);

// Rotate key
const newKey = await client.auth.rotateKey(keyId);
```

### DID Authentication

Recommended for AI agents. Identity is portable and self-sovereign.

```typescript
// Generate new identity
const { identity, privateKeys, mnemonic } = await client.identity.generate({
  generateMnemonic: true
});

// Save mnemonic securely!
console.log('BACKUP THIS:', mnemonic);

// Initialize client with identity
const client = new ValenceClient({
  identity: {
    did: identity.id,
    privateKey: privateKeys.identityKey
  }
});

// Restore from mnemonic
const restored = await client.identity.fromMnemonic(mnemonic);
const client2 = new ValenceClient({
  identity: {
    did: restored.identity.id,
    privateKey: restored.privateKeys.identityKey
  }
});
```

### OAuth 2.0

For third-party applications acting on behalf of users.

```typescript
// Authorization URL
const authUrl = client.auth.getAuthorizationUrl({
  clientId: 'your-client-id',
  redirectUri: 'https://your-app.com/callback',
  scopes: ['beliefs:read', 'beliefs:write'],
  state: randomState
});

// Exchange code for tokens
const tokens = await client.auth.exchangeCode({
  code: authorizationCode,
  redirectUri: 'https://your-app.com/callback'
});

// Initialize with tokens
const client = new ValenceClient({
  oauth: {
    clientId: 'your-client-id',
    clientSecret: 'your-client-secret',
    accessToken: tokens.accessToken,
    refreshToken: tokens.refreshToken
  }
});

// Tokens auto-refresh, or manually:
const newTokens = await client.auth.refreshTokens();
```

---

## Offline-First Architecture

### Storage Adapters

```typescript
// TypeScript
import { IndexedDBStorage, LocalStorageAdapter, MemoryStorage } from '@valence/sdk/storage';

// Browser: IndexedDB (recommended)
const client = new ValenceClient({
  offline: { storage: 'indexeddb' }
});

// Browser fallback: LocalStorage
const client = new ValenceClient({
  offline: { storage: 'localstorage' }
});

// Node.js: SQLite
import { SQLiteStorage } from '@valence/sdk-node';
const client = new ValenceClient({
  offline: { storage: new SQLiteStorage('./valence-cache.db') }
});

// Testing: Memory
const client = new ValenceClient({
  offline: { storage: 'memory' }
});
```

### Sync Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                     Offline Operations                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Local Write → Queue → Periodic Sync → Server               │
│       ↓                    ↑                                │
│  Immediate return      Background                           │
│                                                             │
│  Local Read → Cache → (if miss) → Server → Cache            │
│       ↓                              ↓                      │
│  Immediate return           Update local                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Conflict Resolution

```typescript
// Register conflict handler
client.sync.on('conflict', async (conflict) => {
  // conflict.local: Your local version
  // conflict.server: Server version
  // conflict.base: Last synced version (if available)
  
  switch (conflict.type) {
    case 'belief_update':
      // Server wins by default for beliefs
      return { strategy: 'server_wins' };
      
    case 'trust_update':
      // Local wins for trust (your preferences)
      return { strategy: 'local_wins' };
      
    case 'both_modified':
      // Custom merge
      return {
        strategy: 'custom',
        mergedData: mergeConfidenceVectors(conflict.local, conflict.server)
      };
  }
});

// Or set global strategy
const client = new ValenceClient({
  offline: {
    conflictStrategy: 'server_wins' | 'local_wins' | 'manual'
  }
});
```

---

## Framework Integrations

### React Hook

```typescript
// @valence/react
import { useValence, useBeliefs, useTrust, useFederation } from '@valence/react';

function App() {
  return (
    <ValenceProvider client={client}>
      <BeliefSearch />
    </ValenceProvider>
  );
}

function BeliefSearch() {
  const { query, loading, error, results } = useBeliefs('AI safety', {
    minConfidence: 0.7,
    scope: 'federated'
  });
  
  if (loading) return <Spinner />;
  if (error) return <Error error={error} />;
  
  return (
    <ul>
      {results.map(r => (
        <li key={r.belief.id}>
          [{r.score.final.toFixed(2)}] {r.belief.content}
        </li>
      ))}
    </ul>
  );
}
```

### Vue Composable

```typescript
// @valence/vue
import { useValence, useBeliefs } from '@valence/vue';

export default {
  setup() {
    const { results, loading, error, refetch } = useBeliefs('AI safety', {
      minConfidence: 0.7
    });
    
    return { results, loading, error, refetch };
  }
};
```

### FastAPI Integration

```python
# valence-fastapi
from valence.fastapi import ValenceMiddleware, get_valence_client

app = FastAPI()
app.add_middleware(ValenceMiddleware, api_key="val_sk_...")

@app.get("/search")
async def search(
    query: str,
    client: ValenceClient = Depends(get_valence_client)
):
    results = await client.beliefs.query(query)
    return results
```

### Django Integration

```python
# valence-django
# settings.py
VALENCE_API_KEY = "val_sk_..."

# views.py
from valence.django import valence_client

async def search(request):
    client = valence_client()
    results = await client.beliefs.query(request.GET['q'])
    return JsonResponse({"results": results})
```

---

## Best Practices

### 1. Connection Management

```typescript
// ✅ Reuse client instances
const client = new ValenceClient({ apiKey: '...' });
// Use `client` throughout your application

// ❌ Don't create new clients per request
app.get('/search', async (req, res) => {
  const client = new ValenceClient({ apiKey: '...' });  // Bad!
  // ...
});
```

### 2. Error Handling

```typescript
// ✅ Handle specific errors
try {
  await client.beliefs.create(input);
} catch (error) {
  if (error instanceof RateLimitError) {
    await delay(error.retryAfter);
    return retry();
  }
  if (error instanceof ValidationError) {
    return { error: error.message, field: error.field };
  }
  throw error;  // Re-throw unknown errors
}
```

### 3. Pagination

```typescript
// ✅ Use cursor pagination for large result sets
let cursor: string | undefined;
const allResults: Belief[] = [];

do {
  const result = await client.beliefs.query('topic', { cursor, limit: 100 });
  allResults.push(...result.beliefs.map(r => r.belief));
  cursor = result.nextCursor;
} while (cursor);

// ✅ Or use streaming
for await (const result of client.beliefs.queryStream('topic')) {
  process(result);
}
```

### 4. Confidence Scores

```typescript
// ✅ Be conservative with confidence
await client.beliefs.create({
  content: 'User mentioned they prefer dark mode',
  confidence: {
    sourceReliability: 0.9,   // Direct from user
    methodQuality: 0.95,      // Direct statement
    temporalFreshness: 1.0,   // Just now
    corroboration: 0.2,       // Only one source
    domainApplicability: 0.7  // Specific to this context
  }
});

// ❌ Don't inflate confidence
await client.beliefs.create({
  content: 'This might be true',
  confidence: { sourceReliability: 0.99 }  // Overconfident!
});
```

### 5. Derivation

```typescript
// ✅ Always include derivation when possible
await client.beliefs.create({
  content: 'LoRA reduces memory by 90%',
  derivation: {
    type: 'inference',
    sources: [
      { beliefId: 'source-belief-1' },
      { externalRef: 'https://arxiv.org/...' }
    ],
    methodDescription: 'Compared benchmark results from paper'
  }
});
```

---

## Package Structure

### TypeScript

```
@valence/sdk/
├── index.ts                 # Main exports
├── client.ts               # ValenceClient
├── beliefs/
│   ├── store.ts           # BeliefStore
│   ├── types.ts           # Belief types
│   └── queries.ts         # Query builders
├── trust/
│   ├── graph.ts           # TrustGraph
│   └── types.ts           # Trust types
├── federation/
│   ├── client.ts          # Federation client
│   └── types.ts           # Federation types
├── identity/
│   ├── manager.ts         # Identity management
│   ├── crypto.ts          # Crypto operations
│   └── types.ts           # Identity types
├── verification/
│   ├── client.ts          # Verification client
│   └── types.ts           # Verification types
├── sync/
│   ├── engine.ts          # Sync engine
│   ├── storage/           # Storage adapters
│   │   ├── indexeddb.ts
│   │   ├── localstorage.ts
│   │   └── memory.ts
│   └── conflicts.ts       # Conflict resolution
├── auth/
│   ├── api-key.ts         # API key auth
│   ├── did-auth.ts        # DID auth
│   └── oauth.ts           # OAuth auth
└── utils/
    ├── errors.ts          # Error classes
    ├── retry.ts           # Retry logic
    └── rate-limit.ts      # Rate limiting
```

### Python

```
valence/
├── __init__.py            # Main exports
├── client.py              # ValenceClient
├── beliefs/
│   ├── __init__.py
│   ├── store.py          # BeliefStore
│   └── types.py          # Belief types
├── trust/
│   ├── __init__.py
│   ├── graph.py          # TrustGraph
│   └── types.py          # Trust types
├── federation/
│   ├── __init__.py
│   ├── client.py         # Federation client
│   └── types.py          # Federation types
├── identity/
│   ├── __init__.py
│   ├── manager.py        # Identity management
│   ├── crypto.py         # Crypto operations
│   └── types.py          # Identity types
├── verification/
│   ├── __init__.py
│   ├── client.py         # Verification client
│   └── types.py          # Verification types
├── sync/
│   ├── __init__.py
│   ├── engine.py         # Sync engine
│   ├── storage/          # Storage adapters
│   │   ├── sqlite.py
│   │   └── memory.py
│   └── conflicts.py      # Conflict resolution
├── auth/
│   ├── __init__.py
│   ├── api_key.py        # API key auth
│   ├── did_auth.py       # DID auth
│   └── oauth.py          # OAuth auth
├── exceptions.py         # Exception classes
└── types.py              # Shared types
```

---

*"Make the easy things easy and the hard things possible."*
