# Valence Federation Privacy Guarantees

*How privacy is protected in federated knowledge sharing*

This document specifies the privacy mechanisms and guarantees for Valence federation, ensuring user sovereignty over their knowledge while enabling collective intelligence.

---

## 1. Privacy Philosophy

### 1.1 Guiding Principles

From [PRINCIPLES.md](./PRINCIPLES.md):

- **User Sovereignty**: Users own their data. Data never leaves without explicit consent.
- **Structurally Incapable of Betrayal**: Architecture, not promises. Privacy enforced by design.
- **Aggregation Serves Users**: Privacy-preserving by necessity. If aggregation exposes users, it fails.

### 1.2 Privacy-Utility Tradeoff

```
Maximum Privacy                                    Maximum Utility
      ↓                                                  ↓
┌─────────────────────────────────────────────────────────────┐
│ Private    │    Trusted    │   Federated   │    Public    │
│            │               │               │              │
│ Never      │ Explicit      │ DP-protected  │ Attribution  │
│ shared     │ sharing       │ aggregation   │ allowed      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Visibility Levels

### 2.1 Overview

| Level | Description | Use Case | Privacy Level |
|-------|-------------|----------|---------------|
| `private` | Never leaves local node | Personal notes, sensitive info | Maximum |
| `trusted` | Shared with explicit trust | Team knowledge, semi-private | High |
| `federated` | Shared across network | General knowledge | Medium |
| `public` | Discoverable by anyone | Published research | Low |

### 2.2 Visibility Semantics

**Private**
```python
class PrivateVisibility:
    """Beliefs that never leave the local node."""

    # Cannot be:
    # - Included in sync to any node
    # - Part of any aggregation query response
    # - Visible via federation API

    # Can be:
    # - Queried locally via MCP
    # - Used for local inference
    # - Superseded/archived locally
```

**Trusted**
```python
class TrustedVisibility:
    """Beliefs shared only with explicitly trusted nodes."""

    def can_share_with(self, node: FederationNode) -> bool:
        # Requires explicit trust relationship
        return (
            node.trust_phase >= TrustPhase.PARTICIPANT and
            self.owner.has_trusted(node) and
            node.id in self.explicit_share_list
        )
```

**Federated**
```python
class FederatedVisibility:
    """Beliefs shared across federation network."""

    def can_share_with(self, node: FederationNode) -> bool:
        # Trust threshold check
        return node.trust.overall >= 0.2  # Minimum contributor trust

    def aggregation_rules(self) -> AggregationRules:
        return AggregationRules(
            requires_differential_privacy=True,
            minimum_k_anonymity=5,  # At least 5 contributors
            maximum_hop_count=3     # Limit propagation
        )
```

**Public**
```python
class PublicVisibility:
    """Beliefs discoverable by anyone."""

    # Can be:
    # - Indexed by search engines
    # - Attributed to user (if share_level=full)
    # - Accessed without authentication
    # - Cached by any node
```

### 2.3 Default Visibility

```python
# System defaults (user can override)
DEFAULT_VISIBILITY = {
    "user_created": "private",           # Manual beliefs start private
    "extracted_from_conversation": "private",
    "extracted_from_document": "trusted",
    "inferred": "federated",
    "corroborated": "federated",
}
```

---

## 3. Share Levels

### 3.1 What Gets Shared

| Level | Content | Confidence | Source | Attribution | Metadata |
|-------|---------|------------|--------|-------------|----------|
| `belief_only` | ✓ | ✓ | ✗ | ✗ | ✗ |
| `with_provenance` | ✓ | ✓ | ✓ | ✗ | Partial |
| `full` | ✓ | ✓ | ✓ | ✓ | ✓ |

### 3.2 Share Level Details

**belief_only**
```json
{
  "content": "PostgreSQL JSONB supports efficient nested queries",
  "confidence": {"overall": 0.85},
  "domain_path": ["tech", "databases", "postgresql"]
}
```

**with_provenance**
```json
{
  "content": "PostgreSQL JSONB supports efficient nested queries",
  "confidence": {"overall": 0.85, "source_reliability": 0.9},
  "domain_path": ["tech", "databases", "postgresql"],
  "source": {
    "type": "document",
    "title": "PostgreSQL Documentation",
    "url": "https://www.postgresql.org/docs/current/datatype-json.html"
  }
}
```

**full**
```json
{
  "content": "PostgreSQL JSONB supports efficient nested queries",
  "confidence": {"overall": 0.85, "source_reliability": 0.9, "method_quality": 0.8},
  "domain_path": ["tech", "databases", "postgresql"],
  "source": {
    "type": "document",
    "title": "PostgreSQL Documentation",
    "url": "https://www.postgresql.org/docs/current/datatype-json.html"
  },
  "attribution": {
    "user_did": "did:vkb:user:web:example.com:alice",
    "created_at": "2025-01-15T10:30:00Z"
  },
  "metadata": {
    "extraction_method": "manual",
    "session_context": "database optimization research"
  }
}
```

---

## 4. Differential Privacy

### 4.1 Mechanism

Valence uses differential privacy to protect individual contributions to aggregation queries.

**Definition**: A mechanism M satisfies (ε, δ)-differential privacy if for any two neighboring datasets D and D' (differing in one record) and any output set S:

```
P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S] + δ
```

### 4.2 Privacy Parameters

```python
class PrivacyParameters:
    """Privacy parameters for aggregation queries."""

    # Epsilon: Privacy loss per query
    # Lower = more privacy, less utility
    epsilon: float = 0.1  # Default: strong privacy

    # Delta: Probability of complete privacy failure
    # Should be cryptographically small
    delta: float = 1e-6

    # Minimum contributors for any aggregate
    # Prevents small-n exposure
    min_contributors: int = 5

    # Maximum queries per privacy budget period
    max_queries_per_period: int = 100

    # Budget period (resets epsilon accumulation)
    budget_period: timedelta = timedelta(days=1)
```

### 4.3 Noise Injection

```python
def compute_local_summary(
    beliefs: list[Belief],
    query: AggregationQuery,
    privacy_params: PrivacyParameters
) -> LocalSummary:
    """Compute privacy-preserving local summary."""

    # Compute true statistics
    true_stats = {
        "belief_count": len(beliefs),
        "mean_confidence": mean([b.confidence.overall for b in beliefs]),
        "stance_vector": compute_stance_embedding(beliefs),
    }

    # Calculate sensitivity (how much one belief can change output)
    sensitivity = 1.0 / max(1, len(beliefs))

    # Laplace noise scale from privacy parameters
    noise_scale = sensitivity / privacy_params.epsilon

    # Add calibrated Laplace noise
    noisy_stats = {}
    for key, value in true_stats.items():
        if isinstance(value, (int, float)):
            noise = np.random.laplace(0, noise_scale)
            noisy_stats[key] = max(0, value + noise)  # Clip to non-negative
        elif isinstance(value, list):
            # Vector: add noise to each component
            noisy_stats[key] = [
                v + np.random.laplace(0, noise_scale)
                for v in value
            ]

    return LocalSummary(
        stats=noisy_stats,
        privacy_budget_used=privacy_params.epsilon,
        contributor_count=len(beliefs) if len(beliefs) >= privacy_params.min_contributors else 0
    )
```

### 4.4 Privacy Budget

Each node maintains a privacy budget that limits total information leakage:

```python
class PrivacyBudget:
    """Track and enforce privacy budget."""

    def __init__(self, daily_epsilon: float = 1.0):
        self.daily_epsilon = daily_epsilon
        self.used_epsilon = 0.0
        self.period_start = datetime.now()
        self.query_log: list[QueryRecord] = []

    def can_respond(self, query_epsilon: float) -> bool:
        """Check if budget allows responding to query."""
        self._maybe_reset_period()
        return self.used_epsilon + query_epsilon <= self.daily_epsilon

    def consume(self, query_epsilon: float, query: AggregationQuery):
        """Consume budget for a query."""
        if not self.can_respond(query_epsilon):
            raise PrivacyBudgetExhausted()

        self.used_epsilon += query_epsilon
        self.query_log.append(QueryRecord(
            query=query,
            epsilon=query_epsilon,
            timestamp=datetime.now()
        ))

    def _maybe_reset_period(self):
        """Reset budget if period has elapsed."""
        if datetime.now() - self.period_start > timedelta(days=1):
            self.used_epsilon = 0.0
            self.period_start = datetime.now()
            self.query_log = []

    def remaining(self) -> float:
        """Get remaining privacy budget."""
        self._maybe_reset_period()
        return max(0, self.daily_epsilon - self.used_epsilon)
```

---

## 5. Aggregation Privacy

### 5.1 Secure Aggregation Protocol

```
1. Query Broadcast
   ┌────────────┐         ┌────────────┐
   │ Aggregator │ ──────► │   Node A   │
   │            │         └────────────┘
   │            │         ┌────────────┐
   │            │ ──────► │   Node B   │
   │            │         └────────────┘
   │            │         ┌────────────┐
   │            │ ──────► │   Node C   │
   └────────────┘         └────────────┘

2. Local Computation (each node)
   - Find relevant beliefs
   - Compute statistics
   - Add DP noise
   - Sign summary

3. Secure Collection
   ┌────────────┐         ┌────────────┐
   │   Node A   │ ──┐     │            │
   └────────────┘   │     │            │
   ┌────────────┐   ├───► │ Aggregator │
   │   Node B   │ ──┤     │            │
   └────────────┘   │     │            │
   ┌────────────┐   │     │            │
   │   Node C   │ ──┘     └────────────┘
   └────────────┘

4. Aggregate Computation
   - Combine noisy summaries (trust-weighted)
   - Verify minimum contributor threshold
   - Compute collective confidence
   - Publish aggregate ONLY (not individual summaries)
```

### 5.2 Minimum Contributor Threshold

Aggregation results are only published if enough nodes contribute:

```python
def finalize_aggregation(
    summaries: list[LocalSummary],
    privacy_params: PrivacyParameters
) -> AggregationResult | None:
    """Finalize aggregation with privacy checks."""

    # Check minimum contributors
    valid_summaries = [s for s in summaries if s.contributor_count > 0]

    if len(valid_summaries) < privacy_params.min_contributors:
        # Not enough contributors - abort to protect privacy
        return None

    # Compute weighted aggregate
    total_weight = sum(s.trust_weight for s in valid_summaries)

    result = AggregationResult(
        collective_confidence=sum(
            s.stats["mean_confidence"] * s.trust_weight
            for s in valid_summaries
        ) / total_weight,

        # Report counts, not identities
        contributor_count=sum(s.contributor_count for s in valid_summaries),
        node_count=len(valid_summaries),

        privacy_guarantees={
            "epsilon": privacy_params.epsilon,
            "delta": privacy_params.delta,
            "mechanism": "laplace",
            "min_contributors_satisfied": True
        }
    )

    return result
```

### 5.3 Query Rate Limiting

Prevent privacy erosion through repeated queries:

```python
class AggregationRateLimiter:
    """Rate limit aggregation queries to protect privacy."""

    def __init__(self):
        self.query_history: dict[str, list[datetime]] = {}

    def check_limit(self, query: AggregationQuery) -> bool:
        """Check if query is within rate limits."""
        query_hash = self.normalize_query_hash(query)

        # Same query: 1 per hour
        recent_same = self.recent_queries(query_hash, hours=1)
        if len(recent_same) >= 1:
            return False

        # Similar domain: 10 per hour
        domain_hash = hash(tuple(query.domain_filter))
        recent_domain = self.recent_queries(domain_hash, hours=1)
        if len(recent_domain) >= 10:
            return False

        # Global: 100 per day
        all_recent = self.recent_queries("*", hours=24)
        if len(all_recent) >= 100:
            return False

        return True
```

---

## 6. Cross-Node Tension Privacy

### 6.1 Stance Embedding Approach

Detect contradictions without revealing belief content:

```python
def detect_cross_node_tensions(
    query_domain: list[str],
    participating_nodes: list[FederationNode]
) -> list[TensionSignal]:
    """Detect tensions across nodes without exposing beliefs."""

    # Step 1: Each node computes stance embeddings
    stance_embeddings = {}
    for node in participating_nodes:
        # Node computes locally, shares only embedding
        embedding = node.compute_domain_stance_embedding(query_domain)
        stance_embeddings[node.did] = embedding

    # Step 2: Cluster similar stances (doesn't reveal content)
    clusters = cluster_embeddings(
        list(stance_embeddings.values()),
        similarity_threshold=0.8
    )

    # Step 3: Detect opposing clusters
    tensions = []
    for i, cluster_a in enumerate(clusters):
        for cluster_b in clusters[i+1:]:
            opposition = measure_stance_opposition(cluster_a, cluster_b)
            if opposition > 0.7:  # Strong opposition
                tensions.append(TensionSignal(
                    domain=query_domain,
                    opposition_strength=opposition,
                    # Report counts, not which nodes
                    cluster_a_size=len(cluster_a),
                    cluster_b_size=len(cluster_b)
                ))

    return tensions
```

### 6.2 Tension Investigation Protocol

When tensions are detected:

```python
class TensionInvestigation:
    """Privacy-preserving tension investigation."""

    def investigate(self, tension: TensionSignal) -> InvestigationResult:
        # Step 1: Ask participating nodes if they want to disclose
        disclosures = self.request_voluntary_disclosure(tension)

        # Only proceed if enough voluntary disclosures
        if len(disclosures) < 2:
            return InvestigationResult(
                status="insufficient_disclosure",
                recommendation="Monitor for future signals"
            )

        # Step 2: Compare disclosed beliefs
        # (Only those who opted in)
        comparison = self.compare_disclosed_beliefs(disclosures)

        return InvestigationResult(
            status="investigated",
            tension_type=comparison.detected_type,
            recommendation=comparison.resolution_suggestion
        )
```

---

## 7. GDPR Considerations

### 7.1 Right to Erasure

```python
class DataErasure:
    """Handle GDPR Article 17 - Right to Erasure."""

    def erase_user_data(self, user_did: str) -> ErasureReport:
        """Erase all user data from local and federated systems."""

        report = ErasureReport()

        # 1. Local erasure
        report.local = self.erase_local_beliefs(user_did)

        # 2. Notify federation (can't force, but request)
        for node in self.connected_nodes:
            try:
                response = node.request_erasure(user_did)
                report.federation[node.did] = response
            except Exception as e:
                report.federation[node.did] = ErasureResult(
                    status="request_failed",
                    error=str(e)
                )

        # 3. Aggregates: Cannot erase (DP guarantees plausible deniability)
        report.aggregates = "Not erasable - differential privacy provides plausible deniability"

        return report
```

### 7.2 Data Portability

```python
class DataPortability:
    """Handle GDPR Article 20 - Right to Data Portability."""

    def export_user_data(self, user_did: str) -> DataExport:
        """Export all user data in portable format."""

        return DataExport(
            format="json",
            beliefs=self.export_beliefs(user_did),
            sessions=self.export_sessions(user_did),
            patterns=self.export_patterns(user_did),
            trust_relationships=self.export_trust(user_did),
            metadata={
                "exported_at": datetime.now().isoformat(),
                "format_version": "1.0",
                "can_import_to": ["valence", "generic_json"]
            }
        )
```

### 7.3 Consent Management

```python
class ConsentManager:
    """Manage user consent for federation features."""

    def get_consent_state(self) -> ConsentState:
        return ConsentState(
            federation_enabled=self.prefs.federation_consent,
            aggregation_participation=self.prefs.aggregation_consent,
            public_beliefs=self.prefs.public_consent,
            default_visibility=self.prefs.default_visibility,
            share_level_limit=self.prefs.max_share_level
        )

    def update_consent(self, consent: ConsentUpdate) -> None:
        """Update consent settings."""
        self.prefs.update(consent)

        # Immediately apply: reduce visibility of existing beliefs if needed
        if consent.federation_consent == False:
            self.set_all_beliefs_private()

        if consent.max_share_level:
            self.reduce_share_levels(consent.max_share_level)
```

---

## 8. Privacy Guarantees Summary

### 8.1 What We Guarantee

| Guarantee | Mechanism | Strength |
|-----------|-----------|----------|
| Beliefs don't leak without consent | Visibility levels | Absolute |
| Private beliefs never shared | Architecture | Absolute |
| Aggregation hides individuals | Differential privacy | Mathematical (ε, δ) |
| No deanonymization from aggregates | Min-k threshold | Statistical |
| User can delete their data | Erasure protocol | Best-effort for federation |

### 8.2 What We Don't Guarantee

| Limitation | Reason |
|------------|--------|
| Complete anonymity for public beliefs | User chose public visibility |
| Erasure from all federation nodes | Federated systems can't force |
| Privacy against colluding nodes | Requires majority honest nodes |
| Protection from traffic analysis | Network-level concern |

### 8.3 Privacy Budget Recommendations

```python
# Conservative (maximum privacy)
CONSERVATIVE = PrivacyParameters(
    epsilon=0.01,          # Very low information leakage
    delta=1e-9,
    min_contributors=10,
    max_queries_per_period=10
)

# Balanced (default)
BALANCED = PrivacyParameters(
    epsilon=0.1,
    delta=1e-6,
    min_contributors=5,
    max_queries_per_period=100
)

# Permissive (maximum utility)
PERMISSIVE = PrivacyParameters(
    epsilon=1.0,           # Higher information leakage
    delta=1e-5,
    min_contributors=3,
    max_queries_per_period=1000
)
```

---

## 9. Implementation Checklist

### Phase 3 (Privacy-Preserving Aggregation)
- [ ] Differential privacy implementation
- [ ] Privacy budget tracking
- [ ] Laplace noise injection
- [ ] Minimum contributor threshold
- [ ] Query rate limiting
- [ ] Secure aggregation protocol

### Phase 5 (Integration)
- [ ] GDPR erasure implementation
- [ ] Data portability export
- [ ] Consent management UI
- [ ] Privacy parameter configuration
- [ ] Privacy budget monitoring/alerts

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0-draft | 2025-01-20 | Initial privacy guarantees specification |
