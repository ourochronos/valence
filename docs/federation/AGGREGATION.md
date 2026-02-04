# Cross-Federation Belief Aggregation

*Privacy-preserving aggregation of beliefs across federation nodes.*

**Status:** IMPLEMENTED  
**Issue:** #15  
**Date:** 2026-02-04

---

## Overview

The Federation Aggregation module enables privacy-preserving aggregation of beliefs across multiple federation nodes. It handles:

1. **Cross-federation belief collection** - Gather beliefs from multiple federations on a topic
2. **Conflict detection** - Identify when federations disagree
3. **Trust-weighted aggregation** - Weight contributions by federation trust
4. **Privacy preservation** - Apply differential privacy to results

---

## Quick Start

```python
from valence.federation import (
    FederationAggregator,
    AggregationConfig,
    create_contribution,
    ConflictResolution,
)

# Create aggregator with default config
aggregator = FederationAggregator()

# Or with custom config
config = AggregationConfig(
    conflict_resolution=ConflictResolution.TRUST_WEIGHTED,
    min_federations=3,
)
aggregator = FederationAggregator(config=config)

# Create contributions from each federation
contributions = [
    create_contribution(
        federation_id=fed_1_id,
        node_id=node_1_id,
        federation_did="did:vkb:fed:example-1",
        beliefs=beliefs_from_fed_1,
        trust_score=0.8,
    ),
    create_contribution(
        federation_id=fed_2_id,
        node_id=node_2_id,
        federation_did="did:vkb:fed:example-2",
        beliefs=beliefs_from_fed_2,
        trust_score=0.7,
    ),
    # ... more contributions
]

# Aggregate
result = aggregator.aggregate(
    contributions,
    domain_filter=["ai", "safety"],
    semantic_query="What are the risks of AI?",
)

print(f"Collective confidence: {result.collective_confidence}")
print(f"Agreement score: {result.agreement_score}")
print(f"Conflicts detected: {result.conflict_count}")
```

---

## Components

### FederationContribution

Represents a federation's contribution to an aggregation query:

```python
@dataclass
class FederationContribution:
    federation_id: UUID           # Federation identifier
    node_id: UUID                 # Node identifier
    federation_did: str           # DID string
    trust_score: float = 0.5      # Federation trust (0-1)
    is_anchor: bool = False       # Is this an anchor federation?
    beliefs: list[FederatedBelief] = []  # Beliefs contributed
    local_confidence: float | None = None  # Pre-computed local aggregate
```

### AggregationConfig

Configuration for aggregation behavior:

```python
@dataclass
class AggregationConfig:
    # Strategy
    strategy: AggregationStrategy = WEIGHTED_MEAN
    conflict_resolution: ConflictResolution = TRUST_WEIGHTED
    
    # Weights
    trust_weight: float = 0.5
    recency_weight: float = 0.2
    corroboration_weight: float = 0.3
    
    # Thresholds
    min_federation_trust: float = 0.1
    min_federations: int = 2
    min_total_beliefs: int = 3
    
    # Privacy
    privacy_config: PrivacyConfig = PrivacyConfig()
```

### CrossFederationAggregateResult

The result of aggregation:

```python
@dataclass
class CrossFederationAggregateResult:
    collective_confidence: float  # Aggregate confidence (0-1)
    agreement_score: float        # How much federations agree (0-1)
    federation_count: int         # Number of contributing federations
    total_belief_count: int       # Total beliefs aggregated
    conflicts_detected: list[DetectedConflict]
    k_anonymity_satisfied: bool   # Privacy guarantee met?
    privacy_epsilon: float        # DP epsilon used
```

---

## Conflict Detection

The module detects four types of conflicts:

| Type | Description | Example |
|------|-------------|---------|
| **CONTRADICTION** | Beliefs directly contradict | "X is true" vs "X is false" |
| **DIVERGENCE** | Significant confidence gap | 0.9 vs 0.3 on same topic |
| **TEMPORAL** | Conflicting validity periods | "Valid 2020-2023" vs "Valid 2022-2025" |
| **SCOPE** | Different domain applicability | Belief applies to different contexts |

### Configuration

```python
detector = ConflictDetector(
    semantic_threshold=0.75,      # Min similarity to consider same topic
    confidence_divergence=0.4,    # Min confidence diff for divergence
    stance_threshold=0.6,         # Threshold for contradiction
)
```

### Custom Similarity Function

You can provide a custom similarity function for better topic matching:

```python
def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity using embeddings."""
    emb_a = embed(text_a)
    emb_b = embed(text_b)
    return cosine_similarity(emb_a, emb_b)

result = aggregator.aggregate(
    contributions,
    similarity_fn=semantic_similarity,
)
```

---

## Conflict Resolution Strategies

### TRUST_WEIGHTED (Default)

Keep all beliefs but let trust weights handle the aggregation:
- Higher-trust federations' beliefs have more influence
- Conflicts are reflected in lower agreement scores

### RECENCY_WINS

When conflicts occur, prefer more recent beliefs:
- Compare `signed_at` timestamps
- Exclude older conflicting beliefs

### CORROBORATION

Prefer beliefs with more support:
- Uses federation trust as proxy for corroboration
- Lower-trust federation's belief excluded in conflicts

### EXCLUDE_CONFLICTING

Remove all conflicting beliefs from aggregation:
- Conservative approach
- May result in fewer beliefs

### FLAG_FOR_REVIEW

Don't resolve, just flag conflicts:
- Conflicts appear in result
- Human review expected

---

## Trust-Weighted Aggregation

Contribution weights are computed from three components:

```
weight = trust_component + recency_component + corroboration_component
```

### Trust Component

Based on federation trust score with optional anchor bonus:

```python
trust_component = federation.trust_score * trust_weight
if federation.is_anchor:
    trust_component += ANCHOR_TRUST_BOOST * trust_weight
```

### Recency Component

Exponential decay based on most recent belief:

```python
recency = exp(-ln(2) * age_days / 30)  # Half-life of 30 days
```

### Corroboration Component

Based on relative belief contribution:

```python
corroboration = sqrt(belief_count / total_beliefs)
```

---

## Privacy Preservation

The aggregation applies differential privacy:

1. **k-Anonymity** - Minimum 5 contributors required (10 for sensitive domains)
2. **Noise Injection** - Laplace/Gaussian noise added to statistics
3. **Histogram Suppression** - Distributions hidden below threshold
4. **Budget Tracking** - Daily epsilon limits enforced

### Sensitive Domains

Domains like health, finance, legal automatically get stricter privacy:

```python
SENSITIVE_DOMAINS = [
    "health", "medical", "finance", "legal", "politics", ...
]
```

### Privacy Budget

```python
from valence.federation import PrivacyBudget

budget = PrivacyBudget(
    federation_id=fed_id,
    daily_epsilon_budget=10.0,  # 10 queries at ε=1.0 per day
)

aggregator = FederationAggregator(config=config)
aggregator.privacy_aggregator.privacy_budget = budget
```

---

## Integration with Trust System

The aggregator can use the trust system for federation trust lookup:

```python
from valence.federation import get_effective_trust

def lookup_trust(federation_id: UUID) -> float:
    return get_effective_trust(federation_id)

aggregator = FederationAggregator(
    config=config,
    trust_lookup=lookup_trust,
)
```

---

## Best Practices

1. **Minimum Federations** - Always require at least 2 federations for meaningful aggregation
2. **Trust Thresholds** - Filter out low-trust federations (< 0.1)
3. **Privacy First** - Use default privacy config unless you have specific needs
4. **Conflict Review** - Monitor conflict rates to detect manipulation
5. **Temporal Smoothing** - Enable for membership-change protection

---

## API Reference

### Functions

```python
# Convenience function for one-shot aggregation
aggregate_cross_federation(
    contributions: list[FederationContribution],
    domain_filter: list[str] | None = None,
    semantic_query: str | None = None,
    config: AggregationConfig | None = None,
    trust_lookup: Callable[[UUID], float] | None = None,
) -> CrossFederationAggregateResult

# Helper to create contributions
create_contribution(
    federation_id: UUID,
    node_id: UUID,
    federation_did: str,
    beliefs: list[FederatedBelief],
    trust_score: float = 0.5,
    is_anchor: bool = False,
) -> FederationContribution
```

### Classes

- `FederationAggregator` - Main aggregation engine
- `ConflictDetector` - Detects belief conflicts
- `TrustWeightedAggregator` - Computes trust-weighted statistics
- `PrivacyPreservingAggregator` - Applies differential privacy

---

## Related Documentation

- [Federation Layer Spec](../../spec/components/federation-layer/SPEC.md)
- [Differential Privacy](../../spec/components/federation-layer/DIFFERENTIAL-PRIVACY.md)
- [Privacy Module](../../src/valence/federation/privacy.py)
