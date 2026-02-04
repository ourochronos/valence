# Ranking Algorithm Specification

*How Valence decides what knowledge matters most*

---

## Overview

Ranking in Valence is not just relevance — it's **trusted relevance**. A belief's position in results reflects:

1. **Semantic similarity** — How well it matches what you asked
2. **Confidence** — How certain the belief is (on multiple dimensions)  
3. **Trust** — How much you trust the source (direct and transitive)
4. **Recency** — How fresh the knowledge is
5. **Diversity** — Avoiding echo chambers of similar beliefs

This document specifies the exact formulas and algorithms.

---

## The Master Scoring Formula

### Final Score Computation

Every belief receives a final score ∈ [0.0, 1.0]:

```
final_score = (
    w_semantic × semantic_score +
    w_confidence × confidence_score +
    w_trust × trust_score +
    w_recency × recency_score
) × diversity_multiplier
```

Where:
- `w_semantic` = 0.35 (default)
- `w_confidence` = 0.25 (default)
- `w_trust` = 0.30 (default)
- `w_recency` = 0.10 (default)
- Sum of weights = 1.0

**Weight customization:** Query context can override these weights:

```typescript
context.ranking_weights = {
  semantic: 0.50,    // "I care more about relevance"
  confidence: 0.20,
  trust: 0.20,
  recency: 0.10
}
```

---

## Component 1: Semantic Similarity

### Base Computation

```
semantic_score = cosine_similarity(query_embedding, belief_embedding)
```

Both embeddings are L2-normalized, so:
```
cosine_similarity = dot_product(q, b)  // In [-1, 1], but typically [0, 1] for text
```

### Normalization

Raw cosine scores are transformed to [0, 1]:

```
semantic_score = (raw_cosine + 1) / 2
```

For most text comparisons, scores naturally fall in [0.3, 0.95].

### Query Expansion

For richer matching, queries may be expanded:

```
expanded_query_embedding = normalize(
    original_embedding + 
    0.3 × mean(related_term_embeddings)
)
```

This helps when queries are short or ambiguous.

### Phrase Matching Boost

If query contains quoted phrases and belief content contains exact matches:

```
phrase_boost = 0.1 × (matched_phrase_count / total_phrase_count)
semantic_score = min(1.0, semantic_score + phrase_boost)
```

### Retrieval Strategy

For billion-scale efficiency:

1. **Stage 1: ANN retrieval** — Get top 10K candidates via HNSW index
2. **Stage 2: Reranking** — Apply full scoring to candidates
3. **Stage 3: Filtering** — Remove candidates failing hard filters

HNSW parameters:
- `ef_search = 100` (default, adjustable)
- `M = 32` (index build-time)
- `ef_construction = 200` (index build-time)

---

## Component 2: Confidence Score

### Aggregation Formula

```
confidence_score = geometric_mean(
    source_reliability ^ w_sr,
    method_quality ^ w_mq,
    internal_consistency ^ w_ic,
    temporal_freshness ^ w_tf,
    corroboration ^ w_cor,
    domain_applicability ^ w_da
)
```

With default weights:
- `w_sr` = 0.25 (source reliability)
- `w_mq` = 0.20 (method quality)
- `w_ic` = 0.15 (internal consistency)
- `w_tf` = 0.15 (temporal freshness)
- `w_cor` = 0.15 (corroboration)
- `w_da` = 0.10 (domain applicability)

### Why Geometric Mean?

Geometric mean penalizes beliefs with any single weak dimension:

| Arithmetic Mean | Geometric Mean |
|-----------------|----------------|
| [0.9, 0.9, 0.9, 0.9, 0.9, 0.1] = 0.75 | = **0.58** |
| [0.7, 0.7, 0.7, 0.7, 0.7, 0.7] = 0.70 | = **0.70** |

The geometric mean correctly values consistent confidence over one-dimensional strength.

### Context-Adjusted Weights

Different queries emphasize different dimensions:

```typescript
// For breaking news queries
confidence_weights.temporal_freshness = 0.35;
confidence_weights.corroboration = 0.10;  // Less time for corroboration

// For scientific queries  
confidence_weights.method_quality = 0.35;
confidence_weights.corroboration = 0.25;

// For cross-domain queries
confidence_weights.domain_applicability = 0.25;
```

### Handling Missing Dimensions

If a dimension is `null`:
1. Use default value (0.5) for that dimension
2. Redistribute weight to other dimensions
3. Flag in explanation: "confidence incomplete"

---

## Component 3: Trust Score

Trust scoring is the most complex component, combining direct trust, transitive trust, and reputation.

### Direct Trust

If requester has a trust edge to belief holder:

```
direct_trust = trust_graph.get(holder_id).trust_level

// Domain-specific override
if query_domains.length > 0:
    domain_trust = max(
        trust_graph.get(holder_id).domain_trust.get(d) 
        for d in query_domains
    )
    direct_trust = max(direct_trust, domain_trust)
```

### Transitive Trust

If no direct trust, compute transitive trust:

```
transitive_trust = max over all paths P from requester to holder {
    product(edge.trust × damping^depth for edge in P)
}
```

Where:
- `damping` = 0.7 (default)
- `max_hops` = 3 (default)

**Example:**
```
requester → A (trust=0.9) → B (trust=0.8) → holder (trust=0.7)

path_trust = 0.9 × 0.7 × 0.8 × 0.7 × 0.7 × 0.7
           = 0.9 × 0.8 × 0.7 × (0.7^3)
           = 0.504 × 0.343
           = 0.173
```

### Reputation Factor

Network reputation provides a floor for unknown sources:

```
reputation_factor = holder.reputation.overall × 0.3
```

Reputation alone never exceeds 0.3 contribution.

### Combined Trust Score

```
trust_score = max(
    direct_trust,
    transitive_trust,
    reputation_factor,
    default_trust        // 0.1, baseline for complete strangers
)
```

### Trust Score Bounds

- **Minimum**: `default_trust` (0.1) — No belief is completely untrusted
- **Maximum**: 1.0 (only for self-beliefs)
- **Practical ceiling**: 0.95 (very rarely should external trust hit 1.0)

### Sybil Resistance in Scoring

Transitive trust requires independent paths:

```
if transitive_trust > 0.5:
    distinct_sources = count_distinct_first_hops(paths)
    if distinct_sources < sybil_threshold:
        transitive_trust = min(transitive_trust, 0.5)
```

---

## Component 4: Recency Score

### Base Recency Computation

```
recency_score = temporal_freshness × decay_multiplier

decay_multiplier = exp(-λ × age_days)
```

Where:
- `λ` = domain-specific decay rate
- `age_days` = days since belief creation

### Domain-Specific Decay Rates

| Domain Pattern | λ (per day) | Half-life |
|----------------|-------------|-----------|
| `news/*` | 0.10 | 7 days |
| `prices/*`, `stocks/*` | 0.50 | 1.4 days |
| `weather/*` | 1.00 | 0.7 days |
| `science/*` | 0.002 | 346 days |
| `history/*` | 0.0001 | 6931 days |
| `math/*` | 0.0 | ∞ (no decay) |
| (default) | 0.01 | 69 days |

### Query-Time Recency Boost

For queries with temporal signals:

```
if query contains temporal_keywords ["recent", "latest", "new", "current"]:
    recency_weight *= 2.0  // Double recency importance
    
if query contains historical_keywords ["history", "origin", "first", "original"]:
    recency_weight *= 0.5  // Halve recency importance
```

### Validity Window Enforcement

Hard filter, not soft ranking:

```
if valid_from > query_time OR valid_until < query_time:
    belief is excluded (not ranked)
```

---

## Component 5: Diversity

Diversity prevents the top results from being N near-identical beliefs.

### Maximal Marginal Relevance (MMR)

We use MMR for diversity-aware ranking:

```
MMR(belief) = λ × relevance_score - (1 - λ) × max_similarity_to_selected

where:
- relevance_score = final_score (before diversity)
- max_similarity_to_selected = max(cosine_sim(belief, s) for s in selected_results)
- λ = diversity_balance (default 0.7)
```

### Diversity Algorithm

```python
def diverse_ranking(candidates, limit, diversity_config):
    selected = []
    remaining = candidates.copy()
    
    while len(selected) < limit and remaining:
        # Score each remaining candidate
        for belief in remaining:
            if len(selected) == 0:
                belief.mmr_score = belief.final_score
            else:
                max_sim = max(
                    cosine_similarity(belief.embedding, s.embedding)
                    for s in selected
                )
                belief.mmr_score = (
                    diversity_config.lambda * belief.final_score -
                    (1 - diversity_config.lambda) * max_sim
                )
        
        # Select best MMR score
        best = max(remaining, key=lambda b: b.mmr_score)
        selected.append(best)
        remaining.remove(best)
    
    return selected
```

### Holder Diversity

Limit beliefs from the same source:

```
if count(selected where holder_id == belief.holder_id) >= max_per_holder:
    belief.mmr_score *= 0.5  // Penalty, not exclusion
```

### Domain Diversity

Spread across domains:

```
if count(selected where domains overlap belief.domains) >= max_per_domain:
    belief.mmr_score *= 0.7  // Lighter penalty
```

### Diversity Multiplier

The `diversity_multiplier` in the final score:

```
diversity_multiplier = 1.0 - diversity_penalty

where diversity_penalty = max(
    holder_penalty,      // 0-0.5
    similarity_penalty,  // 0-0.3  
    domain_penalty       // 0-0.3
)
```

---

## Handling Conflicting Beliefs

When multiple beliefs contradict each other:

### Detection

Contradiction detection runs during ranking:

```
for each pair of high-ranked beliefs (A, B):
    contradiction_score = contradiction_model(A.content, B.content)
    if contradiction_score > 0.7:
        mark_as_conflicting(A, B)
```

### Presentation Strategies

**Strategy 1: Flag and Show Both**
```
result.conflicts = [
    { belief_a: A, belief_b: B, type: "direct_contradiction" }
]
```

Both beliefs appear in results with conflict markers.

**Strategy 2: Highest Confidence Wins**
```
if A.confidence_score > B.confidence_score:
    B.final_score *= (1 - contradiction_score × 0.5)
```

Lower-confidence contradiction is penalized (not hidden).

**Strategy 3: Trust-Weighted Resolution**
```
if A.trust_score >> B.trust_score:
    // Heavily trusted source wins
    B.final_score *= (1 - contradiction_score × 0.7)
```

### Contradiction Reporting

Always report contradictions in results:

```typescript
interface ConflictInfo {
  belief_ids: [UUID, UUID];
  contradiction_type: 'direct' | 'implied' | 'temporal';
  resolution: 'both_shown' | 'one_penalized' | 'one_excluded';
  resolution_reason: string;
}
```

---

## Scoring Examples

### Example 1: High-Trust Recent News

Query: "Latest AI announcements"

```
Belief: "Anthropic announced Claude 3 on March 4, 2024"
- semantic_score: 0.88 (good match)
- confidence_score: 0.82 (reliable source, recent)
- trust_score: 0.90 (direct trust to Anthropic feed)
- recency_score: 0.95 (3 days old, news decay)

final_score = 0.35(0.88) + 0.25(0.82) + 0.30(0.90) + 0.10(0.95)
            = 0.308 + 0.205 + 0.270 + 0.095
            = 0.878
```

### Example 2: Low-Trust But Highly Corroborated

Query: "Side effects of medication X"

```
Belief: "Medication X commonly causes drowsiness"
- semantic_score: 0.91 (exact match)
- confidence_score: 0.88 (high corroboration, good method)
- trust_score: 0.25 (unknown source, no trust path)
- recency_score: 0.70 (2 years old, medical decay)

final_score = 0.35(0.91) + 0.25(0.88) + 0.30(0.25) + 0.10(0.70)
            = 0.319 + 0.220 + 0.075 + 0.070
            = 0.684
```

### Example 3: Trusted But Tangential

Query: "Python async patterns"

```
Belief: "Python 3.10 introduced structural pattern matching"
- semantic_score: 0.52 (related but not direct)
- confidence_score: 0.95 (documentation-verified)
- trust_score: 0.92 (trusted coding source)
- recency_score: 0.80 (3 years old, tech decay)

final_score = 0.35(0.52) + 0.25(0.95) + 0.30(0.92) + 0.10(0.80)
            = 0.182 + 0.238 + 0.276 + 0.080
            = 0.776
```

Despite high trust and confidence, poor semantic match limits score.

### Example 4: Diversity Penalty Applied

Query: "Climate change effects"

```
Belief 3 in ranking: "Rising sea levels threaten coastal cities"
- raw_final_score: 0.85

But beliefs 1 and 2 are very similar (coastal flooding focus):
- max_similarity_to_selected: 0.89

MMR with λ=0.7:
mmr_score = 0.7(0.85) - 0.3(0.89)
          = 0.595 - 0.267
          = 0.328

This belief drops in ranking to promote diversity.
```

---

## Performance Optimizations

### Two-Phase Ranking

For billion-scale efficiency:

**Phase 1: Approximate Ranking (fast)**
- HNSW retrieval on semantic similarity only
- Retrieve 10K candidates
- Takes ~10ms

**Phase 2: Precise Reranking (accurate)**
- Apply full scoring formula to candidates
- Compute trust scores (cached where possible)
- Apply diversity algorithm
- Takes ~50ms

### Trust Score Caching

Trust computations are expensive. Cache aggressively:

```
TrustCache {
    key: (requester_id, holder_id, domain)
    value: trust_score
    ttl: 3600s  // 1 hour
}
```

Invalidate on:
- Trust graph updates
- Reputation changes > 0.1

### Embedding Pre-computation

All beliefs have pre-computed embeddings. Never compute embeddings at query time.

### Sharding Strategy

For horizontal scale:

```
shard_key = hash(belief_id) % num_shards

Query routing:
1. Broadcast query to all shards
2. Each shard returns top-K
3. Merge and rerank globally
```

---

## Explainability

Every score component must be explainable.

### Score Decomposition

```typescript
interface ScoreBreakdown {
  final_score: number;
  
  semantic: {
    score: number;
    raw_cosine: number;
    phrase_boost: number;
    explanation: "High semantic match on terms: [AI, announcement, claude]"
  };
  
  confidence: {
    score: number;
    by_dimension: {
      source_reliability: { value: number, weight: number, contribution: number },
      method_quality: { value: number, weight: number, contribution: number },
      // ... other dimensions
    };
    weakest: "corroboration (0.3)";
    explanation: "Good overall confidence, limited by single-source corroboration"
  };
  
  trust: {
    score: number;
    type: "direct" | "transitive" | "reputation_only";
    path?: string;  // "you → alice → bob → holder"
    explanation: "Direct trust relationship with holder (0.85)"
  };
  
  recency: {
    score: number;
    age_days: number;
    decay_rate: number;
    domain_used: string;
    explanation: "Recent belief (3 days), minimal decay applied"
  };
  
  diversity: {
    penalty: number;
    reason?: string;
    explanation: "No diversity penalty - sufficiently distinct from higher-ranked results"
  };
}
```

### Counterfactual Explanations

"What would change this score?"

```typescript
interface Counterfactual {
  dimension: string;
  current_value: number;
  needed_for_top_3: number;
  suggestion: string;
}

// Example output:
[
  {
    dimension: "corroboration",
    current_value: 0.3,
    needed_for_top_3: 0.6,
    suggestion: "Independent verification from 2+ sources would significantly boost ranking"
  }
]
```

---

## Tuning Guidelines

### For Different Use Cases

**Personal knowledge retrieval:**
```
w_semantic: 0.40
w_trust: 0.15      // You trust yourself
w_confidence: 0.35
w_recency: 0.10
```

**News/current events:**
```
w_semantic: 0.30
w_trust: 0.25
w_confidence: 0.15
w_recency: 0.30    // Freshness matters most
```

**Scientific research:**
```
w_semantic: 0.25
w_trust: 0.20
w_confidence: 0.45  // Method and corroboration matter
w_recency: 0.10
```

**Exploratory/brainstorming:**
```
w_semantic: 0.50    // Cast wide net
w_trust: 0.15
w_confidence: 0.15
w_recency: 0.05
diversity.lambda: 0.5  // High diversity
```

### A/B Testing Framework

```typescript
interface RankingExperiment {
  id: string;
  name: string;
  control_weights: RankingWeights;
  treatment_weights: RankingWeights;
  traffic_percentage: number;
  metrics: ["ndcg@10", "click_through_rate", "satisfaction_score"];
}
```

---

## Algorithm Pseudocode

Complete ranking algorithm:

```python
def rank_beliefs(query: Query, candidates: List[Belief]) -> List[RankedBelief]:
    # Phase 1: Score each candidate
    scored = []
    for belief in candidates:
        scores = {
            'semantic': compute_semantic_score(query.embedding, belief.embedding),
            'confidence': compute_confidence_score(belief.confidence, query.context),
            'trust': compute_trust_score(
                query.context.requester_id,
                belief.holder_id,
                query.context.trust_graph,
                query.context.query_domains
            ),
            'recency': compute_recency_score(
                belief.created_at,
                belief.confidence.temporal_freshness,
                get_decay_rate(belief.domains)
            )
        }
        
        weights = query.context.ranking_weights or DEFAULT_WEIGHTS
        
        raw_score = (
            weights.semantic * scores['semantic'] +
            weights.confidence * scores['confidence'] +
            weights.trust * scores['trust'] +
            weights.recency * scores['recency']
        )
        
        scored.append(RankedBelief(
            belief=belief,
            raw_score=raw_score,
            component_scores=scores
        ))
    
    # Phase 2: Detect contradictions
    conflicts = detect_contradictions(scored)
    for conflict in conflicts:
        apply_contradiction_penalty(conflict, scored)
    
    # Phase 3: Apply diversity (MMR)
    if query.options.diversity.enabled:
        ranked = apply_mmr(
            scored,
            query.options.limit,
            query.options.diversity
        )
    else:
        scored.sort(key=lambda x: x.raw_score, reverse=True)
        ranked = scored[:query.options.limit]
    
    # Phase 4: Generate explanations if requested
    if query.options.include_explanation:
        for belief in ranked:
            belief.explanation = generate_explanation(belief, query)
    
    return ranked
```

---

*"Rank by trust, not by tricks. Explain, don't obscure."*
