# Context Query Extension

*Optimized retrieval for LLM context windows.*

---

## The Problem

LLM inference has limited context windows. The challenge:
- What to include? (relevance)
- How much of it? (token budget)
- At what detail level? (abstraction)
- How reliable is it? (confidence)

Standard semantic search returns "most similar" — but similar ≠ useful for the current task.

---

## Context Query Operation

### Interface

```typescript
interface ContextQuery {
  // What we're asking about
  semantic: string;
  
  // Task context
  purpose: 'answer' | 'verify' | 'explore' | 'decide' | 'create';
  
  // Budget constraints
  max_tokens: number;          // Hard limit
  target_tokens?: number;      // Preferred (may go slightly over)
  
  // Abstraction preference
  abstraction: 'detailed' | 'summary' | 'essence' | 'adaptive';
  
  // Trust context
  trust_context: TrustContext;
  
  // Expansion options
  include_derivations?: boolean;  // Include reasoning chains
  include_contradictions?: boolean; // Show conflicting beliefs
  expand_on_demand?: boolean;    // Return expansion IDs for follow-up
}

interface ContextResult {
  // Ranked beliefs within budget
  beliefs: ContextBelief[];
  
  // Metadata
  total_tokens: number;
  truncated: boolean;
  confidence_floor: number;      // Lowest confidence included
  coverage_estimate: number;     // How much relevant knowledge was included
  
  // Expansion capability
  expandable_ids?: string[];     // Beliefs with more detail available
  contradictions_available?: boolean;
  
  // Explanation
  selection_reasoning?: string;  // Why these beliefs were chosen
}

interface ContextBelief {
  id: string;
  content: string;               // At requested abstraction level
  confidence: ConfidenceVector;
  source_trust: number;
  relevance_score: number;
  token_count: number;
  
  // Optional expansion
  has_derivation: boolean;
  has_more_detail: boolean;
  contradiction_count: number;
}
```

### Operation

```typescript
async function context_query(
  query: ContextQuery
): Promise<ContextResult> {
  // 1. Semantic retrieval (get candidates)
  const candidates = await semantic_search(query.semantic, {
    limit: query.max_tokens * 2,  // Overfetch for selection
    trust_context: query.trust_context
  });
  
  // 2. Score for context utility
  const scored = candidates.map(b => ({
    belief: b,
    utility: compute_context_utility(b, query)
  }));
  
  // 3. Select within budget
  const selected = select_within_budget(scored, query);
  
  // 4. Abstract to requested level
  const abstracted = selected.map(b => 
    abstract_belief(b, query.abstraction)
  );
  
  // 5. Package result
  return package_context_result(abstracted, query);
}
```

---

## Context Utility Scoring

Not just relevance — utility for the task:

```typescript
function compute_context_utility(
  belief: Belief, 
  query: ContextQuery
): number {
  const relevance = semantic_similarity(belief, query.semantic);
  const confidence = overall_confidence(belief.confidence);
  const trust = get_source_trust(belief.holder_id, query.trust_context);
  const recency = temporal_freshness(belief);
  
  // Task-specific weighting
  const weights = TASK_WEIGHTS[query.purpose];
  
  // Information density (content per token)
  const density = belief.content.length / estimate_tokens(belief.content);
  
  // Novelty (does this add new information vs already selected?)
  const novelty = 1.0;  // Computed incrementally during selection
  
  return (
    relevance * weights.relevance +
    confidence * weights.confidence +
    trust * weights.trust +
    recency * weights.recency +
    density * weights.density +
    novelty * weights.novelty
  );
}

const TASK_WEIGHTS = {
  answer: { relevance: 0.3, confidence: 0.3, trust: 0.2, recency: 0.1, density: 0.05, novelty: 0.05 },
  verify: { relevance: 0.2, confidence: 0.4, trust: 0.2, recency: 0.1, density: 0.05, novelty: 0.05 },
  explore: { relevance: 0.2, confidence: 0.1, trust: 0.1, recency: 0.1, density: 0.1, novelty: 0.4 },
  decide: { relevance: 0.25, confidence: 0.25, trust: 0.25, recency: 0.15, density: 0.05, novelty: 0.05 },
  create: { relevance: 0.2, confidence: 0.1, trust: 0.1, recency: 0.05, density: 0.15, novelty: 0.4 }
};
```

---

## Abstraction Levels

Same belief at different levels:

### Detailed (L1)
Full content with derivation chain:
```
Belief: "PostgreSQL outperforms MongoDB for complex queries 
involving joins and transactions. In benchmark X, PostgreSQL 
completed query Y in 50ms vs MongoDB's 200ms. This aligns with 
PostgreSQL's ACID compliance and mature query optimizer."

Derived from:
- Benchmark result [belief_123] (confidence: 0.9)
- Documentation review [belief_456] (confidence: 0.8)
- Chris's direct experience [belief_789] (confidence: 0.85)
```

### Summary (L2)
Condensed content:
```
Belief: "PostgreSQL preferred over MongoDB for complex queries 
(joins, transactions). Benchmarked 4× faster."
Confidence: 0.85
```

### Essence (L3)
Minimal content:
```
Belief: "PostgreSQL > MongoDB for complex queries"
Confidence: high
```

### Adaptive
System chooses level based on:
- Available budget
- Belief importance
- Redundancy with other selections

---

## Budget-Aware Selection

Greedy selection with look-ahead:

```typescript
function select_within_budget(
  scored: ScoredBelief[],
  query: ContextQuery
): Belief[] {
  const selected: Belief[] = [];
  let tokens_used = 0;
  
  // Sort by utility
  scored.sort((a, b) => b.utility - a.utility);
  
  for (const item of scored) {
    const belief_tokens = estimate_tokens(
      abstract_belief(item.belief, query.abstraction)
    );
    
    if (tokens_used + belief_tokens <= query.max_tokens) {
      // Check novelty vs already selected
      const novelty = compute_novelty(item.belief, selected);
      if (novelty > NOVELTY_THRESHOLD) {
        selected.push(item.belief);
        tokens_used += belief_tokens;
      }
    }
    
    // Early exit if close to budget
    if (tokens_used >= query.target_tokens) break;
  }
  
  return selected;
}
```

---

## Expansion Protocol

For beliefs with more detail available:

```typescript
// Initial query returns summary
const context = await context_query({
  semantic: "database choice",
  max_tokens: 2000,
  abstraction: 'summary'
});

// If model needs more detail on a specific belief
const expanded = await expand_belief(context.beliefs[0].id, {
  include: ['derivation', 'evidence', 'contradictions'],
  max_additional_tokens: 500
});
```

---

## Integration with LLM Inference

### System Prompt Injection

```typescript
function build_system_prompt(
  base_prompt: string,
  context: ContextResult
): string {
  const context_section = `
## Relevant Knowledge (from Valence)

${context.beliefs.map(b => 
  `- ${b.content} (confidence: ${overall_confidence(b.confidence).toFixed(2)})`
).join('\n')}

Note: ${context.truncated ? 'Additional context available on request.' : 'Full relevant context included.'}
Coverage: ~${(context.coverage_estimate * 100).toFixed(0)}% of relevant knowledge.
`;
  
  return base_prompt + context_section;
}
```

### MCP Tool Usage

```typescript
// In MCP server
tools: [{
  name: "get_context",
  description: "Retrieve relevant knowledge for current task",
  inputSchema: {
    type: "object",
    properties: {
      topic: { type: "string" },
      purpose: { enum: ["answer", "verify", "explore", "decide", "create"] },
      max_tokens: { type: "number", default: 2000 }
    }
  }
}]
```

---

## Performance Targets

| Query Type | P50 Latency | P99 Latency |
|------------|-------------|-------------|
| Local context | 20ms | 100ms |
| + Abstraction | 50ms | 200ms |
| + Expansion | 100ms | 500ms |

Context queries should be fast enough for synchronous use in LLM inference pipelines.

---

*Context query: The right knowledge at the right detail for the right task.*
