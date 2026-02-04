# Valence as Context Window Solution

*"The biggest challenge with inference is context window management."* — Chris

---

## The Problem

Every inference call needs context. But context windows are limited. How do you choose what to include?

**Current approaches fail:**
- **Recency**: Recent ≠ relevant
- **Semantic similarity**: Similar ≠ useful (retrieves related noise)
- **Everything**: Exceeds window, dilutes signal

**What we actually need:**
- **Relevant**: Actually about this query
- **Potent**: High information density
- **Accurate**: Trustworthy, verified
- **Needed**: Fills a gap the model can't fill from training

---

## How Valence Solves This

### 1. Multi-Signal Retrieval
Don't just match semantically. Score on:
- Semantic similarity (is it about X?)
- Confidence vector (is it reliable?)
- Source trust (do I trust who said this?)
- Temporal validity (is it still true?)
- Domain applicability (does it apply here?)

**Result**: Top-K beliefs are the ones most likely to *help*, not just most similar.

### 2. Compression Pyramid
Store beliefs at multiple abstraction levels:
```
L0: Raw source ("Meeting notes from 2024-01-15...")
L1: Extracted beliefs ("Chris prefers X over Y")
L2: Summaries ("Chris's technical preferences")
L3: Essence ("Chris values simplicity")
L4: Tags/embeddings (for retrieval)
```

**For context windows**: Start with L2-L3 (high potency). Drill to L1-L0 only if needed.

### 3. Confidence-Based Prioritization
Limited space? Include high-confidence beliefs first.
- 0.9 confidence belief about X > 0.5 confidence belief about X
- Verified beliefs > unverified
- Recent + stable > old + unstable

### 4. Derivation Chains (On-Demand Detail)
Don't include the full chain. Include the conclusion.
If the model needs to verify, it can request the derivation.

```
Context: "Chris prefers Postgres over MongoDB (confidence: 0.85)"
Model asks: "Why?"
Expand: "Derived from: 3 conversations where he chose Postgres, 
         1 explicit statement, corroborated by project choices"
```

### 5. Query-Aware Filtering
Different queries need different context:
- "What does Chris think about X?" → Personal beliefs, high trust
- "What's the consensus on X?" → Communal knowledge, high corroboration
- "What changed recently about X?" → Temporal freshness weighted high

---

## Implementation in Query Protocol

The query-protocol component should include:

```typescript
interface ContextQuery {
  text: string;                    // What we're asking about
  purpose: 'inference' | 'verification' | 'exploration';
  max_tokens: number;              // Context budget
  abstraction_preference: 'detailed' | 'summary' | 'adaptive';
  trust_context: TrustContext;     // Who's asking, what do they trust
}

interface ContextResult {
  beliefs: RankedBelief[];         // Ordered by utility
  total_tokens: number;            // Actual usage
  compression_level: number;       // What abstraction level dominated
  confidence_floor: number;        // Lowest confidence included
  derivations_available: boolean;  // Can expand if needed
}
```

---

## Key Insight

**Context window management IS epistemic retrieval.**

The question "what context do I need?" is the same as:
- "What do I believe that's relevant?"
- "What's trustworthy enough to use?"
- "What level of detail is appropriate?"

Valence answers all three.

---

## Integration with Existing Systems

This should work with:
- **RAG pipelines**: Replace naive retrieval with Valence queries
- **Agent memory**: Personal beliefs as primary context source
- **MCP**: Valence as context provider tool
- **OpenClaw**: Automatic context injection from belief store

---

## Success Metric

**Good context window management:**
- Higher inference quality per token
- Fewer hallucinations (grounded in trusted beliefs)
- Appropriate confidence in responses
- Ability to cite sources and derivations

**Valence enables**: "I'm answering based on beliefs X, Y, Z with confidences A, B, C"

---

*This isn't a separate feature. This is why Valence exists.*
