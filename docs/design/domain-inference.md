# Learned Domain Inference

**Status:** Deferred

## Summary

Replace manual `domain_path` assignment on beliefs with learned domain inference using embedding-based clustering. Currently deferred because manual assignment combined with `project_context` from session hooks works well at the current scale of the knowledge base.

## Current State

Domain classification happens in two ways:

1. **Explicit assignment** -- When creating a belief via MCP tools, the caller passes a `domain_path` array (e.g., `["tech", "python", "testing"]`). Claude picks the path based on conversational context, which works but is inconsistent across sessions.

2. **Project context from hooks** -- The session-start hook injects `project_context` (derived from the working directory), which gives Claude a strong signal for domain classification. This covers the most common case: beliefs created during active development sessions.

The combination is adequate. Most beliefs get reasonable domains. The pain points are:
- Inconsistent granularity (some paths are 2 levels, some are 4)
- No normalization across sessions (e.g., `["tech", "db"]` vs `["tech", "database"]`)
- Beliefs created outside project contexts sometimes get vague or missing domains

## Future Direction

When belief volume is sufficient to support meaningful clusters, use embedding-based domain inference:

1. **Supervised seeding** -- Use existing `domain_path` labels as training signal. Group belief embeddings by their assigned domain and compute cluster centroids.

2. **k-means clustering** -- Run k-means (or HDBSCAN for variable-density clusters) on belief embeddings. Compare discovered clusters against existing domain labels to validate or propose new domains.

3. **Inference at creation time** -- When a new belief is created without a `domain_path`, find the nearest cluster centroid and suggest a domain. If the caller provides a path, use it as additional training signal.

4. **Periodic reconciliation** -- Batch job that re-evaluates domain assignments, proposes merges for near-duplicate domains (`db` vs `database`), and flags outlier beliefs that don't fit any cluster well.

### Architecture Sketch

```
belief_create (no domain_path)
  -> generate embedding
  -> find k nearest labeled beliefs
  -> majority-vote domain_path from neighbors
  -> assign with confidence based on neighbor distance
```

## Prerequisites

- **Belief volume**: Need enough beliefs per domain for meaningful clusters. Rough target: 50+ beliefs in at least 10 distinct domains.
- **Stable embedding model**: Switching models invalidates centroids. Wait until we have a committed embedding provider (tracked in our-embeddings).
- **Embedding coverage**: Most beliefs need embeddings. Current coverage depends on OPENAI_API_KEY availability.

## Why Deferred

The manual approach works. The inconsistencies are annoying but not blocking. The dedup and corroboration work (#329/#332) reduces noise more effectively than better domain labels would. This becomes valuable when the KB is large enough that browsing by domain matters for curation and when we have the active curation agent to act on inferred domains.
