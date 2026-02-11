# Semantic Compression

**Status:** Deferred

## Summary

Reduce belief verbosity and redundancy by compressing semantically similar beliefs into distilled representations. Currently deferred because the dedup and corroboration work (#329/#332) addresses the acute pain point of duplicate and near-duplicate beliefs without requiring LLM-in-the-loop processing.

## Problem

Beliefs captured from conversations tend to be:
- **Verbose** -- Conversational phrasing carried into belief content ("The user mentioned that they prefer...")
- **Repetitive** -- The same insight captured across multiple sessions with slightly different wording
- **Overly specific** -- Context-bound details that don't generalize ("In the valence repo on Tuesday, we decided...")

Over time this creates a noisy knowledge base where retrieval returns multiple beliefs that say essentially the same thing, wasting context window on redundant information.

## Approaches

### 1. Truncation (current)

Simple character/token limits on belief content. Prevents individual beliefs from being excessively long but does nothing about cross-belief redundancy.

### 2. LLM Distillation

Use an LLM to summarize N related beliefs into a single distilled belief. For example, five beliefs about Python testing preferences become one authoritative belief. The distilled belief supersedes all inputs, maintaining the supersession chain.

### 3. Hybrid

Combine embedding-based grouping with LLM distillation:
1. Group semantically similar beliefs using embedding similarity (threshold ~0.85)
2. Within each group, use an LLM to produce a single distilled belief
3. The distilled belief inherits the highest confidence dimensions from its inputs
4. Corroboration count reflects the number of independent sources across all inputs

## Current Solution

Dedup (#329) and corroboration (#332) solve the most pressing version of this problem:
- **Dedup** prevents creating near-duplicate beliefs at write time
- **Corroboration** tracks when multiple sources confirm the same belief, boosting confidence without creating duplicates

Together these keep the belief count manageable and retrieval clean.

## Future Design

A periodic compression batch job as part of the active curation layer:

```
1. Query all beliefs in a domain
2. Compute pairwise embedding similarity
3. Identify clusters above similarity threshold
4. For each cluster with 3+ beliefs:
   a. Send beliefs to LLM with distillation prompt
   b. Create new distilled belief
   c. Supersede all input beliefs -> distilled belief
   d. Preserve corroboration counts and source links
```

### Key Design Decisions

- **Distilled beliefs are new beliefs**, not edits. The supersession chain preserves full history.
- **Minimum cluster size of 3** avoids compressing pairs where both might carry distinct nuance.
- **Domain-scoped** to avoid cross-domain compression losing important context.
- **Human-reviewable** -- the curation agent proposes compressions, doesn't auto-apply.

## Prerequisites

- Active curation agent (see `active-curation-vision.md`)
- LLM access for distillation (could use Claude API or local model)
- Stable embedding model for reliable similarity grouping

## Why Deferred

Dedup and corroboration handle the current pain point. Semantic compression is an optimization for a larger knowledge base where retrieval noise becomes a real problem. Building it now would add LLM-in-the-loop complexity for a problem that isn't yet acute.
