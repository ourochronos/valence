# Confidence Vector System — Specification

*"Uncertainty as first-class data, not a bug to hide."*

---

## Overview

A ConfidenceVector is a six-dimensional representation of epistemic confidence that travels with every belief. Instead of binary true/false or a single 0-1 score, Valence captures *why* we're confident (or not) across orthogonal dimensions.

This enables:
- **Calibrated reasoning**: Different situations weight dimensions differently
- **Composable aggregation**: Multiple sources combine mathematically
- **Transparent uncertainty**: Humans and agents can see *where* confidence breaks down
- **Temporal awareness**: Knowledge decays; freshness is explicit

---

## The Six Dimensions

### 1. Source Reliability (`source_reliability`)
**Question**: How trustworthy is the origin of this belief?

| Score | Interpretation |
|-------|----------------|
| 1.0 | Highly trusted source with verified track record |
| 0.8 | Known reliable source with occasional errors |
| 0.6 | Unknown source, no negative signals |
| 0.4 | Source with mixed history |
| 0.2 | Source with known credibility issues |
| 0.0 | Untrusted or adversarial source |

**Factors**:
- Historical accuracy (verification outcomes)
- Domain-specific expertise
- Reputation score of the belief holder
- Direct trust relationship (in your trust graph)

**Default**: 0.5 (unknown source, benefit of doubt)

---

### 2. Method Quality (`method_quality`)
**Question**: How was this knowledge derived?

| Score | Method | Description |
|-------|--------|-------------|
| 1.0 | Direct observation | First-hand sensory/measurement |
| 0.9 | Controlled experiment | Systematic testing with controls |
| 0.8 | Formal proof | Mathematical/logical derivation |
| 0.7 | Statistical inference | Data-driven with quantified uncertainty |
| 0.6 | Expert consensus | Multiple domain experts agree |
| 0.5 | Reasoned inference | Logical deduction from premises |
| 0.4 | Analogy/pattern | Similar cases suggest this |
| 0.3 | Secondhand report | Someone reliable told me |
| 0.2 | Hearsay | Multiple degrees of separation |
| 0.1 | Speculation | Educated guess |
| 0.0 | Unknown method | No derivation information |

**Derivation Chain**: If a belief is derived from other beliefs, method_quality is bounded by the lowest-quality step in the chain.

**Default**: 0.3 (treat as secondhand until demonstrated otherwise)

---

### 3. Internal Consistency (`internal_consistency`)
**Question**: Does this belief contradict other beliefs I hold?

| Score | Interpretation |
|-------|----------------|
| 1.0 | Positively supports existing beliefs |
| 0.9 | Consistent with all known beliefs |
| 0.7 | No contradictions found (not exhaustively checked) |
| 0.5 | Minor tension with low-confidence beliefs |
| 0.3 | Contradicts medium-confidence beliefs |
| 0.1 | Contradicts high-confidence beliefs |
| 0.0 | Directly contradicts verified knowledge |

**Computation**: Requires semantic comparison against existing belief store. Expensive to compute exhaustively; often sampled or domain-bounded.

**Dynamic**: This score changes as new beliefs are added. A belief that was consistent may become inconsistent.

**Default**: 0.7 (assumed consistent until checked)

---

### 4. Temporal Freshness (`temporal_freshness`)
**Question**: Is this belief still valid, or has it decayed?

| Score | Interpretation |
|-------|----------------|
| 1.0 | Just verified/created |
| 0.8 | Recently valid, within expected refresh period |
| 0.6 | Aging but likely still accurate |
| 0.4 | Past typical validity window |
| 0.2 | Significantly outdated |
| 0.0 | Expired or explicitly invalidated |

**Decay Function**: Freshness decays over time according to domain-specific half-lives:
- **Timeless facts** (math, physics): Half-life = ∞ (no decay)
- **Stable knowledge** (geography, history): Half-life = years
- **Professional info** (job titles, addresses): Half-life = months
- **Current events** (news, prices): Half-life = hours/days
- **Ephemeral state** (location, mood): Half-life = minutes

**Explicit Bounds**: Beliefs may have `valid_from` and `valid_until` timestamps:
- Before `valid_from`: freshness = 0.0
- After `valid_until`: freshness = 0.0
- Between bounds: decay function applies from creation time

**Default**: 1.0 at creation, then decays

---

### 5. Corroboration (`corroboration`)
**Question**: How many independent sources agree on this belief?

| Score | Interpretation |
|-------|----------------|
| 1.0 | Independently verified by 10+ trusted sources |
| 0.8 | Verified by 5-9 independent sources |
| 0.6 | Verified by 2-4 independent sources |
| 0.4 | Single trusted source, not independently verified |
| 0.2 | Single source of unknown reliability |
| 0.0 | Unverified, origin unclear |

**Independence Requirement**: Sources must be genuinely independent:
- Not derived from each other
- Not from the same upstream origin
- Ideally, different methods of derivation

**Weighting**: Corroboration from high-reputation verifiers counts more than from unknowns.

**Formula**: See MATH.md for exact calculation

**Default**: 0.2 (single source until corroborated)

---

### 6. Domain Applicability (`domain_applicability`)
**Question**: In what contexts does this belief hold?

| Score | Interpretation |
|-------|----------------|
| 1.0 | Universal truth, applies everywhere |
| 0.8 | Broadly applicable across many domains |
| 0.6 | Applies within a specific domain (stated) |
| 0.4 | Context-dependent, conditions apply |
| 0.2 | Narrow applicability, edge case |
| 0.0 | Does not apply in current context |

**Domain Tags**: Beliefs carry domain tags (e.g., `["physics", "classical"]`, `["finance", "US", "2024"]`). Applicability is computed by:
1. How well the query context matches the belief's domains
2. How specific vs. general the belief is
3. Whether the belief explicitly excludes certain contexts

**Dynamic**: Like internal_consistency, this is context-dependent. A belief about "US tax law" has 0.0 applicability when querying about UK taxes.

**Default**: 0.6 (assumes stated domain applicability)

---

## Vector Invariants

### Range
All dimensions are in `[0.0, 1.0]`. Values outside this range are clamped.

### Independence
Dimensions are designed to be orthogonal:
- A belief can have high source_reliability but low corroboration (trusted single source)
- A belief can have high corroboration but low temporal_freshness (many agreed... years ago)
- A belief can have high method_quality but low internal_consistency (well-derived but contradictory)

### Completeness
A full vector has all six dimensions. Partial vectors (missing dimensions) are allowed but must be handled explicitly (see MATH.md for defaults and imputation).

### Comparability
Vectors can be compared, aggregated, and ranked. The `overall_confidence()` function produces a scalar for cases where a single number is needed.

---

## Special Values

### Zero Vector
`[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`

Represents complete uncertainty. A belief with a zero vector is effectively "I have no idea." Still stored (epistemic humility is valuable) but weighted accordingly.

### Unit Vector
`[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

Maximum confidence on all dimensions. Reserved for axiomatic beliefs (mathematical identities, logical tautologies). Any claim asserting unit confidence should be treated with extreme skepticism unless it's truly definitional.

### Unknown/Missing
Individual dimensions may be `null` indicating "not yet computed" or "not applicable." This is distinct from 0.0 (computed, and found to be zero confidence).

---

## Confidence vs. Probability

**ConfidenceVectors are NOT probabilities**. They do not sum to 1.0 across competing hypotheses. They represent *meta-level* confidence about a belief, not the belief's likelihood of being true.

Example:
- Belief: "It will rain tomorrow"
- Confidence: `[0.8, 0.7, 0.9, 1.0, 0.6, 0.8]`
- Interpretation: "I'm fairly confident in this belief" (method was weather model, fresh, mostly corroborated)
- This is NOT the same as: "P(rain) = 0.8"

The belief content might itself contain probabilities ("70% chance of rain"), and the confidence vector describes how much we trust that probabilistic claim.

---

## Example Beliefs with Vectors

### High-Confidence Fact
**Content**: "Water boils at 100°C at sea level"
```
source_reliability:     0.95  (physics textbooks)
method_quality:        0.90  (experimental verification)
internal_consistency:  1.00  (supports other thermodynamics)
temporal_freshness:    1.00  (timeless within stated conditions)
corroboration:         1.00  (independently verified countless times)
domain_applicability:  0.80  (specific conditions: sea level, pure water)
```
**Overall**: ~0.94

### Recent Personal Observation
**Content**: "The coffee shop on Main St. was open at 8am today"
```
source_reliability:     0.95  (self - direct observation)
method_quality:        1.00  (direct observation)
internal_consistency:  0.90  (consistent with their posted hours)
temporal_freshness:    0.95  (observed 2 hours ago)
corroboration:         0.20  (only I observed this)
domain_applicability:  0.40  (very specific: today, that location)
```
**Overall**: ~0.68

### Hearsay with Trust Issues
**Content**: "Company X is planning layoffs"
```
source_reliability:     0.40  (friend of a friend who works there)
method_quality:        0.20  (hearsay)
internal_consistency:  0.70  (consistent with their stock price drop)
temporal_freshness:    0.90  (heard yesterday)
corroboration:         0.30  (single chain, but some news hints)
domain_applicability:  0.60  (specific to that company)
```
**Overall**: ~0.47

### Contradictory Claim
**Content**: "The Earth is flat"
```
source_reliability:     0.10  (known unreliable sources)
method_quality:        0.10  (assertion without evidence)
internal_consistency:  0.00  (contradicts verified physics)
temporal_freshness:    1.00  (claim is current)
corroboration:         0.30  (some people agree, but not independent)
domain_applicability:  0.00  (does not apply in reality)
```
**Overall**: ~0.17

---

## Design Rationale

### Why Six Dimensions?

**Too few** (e.g., single score): Loses crucial information. "0.7 confidence" tells you nothing about *why*.

**Too many** (e.g., 20 dimensions): Cognitive overload, hard to populate, sparse vectors, unclear weighting.

**Six is the sweet spot**: 
- Captures the major epistemic axes
- Human-interpretable (you can look at a vector and understand it)
- Computationally tractable
- Maps to existing epistemological frameworks (reliability, validity, recency, consensus)

### Why These Six?

These dimensions were chosen because they're:
1. **Orthogonal**: Each captures something the others don't
2. **Observable**: Can be computed or estimated from available data
3. **Actionable**: Low scores on specific dimensions suggest remedies (need corroboration? seek verification)
4. **Universal**: Apply to any domain, any belief type
5. **Composable**: Can be aggregated across sources meaningfully

### Why Not Bayesian?

Bayesian updating is valuable for P(hypothesis | evidence), but:
1. Requires well-defined hypothesis spaces (beliefs are open-ended)
2. Prior-sensitive in ways that don't travel well across agents
3. Doesn't capture *why* confidence changed
4. Hard to interpret (log-odds aren't human-friendly)

ConfidenceVectors complement Bayesian reasoning—you might use them to weight priors, or to decide whether to trust a posterior estimate.

---

## Next Steps

- **INTERFACE.md**: Operations on confidence vectors
- **MATH.md**: Formulas for aggregation, decay, and comparison
