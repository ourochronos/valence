# Confidence Vector System — Interface

*"Operations that preserve epistemic meaning."*

---

## Type Definitions

```typescript
// Core vector type
interface ConfidenceVector {
  source_reliability: number | null;     // [0.0, 1.0]
  method_quality: number | null;         // [0.0, 1.0]
  internal_consistency: number | null;   // [0.0, 1.0]
  temporal_freshness: number | null;     // [0.0, 1.0]
  corroboration: number | null;          // [0.0, 1.0]
  domain_applicability: number | null;   // [0.0, 1.0]
}

// Dimension names for iteration
type DimensionName = 
  | 'source_reliability'
  | 'method_quality'
  | 'internal_consistency'
  | 'temporal_freshness'
  | 'corroboration'
  | 'domain_applicability';

// Weights for aggregation/overall calculation
interface DimensionWeights {
  source_reliability: number;
  method_quality: number;
  internal_consistency: number;
  temporal_freshness: number;
  corroboration: number;
  domain_applicability: number;
}

// Comparison result
interface ComparisonResult {
  overall: 'v1_higher' | 'v2_higher' | 'equal' | 'incomparable';
  difference: number;              // Signed: positive = v1 higher
  by_dimension: {
    [K in DimensionName]: {
      comparison: 'v1_higher' | 'v2_higher' | 'equal' | 'incomparable';
      difference: number;
    };
  };
  dominant_dimension: DimensionName | null;  // Largest difference
}

// Decay configuration
interface DecayConfig {
  half_life_seconds: number;       // Time for 50% decay
  floor: number;                   // Minimum value (default: 0.0)
  curve: 'exponential' | 'linear' | 'step';  // Decay shape
}

// Domain-specific decay presets
interface DomainDecayPresets {
  timeless: DecayConfig;           // Mathematical truths
  stable: DecayConfig;             // Geography, history
  professional: DecayConfig;       // Job titles, company info
  current: DecayConfig;            // News, prices
  ephemeral: DecayConfig;          // Location, mood, weather
}

// Aggregation mode
type AggregationMode = 
  | 'weighted_mean'      // Standard weighted average
  | 'pessimistic'        // Weighted mean biased toward lower values
  | 'optimistic'         // Weighted mean biased toward higher values
  | 'min'                // Take minimum across sources
  | 'max'                // Take maximum across sources
  | 'bayesian';          // Bayesian-inspired combination
```

---

## Core Operations

### `create_confidence(dimensions)`

Create a new ConfidenceVector from partial or full dimension values.

**Signature**:
```typescript
function create_confidence(
  dimensions: Partial<ConfidenceVector>,
  options?: {
    fill_defaults?: boolean;        // Fill missing with defaults (default: true)
    validate?: boolean;             // Throw on invalid values (default: true)
  }
): ConfidenceVector;
```

**Behavior**:
1. Validates each provided dimension is in [0.0, 1.0]
2. If `fill_defaults` is true, missing dimensions get default values:
   - `source_reliability`: 0.5
   - `method_quality`: 0.3
   - `internal_consistency`: 0.7
   - `temporal_freshness`: 1.0
   - `corroboration`: 0.2
   - `domain_applicability`: 0.6
3. If `fill_defaults` is false, missing dimensions remain `null`

**Examples**:
```typescript
// Full vector
create_confidence({
  source_reliability: 0.8,
  method_quality: 0.9,
  internal_consistency: 0.95,
  temporal_freshness: 1.0,
  corroboration: 0.7,
  domain_applicability: 0.8
});

// Partial vector with defaults
create_confidence({
  source_reliability: 0.8,
  method_quality: 0.6
});
// → { source_reliability: 0.8, method_quality: 0.6, 
//    internal_consistency: 0.7, temporal_freshness: 1.0,
//    corroboration: 0.2, domain_applicability: 0.6 }

// Partial vector without defaults
create_confidence(
  { source_reliability: 0.8 },
  { fill_defaults: false }
);
// → { source_reliability: 0.8, method_quality: null, ... }
```

**Errors**:
- `InvalidDimensionError`: Value outside [0.0, 1.0]
- `UnknownDimensionError`: Unrecognized dimension name

---

### `aggregate_confidences(vectors, weights)`

Combine multiple confidence vectors into a single aggregate vector.

**Signature**:
```typescript
function aggregate_confidences(
  vectors: ConfidenceVector[],
  weights?: number[],
  options?: {
    mode?: AggregationMode;         // Default: 'weighted_mean'
    dimension_weights?: DimensionWeights;  // Per-dimension multipliers
    null_handling?: 'skip' | 'default' | 'fail';  // Default: 'skip'
  }
): ConfidenceVector;
```

**Behavior**:
1. **Weight normalization**: If `weights` provided, normalize to sum to 1.0. If not provided, equal weights.
2. **Null handling**:
   - `skip`: Ignore vectors with null for that dimension (renormalize weights)
   - `default`: Replace null with dimension default before aggregation
   - `fail`: Throw if any vector has null dimensions
3. **Mode application**: See MATH.md for formulas
4. **Dimension weights**: Optional multipliers applied after aggregation

**Examples**:
```typescript
// Equal-weight aggregation
aggregate_confidences([
  { source_reliability: 0.8, method_quality: 0.9, ... },
  { source_reliability: 0.6, method_quality: 0.7, ... }
]);
// → { source_reliability: 0.7, method_quality: 0.8, ... }

// Weighted by source reputation
aggregate_confidences(
  [highRepVector, lowRepVector],
  [0.8, 0.2]  // High-rep source gets 4x weight
);

// Pessimistic aggregation (safety-critical)
aggregate_confidences(vectors, undefined, { mode: 'pessimistic' });
```

**Special Cases**:
- Empty vector array → throws `EmptyAggregationError`
- Single vector → returns copy of that vector
- All weights zero → throws `ZeroWeightError`

---

### `decay_confidence(vector, time_delta)`

Apply temporal decay to a confidence vector's `temporal_freshness` dimension.

**Signature**:
```typescript
function decay_confidence(
  vector: ConfidenceVector,
  time_delta: number,              // Seconds since creation/last update
  options?: {
    config?: DecayConfig;          // Custom decay parameters
    preset?: keyof DomainDecayPresets;  // Use preset ('timeless', 'stable', etc.)
    decay_all?: boolean;           // Apply decay to ALL dimensions (default: false)
  }
): ConfidenceVector;
```

**Behavior**:
1. If `preset` specified, load from `DOMAIN_DECAY_PRESETS`
2. If `config` specified, use that
3. If neither, use default (stable knowledge preset)
4. Apply decay formula to `temporal_freshness` (or all dimensions if `decay_all`)
5. Clamp result to `[floor, original_value]`

**Presets**:
```typescript
const DOMAIN_DECAY_PRESETS: DomainDecayPresets = {
  timeless: { 
    half_life_seconds: Infinity, 
    floor: 1.0, 
    curve: 'exponential' 
  },
  stable: { 
    half_life_seconds: 365 * 24 * 3600,  // 1 year
    floor: 0.1, 
    curve: 'exponential' 
  },
  professional: { 
    half_life_seconds: 90 * 24 * 3600,   // 90 days
    floor: 0.1, 
    curve: 'exponential' 
  },
  current: { 
    half_life_seconds: 7 * 24 * 3600,    // 1 week
    floor: 0.0, 
    curve: 'exponential' 
  },
  ephemeral: { 
    half_life_seconds: 3600,             // 1 hour
    floor: 0.0, 
    curve: 'exponential' 
  }
};
```

**Examples**:
```typescript
// Default decay after 1 day
const fresh = create_confidence({ temporal_freshness: 1.0 });
decay_confidence(fresh, 86400);  // 24 hours
// → temporal_freshness ≈ 0.998 (stable preset, 1-year half-life)

// Ephemeral belief after 2 hours
decay_confidence(fresh, 7200, { preset: 'ephemeral' });
// → temporal_freshness ≈ 0.25 (1-hour half-life)

// Custom decay
decay_confidence(fresh, 3600, { 
  config: { half_life_seconds: 1800, floor: 0.2, curve: 'exponential' }
});
// → temporal_freshness ≈ 0.5 (halved after 30 minutes)
```

---

### `compare_confidences(v1, v2)`

Compare two confidence vectors and determine which is "more confident."

**Signature**:
```typescript
function compare_confidences(
  v1: ConfidenceVector,
  v2: ConfidenceVector,
  options?: {
    weights?: DimensionWeights;    // Dimension importance weights
    threshold?: number;            // Difference below this → 'equal' (default: 0.01)
  }
): ComparisonResult;
```

**Behavior**:
1. Compute `overall_confidence()` for both vectors using provided weights
2. Compare overall scores
3. Compare each dimension individually
4. Identify the dimension with the largest difference
5. Return structured comparison result

**Examples**:
```typescript
compare_confidences(
  { source_reliability: 0.9, method_quality: 0.8, ... },
  { source_reliability: 0.6, method_quality: 0.9, ... }
);
// → {
//     overall: 'v1_higher',
//     difference: 0.12,
//     by_dimension: {
//       source_reliability: { comparison: 'v1_higher', difference: 0.3 },
//       method_quality: { comparison: 'v2_higher', difference: -0.1 },
//       ...
//     },
//     dominant_dimension: 'source_reliability'
//   }
```

**Incomparability**:
Two vectors are `incomparable` if:
- Either has too many null dimensions to compute overall
- The comparison weights make comparison undefined

---

### `overall_confidence(vector)`

Reduce a confidence vector to a single scalar score.

**Signature**:
```typescript
function overall_confidence(
  vector: ConfidenceVector,
  options?: {
    weights?: DimensionWeights;    // Custom weights (default: equal)
    mode?: 'weighted_mean' | 'geometric_mean' | 'harmonic_mean' | 'min';
    null_handling?: 'skip' | 'default' | 'zero';
  }
): number;
```

**Default Weights**:
```typescript
const DEFAULT_OVERALL_WEIGHTS: DimensionWeights = {
  source_reliability: 1.0,
  method_quality: 1.0,
  internal_consistency: 1.0,
  temporal_freshness: 1.0,
  corroboration: 1.0,
  domain_applicability: 1.0
};
```

**Modes**:
- `weighted_mean`: Standard weighted average (default)
- `geometric_mean`: Emphasizes balance; low dimension hurts more
- `harmonic_mean`: Penalizes outliers more severely
- `min`: Most conservative; overall = lowest dimension

**Examples**:
```typescript
const v = {
  source_reliability: 0.8,
  method_quality: 0.9,
  internal_consistency: 0.95,
  temporal_freshness: 1.0,
  corroboration: 0.7,
  domain_applicability: 0.8
};

overall_confidence(v);  // ≈ 0.858 (weighted mean)
overall_confidence(v, { mode: 'geometric_mean' });  // ≈ 0.851
overall_confidence(v, { mode: 'min' });  // 0.7 (corroboration)

// Custom weights emphasizing corroboration
overall_confidence(v, { 
  weights: { ...DEFAULT_WEIGHTS, corroboration: 3.0 }
});  // ≈ 0.80 (corroboration weighs more)
```

---

## Utility Operations

### `validate_vector(vector)`

Check that a confidence vector is well-formed.

```typescript
function validate_vector(
  vector: ConfidenceVector
): { valid: boolean; errors: string[] };
```

### `normalize_vector(vector)`

Clamp all dimensions to [0.0, 1.0] and replace NaN/undefined with null.

```typescript
function normalize_vector(vector: ConfidenceVector): ConfidenceVector;
```

### `fill_defaults(vector)`

Replace null dimensions with default values.

```typescript
function fill_defaults(vector: Partial<ConfidenceVector>): ConfidenceVector;
```

### `vector_to_array(vector)`

Convert to array format for mathematical operations.

```typescript
function vector_to_array(vector: ConfidenceVector): (number | null)[];
// Order: [source_reliability, method_quality, internal_consistency, 
//         temporal_freshness, corroboration, domain_applicability]
```

### `array_to_vector(array)`

Convert array back to named vector.

```typescript
function array_to_vector(array: (number | null)[]): ConfidenceVector;
```

### `clone_vector(vector)`

Deep copy a vector.

```typescript
function clone_vector(vector: ConfidenceVector): ConfidenceVector;
```

---

## Domain-Specific Operations

### `compute_internal_consistency(belief, belief_store)`

Compute the internal_consistency dimension by checking against existing beliefs.

```typescript
function compute_internal_consistency(
  belief: Belief,
  belief_store: BeliefStore,
  options?: {
    max_comparisons?: number;      // Limit for performance
    domain_filter?: string[];      // Only check within domains
    threshold?: number;            // Semantic similarity threshold
  }
): number;
```

**Note**: This is expensive. Cache results and recompute incrementally.

---

### `compute_corroboration(belief, verifications)`

Compute the corroboration dimension from verification records.

```typescript
function compute_corroboration(
  belief: Belief,
  verifications: Verification[],
  options?: {
    independence_threshold?: number;  // How different sources must be
    reputation_weighted?: boolean;    // Weight by verifier reputation
  }
): number;
```

---

### `compute_domain_applicability(belief, query_context)`

Compute how applicable a belief is to a specific query context.

```typescript
function compute_domain_applicability(
  belief: Belief,
  query_context: {
    domains: string[];
    constraints?: Record<string, any>;
  }
): number;
```

---

## Serialization

### JSON Format

```json
{
  "source_reliability": 0.8,
  "method_quality": 0.9,
  "internal_consistency": 0.95,
  "temporal_freshness": 1.0,
  "corroboration": 0.7,
  "domain_applicability": 0.8
}
```

Null values serialized as JSON `null`.

### Compact Array Format

For storage efficiency:
```json
[0.8, 0.9, 0.95, 1.0, 0.7, 0.8]
```

Order is fixed (as in `vector_to_array`).

### Binary Format

For network transmission:
- 6 × float32 = 24 bytes
- Null encoded as NaN

---

## Error Types

```typescript
class InvalidDimensionError extends Error {
  dimension: DimensionName;
  value: number;
  message: string;  // "Dimension 'source_reliability' value 1.5 out of range [0.0, 1.0]"
}

class UnknownDimensionError extends Error {
  dimension: string;
}

class EmptyAggregationError extends Error {
  message: "Cannot aggregate empty vector array";
}

class ZeroWeightError extends Error {
  message: "All weights are zero; cannot normalize";
}

class IncomparableVectorsError extends Error {
  reason: string;  // "Too many null dimensions"
}
```

---

## Usage Patterns

### Creating a Belief with Confidence

```typescript
const belief = {
  content: "The meeting is at 3pm",
  confidence: create_confidence({
    source_reliability: 0.9,   // Direct from calendar
    method_quality: 1.0,       // System data
    temporal_freshness: 1.0,   // Just checked
    corroboration: 0.4,        // Only one source
    domain_applicability: 0.3  // Very specific context
  })
};
```

### Aggregating Multiple Sources

```typescript
const sources = [
  { vector: v1, reputation: 0.9 },
  { vector: v2, reputation: 0.7 },
  { vector: v3, reputation: 0.3 }
];

const aggregated = aggregate_confidences(
  sources.map(s => s.vector),
  sources.map(s => s.reputation),  // Weight by reputation
  { mode: 'weighted_mean' }
);
```

### Querying with Confidence Threshold

```typescript
const results = await belief_store.query("project deadlines");
const confident_results = results.filter(
  r => overall_confidence(r.confidence) >= 0.6
);
```

### Decaying Stale Beliefs

```typescript
const now = Date.now() / 1000;
const decayed_beliefs = beliefs.map(b => ({
  ...b,
  confidence: decay_confidence(
    b.confidence,
    now - b.created_at,
    { preset: b.decay_preset }
  )
}));
```

---

## Implementation Notes

1. **Immutability**: All operations return new vectors; never mutate inputs
2. **Precision**: Use 64-bit floats internally; round to 3 decimal places for display
3. **Thread Safety**: All operations are pure functions; safe for concurrent use
4. **Performance**: Aggregation is O(n × d) where n = vectors, d = dimensions
5. **Caching**: `overall_confidence` results should be cached when vector is stable
