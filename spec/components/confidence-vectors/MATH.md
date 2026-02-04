# Confidence Vector System — Mathematics

*"Principled formulas for uncertain combination."*

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{v}$ | Confidence vector |
| $v_i$ | Value of dimension $i$ |
| $w_i$ | Weight for dimension $i$ |
| $n$ | Number of vectors being aggregated |
| $d$ | Number of dimensions (always 6) |
| $t$ | Time elapsed (seconds) |
| $\tau$ | Half-life (seconds) |
| $\mathcal{D}$ | Set of dimension names |

**Dimension indices**:
1. `source_reliability` (SR)
2. `method_quality` (MQ)
3. `internal_consistency` (IC)
4. `temporal_freshness` (TF)
5. `corroboration` (CO)
6. `domain_applicability` (DA)

---

## Overall Confidence

### Weighted Arithmetic Mean (Default)

$$
\text{overall}(\mathbf{v}) = \frac{\sum_{i=1}^{d} w_i \cdot v_i}{\sum_{i=1}^{d} w_i}
$$

With equal weights ($w_i = 1$):
$$
\text{overall}(\mathbf{v}) = \frac{1}{d} \sum_{i=1}^{d} v_i
$$

**Properties**:
- Range: $[0, 1]$
- Linear in each dimension
- A zero in one dimension doesn't kill the score

### Weighted Geometric Mean

$$
\text{overall}_{\text{geo}}(\mathbf{v}) = \left( \prod_{i=1}^{d} v_i^{w_i} \right)^{1/\sum w_i}
$$

**Properties**:
- Penalizes imbalance (low score in any dimension hurts more)
- Zero in any dimension → overall = 0
- Undefined for $v_i = 0$; use floor $\epsilon = 0.001$

### Weighted Harmonic Mean

$$
\text{overall}_{\text{harm}}(\mathbf{v}) = \frac{\sum_{i=1}^{d} w_i}{\sum_{i=1}^{d} \frac{w_i}{v_i}}
$$

**Properties**:
- Most conservative of the means
- Strongly penalizes low outliers
- Undefined for $v_i = 0$; use floor $\epsilon$

### Minimum (Most Conservative)

$$
\text{overall}_{\min}(\mathbf{v}) = \min_{i \in \mathcal{D}} v_i
$$

**Properties**:
- Ignores weights
- Chain is only as strong as weakest link
- Use for safety-critical applications

### Relationship: $\text{harm} \leq \text{geo} \leq \text{arith}$

For any vector, harmonic ≤ geometric ≤ arithmetic. Choose based on how much you want to penalize low dimensions.

---

## Aggregation Formulas

Given $n$ vectors $\mathbf{v}^{(1)}, \mathbf{v}^{(2)}, \ldots, \mathbf{v}^{(n)}$ with source weights $s_1, s_2, \ldots, s_n$.

### Weighted Mean Aggregation (Default)

For each dimension $i$:
$$
\text{agg}_i = \frac{\sum_{j=1}^{n} s_j \cdot v_i^{(j)}}{\sum_{j=1}^{n} s_j}
$$

**Properties**:
- Straightforward averaging
- High-weight sources dominate
- Outliers can skew result

### Pessimistic Aggregation

Add a bias toward lower values (for safety-critical beliefs):

$$
\text{agg}_i^{\text{pess}} = \text{agg}_i \cdot \left(1 - \alpha \cdot \sigma_i \right)
$$

Where $\sigma_i$ is the weighted standard deviation of dimension $i$, and $\alpha \in [0, 1]$ controls pessimism.

With $\alpha = 0.5$ (default):
- High agreement → minimal penalty
- High disagreement → significant reduction

### Optimistic Aggregation

Bias toward higher values (for exploratory queries):

$$
\text{agg}_i^{\text{opt}} = \text{agg}_i + \beta \cdot (1 - \text{agg}_i) \cdot \text{max}_j(v_i^{(j)} - \text{agg}_i)
$$

Where $\beta \in [0, 1]$ controls optimism.

### Minimum Aggregation

$$
\text{agg}_i^{\min} = \min_{j=1}^{n} v_i^{(j)}
$$

Most conservative; use when any doubt should propagate.

### Maximum Aggregation

$$
\text{agg}_i^{\max} = \max_{j=1}^{n} v_i^{(j)}
$$

Most optimistic; use when any support is sufficient.

### Bayesian-Inspired Aggregation

Treat each source as providing evidence; combine using logarithmic pooling:

$$
\text{agg}_i^{\text{bayes}} = \frac{\prod_{j=1}^{n} (v_i^{(j)})^{s_j}}{\prod_{j=1}^{n} (v_i^{(j)})^{s_j} + \prod_{j=1}^{n} (1 - v_i^{(j)})^{s_j}}
$$

**Properties**:
- Unanimous agreement → strong result
- Disagreement → mediocre result
- Handles extreme values better than arithmetic mean

---

## Decay Functions

### Exponential Decay (Default)

$$
v_{\text{TF}}(t) = v_{\text{TF}}(0) \cdot 2^{-t/\tau}
$$

Where:
- $t$ = time since creation (seconds)
- $\tau$ = half-life (seconds)
- $v_{\text{TF}}(0)$ = initial freshness (usually 1.0)

**Properties**:
- After one half-life: 50% remains
- After two half-lives: 25% remains
- Never reaches exactly zero
- Smooth, continuous decay

**With floor**:
$$
v_{\text{TF}}(t) = \max\left(f, v_{\text{TF}}(0) \cdot 2^{-t/\tau}\right)
$$

Where $f$ is the floor value.

### Linear Decay

$$
v_{\text{TF}}(t) = \max\left(f, v_{\text{TF}}(0) \cdot \left(1 - \frac{t}{T}\right)\right)
$$

Where $T$ is the total decay time (reaches floor at $t = T$).

**Properties**:
- Simple to understand
- Predictable rate
- Hard cutoff behavior

### Step Decay

$$
v_{\text{TF}}(t) = \begin{cases}
v_{\text{TF}}(0) & \text{if } t < T_1 \\
v_1 & \text{if } T_1 \leq t < T_2 \\
v_2 & \text{if } T_2 \leq t < T_3 \\
\vdots \\
f & \text{if } t \geq T_n
\end{cases}
$$

**Properties**:
- Discrete freshness levels
- Good for categorical distinctions ("current", "recent", "stale", "expired")
- Easier for humans to reason about

### Half-Life Reference Table

| Domain | Half-Life | After 1 day | After 1 week | After 1 month |
|--------|-----------|-------------|--------------|---------------|
| Timeless | ∞ | 100% | 100% | 100% |
| Stable | 1 year | 99.8% | 98.7% | 94.4% |
| Professional | 90 days | 99.2% | 94.7% | 79.4% |
| Current | 1 week | 90.6% | 50% | 6.3% |
| Ephemeral | 1 hour | 0% | 0% | 0% |

---

## Corroboration Calculation

### Basic Formula

$$
\text{CO} = 1 - \prod_{j=1}^{n} (1 - r_j \cdot i_j)
$$

Where:
- $n$ = number of verifications
- $r_j$ = reputation of verifier $j$
- $i_j$ = independence factor of verification $j$ (0 to 1)

**Intuition**: Each independent verification reduces remaining "doubt."

### Independence Factor

$$
i_j = 1 - \max_{k < j}\left(\text{similarity}(s_j, s_k)\right)
$$

Where $\text{similarity}$ measures how related source $j$ is to prior sources.

**Factors reducing independence**:
- Same upstream source (e.g., both citing same paper)
- Same derivation method
- Same organization/network
- Temporal clustering (everyone reacting to same event)

### Saturation

Corroboration saturates; each additional verification provides diminishing returns:

| Verifications | Max CO (all perfect) |
|--------------|---------------------|
| 1 | 0.4 |
| 2 | 0.64 |
| 3 | 0.78 |
| 5 | 0.92 |
| 10 | 0.99 |

This prevents gaming through sheer volume.

### With Reputation Weighting

$$
\text{CO} = \min\left(1, \frac{\sum_{j=1}^{n} r_j \cdot i_j \cdot c_j}{\theta}\right)
$$

Where:
- $c_j$ = confirmation strength (1 = full confirmation, 0.5 = partial, 0 = contradiction)
- $\theta$ = threshold for maximum corroboration (e.g., 5.0)

---

## Weighting Schemes

### Equal Weights (Default)

$$
w_i = 1 \quad \forall i \in \mathcal{D}
$$

Use when no dimension is more important than others.

### Task-Specific Weights

| Task | SR | MQ | IC | TF | CO | DA |
|------|----|----|----|----|----|----|
| General query | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Decision making | 1.5 | 1.2 | 1.5 | 1.0 | 1.5 | 0.8 |
| Research | 1.0 | 2.0 | 1.5 | 0.8 | 1.0 | 1.0 |
| Current events | 1.0 | 0.8 | 1.0 | 2.0 | 1.5 | 1.0 |
| Safety-critical | 1.5 | 1.5 | 2.0 | 1.0 | 1.5 | 1.0 |

### Learned Weights

Over time, weight priors can be updated based on:
- Belief verification outcomes
- User feedback on result quality
- Domain-specific calibration

$$
w_i^{(t+1)} = w_i^{(t)} + \eta \cdot \nabla_w L(\text{predictions}, \text{outcomes})
$$

Where $L$ is a loss function on calibration quality.

---

## Edge Cases

### Missing Dimensions (Null Values)

**Option 1: Skip and renormalize**
$$
\text{overall}(\mathbf{v}) = \frac{\sum_{i \in \text{present}} w_i \cdot v_i}{\sum_{i \in \text{present}} w_i}
$$

**Option 2: Substitute defaults**
Replace null with dimension-specific default before computation.

**Option 3: Pessimistic null**
Treat null as worst-case for that dimension.

**Recommendation**: Skip and renormalize for aggregation; substitute defaults for overall.

### Zero Confidence

If $v_i = 0$ for any dimension:
- **Arithmetic mean**: Works normally
- **Geometric mean**: Use floor $\epsilon = 0.001$ to avoid zero product
- **Harmonic mean**: Use floor to avoid division by zero

A zero in `internal_consistency` (contradiction detected) should typically propagate strongly—consider using geometric mean or min for safety.

### All Vectors Identical

$$
\text{aggregate}([\mathbf{v}, \mathbf{v}, \ldots, \mathbf{v}]) = \mathbf{v}
$$

Aggregating identical vectors returns that vector (idempotent).

### Single Source

$$
\text{aggregate}([\mathbf{v}]) = \mathbf{v}
$$

Aggregation of single source is identity.

### Conflicting Sources

When sources strongly disagree (high variance on a dimension):

1. **Flag the disagreement**: Return aggregate with metadata about variance
2. **Increase uncertainty**: Apply pessimistic adjustment
3. **Reduce corroboration**: Disagreement is evidence against corroboration

$$
\text{CO}_{\text{adj}} = \text{CO} \cdot (1 - \gamma \cdot \text{disagreement\_ratio})
$$

Where $\gamma \in [0, 1]$ is the disagreement penalty factor.

---

## Calibration

### What is Calibration?

A confidence score is **well-calibrated** if:
- Beliefs with 80% confidence are true ~80% of the time
- Beliefs with 50% confidence are true ~50% of the time

### Calibration Error

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |\text{acc}_b - \text{conf}_b|
$$

Where:
- $B$ = number of confidence bins
- $n_b$ = beliefs in bin $b$
- $\text{acc}_b$ = accuracy in bin $b$ (how often verified)
- $\text{conf}_b$ = average confidence in bin $b$

### Per-Dimension Calibration

Each dimension should be independently calibrated:
- Track verification outcomes by dimension
- Adjust dimension calculations if systematically over/under-confident

### Calibration Adjustment

If dimension $i$ is consistently overconfident by factor $k$:

$$
v_i^{\text{calibrated}} = v_i^k \quad \text{for } k > 1
$$

If underconfident:
$$
v_i^{\text{calibrated}} = 1 - (1 - v_i)^k \quad \text{for } k > 1
$$

---

## Numerical Stability

### Floating Point Considerations

- Store as 64-bit floats internally
- Round to 3 decimal places for display/storage
- Use logarithms for products: $\prod v_i = \exp(\sum \log v_i)$

### Avoiding Underflow

When multiplying many small probabilities:
$$
\log \text{agg}^{\text{bayes}} = \sum s_j \log v - \log\left(\exp\sum s_j \log v + \exp\sum s_j \log(1-v)\right)
$$

Use log-sum-exp trick:
$$
\log(e^a + e^b) = \max(a,b) + \log(1 + e^{-|a-b|})
$$

### Clamping

Always clamp results to $[0, 1]$:
```
v_i = max(0.0, min(1.0, v_i))
```

---

## Computational Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `create_confidence` | O(d) | O(d) |
| `overall_confidence` | O(d) | O(1) |
| `aggregate_confidences` | O(n·d) | O(d) |
| `decay_confidence` | O(d) | O(d) |
| `compare_confidences` | O(d) | O(d) |

Where $n$ = number of vectors, $d$ = 6 dimensions.

All operations are constant in practice (d=6 is fixed).

---

## Summary of Formulas

| Operation | Formula |
|-----------|---------|
| **Overall (default)** | $\frac{\sum w_i v_i}{\sum w_i}$ |
| **Overall (geometric)** | $(\prod v_i^{w_i})^{1/\sum w_i}$ |
| **Aggregate (default)** | $\frac{\sum s_j v_i^{(j)}}{\sum s_j}$ per dimension |
| **Decay** | $v(t) = v(0) \cdot 2^{-t/\tau}$ |
| **Corroboration** | $1 - \prod(1 - r_j \cdot i_j)$ |

---

## References

1. Cooke, R.M. (1991). *Experts in Uncertainty*. Oxford University Press.
2. Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation.
3. Clemen, R.T. & Winkler, R.L. (1999). Combining probability distributions from experts in risk analysis.
4. Baron, J. et al. (2014). Two reasons to make aggregated probability forecasts more extreme.
