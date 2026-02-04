# Valence Consensus Module

This module implements VRF-based validator selection for the Valence consensus mechanism, per the [NODE-SELECTION.md](/spec/components/consensus-mechanism/NODE-SELECTION.md) specification.

## Overview

The consensus module provides:

1. **VRF (Verifiable Random Function)** - Unpredictable but verifiable random selection using Ed25519 keys
2. **Validator Selection** - Stake-weighted lottery with diversity constraints
3. **Anti-Gaming Measures** - Tenure penalties, collusion detection, and diversity scoring

## Quick Start

```python
from valence.consensus import (
    VRF,
    ValidatorSelector,
    ValidatorCandidate,
    ValidatorTier,
    StakeRegistration,
    compute_selection_weight,
    derive_epoch_seed,
)

# Generate a VRF key pair for a validator
vrf = VRF.generate()
print(f"Public key: {vrf.public_key_bytes.hex()}")

# Compute a selection ticket
epoch_seed = derive_epoch_seed(previous_seed, block_hash, epoch_number)
output = vrf.prove(epoch_seed + agent_fingerprint)
print(f"Selection ticket: {output.ticket.hex()}")

# Verify a ticket (anyone can do this)
valid = VRF.verify(public_key_bytes, input_data, output)
```

## Components

### VRF (`vrf.py`)

The VRF implementation uses Ed25519 signatures to create verifiable random outputs:

```python
from valence.consensus import VRF, VRFOutput

# Generate new key pair
vrf = VRF.generate()

# Or load from existing private key
vrf = VRF(private_key_bytes=stored_key)

# Compute VRF output
output = vrf.prove(b"input_data")

# Verify output (no private key needed)
valid = VRF.verify(public_key_bytes, b"input_data", output)

# Epoch seed derivation
seed = VRF.derive_epoch_seed(prev_seed, block_hash, epoch_num)
```

### Validator Selection (`selection.py`)

The selection algorithm implements stake-weighted lottery with diversity constraints:

```python
from valence.consensus import (
    ValidatorSelector,
    ValidatorCandidate,
    StakeRegistration,
    ValidatorTier,
)
from datetime import datetime
from uuid import uuid4

# Create a selector
selector = ValidatorSelector()

# Check eligibility
eligible, reasons = selector.check_eligibility(
    reputation=0.7,
    account_age_days=200,
    verification_count=100,
    uphold_rate=0.85,
    attestation_count=1,
    active_slashing=False,
)

# Create candidate
stake = StakeRegistration(
    id=uuid4(),
    agent_id="did:vkb:key:z6MkExample",
    amount=0.20,
    tier=ValidatorTier.STANDARD,
    registered_at=datetime.now(),
    eligible_from_epoch=42,
)

candidate = ValidatorCandidate(
    agent_id=stake.agent_id,
    public_key=vrf.public_key_bytes,
    stake=stake,
    reputation=0.7,
)

# Select validators for epoch
validator_set = selector.select_for_epoch(
    candidates=[candidate, ...],
    epoch_seed=seed,
    epoch_number=42,
)

print(f"Selected {validator_set.validator_count} validators")
print(f"Quorum threshold: {validator_set.quorum_threshold}")
```

### Selection Weight Calculation

Selection probability is based on multiple factors:

```python
from valence.consensus import compute_selection_weight

# Weight factors:
# - Base: tier multiplier (Standard=1.0, Enhanced=1.5, Guardian=2.0)
# - Reputation bonus: up to 1.25× at reputation=1.0
# - Attestation bonus: up to 1.3× with 3+ attestations
# - Tenure penalty: 0.9^(n-4) after 4 consecutive epochs
# - Performance factor: 0.9-1.1× based on last epoch

weight = compute_selection_weight(candidate)
```

### Anti-Gaming Measures (`anti_gaming.py`)

Detect and prevent consensus capture attempts:

```python
from valence.consensus import (
    AntiGamingEngine,
    compute_tenure_penalty,
    compute_diversity_score,
)

# Tenure penalty
penalty = compute_tenure_penalty(12)  # 0.43 after 12 epochs

# Diversity analysis
engine = AntiGamingEngine()
scores = engine.compute_diversity_score(validator_set)
print(f"Diversity score: {scores['overall_score']:.2f}")

# Full analysis with collusion detection
analysis = engine.analyze_validator_set(
    validator_set=validator_set,
    voting_records=records,
    stake_registrations=registrations,
)

if analysis['alerts']:
    print(f"Found {len(analysis['alerts'])} potential issues")
```

## Validator Tiers

| Tier | Min Stake | Max Stake | Weight Multiplier |
|------|-----------|-----------|-------------------|
| Standard | 0.10 | 0.30 | 1.0× |
| Enhanced | 0.30 | 0.50 | 1.5× |
| Guardian | 0.50 | 0.80 | 2.0× |

## Eligibility Requirements

| Requirement | Threshold |
|-------------|-----------|
| Reputation | ≥ 0.5 |
| Account Age | ≥ 180 days |
| Verifications | ≥ 50 |
| Uphold Rate | ≥ 70% |
| Attestation | At least 1 valid |
| No Active Slashing | Required |

## Diversity Constraints

- **Federation limit**: Max 20% from any single federation
- **Returning validators**: Max 60% from previous epoch
- **New validators**: Min 20% must be new
- **Tier diversity**: Min 30% from standard tier

## Slashing Conditions

| Offense | Severity | Slash |
|---------|----------|-------|
| Double-voting | CRITICAL | 100% |
| Equivocation | CRITICAL | 100% |
| Collusion | CRITICAL | 100% |
| Unavailability | HIGH | 50% |
| Censorship | HIGH | 50% |
| Invalid vote | MEDIUM | 20% |
| Late voting | LOW | 5% |

## Security Properties

1. **Unpredictability**: VRF outputs cannot be predicted without the private key
2. **Verifiability**: Anyone can verify a VRF output given the public key
3. **Uniqueness**: Each input produces exactly one valid output per key
4. **Anti-Sybil**: Identity attestations required for eligibility
5. **Anti-Entrenchment**: Tenure penalties reduce long-term validator advantage

## Tests

Run the test suite:

```bash
cd /path/to/valence
python -m pytest tests/consensus/ -v
```

93 tests covering:
- VRF generation and verification
- Selection weight calculation
- Diversity constraint enforcement
- Collusion detection
- Anti-gaming measures
