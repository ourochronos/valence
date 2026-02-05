"""Verification Protocol with Staking for Valence.

This package implements the adversarial verification system where:
- Verifiers stake reputation to validate or challenge beliefs
- Finding contradictions earns higher rewards than confirmations
- Disputes allow challenges to verification results
- Reputation updates based on verification outcomes

Spec references:
- spec/components/verification-protocol/SPEC.md
- spec/components/verification-protocol/INTERFACE.md
- spec/components/verification-protocol/REPUTATION.md
"""

from __future__ import annotations

# Constants
from .constants import ReputationConstants

# Enums
from .enums import (
    ContradictionType,
    DisputeOutcome,
    DisputeStatus,
    DisputeType,
    EvidenceContribution,
    EvidenceType,
    ResolutionMethod,
    StakeType,
    UncertaintyReason,
    VerificationResult,
    VerificationStatus,
)

# Evidence
from .evidence import Evidence, create_evidence

# Models (data classes for evidence sources)
from .models import BeliefReference, DerivationProof, ExternalSource, Observation

# Results
from .results import ResultDetails, Stake

# Verification (main classes, service, and functions)
from .verification import (
    DiscrepancyBounty,
    Dispute,
    ReputationScore,
    ReputationUpdate,
    StakePosition,
    Verification,
    VerificationService,
    calculate_bounty,
    calculate_confirmation_reward,
    calculate_contradiction_reward,
    calculate_dispute_min_stake,
    calculate_holder_confirmation_bonus,
    calculate_holder_contradiction_penalty,
    calculate_max_stake,
    calculate_min_stake,
    calculate_partial_reward,
    validate_dispute_submission,
    validate_evidence_requirements,
    validate_verification_submission,
)

__all__ = [
    # Constants
    "ReputationConstants",
    # Enums
    "VerificationResult",
    "VerificationStatus",
    "StakeType",
    "EvidenceType",
    "EvidenceContribution",
    "ContradictionType",
    "UncertaintyReason",
    "DisputeType",
    "DisputeOutcome",
    "DisputeStatus",
    "ResolutionMethod",
    # Models
    "ExternalSource",
    "BeliefReference",
    "Observation",
    "DerivationProof",
    # Evidence
    "Evidence",
    "create_evidence",
    # Results
    "ResultDetails",
    "Stake",
    # Verification classes
    "Verification",
    "Dispute",
    "ReputationScore",
    "ReputationUpdate",
    "StakePosition",
    "DiscrepancyBounty",
    # Service
    "VerificationService",
    # Functions
    "calculate_min_stake",
    "calculate_max_stake",
    "calculate_dispute_min_stake",
    "calculate_bounty",
    "calculate_confirmation_reward",
    "calculate_contradiction_reward",
    "calculate_holder_confirmation_bonus",
    "calculate_holder_contradiction_penalty",
    "calculate_partial_reward",
    "validate_verification_submission",
    "validate_evidence_requirements",
    "validate_dispute_submission",
]
