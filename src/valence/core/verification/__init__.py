"""Verification Protocol with Staking for Valence.

This package implements the adversarial verification system where:
- Verifiers stake reputation to validate or challenge beliefs
- Finding contradictions earns higher rewards than confirmations
- Disputes allow challenges to verification results
- Reputation updates based on verification outcomes

Submodules:
- constants: Configuration constants for reputation calculations
- enums: Enumeration types for verification states
- models: Data models for evidence, verifications, disputes
- reputation: Stake and reward calculation functions
- validators: Validation functions for submissions
- service: VerificationService class for managing verifications

Spec references:
- spec/components/verification-protocol/SPEC.md
- spec/components/verification-protocol/INTERFACE.md
- spec/components/verification-protocol/REPUTATION.md
"""

# Re-export everything for backward compatibility
from .constants import ReputationConstants

from .enums import (
    VerificationResult,
    VerificationStatus,
    StakeType,
    EvidenceType,
    EvidenceContribution,
    ContradictionType,
    UncertaintyReason,
    DisputeType,
    DisputeOutcome,
    DisputeStatus,
    ResolutionMethod,
)

from .models import (
    ExternalSource,
    BeliefReference,
    Observation,
    DerivationProof,
    Evidence,
    ResultDetails,
    Stake,
    Verification,
    Dispute,
    ReputationScore,
    ReputationUpdate,
    StakePosition,
    DiscrepancyBounty,
)

from .reputation import (
    calculate_min_stake,
    calculate_max_stake,
    calculate_dispute_min_stake,
    calculate_bounty,
    calculate_confirmation_reward,
    calculate_contradiction_reward,
    calculate_holder_confirmation_bonus,
    calculate_holder_contradiction_penalty,
    calculate_partial_reward,
)

from .validators import (
    validate_verification_submission,
    validate_evidence_requirements,
    validate_dispute_submission,
)

from .service import (
    VerificationService,
    create_evidence,
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
    # Evidence models
    "ExternalSource",
    "BeliefReference",
    "Observation",
    "DerivationProof",
    "Evidence",
    "ResultDetails",
    # Core models
    "Stake",
    "Verification",
    "Dispute",
    # Reputation models
    "ReputationScore",
    "ReputationUpdate",
    "StakePosition",
    "DiscrepancyBounty",
    # Calculation functions
    "calculate_min_stake",
    "calculate_max_stake",
    "calculate_dispute_min_stake",
    "calculate_bounty",
    "calculate_confirmation_reward",
    "calculate_contradiction_reward",
    "calculate_holder_confirmation_bonus",
    "calculate_holder_contradiction_penalty",
    "calculate_partial_reward",
    # Validation functions
    "validate_verification_submission",
    "validate_evidence_requirements",
    "validate_dispute_submission",
    # Service
    "VerificationService",
    "create_evidence",
]
