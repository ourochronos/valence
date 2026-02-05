"""Validation functions for the verification protocol.

Contains all validation logic for verification submissions,
evidence requirements, and dispute submissions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .constants import ReputationConstants
from .enums import (
    VerificationResult,
    VerificationStatus,
    EvidenceContribution,
)
from .models import (
    Verification,
    Dispute,
    Evidence,
    ReputationScore,
)
from .reputation import (
    calculate_min_stake,
    calculate_max_stake,
    calculate_dispute_min_stake,
)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_verification_submission(
    verification: Verification,
    belief: dict[str, Any],  # Simplified belief info
    verifier_reputation: ReputationScore,
    existing_verifications: list[Verification],
) -> list[str]:
    """Validate a verification submission.
    
    Returns list of validation errors (empty if valid).
    
    Checks:
    1. Verifier ≠ holder (self-verification)
    2. No duplicate verification by same verifier
    3. Evidence meets minimum requirements
    4. Stake meets minimum
    5. Verifier has sufficient reputation
    """
    errors = []
    
    # Self-verification check
    if verification.verifier_id == verification.holder_id:
        errors.append("Cannot verify your own belief")
    
    # Duplicate check
    for existing in existing_verifications:
        if existing.verifier_id == verification.verifier_id:
            errors.append("Already verified this belief")
            break
    
    # Evidence requirements
    evidence_errors = validate_evidence_requirements(
        verification.result,
        verification.evidence,
    )
    errors.extend(evidence_errors)
    
    # Stake requirements
    belief_confidence = belief.get("confidence", {}).get("overall", 0.7)
    min_stake = calculate_min_stake(
        belief_confidence,
        verifier_reputation.by_domain.get(belief.get("domain_path", ["general"])[0], 0.5),
    )
    
    if verification.stake.amount < min_stake:
        errors.append(f"Stake {verification.stake.amount:.4f} below minimum {min_stake:.4f}")
    
    max_stake = calculate_max_stake(verifier_reputation.overall)
    if verification.stake.amount > max_stake:
        errors.append(f"Stake {verification.stake.amount:.4f} exceeds maximum {max_stake:.4f}")
    
    # Available reputation check
    if verification.stake.amount > verifier_reputation.available_stake():
        errors.append("Insufficient available reputation for stake")
    
    return errors


def validate_evidence_requirements(
    result: VerificationResult,
    evidence: list[Evidence],
) -> list[str]:
    """Validate evidence meets minimum requirements for result type.
    
    Requirements:
    - CONFIRMED: 1 supporting evidence
    - CONTRADICTED: 1 contradicting evidence
    - UNCERTAIN: 0 (but must explain why)
    - PARTIAL: 1 supporting + 1 contradicting
    """
    errors = []
    
    supporting = [e for e in evidence if e.contribution == EvidenceContribution.SUPPORTS]
    contradicting = [e for e in evidence if e.contribution == EvidenceContribution.CONTRADICTS]
    
    if result == VerificationResult.CONFIRMED:
        if len(supporting) < 1:
            errors.append("CONFIRMED requires at least 1 supporting evidence")
    
    elif result == VerificationResult.CONTRADICTED:
        if len(contradicting) < 1:
            errors.append("CONTRADICTED requires at least 1 contradicting evidence")
    
    elif result == VerificationResult.PARTIAL:
        if len(supporting) < 1:
            errors.append("PARTIAL requires at least 1 supporting evidence")
        if len(contradicting) < 1:
            errors.append("PARTIAL requires at least 1 contradicting evidence")
    
    # UNCERTAIN has no minimum (but reasoning is expected)
    
    # Check for invalid evidence
    for e in evidence:
        if e.relevance < 0.1:
            errors.append(f"Evidence {e.id} has very low relevance ({e.relevance})")
    
    return errors


def validate_dispute_submission(
    dispute: Dispute,
    verification: Verification,
    disputer_reputation: ReputationScore,
    is_holder: bool,
) -> list[str]:
    """Validate a dispute submission.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    
    # Verification status check
    if verification.status != VerificationStatus.ACCEPTED:
        errors.append(f"Cannot dispute verification in {verification.status.value} status")
    
    # Dispute window check (7 days from acceptance)
    if verification.accepted_at:
        dispute_deadline = verification.accepted_at + timedelta(days=ReputationConstants.DISPUTE_WINDOW_DAYS)
        if datetime.now() > dispute_deadline:
            errors.append("Dispute window has expired")
    
    # Counter-evidence required
    if not dispute.counter_evidence:
        errors.append("Dispute requires counter-evidence")
    
    # Stake requirements
    min_stake = calculate_dispute_min_stake(verification.stake.amount, is_holder)
    if dispute.stake.amount < min_stake:
        errors.append(f"Dispute stake {dispute.stake.amount:.4f} below minimum {min_stake:.4f}")
    
    # Available reputation check
    if dispute.stake.amount > disputer_reputation.available_stake():
        errors.append("Insufficient available reputation for dispute stake")
    
    return errors
