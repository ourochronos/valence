"""Verification service for managing verifications and disputes.

Contains the main VerificationService class that provides the primary
interface for the verification protocol.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from ..exceptions import ValidationException, NotFoundError
from .constants import ReputationConstants
from .enums import (
    VerificationResult,
    VerificationStatus,
    StakeType,
    DisputeType,
    DisputeOutcome,
    DisputeStatus,
    ResolutionMethod,
    EvidenceType,
    EvidenceContribution,
)
from .models import (
    Verification,
    Dispute,
    Evidence,
    Stake,
    ReputationScore,
    ReputationUpdate,
    StakePosition,
    DiscrepancyBounty,
    ExternalSource,
    BeliefReference,
    Observation,
    DerivationProof,
    ResultDetails,
)
from .reputation import (
    calculate_min_stake,
    calculate_confirmation_reward,
    calculate_contradiction_reward,
    calculate_holder_confirmation_bonus,
    calculate_holder_contradiction_penalty,
    calculate_partial_reward,
)
from .validators import (
    validate_verification_submission,
    validate_dispute_submission,
)

logger = logging.getLogger(__name__)


class VerificationService:
    """Service for managing verifications and disputes.
    
    This is the main interface for the verification protocol.
    In production, this would use database operations via get_cursor().
    """
    
    def __init__(self):
        # In-memory storage for testing
        self._verifications: dict[UUID, Verification] = {}
        self._disputes: dict[UUID, Dispute] = {}
        self._reputations: dict[str, ReputationScore] = {}
        self._stake_positions: dict[UUID, StakePosition] = {}
        self._reputation_events: list[ReputationUpdate] = []
        self._bounties: dict[UUID, DiscrepancyBounty] = {}
    
    def get_or_create_reputation(self, identity_id: str) -> ReputationScore:
        """Get or create reputation score for an identity."""
        if identity_id not in self._reputations:
            self._reputations[identity_id] = ReputationScore(identity_id=identity_id)
        return self._reputations[identity_id]
    
    def submit_verification(
        self,
        belief_id: UUID,
        belief_info: dict[str, Any],
        verifier_id: str,
        result: VerificationResult,
        evidence: list[Evidence],
        stake_amount: float,
        reasoning: str | None = None,
        result_details: ResultDetails | None = None,
    ) -> Verification:
        """Submit a new verification for a belief.
        
        Args:
            belief_id: UUID of the belief being verified
            belief_info: Dictionary with belief details (confidence, holder_id, domain_path)
            verifier_id: DID of the verifier
            result: Verification result
            evidence: List of evidence supporting the verification
            stake_amount: Amount to stake
            reasoning: Optional explanation
            result_details: Optional structured details
        
        Returns:
            The created Verification
        
        Raises:
            ValidationException: If validation fails
        """
        holder_id = belief_info.get("holder_id", "unknown")
        
        # Create stake
        stake = Stake(
            amount=stake_amount,
            type=StakeType.STANDARD,
            locked_until=datetime.now() + timedelta(days=ReputationConstants.STAKE_LOCKUP_DAYS),
            escrow_id=uuid4(),
        )
        
        # Create verification
        verification = Verification(
            id=uuid4(),
            verifier_id=verifier_id,
            belief_id=belief_id,
            holder_id=holder_id,
            result=result,
            evidence=evidence,
            stake=stake,
            reasoning=reasoning,
            result_details=result_details,
            status=VerificationStatus.PENDING,
        )
        
        # Get reputation
        verifier_rep = self.get_or_create_reputation(verifier_id)
        
        # Get existing verifications for this belief
        existing = [v for v in self._verifications.values() if v.belief_id == belief_id]
        
        # Validate
        errors = validate_verification_submission(
            verification, belief_info, verifier_rep, existing
        )
        
        if errors:
            raise ValidationException("; ".join(errors))
        
        # Lock stake
        self._lock_stake(verifier_rep, stake_amount, verification.id)
        
        # Store verification
        self._verifications[verification.id] = verification
        
        logger.info(f"Verification {verification.id} submitted by {verifier_id} for belief {belief_id}")
        
        return verification
    
    def accept_verification(self, verification_id: UUID) -> Verification:
        """Accept a pending verification after validation window.
        
        This would typically be called by a background job after ACCEPTANCE_DELAY_HOURS.
        """
        verification = self._verifications.get(verification_id)
        if not verification:
            raise NotFoundError("Verification", str(verification_id))
        
        if verification.status != VerificationStatus.PENDING:
            raise ValidationException(f"Verification is not pending: {verification.status.value}")
        
        verification.status = VerificationStatus.ACCEPTED
        verification.accepted_at = datetime.now()
        
        # Process reputation updates
        self._process_verification_reputation(verification)
        
        logger.info(f"Verification {verification_id} accepted")
        
        return verification
    
    def dispute_verification(
        self,
        verification_id: UUID,
        disputer_id: str,
        counter_evidence: list[Evidence],
        stake_amount: float,
        dispute_type: DisputeType,
        reasoning: str,
        proposed_result: VerificationResult | None = None,
    ) -> Dispute:
        """Submit a dispute against a verification.
        
        Args:
            verification_id: UUID of the verification to dispute
            disputer_id: DID of the disputer
            counter_evidence: Evidence challenging the verification
            stake_amount: Amount to stake on dispute
            dispute_type: Type of dispute
            reasoning: Explanation for the dispute
            proposed_result: What the result should have been
        
        Returns:
            The created Dispute
        
        Raises:
            NotFoundError: If verification doesn't exist
            ValidationException: If validation fails
        """
        verification = self._verifications.get(verification_id)
        if not verification:
            raise NotFoundError("Verification", str(verification_id))
        
        # Create stake
        stake = Stake(
            amount=stake_amount,
            type=StakeType.CHALLENGE,
            locked_until=datetime.now() + timedelta(days=ReputationConstants.RESOLUTION_TIMEOUT_DAYS),
            escrow_id=uuid4(),
        )
        
        # Create dispute
        dispute = Dispute(
            id=uuid4(),
            verification_id=verification_id,
            disputer_id=disputer_id,
            counter_evidence=counter_evidence,
            stake=stake,
            dispute_type=dispute_type,
            reasoning=reasoning,
            proposed_result=proposed_result,
        )
        
        # Get reputation
        disputer_rep = self.get_or_create_reputation(disputer_id)
        is_holder = disputer_id == verification.holder_id
        
        # Validate
        errors = validate_dispute_submission(dispute, verification, disputer_rep, is_holder)
        
        if errors:
            raise ValidationException("; ".join(errors))
        
        # Lock stake
        self._lock_stake(disputer_rep, stake_amount, None, dispute.id)
        
        # Update verification status
        verification.status = VerificationStatus.DISPUTED
        verification.dispute_id = dispute.id
        
        # Store dispute
        self._disputes[dispute.id] = dispute
        
        logger.info(f"Dispute {dispute.id} submitted by {disputer_id} for verification {verification_id}")
        
        return dispute
    
    def resolve_dispute(
        self,
        dispute_id: UUID,
        outcome: DisputeOutcome,
        resolution_reasoning: str,
        resolution_method: ResolutionMethod = ResolutionMethod.AUTOMATIC,
    ) -> Dispute:
        """Resolve a dispute.
        
        Args:
            dispute_id: UUID of the dispute
            outcome: Resolution outcome
            resolution_reasoning: Explanation of resolution
            resolution_method: How the resolution was determined
        
        Returns:
            The resolved Dispute
        """
        dispute = self._disputes.get(dispute_id)
        if not dispute:
            raise NotFoundError("Dispute", str(dispute_id))
        
        if dispute.status != DisputeStatus.PENDING:
            raise ValidationException(f"Dispute is not pending: {dispute.status.value}")
        
        verification = self._verifications.get(dispute.verification_id)
        if not verification:
            raise NotFoundError("Verification", str(dispute.verification_id))
        
        # Update dispute
        dispute.outcome = outcome
        dispute.resolution_reasoning = resolution_reasoning
        dispute.resolution_method = resolution_method
        dispute.status = DisputeStatus.RESOLVED
        dispute.resolved_at = datetime.now()
        
        # Process based on outcome
        self._process_dispute_resolution(dispute, verification)
        
        logger.info(f"Dispute {dispute_id} resolved with outcome {outcome.value}")
        
        return dispute
    
    def get_verification(self, verification_id: UUID) -> Verification | None:
        """Get a verification by ID."""
        return self._verifications.get(verification_id)
    
    def get_verifications_for_belief(self, belief_id: UUID) -> list[Verification]:
        """Get all verifications for a belief."""
        return [v for v in self._verifications.values() if v.belief_id == belief_id]
    
    def get_dispute(self, dispute_id: UUID) -> Dispute | None:
        """Get a dispute by ID."""
        return self._disputes.get(dispute_id)
    
    def get_reputation(self, identity_id: str) -> ReputationScore | None:
        """Get reputation for an identity."""
        return self._reputations.get(identity_id)
    
    def get_verification_summary(self, belief_id: UUID) -> dict[str, Any]:
        """Get summary of verifications for a belief."""
        verifications = self.get_verifications_for_belief(belief_id)
        
        by_result = {r.value: 0 for r in VerificationResult}
        by_status = {s.value: 0 for s in VerificationStatus}
        total_stake = 0.0
        
        for v in verifications:
            by_result[v.result.value] += 1
            by_status[v.status.value] += 1
            total_stake += v.stake.amount
        
        # Calculate consensus
        accepted = [v for v in verifications if v.status == VerificationStatus.ACCEPTED]
        consensus_result = None
        consensus_confidence = 0.0
        
        if accepted:
            # Reputation-weighted consensus
            result_scores: dict[str, float] = {}
            total_weight = 0.0
            
            for v in accepted:
                rep = self.get_or_create_reputation(v.verifier_id)
                weight = rep.overall * v.stake.amount
                result_scores[v.result.value] = result_scores.get(v.result.value, 0) + weight
                total_weight += weight
            
            if total_weight > 0:
                max_result = max(result_scores.items(), key=lambda x: x[1])
                consensus_result = max_result[0]
                consensus_confidence = max_result[1] / total_weight
        
        return {
            "total": len(verifications),
            "by_result": by_result,
            "by_status": by_status,
            "average_stake": total_stake / len(verifications) if verifications else 0,
            "total_stake": total_stake,
            "consensus_result": consensus_result,
            "consensus_confidence": consensus_confidence,
        }
    
    def _lock_stake(
        self,
        reputation: ReputationScore,
        amount: float,
        verification_id: UUID | None = None,
        dispute_id: UUID | None = None,
    ) -> StakePosition:
        """Lock reputation as stake."""
        position = StakePosition(
            id=uuid4(),
            identity_id=reputation.identity_id,
            amount=amount,
            type=StakeType.CHALLENGE if dispute_id else StakeType.STANDARD,
            verification_id=verification_id,
            dispute_id=dispute_id,
        )
        
        reputation.stake_at_risk += amount
        self._stake_positions[position.id] = position
        
        return position
    
    def _release_stake(
        self,
        identity_id: str,
        verification_id: UUID | None = None,
        dispute_id: UUID | None = None,
        forfeit: bool = False,
    ) -> float:
        """Release or forfeit staked reputation."""
        amount_released = 0.0
        reputation = self.get_or_create_reputation(identity_id)
        
        for position in list(self._stake_positions.values()):
            if position.identity_id != identity_id:
                continue
            if verification_id and position.verification_id != verification_id:
                continue
            if dispute_id and position.dispute_id != dispute_id:
                continue
            
            if forfeit:
                position.status = "forfeited"
            else:
                position.status = "returned"
            
            reputation.stake_at_risk -= position.amount
            amount_released += position.amount
        
        return amount_released
    
    def _process_verification_reputation(self, verification: Verification) -> None:
        """Process reputation updates for an accepted verification."""
        verifier_rep = self.get_or_create_reputation(verification.verifier_id)
        holder_rep = self.get_or_create_reputation(verification.holder_id)
        
        # Get belief info for calculations (simplified)
        belief_confidence = 0.7  # Would come from database
        
        # Count existing verifications
        existing = self.get_verifications_for_belief(verification.belief_id)
        existing_confirmations = len([v for v in existing if v.result == VerificationResult.CONFIRMED and v.id != verification.id])
        existing_contradictions = len([v for v in existing if v.result == VerificationResult.CONTRADICTED and v.id != verification.id])
        
        min_stake = calculate_min_stake(belief_confidence, 0.5)
        
        verifier_delta = 0.0
        holder_delta = 0.0
        
        if verification.result == VerificationResult.CONFIRMED:
            verifier_delta = calculate_confirmation_reward(
                verification.stake.amount, min_stake, belief_confidence, existing_confirmations
            )
            holder_delta = calculate_holder_confirmation_bonus(
                verifier_rep.overall, verification.stake.amount, min_stake
            )
        
        elif verification.result == VerificationResult.CONTRADICTED:
            is_first = existing_contradictions == 0
            verifier_delta = calculate_contradiction_reward(
                verification.stake.amount, min_stake, belief_confidence, is_first, existing_contradictions
            )
            holder_delta = -calculate_holder_contradiction_penalty(belief_confidence, verifier_rep.overall)
            verifier_rep.discrepancy_finds += 1
        
        elif verification.result == VerificationResult.UNCERTAIN:
            verifier_delta = ReputationConstants.UNCERTAINTY_BASE
        
        elif verification.result == VerificationResult.PARTIAL:
            accuracy = 0.5
            if verification.result_details and verification.result_details.accuracy_estimate is not None:
                accuracy = verification.result_details.accuracy_estimate
            
            verifier_delta = calculate_partial_reward(
                accuracy, verification.stake.amount, min_stake, belief_confidence,
                existing_confirmations, existing_contradictions
            )
            
            # Holder gets mixed effect
            confirm_bonus = calculate_holder_confirmation_bonus(verifier_rep.overall, verification.stake.amount, min_stake)
            contradict_penalty = calculate_holder_contradiction_penalty(belief_confidence, verifier_rep.overall)
            holder_delta = confirm_bonus * accuracy - contradict_penalty * (1 - accuracy)
        
        # Apply updates
        self._apply_reputation_update(verifier_rep, verifier_delta, f"Verification {verification.result.value}", verification.id)
        if holder_delta != 0:
            self._apply_reputation_update(holder_rep, holder_delta, f"Belief {verification.result.value}", verification.id)
        
        verifier_rep.verification_count += 1
    
    def _process_dispute_resolution(self, dispute: Dispute, verification: Verification) -> None:
        """Process reputation updates after dispute resolution."""
        verifier_rep = self.get_or_create_reputation(verification.verifier_id)
        disputer_rep = self.get_or_create_reputation(dispute.disputer_id)
        holder_rep = self.get_or_create_reputation(verification.holder_id)
        
        if dispute.outcome == DisputeOutcome.UPHELD:
            # Verifier wins - keeps stake + gets bonus from disputer
            verification.status = VerificationStatus.ACCEPTED
            
            bonus = dispute.stake.amount * 0.8
            self._apply_reputation_update(verifier_rep, bonus, "Dispute upheld - defense bonus", dispute_id=dispute.id)
            
            # Disputer loses stake
            self._release_stake(dispute.disputer_id, dispute_id=dispute.id, forfeit=True)
            self._apply_reputation_update(disputer_rep, -dispute.stake.amount, "Dispute lost - stake forfeited", dispute_id=dispute.id)
        
        elif dispute.outcome == DisputeOutcome.OVERTURNED:
            # Disputer wins - gets verifier's stake
            verification.status = VerificationStatus.OVERTURNED
            
            # Verifier loses stake and penalty
            self._release_stake(verification.verifier_id, verification_id=verification.id, forfeit=True)
            self._apply_reputation_update(verifier_rep, -verification.stake.amount, "Verification overturned - stake forfeited", dispute_id=dispute.id)
            
            # Disputer gets reward
            reward = verification.stake.amount * 0.8
            self._apply_reputation_update(disputer_rep, reward, "Dispute won - reward", dispute_id=dispute.id)
            self._release_stake(dispute.disputer_id, dispute_id=dispute.id)
            
            # If verification was CONTRADICTED and now overturned, restore holder
            if verification.result == VerificationResult.CONTRADICTED:
                restore_amount = calculate_holder_contradiction_penalty(0.7, verifier_rep.overall)
                self._apply_reputation_update(holder_rep, restore_amount, "Contradiction overturned - reputation restored", dispute_id=dispute.id)
        
        elif dispute.outcome == DisputeOutcome.MODIFIED:
            # Partial resolution
            verification.status = VerificationStatus.ACCEPTED
            # Complex proportional logic would go here
            self._release_stake(verification.verifier_id, verification_id=verification.id)
            self._release_stake(dispute.disputer_id, dispute_id=dispute.id)
        
        elif dispute.outcome == DisputeOutcome.DISMISSED:
            # Frivolous dispute
            verification.status = VerificationStatus.ACCEPTED
            
            # Verifier gets harassment compensation
            compensation = dispute.stake.amount * 0.5
            self._apply_reputation_update(verifier_rep, compensation, "Frivolous dispute - compensation", dispute_id=dispute.id)
            
            # Disputer loses stake + penalty
            self._release_stake(dispute.disputer_id, dispute_id=dispute.id, forfeit=True)
            penalty = dispute.stake.amount * (1 + ReputationConstants.FRIVOLOUS_DISPUTE_PENALTY)
            self._apply_reputation_update(disputer_rep, -penalty, "Frivolous dispute - stake + penalty", dispute_id=dispute.id)
    
    def _apply_reputation_update(
        self,
        reputation: ReputationScore,
        delta: float,
        reason: str,
        verification_id: UUID | None = None,
        dispute_id: UUID | None = None,
        dimension: str = "overall",
    ) -> ReputationUpdate:
        """Apply a reputation update and log it."""
        old_value = reputation.overall
        
        # Apply with bounds
        new_value = max(ReputationConstants.REPUTATION_FLOOR, min(1.0, old_value + delta))
        reputation.overall = new_value
        reputation.modified_at = datetime.now()
        
        # Domain update if applicable
        if dimension != "overall" and dimension in reputation.by_domain:
            old_domain = reputation.by_domain[dimension]
            reputation.by_domain[dimension] = max(0.1, min(1.0, old_domain + delta))
        
        # Log event
        event = ReputationUpdate(
            id=uuid4(),
            identity_id=reputation.identity_id,
            delta=delta,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            dimension=dimension,
            verification_id=verification_id,
            dispute_id=dispute_id,
        )
        self._reputation_events.append(event)
        
        logger.info(f"Reputation update for {reputation.identity_id}: {old_value:.4f} -> {new_value:.4f} ({delta:+.4f}) - {reason}")
        
        return event


# ============================================================================
# Module-level convenience functions
# ============================================================================

def create_evidence(
    evidence_type: EvidenceType,
    contribution: EvidenceContribution,
    relevance: float = 0.8,
    **kwargs: Any,
) -> Evidence:
    """Convenience function to create evidence.
    
    Args:
        evidence_type: Type of evidence
        contribution: How it contributes
        relevance: Relevance score
        **kwargs: Type-specific arguments
    
    Returns:
        Evidence object
    """
    evidence = Evidence(
        id=uuid4(),
        type=evidence_type,
        relevance=relevance,
        contribution=contribution,
    )
    
    if evidence_type == EvidenceType.EXTERNAL:
        evidence.external_source = ExternalSource(**kwargs)
    elif evidence_type == EvidenceType.BELIEF:
        evidence.belief_reference = BeliefReference(**kwargs)
    elif evidence_type == EvidenceType.OBSERVATION:
        evidence.observation = Observation(**kwargs)
    elif evidence_type == EvidenceType.DERIVATION:
        evidence.derivation = DerivationProof(**kwargs)
    elif evidence_type == EvidenceType.TESTIMONY:
        evidence.testimony_statement = kwargs.get("statement", "")
    
    return evidence
