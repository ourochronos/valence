"""Verification Protocol with Staking for Valence.

This module implements the adversarial verification system where:
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

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from .confidence import DimensionalConfidence
from .exceptions import ValidationException, NotFoundError, ConflictError

logger = logging.getLogger(__name__)


# ============================================================================
# Constants (from REPUTATION.md)
# ============================================================================

class ReputationConstants:
    """Constants for reputation calculations."""
    
    # Base rewards
    CONFIRMATION_BASE = 0.001
    CONTRADICTION_BASE = 0.005
    UNCERTAINTY_BASE = 0.0002
    
    # Bounties
    BOUNTY_MULTIPLIER = 0.5
    FIRST_FINDER_BONUS = 2.0
    
    # Penalties
    CONTRADICTION_PENALTY_BASE = 0.003
    FRIVOLOUS_DISPUTE_PENALTY = 0.2
    COLLUSION_PENALTY = 0.5
    
    # Calibration
    CALIBRATION_BONUS_BASE = 0.01
    CALIBRATION_PENALTY_THRESHOLD = 0.4
    
    # Decay
    MONTHLY_INACTIVITY_DECAY = 0.02
    VERIFICATION_HALF_LIFE_MONTHS = 12
    
    # Limits
    MAX_DAILY_GAIN = 0.02
    MAX_WEEKLY_GAIN = 0.08
    REPUTATION_FLOOR = 0.1
    MAX_STAKE_RATIO = 0.2
    
    # Recovery
    MAX_MONTHLY_RECOVERY = 0.03
    PROBATION_DURATION_DAYS = 90
    
    # Timing
    VALIDATION_WINDOW_HOURS = 1
    ACCEPTANCE_DELAY_HOURS = 24
    DISPUTE_WINDOW_DAYS = 7
    RESOLUTION_TIMEOUT_DAYS = 14
    STAKE_LOCKUP_DAYS = 7
    BOUNTY_STAKE_LOCKUP_DAYS = 14

    # Rate limits
    MAX_VERIFICATIONS_PER_DAY = 50
    MAX_PENDING_VERIFICATIONS = 10
    MAX_VERIFICATIONS_PER_BELIEF = 100
    MAX_VERIFICATION_VELOCITY_PER_HOUR = 5


# ============================================================================
# Enums
# ============================================================================

class VerificationResult(str, Enum):
    """Result of a verification."""
    CONFIRMED = "confirmed"      # Evidence supports the belief
    CONTRADICTED = "contradicted"  # Evidence refutes the belief
    UNCERTAIN = "uncertain"      # Insufficient evidence either way
    PARTIAL = "partial"          # Partially correct, with qualifications


class VerificationStatus(str, Enum):
    """Status of a verification."""
    PENDING = "pending"          # Awaiting validation
    ACCEPTED = "accepted"        # Validated and accepted
    DISPUTED = "disputed"        # Under dispute
    OVERTURNED = "overturned"    # Dispute overturned the verification
    REJECTED = "rejected"        # Failed validation
    EXPIRED = "expired"          # Timed out during validation


class StakeType(str, Enum):
    """Type of stake."""
    STANDARD = "standard"        # Normal verification stake
    BOUNTY = "bounty"            # Claiming a discrepancy bounty
    CHALLENGE = "challenge"      # Challenging existing verification


class EvidenceType(str, Enum):
    """Type of evidence provided."""
    BELIEF = "belief"            # Reference to another Valence belief
    EXTERNAL = "external"        # External source (URL, paper, etc.)
    OBSERVATION = "observation"  # Verifier's direct observation
    DERIVATION = "derivation"    # Logical/mathematical proof
    TESTIMONY = "testimony"      # Statement from another agent


class EvidenceContribution(str, Enum):
    """How evidence contributes to verification."""
    SUPPORTS = "supports"        # Evidence for the belief
    CONTRADICTS = "contradicts"  # Evidence against the belief
    CONTEXT = "context"          # Adds relevant context
    QUALIFIES = "qualifies"      # Adds conditions/limitations


class ContradictionType(str, Enum):
    """Type of contradiction found."""
    FACTUALLY_FALSE = "factually_false"
    OUTDATED = "outdated"
    MISATTRIBUTED = "misattributed"
    OVERSTATED = "overstated"
    MISSING_CONTEXT = "missing_context"
    LOGICAL_ERROR = "logical_error"


class UncertaintyReason(str, Enum):
    """Reason for uncertain result."""
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_SOURCES = "conflicting_sources"
    OUTSIDE_EXPERTISE = "outside_expertise"
    UNFALSIFIABLE = "unfalsifiable"
    REQUIRES_EXPERIMENT = "requires_experiment"


class DisputeType(str, Enum):
    """Type of dispute."""
    EVIDENCE_INVALID = "evidence_invalid"
    EVIDENCE_FABRICATED = "evidence_fabricated"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    REASONING_FLAWED = "reasoning_flawed"
    CONFLICT_OF_INTEREST = "conflict_of_interest"
    NEW_EVIDENCE = "new_evidence"


class DisputeOutcome(str, Enum):
    """Outcome of a dispute resolution."""
    UPHELD = "upheld"            # Original verification stands
    OVERTURNED = "overturned"    # Verification was wrong
    MODIFIED = "modified"        # Result changed
    DISMISSED = "dismissed"      # Dispute was frivolous


class DisputeStatus(str, Enum):
    """Status of a dispute."""
    PENDING = "pending"          # Awaiting resolution
    RESOLVED = "resolved"        # Resolution complete
    EXPIRED = "expired"          # Timed out


class ResolutionMethod(str, Enum):
    """Method used to resolve a dispute."""
    AUTOMATIC = "automatic"      # Algorithm decides
    JURY = "jury"                # Random selection of jurors
    EXPERT = "expert"            # Domain experts decide
    APPEAL = "appeal"            # Higher-level review


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ExternalSource:
    """External evidence source."""
    url: str | None = None
    doi: str | None = None
    isbn: str | None = None
    citation: str | None = None
    archive_hash: str | None = None
    archived_at: datetime | None = None
    source_reputation: float = 0.5
    
    def to_dict(self) -> dict[str, Any]:
        result = {k: v for k, v in asdict(self).items() if v is not None}
        if self.archived_at:
            result["archived_at"] = self.archived_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalSource:
        if data.get("archived_at") and isinstance(data["archived_at"], str):
            data["archived_at"] = datetime.fromisoformat(data["archived_at"])
        return cls(**data)


@dataclass
class BeliefReference:
    """Reference to another belief as evidence."""
    belief_id: UUID
    holder_id: str  # DID
    content_hash: str
    confidence_at_time: DimensionalConfidence | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_id": str(self.belief_id),
            "holder_id": self.holder_id,
            "content_hash": self.content_hash,
            "confidence_at_time": self.confidence_at_time.to_dict() if self.confidence_at_time else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeliefReference:
        confidence = None
        if data.get("confidence_at_time"):
            confidence = DimensionalConfidence.from_dict(data["confidence_at_time"])
        return cls(
            belief_id=UUID(data["belief_id"]) if isinstance(data["belief_id"], str) else data["belief_id"],
            holder_id=data["holder_id"],
            content_hash=data["content_hash"],
            confidence_at_time=confidence,
        )


@dataclass
class Observation:
    """Direct observation evidence."""
    description: str
    timestamp: datetime
    method: str
    reproducible: bool = False
    reproduction_instructions: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "method": self.method,
            "reproducible": self.reproducible,
            "reproduction_instructions": self.reproduction_instructions,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            description=data["description"],
            timestamp=timestamp,
            method=data["method"],
            reproducible=data.get("reproducible", False),
            reproduction_instructions=data.get("reproduction_instructions"),
        )


@dataclass
class DerivationProof:
    """Logical derivation as evidence."""
    premises: list[UUID]
    logic_type: str  # 'deductive', 'inductive', 'abductive'
    proof_steps: list[str]
    formal_notation: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "premises": [str(p) for p in self.premises],
            "logic_type": self.logic_type,
            "proof_steps": self.proof_steps,
            "formal_notation": self.formal_notation,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DerivationProof:
        return cls(
            premises=[UUID(p) if isinstance(p, str) else p for p in data["premises"]],
            logic_type=data["logic_type"],
            proof_steps=data["proof_steps"],
            formal_notation=data.get("formal_notation"),
        )


@dataclass
class Evidence:
    """Evidence supporting a verification."""
    id: UUID
    type: EvidenceType
    relevance: float  # 0.0-1.0
    contribution: EvidenceContribution
    verifier_notes: str | None = None
    
    # Content (exactly one populated based on type)
    belief_reference: BeliefReference | None = None
    external_source: ExternalSource | None = None
    observation: Observation | None = None
    derivation: DerivationProof | None = None
    testimony_statement: str | None = None
    
    def __post_init__(self):
        if self.relevance < 0.0 or self.relevance > 1.0:
            raise ValidationException("Evidence relevance must be between 0.0 and 1.0", "relevance", self.relevance)
    
    def content_hash(self) -> str:
        """Compute SHA-256 hash of evidence content."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": str(self.id),
            "type": self.type.value,
            "relevance": self.relevance,
            "contribution": self.contribution.value,
            "verifier_notes": self.verifier_notes,
        }
        if self.belief_reference:
            result["belief_reference"] = self.belief_reference.to_dict()
        if self.external_source:
            result["external_source"] = self.external_source.to_dict()
        if self.observation:
            result["observation"] = self.observation.to_dict()
        if self.derivation:
            result["derivation"] = self.derivation.to_dict()
        if self.testimony_statement:
            result["testimony_statement"] = self.testimony_statement
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        evidence_id = data.get("id")
        if evidence_id and isinstance(evidence_id, str):
            evidence_id = UUID(evidence_id)
        elif not evidence_id:
            evidence_id = uuid4()
        
        belief_ref = None
        if data.get("belief_reference"):
            belief_ref = BeliefReference.from_dict(data["belief_reference"])
        
        external = None
        if data.get("external_source"):
            external = ExternalSource.from_dict(data["external_source"])
        
        obs = None
        if data.get("observation"):
            obs = Observation.from_dict(data["observation"])
        
        deriv = None
        if data.get("derivation"):
            deriv = DerivationProof.from_dict(data["derivation"])
        
        return cls(
            id=evidence_id,
            type=EvidenceType(data["type"]),
            relevance=float(data["relevance"]),
            contribution=EvidenceContribution(data["contribution"]),
            verifier_notes=data.get("verifier_notes"),
            belief_reference=belief_ref,
            external_source=external,
            observation=obs,
            derivation=deriv,
            testimony_statement=data.get("testimony_statement"),
        )


@dataclass
class ResultDetails:
    """Structured breakdown of verification result."""
    # For CONFIRMED
    confirmation_strength: str | None = None  # 'strong', 'moderate', 'weak'
    confirmed_aspects: list[str] = field(default_factory=list)
    
    # For CONTRADICTED
    contradiction_type: ContradictionType | None = None
    corrected_belief: str | None = None
    severity: str | None = None  # 'minor', 'moderate', 'major', 'critical'
    
    # For PARTIAL
    accurate_portions: list[str] = field(default_factory=list)
    inaccurate_portions: list[str] = field(default_factory=list)
    accuracy_estimate: float | None = None  # 0.0-1.0
    
    # For UNCERTAIN
    uncertainty_reason: UncertaintyReason | None = None
    additional_evidence_needed: list[str] = field(default_factory=list)
    
    # Universal
    confidence_assessment: list[dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.confirmation_strength:
            result["confirmation_strength"] = self.confirmation_strength
        if self.confirmed_aspects:
            result["confirmed_aspects"] = self.confirmed_aspects
        if self.contradiction_type:
            result["contradiction_type"] = self.contradiction_type.value
        if self.corrected_belief:
            result["corrected_belief"] = self.corrected_belief
        if self.severity:
            result["severity"] = self.severity
        if self.accurate_portions:
            result["accurate_portions"] = self.accurate_portions
        if self.inaccurate_portions:
            result["inaccurate_portions"] = self.inaccurate_portions
        if self.accuracy_estimate is not None:
            result["accuracy_estimate"] = self.accuracy_estimate
        if self.uncertainty_reason:
            result["uncertainty_reason"] = self.uncertainty_reason.value
        if self.additional_evidence_needed:
            result["additional_evidence_needed"] = self.additional_evidence_needed
        if self.confidence_assessment:
            result["confidence_assessment"] = self.confidence_assessment
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResultDetails:
        contradiction_type = None
        if data.get("contradiction_type"):
            contradiction_type = ContradictionType(data["contradiction_type"])
        
        uncertainty_reason = None
        if data.get("uncertainty_reason"):
            uncertainty_reason = UncertaintyReason(data["uncertainty_reason"])
        
        return cls(
            confirmation_strength=data.get("confirmation_strength"),
            confirmed_aspects=data.get("confirmed_aspects", []),
            contradiction_type=contradiction_type,
            corrected_belief=data.get("corrected_belief"),
            severity=data.get("severity"),
            accurate_portions=data.get("accurate_portions", []),
            inaccurate_portions=data.get("inaccurate_portions", []),
            accuracy_estimate=data.get("accuracy_estimate"),
            uncertainty_reason=uncertainty_reason,
            additional_evidence_needed=data.get("additional_evidence_needed", []),
            confidence_assessment=data.get("confidence_assessment", []),
        )


@dataclass
class Stake:
    """Reputation stake for a verification."""
    amount: float
    type: StakeType
    locked_until: datetime
    escrow_id: UUID
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValidationException("Stake amount cannot be negative", "amount", self.amount)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": self.amount,
            "type": self.type.value,
            "locked_until": self.locked_until.isoformat(),
            "escrow_id": str(self.escrow_id),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stake:
        locked_until = data["locked_until"]
        if isinstance(locked_until, str):
            locked_until = datetime.fromisoformat(locked_until)
        
        escrow_id = data["escrow_id"]
        if isinstance(escrow_id, str):
            escrow_id = UUID(escrow_id)
        
        return cls(
            amount=float(data["amount"]),
            type=StakeType(data["type"]),
            locked_until=locked_until,
            escrow_id=escrow_id,
        )


@dataclass
class Verification:
    """A verification of a belief."""
    id: UUID
    verifier_id: str  # DID
    belief_id: UUID
    holder_id: str  # DID (cached)
    result: VerificationResult
    evidence: list[Evidence]
    stake: Stake
    reasoning: str | None = None
    result_details: ResultDetails | None = None
    status: VerificationStatus = VerificationStatus.PENDING
    dispute_id: UUID | None = None
    signature: bytes | None = None
    created_at: datetime = field(default_factory=datetime.now)
    accepted_at: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "verifier_id": self.verifier_id,
            "belief_id": str(self.belief_id),
            "holder_id": self.holder_id,
            "result": self.result.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "stake": self.stake.to_dict(),
            "reasoning": self.reasoning,
            "result_details": self.result_details.to_dict() if self.result_details else None,
            "status": self.status.value,
            "dispute_id": str(self.dispute_id) if self.dispute_id else None,
            "signature": self.signature.hex() if self.signature else None,
            "created_at": self.created_at.isoformat(),
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Verification:
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            verifier_id=data["verifier_id"],
            belief_id=UUID(data["belief_id"]) if isinstance(data["belief_id"], str) else data["belief_id"],
            holder_id=data["holder_id"],
            result=VerificationResult(data["result"]),
            evidence=[Evidence.from_dict(e) for e in data["evidence"]],
            stake=Stake.from_dict(data["stake"]),
            reasoning=data.get("reasoning"),
            result_details=ResultDetails.from_dict(data["result_details"]) if data.get("result_details") else None,
            status=VerificationStatus(data.get("status", "pending")),
            dispute_id=UUID(data["dispute_id"]) if data.get("dispute_id") else None,
            signature=bytes.fromhex(data["signature"]) if data.get("signature") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now()),
            accepted_at=datetime.fromisoformat(data["accepted_at"]) if data.get("accepted_at") and isinstance(data["accepted_at"], str) else data.get("accepted_at"),
        )
    
    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Verification:
        """Create from database row."""
        evidence_data = row.get("evidence", [])
        if isinstance(evidence_data, str):
            evidence_data = json.loads(evidence_data)
        
        stake_data = row.get("stake", {})
        if isinstance(stake_data, str):
            stake_data = json.loads(stake_data)
        
        result_details = None
        if row.get("result_details"):
            details_data = row["result_details"]
            if isinstance(details_data, str):
                details_data = json.loads(details_data)
            result_details = ResultDetails.from_dict(details_data)
        
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            verifier_id=row["verifier_id"],
            belief_id=row["belief_id"] if isinstance(row["belief_id"], UUID) else UUID(row["belief_id"]),
            holder_id=row["holder_id"],
            result=VerificationResult(row["result"]),
            evidence=[Evidence.from_dict(e) for e in evidence_data],
            stake=Stake.from_dict(stake_data),
            reasoning=row.get("reasoning"),
            result_details=result_details,
            status=VerificationStatus(row.get("status", "pending")),
            dispute_id=row["dispute_id"] if row.get("dispute_id") else None,
            signature=row.get("signature"),
            created_at=row["created_at"],
            accepted_at=row.get("accepted_at"),
        )


@dataclass
class Dispute:
    """A dispute challenging a verification."""
    id: UUID
    verification_id: UUID
    disputer_id: str  # DID
    counter_evidence: list[Evidence]
    stake: Stake
    dispute_type: DisputeType
    reasoning: str
    proposed_result: VerificationResult | None = None
    status: DisputeStatus = DisputeStatus.PENDING
    outcome: DisputeOutcome | None = None
    resolution_reasoning: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    resolution_method: ResolutionMethod | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "verification_id": str(self.verification_id),
            "disputer_id": self.disputer_id,
            "counter_evidence": [e.to_dict() for e in self.counter_evidence],
            "stake": self.stake.to_dict(),
            "dispute_type": self.dispute_type.value,
            "reasoning": self.reasoning,
            "proposed_result": self.proposed_result.value if self.proposed_result else None,
            "status": self.status.value,
            "outcome": self.outcome.value if self.outcome else None,
            "resolution_reasoning": self.resolution_reasoning,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_method": self.resolution_method.value if self.resolution_method else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dispute:
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            verification_id=UUID(data["verification_id"]) if isinstance(data["verification_id"], str) else data["verification_id"],
            disputer_id=data["disputer_id"],
            counter_evidence=[Evidence.from_dict(e) for e in data["counter_evidence"]],
            stake=Stake.from_dict(data["stake"]),
            dispute_type=DisputeType(data["dispute_type"]),
            reasoning=data["reasoning"],
            proposed_result=VerificationResult(data["proposed_result"]) if data.get("proposed_result") else None,
            status=DisputeStatus(data.get("status", "pending")),
            outcome=DisputeOutcome(data["outcome"]) if data.get("outcome") else None,
            resolution_reasoning=data.get("resolution_reasoning"),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now()),
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") and isinstance(data["resolved_at"], str) else data.get("resolved_at"),
            resolution_method=ResolutionMethod(data["resolution_method"]) if data.get("resolution_method") else None,
        )
    
    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Dispute:
        """Create from database row."""
        evidence_data = row.get("counter_evidence", [])
        if isinstance(evidence_data, str):
            evidence_data = json.loads(evidence_data)
        
        stake_data = row.get("stake", {})
        if isinstance(stake_data, str):
            stake_data = json.loads(stake_data)
        
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            verification_id=row["verification_id"] if isinstance(row["verification_id"], UUID) else UUID(row["verification_id"]),
            disputer_id=row["disputer_id"],
            counter_evidence=[Evidence.from_dict(e) for e in evidence_data],
            stake=Stake.from_dict(stake_data),
            dispute_type=DisputeType(row["dispute_type"]),
            reasoning=row["reasoning"],
            proposed_result=VerificationResult(row["proposed_result"]) if row.get("proposed_result") else None,
            status=DisputeStatus(row.get("status", "pending")),
            outcome=DisputeOutcome(row["outcome"]) if row.get("outcome") else None,
            resolution_reasoning=row.get("resolution_reasoning"),
            created_at=row["created_at"],
            resolved_at=row.get("resolved_at"),
            resolution_method=ResolutionMethod(row["resolution_method"]) if row.get("resolution_method") else None,
        )


@dataclass 
class ReputationScore:
    """Agent reputation score."""
    identity_id: str  # DID
    overall: float = 0.5  # 0.0-1.0
    by_domain: dict[str, float] = field(default_factory=dict)
    verification_count: int = 0
    discrepancy_finds: int = 0
    stake_at_risk: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # Ensure reputation stays within bounds
        self.overall = max(ReputationConstants.REPUTATION_FLOOR, min(1.0, self.overall))
    
    def available_stake(self) -> float:
        """Calculate available reputation for staking."""
        max_stakeable = self.overall * ReputationConstants.MAX_STAKE_RATIO
        return max(0.0, max_stakeable - self.stake_at_risk)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "overall": self.overall,
            "by_domain": self.by_domain,
            "verification_count": self.verification_count,
            "discrepancy_finds": self.discrepancy_finds,
            "stake_at_risk": self.stake_at_risk,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReputationScore:
        return cls(
            identity_id=data["identity_id"],
            overall=float(data.get("overall", 0.5)),
            by_domain=data.get("by_domain", {}),
            verification_count=int(data.get("verification_count", 0)),
            discrepancy_finds=int(data.get("discrepancy_finds", 0)),
            stake_at_risk=float(data.get("stake_at_risk", 0.0)),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now()),
            modified_at=datetime.fromisoformat(data["modified_at"]) if isinstance(data.get("modified_at"), str) else data.get("modified_at", datetime.now()),
        )
    
    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ReputationScore:
        """Create from database row."""
        by_domain = row.get("by_domain", {})
        if isinstance(by_domain, str):
            by_domain = json.loads(by_domain)
        
        return cls(
            identity_id=row["identity_id"],
            overall=float(row.get("overall", 0.5)),
            by_domain=by_domain,
            verification_count=int(row.get("verification_count", 0)),
            discrepancy_finds=int(row.get("discrepancy_finds", 0)),
            stake_at_risk=float(row.get("stake_at_risk", 0.0)),
            created_at=row["created_at"],
            modified_at=row["modified_at"],
        )


@dataclass
class ReputationUpdate:
    """A reputation change event."""
    id: UUID
    identity_id: str  # DID
    delta: float
    old_value: float
    new_value: float
    reason: str
    dimension: str = "overall"  # 'overall' or domain name
    verification_id: UUID | None = None
    dispute_id: UUID | None = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "identity_id": self.identity_id,
            "delta": self.delta,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "dimension": self.dimension,
            "verification_id": str(self.verification_id) if self.verification_id else None,
            "dispute_id": str(self.dispute_id) if self.dispute_id else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class StakePosition:
    """A staked reputation position."""
    id: UUID
    identity_id: str  # DID
    amount: float
    type: StakeType
    verification_id: UUID | None = None
    dispute_id: UUID | None = None
    locked_at: datetime = field(default_factory=datetime.now)
    unlocks_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=ReputationConstants.STAKE_LOCKUP_DAYS))
    status: str = "locked"  # 'locked', 'pending_return', 'forfeited', 'returned'
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "identity_id": self.identity_id,
            "amount": self.amount,
            "type": self.type.value,
            "verification_id": str(self.verification_id) if self.verification_id else None,
            "dispute_id": str(self.dispute_id) if self.dispute_id else None,
            "locked_at": self.locked_at.isoformat(),
            "unlocks_at": self.unlocks_at.isoformat(),
            "status": self.status,
        }


@dataclass
class DiscrepancyBounty:
    """Bounty for finding contradictions in high-confidence beliefs."""
    belief_id: UUID
    holder_id: str  # DID
    base_amount: float
    confidence_premium: float
    age_factor: float
    total_bounty: float
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    claimed: bool = False
    claimed_by: str | None = None
    claimed_at: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_id": str(self.belief_id),
            "holder_id": self.holder_id,
            "base_amount": self.base_amount,
            "confidence_premium": self.confidence_premium,
            "age_factor": self.age_factor,
            "total_bounty": self.total_bounty,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "claimed": self.claimed,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
        }


# ============================================================================
# Stake Calculation Functions
# ============================================================================

def calculate_min_stake(
    belief_confidence: float,
    verifier_domain_reputation: float = 0.5,
) -> float:
    """Calculate minimum stake required for verification.
    
    Formula: min_stake = base_stake × confidence_multiplier × domain_multiplier
    
    Args:
        belief_confidence: Overall confidence of the belief (0.0-1.0)
        verifier_domain_reputation: Verifier's reputation in the belief's domain
    
    Returns:
        Minimum stake amount
    """
    base_stake = 0.01  # 1% of neutral reputation
    confidence_multiplier = belief_confidence
    domain_multiplier = 1.0 + (verifier_domain_reputation * 0.5)
    
    return base_stake * confidence_multiplier * domain_multiplier


def calculate_max_stake(verifier_reputation: float) -> float:
    """Calculate maximum stake a verifier can make.
    
    Capped at 20% of verifier's current overall reputation.
    
    Args:
        verifier_reputation: Verifier's overall reputation
    
    Returns:
        Maximum stake amount
    """
    return verifier_reputation * ReputationConstants.MAX_STAKE_RATIO


def calculate_dispute_min_stake(
    verification_stake: float,
    is_holder: bool = False,
) -> float:
    """Calculate minimum stake for filing a dispute.
    
    Args:
        verification_stake: Stake of the original verification
        is_holder: True if disputer is the belief holder
    
    Returns:
        Minimum dispute stake
    """
    multiplier = 1.0 if is_holder else 1.5
    return verification_stake * multiplier


def calculate_bounty(
    holder_stake: float,
    belief_confidence: float,
    days_since_creation: int,
    domain_importance: float = 1.0,
) -> float:
    """Calculate discrepancy bounty for a belief.
    
    Formula: bounty = holder_stake × confidence_premium × age_factor × domain_multiplier
    
    Args:
        holder_stake: Stake the holder put on the belief
        belief_confidence: Overall confidence of the belief
        days_since_creation: Days since belief was created
        domain_importance: Importance weight of the belief's domain
    
    Returns:
        Total bounty amount
    """
    confidence_premium = belief_confidence ** 2
    age_factor = min(2.0, 1.0 + days_since_creation / 30)
    
    return holder_stake * ReputationConstants.BOUNTY_MULTIPLIER * confidence_premium * age_factor * domain_importance


# ============================================================================
# Reputation Update Functions
# ============================================================================

def calculate_confirmation_reward(
    stake: float,
    min_stake: float,
    belief_confidence: float,
    existing_confirmations: int,
) -> float:
    """Calculate reward for a CONFIRMED verification.
    
    Formula: base_reward × stake_multiplier × confidence_factor × diminishing_factor
    
    Args:
        stake: Amount staked by verifier
        min_stake: Minimum required stake
        belief_confidence: Confidence of the verified belief
        existing_confirmations: Number of existing confirmations
    
    Returns:
        Reputation reward
    """
    base_reward = ReputationConstants.CONFIRMATION_BASE
    stake_multiplier = min(2.0, stake / min_stake) if min_stake > 0 else 1.0
    confidence_factor = belief_confidence
    diminishing_factor = 1.0 / math.sqrt(existing_confirmations + 1)
    
    return base_reward * stake_multiplier * confidence_factor * diminishing_factor


def calculate_contradiction_reward(
    stake: float,
    min_stake: float,
    belief_confidence: float,
    is_first_contradiction: bool,
    existing_contradictions: int = 0,
) -> float:
    """Calculate reward for a CONTRADICTED verification.
    
    Formula: base_bounty × stake_multiplier × confidence_premium × novelty_bonus
    
    Args:
        stake: Amount staked by verifier
        min_stake: Minimum required stake
        belief_confidence: Confidence of the verified belief
        is_first_contradiction: Whether this is the first contradiction
        existing_contradictions: Number of existing contradictions
    
    Returns:
        Reputation reward
    """
    base_bounty = ReputationConstants.CONTRADICTION_BASE
    stake_multiplier = min(3.0, stake / min_stake) if min_stake > 0 else 1.0
    confidence_premium = belief_confidence ** 2
    
    if is_first_contradiction:
        novelty_bonus = ReputationConstants.FIRST_FINDER_BONUS
    else:
        novelty_bonus = 1.0 / math.sqrt(existing_contradictions + 1)
    
    return base_bounty * stake_multiplier * confidence_premium * novelty_bonus


def calculate_holder_confirmation_bonus(
    verifier_reputation: float,
    stake: float,
    min_stake: float,
) -> float:
    """Calculate holder's bonus when their belief is confirmed.
    
    Args:
        verifier_reputation: Reputation of the verifier
        stake: Stake amount
        min_stake: Minimum required stake
    
    Returns:
        Reputation bonus for holder
    """
    return 0.0005 * verifier_reputation * math.sqrt(stake / min_stake) if min_stake > 0 else 0.0


def calculate_holder_contradiction_penalty(
    belief_confidence: float,
    verifier_reputation: float,
) -> float:
    """Calculate holder's penalty when their belief is contradicted.
    
    Formula: base_penalty × overconfidence_multiplier × verifier_weight
    
    Args:
        belief_confidence: Confidence of the contradicted belief
        verifier_reputation: Reputation of the verifier
    
    Returns:
        Reputation penalty for holder (positive number)
    """
    base_penalty = ReputationConstants.CONTRADICTION_PENALTY_BASE
    overconfidence_multiplier = belief_confidence ** 2
    verifier_weight = verifier_reputation
    
    return base_penalty * overconfidence_multiplier * verifier_weight


def calculate_partial_reward(
    accuracy_estimate: float,
    stake: float,
    min_stake: float,
    belief_confidence: float,
    existing_confirmations: int,
    existing_contradictions: int,
) -> float:
    """Calculate reward for a PARTIAL verification.
    
    Proportional credit based on accuracy estimate.
    
    Args:
        accuracy_estimate: Portion of belief that's accurate (0.0-1.0)
        stake: Amount staked
        min_stake: Minimum required stake
        belief_confidence: Confidence of the belief
        existing_confirmations: Number of existing confirmations
        existing_contradictions: Number of existing contradictions
    
    Returns:
        Reputation reward
    """
    confirm_portion = calculate_confirmation_reward(
        stake, min_stake, belief_confidence, existing_confirmations
    ) * accuracy_estimate
    
    contradict_portion = calculate_contradiction_reward(
        stake, min_stake, belief_confidence, 
        existing_contradictions == 0, existing_contradictions
    ) * (1 - accuracy_estimate)
    
    return confirm_portion + contradict_portion


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
        if e.type == EvidenceType.BELIEF and e.belief_reference:
            # Circular reference check would need belief_id from context
            pass
        
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


# ============================================================================
# Verification Service Functions
# ============================================================================

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
