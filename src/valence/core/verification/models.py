"""Data models for the verification protocol.

Contains all dataclasses for evidence, verifications, disputes,
reputation, and staking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from ..confidence import DimensionalConfidence
from ..exceptions import ValidationException
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


# ============================================================================
# Evidence Models
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


# ============================================================================
# Result Details
# ============================================================================

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


# ============================================================================
# Stake
# ============================================================================

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


# ============================================================================
# Verification
# ============================================================================

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


# ============================================================================
# Dispute
# ============================================================================

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


# ============================================================================
# Reputation Models
# ============================================================================

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
