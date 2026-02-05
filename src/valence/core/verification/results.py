"""Result details and stake models for the verification protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from ..exceptions import ValidationException
from .enums import ContradictionType, StakeType, UncertaintyReason


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
