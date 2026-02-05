"""Evidence class for the verification protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from ..exceptions import ValidationException
from .enums import EvidenceContribution, EvidenceType
from .models import BeliefReference, DerivationProof, ExternalSource, Observation


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
            raise ValidationException(
                "Evidence relevance must be between 0.0 and 1.0",
                "relevance",
                self.relevance,
            )

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
