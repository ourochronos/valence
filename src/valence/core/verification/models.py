"""Data models for the verification protocol.

Contains basic data structures used as evidence sources.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from oro_confidence import DimensionalConfidence


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
            "confidence_at_time": (self.confidence_at_time.to_dict() if self.confidence_at_time else None),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeliefReference:
        confidence = None
        if data.get("confidence_at_time"):
            confidence = DimensionalConfidence.from_dict(data["confidence_at_time"])
        return cls(
            belief_id=(UUID(data["belief_id"]) if isinstance(data["belief_id"], str) else data["belief_id"]),
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
