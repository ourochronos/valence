"""Data models for Valence knowledge substrate.

These models provide a clean Python interface to the PostgreSQL schema.
They use dataclasses for simplicity and can be serialized to/from JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from valence.lib.our_confidence import DimensionalConfidence


class BeliefStatus(StrEnum):
    """Status of a belief."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"
    ARCHIVED = "archived"


class EntityType(StrEnum):
    """Types of entities."""

    PERSON = "person"
    ORGANIZATION = "organization"
    TOOL = "tool"
    CONCEPT = "concept"
    PROJECT = "project"
    LOCATION = "location"
    SERVICE = "service"


class EntityRole(StrEnum):
    """Role of an entity in a belief."""

    SUBJECT = "subject"
    OBJECT = "object"
    CONTEXT = "context"
    SOURCE = "source"


class SessionStatus(StrEnum):
    """Status of a session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class Platform(StrEnum):
    """Supported platforms."""

    CLAUDE_CODE = "claude-code"
    MATRIX = "matrix"
    API = "api"
    SLACK = "slack"
    CLAUDE_WEB = "claude-web"
    CLAUDE_DESKTOP = "claude-desktop"
    CLAUDE_MOBILE = "claude-mobile"
    OPENCLAW = "openclaw"


class ExchangeRole(StrEnum):
    """Role in a conversation exchange."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class PatternStatus(StrEnum):
    """Status of a behavioral pattern."""

    EMERGING = "emerging"
    ESTABLISHED = "established"
    FADING = "fading"
    ARCHIVED = "archived"


class TensionType(StrEnum):
    """Types of tensions between beliefs."""

    CONTRADICTION = "contradiction"
    TEMPORAL_CONFLICT = "temporal_conflict"
    SCOPE_CONFLICT = "scope_conflict"
    PARTIAL_OVERLAP = "partial_overlap"


class TensionSeverity(StrEnum):
    """Severity of a tension."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TensionStatus(StrEnum):
    """Status of a tension."""

    DETECTED = "detected"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"


@dataclass
class Source:
    """Provenance information for knowledge."""

    id: UUID
    type: str
    title: str | None = None
    url: str | None = None
    content_hash: str | None = None
    session_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["id"] = str(self.id)
        if self.session_id:
            result["session_id"] = str(self.session_id)
        result["created_at"] = self.created_at.isoformat()
        return result

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Source:
        """Create from database row."""
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            type=row["type"],
            title=row.get("title"),
            url=row.get("url"),
            content_hash=row.get("content_hash"),
            session_id=row["session_id"] if row.get("session_id") else None,
            metadata=row.get("metadata", {}),
            created_at=row["created_at"],
        )


@dataclass
class Entity:
    """An entity that beliefs can reference."""

    id: UUID
    name: str
    type: EntityType
    description: str | None = None
    aliases: list[str] = field(default_factory=list)
    canonical_id: UUID | None = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["id"] = str(self.id)
        result["type"] = self.type.value
        if self.canonical_id:
            result["canonical_id"] = str(self.canonical_id)
        result["created_at"] = self.created_at.isoformat()
        result["modified_at"] = self.modified_at.isoformat()
        return result

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Entity:
        """Create from database row."""
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            name=row["name"],
            type=EntityType(row["type"]),
            description=row.get("description"),
            aliases=row.get("aliases", []),
            canonical_id=row["canonical_id"] if row.get("canonical_id") else None,
            created_at=row["created_at"],
            modified_at=row["modified_at"],
        )


@dataclass
class Belief:
    """A knowledge claim with confidence and provenance."""

    id: UUID
    content: str
    confidence: DimensionalConfidence
    domain_path: list[str] = field(default_factory=list)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    source_id: UUID | None = None
    extraction_method: str | None = None
    supersedes_id: UUID | None = None
    superseded_by_id: UUID | None = None
    status: BeliefStatus = BeliefStatus.ACTIVE

    # Optional loaded relations
    source: Source | None = None
    entities: list[tuple[Entity, EntityRole]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "id": str(self.id),
            "content": self.content,
            "confidence": self.confidence.to_dict(),
            "domain_path": self.domain_path,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "source_id": str(self.source_id) if self.source_id else None,
            "extraction_method": self.extraction_method,
            "supersedes_id": str(self.supersedes_id) if self.supersedes_id else None,
            "superseded_by_id": (str(self.superseded_by_id) if self.superseded_by_id else None),
            "status": self.status.value,
        }
        if self.source:
            result["source"] = self.source.to_dict()
        if self.entities:
            result["entities"] = [{"entity": e.to_dict(), "role": r.value} for e, r in self.entities]
        return result

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Belief:
        """Create from database row."""
        confidence_data = row.get("confidence", {"overall": 0.7})
        if isinstance(confidence_data, str):
            import json

            confidence_data = json.loads(confidence_data)

        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            content=row["content"],
            confidence=DimensionalConfidence.from_dict(confidence_data),
            domain_path=row.get("domain_path", []),
            valid_from=row.get("valid_from"),
            valid_until=row.get("valid_until"),
            created_at=row["created_at"],
            modified_at=row["modified_at"],
            source_id=row["source_id"] if row.get("source_id") else None,
            extraction_method=row.get("extraction_method"),
            supersedes_id=row["supersedes_id"] if row.get("supersedes_id") else None,
            superseded_by_id=(row["superseded_by_id"] if row.get("superseded_by_id") else None),
            status=BeliefStatus(row.get("status", "active")),
        )


@dataclass
class BeliefEntity:
    """Junction record linking a belief to an entity."""

    belief_id: UUID
    entity_id: UUID
    role: EntityRole
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "belief_id": str(self.belief_id),
            "entity_id": str(self.entity_id),
            "role": self.role.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Tension:
    """A contradiction or conflict between beliefs."""

    id: UUID
    belief_a_id: UUID
    belief_b_id: UUID
    type: TensionType = TensionType.CONTRADICTION
    description: str | None = None
    severity: TensionSeverity = TensionSeverity.MEDIUM
    status: TensionStatus = TensionStatus.DETECTED
    resolution: str | None = None
    resolved_at: datetime | None = None
    detected_at: datetime = field(default_factory=datetime.now)

    # Optional loaded relations
    belief_a: Belief | None = None
    belief_b: Belief | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "id": str(self.id),
            "belief_a_id": str(self.belief_a_id),
            "belief_b_id": str(self.belief_b_id),
            "type": self.type.value,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "detected_at": self.detected_at.isoformat(),
        }
        if self.belief_a:
            result["belief_a"] = self.belief_a.to_dict()
        if self.belief_b:
            result["belief_b"] = self.belief_b.to_dict()
        return result

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Tension:
        """Create from database row."""
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            belief_a_id=(row["belief_a_id"] if isinstance(row["belief_a_id"], UUID) else UUID(row["belief_a_id"])),
            belief_b_id=(row["belief_b_id"] if isinstance(row["belief_b_id"], UUID) else UUID(row["belief_b_id"])),
            type=TensionType(row.get("type", "contradiction")),
            description=row.get("description"),
            severity=TensionSeverity(row.get("severity", "medium")),
            status=TensionStatus(row.get("status", "detected")),
            resolution=row.get("resolution"),
            resolved_at=row.get("resolved_at"),
            detected_at=row["detected_at"],
        )


@dataclass
class Session:
    """A conversation session."""

    id: UUID
    platform: Platform
    project_context: str | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    summary: str | None = None
    themes: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    claude_session_id: str | None = None
    external_room_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Stats (from view)
    exchange_count: int | None = None
    insight_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "platform": self.platform.value,
            "project_context": self.project_context,
            "status": self.status.value,
            "summary": self.summary,
            "themes": self.themes,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "claude_session_id": self.claude_session_id,
            "external_room_id": self.external_room_id,
            "metadata": self.metadata,
            "exchange_count": self.exchange_count,
            "insight_count": self.insight_count,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Session:
        """Create from database row."""
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            platform=Platform(row["platform"]),
            project_context=row.get("project_context"),
            status=SessionStatus(row.get("status", "active")),
            summary=row.get("summary"),
            themes=row.get("themes", []),
            started_at=row["started_at"],
            ended_at=row.get("ended_at"),
            claude_session_id=row.get("claude_session_id"),
            external_room_id=row.get("external_room_id"),
            metadata=row.get("metadata", {}),
            exchange_count=row.get("exchange_count"),
            insight_count=row.get("insight_count"),
        )


@dataclass
class Exchange:
    """A single turn in a conversation."""

    id: UUID
    session_id: UUID
    sequence: int
    role: ExchangeRole
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    tokens_approx: int | None = None
    tool_uses: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "sequence": self.sequence,
            "role": self.role.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "tokens_approx": self.tokens_approx,
            "tool_uses": self.tool_uses,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Exchange:
        """Create from database row."""
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            session_id=(row["session_id"] if isinstance(row["session_id"], UUID) else UUID(row["session_id"])),
            sequence=row["sequence"],
            role=ExchangeRole(row["role"]),
            content=row["content"],
            created_at=row["created_at"],
            tokens_approx=row.get("tokens_approx"),
            tool_uses=row.get("tool_uses", []),
        )


@dataclass
class Pattern:
    """A behavioral pattern observed across sessions."""

    id: UUID
    type: str
    description: str
    evidence: list[UUID] = field(default_factory=list)
    occurrence_count: int = 1
    confidence: float = 0.5
    status: PatternStatus = PatternStatus.EMERGING
    first_observed: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "type": self.type,
            "description": self.description,
            "evidence": [str(e) for e in self.evidence],
            "occurrence_count": self.occurrence_count,
            "confidence": float(self.confidence),
            "status": self.status.value,
            "first_observed": self.first_observed.isoformat(),
            "last_observed": self.last_observed.isoformat(),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Pattern:
        """Create from database row."""
        raw_evidence = row.get("evidence", [])

        # psycopg2 returns UUID[] columns as raw PostgreSQL array strings
        # e.g. '{}' for empty, '{uuid1,uuid2}' for populated
        if isinstance(raw_evidence, str):
            inner = raw_evidence.strip("{}")
            evidence = [UUID(e) for e in inner.split(",") if e] if inner else []
        elif isinstance(raw_evidence, list) and raw_evidence and isinstance(raw_evidence[0], str):
            evidence = [UUID(e) for e in raw_evidence]
        else:
            evidence = list(raw_evidence) if raw_evidence else []

        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            type=row["type"],
            description=row["description"],
            evidence=evidence,
            occurrence_count=row.get("occurrence_count", 1),
            confidence=float(row.get("confidence", 0.5)),
            status=PatternStatus(row.get("status", "emerging")),
            first_observed=row["first_observed"],
            last_observed=row["last_observed"],
        )


@dataclass
class SessionInsight:
    """Link between a session and an extracted belief."""

    id: UUID
    session_id: UUID
    belief_id: UUID
    extraction_method: str = "manual"
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "belief_id": str(self.belief_id),
            "extraction_method": self.extraction_method,
            "extracted_at": self.extracted_at.isoformat(),
        }
