# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Resource models for shared operational knowledge.

Resources represent shareable operational knowledge like prompts, configs,
and patterns. Unlike articles (which are true/false claims), resources are
measured by usefulness and carry risks like prompt injection and data
exfiltration rather than misinformation.

Part of Issue #270: Resource sharing with trust gating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


class ResourceType(StrEnum):
    """Type of shared resource."""

    PROMPT = "prompt"  # Prompt templates, system prompts
    CONFIG = "config"  # Configuration snippets, settings
    PATTERN = "pattern"  # Behavioral patterns, workflows


class SafetyStatus(StrEnum):
    """Safety review status for a resource."""

    UNREVIEWED = "unreviewed"  # Not yet reviewed
    SAFE = "safe"  # Passed safety checks
    SUSPICIOUS = "suspicious"  # Flagged for review
    BLOCKED = "blocked"  # Blocked from sharing


@dataclass
class ResourceReport:
    """A report/flag against a resource."""

    id: UUID
    resource_id: UUID
    reporter_did: str
    reason: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "resource_id": str(self.resource_id),
            "reporter_did": self.reporter_did,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class UsageAttestation:
    """Record of a resource being used, with optional feedback.

    Usage attestations serve the same role for resources that
    corroboration does for articles — they validate quality through use.
    """

    id: UUID
    resource_id: UUID
    user_did: str
    success: bool = True
    feedback: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "resource_id": str(self.resource_id),
            "user_did": self.user_did,
            "success": self.success,
            "feedback": self.feedback,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Resource:
    """A shareable piece of operational knowledge.

    Resources differ from articles:
    - Articles are true/false claims; resources are useful/harmful artifacts.
    - Articles risk misinformation; resources risk prompt injection & data exfil.
    - Articles are validated by corroboration; resources by usage attestations.
    """

    id: UUID
    type: ResourceType
    content: str
    author_did: str

    # Descriptive metadata
    name: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Safety & trust gating
    safety_status: SafetyStatus = SafetyStatus.UNREVIEWED
    trust_level_required: float = 0.5  # Minimum trust to access
    report_count: int = 0

    # Usage tracking
    usage_count: int = 0
    success_rate: float | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "content": self.content,
            "author_did": self.author_did,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
            "safety_status": self.safety_status.value,
            "trust_level_required": self.trust_level_required,
            "report_count": self.report_count,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Resource:
        """Create from a dictionary."""
        return cls(
            id=data["id"] if isinstance(data["id"], UUID) else UUID(data["id"]),
            type=ResourceType(data["type"]),
            content=data["content"],
            author_did=data["author_did"],
            name=data.get("name"),
            description=data.get("description"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            safety_status=SafetyStatus(data.get("safety_status", "unreviewed")),
            trust_level_required=float(data.get("trust_level_required", 0.5)),
            report_count=int(data.get("report_count", 0)),
            usage_count=int(data.get("usage_count", 0)),
            success_rate=(float(data["success_rate"]) if data.get("success_rate") is not None else None),
            created_at=(
                data["created_at"]
                if isinstance(data.get("created_at"), datetime)
                else datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            modified_at=(
                data["modified_at"]
                if isinstance(data.get("modified_at"), datetime)
                else datetime.fromisoformat(data["modified_at"])
                if data.get("modified_at")
                else datetime.now()
            ),
        )
