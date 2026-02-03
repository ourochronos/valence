"""Data models for Valence Federation.

These models represent federation concepts like nodes, federated beliefs,
trust relationships, and aggregation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID

from ..core.confidence import DimensionalConfidence


# =============================================================================
# ENUMS
# =============================================================================


class NodeStatus(str, Enum):
    """Status of a federation node."""
    DISCOVERED = "discovered"    # Found but not yet connected
    CONNECTING = "connecting"    # Connection in progress
    ACTIVE = "active"            # Connected and syncing
    SUSPENDED = "suspended"      # Temporarily suspended
    UNREACHABLE = "unreachable"  # Cannot connect


class TrustPhase(str, Enum):
    """Trust establishment phase for a node."""
    OBSERVER = "observer"        # Days 1-7: Read-only
    CONTRIBUTOR = "contributor"  # Days 7-30: Limited contribution
    PARTICIPANT = "participant"  # Day 30+: Full participation
    ANCHOR = "anchor"            # Earned: Can vouch for others


class Visibility(str, Enum):
    """Visibility level for beliefs."""
    PRIVATE = "private"          # Never shared
    TRUSTED = "trusted"          # Shared with explicit trust
    FEDERATED = "federated"      # Shared across federation
    PUBLIC = "public"            # Discoverable by anyone


class ShareLevel(str, Enum):
    """What information is shared with a belief."""
    BELIEF_ONLY = "belief_only"          # Content + confidence only
    WITH_PROVENANCE = "with_provenance"  # + source information
    FULL = "full"                        # + user attribution, metadata


class SyncStatus(str, Enum):
    """Status of sync with a peer node."""
    IDLE = "idle"
    SYNCING = "syncing"
    ERROR = "error"
    PAUSED = "paused"


class ThreatLevel(str, Enum):
    """Threat level for a node's behavior."""
    NONE = "none"
    LOW = "low"           # Increased scrutiny
    MEDIUM = "medium"     # Reduced influence
    HIGH = "high"         # Quarantine from sensitive ops
    CRITICAL = "critical"  # Functional isolation


class ResolutionProposal(str, Enum):
    """Proposed resolution for a tension."""
    SUPERSEDE_A = "supersede_a"
    SUPERSEDE_B = "supersede_b"
    ACCEPT_BOTH = "accept_both"
    MERGE = "merge"
    REFER_TO_AUTHORITY = "refer_to_authority"


class ResolutionStatus(str, Enum):
    """Status of a tension resolution."""
    PROPOSED = "proposed"
    VOTING = "voting"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


class ConsensusMethod(str, Enum):
    """Method for reaching consensus on resolutions."""
    TRUST_WEIGHTED = "trust_weighted"
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    AUTHORITY = "authority"


class Vote(str, Enum):
    """Vote on a resolution."""
    SUPPORT = "support"
    OPPOSE = "oppose"
    ABSTAIN = "abstain"


class TrustPreference(str, Enum):
    """User preference for node trust."""
    BLOCKED = "blocked"
    REDUCED = "reduced"
    AUTOMATIC = "automatic"
    ELEVATED = "elevated"
    ANCHOR = "anchor"


class AnnotationType(str, Enum):
    """Type of belief trust annotation."""
    CORROBORATION = "corroboration"
    DISPUTE = "dispute"
    ENDORSEMENT = "endorsement"
    FLAG = "flag"


# =============================================================================
# FEDERATION NODE
# =============================================================================


@dataclass
class FederationNode:
    """A node in the federation network."""

    id: UUID
    did: str  # did:vkb:web:domain or did:vkb:key:z6Mk...

    # Connection info
    federation_endpoint: str | None = None
    mcp_endpoint: str | None = None

    # Cryptographic identity
    public_key_multibase: str = ""

    # Profile
    name: str | None = None
    domains: list[str] = field(default_factory=list)

    # Capabilities
    capabilities: list[str] = field(default_factory=list)

    # Status
    status: NodeStatus = NodeStatus.DISCOVERED
    trust_phase: TrustPhase = TrustPhase.OBSERVER
    phase_started_at: datetime = field(default_factory=datetime.now)

    # Protocol
    protocol_version: str = "1.0"

    # Timestamps
    discovered_at: datetime = field(default_factory=datetime.now)
    last_seen_at: datetime | None = None
    last_sync_at: datetime | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "did": self.did,
            "federation_endpoint": self.federation_endpoint,
            "mcp_endpoint": self.mcp_endpoint,
            "public_key_multibase": self.public_key_multibase,
            "name": self.name,
            "domains": self.domains,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "trust_phase": self.trust_phase.value,
            "phase_started_at": self.phase_started_at.isoformat(),
            "protocol_version": self.protocol_version,
            "discovered_at": self.discovered_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> FederationNode:
        """Create from database row."""
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            did=row["did"],
            federation_endpoint=row.get("federation_endpoint"),
            mcp_endpoint=row.get("mcp_endpoint"),
            public_key_multibase=row.get("public_key_multibase", ""),
            name=row.get("name"),
            domains=row.get("domains", []),
            capabilities=row.get("capabilities", []),
            status=NodeStatus(row.get("status", "discovered")),
            trust_phase=TrustPhase(row.get("trust_phase", "observer")),
            phase_started_at=row.get("phase_started_at", datetime.now()),
            protocol_version=row.get("protocol_version", "1.0"),
            discovered_at=row.get("discovered_at", datetime.now()),
            last_seen_at=row.get("last_seen_at"),
            last_sync_at=row.get("last_sync_at"),
            metadata=row.get("metadata", {}),
        )


# =============================================================================
# FEDERATED BELIEF
# =============================================================================


@dataclass
class FederatedBelief:
    """A belief wrapped for federation with cryptographic provenance."""

    # Local identity
    id: UUID
    federation_id: UUID  # Stable across federation
    origin_node_did: str

    # Content (same as local Belief)
    content: str
    confidence: DimensionalConfidence
    domain_path: list[str] = field(default_factory=list)
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Federation metadata
    visibility: Visibility = Visibility.FEDERATED
    share_level: ShareLevel = ShareLevel.BELIEF_ONLY
    hop_count: int = 0
    federation_path: list[str] = field(default_factory=list)

    # Cryptographic proof
    origin_signature: str = ""
    signed_at: datetime = field(default_factory=datetime.now)
    signature_method: str = "Ed25519Signature2020"

    # Source info (if share_level >= with_provenance)
    source_type: str | None = None
    source_title: str | None = None
    source_url: str | None = None

    # Attribution (if share_level == full)
    user_did: str | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": str(self.id),
            "federation_id": str(self.federation_id),
            "origin_node_did": self.origin_node_did,
            "content": self.content,
            "confidence": self.confidence.to_dict(),
            "domain_path": self.domain_path,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "visibility": self.visibility.value,
            "share_level": self.share_level.value,
            "hop_count": self.hop_count,
            "federation_path": self.federation_path,
            "origin_signature": self.origin_signature,
            "signed_at": self.signed_at.isoformat(),
            "signature_method": self.signature_method,
        }

        if self.share_level in (ShareLevel.WITH_PROVENANCE, ShareLevel.FULL):
            result["source"] = {
                "type": self.source_type,
                "title": self.source_title,
                "url": self.source_url,
            }

        if self.share_level == ShareLevel.FULL:
            result["attribution"] = {
                "user_did": self.user_did,
                "created_at": self.created_at.isoformat() if self.created_at else None,
            }

        return result

    def to_signable_content(self) -> dict[str, Any]:
        """Return content for signature generation (canonical form)."""
        return {
            "federation_id": str(self.federation_id),
            "origin_node_did": self.origin_node_did,
            "content": self.content,
            "confidence": self.confidence.to_dict(),
            "domain_path": self.domain_path,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }


# =============================================================================
# BELIEF PROVENANCE
# =============================================================================


@dataclass
class BeliefProvenance:
    """Tracks federation path of beliefs received from other nodes."""

    id: UUID
    belief_id: UUID
    federation_id: UUID
    origin_node_id: UUID
    origin_belief_id: UUID

    # Cryptographic proof
    origin_signature: str
    signed_at: datetime
    signature_verified: bool = False

    # Federation path
    hop_count: int = 1
    federation_path: list[str] = field(default_factory=list)

    # Share level
    share_level: ShareLevel = ShareLevel.BELIEF_ONLY

    # Reception
    received_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "belief_id": str(self.belief_id),
            "federation_id": str(self.federation_id),
            "origin_node_id": str(self.origin_node_id),
            "origin_belief_id": str(self.origin_belief_id),
            "origin_signature": self.origin_signature,
            "signed_at": self.signed_at.isoformat(),
            "signature_verified": self.signature_verified,
            "hop_count": self.hop_count,
            "federation_path": self.federation_path,
            "share_level": self.share_level.value,
            "received_at": self.received_at.isoformat(),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> BeliefProvenance:
        """Create from database row."""
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            belief_id=row["belief_id"] if isinstance(row["belief_id"], UUID) else UUID(row["belief_id"]),
            federation_id=row["federation_id"] if isinstance(row["federation_id"], UUID) else UUID(row["federation_id"]),
            origin_node_id=row["origin_node_id"] if isinstance(row["origin_node_id"], UUID) else UUID(row["origin_node_id"]),
            origin_belief_id=row["origin_belief_id"] if isinstance(row["origin_belief_id"], UUID) else UUID(row["origin_belief_id"]),
            origin_signature=row["origin_signature"],
            signed_at=row["signed_at"],
            signature_verified=row.get("signature_verified", False),
            hop_count=row.get("hop_count", 1),
            federation_path=row.get("federation_path", []),
            share_level=ShareLevel(row.get("share_level", "belief_only")),
            received_at=row.get("received_at", datetime.now()),
        )


# =============================================================================
# NODE TRUST
# =============================================================================


# Default weights for computing overall trust (mirrors DimensionalConfidence)
TRUST_WEIGHTS: dict[str, float] = {
    "belief_accuracy": 0.30,
    "extraction_quality": 0.15,
    "curation_accuracy": 0.10,
    "uptime_reliability": 0.10,
    "contribution_consistency": 0.15,
    "endorsement_strength": 0.15,
}


@dataclass
class NodeTrust:
    """Trust dimensions for a federation node."""

    id: UUID
    node_id: UUID

    # Trust dimensions (stored as JSONB)
    overall: float = 0.1
    belief_accuracy: float | None = None
    extraction_quality: float | None = None
    curation_accuracy: float | None = None
    uptime_reliability: float | None = None
    contribution_consistency: float | None = None
    endorsement_strength: float | None = None
    domain_expertise: dict[str, float] = field(default_factory=dict)

    # Trust factors
    beliefs_received: int = 0
    beliefs_corroborated: int = 0
    beliefs_disputed: int = 0
    sync_requests_served: int = 0
    aggregation_participations: int = 0

    # Social trust
    endorsements_received: int = 0
    endorsements_given: int = 0

    # Relationship timeline
    relationship_started_at: datetime = field(default_factory=datetime.now)
    last_interaction_at: datetime | None = None

    # Manual adjustments
    manual_trust_adjustment: float = 0.0
    adjustment_reason: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate trust values."""
        if self.overall < 0 or self.overall > 1:
            raise ValueError(f"overall trust must be between 0 and 1, got {self.overall}")

    def recalculate_overall(
        self,
        weights: dict[str, float] | None = None
    ) -> NodeTrust:
        """Recalculate overall trust from dimensions."""
        w = weights or TRUST_WEIGHTS
        weighted_sum = 0.0
        total_weight = 0.0

        for dim, weight in w.items():
            value = getattr(self, dim, None)
            if value is not None:
                weighted_sum += value * weight
                total_weight += weight

        # Age bonus (capped at 0.05 after 90 days)
        age_days = (datetime.now() - self.relationship_started_at).days
        age_bonus = min(0.05, age_days / 90 * 0.05)

        if total_weight > 0:
            self.overall = min(1.0, (weighted_sum / total_weight) + age_bonus + self.manual_trust_adjustment)
        else:
            self.overall = 0.1 + age_bonus + self.manual_trust_adjustment

        self.overall = max(0.0, min(1.0, self.overall))
        self.modified_at = datetime.now()
        return self

    def with_dimension(
        self,
        dimension: str,
        value: float,
        recalculate: bool = True,
    ) -> NodeTrust:
        """Return trust with one dimension updated."""
        if value < 0 or value > 1:
            raise ValueError(f"{dimension} must be between 0 and 1, got {value}")

        setattr(self, dimension, value)
        if recalculate:
            self.recalculate_overall()
        return self

    def get_domain_trust(self, domain: str) -> float:
        """Get trust for a specific domain."""
        return self.domain_expertise.get(domain, self.overall)

    def to_trust_dict(self) -> dict[str, Any]:
        """Convert trust dimensions to dictionary (for JSONB storage)."""
        result: dict[str, Any] = {"overall": self.overall}
        for dim in TRUST_WEIGHTS:
            value = getattr(self, dim, None)
            if value is not None:
                result[dim] = value
        if self.domain_expertise:
            result["domain_expertise"] = self.domain_expertise
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to full dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "node_id": str(self.node_id),
            "trust": self.to_trust_dict(),
            "beliefs_received": self.beliefs_received,
            "beliefs_corroborated": self.beliefs_corroborated,
            "beliefs_disputed": self.beliefs_disputed,
            "sync_requests_served": self.sync_requests_served,
            "aggregation_participations": self.aggregation_participations,
            "endorsements_received": self.endorsements_received,
            "endorsements_given": self.endorsements_given,
            "relationship_started_at": self.relationship_started_at.isoformat(),
            "last_interaction_at": self.last_interaction_at.isoformat() if self.last_interaction_at else None,
            "manual_trust_adjustment": float(self.manual_trust_adjustment),
            "adjustment_reason": self.adjustment_reason,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> NodeTrust:
        """Create from database row."""
        trust_data = row.get("trust", {"overall": 0.1})
        if isinstance(trust_data, str):
            import json
            trust_data = json.loads(trust_data)

        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            node_id=row["node_id"] if isinstance(row["node_id"], UUID) else UUID(row["node_id"]),
            overall=trust_data.get("overall", 0.1),
            belief_accuracy=trust_data.get("belief_accuracy"),
            extraction_quality=trust_data.get("extraction_quality"),
            curation_accuracy=trust_data.get("curation_accuracy"),
            uptime_reliability=trust_data.get("uptime_reliability"),
            contribution_consistency=trust_data.get("contribution_consistency"),
            endorsement_strength=trust_data.get("endorsement_strength"),
            domain_expertise=trust_data.get("domain_expertise", {}),
            beliefs_received=row.get("beliefs_received", 0),
            beliefs_corroborated=row.get("beliefs_corroborated", 0),
            beliefs_disputed=row.get("beliefs_disputed", 0),
            sync_requests_served=row.get("sync_requests_served", 0),
            aggregation_participations=row.get("aggregation_participations", 0),
            endorsements_received=row.get("endorsements_received", 0),
            endorsements_given=row.get("endorsements_given", 0),
            relationship_started_at=row.get("relationship_started_at", datetime.now()),
            last_interaction_at=row.get("last_interaction_at"),
            manual_trust_adjustment=float(row.get("manual_trust_adjustment", 0)),
            adjustment_reason=row.get("adjustment_reason"),
            created_at=row.get("created_at", datetime.now()),
            modified_at=row.get("modified_at", datetime.now()),
        )


# =============================================================================
# USER NODE TRUST
# =============================================================================


@dataclass
class UserNodeTrust:
    """User preference overrides for node trust."""

    id: UUID
    node_id: UUID

    trust_preference: TrustPreference = TrustPreference.AUTOMATIC
    manual_trust_score: float | None = None
    reason: str | None = None
    domain_overrides: dict[str, str] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    def get_effective_preference(self, domain: str | None = None) -> TrustPreference:
        """Get effective trust preference, considering domain overrides."""
        if domain and domain in self.domain_overrides:
            return TrustPreference(self.domain_overrides[domain])
        return self.trust_preference

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "node_id": str(self.node_id),
            "trust_preference": self.trust_preference.value,
            "manual_trust_score": self.manual_trust_score,
            "reason": self.reason,
            "domain_overrides": self.domain_overrides,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> UserNodeTrust:
        """Create from database row."""
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            node_id=row["node_id"] if isinstance(row["node_id"], UUID) else UUID(row["node_id"]),
            trust_preference=TrustPreference(row.get("trust_preference", "automatic")),
            manual_trust_score=row.get("manual_trust_score"),
            reason=row.get("reason"),
            domain_overrides=row.get("domain_overrides", {}),
            created_at=row.get("created_at", datetime.now()),
            modified_at=row.get("modified_at", datetime.now()),
        )


# =============================================================================
# BELIEF TRUST ANNOTATION
# =============================================================================


@dataclass
class BeliefTrustAnnotation:
    """Per-belief trust adjustments from federation context."""

    id: UUID
    belief_id: UUID
    type: AnnotationType
    source_node_id: UUID | None = None

    corroboration_attestation: dict[str, Any] | None = None
    confidence_delta: float = 0.0

    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "belief_id": str(self.belief_id),
            "type": self.type.value,
            "source_node_id": str(self.source_node_id) if self.source_node_id else None,
            "corroboration_attestation": self.corroboration_attestation,
            "confidence_delta": float(self.confidence_delta),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> BeliefTrustAnnotation:
        """Create from database row."""
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            belief_id=row["belief_id"] if isinstance(row["belief_id"], UUID) else UUID(row["belief_id"]),
            type=AnnotationType(row["type"]),
            source_node_id=UUID(row["source_node_id"]) if row.get("source_node_id") else None,
            corroboration_attestation=row.get("corroboration_attestation"),
            confidence_delta=float(row.get("confidence_delta", 0)),
            created_at=row.get("created_at", datetime.now()),
            expires_at=row.get("expires_at"),
        )


# =============================================================================
# AGGREGATION
# =============================================================================


@dataclass
class AggregatedBelief:
    """Privacy-preserving aggregate from federation queries."""

    id: UUID
    query_hash: str
    query_domain: list[str]
    query_semantic: str | None = None

    # Results
    collective_confidence: float = 0.0
    agreement_score: float | None = None
    contributor_count: int = 0
    node_count: int = 0
    total_belief_count: int | None = None

    # Summary
    stance_summary: str | None = None
    key_factors: list[str] = field(default_factory=list)

    # Privacy
    privacy_epsilon: float = 0.1
    privacy_delta: float = 1e-6
    privacy_mechanism: str = "laplace"

    # Aggregator
    aggregator_node_id: UUID | None = None

    # Validity
    computed_at: datetime = field(default_factory=datetime.now)
    valid_until: datetime | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "query_hash": self.query_hash,
            "query_domain": self.query_domain,
            "query_semantic": self.query_semantic,
            "collective_confidence": float(self.collective_confidence),
            "agreement_score": float(self.agreement_score) if self.agreement_score else None,
            "contributor_count": self.contributor_count,
            "node_count": self.node_count,
            "total_belief_count": self.total_belief_count,
            "stance_summary": self.stance_summary,
            "key_factors": self.key_factors,
            "privacy_guarantees": {
                "epsilon": float(self.privacy_epsilon),
                "delta": float(self.privacy_delta),
                "mechanism": self.privacy_mechanism,
            },
            "aggregator_node_id": str(self.aggregator_node_id) if self.aggregator_node_id else None,
            "computed_at": self.computed_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "metadata": self.metadata,
        }


@dataclass
class AggregationSource:
    """Links aggregated beliefs to anonymized sources."""

    id: UUID
    aggregated_belief_id: UUID
    source_hash: str  # SHA256(node_did + salt), not reversible

    contribution_weight: float = 0.0
    local_confidence: float | None = None
    local_belief_count: int | None = None

    contributed_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# TENSION RESOLUTION
# =============================================================================


@dataclass
class TensionResolution:
    """Cross-node conflict resolution."""

    id: UUID
    tension_id: UUID | None = None
    is_cross_node: bool = False

    node_a_id: UUID | None = None
    node_b_id: UUID | None = None

    proposed_resolution: ResolutionProposal = ResolutionProposal.ACCEPT_BOTH
    resolution_rationale: str | None = None

    proposed_by_node_id: UUID | None = None
    proposed_at: datetime = field(default_factory=datetime.now)

    consensus_method: ConsensusMethod = ConsensusMethod.TRUST_WEIGHTED
    consensus_threshold: float = 0.6
    current_support: float = 0.0

    status: ResolutionStatus = ResolutionStatus.PROPOSED
    resolved_at: datetime | None = None

    winning_belief_id: UUID | None = None
    superseded_belief_id: UUID | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "tension_id": str(self.tension_id) if self.tension_id else None,
            "is_cross_node": self.is_cross_node,
            "node_a_id": str(self.node_a_id) if self.node_a_id else None,
            "node_b_id": str(self.node_b_id) if self.node_b_id else None,
            "proposed_resolution": self.proposed_resolution.value,
            "resolution_rationale": self.resolution_rationale,
            "proposed_by_node_id": str(self.proposed_by_node_id) if self.proposed_by_node_id else None,
            "proposed_at": self.proposed_at.isoformat(),
            "consensus_method": self.consensus_method.value,
            "consensus_threshold": float(self.consensus_threshold),
            "current_support": float(self.current_support),
            "status": self.status.value,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "winning_belief_id": str(self.winning_belief_id) if self.winning_belief_id else None,
            "superseded_belief_id": str(self.superseded_belief_id) if self.superseded_belief_id else None,
        }


@dataclass
class ConsensusVote:
    """Individual vote on a tension resolution."""

    id: UUID
    resolution_id: UUID
    voter_node_id: UUID

    vote: Vote
    vote_weight: float = 0.0
    rationale: str | None = None
    signature: str = ""

    voted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "resolution_id": str(self.resolution_id),
            "voter_node_id": str(self.voter_node_id),
            "vote": self.vote.value,
            "vote_weight": float(self.vote_weight),
            "rationale": self.rationale,
            "voted_at": self.voted_at.isoformat(),
        }


# =============================================================================
# SYNC
# =============================================================================


@dataclass
class SyncState:
    """Synchronization state with a peer node."""

    id: UUID
    node_id: UUID

    last_received_cursor: str | None = None
    last_sent_cursor: str | None = None
    vector_clock: dict[str, int] = field(default_factory=dict)

    status: SyncStatus = SyncStatus.IDLE

    beliefs_sent: int = 0
    beliefs_received: int = 0
    last_sync_duration_ms: int | None = None

    last_error: str | None = None
    error_count: int = 0

    last_sync_at: datetime | None = None
    next_sync_scheduled: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "node_id": str(self.node_id),
            "last_received_cursor": self.last_received_cursor,
            "last_sent_cursor": self.last_sent_cursor,
            "vector_clock": self.vector_clock,
            "status": self.status.value,
            "beliefs_sent": self.beliefs_sent,
            "beliefs_received": self.beliefs_received,
            "last_sync_duration_ms": self.last_sync_duration_ms,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "next_sync_scheduled": self.next_sync_scheduled.isoformat() if self.next_sync_scheduled else None,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }


@dataclass
class SyncEvent:
    """Audit log entry for sync operations."""

    id: UUID
    node_id: UUID
    event_type: str
    details: dict[str, Any] = field(default_factory=dict)
    direction: str = "inbound"
    belief_id: UUID | None = None
    occurred_at: datetime = field(default_factory=datetime.now)


@dataclass
class SyncOutboundItem:
    """Pending outbound sync operation."""

    id: UUID
    target_node_id: UUID | None = None
    operation: str = "share_belief"
    belief_id: UUID | None = None
    payload: dict[str, Any] | None = None

    priority: int = 5
    status: str = "pending"

    attempts: int = 0
    max_attempts: int = 3
    last_attempt_at: datetime | None = None
    last_error: str | None = None

    created_at: datetime = field(default_factory=datetime.now)
    scheduled_for: datetime = field(default_factory=datetime.now)


# =============================================================================
# PRIVACY
# =============================================================================


@dataclass
class PrivacyParameters:
    """Privacy parameters for aggregation queries."""

    epsilon: float = 0.1  # Privacy loss per query
    delta: float = 1e-6   # Probability of complete privacy failure
    min_contributors: int = 5  # Minimum nodes for aggregate
    max_queries_per_period: int = 100  # Rate limit
    budget_period: timedelta = field(default_factory=lambda: timedelta(days=1))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "min_contributors": self.min_contributors,
            "max_queries_per_period": self.max_queries_per_period,
            "budget_period_seconds": self.budget_period.total_seconds(),
        }


# =============================================================================
# ATTESTATIONS
# =============================================================================


@dataclass
class CorroborationAttestation:
    """Attestation that a claim has been corroborated by multiple nodes."""

    claim_hash: str  # SHA256 of normalized claim
    corroboration_level: float  # 0-1
    participating_nodes_count: int
    confidence_boost: float  # Amount to boost matching beliefs

    domain: list[str] = field(default_factory=list)
    issued_at: datetime = field(default_factory=datetime.now)
    valid_until: datetime | None = None

    aggregator_signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim_hash": self.claim_hash,
            "corroboration_level": float(self.corroboration_level),
            "participating_nodes_count": self.participating_nodes_count,
            "confidence_boost": float(self.confidence_boost),
            "domain": self.domain,
            "issued_at": self.issued_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }


@dataclass
class TrustAttestation:
    """Signed statement of trust from one node to another."""

    issuer_did: str
    subject_did: str
    attestation_type: str  # endorsement, domain_expertise, vouching

    attested_dimensions: dict[str, float] = field(default_factory=dict)
    domains: list[str] | None = None

    issued_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None

    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issuer_did": self.issuer_did,
            "subject_did": self.subject_did,
            "attestation_type": self.attestation_type,
            "attested_dimensions": self.attested_dimensions,
            "domains": self.domains,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


# =============================================================================
# AGGREGATION QUERY/RESULT
# =============================================================================


@dataclass
class AggregationQuery:
    """Query for privacy-preserving aggregation."""

    request_id: UUID
    aggregator_did: str

    domain_filter: list[str] = field(default_factory=list)
    semantic_query: str | None = None
    aggregation_type: str = "stance_summary"

    privacy_parameters: PrivacyParameters = field(default_factory=PrivacyParameters)
    deadline: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": str(self.request_id),
            "aggregator_did": self.aggregator_did,
            "query": {
                "domain_filter": self.domain_filter,
                "semantic_query": self.semantic_query,
                "aggregation_type": self.aggregation_type,
            },
            "privacy_parameters": self.privacy_parameters.to_dict(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
        }


@dataclass
class LocalSummary:
    """Node's response to an aggregation query (with DP noise)."""

    node_did: str
    request_id: UUID

    stats: dict[str, Any] = field(default_factory=dict)
    privacy_budget_used: float = 0.0
    contributor_count: int = 0  # 0 if below threshold

    computed_at: datetime = field(default_factory=datetime.now)
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_did": self.node_did,
            "request_id": str(self.request_id),
            "summary": self.stats,
            "privacy_budget_used": self.privacy_budget_used,
            "contributor_count": self.contributor_count,
            "computed_at": self.computed_at.isoformat(),
        }


@dataclass
class AggregationResult:
    """Final aggregation result (published to requesters)."""

    request_id: UUID

    collective_confidence: float = 0.0
    agreement_score: float | None = None
    contributor_count: int = 0
    node_count: int = 0

    stance_summary: str | None = None
    key_factors: list[str] = field(default_factory=list)

    privacy_guarantees: dict[str, Any] = field(default_factory=dict)
    computed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": str(self.request_id),
            "result": {
                "collective_confidence": float(self.collective_confidence),
                "agreement_score": float(self.agreement_score) if self.agreement_score else None,
                "contributor_count": self.contributor_count,
                "node_count": self.node_count,
                "stance_summary": self.stance_summary,
                "key_factors": self.key_factors,
            },
            "privacy_guarantees": self.privacy_guarantees,
            "computed_at": self.computed_at.isoformat(),
        }
