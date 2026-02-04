"""Elevation proposal workflow for Valence privacy.

Implements Issue #94: Content can be elevated from private to more public
through proposals. Owners can approve/reject elevation requests, optionally
with redacted or transformed versions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
import uuid

from .types import ShareLevel


class ProposalStatus(Enum):
    """Status of an elevation proposal."""
    
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ElevationError(Exception):
    """Base exception for elevation operations."""
    pass


class ProposalNotFoundError(ElevationError):
    """Raised when a proposal is not found."""
    pass


class ProposalAlreadyResolvedError(ElevationError):
    """Raised when trying to approve/reject an already resolved proposal."""
    pass


class InvalidElevationError(ElevationError):
    """Raised when an elevation request is invalid."""
    pass


@dataclass
class ElevationProposal:
    """A proposal to elevate content from one privacy level to another.
    
    Attributes:
        proposal_id: Unique identifier for this proposal
        proposer: DID of the agent proposing the elevation
        belief_id: ID of the belief to be elevated
        from_level: Current privacy level
        to_level: Target privacy level
        reason: Human-readable explanation for the elevation request
        status: Current status of the proposal
        created_at: When the proposal was created
        resolved_at: When the proposal was approved/rejected (if resolved)
        resolved_by: DID of who approved/rejected (if resolved)
        rejection_reason: Explanation if rejected
        redacted_content: Optional redacted/transformed version for elevated sharing
        redaction_notes: Description of what was redacted/transformed
        metadata: Additional metadata for the proposal
    """
    
    proposal_id: str
    proposer: str
    belief_id: str
    from_level: ShareLevel
    to_level: ShareLevel
    reason: str
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    redacted_content: Optional[bytes] = None
    redaction_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "proposal_id": self.proposal_id,
            "proposer": self.proposer,
            "belief_id": self.belief_id,
            "from_level": self.from_level.value,
            "to_level": self.to_level.value,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "rejection_reason": self.rejection_reason,
            "redacted_content": self.redacted_content.decode("utf-8") if self.redacted_content else None,
            "redaction_notes": self.redaction_notes,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElevationProposal":
        """Deserialize from dictionary."""
        return cls(
            proposal_id=data["proposal_id"],
            proposer=data["proposer"],
            belief_id=data["belief_id"],
            from_level=ShareLevel(data["from_level"]),
            to_level=ShareLevel(data["to_level"]),
            reason=data["reason"],
            status=ProposalStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            resolved_by=data.get("resolved_by"),
            rejection_reason=data.get("rejection_reason"),
            redacted_content=data["redacted_content"].encode("utf-8") if data.get("redacted_content") else None,
            redaction_notes=data.get("redaction_notes"),
            metadata=data.get("metadata", {}),
        )
    
    @property
    def is_pending(self) -> bool:
        """Check if proposal is still pending."""
        return self.status == ProposalStatus.PENDING
    
    @property
    def is_resolved(self) -> bool:
        """Check if proposal has been resolved (approved or rejected)."""
        return self.status in (ProposalStatus.APPROVED, ProposalStatus.REJECTED)
    
    @property
    def is_approved(self) -> bool:
        """Check if proposal was approved."""
        return self.status == ProposalStatus.APPROVED
    
    @property
    def is_rejected(self) -> bool:
        """Check if proposal was rejected."""
        return self.status == ProposalStatus.REJECTED


# Level ordering for validation (lower index = more private)
_LEVEL_ORDER = [
    ShareLevel.PRIVATE,
    ShareLevel.DIRECT,
    ShareLevel.BOUNDED,
    ShareLevel.CASCADING,
    ShareLevel.PUBLIC,
]


def _level_index(level: ShareLevel) -> int:
    """Get the ordering index of a share level."""
    return _LEVEL_ORDER.index(level)


def _is_valid_elevation(from_level: ShareLevel, to_level: ShareLevel) -> bool:
    """Check if this is a valid elevation (to_level is more public than from_level)."""
    return _level_index(to_level) > _level_index(from_level)


def propose_elevation(
    proposer: str,
    belief_id: str,
    from_level: ShareLevel,
    to_level: ShareLevel,
    reason: str,
    redacted_content: Optional[bytes] = None,
    redaction_notes: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ElevationProposal:
    """Create an elevation proposal.
    
    Args:
        proposer: DID of the agent proposing the elevation
        belief_id: ID of the belief to elevate
        from_level: Current privacy level of the belief
        to_level: Target privacy level (must be more public than from_level)
        reason: Human-readable explanation for the request
        redacted_content: Optional pre-redacted version to share if approved
        redaction_notes: Description of what was redacted/transformed
        metadata: Additional metadata to attach to the proposal
        
    Returns:
        A new ElevationProposal in PENDING status
        
    Raises:
        InvalidElevationError: If to_level is not more public than from_level
    """
    if not _is_valid_elevation(from_level, to_level):
        raise InvalidElevationError(
            f"Cannot elevate from {from_level.value} to {to_level.value}: "
            f"target level must be more public than source level"
        )
    
    return ElevationProposal(
        proposal_id=str(uuid.uuid4()),
        proposer=proposer,
        belief_id=belief_id,
        from_level=from_level,
        to_level=to_level,
        reason=reason,
        redacted_content=redacted_content,
        redaction_notes=redaction_notes,
        metadata=metadata or {},
    )


def approve_elevation(
    proposal: ElevationProposal,
    approver: str,
    redacted_content: Optional[bytes] = None,
    redaction_notes: Optional[str] = None,
) -> ElevationProposal:
    """Approve an elevation proposal.
    
    The owner can optionally provide a redacted/transformed version of the content
    to be shared at the elevated level.
    
    Args:
        proposal: The proposal to approve
        approver: DID of the owner approving the elevation
        redacted_content: Optional redacted version (overrides proposal's version)
        redaction_notes: Description of redactions (overrides proposal's notes)
        
    Returns:
        Updated proposal with APPROVED status
        
    Raises:
        ProposalAlreadyResolvedError: If proposal is not pending
    """
    if proposal.is_resolved:
        raise ProposalAlreadyResolvedError(
            f"Proposal {proposal.proposal_id} is already {proposal.status.value}"
        )
    
    # Create new proposal with approved status
    return ElevationProposal(
        proposal_id=proposal.proposal_id,
        proposer=proposal.proposer,
        belief_id=proposal.belief_id,
        from_level=proposal.from_level,
        to_level=proposal.to_level,
        reason=proposal.reason,
        status=ProposalStatus.APPROVED,
        created_at=proposal.created_at,
        resolved_at=datetime.now(timezone.utc),
        resolved_by=approver,
        rejection_reason=None,
        redacted_content=redacted_content or proposal.redacted_content,
        redaction_notes=redaction_notes or proposal.redaction_notes,
        metadata=proposal.metadata,
    )


def reject_elevation(
    proposal: ElevationProposal,
    rejector: str,
    reason: Optional[str] = None,
) -> ElevationProposal:
    """Reject an elevation proposal.
    
    Args:
        proposal: The proposal to reject
        rejector: DID of the owner rejecting the elevation
        reason: Optional explanation for the rejection
        
    Returns:
        Updated proposal with REJECTED status
        
    Raises:
        ProposalAlreadyResolvedError: If proposal is not pending
    """
    if proposal.is_resolved:
        raise ProposalAlreadyResolvedError(
            f"Proposal {proposal.proposal_id} is already {proposal.status.value}"
        )
    
    return ElevationProposal(
        proposal_id=proposal.proposal_id,
        proposer=proposal.proposer,
        belief_id=proposal.belief_id,
        from_level=proposal.from_level,
        to_level=proposal.to_level,
        reason=proposal.reason,
        status=ProposalStatus.REJECTED,
        created_at=proposal.created_at,
        resolved_at=datetime.now(timezone.utc),
        resolved_by=rejector,
        rejection_reason=reason,
        redacted_content=proposal.redacted_content,
        redaction_notes=proposal.redaction_notes,
        metadata=proposal.metadata,
    )


# Transformation function type
TransformFunc = Callable[[bytes], bytes]


def create_redacted_content(
    original_content: bytes,
    fields_to_redact: Optional[List[str]] = None,
    transform_func: Optional[TransformFunc] = None,
) -> bytes:
    """Create a redacted/transformed version of content for elevation.
    
    Supports two modes:
    1. Field redaction: Remove specific fields from JSON content
    2. Custom transformation: Apply a custom function to transform content
    
    Args:
        original_content: The original content bytes
        fields_to_redact: List of field paths to remove (dot-notation for nested)
        transform_func: Custom transformation function
        
    Returns:
        Transformed/redacted content bytes
    """
    import json
    import copy
    
    content = original_content
    
    # Apply field redaction if specified
    if fields_to_redact:
        try:
            data = json.loads(content.decode("utf-8"))
            data = copy.deepcopy(data)
            
            for field_path in fields_to_redact:
                _redact_field(data, field_path)
            
            content = json.dumps(data).encode("utf-8")
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON, can't redact fields
            pass
    
    # Apply custom transformation if specified
    if transform_func:
        content = transform_func(content)
    
    return content


def _redact_field(data: Dict[str, Any], field_path: str) -> None:
    """Redact a field in a dictionary using dot-notation path.
    
    Replaces the field value with "[REDACTED]" rather than removing it,
    to preserve structure while indicating something was there.
    """
    parts = field_path.split(".")
    
    current = data
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    
    if isinstance(current, dict) and parts[-1] in current:
        current[parts[-1]] = "[REDACTED]"


class ElevationService:
    """Service for managing elevation proposals.
    
    Provides a higher-level API with storage integration for
    managing the lifecycle of elevation proposals.
    """
    
    def __init__(self):
        """Initialize the elevation service."""
        self._proposals: Dict[str, ElevationProposal] = {}
        self._by_belief: Dict[str, List[str]] = {}  # belief_id -> [proposal_ids]
        self._by_proposer: Dict[str, List[str]] = {}  # proposer -> [proposal_ids]
    
    def propose(
        self,
        proposer: str,
        belief_id: str,
        from_level: ShareLevel,
        to_level: ShareLevel,
        reason: str,
        redacted_content: Optional[bytes] = None,
        redaction_notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ElevationProposal:
        """Create and store an elevation proposal.
        
        Returns:
            The created proposal
            
        Raises:
            InvalidElevationError: If elevation direction is invalid
        """
        proposal = propose_elevation(
            proposer=proposer,
            belief_id=belief_id,
            from_level=from_level,
            to_level=to_level,
            reason=reason,
            redacted_content=redacted_content,
            redaction_notes=redaction_notes,
            metadata=metadata,
        )
        
        self._store_proposal(proposal)
        return proposal
    
    def approve(
        self,
        proposal_id: str,
        approver: str,
        redacted_content: Optional[bytes] = None,
        redaction_notes: Optional[str] = None,
    ) -> ElevationProposal:
        """Approve a proposal by ID.
        
        Returns:
            The updated proposal
            
        Raises:
            ProposalNotFoundError: If proposal doesn't exist
            ProposalAlreadyResolvedError: If already resolved
        """
        proposal = self.get(proposal_id)
        if proposal is None:
            raise ProposalNotFoundError(f"Proposal {proposal_id} not found")
        
        updated = approve_elevation(
            proposal=proposal,
            approver=approver,
            redacted_content=redacted_content,
            redaction_notes=redaction_notes,
        )
        
        self._proposals[proposal_id] = updated
        return updated
    
    def reject(
        self,
        proposal_id: str,
        rejector: str,
        reason: Optional[str] = None,
    ) -> ElevationProposal:
        """Reject a proposal by ID.
        
        Returns:
            The updated proposal
            
        Raises:
            ProposalNotFoundError: If proposal doesn't exist
            ProposalAlreadyResolvedError: If already resolved
        """
        proposal = self.get(proposal_id)
        if proposal is None:
            raise ProposalNotFoundError(f"Proposal {proposal_id} not found")
        
        updated = reject_elevation(
            proposal=proposal,
            rejector=rejector,
            reason=reason,
        )
        
        self._proposals[proposal_id] = updated
        return updated
    
    def get(self, proposal_id: str) -> Optional[ElevationProposal]:
        """Get a proposal by ID."""
        return self._proposals.get(proposal_id)
    
    def get_for_belief(self, belief_id: str) -> List[ElevationProposal]:
        """Get all proposals for a specific belief."""
        proposal_ids = self._by_belief.get(belief_id, [])
        return [self._proposals[pid] for pid in proposal_ids if pid in self._proposals]
    
    def get_by_proposer(self, proposer: str) -> List[ElevationProposal]:
        """Get all proposals by a specific proposer."""
        proposal_ids = self._by_proposer.get(proposer, [])
        return [self._proposals[pid] for pid in proposal_ids if pid in self._proposals]
    
    def get_pending(self) -> List[ElevationProposal]:
        """Get all pending proposals."""
        return [p for p in self._proposals.values() if p.is_pending]
    
    def _store_proposal(self, proposal: ElevationProposal) -> None:
        """Store a proposal and update indices."""
        self._proposals[proposal.proposal_id] = proposal
        
        if proposal.belief_id not in self._by_belief:
            self._by_belief[proposal.belief_id] = []
        self._by_belief[proposal.belief_id].append(proposal.proposal_id)
        
        if proposal.proposer not in self._by_proposer:
            self._by_proposer[proposal.proposer] = []
        self._by_proposer[proposal.proposer].append(proposal.proposal_id)
