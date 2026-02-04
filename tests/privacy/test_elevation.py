"""Tests for elevation proposal workflow (Issue #94)."""

import pytest
from datetime import datetime, timezone
import json

from valence.privacy.elevation import (
    ProposalStatus,
    ElevationProposal,
    ElevationError,
    ProposalNotFoundError,
    ProposalAlreadyResolvedError,
    InvalidElevationError,
    propose_elevation,
    approve_elevation,
    reject_elevation,
    create_redacted_content,
    ElevationService,
)
from valence.privacy.types import ShareLevel


class TestProposalStatus:
    """Tests for ProposalStatus enum."""
    
    def test_all_statuses_exist(self):
        """Verify all proposal statuses are defined."""
        assert ProposalStatus.PENDING.value == "pending"
        assert ProposalStatus.APPROVED.value == "approved"
        assert ProposalStatus.REJECTED.value == "rejected"
    
    def test_status_from_string(self):
        """Test creating status from string value."""
        assert ProposalStatus("pending") == ProposalStatus.PENDING
        assert ProposalStatus("approved") == ProposalStatus.APPROVED
        assert ProposalStatus("rejected") == ProposalStatus.REJECTED


class TestElevationProposal:
    """Tests for ElevationProposal dataclass."""
    
    def test_create_proposal(self):
        """Test creating a basic proposal."""
        proposal = ElevationProposal(
            proposal_id="test-123",
            proposer="did:example:alice",
            belief_id="belief-456",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Need to share with team",
        )
        
        assert proposal.proposal_id == "test-123"
        assert proposal.proposer == "did:example:alice"
        assert proposal.belief_id == "belief-456"
        assert proposal.from_level == ShareLevel.PRIVATE
        assert proposal.to_level == ShareLevel.DIRECT
        assert proposal.reason == "Need to share with team"
        assert proposal.status == ProposalStatus.PENDING
        assert proposal.is_pending
        assert not proposal.is_resolved
        assert not proposal.is_approved
        assert not proposal.is_rejected
    
    def test_proposal_with_redacted_content(self):
        """Test proposal with pre-redacted content."""
        redacted = b'{"data": "[REDACTED]", "public_field": "visible"}'
        
        proposal = ElevationProposal(
            proposal_id="test-123",
            proposer="did:example:alice",
            belief_id="belief-456",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.PUBLIC,
            reason="Public release",
            redacted_content=redacted,
            redaction_notes="Removed sensitive data field",
        )
        
        assert proposal.redacted_content == redacted
        assert proposal.redaction_notes == "Removed sensitive data field"
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        created = datetime(2026, 2, 4, 12, 0, 0, tzinfo=timezone.utc)
        proposal = ElevationProposal(
            proposal_id="test-123",
            proposer="did:example:alice",
            belief_id="belief-456",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.BOUNDED,
            reason="Research collaboration",
            created_at=created,
            metadata={"priority": "high"},
        )
        
        data = proposal.to_dict()
        
        assert data["proposal_id"] == "test-123"
        assert data["proposer"] == "did:example:alice"
        assert data["belief_id"] == "belief-456"
        assert data["from_level"] == "private"
        assert data["to_level"] == "bounded"
        assert data["reason"] == "Research collaboration"
        assert data["status"] == "pending"
        assert data["created_at"] == "2026-02-04T12:00:00+00:00"
        assert data["resolved_at"] is None
        assert data["resolved_by"] is None
        assert data["metadata"] == {"priority": "high"}
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "proposal_id": "test-789",
            "proposer": "did:example:bob",
            "belief_id": "belief-999",
            "from_level": "direct",
            "to_level": "public",
            "reason": "Making public",
            "status": "approved",
            "created_at": "2026-02-04T10:00:00+00:00",
            "resolved_at": "2026-02-04T11:00:00+00:00",
            "resolved_by": "did:example:owner",
            "rejection_reason": None,
            "redacted_content": None,
            "redaction_notes": None,
            "metadata": {"approved_with": "modifications"},
        }
        
        proposal = ElevationProposal.from_dict(data)
        
        assert proposal.proposal_id == "test-789"
        assert proposal.proposer == "did:example:bob"
        assert proposal.from_level == ShareLevel.DIRECT
        assert proposal.to_level == ShareLevel.PUBLIC
        assert proposal.status == ProposalStatus.APPROVED
        assert proposal.is_approved
        assert proposal.is_resolved
        assert proposal.resolved_by == "did:example:owner"
        assert proposal.metadata == {"approved_with": "modifications"}
    
    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip preserves data."""
        original = ElevationProposal(
            proposal_id="roundtrip-test",
            proposer="did:example:user",
            belief_id="belief-123",
            from_level=ShareLevel.BOUNDED,
            to_level=ShareLevel.CASCADING,
            reason="Extended sharing",
            redacted_content=b'{"clean": "data"}',
            redaction_notes="Removed PII",
            metadata={"test": True, "count": 42},
        )
        
        restored = ElevationProposal.from_dict(original.to_dict())
        
        assert restored.proposal_id == original.proposal_id
        assert restored.proposer == original.proposer
        assert restored.belief_id == original.belief_id
        assert restored.from_level == original.from_level
        assert restored.to_level == original.to_level
        assert restored.reason == original.reason
        assert restored.redacted_content == original.redacted_content
        assert restored.redaction_notes == original.redaction_notes
        assert restored.metadata == original.metadata


class TestProposeElevation:
    """Tests for propose_elevation function."""
    
    def test_basic_proposal(self):
        """Test creating a basic elevation proposal."""
        proposal = propose_elevation(
            proposer="did:example:requester",
            belief_id="belief-to-elevate",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Need to share with colleague",
        )
        
        assert proposal.proposal_id  # UUID generated
        assert proposal.proposer == "did:example:requester"
        assert proposal.belief_id == "belief-to-elevate"
        assert proposal.from_level == ShareLevel.PRIVATE
        assert proposal.to_level == ShareLevel.DIRECT
        assert proposal.reason == "Need to share with colleague"
        assert proposal.status == ProposalStatus.PENDING
        assert proposal.created_at is not None
    
    def test_proposal_with_redacted_content(self):
        """Test proposal that includes pre-redacted content."""
        redacted = b'{"safe": "content"}'
        
        proposal = propose_elevation(
            proposer="did:example:user",
            belief_id="sensitive-belief",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.PUBLIC,
            reason="Publish findings",
            redacted_content=redacted,
            redaction_notes="Removed patient identifiers",
        )
        
        assert proposal.redacted_content == redacted
        assert proposal.redaction_notes == "Removed patient identifiers"
    
    def test_proposal_with_metadata(self):
        """Test proposal with additional metadata."""
        proposal = propose_elevation(
            proposer="did:example:user",
            belief_id="belief-123",
            from_level=ShareLevel.DIRECT,
            to_level=ShareLevel.BOUNDED,
            reason="Team access",
            metadata={"urgency": "high", "department": "research"},
        )
        
        assert proposal.metadata == {"urgency": "high", "department": "research"}
    
    def test_invalid_elevation_same_level(self):
        """Test that same-level elevation is rejected."""
        with pytest.raises(InvalidElevationError) as exc_info:
            propose_elevation(
                proposer="did:example:user",
                belief_id="belief-123",
                from_level=ShareLevel.DIRECT,
                to_level=ShareLevel.DIRECT,
                reason="No change",
            )
        
        assert "must be more public" in str(exc_info.value)
    
    def test_invalid_elevation_demotion(self):
        """Test that demotion (more private) is rejected."""
        with pytest.raises(InvalidElevationError) as exc_info:
            propose_elevation(
                proposer="did:example:user",
                belief_id="belief-123",
                from_level=ShareLevel.PUBLIC,
                to_level=ShareLevel.PRIVATE,
                reason="Trying to demote",
            )
        
        assert "must be more public" in str(exc_info.value)
    
    @pytest.mark.parametrize("from_level,to_level", [
        (ShareLevel.PRIVATE, ShareLevel.DIRECT),
        (ShareLevel.PRIVATE, ShareLevel.BOUNDED),
        (ShareLevel.PRIVATE, ShareLevel.CASCADING),
        (ShareLevel.PRIVATE, ShareLevel.PUBLIC),
        (ShareLevel.DIRECT, ShareLevel.BOUNDED),
        (ShareLevel.DIRECT, ShareLevel.CASCADING),
        (ShareLevel.DIRECT, ShareLevel.PUBLIC),
        (ShareLevel.BOUNDED, ShareLevel.CASCADING),
        (ShareLevel.BOUNDED, ShareLevel.PUBLIC),
        (ShareLevel.CASCADING, ShareLevel.PUBLIC),
    ])
    def test_all_valid_elevation_paths(self, from_level, to_level):
        """Test all valid elevation paths are accepted."""
        proposal = propose_elevation(
            proposer="did:example:user",
            belief_id="belief-123",
            from_level=from_level,
            to_level=to_level,
            reason=f"Elevate from {from_level.value} to {to_level.value}",
        )
        
        assert proposal.from_level == from_level
        assert proposal.to_level == to_level


class TestApproveElevation:
    """Tests for approve_elevation function."""
    
    def test_basic_approval(self):
        """Test approving a pending proposal."""
        proposal = propose_elevation(
            proposer="did:example:requester",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Need to share",
        )
        
        approved = approve_elevation(
            proposal=proposal,
            approver="did:example:owner",
        )
        
        assert approved.status == ProposalStatus.APPROVED
        assert approved.is_approved
        assert approved.is_resolved
        assert approved.resolved_by == "did:example:owner"
        assert approved.resolved_at is not None
        # Original data preserved
        assert approved.proposal_id == proposal.proposal_id
        assert approved.proposer == proposal.proposer
        assert approved.belief_id == proposal.belief_id
        assert approved.reason == proposal.reason
    
    def test_approval_with_owner_redaction(self):
        """Test owner providing redacted content at approval time."""
        proposal = propose_elevation(
            proposer="did:example:requester",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.PUBLIC,
            reason="Publish",
        )
        
        owner_redacted = b'{"sanitized": "content"}'
        
        approved = approve_elevation(
            proposal=proposal,
            approver="did:example:owner",
            redacted_content=owner_redacted,
            redaction_notes="Owner sanitized for public release",
        )
        
        assert approved.redacted_content == owner_redacted
        assert approved.redaction_notes == "Owner sanitized for public release"
    
    def test_approval_preserves_proposer_redaction(self):
        """Test that proposer's redacted content is preserved if owner doesn't override."""
        proposer_redacted = b'{"proposer": "version"}'
        
        proposal = propose_elevation(
            proposer="did:example:requester",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Share",
            redacted_content=proposer_redacted,
            redaction_notes="Proposer redaction",
        )
        
        approved = approve_elevation(
            proposal=proposal,
            approver="did:example:owner",
        )
        
        assert approved.redacted_content == proposer_redacted
        assert approved.redaction_notes == "Proposer redaction"
    
    def test_cannot_approve_already_approved(self):
        """Test that approving an approved proposal raises error."""
        proposal = propose_elevation(
            proposer="did:example:user",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Share",
        )
        
        approved = approve_elevation(proposal, "did:example:owner")
        
        with pytest.raises(ProposalAlreadyResolvedError) as exc_info:
            approve_elevation(approved, "did:example:owner2")
        
        assert "already approved" in str(exc_info.value)
    
    def test_cannot_approve_already_rejected(self):
        """Test that approving a rejected proposal raises error."""
        proposal = propose_elevation(
            proposer="did:example:user",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Share",
        )
        
        rejected = reject_elevation(proposal, "did:example:owner", "No")
        
        with pytest.raises(ProposalAlreadyResolvedError) as exc_info:
            approve_elevation(rejected, "did:example:owner")
        
        assert "already rejected" in str(exc_info.value)


class TestRejectElevation:
    """Tests for reject_elevation function."""
    
    def test_basic_rejection(self):
        """Test rejecting a pending proposal."""
        proposal = propose_elevation(
            proposer="did:example:requester",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.PUBLIC,
            reason="Want to publish",
        )
        
        rejected = reject_elevation(
            proposal=proposal,
            rejector="did:example:owner",
            reason="Content too sensitive for public release",
        )
        
        assert rejected.status == ProposalStatus.REJECTED
        assert rejected.is_rejected
        assert rejected.is_resolved
        assert rejected.resolved_by == "did:example:owner"
        assert rejected.resolved_at is not None
        assert rejected.rejection_reason == "Content too sensitive for public release"
    
    def test_rejection_without_reason(self):
        """Test rejection without providing a reason."""
        proposal = propose_elevation(
            proposer="did:example:requester",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Share please",
        )
        
        rejected = reject_elevation(
            proposal=proposal,
            rejector="did:example:owner",
        )
        
        assert rejected.is_rejected
        assert rejected.rejection_reason is None
    
    def test_cannot_reject_already_approved(self):
        """Test that rejecting an approved proposal raises error."""
        proposal = propose_elevation(
            proposer="did:example:user",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Share",
        )
        
        approved = approve_elevation(proposal, "did:example:owner")
        
        with pytest.raises(ProposalAlreadyResolvedError) as exc_info:
            reject_elevation(approved, "did:example:owner", "Changed my mind")
        
        assert "already approved" in str(exc_info.value)
    
    def test_cannot_reject_already_rejected(self):
        """Test that rejecting an already rejected proposal raises error."""
        proposal = propose_elevation(
            proposer="did:example:user",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Share",
        )
        
        rejected = reject_elevation(proposal, "did:example:owner", "No")
        
        with pytest.raises(ProposalAlreadyResolvedError) as exc_info:
            reject_elevation(rejected, "did:example:owner2", "Also no")
        
        assert "already rejected" in str(exc_info.value)


class TestCreateRedactedContent:
    """Tests for create_redacted_content function."""
    
    def test_redact_single_field(self):
        """Test redacting a single top-level field."""
        original = json.dumps({
            "public": "visible",
            "secret": "hidden",
        }).encode("utf-8")
        
        redacted = create_redacted_content(original, fields_to_redact=["secret"])
        
        data = json.loads(redacted.decode("utf-8"))
        assert data["public"] == "visible"
        assert data["secret"] == "[REDACTED]"
    
    def test_redact_nested_field(self):
        """Test redacting a nested field using dot notation."""
        original = json.dumps({
            "user": {
                "name": "Alice",
                "ssn": "123-45-6789",
            },
            "public": True,
        }).encode("utf-8")
        
        redacted = create_redacted_content(original, fields_to_redact=["user.ssn"])
        
        data = json.loads(redacted.decode("utf-8"))
        assert data["user"]["name"] == "Alice"
        assert data["user"]["ssn"] == "[REDACTED]"
        assert data["public"] is True
    
    def test_redact_multiple_fields(self):
        """Test redacting multiple fields."""
        original = json.dumps({
            "name": "Test",
            "password": "secret123",
            "metadata": {
                "internal_id": "xyz",
                "public_tag": "foo",
            },
        }).encode("utf-8")
        
        redacted = create_redacted_content(
            original,
            fields_to_redact=["password", "metadata.internal_id"],
        )
        
        data = json.loads(redacted.decode("utf-8"))
        assert data["name"] == "Test"
        assert data["password"] == "[REDACTED]"
        assert data["metadata"]["internal_id"] == "[REDACTED]"
        assert data["metadata"]["public_tag"] == "foo"
    
    def test_redact_missing_field_is_noop(self):
        """Test that redacting a non-existent field doesn't error."""
        original = json.dumps({"existing": "value"}).encode("utf-8")
        
        redacted = create_redacted_content(
            original,
            fields_to_redact=["nonexistent", "also.missing"],
        )
        
        data = json.loads(redacted.decode("utf-8"))
        assert data == {"existing": "value"}
    
    def test_custom_transform_function(self):
        """Test applying a custom transformation function."""
        original = b"SENSITIVE DATA"
        
        def anonymize(content: bytes) -> bytes:
            return content.replace(b"SENSITIVE", b"ANONYMOUS")
        
        redacted = create_redacted_content(original, transform_func=anonymize)
        
        assert redacted == b"ANONYMOUS DATA"
    
    def test_combined_redaction_and_transform(self):
        """Test combining field redaction with custom transform."""
        original = json.dumps({
            "secret": "hidden",
            "text": "hello world",
        }).encode("utf-8")
        
        def uppercase(content: bytes) -> bytes:
            return content.upper()
        
        redacted = create_redacted_content(
            original,
            fields_to_redact=["secret"],
            transform_func=uppercase,
        )
        
        # Transform is applied after redaction
        assert b'"SECRET": "[REDACTED]"' in redacted
        assert b'"TEXT": "HELLO WORLD"' in redacted
    
    def test_non_json_content_skips_field_redaction(self):
        """Test that non-JSON content skips field redaction gracefully."""
        original = b"This is plain text, not JSON"
        
        redacted = create_redacted_content(original, fields_to_redact=["field"])
        
        # Unchanged because it's not JSON
        assert redacted == original
    
    def test_no_transformations_returns_original(self):
        """Test that no transformations returns original content."""
        original = b'{"data": "unchanged"}'
        
        redacted = create_redacted_content(original)
        
        assert redacted == original


class TestElevationService:
    """Tests for ElevationService class."""
    
    @pytest.fixture
    def service(self):
        """Create a fresh elevation service for each test."""
        return ElevationService()
    
    def test_propose_creates_and_stores(self, service):
        """Test that propose creates and stores a proposal."""
        proposal = service.propose(
            proposer="did:example:alice",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Share with Bob",
        )
        
        assert proposal.proposal_id
        assert service.get(proposal.proposal_id) == proposal
    
    def test_approve_by_id(self, service):
        """Test approving a proposal by ID."""
        proposal = service.propose(
            proposer="did:example:alice",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Share",
        )
        
        approved = service.approve(
            proposal_id=proposal.proposal_id,
            approver="did:example:owner",
        )
        
        assert approved.is_approved
        assert service.get(proposal.proposal_id).is_approved
    
    def test_reject_by_id(self, service):
        """Test rejecting a proposal by ID."""
        proposal = service.propose(
            proposer="did:example:alice",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.PUBLIC,
            reason="Publish",
        )
        
        rejected = service.reject(
            proposal_id=proposal.proposal_id,
            rejector="did:example:owner",
            reason="Too sensitive",
        )
        
        assert rejected.is_rejected
        assert service.get(proposal.proposal_id).is_rejected
        assert service.get(proposal.proposal_id).rejection_reason == "Too sensitive"
    
    def test_get_nonexistent_returns_none(self, service):
        """Test getting a non-existent proposal returns None."""
        assert service.get("nonexistent-id") is None
    
    def test_approve_nonexistent_raises(self, service):
        """Test approving a non-existent proposal raises error."""
        with pytest.raises(ProposalNotFoundError):
            service.approve("nonexistent-id", "did:example:owner")
    
    def test_reject_nonexistent_raises(self, service):
        """Test rejecting a non-existent proposal raises error."""
        with pytest.raises(ProposalNotFoundError):
            service.reject("nonexistent-id", "did:example:owner")
    
    def test_get_for_belief(self, service):
        """Test getting all proposals for a specific belief."""
        # Create proposals for different beliefs
        p1 = service.propose(
            proposer="did:example:alice",
            belief_id="belief-A",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Share",
        )
        p2 = service.propose(
            proposer="did:example:bob",
            belief_id="belief-A",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.PUBLIC,
            reason="Publish",
        )
        p3 = service.propose(
            proposer="did:example:alice",
            belief_id="belief-B",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Different belief",
        )
        
        belief_a_proposals = service.get_for_belief("belief-A")
        
        assert len(belief_a_proposals) == 2
        assert p1 in belief_a_proposals
        assert p2 in belief_a_proposals
        assert p3 not in belief_a_proposals
    
    def test_get_by_proposer(self, service):
        """Test getting all proposals by a specific proposer."""
        p1 = service.propose(
            proposer="did:example:alice",
            belief_id="belief-1",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="First",
        )
        p2 = service.propose(
            proposer="did:example:alice",
            belief_id="belief-2",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.PUBLIC,
            reason="Second",
        )
        p3 = service.propose(
            proposer="did:example:bob",
            belief_id="belief-3",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Bob's proposal",
        )
        
        alice_proposals = service.get_by_proposer("did:example:alice")
        
        assert len(alice_proposals) == 2
        assert p1 in alice_proposals
        assert p2 in alice_proposals
        assert p3 not in alice_proposals
    
    def test_get_pending(self, service):
        """Test getting all pending proposals."""
        p1 = service.propose(
            proposer="did:example:alice",
            belief_id="belief-1",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.DIRECT,
            reason="Pending",
        )
        p2 = service.propose(
            proposer="did:example:bob",
            belief_id="belief-2",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.PUBLIC,
            reason="Will be approved",
        )
        p3 = service.propose(
            proposer="did:example:carol",
            belief_id="belief-3",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.BOUNDED,
            reason="Will be rejected",
        )
        
        service.approve(p2.proposal_id, "did:example:owner")
        service.reject(p3.proposal_id, "did:example:owner", "No")
        
        pending = service.get_pending()
        
        assert len(pending) == 1
        assert pending[0].proposal_id == p1.proposal_id
    
    def test_approve_with_owner_redaction(self, service):
        """Test owner providing redacted content when approving."""
        proposal = service.propose(
            proposer="did:example:alice",
            belief_id="belief-123",
            from_level=ShareLevel.PRIVATE,
            to_level=ShareLevel.PUBLIC,
            reason="Publish",
        )
        
        owner_content = b'{"sanitized": true}'
        approved = service.approve(
            proposal_id=proposal.proposal_id,
            approver="did:example:owner",
            redacted_content=owner_content,
            redaction_notes="Owner sanitized",
        )
        
        assert approved.redacted_content == owner_content
        assert approved.redaction_notes == "Owner sanitized"


class TestElevationExceptions:
    """Tests for elevation exception hierarchy."""
    
    def test_exception_hierarchy(self):
        """Verify exception inheritance."""
        assert issubclass(ProposalNotFoundError, ElevationError)
        assert issubclass(ProposalAlreadyResolvedError, ElevationError)
        assert issubclass(InvalidElevationError, ElevationError)
        assert issubclass(ElevationError, Exception)
    
    def test_exceptions_have_messages(self):
        """Test that exceptions carry meaningful messages."""
        e1 = ProposalNotFoundError("Proposal xyz not found")
        e2 = ProposalAlreadyResolvedError("Already approved")
        e3 = InvalidElevationError("Invalid direction")
        
        assert "xyz" in str(e1)
        assert "approved" in str(e2)
        assert "direction" in str(e3)
