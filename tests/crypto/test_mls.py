"""Tests for MLS (Messaging Layer Security) abstraction layer.

These tests verify the MLS interface and mock implementation work correctly.
"""

import pytest
from datetime import datetime

from valence.crypto.mls import (
    MLSGroup,
    MLSMember,
    MLSKeySchedule,
    MLSProposal,
    MLSCommit,
    MLSBackend,
    MockMLSBackend,
    ProposalType,
    MLSError,
    MLSGroupNotFoundError,
    MLSMemberNotFoundError,
    MLSEpochMismatchError,
)


# =============================================================================
# Data Class Tests
# =============================================================================


class TestMLSMember:
    """Tests for MLSMember dataclass."""
    
    def test_create_member(self):
        """Test creating a member."""
        member = MLSMember(
            member_id=b"alice",
            leaf_index=0,
            credential=b"alice-cert",
            key_package=b"alice-kp",
        )
        
        assert member.member_id == b"alice"
        assert member.leaf_index == 0
        assert member.credential == b"alice-cert"
        assert member.key_package == b"alice-kp"
        assert isinstance(member.joined_at, datetime)
    
    def test_member_serialization(self):
        """Test member to_dict/from_dict roundtrip."""
        member = MLSMember(
            member_id=b"bob",
            leaf_index=1,
            credential=b"bob-cert",
            key_package=b"bob-kp",
        )
        
        data = member.to_dict()
        restored = MLSMember.from_dict(data)
        
        assert restored.member_id == member.member_id
        assert restored.leaf_index == member.leaf_index
        assert restored.credential == member.credential
        assert restored.key_package == member.key_package


class TestMLSKeySchedule:
    """Tests for MLSKeySchedule dataclass."""
    
    @pytest.fixture
    def key_schedule(self):
        """Create a test key schedule."""
        return MLSKeySchedule(
            epoch=5,
            epoch_secret=b"0" * 32,
            application_secret=b"1" * 32,
            confirmation_key=b"2" * 32,
            membership_key=b"3" * 32,
            resumption_psk=b"4" * 32,
            exporter_secret=b"5" * 32,
        )
    
    def test_derive_application_key(self, key_schedule):
        """Test application key derivation."""
        key1 = key_schedule.derive_application_key(0)
        key2 = key_schedule.derive_application_key(1)
        key1_again = key_schedule.derive_application_key(0)
        
        assert len(key1) == 32
        assert len(key2) == 32
        assert key1 != key2  # Different generations produce different keys
        assert key1 == key1_again  # Same generation is deterministic
    
    def test_derive_nonce(self, key_schedule):
        """Test nonce derivation."""
        nonce = key_schedule.derive_nonce(0)
        
        assert len(nonce) == 12  # AEAD nonce size
    
    def test_export_secret(self, key_schedule):
        """Test secret export."""
        secret16 = key_schedule.export_secret(b"label1", b"context1", 16)
        secret32 = key_schedule.export_secret(b"label1", b"context1", 32)
        secret_diff_label = key_schedule.export_secret(b"label2", b"context1", 32)
        secret_diff_context = key_schedule.export_secret(b"label1", b"context2", 32)
        
        assert len(secret16) == 16
        assert len(secret32) == 32
        # Different lengths produce different secrets (length is part of derivation)
        assert secret16 != secret32[:16]
        # But same params are deterministic
        assert secret32 == key_schedule.export_secret(b"label1", b"context1", 32)
        # Different labels/contexts produce different secrets
        assert secret32 != secret_diff_label
        assert secret32 != secret_diff_context
    
    def test_key_schedule_serialization(self, key_schedule):
        """Test key schedule to_dict/from_dict roundtrip."""
        data = key_schedule.to_dict()
        restored = MLSKeySchedule.from_dict(data)
        
        assert restored.epoch == key_schedule.epoch
        assert restored.application_secret == key_schedule.application_secret
        assert restored.exporter_secret == key_schedule.exporter_secret


class TestMLSGroup:
    """Tests for MLSGroup dataclass."""
    
    def test_create_group(self):
        """Test creating a group."""
        group = MLSGroup(
            group_id=b"test-group",
            epoch=0,
        )
        
        assert group.group_id == b"test-group"
        assert group.epoch == 0
        assert group.cipher_suite == 0x0001
        assert group.members == []
    
    def test_group_with_members(self):
        """Test group member management."""
        alice = MLSMember(member_id=b"alice", leaf_index=0)
        bob = MLSMember(member_id=b"bob", leaf_index=1)
        
        group = MLSGroup(
            group_id=b"test-group",
            members=[alice, bob],
        )
        
        assert group.member_count == 2
        assert group.get_member(b"alice") == alice
        assert group.get_member(b"bob") == bob
        assert group.get_member(b"charlie") is None
        assert group.get_member_by_index(0) == alice
        assert group.get_member_by_index(1) == bob
        assert b"alice" in group.member_ids
        assert b"bob" in group.member_ids
    
    def test_group_serialization(self):
        """Test group to_dict/from_dict roundtrip."""
        alice = MLSMember(member_id=b"alice", leaf_index=0)
        group = MLSGroup(
            group_id=b"test-group",
            epoch=3,
            cipher_suite=0x0002,
            members=[alice],
        )
        
        data = group.to_dict()
        restored = MLSGroup.from_dict(data)
        
        assert restored.group_id == group.group_id
        assert restored.epoch == group.epoch
        assert restored.cipher_suite == group.cipher_suite
        assert len(restored.members) == 1
        assert restored.members[0].member_id == b"alice"


class TestMLSProposal:
    """Tests for MLSProposal dataclass."""
    
    def test_create_proposal(self):
        """Test creating a proposal."""
        proposal = MLSProposal(
            proposal_type=ProposalType.ADD,
            sender=b"alice",
            epoch=5,
            payload=b"new-member-data",
        )
        
        assert proposal.proposal_type == ProposalType.ADD
        assert proposal.sender == b"alice"
        assert proposal.epoch == 5
        assert len(proposal.proposal_ref) == 16  # Random ref
    
    def test_proposal_serialization(self):
        """Test proposal to_dict/from_dict roundtrip."""
        proposal = MLSProposal(
            proposal_type=ProposalType.REMOVE,
            sender=b"alice",
            epoch=3,
            payload=b"bob",
        )
        
        data = proposal.to_dict()
        restored = MLSProposal.from_dict(data)
        
        assert restored.proposal_type == ProposalType.REMOVE
        assert restored.sender == b"alice"
        assert restored.epoch == 3
        assert restored.payload == b"bob"


# =============================================================================
# Mock Backend Tests
# =============================================================================


class TestMockMLSBackend:
    """Tests for MockMLSBackend implementation."""
    
    @pytest.fixture
    def backend(self):
        """Create a fresh mock backend."""
        return MockMLSBackend()
    
    def test_create_group(self, backend):
        """Test group creation."""
        group = backend.create_group(
            group_id=b"test-group",
            creator_id=b"alice",
            credential=b"alice-cert",
        )
        
        assert group.group_id == b"test-group"
        assert group.epoch == 0
        assert len(group.members) == 1
        assert group.members[0].member_id == b"alice"
        assert group.members[0].credential == b"alice-cert"
    
    def test_get_group(self, backend):
        """Test getting a group."""
        backend.create_group(b"test-group", b"alice")
        
        group = backend.get_group(b"test-group")
        assert group is not None
        assert group.group_id == b"test-group"
        
        missing = backend.get_group(b"nonexistent")
        assert missing is None
    
    def test_add_member(self, backend):
        """Test adding members to a group."""
        group = backend.create_group(b"test-group", b"alice")
        initial_epoch = group.epoch
        
        group = backend.add_member(
            group_id=b"test-group",
            member_id=b"bob",
            key_package=b"bob-kp",
        )
        
        assert len(group.members) == 2
        assert group.get_member(b"bob") is not None
        assert group.epoch == initial_epoch + 1  # Epoch advanced
    
    def test_add_member_to_nonexistent_group(self, backend):
        """Test adding member to nonexistent group raises error."""
        with pytest.raises(MLSGroupNotFoundError):
            backend.add_member(b"nonexistent", b"bob", b"bob-kp")
    
    def test_add_duplicate_member(self, backend):
        """Test adding duplicate member raises error."""
        backend.create_group(b"test-group", b"alice")
        
        with pytest.raises(MLSError):
            backend.add_member(b"test-group", b"alice", b"alice-kp")
    
    def test_remove_member(self, backend):
        """Test removing members from a group."""
        backend.create_group(b"test-group", b"alice")
        backend.add_member(b"test-group", b"bob", b"bob-kp")
        
        group = backend.remove_member(
            group_id=b"test-group",
            member_id=b"bob",
            remover_id=b"alice",
        )
        
        assert len(group.members) == 1
        assert group.get_member(b"bob") is None
        assert group.get_member(b"alice") is not None
    
    def test_remove_nonexistent_member(self, backend):
        """Test removing nonexistent member raises error."""
        backend.create_group(b"test-group", b"alice")
        
        with pytest.raises(MLSMemberNotFoundError):
            backend.remove_member(b"test-group", b"charlie", b"alice")
    
    def test_update_keys(self, backend):
        """Test key update."""
        backend.create_group(b"test-group", b"alice")
        group = backend.get_group(b"test-group")
        initial_epoch = group.epoch
        
        group = backend.update_keys(b"test-group", b"alice")
        
        assert group.epoch == initial_epoch + 1
    
    def test_key_schedule(self, backend):
        """Test key schedule retrieval."""
        backend.create_group(b"test-group", b"alice")
        
        ks = backend.get_key_schedule(b"test-group")
        
        assert ks.epoch == 0
        assert len(ks.application_secret) == 32
        assert len(ks.epoch_secret) == 32
    
    def test_key_schedule_changes_on_commit(self, backend):
        """Test that key schedule changes when epoch advances."""
        backend.create_group(b"test-group", b"alice")
        ks1 = backend.get_key_schedule(b"test-group")
        
        backend.add_member(b"test-group", b"bob", b"bob-kp")
        ks2 = backend.get_key_schedule(b"test-group")
        
        assert ks2.epoch == ks1.epoch + 1
        assert ks2.application_secret != ks1.application_secret
        assert ks2.epoch_secret != ks1.epoch_secret
    
    def test_proposal_and_commit(self, backend):
        """Test proposal/commit flow."""
        backend.create_group(b"test-group", b"alice")
        
        # Create proposals
        p1 = backend.propose_add(b"test-group", b"alice", b"bob", b"bob-kp")
        p2 = backend.propose_add(b"test-group", b"alice", b"charlie", b"charlie-kp")
        
        group = backend.get_group(b"test-group")
        assert len(group.pending_proposals) == 2
        assert len(group.members) == 1  # Not committed yet
        
        # Commit proposals
        group, commit = backend.commit(b"test-group", b"alice")
        
        assert len(group.members) == 3
        assert len(group.pending_proposals) == 0
        assert group.get_member(b"bob") is not None
        assert group.get_member(b"charlie") is not None
    
    def test_partial_commit(self, backend):
        """Test committing only some proposals."""
        backend.create_group(b"test-group", b"alice")
        
        p1 = backend.propose_add(b"test-group", b"alice", b"bob", b"bob-kp")
        p2 = backend.propose_add(b"test-group", b"alice", b"charlie", b"charlie-kp")
        
        # Only commit first proposal
        group, commit = backend.commit(b"test-group", b"alice", [p1.proposal_ref])
        
        assert len(group.members) == 2  # Alice + Bob
        assert len(group.pending_proposals) == 1  # Charlie still pending
        assert group.get_member(b"bob") is not None
        assert group.get_member(b"charlie") is None
    
    def test_propose_remove_and_commit(self, backend):
        """Test remove proposal and commit."""
        backend.create_group(b"test-group", b"alice")
        backend.add_member(b"test-group", b"bob", b"bob-kp")
        backend.add_member(b"test-group", b"charlie", b"charlie-kp")
        
        # Propose removing bob
        backend.propose_remove(b"test-group", b"alice", b"bob")
        
        group = backend.get_group(b"test-group")
        assert len(group.members) == 3  # Not committed yet
        
        # Commit
        group, commit = backend.commit(b"test-group", b"alice")
        
        assert len(group.members) == 2
        assert group.get_member(b"bob") is None
    
    def test_process_commit(self, backend):
        """Test processing a received commit."""
        backend.create_group(b"test-group", b"alice")
        group = backend.get_group(b"test-group")
        initial_epoch = group.epoch
        
        # Simulate receiving a commit from another member
        commit = MLSCommit(
            group_id=b"test-group",
            epoch=initial_epoch,
            proposals=[],
            committer=b"alice",
            commit_secret=b"x" * 32,
        )
        
        group = backend.process_commit(b"test-group", commit)
        
        assert group.epoch == initial_epoch + 1
    
    def test_process_commit_wrong_epoch(self, backend):
        """Test processing commit with wrong epoch raises error."""
        backend.create_group(b"test-group", b"alice")
        
        commit = MLSCommit(
            group_id=b"test-group",
            epoch=99,  # Wrong epoch
            proposals=[],
            committer=b"alice",
            commit_secret=b"x" * 32,
        )
        
        with pytest.raises(MLSEpochMismatchError):
            backend.process_commit(b"test-group", commit)
    
    def test_clear(self, backend):
        """Test clearing all groups."""
        backend.create_group(b"group1", b"alice")
        backend.create_group(b"group2", b"bob")
        
        backend.clear()
        
        assert backend.get_group(b"group1") is None
        assert backend.get_group(b"group2") is None


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestMLSScenarios:
    """Test realistic MLS scenarios."""
    
    @pytest.fixture
    def backend(self):
        """Create a fresh mock backend."""
        return MockMLSBackend()
    
    def test_group_lifecycle(self, backend):
        """Test complete group lifecycle."""
        # Create group
        group = backend.create_group(b"team-chat", b"alice")
        assert group.member_count == 1
        
        # Add members
        backend.add_member(b"team-chat", b"bob", b"bob-kp")
        backend.add_member(b"team-chat", b"charlie", b"charlie-kp")
        group = backend.get_group(b"team-chat")
        assert group.member_count == 3
        
        # Bob updates their keys (post-compromise security)
        backend.update_keys(b"team-chat", b"bob")
        
        # Charlie leaves
        backend.remove_member(b"team-chat", b"charlie", b"alice")
        group = backend.get_group(b"team-chat")
        assert group.member_count == 2
        assert group.get_member(b"charlie") is None
        
        # Verify key schedule evolved
        ks = backend.get_key_schedule(b"team-chat")
        assert ks.epoch == 4  # Create + 2 adds + update + remove
    
    def test_forward_secrecy_via_epochs(self, backend):
        """Test that key schedule changes maintain forward secrecy."""
        backend.create_group(b"secure-chat", b"alice")
        
        secrets = []
        for i in range(5):
            ks = backend.get_key_schedule(b"secure-chat")
            secrets.append(ks.application_secret)
            backend.add_member(b"secure-chat", f"user{i}".encode(), b"kp")
        
        # All secrets should be different
        assert len(set(secrets)) == 5
    
    def test_application_key_derivation(self, backend):
        """Test deriving application keys for message encryption."""
        backend.create_group(b"chat", b"alice")
        
        ks = backend.get_key_schedule(b"chat")
        
        # Derive keys for multiple messages
        key0 = ks.derive_application_key(0)
        nonce0 = ks.derive_nonce(0)
        
        key1 = ks.derive_application_key(1)
        nonce1 = ks.derive_nonce(1)
        
        # Keys and nonces should be unique per generation
        assert key0 != key1
        assert nonce0 != nonce1
        
        # But deterministic
        assert key0 == ks.derive_application_key(0)
    
    def test_export_secret_for_external_use(self, backend):
        """Test exporting secrets for external protocols."""
        backend.create_group(b"chat", b"alice")
        ks = backend.get_key_schedule(b"chat")
        
        # Export secrets for different purposes
        auth_secret = ks.export_secret(b"authentication", b"context1")
        binding_secret = ks.export_secret(b"channel-binding", b"context2")
        
        assert len(auth_secret) == 32
        assert len(binding_secret) == 32
        assert auth_secret != binding_secret
    
    def test_multi_group_isolation(self, backend):
        """Test that groups are properly isolated."""
        backend.create_group(b"group-a", b"alice")
        backend.create_group(b"group-b", b"alice")
        
        backend.add_member(b"group-a", b"bob", b"bob-kp")
        
        group_a = backend.get_group(b"group-a")
        group_b = backend.get_group(b"group-b")
        
        assert group_a.member_count == 2
        assert group_b.member_count == 1
        
        # Key schedules should be different
        ks_a = backend.get_key_schedule(b"group-a")
        ks_b = backend.get_key_schedule(b"group-b")
        
        assert ks_a.application_secret != ks_b.application_secret


# =============================================================================
# Backend Interface Tests
# =============================================================================


class TestMLSBackendInterface:
    """Test that MockMLSBackend properly implements MLSBackend interface."""
    
    def test_is_mls_backend(self):
        """Verify MockMLSBackend is a proper MLSBackend subclass."""
        backend = MockMLSBackend()
        assert isinstance(backend, MLSBackend)
    
    def test_all_abstract_methods_implemented(self):
        """Verify all abstract methods are implemented."""
        backend = MockMLSBackend()
        
        # These should not raise NotImplementedError
        group = backend.create_group(b"test", b"alice")
        backend.add_member(b"test", b"bob", b"kp")
        backend.remove_member(b"test", b"bob", b"alice")
        backend.update_keys(b"test", b"alice")
        backend.get_group(b"test")
        backend.get_key_schedule(b"test")
        
        # Add bob back for proposal tests
        backend.add_member(b"test", b"bob", b"kp")
        backend.propose_add(b"test", b"alice", b"charlie", b"kp")
        backend.propose_remove(b"test", b"alice", b"bob")
        backend.commit(b"test", b"alice")
        
        commit = MLSCommit(
            group_id=b"test",
            epoch=backend.get_group(b"test").epoch,
            proposals=[],
            committer=b"alice",
            commit_secret=b"x" * 32,
        )
        backend.process_commit(b"test", commit)
