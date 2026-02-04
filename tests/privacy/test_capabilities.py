"""Tests for capability-based access control.

Tests cover:
- Capability issuance with TTL
- Signature verification
- Expiration handling
- Revocation
- Delegation chains
- Store operations
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from valence.privacy.capabilities import (
    Capability,
    CapabilityAction,
    CapabilityService,
    CapabilityStore,
    InMemoryCapabilityStore,
    CapabilityError,
    CapabilityExpiredError,
    CapabilityInvalidSignatureError,
    CapabilityRevokedError,
    CapabilityNotFoundError,
    CapabilityTTLExceededError,
    CapabilityInsufficientPermissionError,
    DEFAULT_TTL_SECONDS,
    MAX_TTL_SECONDS,
    get_capability_service,
    set_capability_service,
    issue_capability,
    verify_capability,
    revoke_capability,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def issuer_keypair():
    """Generate an issuer Ed25519 keypair."""
    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


@pytest.fixture
def holder_keypair():
    """Generate a holder Ed25519 keypair."""
    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


@pytest.fixture
def issuer_did():
    """Issuer DID for tests."""
    return "did:valence:issuer123"


@pytest.fixture
def holder_did():
    """Holder DID for tests."""
    return "did:valence:holder456"


@pytest.fixture
def resource():
    """Resource identifier for tests."""
    return "valence://beliefs/domain/science"


@pytest.fixture
def actions():
    """Default actions for tests."""
    return [CapabilityAction.READ.value, CapabilityAction.QUERY.value]


@pytest.fixture
def store():
    """In-memory capability store for tests."""
    return InMemoryCapabilityStore()


@pytest.fixture
def service(store, issuer_keypair):
    """Capability service with in-memory store."""
    private_key, public_key = issuer_keypair
    
    async def key_resolver(did: str):
        if "issuer" in did:
            return public_key
        return None
    
    return CapabilityService(
        store=store,
        key_resolver=key_resolver,
    )


# =============================================================================
# CAPABILITY MODEL TESTS
# =============================================================================

class TestCapabilityModel:
    """Tests for the Capability dataclass."""
    
    def test_capability_creation(self, issuer_did, holder_did, resource, actions):
        """Test basic capability creation."""
        now = datetime.now(timezone.utc)
        cap = Capability(
            id="cap-123",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        
        assert cap.id == "cap-123"
        assert cap.issuer_did == issuer_did
        assert cap.holder_did == holder_did
        assert cap.resource == resource
        assert cap.actions == actions
        assert not cap.is_expired
        assert not cap.is_revoked
        assert cap.is_valid
    
    def test_capability_requires_actions(self, issuer_did, holder_did, resource):
        """Test that capability requires at least one action."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="at least one action"):
            Capability(
                id="cap-123",
                issuer_did=issuer_did,
                holder_did=holder_did,
                resource=resource,
                actions=[],
                issued_at=now,
                expires_at=now + timedelta(hours=1),
            )
    
    def test_capability_expires_at_must_be_after_issued_at(self, issuer_did, holder_did, resource, actions):
        """Test that expires_at must be after issued_at."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="expires_at must be after issued_at"):
            Capability(
                id="cap-123",
                issuer_did=issuer_did,
                holder_did=holder_did,
                resource=resource,
                actions=actions,
                issued_at=now,
                expires_at=now - timedelta(hours=1),
            )
    
    def test_capability_expiration(self, issuer_did, holder_did, resource, actions):
        """Test capability expiration detection."""
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        cap = Capability(
            id="cap-123",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=past - timedelta(hours=1),
            expires_at=past,
        )
        
        assert cap.is_expired
        assert not cap.is_valid
        assert cap.ttl_seconds < 0
    
    def test_capability_revocation(self, issuer_did, holder_did, resource, actions):
        """Test capability revocation detection."""
        now = datetime.now(timezone.utc)
        cap = Capability(
            id="cap-123",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
            revoked_at=now,
        )
        
        assert cap.is_revoked
        assert not cap.is_valid
    
    def test_capability_has_action(self, issuer_did, holder_did, resource):
        """Test action checking."""
        now = datetime.now(timezone.utc)
        cap = Capability(
            id="cap-123",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read", "query"],
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        
        assert cap.has_action("read")
        assert cap.has_action("query")
        assert not cap.has_action("write")
    
    def test_capability_admin_grants_all_actions(self, issuer_did, holder_did, resource):
        """Test that admin action grants all actions."""
        now = datetime.now(timezone.utc)
        cap = Capability(
            id="cap-123",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=[CapabilityAction.ADMIN.value],
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        
        assert cap.has_action("read")
        assert cap.has_action("write")
        assert cap.has_action("delete")
        assert cap.has_action("anything")
    
    def test_capability_serialization(self, issuer_did, holder_did, resource, actions):
        """Test capability serialization/deserialization."""
        now = datetime.now(timezone.utc)
        cap = Capability(
            id="cap-123",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
            signature="abc123",
            parent_id="parent-456",
            metadata={"domain": "science"},
        )
        
        data = cap.to_dict()
        restored = Capability.from_dict(data)
        
        assert restored.id == cap.id
        assert restored.issuer_did == cap.issuer_did
        assert restored.holder_did == cap.holder_did
        assert restored.resource == cap.resource
        assert restored.actions == cap.actions
        assert restored.signature == cap.signature
        assert restored.parent_id == cap.parent_id
        assert restored.metadata == cap.metadata
    
    def test_capability_payload_bytes_is_deterministic(self, issuer_did, holder_did, resource, actions):
        """Test that payload bytes are deterministic for signing."""
        now = datetime.now(timezone.utc)
        cap1 = Capability(
            id="cap-123",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        
        # Create with actions in different order
        cap2 = Capability(
            id="cap-123",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=list(reversed(actions)),
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        
        # Should produce same payload (actions are sorted)
        assert cap1.payload_bytes() == cap2.payload_bytes()


# =============================================================================
# CAPABILITY STORE TESTS
# =============================================================================

class TestInMemoryCapabilityStore:
    """Tests for the in-memory capability store."""
    
    @pytest.fixture
    def sample_capability(self, issuer_did, holder_did, resource, actions):
        """Create a sample capability."""
        now = datetime.now(timezone.utc)
        return Capability(
            id="cap-123",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
    
    @pytest.mark.asyncio
    async def test_save_and_get(self, store, sample_capability):
        """Test saving and retrieving a capability."""
        await store.save(sample_capability)
        retrieved = await store.get(sample_capability.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_capability.id
    
    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test retrieving a nonexistent capability."""
        result = await store.get("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_by_holder(self, store, issuer_did, holder_did, resource, actions):
        """Test listing capabilities by holder."""
        now = datetime.now(timezone.utc)
        
        # Create capabilities for different holders
        cap1 = Capability(
            id="cap-1",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        cap2 = Capability(
            id="cap-2",
            issuer_did=issuer_did,
            holder_did="did:valence:other",
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        
        await store.save(cap1)
        await store.save(cap2)
        
        holder_caps = await store.list_by_holder(holder_did)
        assert len(holder_caps) == 1
        assert holder_caps[0].id == "cap-1"
    
    @pytest.mark.asyncio
    async def test_list_by_holder_excludes_expired(self, store, issuer_did, holder_did, resource, actions):
        """Test that expired capabilities are excluded by default."""
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        now = datetime.now(timezone.utc)
        
        expired_cap = Capability(
            id="cap-expired",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=past - timedelta(hours=1),
            expires_at=past,
        )
        valid_cap = Capability(
            id="cap-valid",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        
        await store.save(expired_cap)
        await store.save(valid_cap)
        
        # Should only return valid
        caps = await store.list_by_holder(holder_did, include_expired=False)
        assert len(caps) == 1
        assert caps[0].id == "cap-valid"
        
        # Should return both when include_expired=True
        all_caps = await store.list_by_holder(holder_did, include_expired=True)
        assert len(all_caps) == 2
    
    @pytest.mark.asyncio
    async def test_list_by_issuer(self, store, issuer_did, holder_did, resource, actions):
        """Test listing capabilities by issuer."""
        now = datetime.now(timezone.utc)
        
        cap = Capability(
            id="cap-1",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        
        await store.save(cap)
        
        issuer_caps = await store.list_by_issuer(issuer_did)
        assert len(issuer_caps) == 1
        assert issuer_caps[0].id == "cap-1"
    
    @pytest.mark.asyncio
    async def test_list_by_resource(self, store, issuer_did, holder_did, resource, actions):
        """Test listing capabilities by resource."""
        now = datetime.now(timezone.utc)
        
        cap = Capability(
            id="cap-1",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        
        await store.save(cap)
        
        resource_caps = await store.list_by_resource(resource)
        assert len(resource_caps) == 1
        assert resource_caps[0].id == "cap-1"
    
    @pytest.mark.asyncio
    async def test_revoke(self, store, sample_capability):
        """Test revoking a capability."""
        await store.save(sample_capability)
        
        result = await store.revoke(sample_capability.id)
        assert result is True
        
        # Check it's marked as revoked
        cap = await store.get(sample_capability.id)
        assert cap.is_revoked
        assert await store.is_revoked(sample_capability.id)
    
    @pytest.mark.asyncio
    async def test_revoke_nonexistent(self, store):
        """Test revoking a nonexistent capability."""
        result = await store.revoke("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cleanup_expired(self, store, issuer_did, holder_did, resource, actions):
        """Test cleanup of old expired capabilities."""
        old_time = datetime.now(timezone.utc) - timedelta(days=60)
        recent_time = datetime.now(timezone.utc) - timedelta(days=5)
        now = datetime.now(timezone.utc)
        
        # Old expired capability (should be cleaned up)
        old_cap = Capability(
            id="cap-old",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=old_time - timedelta(hours=1),
            expires_at=old_time,
        )
        
        # Recently expired (should not be cleaned up)
        recent_cap = Capability(
            id="cap-recent",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=recent_time - timedelta(hours=1),
            expires_at=recent_time,
        )
        
        # Valid capability (should not be cleaned up)
        valid_cap = Capability(
            id="cap-valid",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        
        await store.save(old_cap)
        await store.save(recent_cap)
        await store.save(valid_cap)
        
        # Cleanup capabilities expired more than 30 days ago
        removed = await store.cleanup_expired(older_than_days=30)
        assert removed == 1
        
        # Check old is gone, recent and valid remain
        assert await store.get("cap-old") is None
        assert await store.get("cap-recent") is not None
        assert await store.get("cap-valid") is not None


# =============================================================================
# CAPABILITY SERVICE TESTS
# =============================================================================

class TestCapabilityService:
    """Tests for the capability service."""
    
    @pytest.mark.asyncio
    async def test_issue_capability(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test issuing a capability."""
        private_key, _ = issuer_keypair
        
        cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
        )
        
        assert cap.issuer_did == issuer_did
        assert cap.holder_did == holder_did
        assert cap.resource == resource
        assert cap.actions == actions
        assert cap.signature is not None
        assert cap.is_valid
    
    @pytest.mark.asyncio
    async def test_issue_with_custom_ttl(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test issuing a capability with custom TTL."""
        private_key, _ = issuer_keypair
        
        cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
            ttl_seconds=1800,  # 30 minutes
        )
        
        # Check TTL is approximately 30 minutes
        assert 1790 < cap.ttl_seconds <= 1800
    
    @pytest.mark.asyncio
    async def test_issue_enforces_max_ttl(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test that issue enforces maximum TTL."""
        private_key, _ = issuer_keypair
        
        with pytest.raises(CapabilityTTLExceededError):
            await service.issue(
                issuer_did=issuer_did,
                holder_did=holder_did,
                resource=resource,
                actions=actions,
                issuer_private_key=private_key,
                ttl_seconds=MAX_TTL_SECONDS + 1,
            )
    
    @pytest.mark.asyncio
    async def test_issue_stores_capability(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test that issued capabilities are stored."""
        private_key, _ = issuer_keypair
        
        cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
        )
        
        # Should be retrievable
        retrieved = await service.get(cap.id)
        assert retrieved is not None
        assert retrieved.id == cap.id
    
    @pytest.mark.asyncio
    async def test_verify_valid_capability(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test verifying a valid capability."""
        private_key, public_key = issuer_keypair
        
        cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
        )
        
        # Should verify with explicit key
        result = await service.verify(cap, public_key)
        assert result is True
        
        # Should also verify via key resolver
        result = await service.verify(cap)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_expired_capability(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test verifying an expired capability raises error."""
        private_key, _ = issuer_keypair
        
        # Create an expired capability manually
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        cap = Capability(
            id="cap-expired",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=past - timedelta(hours=1),
            expires_at=past,
            signature="invalid",
        )
        
        with pytest.raises(CapabilityExpiredError):
            await service.verify(cap)
    
    @pytest.mark.asyncio
    async def test_verify_revoked_capability(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test verifying a revoked capability raises error."""
        private_key, public_key = issuer_keypair
        
        cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
        )
        
        # Revoke it
        await service.revoke(cap.id)
        
        with pytest.raises(CapabilityRevokedError):
            await service.verify(cap, public_key)
    
    @pytest.mark.asyncio
    async def test_verify_tampered_capability(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test verifying a tampered capability raises error."""
        private_key, public_key = issuer_keypair
        
        cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
        )
        
        # Tamper with actions
        cap.actions.append("admin")
        
        with pytest.raises(CapabilityInvalidSignatureError):
            await service.verify(cap, public_key)
    
    @pytest.mark.asyncio
    async def test_revoke_capability(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test revoking a capability."""
        private_key, _ = issuer_keypair
        
        cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
        )
        
        result = await service.revoke(cap.id)
        assert result is True
        
        # Check it's revoked in store
        retrieved = await service.get(cap.id)
        assert retrieved.is_revoked
    
    @pytest.mark.asyncio
    async def test_list_holder_capabilities(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test listing capabilities for a holder."""
        private_key, _ = issuer_keypair
        
        await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
        )
        
        caps = await service.list_holder_capabilities(holder_did)
        assert len(caps) == 1
        assert caps[0].holder_did == holder_did
    
    @pytest.mark.asyncio
    async def test_list_issuer_capabilities(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test listing capabilities for an issuer."""
        private_key, _ = issuer_keypair
        
        await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
        )
        
        caps = await service.list_issuer_capabilities(issuer_did)
        assert len(caps) == 1
        assert caps[0].issuer_did == issuer_did
    
    @pytest.mark.asyncio
    async def test_check_access_granted(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test checking access when granted."""
        private_key, public_key = issuer_keypair
        
        await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
        )
        
        cap = await service.check_access(holder_did, resource, "read", public_key)
        assert cap is not None
        assert cap.holder_did == holder_did
    
    @pytest.mark.asyncio
    async def test_check_access_denied_wrong_action(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test checking access when action not permitted."""
        private_key, public_key = issuer_keypair
        
        await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read"],  # Only read, not write
            issuer_private_key=private_key,
        )
        
        cap = await service.check_access(holder_did, resource, "write", public_key)
        assert cap is None
    
    @pytest.mark.asyncio
    async def test_check_access_denied_wrong_resource(self, service, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test checking access when resource doesn't match."""
        private_key, public_key = issuer_keypair
        
        await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
        )
        
        cap = await service.check_access(holder_did, "different/resource", "read", public_key)
        assert cap is None


# =============================================================================
# DELEGATION TESTS
# =============================================================================

class TestCapabilityDelegation:
    """Tests for capability delegation."""
    
    @pytest.mark.asyncio
    async def test_delegate_capability(self, service, issuer_keypair, holder_keypair, issuer_did, holder_did, resource):
        """Test delegating a capability."""
        issuer_private, _ = issuer_keypair
        holder_private, _ = holder_keypair
        
        # Issue capability with delegate action
        parent_cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read", "delegate"],
            issuer_private_key=issuer_private,
        )
        
        # Delegate to new holder
        new_holder_did = "did:valence:delegatee789"
        delegated_cap = await service.delegate(
            parent_capability=parent_cap,
            new_holder_did=new_holder_did,
            delegator_private_key=holder_private,
            actions=["read"],
        )
        
        assert delegated_cap.holder_did == new_holder_did
        assert delegated_cap.issuer_did == holder_did  # Holder becomes issuer
        assert delegated_cap.parent_id == parent_cap.id
        assert "read" in delegated_cap.actions
        assert "delegate" not in delegated_cap.actions  # Didn't delegate this
    
    @pytest.mark.asyncio
    async def test_delegate_requires_delegate_action(self, service, issuer_keypair, holder_keypair, issuer_did, holder_did, resource):
        """Test that delegation requires delegate action."""
        issuer_private, _ = issuer_keypair
        holder_private, _ = holder_keypair
        
        # Issue capability WITHOUT delegate action
        parent_cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read"],  # No delegate
            issuer_private_key=issuer_private,
        )
        
        # Try to delegate - should fail
        with pytest.raises(CapabilityInsufficientPermissionError, match="delegation rights"):
            await service.delegate(
                parent_capability=parent_cap,
                new_holder_did="did:valence:delegatee",
                delegator_private_key=holder_private,
                actions=["read"],
            )
    
    @pytest.mark.asyncio
    async def test_delegate_cannot_exceed_parent_actions(self, service, issuer_keypair, holder_keypair, issuer_did, holder_did, resource):
        """Test that delegated actions cannot exceed parent's."""
        issuer_private, _ = issuer_keypair
        holder_private, _ = holder_keypair
        
        parent_cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read", "delegate"],
            issuer_private_key=issuer_private,
        )
        
        with pytest.raises(CapabilityInsufficientPermissionError, match="Cannot delegate actions"):
            await service.delegate(
                parent_capability=parent_cap,
                new_holder_did="did:valence:delegatee",
                delegator_private_key=holder_private,
                actions=["read", "write"],  # write not in parent
            )
    
    @pytest.mark.asyncio
    async def test_delegate_ttl_capped_by_parent(self, service, issuer_keypair, holder_keypair, issuer_did, holder_did, resource):
        """Test that delegated TTL cannot exceed parent's remaining TTL."""
        issuer_private, _ = issuer_keypair
        holder_private, _ = holder_keypair
        
        parent_cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read", "delegate"],
            issuer_private_key=issuer_private,
            ttl_seconds=600,  # 10 minutes
        )
        
        # Try to delegate with longer TTL
        delegated_cap = await service.delegate(
            parent_capability=parent_cap,
            new_holder_did="did:valence:delegatee",
            delegator_private_key=holder_private,
            ttl_seconds=7200,  # 2 hours (longer than parent)
        )
        
        # Should be capped to parent's TTL
        assert delegated_cap.ttl_seconds < 610  # Less than parent's original
    
    @pytest.mark.asyncio
    async def test_delegate_from_expired_parent_fails(self, service, issuer_keypair, holder_keypair, issuer_did, holder_did, resource):
        """Test that delegation from expired parent fails."""
        issuer_private, _ = issuer_keypair
        holder_private, _ = holder_keypair
        
        # Create an expired parent manually
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        parent_cap = Capability(
            id="cap-expired-parent",
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read", "delegate"],
            issued_at=past - timedelta(hours=1),
            expires_at=past,
            signature="fake",
        )
        
        with pytest.raises(CapabilityExpiredError):
            await service.delegate(
                parent_capability=parent_cap,
                new_holder_did="did:valence:delegatee",
                delegator_private_key=holder_private,
            )
    
    @pytest.mark.asyncio
    async def test_delegate_from_revoked_parent_fails(self, service, issuer_keypair, holder_keypair, issuer_did, holder_did, resource):
        """Test that delegation from revoked parent fails."""
        issuer_private, _ = issuer_keypair
        holder_private, _ = holder_keypair
        
        parent_cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read", "delegate"],
            issuer_private_key=issuer_private,
        )
        
        # Revoke the parent capability (set revoked_at)
        parent_cap.revoked_at = datetime.now(timezone.utc)
        
        with pytest.raises(CapabilityRevokedError):
            await service.delegate(
                parent_capability=parent_cap,
                new_holder_did="did:valence:delegatee",
                delegator_private_key=holder_private,
            )


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestCapabilityConfiguration:
    """Tests for capability service configuration."""
    
    @pytest.mark.asyncio
    async def test_custom_default_ttl(self, store, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test custom default TTL."""
        private_key, _ = issuer_keypair
        
        service = CapabilityService(
            store=store,
            default_ttl_seconds=300,  # 5 minutes
        )
        
        cap = await service.issue(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issuer_private_key=private_key,
        )
        
        assert 290 < cap.ttl_seconds <= 300
    
    @pytest.mark.asyncio
    async def test_custom_max_ttl(self, store, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test custom maximum TTL."""
        private_key, _ = issuer_keypair
        
        service = CapabilityService(
            store=store,
            max_ttl_seconds=3600,  # 1 hour max
        )
        
        with pytest.raises(CapabilityTTLExceededError):
            await service.issue(
                issuer_did=issuer_did,
                holder_did=holder_did,
                resource=resource,
                actions=actions,
                issuer_private_key=private_key,
                ttl_seconds=7200,  # 2 hours
            )


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestSingletonAndConvenience:
    """Tests for module-level singleton and convenience functions."""
    
    def test_get_capability_service_returns_singleton(self):
        """Test that get_capability_service returns same instance."""
        service1 = get_capability_service()
        service2 = get_capability_service()
        assert service1 is service2
    
    def test_set_capability_service(self, service):
        """Test setting custom service."""
        original = get_capability_service()
        try:
            set_capability_service(service)
            assert get_capability_service() is service
        finally:
            set_capability_service(original)
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self, issuer_keypair, issuer_did, holder_did, resource, actions):
        """Test module-level convenience functions."""
        private_key, public_key = issuer_keypair
        
        # Create a fresh service
        async def key_resolver(did):
            if "issuer" in did:
                return public_key
            return None
        
        test_service = CapabilityService(
            store=InMemoryCapabilityStore(),
            key_resolver=key_resolver,
        )
        
        original = get_capability_service()
        try:
            set_capability_service(test_service)
            
            # Issue
            cap = await issue_capability(
                issuer_did=issuer_did,
                holder_did=holder_did,
                resource=resource,
                actions=actions,
                issuer_private_key=private_key,
            )
            assert cap is not None
            
            # Verify
            result = await verify_capability(cap, public_key)
            assert result is True
            
            # Revoke
            result = await revoke_capability(cap.id)
            assert result is True
        finally:
            set_capability_service(original)


# =============================================================================
# JWT SERIALIZATION TESTS (Issue #77)
# =============================================================================

import time
import jwt as pyjwt

from valence.privacy.capabilities import CapabilityInvalidError


class TestCapabilityJWTSerialization:
    """Tests for JWT serialization/deserialization (Issue #77).
    
    Capabilities can be serialized to JWT for stateless transport over HTTP APIs,
    separate from the Ed25519 signatures used for cryptographic verification.
    """
    
    JWT_SECRET = "test-secret-key-at-least-32-bytes-long"

    def test_to_jwt(self, issuer_did, holder_did, resource, actions):
        """Test serializing capability to JWT."""
        cap = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
        )

        token = cap.to_jwt(self.JWT_SECRET)

        # Should be a valid JWT string
        assert isinstance(token, str)
        assert token.count(".") == 2  # JWT has 3 parts

        # Decode without verification to inspect
        payload = pyjwt.decode(token, self.JWT_SECRET, algorithms=["HS256"])
        assert payload["jti"] == cap.id
        assert payload["iss"] == issuer_did
        assert payload["sub"] == holder_did
        assert payload["resource"] == resource
        assert payload["actions"] == actions

    def test_from_jwt(self, issuer_did, holder_did, resource, actions):
        """Test deserializing capability from JWT."""
        original = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            metadata={"scope": "limited"},
        )

        token = original.to_jwt(self.JWT_SECRET)
        restored = Capability.from_jwt(token, self.JWT_SECRET)

        assert restored.id == original.id
        assert restored.issuer_did == original.issuer_did
        assert restored.holder_did == original.holder_did
        assert restored.resource == original.resource
        assert restored.actions == original.actions
        assert restored.metadata == original.metadata

    def test_jwt_roundtrip(self, issuer_did, holder_did, resource):
        """Test JWT serialization roundtrip."""
        original = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read", "write", "share"],
            metadata={"max_hops": 2},
        )

        token = original.to_jwt(self.JWT_SECRET)
        restored = Capability.from_jwt(token, self.JWT_SECRET)

        # All fields should match
        assert restored.id == original.id
        assert restored.issuer_did == original.issuer_did
        assert restored.holder_did == original.holder_did
        assert restored.resource == original.resource
        assert restored.actions == original.actions
        assert restored.metadata == original.metadata
        # Timestamps within 1 second (JWT uses integer timestamps)
        assert abs((restored.issued_at - original.issued_at).total_seconds()) < 1
        assert abs((restored.expires_at - original.expires_at).total_seconds()) < 1

    def test_from_jwt_expired(self, issuer_did, holder_did, resource):
        """Test that expired JWT raises CapabilityExpiredError."""
        cap = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read"],
            ttl_seconds=1,  # Very short TTL
        )

        token = cap.to_jwt(self.JWT_SECRET)

        # Wait for expiration
        time.sleep(1.1)

        with pytest.raises(CapabilityExpiredError) as exc_info:
            Capability.from_jwt(token, self.JWT_SECRET)

        assert "expired" in str(exc_info.value).lower()

    def test_from_jwt_expired_skip_verification(self, issuer_did, holder_did, resource):
        """Test that expired JWT can be parsed with verify_exp=False."""
        cap = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read"],
            ttl_seconds=1,
        )

        token = cap.to_jwt(self.JWT_SECRET)
        time.sleep(1.1)

        # Should not raise with verify_exp=False
        restored = Capability.from_jwt(token, self.JWT_SECRET, verify_exp=False)
        assert restored.id == cap.id

    def test_from_jwt_invalid_secret(self, issuer_did, holder_did, resource):
        """Test that wrong secret raises CapabilityInvalidSignatureError."""
        cap = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read"],
        )

        token = cap.to_jwt(self.JWT_SECRET)

        with pytest.raises(CapabilityInvalidSignatureError):
            Capability.from_jwt(token, "wrong-secret-key-at-least-32-bytes")

    def test_from_jwt_malformed(self):
        """Test that malformed JWT raises CapabilityInvalidSignatureError."""
        with pytest.raises(CapabilityInvalidSignatureError):
            Capability.from_jwt("not.a.valid.jwt", self.JWT_SECRET)

    def test_jwt_custom_algorithm(self, issuer_did, holder_did, resource):
        """Test JWT with custom algorithm."""
        cap = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read"],
        )

        # Use HS384 instead of default HS256
        # Note: Using longer secret to avoid warning
        long_secret = self.JWT_SECRET + "0" * 20
        token = cap.to_jwt(long_secret, algorithm="HS384")
        restored = Capability.from_jwt(token, long_secret, algorithm="HS384")

        assert restored.id == cap.id

    def test_invalid_error_alias(self):
        """Test CapabilityInvalidError is alias for CapabilityInvalidSignatureError."""
        assert CapabilityInvalidError is CapabilityInvalidSignatureError


class TestCapabilityCreateFactory:
    """Tests for Capability.create() factory method (Issue #77)."""

    def test_create_with_defaults(self, issuer_did, holder_did, resource, actions):
        """Test creating a capability with default TTL."""
        cap = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
        )

        assert cap.issuer_did == issuer_did
        assert cap.holder_did == holder_did
        assert cap.resource == resource
        assert cap.actions == actions
        assert cap.metadata == {}
        assert cap.id is not None
        assert len(cap.id) == 36  # UUID format

        # Check TTL is approximately 1 hour
        ttl = cap.ttl_seconds
        assert 3599 < ttl <= DEFAULT_TTL_SECONDS

    def test_create_with_custom_ttl(self, issuer_did, holder_did, resource):
        """Test creating a capability with custom TTL."""
        cap = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read"],
            ttl_seconds=300,  # 5 minutes
        )

        ttl = cap.ttl_seconds
        assert 299 < ttl <= 300

    def test_create_with_metadata(self, issuer_did, holder_did, resource):
        """Test creating a capability with metadata."""
        metadata = {
            "max_uses": 10,
            "ip_whitelist": ["192.168.1.0/24"],
            "rate_limit": "10/minute",
        }

        cap = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read", "write"],
            metadata=metadata,
        )

        assert cap.metadata == metadata
        assert cap.metadata["max_uses"] == 10

    def test_create_with_explicit_id(self, issuer_did, holder_did, resource):
        """Test creating a capability with explicit ID."""
        explicit_id = "custom-capability-id-12345"
        cap = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=["read"],
            capability_id=explicit_id,
        )

        assert cap.id == explicit_id

    def test_actions_are_copied(self, issuer_did, holder_did, resource):
        """Test that actions list is copied, not referenced."""
        original_actions = ["read", "write"]
        cap = Capability.create(
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=original_actions,
        )

        # Modify original list
        original_actions.append("delete")

        # Capability should not be affected
        assert cap.actions == ["read", "write"]
        assert "delete" not in cap.actions


class TestCapabilityDictNaiveDatetime:
    """Tests for from_dict handling of naive datetimes (Issue #77)."""

    def test_from_dict_naive_datetime(self, issuer_did, holder_did, resource):
        """Test deserializing with naive datetime (assumes UTC)."""
        data = {
            "id": "test-id",
            "issuer_did": issuer_did,
            "holder_did": holder_did,
            "resource": resource,
            "actions": ["read"],
            "issued_at": "2026-01-15T10:00:00",  # No timezone
            "expires_at": "2026-01-15T11:00:00",
        }

        cap = Capability.from_dict(data)

        # Should assume UTC
        assert cap.issued_at.tzinfo == timezone.utc
        assert cap.expires_at.tzinfo == timezone.utc
