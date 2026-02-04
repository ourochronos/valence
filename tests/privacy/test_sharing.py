"""Tests for sharing service - consent chains, encryption, and policy validation."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass
from typing import Optional, Any
import json
import hashlib

from valence.privacy.sharing import (
    ShareRequest,
    ShareResult,
    ReceiveRequest,
    ReceiveResult,
    RevokeRequest,
    RevokeResult,
    ReshareRequest,
    ReshareResult,
    RevocationNotification,
    Notification,
    SharingService,
    Share,
    ConsentChainEntry,
)
from valence.privacy.types import SharePolicy, ShareLevel, EnforcementType, PropagationRules
from valence.privacy.encryption import generate_keypair, EncryptionEnvelope


@dataclass
class MockBelief:
    """Mock belief for testing."""
    id: str
    content: str


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self):
        self.beliefs: dict[str, MockBelief] = {}
        self.consent_chains: dict[str, dict] = {}
        self.shares: dict[str, dict] = {}
        self.notifications: dict[str, dict] = {}
    
    async def get_belief(self, belief_id: str) -> Optional[MockBelief]:
        return self.beliefs.get(belief_id)
    
    async def create_consent_chain(
        self,
        id: str,
        belief_id: str,
        origin_sharer: str,
        origin_timestamp: float,
        origin_policy: dict,
        origin_signature: bytes,
        chain_hash: bytes,
    ) -> None:
        self.consent_chains[id] = {
            "id": id,
            "belief_id": belief_id,
            "origin_sharer": origin_sharer,
            "origin_timestamp": origin_timestamp,
            "origin_policy": origin_policy,
            "origin_signature": origin_signature,
            "chain_hash": chain_hash,
            "hops": [],
            "current_hop": 0,
            "revoked": False,
            "revoked_at": None,
            "revoked_by": None,
            "revoke_reason": None,
        }
    
    async def create_share(
        self,
        id: str,
        consent_chain_id: str,
        encrypted_envelope: dict,
        recipient_did: str,
        belief_id: str = None,
    ) -> None:
        import time
        self.shares[id] = {
            "id": id,
            "consent_chain_id": consent_chain_id,
            "encrypted_envelope": encrypted_envelope,
            "recipient_did": recipient_did,
            "belief_id": belief_id,
            "created_at": time.time(),
            "accessed_at": None,
            "received_at": None,
        }
    
    async def get_share(self, share_id: str) -> Optional[Share]:
        data = self.shares.get(share_id)
        if not data:
            return None
        return Share(
            id=data["id"],
            consent_chain_id=data["consent_chain_id"],
            encrypted_envelope=data["encrypted_envelope"],
            recipient_did=data["recipient_did"],
            created_at=data["created_at"],
            belief_id=data.get("belief_id"),
            accessed_at=data.get("accessed_at"),
            received_at=data.get("received_at"),
        )
    
    async def list_shares(
        self,
        sharer_did: Optional[str] = None,
        recipient_did: Optional[str] = None,
        limit: int = 100,
        include_revoked: bool = False,
    ) -> list[Share]:
        results = []
        for data in self.shares.values():
            if recipient_did and data["recipient_did"] != recipient_did:
                continue
            # Filter out revoked shares unless include_revoked is True
            if not include_revoked:
                consent_chain = self.consent_chains.get(data["consent_chain_id"])
                if consent_chain and consent_chain.get("revoked"):
                    continue
            # Note: sharer_did filtering would need consent_chain lookup
            results.append(Share(
                id=data["id"],
                consent_chain_id=data["consent_chain_id"],
                encrypted_envelope=data["encrypted_envelope"],
                recipient_did=data["recipient_did"],
                created_at=data["created_at"],
                belief_id=data.get("belief_id"),
                accessed_at=data.get("accessed_at"),
                received_at=data.get("received_at"),
            ))
            if len(results) >= limit:
                break
        return results
    
    async def get_consent_chain(self, chain_id: str) -> Optional[ConsentChainEntry]:
        data = self.consent_chains.get(chain_id)
        if not data:
            return None
        return ConsentChainEntry(
            id=data["id"],
            belief_id=data["belief_id"],
            origin_sharer=data["origin_sharer"],
            origin_timestamp=data["origin_timestamp"],
            origin_policy=data["origin_policy"],
            origin_signature=data["origin_signature"],
            hops=data["hops"],
            chain_hash=data["chain_hash"],
            current_hop=data.get("current_hop", 0),
            revoked=data["revoked"],
            revoked_at=data.get("revoked_at"),
            revoked_by=data.get("revoked_by"),
            revoke_reason=data.get("revoke_reason"),
        )
    
    async def revoke_consent_chain(
        self,
        consent_chain_id: str,
        revoked_at: float,
        revoked_by: str,
        reason: Optional[str] = None,
    ) -> None:
        """Mark a consent chain as revoked."""
        if consent_chain_id in self.consent_chains:
            self.consent_chains[consent_chain_id]["revoked"] = True
            self.consent_chains[consent_chain_id]["revoked_at"] = revoked_at
            self.consent_chains[consent_chain_id]["revoked_by"] = revoked_by
            self.consent_chains[consent_chain_id]["revoke_reason"] = reason
    
    async def get_shares_by_consent_chain(
        self,
        consent_chain_id: str,
    ) -> list[Share]:
        """Get all shares associated with a consent chain."""
        results = []
        for data in self.shares.values():
            if data["consent_chain_id"] == consent_chain_id:
                results.append(Share(
                    id=data["id"],
                    consent_chain_id=data["consent_chain_id"],
                    encrypted_envelope=data["encrypted_envelope"],
                    recipient_did=data["recipient_did"],
                    created_at=data["created_at"],
                    belief_id=data.get("belief_id"),
                    accessed_at=data.get("accessed_at"),
                    received_at=data.get("received_at"),
                ))
        return results
    
    async def acknowledge_share(
        self,
        share_id: str,
        received_at: float,
    ) -> None:
        """Mark a share as received/acknowledged."""
        if share_id in self.shares:
            self.shares[share_id]["received_at"] = received_at
    
    async def add_consent_chain_hop(
        self,
        consent_chain_id: str,
        hop: dict,
    ) -> None:
        """Add a hop to a consent chain."""
        if consent_chain_id in self.consent_chains:
            self.consent_chains[consent_chain_id]["hops"].append(hop)
    
    async def update_consent_chain_hop_count(
        self,
        consent_chain_id: str,
        current_hop: int,
    ) -> None:
        """Update the current hop count of a consent chain."""
        if consent_chain_id in self.consent_chains:
            self.consent_chains[consent_chain_id]["current_hop"] = current_hop
    
    async def get_pending_shares(
        self,
        recipient_did: str,
    ) -> list[Share]:
        """Get shares that haven't been received yet for a recipient."""
        results = []
        for data in self.shares.values():
            if data["recipient_did"] != recipient_did:
                continue
            if data.get("received_at") is not None:
                continue
            # Filter out revoked shares
            consent_chain = self.consent_chains.get(data["consent_chain_id"])
            if consent_chain and consent_chain.get("revoked"):
                continue
            results.append(Share(
                id=data["id"],
                consent_chain_id=data["consent_chain_id"],
                encrypted_envelope=data["encrypted_envelope"],
                recipient_did=data["recipient_did"],
                created_at=data["created_at"],
                belief_id=data.get("belief_id"),
                accessed_at=data.get("accessed_at"),
                received_at=data.get("received_at"),
            ))
        return results
    
    async def create_notification(
        self,
        id: str,
        recipient_did: str,
        notification_type: str,
        payload: dict,
        created_at: float,
    ) -> None:
        """Create a notification for a recipient."""
        self.notifications[id] = {
            "id": id,
            "recipient_did": recipient_did,
            "notification_type": notification_type,
            "payload": payload,
            "created_at": created_at,
            "acknowledged_at": None,
        }
    
    async def get_pending_notifications(
        self,
        recipient_did: str,
    ) -> list[Notification]:
        """Get pending notifications for a recipient."""
        results = []
        for data in self.notifications.values():
            if data["recipient_did"] != recipient_did:
                continue
            if data.get("acknowledged_at") is not None:
                continue
            results.append(Notification(
                id=data["id"],
                recipient_did=data["recipient_did"],
                notification_type=data["notification_type"],
                payload=data["payload"],
                created_at=data["created_at"],
                acknowledged_at=data.get("acknowledged_at"),
            ))
        return results
    
    async def acknowledge_notification(
        self,
        notification_id: str,
        acknowledged_at: float,
    ) -> None:
        """Mark a notification as acknowledged/delivered."""
        if notification_id in self.notifications:
            self.notifications[notification_id]["acknowledged_at"] = acknowledged_at
    
    async def get_notification(
        self,
        notification_id: str,
    ) -> Optional[Notification]:
        """Get a notification by ID."""
        data = self.notifications.get(notification_id)
        if not data:
            return None
        return Notification(
            id=data["id"],
            recipient_did=data["recipient_did"],
            notification_type=data["notification_type"],
            payload=data["payload"],
            created_at=data["created_at"],
            acknowledged_at=data.get("acknowledged_at"),
        )


class MockIdentityService:
    """Mock identity service for testing."""
    
    def __init__(self, did: str = "did:key:test-sharer"):
        self.did = did
        # Generate real X25519 keypairs for encryption testing
        self.keypairs: dict[str, tuple[bytes, bytes]] = {}
        # Ed25519-like signing (simplified for testing)
        self._sign_key = b"test-signing-key-32-bytes-long!!"
    
    def add_identity(self, did: str) -> bytes:
        """Add an identity and return its public key."""
        private, public = generate_keypair()
        self.keypairs[did] = (private, public)
        return public
    
    async def get_public_key(self, did: str) -> Optional[bytes]:
        if did in self.keypairs:
            return self.keypairs[did][1]
        return None
    
    async def get_private_key(self, did: str) -> Optional[bytes]:
        if did in self.keypairs:
            return self.keypairs[did][0]
        return None
    
    def sign(self, data: dict) -> bytes:
        """Sign data (simplified for testing)."""
        data_json = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(self._sign_key + data_json).digest()
    
    def get_did(self) -> str:
        return self.did


class MockDomainService:
    """Mock domain service for testing BOUNDED sharing."""
    
    def __init__(self):
        # Maps DID -> set of domains they belong to
        self.memberships: dict[str, set[str]] = {}
    
    def add_member(self, did: str, domain: str) -> None:
        """Add a DID to a domain."""
        if did not in self.memberships:
            self.memberships[did] = set()
        self.memberships[did].add(domain)
    
    def remove_member(self, did: str, domain: str) -> None:
        """Remove a DID from a domain."""
        if did in self.memberships:
            self.memberships[did].discard(domain)
    
    async def is_member(self, did: str, domain: str) -> bool:
        """Check if a DID is a member of a domain."""
        if did not in self.memberships:
            return False
        return domain in self.memberships[did]
    
    async def get_domains_for_did(self, did: str) -> list[str]:
        """Get all domains a DID belongs to."""
        if did not in self.memberships:
            return []
        return list(self.memberships[did])


class TestSharingService:
    """Tests for SharingService."""
    
    @pytest.fixture
    def db(self):
        """Create a mock database."""
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        """Create a mock identity service."""
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        """Create a sharing service."""
        return SharingService(db, identity)
    
    @pytest.mark.asyncio
    async def test_share_basic(self, service, db, identity):
        """Test basic share operation."""
        # Setup: add belief and recipient identity
        belief_id = "belief-123"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Test belief content")
        
        recipient_did = "did:key:recipient"
        identity.add_identity(recipient_did)
        
        # Share
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
        )
        
        result = await service.share(request, identity.get_did())
        
        # Verify result
        assert result.share_id is not None
        assert result.consent_chain_id is not None
        assert result.encrypted_for == recipient_did
        assert result.created_at > 0
        
        # Verify consent chain was created
        assert result.consent_chain_id in db.consent_chains
        chain = db.consent_chains[result.consent_chain_id]
        assert chain["belief_id"] == belief_id
        assert chain["origin_sharer"] == identity.get_did()
        
        # Verify share was created
        assert result.share_id in db.shares
        share = db.shares[result.share_id]
        assert share["recipient_did"] == recipient_did
        assert "encrypted_content" in share["encrypted_envelope"]
    
    @pytest.mark.asyncio
    async def test_share_with_explicit_policy(self, service, db, identity):
        """Test share with explicit DIRECT policy."""
        belief_id = "belief-456"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Another test belief")
        
        recipient_did = "did:key:bob"
        identity.add_identity(recipient_did)
        
        policy = SharePolicy.direct([recipient_did])
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        result = await service.share(request, identity.get_did())
        
        # Verify policy was stored in consent chain
        chain = db.consent_chains[result.consent_chain_id]
        assert chain["origin_policy"]["level"] == "direct"
        assert chain["origin_policy"]["enforcement"] == "cryptographic"
    
    @pytest.mark.asyncio
    async def test_share_belief_not_found(self, service, db, identity):
        """Test share with non-existent belief."""
        identity.add_identity("did:key:recipient")
        
        request = ShareRequest(
            belief_id="nonexistent",
            recipient_did="did:key:recipient",
        )
        
        with pytest.raises(ValueError, match="Belief not found"):
            await service.share(request, identity.get_did())
    
    @pytest.mark.asyncio
    async def test_share_recipient_not_found(self, service, db, identity):
        """Test share with non-existent recipient."""
        belief_id = "belief-789"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did="did:key:unknown",
        )
        
        with pytest.raises(ValueError, match="Recipient not found"):
            await service.share(request, identity.get_did())
    
    @pytest.mark.asyncio
    async def test_share_unsupported_level_rejected(self, service, db, identity):
        """Test that unsupported levels (CASCADING, PUBLIC) are rejected."""
        belief_id = "belief-001"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        
        recipient_did = "did:key:recipient"
        identity.add_identity(recipient_did)
        
        # Try PUBLIC level (not yet supported)
        policy = SharePolicy.public()
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        with pytest.raises(ValueError, match="Only DIRECT and BOUNDED sharing supported"):
            await service.share(request, identity.get_did())
    
    @pytest.mark.asyncio
    async def test_share_recipient_not_in_policy(self, service, db, identity):
        """Test that recipient must be in policy recipients list."""
        belief_id = "belief-002"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        
        recipient_did = "did:key:bob"
        other_did = "did:key:alice"
        identity.add_identity(recipient_did)
        
        # Policy with different recipient
        policy = SharePolicy.direct([other_did])
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        with pytest.raises(ValueError, match="Recipient not in policy"):
            await service.share(request, identity.get_did())


class TestEncryptionVerification:
    """Tests verifying encryption works end-to-end."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        return SharingService(db, identity)
    
    @pytest.mark.asyncio
    async def test_encrypted_content_decryptable(self, service, db, identity):
        """Test that shared content can be decrypted by recipient."""
        belief_id = "belief-enc-001"
        original_content = "This is secret content that should be encrypted."
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=original_content)
        
        recipient_did = "did:key:decryptor"
        recipient_public = identity.add_identity(recipient_did)
        recipient_private = identity.keypairs[recipient_did][0]
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
        )
        
        result = await service.share(request, identity.get_did())
        
        # Get the encrypted envelope
        share = db.shares[result.share_id]
        envelope = EncryptionEnvelope.from_dict(share["encrypted_envelope"])
        
        # Decrypt as recipient
        decrypted = EncryptionEnvelope.decrypt(envelope, recipient_private)
        
        assert decrypted.decode("utf-8") == original_content
    
    @pytest.mark.asyncio
    async def test_encrypted_content_different_per_recipient(self, service, db, identity):
        """Test that encryption produces different ciphertext for same content."""
        belief_id = "belief-enc-002"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Same content")
        
        # Two different recipients
        identity.add_identity("did:key:alice")
        identity.add_identity("did:key:bob")
        
        request1 = ShareRequest(belief_id=belief_id, recipient_did="did:key:alice")
        request2 = ShareRequest(belief_id=belief_id, recipient_did="did:key:bob")
        
        result1 = await service.share(request1, identity.get_did())
        result2 = await service.share(request2, identity.get_did())
        
        share1 = db.shares[result1.share_id]
        share2 = db.shares[result2.share_id]
        
        # Different encrypted content (due to different ephemeral keys and recipients)
        assert share1["encrypted_envelope"]["encrypted_content"] != share2["encrypted_envelope"]["encrypted_content"]


class TestConsentChain:
    """Tests for consent chain creation and integrity."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        return SharingService(db, identity)
    
    @pytest.mark.asyncio
    async def test_consent_chain_has_signature(self, service, db, identity):
        """Test that consent chain includes a signature."""
        belief_id = "belief-cc-001"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        
        recipient_did = "did:key:recipient"
        identity.add_identity(recipient_did)
        
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        result = await service.share(request, identity.get_did())
        
        chain = db.consent_chains[result.consent_chain_id]
        
        assert chain["origin_signature"] is not None
        assert len(chain["origin_signature"]) == 32  # SHA256 digest
    
    @pytest.mark.asyncio
    async def test_consent_chain_has_hash(self, service, db, identity):
        """Test that consent chain has integrity hash."""
        belief_id = "belief-cc-002"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        
        recipient_did = "did:key:recipient"
        identity.add_identity(recipient_did)
        
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        result = await service.share(request, identity.get_did())
        
        chain = db.consent_chains[result.consent_chain_id]
        
        assert chain["chain_hash"] is not None
        assert len(chain["chain_hash"]) == 32  # SHA256 digest
    
    @pytest.mark.asyncio
    async def test_consent_chain_includes_policy(self, service, db, identity):
        """Test that consent chain stores the policy."""
        belief_id = "belief-cc-003"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        
        recipient_did = "did:key:recipient"
        identity.add_identity(recipient_did)
        
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        result = await service.share(request, identity.get_did())
        
        chain = db.consent_chains[result.consent_chain_id]
        
        assert chain["origin_policy"]["level"] == "direct"
        assert chain["origin_policy"]["enforcement"] == "cryptographic"
        assert recipient_did in chain["origin_policy"]["recipients"]


class TestShareRetrieval:
    """Tests for share retrieval operations."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        return SharingService(db, identity)
    
    @pytest.mark.asyncio
    async def test_get_share(self, service, db, identity):
        """Test getting a share by ID."""
        # Create a share
        belief_id = "belief-get-001"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        identity.add_identity("did:key:recipient")
        
        request = ShareRequest(belief_id=belief_id, recipient_did="did:key:recipient")
        result = await service.share(request, identity.get_did())
        
        # Retrieve it
        share = await service.get_share(result.share_id)
        
        assert share is not None
        assert share.id == result.share_id
        assert share.recipient_did == "did:key:recipient"
    
    @pytest.mark.asyncio
    async def test_get_share_not_found(self, service, db, identity):
        """Test getting a non-existent share."""
        share = await service.get_share("nonexistent-id")
        assert share is None
    
    @pytest.mark.asyncio
    async def test_list_shares(self, service, db, identity):
        """Test listing shares."""
        # Create multiple shares
        for i in range(3):
            belief_id = f"belief-list-{i}"
            db.beliefs[belief_id] = MockBelief(id=belief_id, content=f"Content {i}")
            identity.add_identity(f"did:key:recipient-{i}")
            
            request = ShareRequest(
                belief_id=belief_id,
                recipient_did=f"did:key:recipient-{i}",
            )
            await service.share(request, identity.get_did())
        
        # List all
        shares = await service.list_shares()
        assert len(shares) == 3
    
    @pytest.mark.asyncio
    async def test_list_shares_by_recipient(self, service, db, identity):
        """Test listing shares filtered by recipient."""
        # Create shares to different recipients
        for i in range(3):
            belief_id = f"belief-filter-{i}"
            db.beliefs[belief_id] = MockBelief(id=belief_id, content=f"Content {i}")
            identity.add_identity(f"did:key:r-{i}")
            
            request = ShareRequest(
                belief_id=belief_id,
                recipient_did=f"did:key:r-{i}",
            )
            await service.share(request, identity.get_did())
        
        # Filter by recipient
        shares = await service.list_shares(recipient_did="did:key:r-1")
        assert len(shares) == 1
        assert shares[0].recipient_did == "did:key:r-1"


class TestRevocation:
    """Tests for share revocation functionality."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        return SharingService(db, identity)
    
    async def _create_share(self, service, db, identity, belief_id="belief-revoke-001"):
        """Helper to create a share for testing."""
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content to revoke")
        recipient_did = "did:key:recipient-revoke"
        identity.add_identity(recipient_did)
        
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        return await service.share(request, identity.get_did())
    
    @pytest.mark.asyncio
    async def test_revoke_share_success(self, service, db, identity):
        """Test successful revocation of a share."""
        # Create a share
        share_result = await self._create_share(service, db, identity)
        
        # Revoke it
        revoke_request = RevokeRequest(
            share_id=share_result.share_id,
            reason="Testing revocation",
        )
        result = await service.revoke_share(revoke_request, identity.get_did())
        
        # Verify result
        assert result.share_id == share_result.share_id
        assert result.consent_chain_id == share_result.consent_chain_id
        assert result.revoked_at > 0
        assert result.affected_recipients == 1
        
        # Verify consent chain is marked revoked
        chain = db.consent_chains[share_result.consent_chain_id]
        assert chain["revoked"] is True
        assert chain["revoked_at"] == result.revoked_at
        assert chain["revoked_by"] == identity.get_did()
        assert chain["revoke_reason"] == "Testing revocation"
    
    @pytest.mark.asyncio
    async def test_revoke_share_not_found(self, service, db, identity):
        """Test revocation of non-existent share."""
        revoke_request = RevokeRequest(share_id="nonexistent-share-id")
        
        with pytest.raises(ValueError, match="Share not found"):
            await service.revoke_share(revoke_request, identity.get_did())
    
    @pytest.mark.asyncio
    async def test_revoke_share_permission_denied(self, service, db, identity):
        """Test that only original sharer can revoke."""
        # Create a share
        share_result = await self._create_share(service, db, identity)
        
        # Try to revoke as different DID
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        
        with pytest.raises(PermissionError, match="Only original sharer can revoke"):
            await service.revoke_share(revoke_request, "did:key:imposter")
    
    @pytest.mark.asyncio
    async def test_revoke_share_already_revoked(self, service, db, identity):
        """Test that already revoked share cannot be revoked again."""
        # Create and revoke a share
        share_result = await self._create_share(service, db, identity)
        
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Try to revoke again
        with pytest.raises(ValueError, match="Share already revoked"):
            await service.revoke_share(revoke_request, identity.get_did())
    
    @pytest.mark.asyncio
    async def test_revoke_without_reason(self, service, db, identity):
        """Test revocation without providing a reason."""
        # Create a share
        share_result = await self._create_share(service, db, identity, "belief-no-reason")
        
        # Revoke without reason
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        result = await service.revoke_share(revoke_request, identity.get_did())
        
        assert result.share_id == share_result.share_id
        
        # Verify reason is None in consent chain
        chain = db.consent_chains[share_result.consent_chain_id]
        assert chain["revoke_reason"] is None


class TestRevocationFiltering:
    """Tests for filtering revoked shares from listings."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        return SharingService(db, identity)
    
    @pytest.mark.asyncio
    async def test_list_shares_excludes_revoked_by_default(self, service, db, identity):
        """Test that revoked shares are excluded from listings by default."""
        # Create two shares
        for i in range(2):
            belief_id = f"belief-filter-revoke-{i}"
            db.beliefs[belief_id] = MockBelief(id=belief_id, content=f"Content {i}")
            identity.add_identity(f"did:key:filter-r-{i}")
            
            request = ShareRequest(
                belief_id=belief_id,
                recipient_did=f"did:key:filter-r-{i}",
            )
            result = await service.share(request, identity.get_did())
            
            # Revoke the first share
            if i == 0:
                revoke_request = RevokeRequest(share_id=result.share_id)
                await service.revoke_share(revoke_request, identity.get_did())
        
        # List without including revoked
        shares = await service.list_shares()
        assert len(shares) == 1
        assert shares[0].recipient_did == "did:key:filter-r-1"
    
    @pytest.mark.asyncio
    async def test_list_shares_includes_revoked_when_requested(self, service, db, identity):
        """Test that revoked shares are included when explicitly requested."""
        # Create two shares
        share_ids = []
        for i in range(2):
            belief_id = f"belief-include-revoke-{i}"
            db.beliefs[belief_id] = MockBelief(id=belief_id, content=f"Content {i}")
            identity.add_identity(f"did:key:include-r-{i}")
            
            request = ShareRequest(
                belief_id=belief_id,
                recipient_did=f"did:key:include-r-{i}",
            )
            result = await service.share(request, identity.get_did())
            share_ids.append(result.share_id)
            
            # Revoke the first share
            if i == 0:
                revoke_request = RevokeRequest(share_id=result.share_id)
                await service.revoke_share(revoke_request, identity.get_did())
        
        # List with including revoked
        shares = await service.list_shares(include_revoked=True)
        assert len(shares) == 2
    
    @pytest.mark.asyncio
    async def test_get_consent_chain_shows_revocation_details(self, service, db, identity):
        """Test that consent chain includes revocation details after revocation."""
        # Create and revoke a share
        belief_id = "belief-chain-details"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        identity.add_identity("did:key:chain-recipient")
        
        request = ShareRequest(belief_id=belief_id, recipient_did="did:key:chain-recipient")
        share_result = await service.share(request, identity.get_did())
        
        revoke_request = RevokeRequest(
            share_id=share_result.share_id,
            reason="Detailed revocation test",
        )
        revoke_result = await service.revoke_share(revoke_request, identity.get_did())
        
        # Get consent chain and check revocation details
        chain = await service.get_consent_chain(share_result.consent_chain_id)
        
        assert chain is not None
        assert chain.revoked is True
        assert chain.revoked_at == revoke_result.revoked_at
        assert chain.revoked_by == identity.get_did()
        assert chain.revoke_reason == "Detailed revocation test"


class TestReceive:
    """Tests for share receive/acknowledgment functionality."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        return SharingService(db, identity)
    
    async def _create_share(
        self, service, db, identity, 
        belief_id="belief-receive-001",
        recipient_did="did:key:recipient-receive",
        content="Secret content to share"
    ):
        """Helper to create a share for testing."""
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=content)
        identity.add_identity(recipient_did)
        
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        return await service.share(request, identity.get_did()), recipient_did
    
    @pytest.mark.asyncio
    async def test_receive_success(self, service, db, identity):
        """Test successful receive and decryption of a share."""
        original_content = "This is the secret shared content."
        share_result, recipient_did = await self._create_share(
            service, db, identity,
            content=original_content
        )
        
        # Receive the share
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        result = await service.receive(receive_request, recipient_did)
        
        # Verify result
        assert result.share_id == share_result.share_id
        assert result.belief_id == "belief-receive-001"
        assert result.content.decode("utf-8") == original_content
        assert result.sharer_did == identity.get_did()
        assert result.received_at > 0
        assert result.consent_chain_id == share_result.consent_chain_id
    
    @pytest.mark.asyncio
    async def test_receive_decryption_verification(self, service, db, identity):
        """Test that decryption produces correct content."""
        test_cases = [
            "Simple text",
            "Unicode: 你好世界 🌍",
            "Long content " * 100,
            '{"json": "data", "number": 123}',
        ]
        
        for i, content in enumerate(test_cases):
            belief_id = f"belief-decrypt-{i}"
            recipient_did = f"did:key:decrypt-recipient-{i}"
            
            share_result, recipient = await self._create_share(
                service, db, identity,
                belief_id=belief_id,
                recipient_did=recipient_did,
                content=content
            )
            
            receive_request = ReceiveRequest(share_id=share_result.share_id)
            result = await service.receive(receive_request, recipient)
            
            assert result.content.decode("utf-8") == content
    
    @pytest.mark.asyncio
    async def test_receive_not_intended_recipient(self, service, db, identity):
        """Test that only intended recipient can receive."""
        share_result, recipient_did = await self._create_share(service, db, identity)
        
        # Try to receive as different DID
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        
        with pytest.raises(PermissionError, match="Not the intended recipient"):
            await service.receive(receive_request, "did:key:imposter")
    
    @pytest.mark.asyncio
    async def test_receive_already_received(self, service, db, identity):
        """Test that share can only be received once."""
        share_result, recipient_did = await self._create_share(
            service, db, identity,
            belief_id="belief-once",
            recipient_did="did:key:once-recipient"
        )
        
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        
        # First receive should succeed
        await service.receive(receive_request, recipient_did)
        
        # Second receive should fail
        with pytest.raises(ValueError, match="Already received"):
            await service.receive(receive_request, recipient_did)
    
    @pytest.mark.asyncio
    async def test_receive_revoked_share(self, service, db, identity):
        """Test that revoked share cannot be received."""
        share_result, recipient_did = await self._create_share(
            service, db, identity,
            belief_id="belief-revoke-before-receive",
            recipient_did="did:key:revoke-recipient"
        )
        
        # Revoke the share first
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Try to receive
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        
        with pytest.raises(ValueError, match="Share has been revoked"):
            await service.receive(receive_request, recipient_did)
    
    @pytest.mark.asyncio
    async def test_receive_share_not_found(self, service, db, identity):
        """Test receive with non-existent share."""
        receive_request = ReceiveRequest(share_id="nonexistent-share-id")
        
        with pytest.raises(ValueError, match="Share not found"):
            await service.receive(receive_request, "did:key:anyone")
    
    @pytest.mark.asyncio
    async def test_receive_creates_consent_chain_hop(self, service, db, identity):
        """Test that receiving adds a hop to the consent chain."""
        share_result, recipient_did = await self._create_share(
            service, db, identity,
            belief_id="belief-hop",
            recipient_did="did:key:hop-recipient"
        )
        
        # Verify no hops initially
        chain_before = db.consent_chains[share_result.consent_chain_id]
        assert len(chain_before["hops"]) == 0
        
        # Receive
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        result = await service.receive(receive_request, recipient_did)
        
        # Verify hop was added
        chain_after = db.consent_chains[share_result.consent_chain_id]
        assert len(chain_after["hops"]) == 1
        
        hop = chain_after["hops"][0]
        assert hop["recipient"] == recipient_did
        assert hop["received_at"] == result.received_at
        assert hop["acknowledged"] is True
        assert "signature" in hop
    
    @pytest.mark.asyncio
    async def test_receive_updates_share_received_at(self, service, db, identity):
        """Test that receiving updates the share's received_at field."""
        share_result, recipient_did = await self._create_share(
            service, db, identity,
            belief_id="belief-timestamp",
            recipient_did="did:key:timestamp-recipient"
        )
        
        # Verify not received initially
        share_before = await service.get_share(share_result.share_id)
        assert share_before.received_at is None
        
        # Receive
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        result = await service.receive(receive_request, recipient_did)
        
        # Verify received_at was set
        share_after = await service.get_share(share_result.share_id)
        assert share_after.received_at == result.received_at


class TestListPendingShares:
    """Tests for listing pending shares."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        return SharingService(db, identity)
    
    @pytest.mark.asyncio
    async def test_list_pending_shares(self, service, db, identity):
        """Test listing pending shares for a recipient."""
        recipient_did = "did:key:pending-recipient"
        identity.add_identity(recipient_did)
        
        # Create multiple shares to same recipient
        for i in range(3):
            belief_id = f"belief-pending-{i}"
            db.beliefs[belief_id] = MockBelief(id=belief_id, content=f"Content {i}")
            
            request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
            await service.share(request, identity.get_did())
        
        # All should be pending
        pending = await service.list_pending_shares(recipient_did)
        assert len(pending) == 3
    
    @pytest.mark.asyncio
    async def test_list_pending_excludes_received(self, service, db, identity):
        """Test that received shares are excluded from pending."""
        recipient_did = "did:key:pending-exclude"
        identity.add_identity(recipient_did)
        
        # Create 3 shares
        share_ids = []
        for i in range(3):
            belief_id = f"belief-pending-ex-{i}"
            db.beliefs[belief_id] = MockBelief(id=belief_id, content=f"Content {i}")
            
            request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
            result = await service.share(request, identity.get_did())
            share_ids.append(result.share_id)
        
        # Receive one of them
        receive_request = ReceiveRequest(share_id=share_ids[0])
        await service.receive(receive_request, recipient_did)
        
        # Only 2 should be pending
        pending = await service.list_pending_shares(recipient_did)
        assert len(pending) == 2
        
        pending_ids = [s.id for s in pending]
        assert share_ids[0] not in pending_ids
        assert share_ids[1] in pending_ids
        assert share_ids[2] in pending_ids
    
    @pytest.mark.asyncio
    async def test_list_pending_excludes_revoked(self, service, db, identity):
        """Test that revoked shares are excluded from pending."""
        recipient_did = "did:key:pending-revoked"
        identity.add_identity(recipient_did)
        
        # Create 2 shares
        share_ids = []
        for i in range(2):
            belief_id = f"belief-pending-rev-{i}"
            db.beliefs[belief_id] = MockBelief(id=belief_id, content=f"Content {i}")
            
            request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
            result = await service.share(request, identity.get_did())
            share_ids.append(result.share_id)
        
        # Revoke one of them
        revoke_request = RevokeRequest(share_id=share_ids[0])
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Only 1 should be pending
        pending = await service.list_pending_shares(recipient_did)
        assert len(pending) == 1
        assert pending[0].id == share_ids[1]
    
    @pytest.mark.asyncio
    async def test_list_pending_empty_for_other_recipient(self, service, db, identity):
        """Test that pending list is empty for unrelated recipient."""
        recipient_did = "did:key:actual-recipient"
        identity.add_identity(recipient_did)
        
        belief_id = "belief-other-recipient"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        await service.share(request, identity.get_did())
        
        # Different recipient should see no pending shares
        pending = await service.list_pending_shares("did:key:other-person")
        assert len(pending) == 0


class TestRevocationPropagation:
    """Tests for revocation propagation to recipients (Issue #55)."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        return SharingService(db, identity)
    
    async def _create_share(
        self, service, db, identity, 
        belief_id="belief-prop-001",
        recipient_did="did:key:recipient-prop",
        content="Content to propagate revocation"
    ):
        """Helper to create a share for testing."""
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=content)
        identity.add_identity(recipient_did)
        
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        return await service.share(request, identity.get_did()), recipient_did
    
    @pytest.mark.asyncio
    async def test_revocation_creates_notification(self, service, db, identity):
        """Test that revoking a share creates a notification for the recipient."""
        share_result, recipient_did = await self._create_share(service, db, identity)
        
        # Revoke the share
        revoke_request = RevokeRequest(
            share_id=share_result.share_id,
            reason="Testing notification creation",
        )
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Check notification was created
        assert len(db.notifications) == 1
        
        notification = list(db.notifications.values())[0]
        assert notification["recipient_did"] == recipient_did
        assert notification["notification_type"] == "revocation"
        assert notification["payload"]["consent_chain_id"] == share_result.consent_chain_id
        assert notification["payload"]["belief_id"] == "belief-prop-001"
        assert notification["payload"]["revoked_by"] == identity.get_did()
        assert notification["payload"]["reason"] == "Testing notification creation"
    
    @pytest.mark.asyncio
    async def test_revocation_notification_in_memory_queue(self, service, db, identity):
        """Test that revocation notification is added to in-memory queue."""
        share_result, recipient_did = await self._create_share(
            service, db, identity,
            belief_id="belief-queue",
            recipient_did="did:key:queue-recipient"
        )
        
        # Verify queue is empty initially
        assert len(service._revocation_queue) == 0
        
        # Revoke the share
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Check notification is in queue
        assert len(service._revocation_queue) == 1
        queued_recipient, queued_notification = service._revocation_queue[0]
        assert queued_recipient == recipient_did
        assert queued_notification.consent_chain_id == share_result.consent_chain_id
    
    @pytest.mark.asyncio
    async def test_process_revocation_queue(self, service, db, identity):
        """Test processing the revocation queue."""
        share_result, recipient_did = await self._create_share(
            service, db, identity,
            belief_id="belief-process",
            recipient_did="did:key:process-recipient"
        )
        
        # Revoke to add to queue
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Process the queue
        processed = []
        async for recipient, notification in service.process_revocation_queue():
            processed.append((recipient, notification))
        
        assert len(processed) == 1
        assert processed[0][0] == recipient_did
        assert processed[0][1].consent_chain_id == share_result.consent_chain_id
        
        # Queue should be empty after processing
        assert len(service._revocation_queue) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_shares_same_consent_chain(self, service, db, identity):
        """Test that all recipients in a consent chain are notified."""
        # For DIRECT shares, there's only one recipient per consent chain
        # This test verifies the mechanism works correctly
        recipient_did = "did:key:multi-recipient"
        identity.add_identity(recipient_did)
        
        belief_id = "belief-multi"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Multi content")
        
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        share_result = await service.share(request, identity.get_did())
        
        # Revoke
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Verify notification was created
        notifications = await service.get_pending_notifications(recipient_did)
        assert len(notifications) == 1
        assert notifications[0].notification_type == "revocation"
    
    @pytest.mark.asyncio
    async def test_revocation_propagation_with_hops(self, service, db, identity):
        """Test that revocation propagates to hop recipients."""
        recipient_did = "did:key:hop-notify-recipient"
        identity.add_identity(recipient_did)
        
        belief_id = "belief-hop-notify"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Hop content")
        
        # Create share
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        share_result = await service.share(request, identity.get_did())
        
        # Receive the share (creates a hop)
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        await service.receive(receive_request, recipient_did)
        
        # Manually add a reshared_to hop for testing BOUNDED/CASCADING scenario
        downstream_did = "did:key:downstream"
        db.consent_chains[share_result.consent_chain_id]["hops"].append({
            "recipient": "did:key:intermediate",
            "reshared_to": [downstream_did],
        })
        
        # Revoke
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Should have notifications for:
        # 1. Original recipient
        # 2. Intermediate hop recipient (already notified via first hop)
        # 3. Downstream recipient
        # Note: recipient_did is notified once (deduplication)
        assert len(db.notifications) >= 2  # At least recipient + downstream
        
        # Verify downstream got notification
        downstream_notifications = [
            n for n in db.notifications.values()
            if n["recipient_did"] == downstream_did
        ]
        assert len(downstream_notifications) == 1
    
    @pytest.mark.asyncio
    async def test_no_duplicate_notifications(self, service, db, identity):
        """Test that recipients don't get duplicate notifications."""
        recipient_did = "did:key:no-dup-recipient"
        identity.add_identity(recipient_did)
        
        belief_id = "belief-no-dup"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="No dup content")
        
        # Create share
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        share_result = await service.share(request, identity.get_did())
        
        # Receive (creates hop with same recipient)
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        await service.receive(receive_request, recipient_did)
        
        # Revoke
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Should only have 1 notification despite appearing in shares and hops
        notifications_for_recipient = [
            n for n in db.notifications.values()
            if n["recipient_did"] == recipient_did
        ]
        assert len(notifications_for_recipient) == 1


class TestNotificationHandling:
    """Tests for notification retrieval and acknowledgment."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        return SharingService(db, identity)
    
    async def _create_and_revoke_share(
        self, service, db, identity, 
        belief_id="belief-notify-001",
        recipient_did="did:key:notify-recipient",
    ):
        """Helper to create and revoke a share."""
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        identity.add_identity(recipient_did)
        
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        share_result = await service.share(request, identity.get_did())
        
        revoke_request = RevokeRequest(share_id=share_result.share_id, reason="Test")
        await service.revoke_share(revoke_request, identity.get_did())
        
        return share_result, recipient_did
    
    @pytest.mark.asyncio
    async def test_get_pending_notifications(self, service, db, identity):
        """Test retrieving pending notifications for a recipient."""
        share_result, recipient_did = await self._create_and_revoke_share(
            service, db, identity
        )
        
        # Get pending notifications
        notifications = await service.get_pending_notifications(recipient_did)
        
        assert len(notifications) == 1
        assert notifications[0].notification_type == "revocation"
        assert notifications[0].recipient_did == recipient_did
        assert notifications[0].acknowledged_at is None
    
    @pytest.mark.asyncio
    async def test_acknowledge_notification(self, service, db, identity):
        """Test acknowledging a notification."""
        share_result, recipient_did = await self._create_and_revoke_share(
            service, db, identity,
            belief_id="belief-ack",
            recipient_did="did:key:ack-recipient"
        )
        
        # Get notification ID
        notifications = await service.get_pending_notifications(recipient_did)
        notification_id = notifications[0].id
        
        # Acknowledge it
        success = await service.acknowledge_notification(notification_id, recipient_did)
        assert success is True
        
        # Should no longer be in pending
        pending = await service.get_pending_notifications(recipient_did)
        assert len(pending) == 0
    
    @pytest.mark.asyncio
    async def test_acknowledge_notification_not_found(self, service, db, identity):
        """Test acknowledging a non-existent notification."""
        success = await service.acknowledge_notification(
            "nonexistent-notification-id",
            "did:key:anyone"
        )
        assert success is False
    
    @pytest.mark.asyncio
    async def test_acknowledge_notification_wrong_recipient(self, service, db, identity):
        """Test that wrong recipient cannot acknowledge notification."""
        share_result, recipient_did = await self._create_and_revoke_share(
            service, db, identity,
            belief_id="belief-wrong-ack",
            recipient_did="did:key:correct-recipient"
        )
        
        # Get notification ID
        notifications = await service.get_pending_notifications(recipient_did)
        notification_id = notifications[0].id
        
        # Try to acknowledge as wrong recipient
        with pytest.raises(PermissionError, match="Not the notification recipient"):
            await service.acknowledge_notification(notification_id, "did:key:imposter")
    
    @pytest.mark.asyncio
    async def test_acknowledge_already_acknowledged(self, service, db, identity):
        """Test acknowledging an already acknowledged notification."""
        share_result, recipient_did = await self._create_and_revoke_share(
            service, db, identity,
            belief_id="belief-double-ack",
            recipient_did="did:key:double-ack-recipient"
        )
        
        # Get notification ID
        notifications = await service.get_pending_notifications(recipient_did)
        notification_id = notifications[0].id
        
        # Acknowledge twice
        success1 = await service.acknowledge_notification(notification_id, recipient_did)
        success2 = await service.acknowledge_notification(notification_id, recipient_did)
        
        assert success1 is True
        assert success2 is True  # Idempotent
    
    @pytest.mark.asyncio
    async def test_multiple_notifications_for_recipient(self, service, db, identity):
        """Test handling multiple notifications for the same recipient."""
        recipient_did = "did:key:multi-notify-recipient"
        identity.add_identity(recipient_did)
        
        # Create and revoke multiple shares
        share_ids = []
        for i in range(3):
            belief_id = f"belief-multi-notify-{i}"
            db.beliefs[belief_id] = MockBelief(id=belief_id, content=f"Content {i}")
            
            request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
            share_result = await service.share(request, identity.get_did())
            share_ids.append(share_result.share_id)
            
            revoke_request = RevokeRequest(share_id=share_result.share_id)
            await service.revoke_share(revoke_request, identity.get_did())
        
        # Should have 3 notifications
        notifications = await service.get_pending_notifications(recipient_did)
        assert len(notifications) == 3
        
        # Acknowledge one
        await service.acknowledge_notification(notifications[0].id, recipient_did)
        
        # Should have 2 remaining
        pending = await service.get_pending_notifications(recipient_did)
        assert len(pending) == 2
    
    @pytest.mark.asyncio
    async def test_notifications_for_different_recipients_isolated(self, service, db, identity):
        """Test that notifications are isolated per recipient."""
        identity.add_identity("did:key:alice-notify")
        identity.add_identity("did:key:bob-notify")
        
        # Create and revoke shares for both
        for i, recipient in enumerate(["did:key:alice-notify", "did:key:bob-notify"]):
            belief_id = f"belief-isolated-{i}"
            db.beliefs[belief_id] = MockBelief(id=belief_id, content=f"Content {i}")
            
            request = ShareRequest(belief_id=belief_id, recipient_did=recipient)
            share_result = await service.share(request, identity.get_did())
            
            revoke_request = RevokeRequest(share_id=share_result.share_id)
            await service.revoke_share(revoke_request, identity.get_did())
        
        # Each should only see their own notifications
        alice_notifications = await service.get_pending_notifications("did:key:alice-notify")
        bob_notifications = await service.get_pending_notifications("did:key:bob-notify")
        
        assert len(alice_notifications) == 1
        assert len(bob_notifications) == 1
        
        # They should be different notifications
        assert alice_notifications[0].id != bob_notifications[0].id


class TestOfflineRecipientHandling:
    """Tests for handling offline recipients via notification persistence."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        return SharingService(db, identity)
    
    @pytest.mark.asyncio
    async def test_notifications_persist_for_offline_recipients(self, service, db, identity):
        """Test that notifications are stored in DB for later retrieval."""
        recipient_did = "did:key:offline-recipient"
        identity.add_identity(recipient_did)
        
        belief_id = "belief-offline"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Offline content")
        
        # Create and revoke share
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        share_result = await service.share(request, identity.get_did())
        
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Clear in-memory queue (simulating restart)
        service._revocation_queue.clear()
        
        # Notification should still be in database
        notifications = await service.get_pending_notifications(recipient_did)
        assert len(notifications) == 1
        assert notifications[0].notification_type == "revocation"
    
    @pytest.mark.asyncio
    async def test_recipient_can_poll_notifications(self, service, db, identity):
        """Test that recipient can poll for notifications after coming online."""
        recipient_did = "did:key:polling-recipient"
        identity.add_identity(recipient_did)
        
        # Create multiple revocations while "offline"
        for i in range(3):
            belief_id = f"belief-poll-{i}"
            db.beliefs[belief_id] = MockBelief(id=belief_id, content=f"Poll content {i}")
            
            request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
            share_result = await service.share(request, identity.get_did())
            
            revoke_request = RevokeRequest(share_id=share_result.share_id)
            await service.revoke_share(revoke_request, identity.get_did())
        
        # Clear in-memory queue (simulating recipient was offline)
        service._revocation_queue.clear()
        
        # "Come online" and poll
        notifications = await service.get_pending_notifications(recipient_did)
        assert len(notifications) == 3
        
        # Acknowledge all
        for notification in notifications:
            await service.acknowledge_notification(notification.id, recipient_did)
        
        # Should be empty now
        pending = await service.get_pending_notifications(recipient_did)
        assert len(pending) == 0
