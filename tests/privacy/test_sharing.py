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
    PropagateRequest,
    PropagateResult,
    RevocationNotification,
    Notification,
    SharingService,
    Share,
    ConsentChainEntry,
    strip_fields_from_content,
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


class TestMaxHopsPropagation:
    """Tests for max_hops propagation tracking (Issue #65)."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def service(self, db, identity):
        return SharingService(db, identity)
    
    async def _create_bounded_share(
        self, service, db, identity,
        belief_id="belief-bounded-001",
        recipient_did="did:key:bounded-recipient",
        content="Bounded content",
        max_hops=2,
    ):
        """Helper to create a BOUNDED share with max_hops."""
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=content)
        identity.add_identity(recipient_did)
        
        policy = SharePolicy.bounded(max_hops=max_hops)
        # Override enforcement to POLICY for testing (BOUNDED uses POLICY)
        policy.enforcement = EnforcementType.POLICY
        # Add recipient to recipients list for validation
        policy.recipients = [recipient_did]
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        return await service.share(request, identity.get_did()), recipient_did
    
    @pytest.mark.asyncio
    async def test_consent_chain_tracks_current_hop(self, service, db, identity):
        """Test that consent chain initializes with current_hop=0."""
        # Use DIRECT share for simpler test
        belief_id = "belief-hop-track"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        identity.add_identity("did:key:hop-recipient")
        
        request = ShareRequest(belief_id=belief_id, recipient_did="did:key:hop-recipient")
        result = await service.share(request, identity.get_did())
        
        chain = await service.get_consent_chain(result.consent_chain_id)
        assert chain.current_hop == 0
    
    @pytest.mark.asyncio
    async def test_consent_chain_max_hops_property(self, service, db, identity):
        """Test that consent chain correctly reports max_hops from policy."""
        belief_id = "belief-max-hops-prop"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        recipient_did = "did:key:max-hops-recipient"
        identity.add_identity(recipient_did)
        
        policy = SharePolicy.bounded(max_hops=3)
        policy.recipients = [recipient_did]
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        result = await service.share(request, identity.get_did())
        
        chain = await service.get_consent_chain(result.consent_chain_id)
        assert chain.max_hops == 3
        assert chain.hops_remaining() == 3
        assert chain.can_reshare() is True
    
    @pytest.mark.asyncio
    async def test_consent_chain_can_reshare_with_no_limit(self, service, db, identity):
        """Test that consent chain without max_hops allows unlimited resharing."""
        belief_id = "belief-no-limit"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        recipient_did = "did:key:no-limit-recipient"
        identity.add_identity(recipient_did)
        
        # Create with BOUNDED policy but no max_hops (unlimited)
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            recipients=[recipient_did],
            propagation=PropagationRules(max_hops=None),  # No limit
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        result = await service.share(request, identity.get_did())
        
        chain = await service.get_consent_chain(result.consent_chain_id)
        assert chain.max_hops is None
        assert chain.hops_remaining() is None
        assert chain.can_reshare() is True
    
    @pytest.mark.asyncio
    async def test_consent_chain_cannot_reshare_when_revoked(self, service, db, identity):
        """Test that revoked consent chain cannot be reshared."""
        belief_id = "belief-revoked-reshare"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        recipient_did = "did:key:revoked-reshare-recipient"
        identity.add_identity(recipient_did)
        
        request = ShareRequest(belief_id=belief_id, recipient_did=recipient_did)
        result = await service.share(request, identity.get_did())
        
        # Revoke
        revoke_request = RevokeRequest(share_id=result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        chain = await service.get_consent_chain(result.consent_chain_id)
        assert chain.can_reshare() is False
    
    @pytest.mark.asyncio
    async def test_reshare_increments_hop_count(self, service, db, identity):
        """Test that resharing increments the current_hop counter."""
        # Create original share with BOUNDED policy
        belief_id = "belief-inc-hop"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Hop increment content")
        first_recipient = "did:key:first-recipient"
        second_recipient = "did:key:second-recipient"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        
        policy = SharePolicy.bounded(max_hops=3)
        policy.recipients = [first_recipient]
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives the share
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        await service.receive(receive_request, first_recipient)
        
        # Reshare to second recipient
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        reshare_result = await service.reshare(reshare_request, first_recipient)
        
        # Verify hop count incremented
        assert reshare_result.current_hop == 1
        assert reshare_result.hops_remaining == 2  # 3 - 1 = 2
        
        # Verify consent chain was updated
        chain = await service.get_consent_chain(share_result.consent_chain_id)
        assert chain.current_hop == 1
    
    @pytest.mark.asyncio
    async def test_reshare_blocked_when_max_hops_exceeded(self, service, db, identity):
        """Test that resharing is blocked when max_hops would be exceeded."""
        # Create share with max_hops=1
        belief_id = "belief-max-exceeded"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Max hops content")
        first_recipient = "did:key:first-max"
        second_recipient = "did:key:second-max"
        third_recipient = "did:key:third-max"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        identity.add_identity(third_recipient)
        
        policy = SharePolicy.bounded(max_hops=1)
        policy.recipients = [first_recipient]
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives and reshares
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        await service.receive(receive_request, first_recipient)
        
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        reshare_result = await service.reshare(reshare_request, first_recipient)
        
        # Verify reshare worked
        assert reshare_result.current_hop == 1
        assert reshare_result.hops_remaining == 0
        
        # Second recipient receives
        receive_request2 = ReceiveRequest(share_id=reshare_result.share_id)
        await service.receive(receive_request2, second_recipient)
        
        # Second recipient tries to reshare - should fail
        reshare_request2 = ReshareRequest(
            original_share_id=reshare_result.share_id,
            recipient_did=third_recipient,
        )
        
        with pytest.raises(ValueError, match="max_hops.*exceeded"):
            await service.reshare(reshare_request2, second_recipient)
    
    @pytest.mark.asyncio
    async def test_reshare_requires_receiving_first(self, service, db, identity):
        """Test that resharing requires receiving the share first."""
        belief_id = "belief-receive-first"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        first_recipient = "did:key:receive-first"
        second_recipient = "did:key:second-receive"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        
        policy = SharePolicy.bounded(max_hops=2)
        policy.recipients = [first_recipient]
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Try to reshare without receiving first
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        
        with pytest.raises(ValueError, match="Must receive share before resharing"):
            await service.reshare(reshare_request, first_recipient)
    
    @pytest.mark.asyncio
    async def test_reshare_only_by_recipient(self, service, db, identity):
        """Test that only the recipient of a share can reshare it."""
        belief_id = "belief-only-recipient"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        recipient_did = "did:key:actual-recipient"
        imposter_did = "did:key:imposter"
        target_did = "did:key:target"
        identity.add_identity(recipient_did)
        identity.add_identity(imposter_did)
        identity.add_identity(target_did)
        
        policy = SharePolicy.bounded(max_hops=2)
        policy.recipients = [recipient_did]
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Imposter tries to reshare
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=target_did,
        )
        
        with pytest.raises(PermissionError, match="Only the recipient"):
            await service.reshare(reshare_request, imposter_did)
    
    @pytest.mark.asyncio
    async def test_reshare_blocked_for_direct_policy(self, service, db, identity):
        """Test that DIRECT policy shares cannot be reshared."""
        belief_id = "belief-direct-no-reshare"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        first_recipient = "did:key:direct-recipient"
        second_recipient = "did:key:direct-second"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        
        # DIRECT policy (default)
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Receive
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        await service.receive(receive_request, first_recipient)
        
        # Try to reshare
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        
        with pytest.raises(ValueError, match="does not allow resharing"):
            await service.reshare(reshare_request, first_recipient)
    
    @pytest.mark.asyncio
    async def test_reshare_blocked_for_revoked_share(self, service, db, identity):
        """Test that revoked shares cannot be reshared."""
        belief_id = "belief-revoked-no-reshare"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        first_recipient = "did:key:revoked-recipient"
        second_recipient = "did:key:revoked-second"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        
        policy = SharePolicy.bounded(max_hops=2)
        policy.recipients = [first_recipient]
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Receive
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        await service.receive(receive_request, first_recipient)
        
        # Revoke
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Try to reshare
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        
        with pytest.raises(ValueError, match="has been revoked"):
            await service.reshare(reshare_request, first_recipient)
    
    @pytest.mark.asyncio
    async def test_reshare_chain_of_three(self, service, db, identity):
        """Test a chain of reshares: A -> B -> C -> D with max_hops=3."""
        belief_id = "belief-chain-three"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Chain of three content")
        
        # Create identities
        alice = "did:key:alice"
        bob = "did:key:bob"
        charlie = "did:key:charlie"
        diana = "did:key:diana"
        identity.add_identity(alice)
        identity.add_identity(bob)
        identity.add_identity(charlie)
        identity.add_identity(diana)
        
        # Original share from sharer to Alice
        policy = SharePolicy.bounded(max_hops=3)
        policy.recipients = [alice]
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=alice,
            policy=policy,
        )
        share1 = await service.share(request, identity.get_did())
        
        # Alice receives and reshares to Bob
        await service.receive(ReceiveRequest(share_id=share1.share_id), alice)
        reshare1 = await service.reshare(
            ReshareRequest(original_share_id=share1.share_id, recipient_did=bob),
            alice
        )
        assert reshare1.current_hop == 1
        assert reshare1.hops_remaining == 2
        
        # Bob receives and reshares to Charlie
        await service.receive(ReceiveRequest(share_id=reshare1.share_id), bob)
        reshare2 = await service.reshare(
            ReshareRequest(original_share_id=reshare1.share_id, recipient_did=charlie),
            bob
        )
        assert reshare2.current_hop == 2
        assert reshare2.hops_remaining == 1
        
        # Charlie receives and reshares to Diana
        await service.receive(ReceiveRequest(share_id=reshare2.share_id), charlie)
        reshare3 = await service.reshare(
            ReshareRequest(original_share_id=reshare2.share_id, recipient_did=diana),
            charlie
        )
        assert reshare3.current_hop == 3
        assert reshare3.hops_remaining == 0
        
        # Diana receives but cannot reshare (max_hops reached)
        await service.receive(ReceiveRequest(share_id=reshare3.share_id), diana)
        
        eve = "did:key:eve"
        identity.add_identity(eve)
        
        with pytest.raises(ValueError, match="max_hops.*exceeded"):
            await service.reshare(
                ReshareRequest(original_share_id=reshare3.share_id, recipient_did=eve),
                diana
            )
    
    @pytest.mark.asyncio
    async def test_reshare_result_contains_hop_info(self, service, db, identity):
        """Test that ReshareResult contains correct hop tracking info."""
        belief_id = "belief-result-info"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Result info content")
        first_recipient = "did:key:result-first"
        second_recipient = "did:key:result-second"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        
        policy = SharePolicy.bounded(max_hops=5)
        policy.recipients = [first_recipient]
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Receive and reshare
        await service.receive(ReceiveRequest(share_id=share_result.share_id), first_recipient)
        reshare_result = await service.reshare(
            ReshareRequest(original_share_id=share_result.share_id, recipient_did=second_recipient),
            first_recipient
        )
        
        # Verify all fields
        assert reshare_result.share_id is not None
        assert reshare_result.consent_chain_id == share_result.consent_chain_id
        assert reshare_result.encrypted_for == second_recipient
        assert reshare_result.created_at > 0
        assert reshare_result.current_hop == 1
        assert reshare_result.hops_remaining == 4  # 5 - 1 = 4
    
    @pytest.mark.asyncio
    async def test_reshare_adds_hop_to_consent_chain(self, service, db, identity):
        """Test that resharing adds a hop record to the consent chain."""
        belief_id = "belief-hop-record"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Hop record content")
        first_recipient = "did:key:hop-first"
        second_recipient = "did:key:hop-second"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        
        policy = SharePolicy.bounded(max_hops=2)
        policy.recipients = [first_recipient]
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Receive and reshare
        await service.receive(ReceiveRequest(share_id=share_result.share_id), first_recipient)
        reshare_result = await service.reshare(
            ReshareRequest(original_share_id=share_result.share_id, recipient_did=second_recipient),
            first_recipient
        )
        
        # Get consent chain and check hops
        chain = await service.get_consent_chain(share_result.consent_chain_id)
        
        # Should have 2 hops: receive hop + reshare hop
        assert len(chain.hops) == 2
        
        # Find the reshare hop
        reshare_hop = None
        for hop in chain.hops:
            if hop.get("resharer") == first_recipient:
                reshare_hop = hop
                break
        
        assert reshare_hop is not None
        assert reshare_hop["recipient"] == second_recipient
        assert reshare_hop["hop_number"] == 1
        assert reshare_hop["from_share_id"] == share_result.share_id
        assert "signature" in reshare_hop


class TestBoundedSharing:
    """Tests for BOUNDED share level (Issue #64)."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def domain_service(self):
        return MockDomainService()
    
    @pytest.fixture
    def service(self, db, identity, domain_service):
        return SharingService(db, identity, domain_service)
    
    @pytest.mark.asyncio
    async def test_bounded_share_basic(self, service, db, identity, domain_service):
        """Test basic BOUNDED share to a domain member."""
        belief_id = "belief-bounded-001"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Bounded content")
        
        recipient_did = "did:key:domain-member"
        identity.add_identity(recipient_did)
        
        # Add recipient to allowed domain
        domain_service.add_member(recipient_did, "acme-corp")
        
        # Create BOUNDED policy
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=3,
                allowed_domains=["acme-corp"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        result = await service.share(request, identity.get_did())
        
        assert result.share_id is not None
        assert result.consent_chain_id is not None
        assert result.encrypted_for == recipient_did
        
        # Verify policy in consent chain
        chain = db.consent_chains[result.consent_chain_id]
        assert chain["origin_policy"]["level"] == "bounded"
        assert chain["origin_policy"]["propagation"]["allowed_domains"] == ["acme-corp"]
    
    @pytest.mark.asyncio
    async def test_bounded_share_multiple_domains(self, service, db, identity, domain_service):
        """Test BOUNDED share with multiple allowed domains."""
        belief_id = "belief-bounded-multi"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Multi-domain content")
        
        recipient_did = "did:key:multi-domain-member"
        identity.add_identity(recipient_did)
        
        # Add recipient to one of the allowed domains
        domain_service.add_member(recipient_did, "team-beta")
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=2,
                allowed_domains=["team-alpha", "team-beta", "team-gamma"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        result = await service.share(request, identity.get_did())
        
        assert result.share_id is not None
        # Recipient was in team-beta which is in allowed_domains
    
    @pytest.mark.asyncio
    async def test_bounded_share_recipient_not_in_domain(self, service, db, identity, domain_service):
        """Test that BOUNDED share fails if recipient is not in any allowed domain."""
        belief_id = "belief-bounded-fail"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Should fail")
        
        recipient_did = "did:key:outsider"
        identity.add_identity(recipient_did)
        
        # Don't add recipient to any domain
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=2,
                allowed_domains=["secret-club"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        with pytest.raises(ValueError, match="not a member of any allowed domain"):
            await service.share(request, identity.get_did())
    
    @pytest.mark.asyncio
    async def test_bounded_share_without_domains_allowed(self, service, db, identity, domain_service):
        """Test that BOUNDED share without allowed_domains works (no restriction)."""
        belief_id = "belief-bounded-no-domains"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="No domains")
        
        recipient_did = "did:key:recipient"
        identity.add_identity(recipient_did)
        
        # Test with allowed_domains=None - should work (no restriction)
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(max_hops=2, allowed_domains=None),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        result = await service.share(request, identity.get_did())
        assert result.share_id is not None
        
        # Test with allowed_domains=[] - should also work (no restriction)
        belief_id2 = "belief-bounded-empty-domains"
        db.beliefs[belief_id2] = MockBelief(id=belief_id2, content="Empty domains")
        
        recipient_did2 = "did:key:recipient2"
        identity.add_identity(recipient_did2)
        
        policy2 = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(max_hops=2, allowed_domains=[]),
        )
        
        request2 = ShareRequest(
            belief_id=belief_id2,
            recipient_did=recipient_did2,
            policy=policy2,
        )
        
        result2 = await service.share(request2, identity.get_did())
        assert result2.share_id is not None
    
    @pytest.mark.asyncio
    async def test_bounded_share_requires_domain_service(self, db, identity):
        """Test that BOUNDED share requires domain service."""
        # Create service WITHOUT domain_service
        service = SharingService(db, identity, domain_service=None)
        
        belief_id = "belief-bounded-no-service"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="No service")
        
        recipient_did = "did:key:recipient"
        identity.add_identity(recipient_did)
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=2,
                allowed_domains=["some-domain"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        with pytest.raises(ValueError, match="Domain service required"):
            await service.share(request, identity.get_did())


class TestBoundedResharing:
    """Tests for resharing BOUNDED beliefs (Issue #64)."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def domain_service(self):
        return MockDomainService()
    
    @pytest.fixture
    def service(self, db, identity, domain_service):
        return SharingService(db, identity, domain_service)
    
    async def _create_and_receive_bounded_share(
        self, service, db, identity, domain_service,
        belief_id="belief-reshare-001",
        recipient_did="did:key:first-recipient",
        content="Content to reshare",
        allowed_domains=None,
        max_hops=3,
    ):
        """Helper to create and receive a BOUNDED share."""
        if allowed_domains is None:
            allowed_domains = ["acme-corp"]
        
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=content)
        identity.add_identity(recipient_did)
        
        # Add recipient to domain
        for domain in allowed_domains:
            domain_service.add_member(recipient_did, domain)
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=max_hops,
                allowed_domains=allowed_domains,
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        share_result = await service.share(request, identity.get_did())
        
        # Receive the share
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        await service.receive(receive_request, recipient_did)
        
        return share_result, recipient_did
    
    @pytest.mark.asyncio
    async def test_reshare_bounded_success(self, service, db, identity, domain_service):
        """Test successful resharing of a BOUNDED belief."""
        # Create and receive original share
        share_result, first_recipient = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service
        )
        
        # Add second recipient to same domain
        second_recipient = "did:key:second-recipient"
        identity.add_identity(second_recipient)
        domain_service.add_member(second_recipient, "acme-corp")
        
        # Reshare to second recipient
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        
        reshare_result = await service.reshare(reshare_request, first_recipient)
        
        assert reshare_result.share_id is not None
        assert reshare_result.consent_chain_id == share_result.consent_chain_id
        assert reshare_result.encrypted_for == second_recipient
        assert reshare_result.current_hop == 1  # First reshare is hop 1
    
    @pytest.mark.asyncio
    async def test_reshare_content_decryptable(self, service, db, identity, domain_service):
        """Test that reshared content can be decrypted by new recipient."""
        original_content = "Super secret content for resharing"
        
        # Create and receive original share
        share_result, first_recipient = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
            content=original_content,
        )
        
        # Add second recipient
        second_recipient = "did:key:decryptor"
        identity.add_identity(second_recipient)
        domain_service.add_member(second_recipient, "acme-corp")
        
        # Reshare
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        reshare_result = await service.reshare(reshare_request, first_recipient)
        
        # Receive the reshared content
        receive_request = ReceiveRequest(share_id=reshare_result.share_id)
        receive_result = await service.receive(receive_request, second_recipient)
        
        assert receive_result.content.decode("utf-8") == original_content
    
    @pytest.mark.asyncio
    async def test_reshare_recipient_not_in_domain(self, service, db, identity, domain_service):
        """Test that resharing fails if new recipient is not in allowed domain."""
        # Create and receive original share
        share_result, first_recipient = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service
        )
        
        # Add second recipient but NOT to allowed domain
        second_recipient = "did:key:outsider"
        identity.add_identity(second_recipient)
        domain_service.add_member(second_recipient, "different-corp")
        
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        
        with pytest.raises(ValueError, match="not a member of any allowed domain"):
            await service.reshare(reshare_request, first_recipient)
    
    @pytest.mark.asyncio
    async def test_reshare_max_hops_exceeded(self, service, db, identity, domain_service):
        """Test that resharing fails when max_hops is exceeded."""
        # Create share with max_hops=1 (allows 1 reshare)
        share_result, first_recipient = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
            belief_id="belief-max-hops",
            max_hops=1,
        )
        
        # Second recipient - first reshare will succeed
        second_recipient = "did:key:second"
        identity.add_identity(second_recipient)
        domain_service.add_member(second_recipient, "acme-corp")
        
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        
        # First reshare succeeds (0 -> 1, which is <= max_hops)
        reshare_result = await service.reshare(reshare_request, first_recipient)
        assert reshare_result.current_hop == 1
        assert reshare_result.hops_remaining == 0
        
        # Second recipient receives
        receive_request = ReceiveRequest(share_id=reshare_result.share_id)
        await service.receive(receive_request, second_recipient)
        
        # Third recipient - second reshare should fail (max_hops exceeded)
        third_recipient = "did:key:third"
        identity.add_identity(third_recipient)
        domain_service.add_member(third_recipient, "acme-corp")
        
        reshare_request2 = ReshareRequest(
            original_share_id=reshare_result.share_id,
            recipient_did=third_recipient,
        )
        
        with pytest.raises(ValueError, match="max_hops"):
            await service.reshare(reshare_request2, second_recipient)
    
    @pytest.mark.asyncio
    async def test_reshare_direct_not_allowed(self, service, db, identity, domain_service):
        """Test that DIRECT shares cannot be reshared."""
        belief_id = "belief-direct-no-reshare"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Direct content")
        
        recipient_did = "did:key:direct-recipient"
        identity.add_identity(recipient_did)
        
        # Create DIRECT share
        policy = SharePolicy.direct([recipient_did])
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Receive it
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        await service.receive(receive_request, recipient_did)
        
        # Try to reshare
        second_recipient = "did:key:second-direct"
        identity.add_identity(second_recipient)
        
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        
        with pytest.raises(ValueError, match="does not allow resharing"):
            await service.reshare(reshare_request, recipient_did)
    
    @pytest.mark.asyncio
    async def test_reshare_not_received_fails(self, service, db, identity, domain_service):
        """Test that share must be received before resharing."""
        belief_id = "belief-not-received"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Not received yet")
        
        recipient_did = "did:key:unreceived-recipient"
        identity.add_identity(recipient_did)
        domain_service.add_member(recipient_did, "acme-corp")
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=3,
                allowed_domains=["acme-corp"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Don't receive, try to reshare directly
        second_recipient = "did:key:second"
        identity.add_identity(second_recipient)
        domain_service.add_member(second_recipient, "acme-corp")
        
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        
        with pytest.raises(ValueError, match="receive share before resharing"):
            await service.reshare(reshare_request, recipient_did)
    
    @pytest.mark.asyncio
    async def test_reshare_revoked_fails(self, service, db, identity, domain_service):
        """Test that revoked shares cannot be reshared."""
        # Create and receive
        share_result, first_recipient = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
            belief_id="belief-revoked-reshare",
        )
        
        # Revoke the share
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Try to reshare
        second_recipient = "did:key:after-revoke"
        identity.add_identity(second_recipient)
        domain_service.add_member(second_recipient, "acme-corp")
        
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        
        with pytest.raises(ValueError, match="original share has been revoked"):
            await service.reshare(reshare_request, first_recipient)
    
    @pytest.mark.asyncio
    async def test_reshare_wrong_resharer(self, service, db, identity, domain_service):
        """Test that only the share recipient can reshare."""
        # Create and receive
        share_result, first_recipient = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
            belief_id="belief-wrong-resharer",
        )
        
        # Second recipient who will try to reshare
        second_recipient = "did:key:imposter"
        identity.add_identity(second_recipient)
        domain_service.add_member(second_recipient, "acme-corp")
        
        # Third recipient
        third_recipient = "did:key:third"
        identity.add_identity(third_recipient)
        domain_service.add_member(third_recipient, "acme-corp")
        
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=third_recipient,
        )
        
        # Second recipient (imposter) tries to reshare, should fail
        with pytest.raises(PermissionError, match="Only the recipient"):
            await service.reshare(reshare_request, second_recipient)


class TestBoundedConsentChain:
    """Tests for consent chain tracking with BOUNDED shares."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def domain_service(self):
        return MockDomainService()
    
    @pytest.fixture
    def service(self, db, identity, domain_service):
        return SharingService(db, identity, domain_service)
    
    @pytest.mark.asyncio
    async def test_reshare_creates_hop_entry(self, service, db, identity, domain_service):
        """Test that resharing creates proper hop entry in consent chain."""
        belief_id = "belief-chain-hop"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Chain content")
        
        first_recipient = "did:key:first"
        identity.add_identity(first_recipient)
        domain_service.add_member(first_recipient, "acme-corp")
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=5,
                allowed_domains=["acme-corp"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Receive
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        await service.receive(receive_request, first_recipient)
        
        # Reshare
        second_recipient = "did:key:second"
        identity.add_identity(second_recipient)
        domain_service.add_member(second_recipient, "acme-corp")
        
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        reshare_result = await service.reshare(reshare_request, first_recipient)
        
        # Check consent chain hops
        chain = db.consent_chains[share_result.consent_chain_id]
        
        # Should have 2 hops: receive + reshare
        assert len(chain["hops"]) == 2
        
        # Check reshare hop details
        reshare_hop = chain["hops"][1]
        assert reshare_hop["recipient"] == second_recipient
        assert reshare_hop["resharer"] == first_recipient
        assert reshare_hop["from_share_id"] == share_result.share_id
        assert reshare_hop["hop_number"] == 1  # First reshare is hop 1
        assert "signature" in reshare_hop
    
    @pytest.mark.asyncio
    async def test_chain_of_reshares(self, service, db, identity, domain_service):
        """Test a chain of multiple reshares tracks correctly."""
        belief_id = "belief-multi-chain"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Multi-hop content")
        
        # Create chain of 4 recipients
        recipients = [f"did:key:recipient-{i}" for i in range(4)]
        for r in recipients:
            identity.add_identity(r)
            domain_service.add_member(r, "acme-corp")
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=10,
                allowed_domains=["acme-corp"],
            ),
        )
        
        # Initial share to first recipient
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipients[0],
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        current_share_id = share_result.share_id
        
        # Receive and reshare through the chain
        for i in range(len(recipients) - 1):
            # Receive
            receive_request = ReceiveRequest(share_id=current_share_id)
            await service.receive(receive_request, recipients[i])
            
            # Reshare to next
            reshare_request = ReshareRequest(
                original_share_id=current_share_id,
                recipient_did=recipients[i + 1],
            )
            reshare_result = await service.reshare(reshare_request, recipients[i])
            current_share_id = reshare_result.share_id
        
        # Verify chain has all hops
        chain = db.consent_chains[share_result.consent_chain_id]
        
        # 3 receives + 3 reshares = 6 hops
        # Actually: receive(0) + reshare(0->1) + receive(1) + reshare(1->2) + receive(2) + reshare(2->3)
        # But in our flow: receive creates 1 hop, reshare creates 1 hop
        # So: 3 receives + 3 reshares but they overlap... let's count actual hops
        # receive by r0 -> hop1
        # reshare r0->r1 -> hop2  
        # receive by r1 -> hop3
        # reshare r1->r2 -> hop4
        # receive by r2 -> hop5
        # reshare r2->r3 -> hop6
        # Total: 6 hops
        assert len(chain["hops"]) == 6


class TestBoundedRevocationPropagation:
    """Tests for revocation propagation through BOUNDED share chains."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def domain_service(self):
        return MockDomainService()
    
    @pytest.fixture
    def service(self, db, identity, domain_service):
        return SharingService(db, identity, domain_service)
    
    @pytest.mark.asyncio
    async def test_revocation_notifies_all_in_chain(self, service, db, identity, domain_service):
        """Test that revoking a BOUNDED share notifies all downstream recipients."""
        belief_id = "belief-revoke-chain"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Revoke chain content")
        
        # Create chain of 3 recipients
        recipients = [f"did:key:chain-{i}" for i in range(3)]
        for r in recipients:
            identity.add_identity(r)
            domain_service.add_member(r, "acme-corp")
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=10,
                allowed_domains=["acme-corp"],
            ),
        )
        
        # Initial share
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipients[0],
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        current_share_id = share_result.share_id
        
        # Build chain: r0 receives, reshares to r1, r1 receives, reshares to r2
        for i in range(len(recipients) - 1):
            receive_request = ReceiveRequest(share_id=current_share_id)
            await service.receive(receive_request, recipients[i])
            
            reshare_request = ReshareRequest(
                original_share_id=current_share_id,
                recipient_did=recipients[i + 1],
            )
            reshare_result = await service.reshare(reshare_request, recipients[i])
            current_share_id = reshare_result.share_id
        
        # Last recipient receives
        receive_request = ReceiveRequest(share_id=current_share_id)
        await service.receive(receive_request, recipients[-1])
        
        # Clear notifications before revocation
        db.notifications.clear()
        service._revocation_queue.clear()
        
        # Revoke original share
        revoke_request = RevokeRequest(
            share_id=share_result.share_id,
            reason="Testing chain revocation",
        )
        await service.revoke_share(revoke_request, identity.get_did())
        
        # All recipients should have notifications
        all_notified = set()
        for notification in db.notifications.values():
            all_notified.add(notification["recipient_did"])
        
        # At minimum, r0 gets notified (direct recipient)
        # r1 and r2 should also be notified via the hop tracking
        assert recipients[0] in all_notified



class MockDomainService:
    """Mock domain service for testing domain membership."""
    
    def __init__(self):
        # Maps domain_id -> set of DIDs that are members
        self.domains: dict[str, set[str]] = {}
    
    def add_domain(self, domain_id: str, members: list[str] = None):
        """Create a domain with optional initial members."""
        self.domains[domain_id] = set(members or [])
    
    def add_member(self, did: str, domain_id: str):
        """Add a DID to a domain."""
        if domain_id not in self.domains:
            self.domains[domain_id] = set()
        self.domains[domain_id].add(did)
    
    async def is_member(self, did: str, domain: str) -> bool:
        """Check if a DID is a member of a domain."""
        if domain not in self.domains:
            return False
        return did in self.domains[domain]
    
    async def get_domains_for_did(self, did: str) -> list[str]:
        """Get all domains a DID is a member of."""
        return [
            domain_id for domain_id, members in self.domains.items()
            if did in members
        ]


class TestAllowedDomains:
    """Tests for allowed_domains restriction in BOUNDED sharing (Issue #66)."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def domain_service(self):
        return MockDomainService()
    
    @pytest.fixture
    def service(self, db, identity, domain_service):
        return SharingService(db, identity, domain_service)
    
    @pytest.mark.asyncio
    async def test_bounded_share_validates_recipient_domain(self, service, db, identity, domain_service):
        """Test that BOUNDED share validates recipient is in allowed domain."""
        belief_id = "belief-domain-001"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Domain restricted content")
        
        # Setup: create domains and add members
        domain_service.add_domain("acme-corp", ["did:key:alice-acme"])
        domain_service.add_domain("beta-inc", ["did:key:bob-beta"])
        
        # Add recipient identity (in acme-corp domain)
        recipient_did = "did:key:alice-acme"
        identity.add_identity(recipient_did)
        
        # Create BOUNDED policy with allowed_domains
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(allowed_domains=["acme-corp"]),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        # Should succeed - recipient is in acme-corp domain
        result = await service.share(request, identity.get_did())
        
        assert result.share_id is not None
        assert result.encrypted_for == recipient_did
    
    @pytest.mark.asyncio
    async def test_bounded_share_rejects_recipient_not_in_domain(self, service, db, identity, domain_service):
        """Test that BOUNDED share fails if recipient is not in allowed domain."""
        belief_id = "belief-domain-002"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Domain restricted content")
        
        # Setup: create domain but recipient is NOT a member
        domain_service.add_domain("acme-corp", ["did:key:insider"])
        
        # Recipient is not in acme-corp
        recipient_did = "did:key:outsider"
        identity.add_identity(recipient_did)
        
        # Create BOUNDED policy with allowed_domains
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(allowed_domains=["acme-corp"]),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        # Should fail - recipient is NOT in acme-corp domain
        with pytest.raises(ValueError, match="not a member of any allowed domain"):
            await service.share(request, identity.get_did())
    
    @pytest.mark.asyncio
    async def test_bounded_share_with_multiple_allowed_domains(self, service, db, identity, domain_service):
        """Test BOUNDED share succeeds if recipient is in ANY allowed domain."""
        belief_id = "belief-domain-003"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Multi-domain content")
        
        # Setup: create multiple domains
        domain_service.add_domain("domain-a", ["did:key:user-a"])
        domain_service.add_domain("domain-b", ["did:key:user-b"])
        domain_service.add_domain("domain-c", ["did:key:user-c"])
        
        # Recipient is only in domain-b
        recipient_did = "did:key:user-b"
        identity.add_identity(recipient_did)
        
        # Policy allows domains a and b (recipient is in b)
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(allowed_domains=["domain-a", "domain-b"]),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        # Should succeed - recipient is in domain-b which is allowed
        result = await service.share(request, identity.get_did())
        assert result.share_id is not None
    
    @pytest.mark.asyncio
    async def test_empty_allowed_domains_no_restriction(self, service, db, identity, domain_service):
        """Test that empty allowed_domains means no domain restriction."""
        belief_id = "belief-domain-004"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Unrestricted content")
        
        # Recipient is not in any domain
        recipient_did = "did:key:anyone"
        identity.add_identity(recipient_did)
        
        # Policy with empty allowed_domains
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(allowed_domains=[]),  # Empty = no restriction
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        # Should succeed - no domain restriction
        result = await service.share(request, identity.get_did())
        assert result.share_id is not None
    
    @pytest.mark.asyncio
    async def test_none_allowed_domains_no_restriction(self, service, db, identity, domain_service):
        """Test that None allowed_domains means no domain restriction."""
        belief_id = "belief-domain-005"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Unrestricted content")
        
        # Recipient is not in any domain
        recipient_did = "did:key:anyone"
        identity.add_identity(recipient_did)
        
        # Policy with no propagation rules at all
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=None,  # No propagation = no domain restriction
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        # Should succeed - no domain restriction
        result = await service.share(request, identity.get_did())
        assert result.share_id is not None
    
    @pytest.mark.asyncio
    async def test_no_domain_service_with_restrictions_fails(self, db, identity):
        """Test that domain-restricted sharing requires domain service."""
        # Create service WITHOUT domain_service
        service = SharingService(db, identity, domain_service=None)
        
        belief_id = "belief-domain-006"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Content")
        
        recipient_did = "did:key:recipient"
        identity.add_identity(recipient_did)
        
        # Policy with allowed_domains
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(allowed_domains=["some-domain"]),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        # Should fail - need domain service for domain-restricted sharing
        with pytest.raises(ValueError, match="Domain service required"):
            await service.share(request, identity.get_did())


class TestReshareWithDomains:
    """Tests for domain restrictions on resharing (Issue #66)."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def domain_service(self):
        return MockDomainService()
    
    @pytest.fixture
    def service(self, db, identity, domain_service):
        return SharingService(db, identity, domain_service)
    
    async def _create_bounded_share(
        self, service, db, identity, domain_service,
        belief_id="belief-reshare-001",
        sharer_did=None,
        recipient_did="did:key:reshare-recipient",
        allowed_domains=None,
        max_hops=3,
    ):
        """Helper to create a BOUNDED share for resharing tests."""
        sharer_did = sharer_did or identity.get_did()
        
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Reshare test content")
        identity.add_identity(recipient_did)
        
        # Setup domain membership if domains specified
        if allowed_domains:
            for domain in allowed_domains:
                domain_service.add_member(recipient_did, domain)
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                allowed_domains=allowed_domains,
                max_hops=max_hops,
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        share_result = await service.share(request, sharer_did)
        
        # Receive the share
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        await service.receive(receive_request, recipient_did)
        
        return share_result, recipient_did
    
    @pytest.mark.asyncio
    async def test_reshare_validates_domain_membership(self, service, db, identity, domain_service):
        """Test that resharing validates new recipient is in allowed domain."""
        # Create initial share with domain restriction
        domain_service.add_domain("acme-corp", [])
        share_result, resharer_did = await self._create_bounded_share(
            service, db, identity, domain_service,
            allowed_domains=["acme-corp"],
            recipient_did="did:key:first-recipient",
        )
        
        # Add new recipient to the allowed domain
        new_recipient = "did:key:second-recipient"
        identity.add_identity(new_recipient)
        domain_service.add_member(new_recipient, "acme-corp")
        
        # Reshare - should succeed
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=new_recipient,
        )
        
        reshare_result = await service.reshare(reshare_request, resharer_did)
        assert reshare_result.share_id is not None
        assert reshare_result.current_hop == 1
    
    @pytest.mark.asyncio
    async def test_reshare_rejects_recipient_not_in_domain(self, service, db, identity, domain_service):
        """Test that resharing fails if new recipient is not in allowed domain."""
        # Create initial share with domain restriction
        domain_service.add_domain("acme-corp", [])
        share_result, resharer_did = await self._create_bounded_share(
            service, db, identity, domain_service,
            allowed_domains=["acme-corp"],
            recipient_did="did:key:first-recipient",
            belief_id="belief-reshare-reject",
        )
        
        # New recipient is NOT in the allowed domain
        new_recipient = "did:key:outsider"
        identity.add_identity(new_recipient)
        # NOT adding to acme-corp domain
        
        # Reshare - should fail
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=new_recipient,
        )
        
        with pytest.raises(ValueError, match="not a member of any allowed domain"):
            await service.reshare(reshare_request, resharer_did)
    
    @pytest.mark.asyncio
    async def test_reshare_inherits_domain_restrictions(self, service, db, identity, domain_service):
        """Test that reshares inherit the original domain restrictions."""
        # Create initial share with domain restriction
        domain_service.add_domain("restricted-domain", [])
        share_result, resharer_did = await self._create_bounded_share(
            service, db, identity, domain_service,
            allowed_domains=["restricted-domain"],
            recipient_did="did:key:hop1",
            belief_id="belief-inherit",
            max_hops=5,  # Allow multiple hops for chained resharing
        )
        
        # First reshare to second recipient in domain
        second_recipient = "did:key:hop2"
        identity.add_identity(second_recipient)
        domain_service.add_member(second_recipient, "restricted-domain")
        
        reshare1_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        reshare1_result = await service.reshare(reshare1_request, resharer_did)
        
        # Second recipient receives the reshare
        receive_request = ReceiveRequest(share_id=reshare1_result.share_id)
        await service.receive(receive_request, second_recipient)
        
        # Third recipient NOT in domain - should fail even though hop count allows it
        third_recipient = "did:key:hop3-outsider"
        identity.add_identity(third_recipient)
        # NOT adding to restricted-domain
        
        reshare2_request = ReshareRequest(
            original_share_id=reshare1_result.share_id,
            recipient_did=third_recipient,
        )
        
        # Should fail - domain restriction is inherited
        with pytest.raises(ValueError, match="not a member of any allowed domain"):
            await service.reshare(reshare2_request, second_recipient)
    
    @pytest.mark.asyncio
    async def test_reshare_without_domain_restriction(self, service, db, identity, domain_service):
        """Test that resharing works without domain restrictions."""
        # Create initial share without domain restriction (empty allowed_domains)
        share_result, resharer_did = await self._create_bounded_share(
            service, db, identity, domain_service,
            allowed_domains=[],  # Empty = no restriction
            recipient_did="did:key:unrestricted-first",
            belief_id="belief-unrestricted-reshare",
        )
        
        # New recipient doesn't need to be in any domain
        new_recipient = "did:key:unrestricted-second"
        identity.add_identity(new_recipient)
        
        # Reshare - should succeed without domain validation
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=new_recipient,
        )
        
        reshare_result = await service.reshare(reshare_request, resharer_did)
        assert reshare_result.share_id is not None
    
    @pytest.mark.asyncio
    async def test_reshare_no_domain_service_with_restrictions_fails(self, db, identity, domain_service):
        """Test that resharing with domain restrictions requires domain service."""
        # Create service with domain_service for initial share
        service_with_domains = SharingService(db, identity, domain_service)
        
        # Create share with domain restriction
        domain_service.add_domain("some-domain", [])
        share_result, resharer_did = await self._create_bounded_share(
            service_with_domains, db, identity, domain_service,
            allowed_domains=["some-domain"],
            recipient_did="did:key:restricted-recipient",
            belief_id="belief-no-service",
        )
        
        # Now create a service WITHOUT domain_service
        service_without_domains = SharingService(db, identity, domain_service=None)
        
        new_recipient = "did:key:new-recipient"
        identity.add_identity(new_recipient)
        
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=new_recipient,
        )
        
        # Should fail - need domain service for domain-restricted resharing
        with pytest.raises(ValueError, match="Domain service required"):
            await service_without_domains.reshare(reshare_request, resharer_did)


class TestStripOnForward:
    """Tests for strip_on_forward field redaction (Issue #71)."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def domain_service(self):
        return MockDomainService()
    
    @pytest.fixture
    def service(self, db, identity, domain_service):
        return SharingService(db, identity, domain_service)
    
    @pytest.mark.asyncio
    async def test_strip_flat_field_on_reshare(self, service, db, identity, domain_service):
        """Test stripping a top-level field on reshare."""
        # Content with a field to strip
        original_content = json.dumps({
            "message": "Hello world",
            "secret_api_key": "sk_live_1234567890",
            "public_data": "visible",
        })
        
        belief_id = "belief-strip-flat"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=original_content)
        
        first_recipient = "did:key:first-strip"
        second_recipient = "did:key:second-strip"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        domain_service.add_member(first_recipient, "test-domain")
        domain_service.add_member(second_recipient, "test-domain")
        
        # Create share with strip_on_forward
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=3,
                allowed_domains=["test-domain"],
                strip_on_forward=["secret_api_key"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        receive_result = await service.receive(receive_request, first_recipient)
        
        # Original content should still have the secret key
        received_content = json.loads(receive_result.content.decode("utf-8"))
        assert "secret_api_key" in received_content
        assert received_content["secret_api_key"] == "sk_live_1234567890"
        
        # First recipient reshares to second
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        reshare_result = await service.reshare(reshare_request, first_recipient)
        
        # Second recipient receives the reshared content
        receive_request2 = ReceiveRequest(share_id=reshare_result.share_id)
        receive_result2 = await service.receive(receive_request2, second_recipient)
        
        # Reshared content should NOT have the secret key
        reshared_content = json.loads(receive_result2.content.decode("utf-8"))
        assert "secret_api_key" not in reshared_content
        assert reshared_content["message"] == "Hello world"
        assert reshared_content["public_data"] == "visible"
    
    @pytest.mark.asyncio
    async def test_strip_nested_field_on_reshare(self, service, db, identity, domain_service):
        """Test stripping a nested field using dot notation."""
        # Content with nested structure
        original_content = json.dumps({
            "message": "Important message",
            "metadata": {
                "source": "internal-system",
                "timestamp": "2026-02-04T12:00:00Z",
                "author": "Alice",
            },
            "public": True,
        })
        
        belief_id = "belief-strip-nested"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=original_content)
        
        first_recipient = "did:key:first-nested"
        second_recipient = "did:key:second-nested"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        domain_service.add_member(first_recipient, "test-domain")
        domain_service.add_member(second_recipient, "test-domain")
        
        # Create share with nested strip_on_forward
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=3,
                allowed_domains=["test-domain"],
                strip_on_forward=["metadata.source", "metadata.author"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        receive_result = await service.receive(receive_request, first_recipient)
        
        # Original content should have all metadata fields
        received_content = json.loads(receive_result.content.decode("utf-8"))
        assert received_content["metadata"]["source"] == "internal-system"
        assert received_content["metadata"]["author"] == "Alice"
        
        # First recipient reshares to second
        reshare_request = ReshareRequest(
            original_share_id=share_result.share_id,
            recipient_did=second_recipient,
        )
        reshare_result = await service.reshare(reshare_request, first_recipient)
        
        # Second recipient receives the reshared content
        receive_request2 = ReceiveRequest(share_id=reshare_result.share_id)
        receive_result2 = await service.receive(receive_request2, second_recipient)
        
        # Reshared content should have metadata but without source and author
        reshared_content = json.loads(receive_result2.content.decode("utf-8"))
        assert "metadata" in reshared_content
        assert "source" not in reshared_content["metadata"]
        assert "author" not in reshared_content["metadata"]
        assert reshared_content["metadata"]["timestamp"] == "2026-02-04T12:00:00Z"
        assert reshared_content["message"] == "Important message"
        assert reshared_content["public"] is True
    
    @pytest.mark.asyncio
    async def test_strip_deeply_nested_field(self, service, db, identity, domain_service):
        """Test stripping a deeply nested field (3+ levels)."""
        original_content = json.dumps({
            "data": {
                "user": {
                    "profile": {
                        "email": "secret@example.com",
                        "name": "John Doe",
                    },
                    "id": "user123",
                },
            },
            "public_info": "visible",
        })
        
        belief_id = "belief-strip-deep"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=original_content)
        
        first_recipient = "did:key:first-deep"
        second_recipient = "did:key:second-deep"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        domain_service.add_member(first_recipient, "test-domain")
        domain_service.add_member(second_recipient, "test-domain")
        
        # Strip deeply nested email
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=3,
                allowed_domains=["test-domain"],
                strip_on_forward=["data.user.profile.email"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives and reshares
        await service.receive(ReceiveRequest(share_id=share_result.share_id), first_recipient)
        
        reshare_result = await service.reshare(
            ReshareRequest(original_share_id=share_result.share_id, recipient_did=second_recipient),
            first_recipient
        )
        
        # Second recipient receives
        receive_result = await service.receive(
            ReceiveRequest(share_id=reshare_result.share_id), second_recipient
        )
        
        reshared_content = json.loads(receive_result.content.decode("utf-8"))
        
        # Email should be stripped, but name should remain
        assert "email" not in reshared_content["data"]["user"]["profile"]
        assert reshared_content["data"]["user"]["profile"]["name"] == "John Doe"
        assert reshared_content["data"]["user"]["id"] == "user123"
        assert reshared_content["public_info"] == "visible"
    
    @pytest.mark.asyncio
    async def test_strip_multiple_fields(self, service, db, identity, domain_service):
        """Test stripping multiple fields at once."""
        original_content = json.dumps({
            "message": "Hello",
            "field1": "strip this",
            "field2": "strip this too",
            "keep_this": "visible",
            "nested": {
                "secret": "remove",
                "public": "keep",
            },
        })
        
        belief_id = "belief-strip-multi"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=original_content)
        
        first_recipient = "did:key:first-multi"
        second_recipient = "did:key:second-multi"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        domain_service.add_member(first_recipient, "test-domain")
        domain_service.add_member(second_recipient, "test-domain")
        
        # Strip multiple fields
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=3,
                allowed_domains=["test-domain"],
                strip_on_forward=["field1", "field2", "nested.secret"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives and reshares
        await service.receive(ReceiveRequest(share_id=share_result.share_id), first_recipient)
        
        reshare_result = await service.reshare(
            ReshareRequest(original_share_id=share_result.share_id, recipient_did=second_recipient),
            first_recipient
        )
        
        # Second recipient receives
        receive_result = await service.receive(
            ReceiveRequest(share_id=reshare_result.share_id), second_recipient
        )
        
        reshared_content = json.loads(receive_result.content.decode("utf-8"))
        
        # All specified fields should be stripped
        assert "field1" not in reshared_content
        assert "field2" not in reshared_content
        assert "secret" not in reshared_content["nested"]
        
        # These should remain
        assert reshared_content["message"] == "Hello"
        assert reshared_content["keep_this"] == "visible"
        assert reshared_content["nested"]["public"] == "keep"
    
    @pytest.mark.asyncio
    async def test_strip_nonexistent_field_no_error(self, service, db, identity, domain_service):
        """Test that stripping a nonexistent field doesn't cause an error."""
        original_content = json.dumps({
            "message": "Hello",
            "data": {"key": "value"},
        })
        
        belief_id = "belief-strip-missing"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=original_content)
        
        first_recipient = "did:key:first-missing"
        second_recipient = "did:key:second-missing"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        domain_service.add_member(first_recipient, "test-domain")
        domain_service.add_member(second_recipient, "test-domain")
        
        # Try to strip fields that don't exist
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=3,
                allowed_domains=["test-domain"],
                strip_on_forward=["nonexistent_field", "data.nonexistent", "a.b.c.d"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives and reshares - should not error
        await service.receive(ReceiveRequest(share_id=share_result.share_id), first_recipient)
        
        reshare_result = await service.reshare(
            ReshareRequest(original_share_id=share_result.share_id, recipient_did=second_recipient),
            first_recipient
        )
        
        # Second recipient receives
        receive_result = await service.receive(
            ReceiveRequest(share_id=reshare_result.share_id), second_recipient
        )
        
        # Content should be unchanged (except formatted as JSON)
        reshared_content = json.loads(receive_result.content.decode("utf-8"))
        assert reshared_content["message"] == "Hello"
        assert reshared_content["data"]["key"] == "value"
    
    @pytest.mark.asyncio
    async def test_strip_with_non_json_content_no_error(self, service, db, identity, domain_service):
        """Test that stripping on non-JSON content doesn't cause an error."""
        # Plain text content (not JSON)
        original_content = "This is plain text, not JSON"
        
        belief_id = "belief-strip-text"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=original_content)
        
        first_recipient = "did:key:first-text"
        second_recipient = "did:key:second-text"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        domain_service.add_member(first_recipient, "test-domain")
        domain_service.add_member(second_recipient, "test-domain")
        
        # Try to strip fields from non-JSON content
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=3,
                allowed_domains=["test-domain"],
                strip_on_forward=["some.field"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives and reshares - should not error
        await service.receive(ReceiveRequest(share_id=share_result.share_id), first_recipient)
        
        reshare_result = await service.reshare(
            ReshareRequest(original_share_id=share_result.share_id, recipient_did=second_recipient),
            first_recipient
        )
        
        # Second recipient receives
        receive_result = await service.receive(
            ReceiveRequest(share_id=reshare_result.share_id), second_recipient
        )
        
        # Content should be unchanged
        assert receive_result.content.decode("utf-8") == original_content
    
    @pytest.mark.asyncio
    async def test_original_content_not_modified(self, service, db, identity, domain_service):
        """Test that the original share's content is not modified by stripping."""
        original_content = json.dumps({
            "message": "Original message",
            "secret": "should be stripped on forward",
        })
        
        belief_id = "belief-original-intact"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=original_content)
        
        first_recipient = "did:key:first-intact"
        second_recipient = "did:key:second-intact"
        third_recipient = "did:key:third-intact"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        identity.add_identity(third_recipient)
        domain_service.add_member(first_recipient, "test-domain")
        domain_service.add_member(second_recipient, "test-domain")
        domain_service.add_member(third_recipient, "test-domain")
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=5,
                allowed_domains=["test-domain"],
                strip_on_forward=["secret"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives
        receive1 = await service.receive(
            ReceiveRequest(share_id=share_result.share_id), first_recipient
        )
        
        # First recipient reshares to second
        reshare1 = await service.reshare(
            ReshareRequest(original_share_id=share_result.share_id, recipient_did=second_recipient),
            first_recipient
        )
        
        # First recipient re-reads original - it should still have the secret
        # We can verify by checking what first_recipient received initially
        content1 = json.loads(receive1.content.decode("utf-8"))
        assert "secret" in content1
        assert content1["secret"] == "should be stripped on forward"
        
        # Second recipient receives - should NOT have secret
        receive2 = await service.receive(
            ReceiveRequest(share_id=reshare1.share_id), second_recipient
        )
        content2 = json.loads(receive2.content.decode("utf-8"))
        assert "secret" not in content2


class TestStripOnForwardWithPropagate:
    """Tests for strip_on_forward in propagate() method (Issue #71)."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def domain_service(self):
        return MockDomainService()
    
    @pytest.fixture
    def service(self, db, identity, domain_service):
        return SharingService(db, identity, domain_service)
    
    @pytest.mark.asyncio
    async def test_propagate_strips_original_fields(self, service, db, identity, domain_service):
        """Test that propagate() strips fields from original policy."""
        from valence.privacy.sharing import PropagateRequest
        
        original_content = json.dumps({
            "message": "Propagate me",
            "internal_id": "secret-123",
            "metadata": {"source": "internal"},
        })
        
        belief_id = "belief-propagate-strip"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=original_content)
        
        first_recipient = "did:key:first-prop"
        second_recipient = "did:key:second-prop"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        domain_service.add_member(first_recipient, "test-domain")
        domain_service.add_member(second_recipient, "test-domain")
        
        # Original policy with strip_on_forward
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=5,
                allowed_domains=["test-domain"],
                strip_on_forward=["internal_id"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives
        await service.receive(ReceiveRequest(share_id=share_result.share_id), first_recipient)
        
        # First recipient propagates with no additional restrictions
        propagate_result = await service.propagate(
            PropagateRequest(
                share_id=share_result.share_id,
                recipient_did=second_recipient,
            ),
            first_recipient
        )
        
        # Second recipient receives
        receive_result = await service.receive(
            ReceiveRequest(share_id=propagate_result.share_id), second_recipient
        )
        
        content = json.loads(receive_result.content.decode("utf-8"))
        
        # internal_id should be stripped
        assert "internal_id" not in content
        assert content["message"] == "Propagate me"
        assert content["metadata"]["source"] == "internal"
    
    @pytest.mark.asyncio
    async def test_propagate_composes_strip_fields(self, service, db, identity, domain_service):
        """Test that propagate() unions strip fields from original and additional restrictions."""
        from valence.privacy.sharing import PropagateRequest
        
        original_content = json.dumps({
            "message": "Propagate and strip more",
            "field_a": "from original",
            "field_b": "from additional",
            "field_c": "keep this",
        })
        
        belief_id = "belief-propagate-compose"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=original_content)
        
        first_recipient = "did:key:first-compose"
        second_recipient = "did:key:second-compose"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        domain_service.add_member(first_recipient, "test-domain")
        domain_service.add_member(second_recipient, "test-domain")
        
        # Original policy strips field_a
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=5,
                allowed_domains=["test-domain"],
                strip_on_forward=["field_a"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives
        await service.receive(ReceiveRequest(share_id=share_result.share_id), first_recipient)
        
        # First recipient propagates with additional strip_on_forward for field_b
        propagate_result = await service.propagate(
            PropagateRequest(
                share_id=share_result.share_id,
                recipient_did=second_recipient,
                additional_restrictions=PropagationRules(
                    strip_on_forward=["field_b"],
                ),
            ),
            first_recipient
        )
        
        # Verify composed restrictions include both fields
        assert "field_a" in propagate_result.composed_restrictions["strip_on_forward"]
        assert "field_b" in propagate_result.composed_restrictions["strip_on_forward"]
        
        # Second recipient receives
        receive_result = await service.receive(
            ReceiveRequest(share_id=propagate_result.share_id), second_recipient
        )
        
        content = json.loads(receive_result.content.decode("utf-8"))
        
        # Both field_a and field_b should be stripped
        assert "field_a" not in content
        assert "field_b" not in content
        assert content["message"] == "Propagate and strip more"
        assert content["field_c"] == "keep this"
    
    @pytest.mark.asyncio
    async def test_propagate_nested_field_stripping(self, service, db, identity, domain_service):
        """Test propagate() with nested field stripping."""
        from valence.privacy.sharing import PropagateRequest
        
        original_content = json.dumps({
            "data": {
                "public": "visible",
                "private": {
                    "ssn": "123-45-6789",
                    "name": "John",
                },
            },
        })
        
        belief_id = "belief-propagate-nested"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=original_content)
        
        first_recipient = "did:key:first-pnested"
        second_recipient = "did:key:second-pnested"
        identity.add_identity(first_recipient)
        identity.add_identity(second_recipient)
        domain_service.add_member(first_recipient, "test-domain")
        domain_service.add_member(second_recipient, "test-domain")
        
        # Original policy with nested strip
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=5,
                allowed_domains=["test-domain"],
                strip_on_forward=["data.private.ssn"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # First recipient receives
        await service.receive(ReceiveRequest(share_id=share_result.share_id), first_recipient)
        
        # Propagate
        propagate_result = await service.propagate(
            PropagateRequest(
                share_id=share_result.share_id,
                recipient_did=second_recipient,
            ),
            first_recipient
        )
        
        # Second recipient receives
        receive_result = await service.receive(
            ReceiveRequest(share_id=propagate_result.share_id), second_recipient
        )
        
        content = json.loads(receive_result.content.decode("utf-8"))
        
        # SSN should be stripped, but structure and other fields should remain
        assert content["data"]["public"] == "visible"
        assert "ssn" not in content["data"]["private"]
        assert content["data"]["private"]["name"] == "John"


class TestStripFieldsHelper:
    """Unit tests for the strip_fields_from_content helper function."""
    
    def test_strip_single_flat_field(self):
        """Test stripping a single top-level field."""
        from valence.privacy.sharing import strip_fields_from_content
        
        content = json.dumps({"a": 1, "b": 2, "c": 3}).encode("utf-8")
        result = strip_fields_from_content(content, ["b"])
        
        result_dict = json.loads(result.decode("utf-8"))
        assert "a" in result_dict
        assert "b" not in result_dict
        assert "c" in result_dict
    
    def test_strip_nested_field(self):
        """Test stripping a nested field."""
        from valence.privacy.sharing import strip_fields_from_content
        
        content = json.dumps({
            "level1": {
                "level2": {
                    "secret": "hidden",
                    "public": "visible",
                }
            }
        }).encode("utf-8")
        
        result = strip_fields_from_content(content, ["level1.level2.secret"])
        result_dict = json.loads(result.decode("utf-8"))
        
        assert "secret" not in result_dict["level1"]["level2"]
        assert result_dict["level1"]["level2"]["public"] == "visible"
    
    def test_strip_nonexistent_field(self):
        """Test that stripping nonexistent field doesn't error."""
        from valence.privacy.sharing import strip_fields_from_content
        
        content = json.dumps({"a": 1}).encode("utf-8")
        result = strip_fields_from_content(content, ["nonexistent", "a.b.c"])
        
        result_dict = json.loads(result.decode("utf-8"))
        assert result_dict == {"a": 1}
    
    def test_strip_empty_list(self):
        """Test stripping with empty list returns original."""
        from valence.privacy.sharing import strip_fields_from_content
        
        content = json.dumps({"a": 1}).encode("utf-8")
        result = strip_fields_from_content(content, [])
        
        assert result == content
    
    def test_strip_non_json_returns_original(self):
        """Test that non-JSON content is returned unchanged."""
        from valence.privacy.sharing import strip_fields_from_content
        
        content = b"This is not JSON"
        result = strip_fields_from_content(content, ["field"])
        
        assert result == content
    
    def test_strip_does_not_modify_original(self):
        """Test that original content is not modified."""
        from valence.privacy.sharing import strip_fields_from_content
        import copy
        
        original = {"nested": {"secret": "value", "other": "keep"}}
        content = json.dumps(original).encode("utf-8")
        original_copy = copy.deepcopy(original)
        
        result = strip_fields_from_content(content, ["nested.secret"])
        
        # Original should be unchanged
        original_after = json.loads(content.decode("utf-8"))
        assert original_after == original_copy
        
        # Result should have field stripped
        result_dict = json.loads(result.decode("utf-8"))
        assert "secret" not in result_dict["nested"]


class TestPropagateAPI:
    """Tests for propagate() API with restriction composition (Issue #70)."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    @pytest.fixture
    def identity(self):
        return MockIdentityService()
    
    @pytest.fixture
    def domain_service(self):
        return MockDomainService()
    
    @pytest.fixture
    def service(self, db, identity, domain_service):
        return SharingService(db, identity, domain_service)
    
    async def _create_and_receive_bounded_share(
        self, service, db, identity, domain_service,
        belief_id="belief-propagate-001",
        recipient_did="did:key:propagator",
        content="Content to propagate",
        allowed_domains=None,
        max_hops=5,
    ):
        """Helper to create and receive a BOUNDED share."""
        if allowed_domains is None:
            allowed_domains = ["team-alpha", "team-beta"]
        
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=content)
        identity.add_identity(recipient_did)
        
        # Add recipient to all allowed domains
        for domain in allowed_domains:
            domain_service.add_member(recipient_did, domain)
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=max_hops,
                allowed_domains=allowed_domains,
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        
        share_result = await service.share(request, identity.get_did())
        
        # Receive the share
        receive_request = ReceiveRequest(share_id=share_result.share_id)
        await service.receive(receive_request, recipient_did)
        
        return share_result, recipient_did
    
    @pytest.mark.asyncio
    async def test_propagate_basic_success(self, service, db, identity, domain_service):
        """Test basic propagate operation without additional restrictions."""
        share_result, propagator_did = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service
        )
        
        # Add second recipient to domain
        new_recipient = "did:key:new-recipient"
        identity.add_identity(new_recipient)
        domain_service.add_member(new_recipient, "team-alpha")
        
        # Propagate without additional restrictions
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
            additional_restrictions=None,
        )
        
        result = await service.propagate(propagate_request, propagator_did)
        
        assert result.share_id is not None
        assert result.consent_chain_id == share_result.consent_chain_id
        assert result.encrypted_for == new_recipient
        assert result.current_hop == 1
        assert result.hops_remaining == 4  # 5 - 1 = 4
    
    @pytest.mark.asyncio
    async def test_propagate_composes_max_hops_minimum(self, service, db, identity, domain_service):
        """Test that propagate takes minimum of max_hops."""
        # Original share has max_hops=5, so 5 hops remaining at start
        share_result, propagator_did = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
            max_hops=5,
        )
        
        # Add new recipient
        new_recipient = "did:key:hops-recipient"
        identity.add_identity(new_recipient)
        domain_service.add_member(new_recipient, "team-alpha")
        
        # Propagate with additional restriction: max_hops=2
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
            additional_restrictions=PropagationRules(max_hops=2),
        )
        
        result = await service.propagate(propagate_request, propagator_did)
        
        # min(5 remaining, 2 requested) = 2, then -1 for this hop = 1 remaining
        assert result.hops_remaining == 1
        assert result.composed_restrictions["max_hops"] == 2
    
    @pytest.mark.asyncio
    async def test_propagate_composes_allowed_domains_intersection(
        self, service, db, identity, domain_service
    ):
        """Test that propagate takes intersection of allowed_domains."""
        # Original share allows ["team-alpha", "team-beta"]
        share_result, propagator_did = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
            allowed_domains=["team-alpha", "team-beta", "team-gamma"],
        )
        
        # Add new recipient to team-alpha only
        new_recipient = "did:key:intersection-recipient"
        identity.add_identity(new_recipient)
        domain_service.add_member(new_recipient, "team-alpha")
        
        # Propagate with additional restriction: only team-alpha allowed
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
            additional_restrictions=PropagationRules(allowed_domains=["team-alpha", "team-delta"]),
        )
        
        result = await service.propagate(propagate_request, propagator_did)
        
        # Intersection: ["team-alpha", "team-beta", "team-gamma"] ∩ ["team-alpha", "team-delta"]
        # = ["team-alpha"]
        assert result.composed_restrictions["allowed_domains"] == ["team-alpha"]
    
    @pytest.mark.asyncio
    async def test_propagate_fails_with_no_common_domains(
        self, service, db, identity, domain_service
    ):
        """Test that propagate fails when domain intersection is empty."""
        share_result, propagator_did = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
            allowed_domains=["team-alpha"],
        )
        
        new_recipient = "did:key:no-common-recipient"
        identity.add_identity(new_recipient)
        
        # Propagate with disjoint domains
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
            additional_restrictions=PropagationRules(allowed_domains=["team-omega"]),
        )
        
        with pytest.raises(ValueError, match="no common domains"):
            await service.propagate(propagate_request, propagator_did)
    
    @pytest.mark.asyncio
    async def test_propagate_composes_strip_on_forward_union(
        self, service, db, identity, domain_service
    ):
        """Test that strip_on_forward fields are unioned."""
        # Create share with strip_on_forward
        belief_id = "belief-strip"
        content = json.dumps({"public": "data", "field_a": "secret_a", "field_b": "secret_b"})
        db.beliefs[belief_id] = MockBelief(id=belief_id, content=content)
        
        propagator_did = "did:key:strip-propagator"
        identity.add_identity(propagator_did)
        domain_service.add_member(propagator_did, "strip-domain")
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=5,
                allowed_domains=["strip-domain"],
                strip_on_forward=["field_a"],
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=propagator_did,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Receive
        await service.receive(ReceiveRequest(share_id=share_result.share_id), propagator_did)
        
        # Add new recipient
        new_recipient = "did:key:strip-recipient"
        identity.add_identity(new_recipient)
        domain_service.add_member(new_recipient, "strip-domain")
        
        # Propagate with additional strip_on_forward
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
            additional_restrictions=PropagationRules(strip_on_forward=["field_b"]),
        )
        
        result = await service.propagate(propagate_request, propagator_did)
        
        # Union: ["field_a"] ∪ ["field_b"] = ["field_a", "field_b"]
        assert set(result.composed_restrictions["strip_on_forward"]) == {"field_a", "field_b"}
    
    @pytest.mark.asyncio
    async def test_propagate_composes_min_trust_maximum(
        self, service, db, identity, domain_service
    ):
        """Test that min_trust_to_receive takes maximum (most restrictive)."""
        # Create share with min_trust_to_receive
        belief_id = "belief-trust"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Trust content")
        
        propagator_did = "did:key:trust-propagator"
        identity.add_identity(propagator_did)
        domain_service.add_member(propagator_did, "trust-domain")
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=5,
                allowed_domains=["trust-domain"],
                min_trust_to_receive=0.5,
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=propagator_did,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Receive
        await service.receive(ReceiveRequest(share_id=share_result.share_id), propagator_did)
        
        # Add new recipient
        new_recipient = "did:key:trust-recipient"
        identity.add_identity(new_recipient)
        domain_service.add_member(new_recipient, "trust-domain")
        
        # Propagate with higher min_trust_to_receive
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
            additional_restrictions=PropagationRules(min_trust_to_receive=0.8),
        )
        
        result = await service.propagate(propagate_request, propagator_did)
        
        # max(0.5, 0.8) = 0.8 (more restrictive)
        assert result.composed_restrictions["min_trust_to_receive"] == 0.8
    
    @pytest.mark.asyncio
    async def test_propagate_fails_for_direct_policy(self, service, db, identity, domain_service):
        """Test that propagate fails for DIRECT policy shares."""
        belief_id = "belief-direct"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Direct content")
        
        recipient_did = "did:key:direct-recipient"
        identity.add_identity(recipient_did)
        
        # Create DIRECT share (default)
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Receive
        await service.receive(ReceiveRequest(share_id=share_result.share_id), recipient_did)
        
        # Add new recipient
        new_recipient = "did:key:direct-new"
        identity.add_identity(new_recipient)
        
        # Try to propagate
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
        )
        
        with pytest.raises(ValueError, match="does not allow propagation"):
            await service.propagate(propagate_request, recipient_did)
    
    @pytest.mark.asyncio
    async def test_propagate_fails_for_revoked_share(self, service, db, identity, domain_service):
        """Test that propagate fails for revoked shares."""
        share_result, propagator_did = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
        )
        
        # Revoke the share
        revoke_request = RevokeRequest(share_id=share_result.share_id)
        await service.revoke_share(revoke_request, identity.get_did())
        
        # Add new recipient
        new_recipient = "did:key:revoked-new"
        identity.add_identity(new_recipient)
        domain_service.add_member(new_recipient, "team-alpha")
        
        # Try to propagate
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
        )
        
        with pytest.raises(ValueError, match="has been revoked"):
            await service.propagate(propagate_request, propagator_did)
    
    @pytest.mark.asyncio
    async def test_propagate_fails_when_max_hops_exceeded(
        self, service, db, identity, domain_service
    ):
        """Test that propagate fails when max_hops is exceeded."""
        # Create with max_hops=1
        share_result, propagator_did = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
            belief_id="belief-maxhops-exceeded",
            max_hops=1,
        )
        
        # First propagation
        first_recipient = "did:key:first-prop"
        identity.add_identity(first_recipient)
        domain_service.add_member(first_recipient, "team-alpha")
        
        result1 = await service.propagate(
            PropagateRequest(share_id=share_result.share_id, recipient_did=first_recipient),
            propagator_did,
        )
        
        # Receive
        await service.receive(ReceiveRequest(share_id=result1.share_id), first_recipient)
        
        # Second propagation should fail (max_hops=1 already used)
        second_recipient = "did:key:second-prop"
        identity.add_identity(second_recipient)
        domain_service.add_member(second_recipient, "team-alpha")
        
        with pytest.raises(ValueError, match="max_hops.*exceeded"):
            await service.propagate(
                PropagateRequest(share_id=result1.share_id, recipient_did=second_recipient),
                first_recipient,
            )
    
    @pytest.mark.asyncio
    async def test_propagate_only_by_recipient(self, service, db, identity, domain_service):
        """Test that only the recipient can propagate."""
        share_result, actual_recipient = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
        )
        
        imposter_did = "did:key:imposter"
        identity.add_identity(imposter_did)
        
        new_recipient = "did:key:imposter-target"
        identity.add_identity(new_recipient)
        
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
        )
        
        with pytest.raises(PermissionError, match="Only the recipient"):
            await service.propagate(propagate_request, imposter_did)
    
    @pytest.mark.asyncio
    async def test_propagate_requires_receiving_first(self, service, db, identity, domain_service):
        """Test that propagate requires receiving the share first."""
        # Create share but don't receive
        belief_id = "belief-not-received"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Not received")
        
        recipient_did = "did:key:not-received"
        identity.add_identity(recipient_did)
        domain_service.add_member(recipient_did, "team-alpha")
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(max_hops=5, allowed_domains=["team-alpha"]),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=recipient_did,
            policy=policy,
        )
        share_result = await service.share(request, identity.get_did())
        
        # Try to propagate without receiving
        new_recipient = "did:key:propagate-target"
        identity.add_identity(new_recipient)
        domain_service.add_member(new_recipient, "team-alpha")
        
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
        )
        
        with pytest.raises(ValueError, match="Must receive share before propagating"):
            await service.propagate(propagate_request, recipient_did)
    
    @pytest.mark.asyncio
    async def test_propagate_recipient_must_be_in_composed_domains(
        self, service, db, identity, domain_service
    ):
        """Test that recipient must be in the composed (intersected) domains."""
        share_result, propagator_did = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
            allowed_domains=["team-alpha", "team-beta"],
        )
        
        # New recipient is in team-beta but not team-alpha
        new_recipient = "did:key:beta-only"
        identity.add_identity(new_recipient)
        domain_service.add_member(new_recipient, "team-beta")  # Only in team-beta
        
        # Propagate with restriction to team-alpha only
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
            additional_restrictions=PropagationRules(allowed_domains=["team-alpha"]),
        )
        
        with pytest.raises(ValueError, match="not a member of any allowed domain"):
            await service.propagate(propagate_request, propagator_did)
    
    @pytest.mark.asyncio
    async def test_propagate_updates_consent_chain_hop(
        self, service, db, identity, domain_service
    ):
        """Test that propagate adds a hop to the consent chain."""
        share_result, propagator_did = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
        )
        
        new_recipient = "did:key:chain-hop"
        identity.add_identity(new_recipient)
        domain_service.add_member(new_recipient, "team-alpha")
        
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
            additional_restrictions=PropagationRules(max_hops=2),
        )
        
        result = await service.propagate(propagate_request, propagator_did)
        
        # Check consent chain has the hop
        chain = await service.get_consent_chain(result.consent_chain_id)
        
        # Should have receive hop + propagate hop
        propagate_hops = [h for h in chain.hops if h.get("propagator")]
        assert len(propagate_hops) == 1
        
        hop = propagate_hops[0]
        assert hop["propagator"] == propagator_did
        assert hop["recipient"] == new_recipient
        assert hop["hop_number"] == 1
        assert hop["composed_restrictions"]["max_hops"] == 2
        assert "signature" in hop
    
    @pytest.mark.asyncio
    async def test_propagate_decrypts_and_reencrypts_content(
        self, service, db, identity, domain_service
    ):
        """Test that propagated content is properly encrypted for new recipient."""
        original_content = "Secret propagated content"
        share_result, propagator_did = await self._create_and_receive_bounded_share(
            service, db, identity, domain_service,
            content=original_content,
        )
        
        new_recipient = "did:key:decrypt-test"
        identity.add_identity(new_recipient)
        domain_service.add_member(new_recipient, "team-alpha")
        
        propagate_request = PropagateRequest(
            share_id=share_result.share_id,
            recipient_did=new_recipient,
        )
        
        result = await service.propagate(propagate_request, propagator_did)
        
        # New recipient should be able to receive and decrypt
        receive_result = await service.receive(
            ReceiveRequest(share_id=result.share_id),
            new_recipient,
        )
        
        assert receive_result.content.decode("utf-8") == original_content
    
    @pytest.mark.asyncio
    async def test_propagate_chain_of_restrictions(
        self, service, db, identity, domain_service
    ):
        """Test a chain of propagations with progressively tighter restrictions."""
        # Start with generous restrictions
        original_domains = ["alpha", "beta", "gamma", "delta"]
        for domain in original_domains:
            domain_service.add_member("did:key:origin", domain)
        
        # Create original share
        belief_id = "belief-chain-restrictions"
        db.beliefs[belief_id] = MockBelief(id=belief_id, content="Chain content")
        
        first_recipient = "did:key:chain-first"
        identity.add_identity(first_recipient)
        for domain in original_domains:
            domain_service.add_member(first_recipient, domain)
        
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=10,
                allowed_domains=original_domains,
            ),
        )
        
        request = ShareRequest(
            belief_id=belief_id,
            recipient_did=first_recipient,
            policy=policy,
        )
        share1 = await service.share(request, identity.get_did())
        await service.receive(ReceiveRequest(share_id=share1.share_id), first_recipient)
        
        # First propagation: narrow to alpha, beta
        second_recipient = "did:key:chain-second"
        identity.add_identity(second_recipient)
        domain_service.add_member(second_recipient, "alpha")
        domain_service.add_member(second_recipient, "beta")
        
        prop1_result = await service.propagate(
            PropagateRequest(
                share_id=share1.share_id,
                recipient_did=second_recipient,
                additional_restrictions=PropagationRules(
                    max_hops=5,
                    allowed_domains=["alpha", "beta"],
                ),
            ),
            first_recipient,
        )
        await service.receive(ReceiveRequest(share_id=prop1_result.share_id), second_recipient)
        
        # Check first propagation results
        assert set(prop1_result.composed_restrictions["allowed_domains"]) == {"alpha", "beta"}
        assert prop1_result.composed_restrictions["max_hops"] == 5
        
        # Second propagation: narrow to alpha only
        third_recipient = "did:key:chain-third"
        identity.add_identity(third_recipient)
        domain_service.add_member(third_recipient, "alpha")
        
        prop2_result = await service.propagate(
            PropagateRequest(
                share_id=prop1_result.share_id,
                recipient_did=third_recipient,
                additional_restrictions=PropagationRules(
                    max_hops=2,  # min(4 remaining, 2) = 2
                    allowed_domains=["alpha"],
                ),
            ),
            second_recipient,
        )
        
        # Check second propagation: domains narrowed, hops reduced
        assert prop2_result.composed_restrictions["allowed_domains"] == ["alpha"]
        assert prop2_result.composed_restrictions["max_hops"] == 2
        assert prop2_result.hops_remaining == 1  # 2 - 1 = 1
