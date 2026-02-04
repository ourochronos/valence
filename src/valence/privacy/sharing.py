"""Sharing service for Valence belief sharing.

Implements basic share() API with DIRECT level support, consent chains,
and cryptographic enforcement.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Any
import uuid
import time
import hashlib
import json
import logging

from .types import SharePolicy, ShareLevel, EnforcementType
from .encryption import EncryptionEnvelope

logger = logging.getLogger(__name__)


@dataclass
class ShareRequest:
    """Request to share a belief with a specific recipient."""
    
    belief_id: str
    recipient_did: str
    policy: Optional[SharePolicy] = None  # Defaults to DIRECT


@dataclass
class ReceiveRequest:
    """Request to receive and acknowledge a shared belief."""
    
    share_id: str


@dataclass
class ShareResult:
    """Result of a successful share operation."""
    
    share_id: str
    consent_chain_id: str
    encrypted_for: str
    created_at: float


@dataclass
class RevokeRequest:
    """Request to revoke a previously shared belief."""
    
    share_id: str
    reason: Optional[str] = None


@dataclass
class RevokeResult:
    """Result of a successful revoke operation."""
    
    share_id: str
    consent_chain_id: str
    revoked_at: float
    affected_recipients: int


@dataclass
class ReceiveResult:
    """Result of a successful receive operation."""
    
    share_id: str
    belief_id: str
    content: bytes  # Decrypted content
    sharer_did: str
    received_at: float
    consent_chain_id: str


@dataclass
class ConsentChainEntry:
    """A consent chain tracking the origin and path of a share."""
    
    id: str
    belief_id: str
    origin_sharer: str  # DID
    origin_timestamp: float
    origin_policy: dict
    origin_signature: bytes
    hops: list
    chain_hash: bytes
    revoked: bool = False
    revoked_at: Optional[float] = None
    revoked_by: Optional[str] = None
    revoke_reason: Optional[str] = None


@dataclass
class Share:
    """A share record linking encrypted content to a consent chain."""
    
    id: str
    consent_chain_id: str
    encrypted_envelope: dict
    recipient_did: str
    created_at: float
    belief_id: Optional[str] = None
    accessed_at: Optional[float] = None
    received_at: Optional[float] = None


class DatabaseProtocol(Protocol):
    """Protocol for database operations required by SharingService."""
    
    async def get_belief(self, belief_id: str) -> Optional[Any]:
        """Get a belief by ID."""
        ...
    
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
        """Create a consent chain record."""
        ...
    
    async def create_share(
        self,
        id: str,
        consent_chain_id: str,
        encrypted_envelope: dict,
        recipient_did: str,
        belief_id: Optional[str] = None,
    ) -> None:
        """Create a share record."""
        ...
    
    async def get_share(self, share_id: str) -> Optional[Share]:
        """Get a share by ID."""
        ...
    
    async def list_shares(
        self,
        sharer_did: Optional[str] = None,
        recipient_did: Optional[str] = None,
        limit: int = 100,
        include_revoked: bool = False,
    ) -> list[Share]:
        """List shares, optionally filtered by sharer or recipient."""
        ...
    
    async def get_consent_chain(self, chain_id: str) -> Optional[ConsentChainEntry]:
        """Get a consent chain by ID."""
        ...
    
    async def revoke_consent_chain(
        self,
        consent_chain_id: str,
        revoked_at: float,
        revoked_by: str,
        reason: Optional[str] = None,
    ) -> None:
        """Mark a consent chain as revoked."""
        ...
    
    async def get_shares_by_consent_chain(
        self,
        consent_chain_id: str,
    ) -> list[Share]:
        """Get all shares associated with a consent chain."""
        ...
    
    async def acknowledge_share(
        self,
        share_id: str,
        received_at: float,
    ) -> None:
        """Mark a share as received/acknowledged."""
        ...
    
    async def add_consent_chain_hop(
        self,
        consent_chain_id: str,
        hop: dict,
    ) -> None:
        """Add a hop to a consent chain."""
        ...
    
    async def get_pending_shares(
        self,
        recipient_did: str,
    ) -> list[Share]:
        """Get shares that haven't been received yet for a recipient."""
        ...


class IdentityProtocol(Protocol):
    """Protocol for identity operations required by SharingService."""
    
    async def get_public_key(self, did: str) -> Optional[bytes]:
        """Get the X25519 public key for a DID."""
        ...
    
    async def get_private_key(self, did: str) -> Optional[bytes]:
        """Get the X25519 private key for a DID (only for local identities)."""
        ...
    
    def sign(self, data: dict) -> bytes:
        """Sign data with the local identity's Ed25519 key."""
        ...
    
    def get_did(self) -> str:
        """Get the local node's DID."""
        ...


class SharingService:
    """Service for sharing beliefs with specific recipients.
    
    Implements DIRECT sharing level with:
    - Cryptographic enforcement (encryption for recipient)
    - Consent chain creation and signing
    - Policy validation
    """
    
    def __init__(self, db: DatabaseProtocol, identity_service: IdentityProtocol):
        self.db = db
        self.identity = identity_service
    
    async def share(self, request: ShareRequest, sharer_did: str) -> ShareResult:
        """Share a belief with a specific recipient.
        
        Args:
            request: The share request containing belief_id, recipient, and optional policy
            sharer_did: The DID of the entity sharing the belief
            
        Returns:
            ShareResult with share_id, consent_chain_id, and metadata
            
        Raises:
            ValueError: If validation fails (belief not found, recipient not found, etc.)
        """
        # Default to DIRECT policy
        policy = request.policy or SharePolicy(
            level=ShareLevel.DIRECT,
            enforcement=EnforcementType.CRYPTOGRAPHIC,
            recipients=[request.recipient_did],
        )
        
        # Validate: DIRECT level only for now
        if policy.level != ShareLevel.DIRECT:
            raise ValueError("Only DIRECT sharing supported in v1")
        
        # Validate: must be cryptographic enforcement for DIRECT
        if policy.enforcement != EnforcementType.CRYPTOGRAPHIC:
            raise ValueError("DIRECT sharing requires CRYPTOGRAPHIC enforcement")
        
        # Validate: recipient must be in policy recipients
        if policy.recipients is None or request.recipient_did not in policy.recipients:
            raise ValueError("Recipient not in policy recipients list")
        
        # Get the belief
        belief = await self.db.get_belief(request.belief_id)
        if not belief:
            raise ValueError("Belief not found")
        
        # Get belief content - handle both dict and object
        belief_content = (
            belief.get("content") if isinstance(belief, dict)
            else getattr(belief, "content", None)
        )
        if not belief_content:
            raise ValueError("Belief has no content")
        
        # Get recipient's public key
        recipient_key = await self.identity.get_public_key(request.recipient_did)
        if not recipient_key:
            raise ValueError("Recipient not found or has no public key")
        
        # Encrypt for recipient
        envelope = EncryptionEnvelope.encrypt(
            content=belief_content.encode("utf-8"),
            recipient_public_key=recipient_key,
        )
        
        # Create consent chain origin
        timestamp = time.time()
        consent_origin = {
            "sharer": sharer_did,
            "recipient": request.recipient_did,
            "belief_id": request.belief_id,
            "policy": policy.to_dict(),
            "timestamp": timestamp,
        }
        
        # Sign the consent
        signature = self.identity.sign(consent_origin)
        
        # Compute chain hash
        consent_json = json.dumps(consent_origin, sort_keys=True).encode("utf-8")
        chain_hash = hashlib.sha256(consent_json + signature).digest()
        
        # Store consent chain
        consent_chain_id = str(uuid.uuid4())
        await self.db.create_consent_chain(
            id=consent_chain_id,
            belief_id=request.belief_id,
            origin_sharer=sharer_did,
            origin_timestamp=timestamp,
            origin_policy=policy.to_dict(),
            origin_signature=signature,
            chain_hash=chain_hash,
        )
        
        # Store encrypted share
        share_id = str(uuid.uuid4())
        await self.db.create_share(
            id=share_id,
            consent_chain_id=consent_chain_id,
            encrypted_envelope=envelope.to_dict(),
            recipient_did=request.recipient_did,
            belief_id=request.belief_id,
        )
        
        logger.info(
            f"Shared belief {request.belief_id} with {request.recipient_did} "
            f"(share_id={share_id}, consent_chain_id={consent_chain_id})"
        )
        
        return ShareResult(
            share_id=share_id,
            consent_chain_id=consent_chain_id,
            encrypted_for=request.recipient_did,
            created_at=timestamp,
        )
    
    async def get_share(self, share_id: str) -> Optional[Share]:
        """Get share details by ID.
        
        Args:
            share_id: The share UUID
            
        Returns:
            Share details or None if not found
        """
        return await self.db.get_share(share_id)
    
    async def list_shares(
        self,
        sharer_did: Optional[str] = None,
        recipient_did: Optional[str] = None,
        limit: int = 100,
        include_revoked: bool = False,
    ) -> list[Share]:
        """List shares, optionally filtered.
        
        Args:
            sharer_did: Filter by sharer DID
            recipient_did: Filter by recipient DID
            limit: Maximum number of results
            include_revoked: Whether to include revoked shares (default False)
            
        Returns:
            List of Share objects
        """
        return await self.db.list_shares(
            sharer_did=sharer_did,
            recipient_did=recipient_did,
            limit=limit,
            include_revoked=include_revoked,
        )
    
    async def get_consent_chain(self, chain_id: str) -> Optional[ConsentChainEntry]:
        """Get consent chain details.
        
        Args:
            chain_id: The consent chain UUID
            
        Returns:
            ConsentChainEntry or None if not found
        """
        return await self.db.get_consent_chain(chain_id)
    
    async def revoke_share(self, request: RevokeRequest, revoker_did: str) -> RevokeResult:
        """Revoke a previously shared belief.
        
        Args:
            request: The revoke request containing share_id and optional reason
            revoker_did: The DID of the entity revoking the share
            
        Returns:
            RevokeResult with revocation details
            
        Raises:
            ValueError: If share not found, consent chain not found, or already revoked
            PermissionError: If revoker is not the original sharer
        """
        # Get the share
        share = await self.db.get_share(request.share_id)
        if not share:
            raise ValueError("Share not found")
        
        # Get consent chain
        consent_chain = await self.db.get_consent_chain(share.consent_chain_id)
        if not consent_chain:
            raise ValueError("Consent chain not found")
        
        # Verify revoker is the original sharer
        if consent_chain.origin_sharer != revoker_did:
            raise PermissionError("Only original sharer can revoke")
        
        # Check if already revoked
        if consent_chain.revoked:
            raise ValueError("Share already revoked")
        
        # Mark as revoked
        revoked_at = time.time()
        await self.db.revoke_consent_chain(
            consent_chain_id=share.consent_chain_id,
            revoked_at=revoked_at,
            revoked_by=revoker_did,
            reason=request.reason,
        )
        
        # Count affected recipients (for DIRECT, just 1)
        affected = 1
        
        # Trigger revocation propagation (for future BOUNDED/CASCADING)
        # This will notify downstream recipients
        await self._propagate_revocation(consent_chain, revoker_did)
        
        logger.info(
            f"Revoked share {request.share_id} (consent_chain={consent_chain.id}) "
            f"by {revoker_did}, reason: {request.reason}"
        )
        
        return RevokeResult(
            share_id=request.share_id,
            consent_chain_id=share.consent_chain_id,
            revoked_at=revoked_at,
            affected_recipients=affected,
        )
    
    async def receive(
        self, request: ReceiveRequest, recipient_did: str
    ) -> ReceiveResult:
        """Receive and acknowledge a shared belief.
        
        This method:
        1. Validates the recipient is the intended recipient
        2. Checks the share hasn't been revoked or already received
        3. Decrypts the content using the recipient's private key
        4. Records the acknowledgment
        5. Adds a hop to the consent chain
        
        Args:
            request: The receive request containing share_id
            recipient_did: The DID of the recipient receiving the share
            
        Returns:
            ReceiveResult with decrypted content and metadata
            
        Raises:
            ValueError: If share not found, already received, or revoked
            PermissionError: If caller is not the intended recipient
        """
        # Get the share
        share = await self.db.get_share(request.share_id)
        if not share:
            raise ValueError("Share not found")
        
        # Verify recipient
        if share.recipient_did != recipient_did:
            raise PermissionError("Not the intended recipient")
        
        # Check if already received
        if share.received_at:
            raise ValueError("Already received")
        
        # Get consent chain and check if revoked
        consent_chain = await self.db.get_consent_chain(share.consent_chain_id)
        if not consent_chain:
            raise ValueError("Consent chain not found")
        
        if consent_chain.revoked:
            raise ValueError("Share has been revoked")
        
        # Decrypt content
        recipient_private_key = await self.identity.get_private_key(recipient_did)
        if not recipient_private_key:
            raise ValueError("Recipient private key not available")
        
        envelope = EncryptionEnvelope.from_dict(share.encrypted_envelope)
        content = EncryptionEnvelope.decrypt(envelope, recipient_private_key)
        
        # Record acknowledgment
        received_at = time.time()
        await self.db.acknowledge_share(
            share_id=request.share_id,
            received_at=received_at,
        )
        
        # Add hop to consent chain
        hop = {
            "recipient": recipient_did,
            "received_at": received_at,
            "acknowledged": True,
            "signature": self.identity.sign({
                "share_id": request.share_id,
                "received_at": received_at,
            }),
        }
        await self.db.add_consent_chain_hop(
            consent_chain_id=share.consent_chain_id,
            hop=hop,
        )
        
        logger.info(
            f"Received share {request.share_id} by {recipient_did} "
            f"(consent_chain={share.consent_chain_id})"
        )
        
        return ReceiveResult(
            share_id=request.share_id,
            belief_id=share.belief_id or consent_chain.belief_id,
            content=content,
            sharer_did=consent_chain.origin_sharer,
            received_at=received_at,
            consent_chain_id=share.consent_chain_id,
        )
    
    async def list_pending_shares(self, recipient_did: str) -> list[Share]:
        """List shares waiting to be received by a recipient.
        
        Args:
            recipient_did: The DID of the recipient
            
        Returns:
            List of shares that haven't been received yet
        """
        return await self.db.get_pending_shares(recipient_did)
    
    async def _propagate_revocation(
        self, consent_chain: ConsentChainEntry, revoker_did: str
    ) -> None:
        """Propagate revocation to downstream recipients.
        
        For DIRECT shares, notifies the single recipient.
        For BOUNDED/CASCADING, traverses hops and notifies all.
        
        Args:
            consent_chain: The consent chain being revoked
            revoker_did: The DID of the revoker
        """
        # Get all shares from this consent chain
        shares = await self.db.get_shares_by_consent_chain(consent_chain.id)
        
        for share in shares:
            # Queue revocation notification for recipient
            await self._queue_revocation_notification(
                recipient_did=share.recipient_did,
                consent_chain_id=consent_chain.id,
                revoked_by=revoker_did,
            )
    
    async def _queue_revocation_notification(
        self,
        recipient_did: str,
        consent_chain_id: str,
        revoked_by: str,
    ) -> None:
        """Queue a revocation notification for delivery.
        
        This will be sent via the network layer when available.
        For now, just logs the notification intent.
        
        Args:
            recipient_did: The DID of the notification recipient
            consent_chain_id: The revoked consent chain ID
            revoked_by: The DID of the revoker
        """
        # This will be sent via the network layer when available
        # For now, just log it
        logger.debug(
            f"Queued revocation notification: recipient={recipient_did}, "
            f"consent_chain={consent_chain_id}, revoked_by={revoked_by}"
        )
