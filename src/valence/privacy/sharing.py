"""Sharing service for Valence belief sharing.

Implements share() API with DIRECT and BOUNDED level support, consent chains,
and cryptographic/policy enforcement.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Protocol, Any, List, AsyncIterator, Set
import uuid
import time
import hashlib
import json
import logging

from .types import SharePolicy, ShareLevel, EnforcementType, PropagationRules
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
class ReshareRequest:
    """Request to reshare a previously received belief."""
    
    original_share_id: str  # The share we received and want to reshare
    recipient_did: str  # Who we're resharing with


@dataclass
class ReshareResult:
    """Result of a successful reshare operation."""
    
    share_id: str
    consent_chain_id: str
    encrypted_for: str
    created_at: float
    current_hop: int  # The hop number of this reshare
    hops_remaining: Optional[int]  # Remaining hops, or None if unlimited


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
class RevocationNotification:
    """Notification sent to recipients when a share is revoked."""
    
    id: str
    consent_chain_id: str
    belief_id: str
    revoked_by: str
    revoked_at: float
    reason: Optional[str] = None


@dataclass
class Notification:
    """A stored notification pending delivery."""
    
    id: str
    recipient_did: str
    notification_type: str
    payload: dict
    created_at: float
    acknowledged_at: Optional[float] = None


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
    current_hop: int = 0  # Current depth in propagation chain
    revoked: bool = False
    revoked_at: Optional[float] = None
    revoked_by: Optional[str] = None
    revoke_reason: Optional[str] = None
    
    @property
    def max_hops(self) -> Optional[int]:
        """Get max_hops from the origin policy's propagation rules."""
        propagation = self.origin_policy.get("propagation")
        if propagation:
            return propagation.get("max_hops")
        return None
    
    def can_reshare(self) -> bool:
        """Check if this consent chain allows another reshare hop."""
        if self.revoked:
            return False
        if self.max_hops is None:
            return True  # No limit
        return self.current_hop < self.max_hops
    
    def hops_remaining(self) -> Optional[int]:
        """Get the number of reshare hops remaining, or None if unlimited."""
        if self.max_hops is None:
            return None
        return max(0, self.max_hops - self.current_hop)


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
    
    async def create_notification(
        self,
        id: str,
        recipient_did: str,
        notification_type: str,
        payload: dict,
        created_at: float,
    ) -> None:
        """Create a notification for a recipient."""
        ...
    
    async def get_pending_notifications(
        self,
        recipient_did: str,
    ) -> list[Notification]:
        """Get pending notifications for a recipient."""
        ...
    
    async def acknowledge_notification(
        self,
        notification_id: str,
        acknowledged_at: float,
    ) -> None:
        """Mark a notification as acknowledged/delivered."""
        ...
    
    async def get_notification(
        self,
        notification_id: str,
    ) -> Optional[Notification]:
        """Get a notification by ID."""
        ...
    
    async def update_consent_chain_hop_count(
        self,
        consent_chain_id: str,
        current_hop: int,
    ) -> None:
        """Update the current hop count of a consent chain."""
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


class DomainServiceProtocol(Protocol):
    """Protocol for domain membership operations required by SharingService.
    
    Domains are logical groupings (e.g., organizations, teams, communities)
    that constrain BOUNDED sharing. Recipients can only reshare within
    allowed domains.
    """
    
    async def is_member(self, did: str, domain: str) -> bool:
        """Check if a DID is a member of a domain.
        
        Args:
            did: The DID to check membership for
            domain: The domain identifier to check
            
        Returns:
            True if the DID is a member of the domain
        """
        ...
    
    async def get_domains_for_did(self, did: str) -> List[str]:
        """Get all domains a DID is a member of.
        
        Args:
            did: The DID to get domains for
            
        Returns:
            List of domain identifiers the DID belongs to
        """
        ...


@dataclass
class ReshareRequest:
    """Request to reshare a previously received belief.
    
    Only valid for BOUNDED shares within allowed domains.
    """
    
    original_share_id: str  # The share that was received
    recipient_did: str  # New recipient to reshare with


@dataclass
class ReshareResult:
    """Result of a successful reshare operation."""
    
    share_id: str
    consent_chain_id: str  # Same as original (extended with new hop)
    encrypted_for: str
    created_at: float
    current_hop: int  # Position in the consent chain
    hops_remaining: Optional[int] = None  # Remaining hops allowed, or None if unlimited


class SharingService:
    """Service for sharing beliefs with specific recipients.
    
    Implements DIRECT and BOUNDED sharing levels with:
    - Cryptographic enforcement (encryption for recipient)
    - Policy enforcement (domain membership validation)
    - Consent chain creation, signing, and hop tracking
    - Revocation propagation to all recipients in consent chain
    - Resharing within BOUNDED domain constraints
    """
    
    def __init__(
        self,
        db: DatabaseProtocol,
        identity_service: IdentityProtocol,
        domain_service: Optional[DomainServiceProtocol] = None,
    ):
        self.db = db
        self.identity = identity_service
        self.domain_service = domain_service
        # In-memory queue for immediate delivery attempts
        self._revocation_queue: List[tuple[str, RevocationNotification]] = []
    
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
        
        # Validate: only DIRECT and BOUNDED supported
        if policy.level not in (ShareLevel.DIRECT, ShareLevel.BOUNDED):
            raise ValueError("Only DIRECT and BOUNDED sharing supported")
        
        # Validate level-specific requirements
        if policy.level == ShareLevel.DIRECT:
            # DIRECT requires cryptographic enforcement
            if policy.enforcement != EnforcementType.CRYPTOGRAPHIC:
                raise ValueError("DIRECT sharing requires CRYPTOGRAPHIC enforcement")
            
            # Validate: recipient must be in policy recipients
            if policy.recipients is None or request.recipient_did not in policy.recipients:
                raise ValueError("Recipient not in policy recipients list")
        
        elif policy.level == ShareLevel.BOUNDED:
            # BOUNDED requires policy enforcement (domain membership checks)
            if policy.enforcement not in (EnforcementType.POLICY, EnforcementType.CRYPTOGRAPHIC):
                raise ValueError("BOUNDED sharing requires POLICY or CRYPTOGRAPHIC enforcement")
            
            # Get allowed_domains from policy (empty/None = no restriction)
            allowed_domains = (
                policy.propagation.allowed_domains
                if policy.propagation else None
            )
            
            # If allowed_domains is specified and non-empty, validate domain membership
            if allowed_domains:
                # Require domain service when domain restrictions are specified
                if self.domain_service is None:
                    raise ValueError("Domain service required for domain-restricted sharing")
                
                # Validate: recipient must be member of at least one allowed domain
                recipient_in_domain = False
                for domain in allowed_domains:
                    if await self.domain_service.is_member(request.recipient_did, domain):
                        recipient_in_domain = True
                        break
                
                if not recipient_in_domain:
                    raise ValueError(
                        f"Recipient {request.recipient_did} is not a member of any allowed domain"
                    )
            # Empty allowed_domains = no domain restriction (anyone can receive)
        
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
    
    async def reshare(
        self, request: ReshareRequest, resharer_did: str
    ) -> ReshareResult:
        """Reshare a previously received belief with a new recipient.
        
        This implements max_hops propagation tracking:
        - Validates the resharer received the original share
        - Checks if max_hops allows another hop
        - Increments the hop counter
        - Creates a new share linked to the original consent chain
        
        Args:
            request: The reshare request containing original_share_id and recipient
            resharer_did: The DID of the entity resharing (must have received the share)
            
        Returns:
            ReshareResult with the new share details and hop tracking info
            
        Raises:
            ValueError: If validation fails
            PermissionError: If resharer didn't receive the original share
        """
        # Get the original share
        original_share = await self.db.get_share(request.original_share_id)
        if not original_share:
            raise ValueError("Original share not found")
        
        # Verify resharer received the original share
        if original_share.recipient_did != resharer_did:
            raise PermissionError("Only the recipient of a share can reshare it")
        
        # Verify the share was actually received
        if original_share.received_at is None:
            raise ValueError("Must receive share before resharing")
        
        # Get the consent chain
        consent_chain = await self.db.get_consent_chain(original_share.consent_chain_id)
        if not consent_chain:
            raise ValueError("Consent chain not found")
        
        # Check if revoked
        if consent_chain.revoked:
            raise ValueError("Cannot reshare: original share has been revoked")
        
        # Check policy allows resharing
        policy_level = consent_chain.origin_policy.get("level")
        if policy_level not in ("bounded", "cascading", "public"):
            raise ValueError(
                f"Cannot reshare: policy level '{policy_level}' does not allow resharing"
            )
        
        # Check max_hops
        if not consent_chain.can_reshare():
            raise ValueError(
                f"Cannot reshare: max_hops ({consent_chain.max_hops}) exceeded "
                f"(current_hop={consent_chain.current_hop})"
            )
        
        # Validate recipient is allowed (check allowed_domains if set)
        propagation = consent_chain.origin_policy.get("propagation", {})
        allowed_domains = propagation.get("allowed_domains")
        
        # If allowed_domains is specified and non-empty, validate domain membership
        if allowed_domains:
            if self.domain_service is None:
                raise ValueError("Domain service required for domain-restricted resharing")
            
            # Recipient must be member of at least one allowed domain
            recipient_in_domain = False
            for domain in allowed_domains:
                if await self.domain_service.is_member(request.recipient_did, domain):
                    recipient_in_domain = True
                    break
            
            if not recipient_in_domain:
                raise ValueError(
                    f"Recipient {request.recipient_did} is not a member of any allowed domain"
                )
        # Empty allowed_domains = no domain restriction
        
        # Check expiration
        expires_at = propagation.get("expires_at")
        if expires_at:
            from datetime import datetime
            exp_time = datetime.fromisoformat(expires_at)
            if datetime.now() > exp_time:
                raise ValueError("Cannot reshare: share has expired")
        
        # Get the original belief content (need to decrypt first)
        recipient_private_key = await self.identity.get_private_key(resharer_did)
        if not recipient_private_key:
            raise ValueError("Resharer private key not available")
        
        envelope = EncryptionEnvelope.from_dict(original_share.encrypted_envelope)
        original_content = EncryptionEnvelope.decrypt(envelope, recipient_private_key)
        
        # Strip fields if configured
        strip_fields = propagation.get("strip_on_forward")
        content_to_share = original_content
        if strip_fields:
            try:
                content_dict = json.loads(original_content.decode("utf-8"))
                for field in strip_fields:
                    content_dict.pop(field, None)
                content_to_share = json.dumps(content_dict).encode("utf-8")
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Not JSON content, can't strip fields
                pass
        
        # Get new recipient's public key
        recipient_key = await self.identity.get_public_key(request.recipient_did)
        if not recipient_key:
            raise ValueError("Recipient not found or has no public key")
        
        # Encrypt for new recipient
        new_envelope = EncryptionEnvelope.encrypt(
            content=content_to_share,
            recipient_public_key=recipient_key,
        )
        
        # Calculate new hop count
        new_hop_count = consent_chain.current_hop + 1
        
        # Store new share (linked to same consent chain)
        share_id = str(uuid.uuid4())
        timestamp = time.time()
        await self.db.create_share(
            id=share_id,
            consent_chain_id=consent_chain.id,
            encrypted_envelope=new_envelope.to_dict(),
            recipient_did=request.recipient_did,
            belief_id=original_share.belief_id,
        )
        
        # Add hop to consent chain
        hop = {
            "resharer": resharer_did,
            "recipient": request.recipient_did,
            "reshared_at": timestamp,
            "hop_number": new_hop_count,
            "from_share_id": request.original_share_id,
            "signature": self.identity.sign({
                "original_share_id": request.original_share_id,
                "recipient": request.recipient_did,
                "reshared_at": timestamp,
                "hop_number": new_hop_count,
            }),
        }
        await self.db.add_consent_chain_hop(
            consent_chain_id=consent_chain.id,
            hop=hop,
        )
        
        # Update hop count in consent chain
        await self.db.update_consent_chain_hop_count(
            consent_chain_id=consent_chain.id,
            current_hop=new_hop_count,
        )
        
        logger.info(
            f"Reshared belief {original_share.belief_id} from {resharer_did} "
            f"to {request.recipient_did} (share_id={share_id}, "
            f"consent_chain_id={consent_chain.id}, hop={new_hop_count})"
        )
        
        return ReshareResult(
            share_id=share_id,
            consent_chain_id=consent_chain.id,
            encrypted_for=request.recipient_did,
            created_at=timestamp,
            current_hop=new_hop_count,
            hops_remaining=consent_chain.max_hops - new_hop_count if consent_chain.max_hops else None,
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
        # Note: Pass the reason explicitly since consent_chain was fetched before update
        await self._propagate_revocation(consent_chain, revoker_did, request.reason)
        
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
    
    async def reshare(
        self, request: ReshareRequest, resharer_did: str
    ) -> ReshareResult:
        """Reshare a previously received belief with a new recipient.
        
        Only valid for BOUNDED shares. The new recipient must be a member
        of at least one of the allowed domains specified in the original
        share's propagation rules.
        
        Args:
            request: The reshare request containing original share_id and new recipient
            resharer_did: The DID of the entity resharing (must have received the share)
            
        Returns:
            ReshareResult with new share details and hop number
            
        Raises:
            ValueError: If validation fails (share not found, not received,
                        policy doesn't allow resharing, max hops exceeded, etc.)
            PermissionError: If resharer hasn't received the share or recipient
                            is not in allowed domains
        """
        # Get the original share
        original_share = await self.db.get_share(request.original_share_id)
        if not original_share:
            raise ValueError("Original share not found")
        
        # Verify resharer is the recipient of the original share
        if original_share.recipient_did != resharer_did:
            raise PermissionError("Only the share recipient can reshare")
        
        # Verify the share was received (acknowledged)
        if original_share.received_at is None:
            raise ValueError("Share must be received before resharing")
        
        # Get consent chain
        consent_chain = await self.db.get_consent_chain(original_share.consent_chain_id)
        if not consent_chain:
            raise ValueError("Consent chain not found")
        
        # Check if revoked
        if consent_chain.revoked:
            raise ValueError("Cannot reshare: original share has been revoked")
        
        # Get original policy
        policy = SharePolicy.from_dict(consent_chain.origin_policy)
        
        # Validate: resharing is only allowed for BOUNDED level
        if policy.level != ShareLevel.BOUNDED:
            raise ValueError(
                f"Resharing not allowed for {policy.level.value} shares. "
                "Only BOUNDED shares can be reshared."
            )
        
        # Validate: check propagation rules
        if policy.propagation is None:
            raise ValueError("BOUNDED share missing propagation rules")
        
        # Check max_hops if specified
        current_hop = len(consent_chain.hops)
        if policy.propagation.max_hops is not None:
            if current_hop >= policy.propagation.max_hops:
                raise ValueError(
                    f"Maximum reshare hops reached ({policy.propagation.max_hops})"
                )
        
        # Check expiration
        if policy.is_expired():
            raise ValueError("Share has expired and cannot be reshared")
        
        # Validate: new recipient must be member of allowed domains
        if not policy.propagation.allowed_domains:
            raise ValueError("No allowed domains specified for BOUNDED share")
        
        # Require domain service
        if self.domain_service is None:
            raise ValueError("Domain service required for BOUNDED resharing")
        
        # Check recipient domain membership
        recipient_in_domain = False
        recipient_domains: List[str] = []
        for domain in policy.propagation.allowed_domains:
            if await self.domain_service.is_member(request.recipient_did, domain):
                recipient_in_domain = True
                recipient_domains.append(domain)
        
        if not recipient_in_domain:
            raise ValueError(
                f"Recipient {request.recipient_did} is not a member of any allowed domain "
                f"({', '.join(policy.propagation.allowed_domains)})"
            )
        
        # Get recipient's public key
        recipient_key = await self.identity.get_public_key(request.recipient_did)
        if not recipient_key:
            raise ValueError("Recipient not found or has no public key")
        
        # Get the decrypted content from resharer's copy
        # Note: We need to decrypt from the resharer's perspective first
        resharer_private_key = await self.identity.get_private_key(resharer_did)
        if not resharer_private_key:
            raise ValueError("Resharer private key not available")
        
        original_envelope = EncryptionEnvelope.from_dict(original_share.encrypted_envelope)
        content = EncryptionEnvelope.decrypt(original_envelope, resharer_private_key)
        
        # Re-encrypt for new recipient
        new_envelope = EncryptionEnvelope.encrypt(
            content=content,
            recipient_public_key=recipient_key,
        )
        
        # Create reshare record
        timestamp = time.time()
        new_share_id = str(uuid.uuid4())
        
        await self.db.create_share(
            id=new_share_id,
            consent_chain_id=consent_chain.id,  # Same consent chain
            encrypted_envelope=new_envelope.to_dict(),
            recipient_did=request.recipient_did,
            belief_id=consent_chain.belief_id,
        )
        
        # Add reshare hop to consent chain
        hop_number = current_hop + 1
        reshare_hop = {
            "recipient": request.recipient_did,
            "reshared_by": resharer_did,
            "reshared_at": timestamp,
            "from_share_id": request.original_share_id,
            "new_share_id": new_share_id,
            "hop_number": hop_number,
            "recipient_domains": recipient_domains,
            "signature": self.identity.sign({
                "original_share_id": request.original_share_id,
                "new_share_id": new_share_id,
                "recipient": request.recipient_did,
                "reshared_at": timestamp,
            }),
        }
        
        await self.db.add_consent_chain_hop(
            consent_chain_id=consent_chain.id,
            hop=reshare_hop,
        )
        
        logger.info(
            f"Reshared belief from share {request.original_share_id} to {request.recipient_did} "
            f"(new_share_id={new_share_id}, hop={hop_number}, "
            f"consent_chain={consent_chain.id})"
        )
        
        return ReshareResult(
            share_id=new_share_id,
            consent_chain_id=consent_chain.id,
            encrypted_for=request.recipient_did,
            created_at=timestamp,
            hop_number=hop_number,
        )
    
    def _get_domain_constraints_from_chain(
        self, consent_chain: ConsentChainEntry
    ) -> Optional[List[str]]:
        """Extract domain constraints from a consent chain.
        
        Returns the allowed_domains from the original policy's propagation rules.
        
        Args:
            consent_chain: The consent chain to extract domains from
            
        Returns:
            List of allowed domain identifiers, or None if not a BOUNDED share
        """
        policy = SharePolicy.from_dict(consent_chain.origin_policy)
        if policy.level != ShareLevel.BOUNDED:
            return None
        if policy.propagation is None:
            return None
        return policy.propagation.allowed_domains
    
    async def _propagate_revocation(
        self, consent_chain: ConsentChainEntry, revoker_did: str, reason: Optional[str] = None
    ) -> int:
        """Propagate revocation to all downstream recipients.
        
        For DIRECT shares, notifies the single recipient.
        For BOUNDED/CASCADING, traverses hops and notifies all downstream recipients.
        
        Args:
            consent_chain: The consent chain being revoked
            revoker_did: The DID of the revoker
            reason: The revocation reason (passed explicitly since consent_chain
                    may have been fetched before the update)
            
        Returns:
            Number of notifications sent
        """
        # Get all shares from this consent chain
        shares = await self.db.get_shares_by_consent_chain(consent_chain.id)
        
        notifications_sent = 0
        revoked_at = time.time()
        
        # Track recipients to avoid duplicate notifications
        notified_recipients: set[str] = set()
        
        # Notify direct share recipients
        for share in shares:
            if share.recipient_did not in notified_recipients:
                notification = RevocationNotification(
                    id=str(uuid.uuid4()),
                    consent_chain_id=consent_chain.id,
                    belief_id=consent_chain.belief_id,
                    revoked_by=revoker_did,
                    revoked_at=revoked_at,
                    reason=reason,
                )
                
                await self._queue_revocation_notification(
                    recipient_did=share.recipient_did,
                    notification=notification,
                )
                notified_recipients.add(share.recipient_did)
                notifications_sent += 1
        
        # For BOUNDED/CASCADING shares, traverse hops and notify downstream
        for hop in consent_chain.hops:
            # Notify the hop recipient if not already notified
            hop_recipient = hop.get("recipient")
            if hop_recipient and hop_recipient not in notified_recipients:
                notification = RevocationNotification(
                    id=str(uuid.uuid4()),
                    consent_chain_id=consent_chain.id,
                    belief_id=consent_chain.belief_id,
                    revoked_by=revoker_did,
                    revoked_at=revoked_at,
                    reason=reason,
                )
                
                await self._queue_revocation_notification(
                    recipient_did=hop_recipient,
                    notification=notification,
                )
                notified_recipients.add(hop_recipient)
                notifications_sent += 1
            
            # Handle reshared_to for future BOUNDED/CASCADING support
            reshared_to = hop.get("reshared_to", [])
            for downstream_recipient in reshared_to:
                if downstream_recipient not in notified_recipients:
                    notification = RevocationNotification(
                        id=str(uuid.uuid4()),
                        consent_chain_id=consent_chain.id,
                        belief_id=consent_chain.belief_id,
                        revoked_by=revoker_did,
                        revoked_at=revoked_at,
                        reason=reason,
                    )
                    
                    await self._queue_revocation_notification(
                        recipient_did=downstream_recipient,
                        notification=notification,
                    )
                    notified_recipients.add(downstream_recipient)
                    notifications_sent += 1
        
        logger.info(
            f"Propagated revocation for consent_chain={consent_chain.id}: "
            f"{notifications_sent} notifications queued"
        )
        
        return notifications_sent
    
    async def _queue_revocation_notification(
        self,
        recipient_did: str,
        notification: RevocationNotification,
    ) -> None:
        """Queue a revocation notification for delivery.
        
        Stores notification in database for persistence (survives restarts)
        and adds to in-memory queue for immediate delivery attempt.
        
        Args:
            recipient_did: The DID of the notification recipient
            notification: The revocation notification to deliver
        """
        # Store in database for persistence
        await self.db.create_notification(
            id=notification.id,
            recipient_did=recipient_did,
            notification_type="revocation",
            payload=asdict(notification),
            created_at=notification.revoked_at,
        )
        
        # Add to in-memory queue for immediate delivery attempt
        self._revocation_queue.append((recipient_did, notification))
        
        logger.debug(
            f"Queued revocation notification: recipient={recipient_did}, "
            f"notification_id={notification.id}, "
            f"consent_chain={notification.consent_chain_id}"
        )
    
    async def process_revocation_queue(self) -> AsyncIterator[tuple[str, RevocationNotification]]:
        """Process queued revocation notifications.
        
        Called by the network layer to get pending notifications for delivery.
        
        Yields:
            Tuples of (recipient_did, notification) for each queued notification
        """
        while self._revocation_queue:
            recipient_did, notification = self._revocation_queue.pop(0)
            yield recipient_did, notification
    
    async def get_pending_notifications(self, recipient_did: str) -> list[Notification]:
        """Get pending notifications for a recipient.
        
        Used by recipients to poll for notifications (e.g., when they come online).
        
        Args:
            recipient_did: The DID of the recipient
            
        Returns:
            List of pending notifications
        """
        return await self.db.get_pending_notifications(recipient_did)
    
    async def acknowledge_notification(self, notification_id: str, recipient_did: str) -> bool:
        """Mark a notification as acknowledged/delivered.
        
        Args:
            notification_id: The ID of the notification
            recipient_did: The DID of the recipient (for authorization)
            
        Returns:
            True if acknowledged successfully, False if not found or unauthorized
            
        Raises:
            PermissionError: If recipient is not the notification recipient
        """
        notification = await self.db.get_notification(notification_id)
        if not notification:
            return False
        
        if notification.recipient_did != recipient_did:
            raise PermissionError("Not the notification recipient")
        
        if notification.acknowledged_at is not None:
            # Already acknowledged
            return True
        
        await self.db.acknowledge_notification(notification_id, time.time())
        
        logger.debug(
            f"Acknowledged notification: id={notification_id}, "
            f"recipient={recipient_did}"
        )
        
        return True
