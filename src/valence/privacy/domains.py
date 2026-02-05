"""Domain management for Valence.

Domains are named scopes for trust (e.g., "research-team", "family").
They provide a way to organize members and control access to beliefs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Protocol, Any, Dict
import uuid
import logging
import hashlib
import hmac
import re

logger = logging.getLogger(__name__)


class DomainRole(Enum):
    """Roles within a domain."""
    
    OWNER = "owner"    # Full control, can delete domain
    ADMIN = "admin"    # Can manage members
    MEMBER = "member"  # Basic access


class VerificationMethod(Enum):
    """Supported verification methods for domain membership."""
    
    NONE = "none"                    # No verification required
    ADMIN_SIGNATURE = "admin_sig"    # Requires signature from domain admin
    DNS_TXT = "dns_txt"              # External DNS TXT record verification
    CUSTOM = "custom"                # Custom/pluggable verification


@dataclass
class VerificationRequirement:
    """Configuration for domain membership verification.
    
    Specifies what verification is needed to prove membership.
    """
    
    method: VerificationMethod
    config: Dict[str, Any] = field(default_factory=dict)
    required: bool = True  # If False, verification is optional
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "method": self.method.value,
            "config": self.config,
            "required": self.required,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VerificationRequirement":
        """Deserialize from dictionary."""
        return cls(
            method=VerificationMethod(data["method"]),
            config=data.get("config", {}),
            required=data.get("required", True),
        )


@dataclass
class VerificationResult:
    """Result of a membership verification attempt."""
    
    verified: bool
    method: VerificationMethod
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None  # Proof/signature data
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "verified": self.verified,
            "method": self.method.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "details": self.details,
            "evidence": self.evidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VerificationResult":
        """Deserialize from dictionary."""
        timestamp = data.get("timestamp")
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        return cls(
            verified=data["verified"],
            method=VerificationMethod(data["method"]),
            timestamp=timestamp,
            details=data.get("details"),
            evidence=data.get("evidence"),
        )


class Verifier(Protocol):
    """Protocol for membership verification strategies."""
    
    async def verify(
        self,
        domain_id: str,
        member_did: str,
        requirement: VerificationRequirement,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Verify a membership claim.
        
        Args:
            domain_id: The domain UUID
            member_did: DID claiming membership
            requirement: The verification requirement configuration
            evidence: Optional proof/credentials for verification
            
        Returns:
            VerificationResult indicating success/failure
        """
        ...


class AdminSignatureVerifier:
    """Verifier that requires a signature from a domain admin.
    
    The admin signs a message containing the domain_id and member_did,
    proving they authorize the membership.
    """
    
    def __init__(self, get_admin_key_func=None):
        """Initialize with optional key lookup function.
        
        Args:
            get_admin_key_func: Async function(admin_did) -> public_key
                               If None, uses simple HMAC verification
        """
        self.get_admin_key = get_admin_key_func
    
    async def verify(
        self,
        domain_id: str,
        member_did: str,
        requirement: VerificationRequirement,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Verify membership via admin signature.
        
        Evidence should contain:
        - admin_did: DID of the signing admin
        - signature: The signature bytes (hex encoded)
        - message: Optional custom message (defaults to domain_id:member_did)
        """
        if not evidence:
            return VerificationResult(
                verified=False,
                method=VerificationMethod.ADMIN_SIGNATURE,
                details="No evidence provided",
            )
        
        admin_did = evidence.get("admin_did")
        signature = evidence.get("signature")
        
        if not admin_did or not signature:
            return VerificationResult(
                verified=False,
                method=VerificationMethod.ADMIN_SIGNATURE,
                details="Missing admin_did or signature in evidence",
            )
        
        # Build the message to verify
        message = evidence.get("message", f"{domain_id}:{member_did}")
        
        # Get admin's key for verification
        if self.get_admin_key:
            try:
                admin_key = await self.get_admin_key(admin_did)
                if not admin_key:
                    return VerificationResult(
                        verified=False,
                        method=VerificationMethod.ADMIN_SIGNATURE,
                        details=f"Could not retrieve key for admin {admin_did}",
                    )
                # Full cryptographic verification would go here
                # For now, use HMAC as a placeholder
                expected_sig = hmac.new(
                    admin_key.encode() if isinstance(admin_key, str) else admin_key,
                    message.encode(),
                    hashlib.sha256
                ).hexdigest()
                verified = hmac.compare_digest(signature, expected_sig)
            except Exception as e:  # Intentionally broad: return verification failure, don't propagate
                return VerificationResult(
                    verified=False,
                    method=VerificationMethod.ADMIN_SIGNATURE,
                    details=f"Verification error: {e}",
                )
        else:
            # Simple mode: verify signature format matches expected pattern
            # In production, would use proper cryptographic verification
            expected_sig = hashlib.sha256(
                f"{domain_id}:{member_did}:{admin_did}".encode()
            ).hexdigest()
            verified = hmac.compare_digest(signature, expected_sig)
        
        return VerificationResult(
            verified=verified,
            method=VerificationMethod.ADMIN_SIGNATURE,
            details="Signature verified" if verified else "Signature mismatch",
            evidence={"admin_did": admin_did, "verified_at": datetime.now(timezone.utc).isoformat()},
        )


class DNSTxtVerifier:
    """Verifier that checks DNS TXT records for domain verification.
    
    Used for organizational domains where members can prove affiliation
    by showing a TXT record at a specific DNS location.
    
    Config should specify:
    - domain_pattern: DNS domain pattern (e.g., "_valence.{org}.example.com")
    - expected_prefix: Expected TXT record prefix (e.g., "valence-member=")
    """
    
    def __init__(self, dns_resolver=None):
        """Initialize with optional DNS resolver.
        
        Args:
            dns_resolver: Async function(hostname) -> List[str] of TXT records
                         If None, uses a mock implementation
        """
        self.dns_resolver = dns_resolver
    
    async def _default_dns_lookup(self, hostname: str) -> List[str]:
        """Default DNS lookup - returns empty (override for real DNS)."""
        # In production, would use aiodns or similar
        return []
    
    async def verify(
        self,
        domain_id: str,
        member_did: str,
        requirement: VerificationRequirement,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Verify membership via DNS TXT record.
        
        Evidence should contain:
        - dns_domain: The DNS domain to check (e.g., "example.com")
        - Or the config should have domain_pattern
        """
        config = requirement.config
        
        # Get the DNS domain to query
        if evidence and "dns_domain" in evidence:
            dns_domain = evidence["dns_domain"]
        elif "dns_domain" in config:
            dns_domain = config["dns_domain"]
        else:
            return VerificationResult(
                verified=False,
                method=VerificationMethod.DNS_TXT,
                details="No DNS domain specified in evidence or config",
            )
        
        # Build the hostname to query
        pattern = config.get("domain_pattern", "_valence.{dns_domain}")
        hostname = pattern.format(dns_domain=dns_domain, domain_id=domain_id)
        
        # Expected TXT record content
        expected_prefix = config.get("expected_prefix", "valence-member=")
        expected_value = f"{expected_prefix}{member_did}"
        
        # Look up TXT records
        resolver = self.dns_resolver or self._default_dns_lookup
        try:
            txt_records = await resolver(hostname)
        except Exception as e:  # Intentionally broad: external resolver may raise various errors
            return VerificationResult(
                verified=False,
                method=VerificationMethod.DNS_TXT,
                details=f"DNS lookup failed: {e}",
            )
        
        # Check if expected record exists
        for record in txt_records:
            if expected_value in record or record.startswith(expected_prefix):
                # For exact member verification
                if member_did in record:
                    return VerificationResult(
                        verified=True,
                        method=VerificationMethod.DNS_TXT,
                        details=f"Found matching TXT record at {hostname}",
                        evidence={"hostname": hostname, "record": record},
                    )
        
        return VerificationResult(
            verified=False,
            method=VerificationMethod.DNS_TXT,
            details=f"No matching TXT record found at {hostname}",
            evidence={"hostname": hostname, "records_found": len(txt_records)},
        )


@dataclass
class Domain:
    """A named scope for trust.
    
    Domains group members who share beliefs within a common context.
    Names must be unique per owner.
    """
    
    domain_id: str
    name: str
    owner_did: str
    description: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    verification_requirement: Optional[VerificationRequirement] = None
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "domain_id": self.domain_id,
            "name": self.name,
            "owner_did": self.owner_did,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "verification_requirement": (
                self.verification_requirement.to_dict() 
                if self.verification_requirement else None
            ),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Domain":
        """Deserialize from dictionary."""
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
        
        verification_req = data.get("verification_requirement")
        if verification_req and isinstance(verification_req, dict):
            verification_req = VerificationRequirement.from_dict(verification_req)
            
        return cls(
            domain_id=data["domain_id"],
            name=data["name"],
            owner_did=data["owner_did"],
            description=data.get("description"),
            created_at=created_at,
            verification_requirement=verification_req,
        )


@dataclass
class DomainMembership:
    """Membership record linking a DID to a domain with a role."""
    
    domain_id: str
    member_did: str
    role: DomainRole
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "domain_id": self.domain_id,
            "member_did": self.member_did,
            "role": self.role.value,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DomainMembership":
        """Deserialize from dictionary."""
        joined_at = data.get("joined_at")
        if joined_at and isinstance(joined_at, str):
            joined_at = datetime.fromisoformat(joined_at)
        elif joined_at is None:
            joined_at = datetime.now(timezone.utc)
            
        return cls(
            domain_id=data["domain_id"],
            member_did=data["member_did"],
            role=DomainRole(data["role"]),
            joined_at=joined_at,
        )


class DomainDatabaseProtocol(Protocol):
    """Protocol for database operations required by DomainService."""
    
    async def create_domain(
        self,
        domain_id: str,
        name: str,
        owner_did: str,
        description: Optional[str],
    ) -> None:
        """Create a domain record."""
        ...
    
    async def get_domain(self, domain_id: str) -> Optional[dict]:
        """Get a domain by ID."""
        ...
    
    async def get_domain_by_name(self, name: str, owner_did: str) -> Optional[dict]:
        """Get a domain by name and owner."""
        ...
    
    async def delete_domain(self, domain_id: str) -> bool:
        """Delete a domain."""
        ...
    
    async def add_membership(
        self,
        domain_id: str,
        member_did: str,
        role: str,
    ) -> None:
        """Add a member to a domain."""
        ...
    
    async def remove_membership(self, domain_id: str, member_did: str) -> bool:
        """Remove a member from a domain."""
        ...
    
    async def get_membership(
        self, domain_id: str, member_did: str
    ) -> Optional[dict]:
        """Get a specific membership."""
        ...
    
    async def list_memberships(self, domain_id: str) -> List[dict]:
        """List all members of a domain."""
        ...
    
    async def list_domains_for_member(self, member_did: str) -> List[dict]:
        """List all domains a member belongs to."""
        ...
    
    async def set_verification_requirement(
        self,
        domain_id: str,
        requirement: Optional[dict],
    ) -> None:
        """Set or clear verification requirement for a domain."""
        ...
    
    async def store_verification_result(
        self,
        domain_id: str,
        member_did: str,
        result: dict,
    ) -> None:
        """Store a verification result for a membership."""
        ...
    
    async def get_verification_result(
        self,
        domain_id: str,
        member_did: str,
    ) -> Optional[dict]:
        """Get the latest verification result for a membership."""
        ...


class DomainError(Exception):
    """Base exception for domain operations."""
    pass


class DomainNotFoundError(DomainError):
    """Raised when a domain is not found."""
    pass


class DomainExistsError(DomainError):
    """Raised when trying to create a domain that already exists."""
    pass


class MembershipExistsError(DomainError):
    """Raised when trying to add a member who is already in the domain."""
    pass


class MembershipNotFoundError(DomainError):
    """Raised when a membership is not found."""
    pass


class PermissionDeniedError(DomainError):
    """Raised when an operation is not permitted."""
    pass


class VerificationError(DomainError):
    """Raised when membership verification fails."""
    pass


class VerificationRequiredError(DomainError):
    """Raised when verification is required but not provided."""
    pass


class DomainService:
    """Service for managing domains and memberships.
    
    Handles domain creation, member management, and access control.
    """
    
    def __init__(self, db: DomainDatabaseProtocol):
        """Initialize with database backend.
        
        Args:
            db: Database implementation following DomainDatabaseProtocol
        """
        self.db = db
    
    async def create_domain(
        self,
        name: str,
        owner_did: str,
        description: Optional[str] = None,
    ) -> Domain:
        """Create a new domain.
        
        Args:
            name: Domain name (unique per owner)
            owner_did: DID of the domain owner
            description: Optional description
            
        Returns:
            The created Domain
            
        Raises:
            DomainExistsError: If a domain with this name already exists for the owner
        """
        # Check if domain already exists for this owner
        existing = await self.db.get_domain_by_name(name, owner_did)
        if existing:
            raise DomainExistsError(
                f"Domain '{name}' already exists for owner {owner_did}"
            )
        
        domain_id = str(uuid.uuid4())
        await self.db.create_domain(
            domain_id=domain_id,
            name=name,
            owner_did=owner_did,
            description=description,
        )
        
        # Automatically add owner as a member with OWNER role
        await self.db.add_membership(
            domain_id=domain_id,
            member_did=owner_did,
            role=DomainRole.OWNER.value,
        )
        
        logger.info(f"Created domain '{name}' (id={domain_id}) for owner {owner_did}")
        
        return Domain(
            domain_id=domain_id,
            name=name,
            owner_did=owner_did,
            description=description,
        )
    
    async def get_domain(self, domain_id: str) -> Domain:
        """Get a domain by ID.
        
        Args:
            domain_id: The domain UUID
            
        Returns:
            The Domain
            
        Raises:
            DomainNotFoundError: If domain doesn't exist
        """
        data = await self.db.get_domain(domain_id)
        if not data:
            raise DomainNotFoundError(f"Domain {domain_id} not found")
        return Domain.from_dict(data)
    
    async def add_member(
        self,
        domain_id: str,
        member_did: str,
        role: DomainRole = DomainRole.MEMBER,
        requester_did: Optional[str] = None,
    ) -> DomainMembership:
        """Add a member to a domain.
        
        Args:
            domain_id: The domain UUID
            member_did: DID of the new member
            role: Role to assign (default: MEMBER)
            requester_did: DID of the user making the request (for auth check)
            
        Returns:
            The created DomainMembership
            
        Raises:
            DomainNotFoundError: If domain doesn't exist
            MembershipExistsError: If member is already in the domain
            PermissionDeniedError: If requester doesn't have permission
        """
        # Verify domain exists
        domain = await self.get_domain(domain_id)
        
        # Check permissions if requester provided
        if requester_did:
            await self._check_can_manage_members(domain_id, requester_did)
        
        # Check if already a member
        existing = await self.db.get_membership(domain_id, member_did)
        if existing:
            raise MembershipExistsError(
                f"Member {member_did} is already in domain {domain_id}"
            )
        
        await self.db.add_membership(
            domain_id=domain_id,
            member_did=member_did,
            role=role.value,
        )
        
        logger.info(
            f"Added member {member_did} to domain {domain_id} with role {role.value}"
        )
        
        return DomainMembership(
            domain_id=domain_id,
            member_did=member_did,
            role=role,
        )
    
    async def remove_member(
        self,
        domain_id: str,
        member_did: str,
        requester_did: Optional[str] = None,
    ) -> bool:
        """Remove a member from a domain.
        
        Args:
            domain_id: The domain UUID
            member_did: DID of the member to remove
            requester_did: DID of the user making the request (for auth check)
            
        Returns:
            True if removed, False if wasn't a member
            
        Raises:
            DomainNotFoundError: If domain doesn't exist
            PermissionDeniedError: If trying to remove owner or requester lacks permission
        """
        domain = await self.get_domain(domain_id)
        
        # Cannot remove the owner
        if member_did == domain.owner_did:
            raise PermissionDeniedError("Cannot remove domain owner")
        
        # Check permissions if requester provided
        if requester_did:
            await self._check_can_manage_members(domain_id, requester_did)
        
        result = await self.db.remove_membership(domain_id, member_did)
        
        if result:
            logger.info(f"Removed member {member_did} from domain {domain_id}")
        
        return result
    
    async def list_members(self, domain_id: str) -> List[DomainMembership]:
        """List all members of a domain.
        
        Args:
            domain_id: The domain UUID
            
        Returns:
            List of DomainMembership objects
            
        Raises:
            DomainNotFoundError: If domain doesn't exist
        """
        # Verify domain exists
        await self.get_domain(domain_id)
        
        memberships = await self.db.list_memberships(domain_id)
        return [DomainMembership.from_dict(m) for m in memberships]
    
    async def get_member_role(
        self, domain_id: str, member_did: str
    ) -> Optional[DomainRole]:
        """Get a member's role in a domain.
        
        Args:
            domain_id: The domain UUID
            member_did: The member's DID
            
        Returns:
            The member's role, or None if not a member
        """
        membership = await self.db.get_membership(domain_id, member_did)
        if membership:
            return DomainRole(membership["role"])
        return None
    
    async def is_member(self, domain_id: str, member_did: str) -> bool:
        """Check if a DID is a member of a domain.
        
        Args:
            domain_id: The domain UUID
            member_did: The DID to check
            
        Returns:
            True if member, False otherwise
        """
        membership = await self.db.get_membership(domain_id, member_did)
        return membership is not None
    
    async def list_domains_for_member(self, member_did: str) -> List[Domain]:
        """List all domains a member belongs to.
        
        Args:
            member_did: The member's DID
            
        Returns:
            List of Domain objects
        """
        domains = await self.db.list_domains_for_member(member_did)
        return [Domain.from_dict(d) for d in domains]
    
    async def _check_can_manage_members(
        self, domain_id: str, requester_did: str
    ) -> None:
        """Check if requester can manage members.
        
        Only owners and admins can manage members.
        
        Raises:
            PermissionDeniedError: If requester lacks permission
        """
        role = await self.get_member_role(domain_id, requester_did)
        if role not in (DomainRole.OWNER, DomainRole.ADMIN):
            raise PermissionDeniedError(
                f"User {requester_did} cannot manage members in domain {domain_id}"
            )
    
    async def set_verification_requirement(
        self,
        domain_id: str,
        requirement: Optional[VerificationRequirement],
        requester_did: Optional[str] = None,
    ) -> Domain:
        """Set or clear the verification requirement for a domain.
        
        Only domain owners can modify verification requirements.
        
        Args:
            domain_id: The domain UUID
            requirement: Verification requirement (None to clear)
            requester_did: DID of the user making the request (for auth check)
            
        Returns:
            The updated Domain
            
        Raises:
            DomainNotFoundError: If domain doesn't exist
            PermissionDeniedError: If requester is not the owner
        """
        domain = await self.get_domain(domain_id)
        
        # Only owner can set verification requirements
        if requester_did and requester_did != domain.owner_did:
            raise PermissionDeniedError(
                f"Only the domain owner can set verification requirements"
            )
        
        await self.db.set_verification_requirement(
            domain_id=domain_id,
            requirement=requirement.to_dict() if requirement else None,
        )
        
        logger.info(
            f"Set verification requirement for domain {domain_id}: "
            f"{requirement.method.value if requirement else 'none'}"
        )
        
        # Return updated domain
        domain.verification_requirement = requirement
        return domain
    
    async def verify_membership(
        self,
        domain_id: str,
        member_did: str,
        evidence: Optional[Dict[str, Any]] = None,
        verifier: Optional[Verifier] = None,
    ) -> VerificationResult:
        """Verify a membership claim for a domain.
        
        Checks if the member_did can prove membership according to the
        domain's verification requirements.
        
        Args:
            domain_id: The domain UUID
            member_did: DID claiming membership
            evidence: Proof/credentials for verification
            verifier: Optional custom verifier (uses default based on method)
            
        Returns:
            VerificationResult indicating success/failure
            
        Raises:
            DomainNotFoundError: If domain doesn't exist
            MembershipNotFoundError: If not a member of the domain
        """
        domain = await self.get_domain(domain_id)
        
        # Check if actually a member first
        if not await self.is_member(domain_id, member_did):
            raise MembershipNotFoundError(
                f"Member {member_did} is not in domain {domain_id}"
            )
        
        requirement = domain.verification_requirement
        
        # If no verification required, auto-verify
        if not requirement or requirement.method == VerificationMethod.NONE:
            result = VerificationResult(
                verified=True,
                method=VerificationMethod.NONE,
                details="No verification required",
            )
            await self.db.store_verification_result(
                domain_id, member_did, result.to_dict()
            )
            return result
        
        # Get appropriate verifier
        if verifier is None:
            verifier = self._get_default_verifier(requirement.method)
        
        if verifier is None:
            result = VerificationResult(
                verified=False,
                method=requirement.method,
                details=f"No verifier available for method {requirement.method.value}",
            )
            return result
        
        # Perform verification
        result = await verifier.verify(
            domain_id=domain_id,
            member_did=member_did,
            requirement=requirement,
            evidence=evidence,
        )
        
        # Store result
        await self.db.store_verification_result(
            domain_id, member_did, result.to_dict()
        )
        
        logger.info(
            f"Verification for {member_did} in domain {domain_id}: "
            f"{'success' if result.verified else 'failed'}"
        )
        
        return result
    
    async def get_verification_status(
        self,
        domain_id: str,
        member_did: str,
    ) -> Optional[VerificationResult]:
        """Get the current verification status for a membership.
        
        Args:
            domain_id: The domain UUID
            member_did: The member's DID
            
        Returns:
            The latest VerificationResult, or None if never verified
        """
        data = await self.db.get_verification_result(domain_id, member_did)
        if data:
            return VerificationResult.from_dict(data)
        return None
    
    async def is_verified_member(
        self,
        domain_id: str,
        member_did: str,
    ) -> bool:
        """Check if a member has verified their membership.
        
        Args:
            domain_id: The domain UUID
            member_did: The member's DID
            
        Returns:
            True if member is verified (or no verification required)
        """
        domain = await self.get_domain(domain_id)
        
        # Check membership first
        if not await self.is_member(domain_id, member_did):
            return False
        
        # If no verification required, they're verified by default
        requirement = domain.verification_requirement
        if not requirement or requirement.method == VerificationMethod.NONE:
            return True
        
        # If verification is optional, membership is enough
        if not requirement.required:
            return True
        
        # Check verification status
        result = await self.get_verification_status(domain_id, member_did)
        return result is not None and result.verified
    
    def _get_default_verifier(self, method: VerificationMethod) -> Optional[Verifier]:
        """Get the default verifier for a verification method.
        
        Args:
            method: The verification method
            
        Returns:
            A Verifier instance, or None if not supported
        """
        if method == VerificationMethod.ADMIN_SIGNATURE:
            return AdminSignatureVerifier()
        elif method == VerificationMethod.DNS_TXT:
            return DNSTxtVerifier()
        return None
    
    def register_verifier(
        self,
        method: VerificationMethod,
        verifier: Verifier,
    ) -> None:
        """Register a custom verifier for a verification method.
        
        Args:
            method: The verification method to handle
            verifier: The Verifier implementation
        """
        if not hasattr(self, '_custom_verifiers'):
            self._custom_verifiers: Dict[VerificationMethod, Verifier] = {}
        self._custom_verifiers[method] = verifier
