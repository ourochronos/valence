"""Domain management for Valence.

Domains are named scopes for trust (e.g., "research-team", "family").
They provide a way to organize members and control access to beliefs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Protocol, Any
import uuid
import logging

logger = logging.getLogger(__name__)


class DomainRole(Enum):
    """Roles within a domain."""
    
    OWNER = "owner"    # Full control, can delete domain
    ADMIN = "admin"    # Can manage members
    MEMBER = "member"  # Basic access


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
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "domain_id": self.domain_id,
            "name": self.name,
            "owner_did": self.owner_did,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Domain":
        """Deserialize from dictionary."""
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
            
        return cls(
            domain_id=data["domain_id"],
            name=data["name"],
            owner_did=data["owner_did"],
            description=data.get("description"),
            created_at=created_at,
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
