"""Tests for Domain schema and DomainService."""

import pytest
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from valence.privacy.domains import (
    Domain,
    DomainMembership,
    DomainRole,
    DomainService,
    DomainError,
    DomainNotFoundError,
    DomainExistsError,
    MembershipExistsError,
    MembershipNotFoundError,
    PermissionDeniedError,
    VerificationMethod,
    VerificationRequirement,
    VerificationResult,
    VerificationError,
    VerificationRequiredError,
    AdminSignatureVerifier,
    DNSTxtVerifier,
)


class TestDomainRole:
    """Tests for DomainRole enum."""
    
    def test_all_roles_exist(self):
        """Verify all roles are defined."""
        assert DomainRole.OWNER.value == "owner"
        assert DomainRole.ADMIN.value == "admin"
        assert DomainRole.MEMBER.value == "member"
    
    def test_role_from_string(self):
        """Test creating role from string value."""
        assert DomainRole("owner") == DomainRole.OWNER
        assert DomainRole("admin") == DomainRole.ADMIN
        assert DomainRole("member") == DomainRole.MEMBER


class TestDomain:
    """Tests for Domain dataclass."""
    
    def test_create_domain(self):
        """Test basic domain creation."""
        domain = Domain(
            domain_id="test-uuid",
            name="research-team",
            owner_did="did:example:owner",
            description="A research team domain",
        )
        
        assert domain.domain_id == "test-uuid"
        assert domain.name == "research-team"
        assert domain.owner_did == "did:example:owner"
        assert domain.description == "A research team domain"
        assert domain.created_at is not None
    
    def test_domain_to_dict(self):
        """Test serialization to dict."""
        created = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        domain = Domain(
            domain_id="test-uuid",
            name="family",
            owner_did="did:example:alice",
            description="Family domain",
            created_at=created,
        )
        
        data = domain.to_dict()
        assert data["domain_id"] == "test-uuid"
        assert data["name"] == "family"
        assert data["owner_did"] == "did:example:alice"
        assert data["description"] == "Family domain"
        assert "2025-06-15" in data["created_at"]
    
    def test_domain_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "domain_id": "test-uuid",
            "name": "team",
            "owner_did": "did:example:bob",
            "description": None,
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        
        domain = Domain.from_dict(data)
        assert domain.domain_id == "test-uuid"
        assert domain.name == "team"
        assert domain.owner_did == "did:example:bob"
        assert domain.description is None
        assert domain.created_at.year == 2025
    
    def test_domain_roundtrip(self):
        """Test serialization roundtrip."""
        original = Domain(
            domain_id="roundtrip-id",
            name="test-domain",
            owner_did="did:example:test",
            description="Test description",
        )
        
        restored = Domain.from_dict(original.to_dict())
        assert restored.domain_id == original.domain_id
        assert restored.name == original.name
        assert restored.owner_did == original.owner_did
        assert restored.description == original.description


class TestDomainMembership:
    """Tests for DomainMembership dataclass."""
    
    def test_create_membership(self):
        """Test basic membership creation."""
        membership = DomainMembership(
            domain_id="domain-uuid",
            member_did="did:example:member",
            role=DomainRole.MEMBER,
        )
        
        assert membership.domain_id == "domain-uuid"
        assert membership.member_did == "did:example:member"
        assert membership.role == DomainRole.MEMBER
        assert membership.joined_at is not None
    
    def test_membership_to_dict(self):
        """Test serialization to dict."""
        joined = datetime(2025, 3, 20, 10, 30, 0, tzinfo=timezone.utc)
        membership = DomainMembership(
            domain_id="domain-uuid",
            member_did="did:example:admin",
            role=DomainRole.ADMIN,
            joined_at=joined,
        )
        
        data = membership.to_dict()
        assert data["domain_id"] == "domain-uuid"
        assert data["member_did"] == "did:example:admin"
        assert data["role"] == "admin"
        assert "2025-03-20" in data["joined_at"]
    
    def test_membership_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "domain_id": "domain-uuid",
            "member_did": "did:example:owner",
            "role": "owner",
            "joined_at": "2025-02-15T08:00:00+00:00",
        }
        
        membership = DomainMembership.from_dict(data)
        assert membership.domain_id == "domain-uuid"
        assert membership.member_did == "did:example:owner"
        assert membership.role == DomainRole.OWNER
        assert membership.joined_at.month == 2
    
    def test_membership_roundtrip(self):
        """Test serialization roundtrip."""
        original = DomainMembership(
            domain_id="roundtrip-domain",
            member_did="did:example:roundtrip",
            role=DomainRole.ADMIN,
        )
        
        restored = DomainMembership.from_dict(original.to_dict())
        assert restored.domain_id == original.domain_id
        assert restored.member_did == original.member_did
        assert restored.role == original.role


class MockDomainDatabase:
    """In-memory mock database for testing DomainService."""
    
    def __init__(self):
        self.domains: Dict[str, dict] = {}
        self.memberships: Dict[str, Dict[str, dict]] = {}  # domain_id -> member_did -> membership
        self.verification_results: Dict[str, Dict[str, dict]] = {}  # domain_id -> member_did -> result
    
    async def create_domain(
        self,
        domain_id: str,
        name: str,
        owner_did: str,
        description: Optional[str],
    ) -> None:
        self.domains[domain_id] = {
            "domain_id": domain_id,
            "name": name,
            "owner_did": owner_did,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "verification_requirement": None,
        }
        self.memberships[domain_id] = {}
        self.verification_results[domain_id] = {}
    
    async def get_domain(self, domain_id: str) -> Optional[dict]:
        return self.domains.get(domain_id)
    
    async def get_domain_by_name(self, name: str, owner_did: str) -> Optional[dict]:
        for domain in self.domains.values():
            if domain["name"] == name and domain["owner_did"] == owner_did:
                return domain
        return None
    
    async def delete_domain(self, domain_id: str) -> bool:
        if domain_id in self.domains:
            del self.domains[domain_id]
            if domain_id in self.memberships:
                del self.memberships[domain_id]
            return True
        return False
    
    async def add_membership(
        self,
        domain_id: str,
        member_did: str,
        role: str,
    ) -> None:
        if domain_id not in self.memberships:
            self.memberships[domain_id] = {}
        self.memberships[domain_id][member_did] = {
            "domain_id": domain_id,
            "member_did": member_did,
            "role": role,
            "joined_at": datetime.now(timezone.utc).isoformat(),
        }
    
    async def remove_membership(self, domain_id: str, member_did: str) -> bool:
        if domain_id in self.memberships and member_did in self.memberships[domain_id]:
            del self.memberships[domain_id][member_did]
            return True
        return False
    
    async def get_membership(
        self, domain_id: str, member_did: str
    ) -> Optional[dict]:
        if domain_id in self.memberships:
            return self.memberships[domain_id].get(member_did)
        return None
    
    async def list_memberships(self, domain_id: str) -> List[dict]:
        if domain_id in self.memberships:
            return list(self.memberships[domain_id].values())
        return []
    
    async def list_domains_for_member(self, member_did: str) -> List[dict]:
        result = []
        for domain_id, members in self.memberships.items():
            if member_did in members:
                domain = self.domains.get(domain_id)
                if domain:
                    result.append(domain)
        return result
    
    async def set_verification_requirement(
        self,
        domain_id: str,
        requirement: Optional[dict],
    ) -> None:
        if domain_id in self.domains:
            self.domains[domain_id]["verification_requirement"] = requirement
    
    async def store_verification_result(
        self,
        domain_id: str,
        member_did: str,
        result: dict,
    ) -> None:
        if domain_id not in self.verification_results:
            self.verification_results[domain_id] = {}
        self.verification_results[domain_id][member_did] = result
    
    async def get_verification_result(
        self,
        domain_id: str,
        member_did: str,
    ) -> Optional[dict]:
        if domain_id in self.verification_results:
            return self.verification_results[domain_id].get(member_did)
        return None


@pytest.fixture
def mock_db():
    """Create a mock database for testing."""
    return MockDomainDatabase()


@pytest.fixture
def domain_service(mock_db):
    """Create a DomainService with mock database."""
    return DomainService(mock_db)


class TestDomainService:
    """Tests for DomainService."""
    
    @pytest.mark.asyncio
    async def test_create_domain(self, domain_service):
        """Test creating a domain."""
        domain = await domain_service.create_domain(
            name="test-team",
            owner_did="did:example:alice",
            description="Test team",
        )
        
        assert domain.name == "test-team"
        assert domain.owner_did == "did:example:alice"
        assert domain.description == "Test team"
        assert domain.domain_id is not None
    
    @pytest.mark.asyncio
    async def test_create_domain_adds_owner_as_member(self, domain_service, mock_db):
        """Test that creating a domain automatically adds owner as member."""
        domain = await domain_service.create_domain(
            name="auto-member-test",
            owner_did="did:example:owner",
        )
        
        # Owner should be added as a member with OWNER role
        membership = await mock_db.get_membership(domain.domain_id, "did:example:owner")
        assert membership is not None
        assert membership["role"] == "owner"
    
    @pytest.mark.asyncio
    async def test_create_duplicate_domain_fails(self, domain_service):
        """Test that creating a domain with same name/owner fails."""
        await domain_service.create_domain(
            name="unique-name",
            owner_did="did:example:owner",
        )
        
        with pytest.raises(DomainExistsError):
            await domain_service.create_domain(
                name="unique-name",
                owner_did="did:example:owner",
            )
    
    @pytest.mark.asyncio
    async def test_same_name_different_owners_allowed(self, domain_service):
        """Test that different owners can have domains with the same name."""
        domain1 = await domain_service.create_domain(
            name="team",
            owner_did="did:example:alice",
        )
        domain2 = await domain_service.create_domain(
            name="team",
            owner_did="did:example:bob",
        )
        
        assert domain1.domain_id != domain2.domain_id
    
    @pytest.mark.asyncio
    async def test_get_domain(self, domain_service):
        """Test getting a domain by ID."""
        created = await domain_service.create_domain(
            name="get-test",
            owner_did="did:example:owner",
        )
        
        retrieved = await domain_service.get_domain(created.domain_id)
        assert retrieved.name == "get-test"
        assert retrieved.owner_did == "did:example:owner"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_domain_fails(self, domain_service):
        """Test that getting a non-existent domain raises error."""
        with pytest.raises(DomainNotFoundError):
            await domain_service.get_domain("nonexistent-uuid")
    
    @pytest.mark.asyncio
    async def test_add_member(self, domain_service):
        """Test adding a member to a domain."""
        domain = await domain_service.create_domain(
            name="member-test",
            owner_did="did:example:owner",
        )
        
        membership = await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:newmember",
            role=DomainRole.MEMBER,
        )
        
        assert membership.domain_id == domain.domain_id
        assert membership.member_did == "did:example:newmember"
        assert membership.role == DomainRole.MEMBER
    
    @pytest.mark.asyncio
    async def test_add_admin_member(self, domain_service):
        """Test adding an admin member."""
        domain = await domain_service.create_domain(
            name="admin-test",
            owner_did="did:example:owner",
        )
        
        membership = await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:admin",
            role=DomainRole.ADMIN,
        )
        
        assert membership.role == DomainRole.ADMIN
    
    @pytest.mark.asyncio
    async def test_add_duplicate_member_fails(self, domain_service):
        """Test that adding the same member twice fails."""
        domain = await domain_service.create_domain(
            name="dup-test",
            owner_did="did:example:owner",
        )
        
        await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:member",
        )
        
        with pytest.raises(MembershipExistsError):
            await domain_service.add_member(
                domain_id=domain.domain_id,
                member_did="did:example:member",
            )
    
    @pytest.mark.asyncio
    async def test_add_member_to_nonexistent_domain_fails(self, domain_service):
        """Test that adding member to non-existent domain fails."""
        with pytest.raises(DomainNotFoundError):
            await domain_service.add_member(
                domain_id="nonexistent",
                member_did="did:example:member",
            )
    
    @pytest.mark.asyncio
    async def test_remove_member(self, domain_service):
        """Test removing a member from a domain."""
        domain = await domain_service.create_domain(
            name="remove-test",
            owner_did="did:example:owner",
        )
        
        await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:member",
        )
        
        result = await domain_service.remove_member(
            domain_id=domain.domain_id,
            member_did="did:example:member",
        )
        
        assert result is True
        
        # Verify member is gone
        is_member = await domain_service.is_member(
            domain.domain_id, "did:example:member"
        )
        assert is_member is False
    
    @pytest.mark.asyncio
    async def test_cannot_remove_owner(self, domain_service):
        """Test that the domain owner cannot be removed."""
        domain = await domain_service.create_domain(
            name="owner-protect-test",
            owner_did="did:example:owner",
        )
        
        with pytest.raises(PermissionDeniedError):
            await domain_service.remove_member(
                domain_id=domain.domain_id,
                member_did="did:example:owner",
            )
    
    @pytest.mark.asyncio
    async def test_list_members(self, domain_service):
        """Test listing all members of a domain."""
        domain = await domain_service.create_domain(
            name="list-test",
            owner_did="did:example:owner",
        )
        
        await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:member1",
        )
        await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:member2",
            role=DomainRole.ADMIN,
        )
        
        members = await domain_service.list_members(domain.domain_id)
        
        # Should have owner + 2 members
        assert len(members) == 3
        
        member_dids = {m.member_did for m in members}
        assert "did:example:owner" in member_dids
        assert "did:example:member1" in member_dids
        assert "did:example:member2" in member_dids
    
    @pytest.mark.asyncio
    async def test_get_member_role(self, domain_service):
        """Test getting a member's role."""
        domain = await domain_service.create_domain(
            name="role-test",
            owner_did="did:example:owner",
        )
        
        await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:admin",
            role=DomainRole.ADMIN,
        )
        
        owner_role = await domain_service.get_member_role(
            domain.domain_id, "did:example:owner"
        )
        admin_role = await domain_service.get_member_role(
            domain.domain_id, "did:example:admin"
        )
        nonmember_role = await domain_service.get_member_role(
            domain.domain_id, "did:example:stranger"
        )
        
        assert owner_role == DomainRole.OWNER
        assert admin_role == DomainRole.ADMIN
        assert nonmember_role is None
    
    @pytest.mark.asyncio
    async def test_is_member(self, domain_service):
        """Test checking membership."""
        domain = await domain_service.create_domain(
            name="ismember-test",
            owner_did="did:example:owner",
        )
        
        await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:member",
        )
        
        assert await domain_service.is_member(domain.domain_id, "did:example:owner")
        assert await domain_service.is_member(domain.domain_id, "did:example:member")
        assert not await domain_service.is_member(domain.domain_id, "did:example:stranger")
    
    @pytest.mark.asyncio
    async def test_list_domains_for_member(self, domain_service):
        """Test listing domains a member belongs to."""
        domain1 = await domain_service.create_domain(
            name="domain1",
            owner_did="did:example:owner1",
        )
        domain2 = await domain_service.create_domain(
            name="domain2",
            owner_did="did:example:owner2",
        )
        
        # Add same member to both domains
        await domain_service.add_member(
            domain_id=domain1.domain_id,
            member_did="did:example:member",
        )
        await domain_service.add_member(
            domain_id=domain2.domain_id,
            member_did="did:example:member",
        )
        
        domains = await domain_service.list_domains_for_member("did:example:member")
        
        assert len(domains) == 2
        domain_names = {d.name for d in domains}
        assert "domain1" in domain_names
        assert "domain2" in domain_names
    
    @pytest.mark.asyncio
    async def test_permission_check_owner_can_manage(self, domain_service):
        """Test that owner can manage members."""
        domain = await domain_service.create_domain(
            name="perm-test",
            owner_did="did:example:owner",
        )
        
        # Owner should be able to add members
        membership = await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:newmember",
            requester_did="did:example:owner",
        )
        
        assert membership is not None
    
    @pytest.mark.asyncio
    async def test_permission_check_admin_can_manage(self, domain_service):
        """Test that admin can manage members."""
        domain = await domain_service.create_domain(
            name="admin-perm-test",
            owner_did="did:example:owner",
        )
        
        await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:admin",
            role=DomainRole.ADMIN,
        )
        
        # Admin should be able to add members
        membership = await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:newmember",
            requester_did="did:example:admin",
        )
        
        assert membership is not None
    
    @pytest.mark.asyncio
    async def test_permission_check_member_cannot_manage(self, domain_service):
        """Test that regular members cannot manage other members."""
        domain = await domain_service.create_domain(
            name="member-perm-test",
            owner_did="did:example:owner",
        )
        
        await domain_service.add_member(
            domain_id=domain.domain_id,
            member_did="did:example:member",
            role=DomainRole.MEMBER,
        )
        
        # Regular member should NOT be able to add members
        with pytest.raises(PermissionDeniedError):
            await domain_service.add_member(
                domain_id=domain.domain_id,
                member_did="did:example:unauthorized",
                requester_did="did:example:member",
            )
