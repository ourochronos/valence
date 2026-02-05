"""Tests for cross-federation consent chains.

Tests cover:
- CrossFederationHop creation and serialization
- FederationConsentPolicy enforcement
- CrossFederationConsentChain management
- Consent validation across federation boundaries
- Revocation propagation across federations
- Provenance preservation

Issue #89
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

# Import directly from module to avoid broken __init__.py imports (Issue #X - fix groups exports)
from valence.federation.consent import (
    ConsentValidationResult,
    CrossFederationConsentChain,
    # Service
    CrossFederationConsentService,
    # Data classes
    CrossFederationHop,
    # Enums
    CrossFederationPolicy,
    FederationConsentPolicy,
    # In-memory implementations
    InMemoryConsentChainStore,
    InMemoryPolicyStore,
    MockGatewaySigner,
    RevocationScope,
    # Helper functions (Issue #145)
    compute_policy_hash,
    verify_policy_hash,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def gateway_signer_a() -> MockGatewaySigner:
    """Gateway signer for federation A."""
    return MockGatewaySigner("gateway-a")


@pytest.fixture
def gateway_signer_b() -> MockGatewaySigner:
    """Gateway signer for federation B."""
    return MockGatewaySigner("gateway-b")


@pytest.fixture
def gateway_signer_c() -> MockGatewaySigner:
    """Gateway signer for federation C."""
    return MockGatewaySigner("gateway-c")


@pytest.fixture
def chain_store_a() -> InMemoryConsentChainStore:
    """In-memory consent chain store for federation A."""
    return InMemoryConsentChainStore()


@pytest.fixture
def chain_store_b() -> InMemoryConsentChainStore:
    """In-memory consent chain store for federation B."""
    return InMemoryConsentChainStore()


@pytest.fixture
def chain_store_c() -> InMemoryConsentChainStore:
    """In-memory consent chain store for federation C."""
    return InMemoryConsentChainStore()


@pytest.fixture
def chain_store(chain_store_a: InMemoryConsentChainStore) -> InMemoryConsentChainStore:
    """Default chain store (alias for A) for single-federation tests."""
    return chain_store_a


@pytest.fixture
def policy_store() -> InMemoryPolicyStore:
    """Shared policy store (in practice policies are known across federations)."""
    return InMemoryPolicyStore()


@pytest.fixture
def consent_service_a(
    gateway_signer_a: MockGatewaySigner,
    chain_store_a: InMemoryConsentChainStore,
    policy_store: InMemoryPolicyStore,
) -> CrossFederationConsentService:
    """Consent service for federation A."""
    return CrossFederationConsentService(
        federation_id="federation-a",
        gateway_signer=gateway_signer_a,
        chain_store=chain_store_a,
        policy_store=policy_store,
    )


@pytest.fixture
def consent_service_b(
    gateway_signer_b: MockGatewaySigner,
    chain_store_b: InMemoryConsentChainStore,
    policy_store: InMemoryPolicyStore,
) -> CrossFederationConsentService:
    """Consent service for federation B."""
    return CrossFederationConsentService(
        federation_id="federation-b",
        gateway_signer=gateway_signer_b,
        chain_store=chain_store_b,
        policy_store=policy_store,
    )


@pytest.fixture
def consent_service_c(
    gateway_signer_c: MockGatewaySigner,
    chain_store_c: InMemoryConsentChainStore,
    policy_store: InMemoryPolicyStore,
) -> CrossFederationConsentService:
    """Consent service for federation C."""
    return CrossFederationConsentService(
        federation_id="federation-c",
        gateway_signer=gateway_signer_c,
        chain_store=chain_store_c,
        policy_store=policy_store,
    )


# =============================================================================
# CROSS FEDERATION HOP TESTS
# =============================================================================


class TestPolicyHashVerification:
    """Tests for policy snapshot hash verification (Issue #145)."""

    def test_compute_policy_hash(self) -> None:
        """Test computing policy hash."""
        policy = {"max_hops": 3, "federation_id": "fed-a"}
        hash1 = compute_policy_hash(policy)
        hash2 = compute_policy_hash(policy)

        assert hash1 == hash2
        assert len(hash1) == 32  # SHA-256

    def test_policy_hash_deterministic(self) -> None:
        """Test that hash is deterministic regardless of key order."""
        policy1 = {"b": 2, "a": 1}
        policy2 = {"a": 1, "b": 2}

        assert compute_policy_hash(policy1) == compute_policy_hash(policy2)

    def test_policy_hash_different_for_different_policies(self) -> None:
        """Test that different policies produce different hashes."""
        policy1 = {"max_hops": 3}
        policy2 = {"max_hops": 5}

        assert compute_policy_hash(policy1) != compute_policy_hash(policy2)

    def test_verify_policy_hash_valid(self) -> None:
        """Test verifying a valid policy hash."""
        policy = {"federation_id": "fed-a", "max_hops": 3}
        policy_hash = compute_policy_hash(policy)

        assert verify_policy_hash(policy, policy_hash)

    def test_verify_policy_hash_invalid(self) -> None:
        """Test detecting a tampered policy."""
        original_policy = {"max_hops": 3}
        tampered_policy = {"max_hops": 10}  # Attacker changed it

        # Hash was computed on original
        policy_hash = compute_policy_hash(original_policy)

        # Verification on tampered should fail
        assert not verify_policy_hash(tampered_policy, policy_hash)

    def test_empty_policy_hash(self) -> None:
        """Test hashing empty policy."""
        policy = {}
        policy_hash = compute_policy_hash(policy)

        assert verify_policy_hash(policy, policy_hash)
        assert len(policy_hash) == 32


class TestCrossFederationHop:
    """Tests for CrossFederationHop data class."""

    def test_create_hop(self) -> None:
        """Test creating a cross-federation hop."""
        hop = CrossFederationHop(
            hop_id="hop-1",
            from_federation_id="fed-a",
            from_gateway_id="gateway-a",
            to_federation_id="fed-b",
            to_gateway_id="gateway-b",
            timestamp=time.time(),
            signature=b"test-signature",
            original_consent_chain_id="chain-1",
            hop_number=1,
        )

        assert hop.hop_id == "hop-1"
        assert hop.from_federation_id == "fed-a"
        assert hop.to_federation_id == "fed-b"
        assert hop.hop_number == 1

    def test_hop_with_policy_hash(self) -> None:
        """Test creating a hop with policy hash (Issue #145)."""
        policy_snapshot = {"max_hops": 3, "federation_id": "fed-a"}
        policy_hash = compute_policy_hash(policy_snapshot)

        hop = CrossFederationHop(
            hop_id="hop-1",
            from_federation_id="fed-a",
            from_gateway_id="gateway-a",
            to_federation_id="fed-b",
            to_gateway_id="gateway-b",
            timestamp=time.time(),
            signature=b"test-signature",
            original_consent_chain_id="chain-1",
            hop_number=1,
            policy_snapshot=policy_snapshot,
            policy_hash=policy_hash,
        )

        assert hop.policy_hash == policy_hash
        assert verify_policy_hash(hop.policy_snapshot, hop.policy_hash)

    def test_hop_serialization(self) -> None:
        """Test hop serialization/deserialization."""
        hop = CrossFederationHop(
            hop_id="hop-1",
            from_federation_id="fed-a",
            from_gateway_id="gateway-a",
            to_federation_id="fed-b",
            to_gateway_id="gateway-b",
            timestamp=1234567890.0,
            signature=b"test-signature",
            original_consent_chain_id="chain-1",
            hop_number=1,
            policy_snapshot={"max_hops": 3},
            reason="Data sharing agreement",
            requester_did="did:vkb:user:alice",
        )

        data = hop.to_dict()
        restored = CrossFederationHop.from_dict(data)

        assert restored.hop_id == hop.hop_id
        assert restored.from_federation_id == hop.from_federation_id
        assert restored.to_federation_id == hop.to_federation_id
        assert restored.signature == hop.signature
        assert restored.policy_snapshot == hop.policy_snapshot
        assert restored.reason == hop.reason
        assert restored.requester_did == hop.requester_did

    def test_hop_serialization_with_policy_hash(self) -> None:
        """Test hop serialization preserves policy hash (Issue #145)."""
        policy_snapshot = {"max_hops": 3}
        policy_hash = compute_policy_hash(policy_snapshot)

        hop = CrossFederationHop(
            hop_id="hop-1",
            from_federation_id="fed-a",
            from_gateway_id="gateway-a",
            to_federation_id="fed-b",
            to_gateway_id="gateway-b",
            timestamp=1234567890.0,
            signature=b"test-signature",
            original_consent_chain_id="chain-1",
            hop_number=1,
            policy_snapshot=policy_snapshot,
            policy_hash=policy_hash,
        )

        data = hop.to_dict()
        restored = CrossFederationHop.from_dict(data)

        assert restored.policy_hash == policy_hash
        assert verify_policy_hash(restored.policy_snapshot, restored.policy_hash)


# =============================================================================
# FEDERATION CONSENT POLICY TESTS
# =============================================================================


class TestFederationConsentPolicy:
    """Tests for FederationConsentPolicy."""

    def test_create_default_policy(self) -> None:
        """Test creating a policy with defaults."""
        policy = FederationConsentPolicy(federation_id="fed-a")

        assert policy.outgoing_policy == CrossFederationPolicy.ALLOW_TRUSTED
        assert policy.incoming_policy == CrossFederationPolicy.ALLOW_TRUSTED
        assert policy.min_trust_for_outgoing == 0.5
        assert policy.max_outgoing_hops == 3

    def test_create_restrictive_policy(self) -> None:
        """Test creating a restrictive policy."""
        policy = FederationConsentPolicy(
            federation_id="fed-secure",
            outgoing_policy=CrossFederationPolicy.ALLOW_LISTED,
            allowed_outgoing_federations=["fed-partner-1", "fed-partner-2"],
            incoming_policy=CrossFederationPolicy.DENY_ALL,
            min_trust_for_outgoing=0.8,
            max_outgoing_hops=1,
            strip_fields_on_outgoing=["metadata.internal"],
            revocation_scope=RevocationScope.FULL_CHAIN,
        )

        assert policy.outgoing_policy == CrossFederationPolicy.ALLOW_LISTED
        assert len(policy.allowed_outgoing_federations) == 2
        assert policy.incoming_policy == CrossFederationPolicy.DENY_ALL
        assert policy.max_outgoing_hops == 1

    def test_policy_serialization(self) -> None:
        """Test policy serialization/deserialization."""
        policy = FederationConsentPolicy(
            federation_id="fed-a",
            outgoing_policy=CrossFederationPolicy.ALLOW_LISTED,
            allowed_outgoing_federations=["fed-b"],
            blocked_incoming_federations=["fed-untrusted"],
            min_trust_for_outgoing=0.7,
        )

        data = policy.to_dict()
        restored = FederationConsentPolicy.from_dict(data)

        assert restored.federation_id == policy.federation_id
        assert restored.outgoing_policy == policy.outgoing_policy
        assert restored.allowed_outgoing_federations == policy.allowed_outgoing_federations
        assert restored.blocked_incoming_federations == policy.blocked_incoming_federations
        assert restored.min_trust_for_outgoing == policy.min_trust_for_outgoing


# =============================================================================
# CROSS FEDERATION CONSENT CHAIN TESTS
# =============================================================================


class TestCrossFederationConsentChain:
    """Tests for CrossFederationConsentChain."""

    def test_create_chain(self) -> None:
        """Test creating a consent chain."""
        chain = CrossFederationConsentChain(
            id="chain-1",
            original_chain_id="original-chain-1",
            origin_federation_id="fed-a",
            origin_gateway_id="gateway-a",
        )

        assert chain.id == "chain-1"
        assert chain.origin_federation_id == "fed-a"
        assert chain.get_hop_count() == 0
        assert chain.get_current_federation() == "fed-a"
        assert not chain.revoked

    def test_add_hop(self) -> None:
        """Test adding a hop to the chain."""
        chain = CrossFederationConsentChain(
            id="chain-1",
            original_chain_id="original-chain-1",
            origin_federation_id="fed-a",
            origin_gateway_id="gateway-a",
        )

        hop = CrossFederationHop(
            hop_id="hop-1",
            from_federation_id="fed-a",
            from_gateway_id="gateway-a",
            to_federation_id="fed-b",
            to_gateway_id="gateway-b",
            timestamp=time.time(),
            signature=b"sig",
            original_consent_chain_id="original-chain-1",
            hop_number=1,
        )

        chain.add_hop(hop)

        assert chain.get_hop_count() == 1
        assert chain.get_current_federation() == "fed-b"
        assert chain.get_federation_path() == ["fed-a", "fed-b"]
        assert len(chain.provenance_chain) == 1

    def test_federation_path_multi_hop(self) -> None:
        """Test federation path with multiple hops."""
        chain = CrossFederationConsentChain(
            id="chain-1",
            original_chain_id="original-chain-1",
            origin_federation_id="fed-a",
            origin_gateway_id="gateway-a",
        )

        # Add hop A -> B
        chain.add_hop(
            CrossFederationHop(
                hop_id="hop-1",
                from_federation_id="fed-a",
                from_gateway_id="gateway-a",
                to_federation_id="fed-b",
                to_gateway_id="gateway-b",
                timestamp=time.time(),
                signature=b"sig1",
                original_consent_chain_id="original-chain-1",
                hop_number=1,
            )
        )

        # Add hop B -> C
        chain.add_hop(
            CrossFederationHop(
                hop_id="hop-2",
                from_federation_id="fed-b",
                from_gateway_id="gateway-b",
                to_federation_id="fed-c",
                to_gateway_id="gateway-c",
                timestamp=time.time(),
                signature=b"sig2",
                original_consent_chain_id="original-chain-1",
                hop_number=2,
            )
        )

        assert chain.get_hop_count() == 2
        assert chain.get_federation_path() == ["fed-a", "fed-b", "fed-c"]
        assert chain.get_current_federation() == "fed-c"
        assert chain.get_current_gateway() == "gateway-c"

    def test_chain_serialization(self) -> None:
        """Test chain serialization/deserialization."""
        chain = CrossFederationConsentChain(
            id="chain-1",
            original_chain_id="original-chain-1",
            origin_federation_id="fed-a",
            origin_gateway_id="gateway-a",
        )

        chain.add_hop(
            CrossFederationHop(
                hop_id="hop-1",
                from_federation_id="fed-a",
                from_gateway_id="gateway-a",
                to_federation_id="fed-b",
                to_gateway_id="gateway-b",
                timestamp=1234567890.0,
                signature=b"sig",
                original_consent_chain_id="original-chain-1",
                hop_number=1,
            )
        )

        data = chain.to_dict()
        restored = CrossFederationConsentChain.from_dict(data)

        assert restored.id == chain.id
        assert restored.origin_federation_id == chain.origin_federation_id
        assert len(restored.cross_federation_hops) == 1
        assert restored.get_federation_path() == chain.get_federation_path()


# =============================================================================
# CONSENT SERVICE TESTS
# =============================================================================


class TestCrossFederationConsentService:
    """Tests for CrossFederationConsentService."""

    @pytest.mark.asyncio
    async def test_create_cross_federation_hop(
        self,
        consent_service_a: CrossFederationConsentService,
        chain_store_a: InMemoryConsentChainStore,
    ) -> None:
        """Test creating a cross-federation hop."""
        original_chain_id = str(uuid4())

        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
            requester_did="did:vkb:user:alice",
            reason="Data sharing agreement",
        )

        assert hop.from_federation_id == "federation-a"
        assert hop.to_federation_id == "federation-b"
        assert hop.hop_number == 1
        assert hop.requester_did == "did:vkb:user:alice"
        assert hop.signature is not None

        # Verify chain was stored in federation A's store
        chain = await chain_store_a.get_cross_chain_by_original(original_chain_id)
        assert chain is not None
        assert chain.get_hop_count() == 1

    @pytest.mark.asyncio
    async def test_receive_cross_federation_hop(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        gateway_signer_a: MockGatewaySigner,
    ) -> None:
        """Test receiving a cross-federation hop."""
        original_chain_id = str(uuid4())

        # Create hop from A
        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )

        # Receive in B
        chain = await consent_service_b.receive_cross_federation_hop(
            hop=hop,
            source_gateway_verifier=gateway_signer_a,
        )

        assert chain.get_current_federation() == "federation-b"
        assert len(chain.cross_federation_hops) == 1

    @pytest.mark.asyncio
    async def test_receive_hop_wrong_federation(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_c: CrossFederationConsentService,
    ) -> None:
        """Test receiving a hop directed to a different federation."""
        original_chain_id = str(uuid4())

        # Create hop from A to B
        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )

        # Try to receive in C (should fail)
        with pytest.raises(ValueError, match="not directed to this federation"):
            await consent_service_c.receive_cross_federation_hop(hop=hop)

    @pytest.mark.asyncio
    async def test_hop_includes_policy_hash(
        self,
        consent_service_a: CrossFederationConsentService,
        policy_store: InMemoryPolicyStore,
    ) -> None:
        """Test that created hops include policy hash (Issue #145)."""
        # Set up a policy
        policy = FederationConsentPolicy(
            federation_id="federation-a",
            max_outgoing_hops=5,
            min_trust_for_outgoing=0.7,
        )
        await policy_store.store_policy(policy)

        original_chain_id = str(uuid4())
        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )

        # Verify hash is set and matches snapshot
        assert hop.policy_hash != b""
        assert verify_policy_hash(hop.policy_snapshot, hop.policy_hash)

    @pytest.mark.asyncio
    async def test_receive_hop_detects_tampered_policy(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        gateway_signer_a: MockGatewaySigner,
        policy_store: InMemoryPolicyStore,
    ) -> None:
        """Test that tampered policy snapshots are detected (Issue #145)."""
        # Set up a policy
        policy = FederationConsentPolicy(
            federation_id="federation-a",
            max_outgoing_hops=3,
        )
        await policy_store.store_policy(policy)

        original_chain_id = str(uuid4())
        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )

        # Attacker tampers with the policy snapshot
        hop.policy_snapshot["max_outgoing_hops"] = 100  # Changed!

        # Receiver should detect tampering
        with pytest.raises(ValueError, match="Policy snapshot hash mismatch"):
            await consent_service_b.receive_cross_federation_hop(
                hop=hop,
                source_gateway_verifier=gateway_signer_a,
            )

    @pytest.mark.asyncio
    async def test_receive_hop_accepts_valid_policy_hash(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        gateway_signer_a: MockGatewaySigner,
        policy_store: InMemoryPolicyStore,
    ) -> None:
        """Test that valid policy hashes are accepted (Issue #145)."""
        # Set up a policy
        policy = FederationConsentPolicy(
            federation_id="federation-a",
            max_outgoing_hops=5,
        )
        await policy_store.store_policy(policy)

        original_chain_id = str(uuid4())
        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )

        # Should succeed with valid hash
        chain = await consent_service_b.receive_cross_federation_hop(
            hop=hop,
            source_gateway_verifier=gateway_signer_a,
        )

        assert chain is not None
        assert len(chain.cross_federation_hops) == 1

    @pytest.mark.asyncio
    async def test_multi_hop_chain(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        consent_service_c: CrossFederationConsentService,
        gateway_signer_a: MockGatewaySigner,
        gateway_signer_b: MockGatewaySigner,
        chain_store_b: InMemoryConsentChainStore,
    ) -> None:
        """Test a chain with multiple hops across federations."""
        original_chain_id = str(uuid4())

        # A -> B
        hop1 = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )
        await consent_service_b.receive_cross_federation_hop(
            hop=hop1,
            source_gateway_verifier=gateway_signer_a,
        )

        # B -> C
        hop2 = await consent_service_b.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-c",
            target_gateway_id="gateway-c",
        )
        chain_c = await consent_service_c.receive_cross_federation_hop(
            hop=hop2,
            source_gateway_verifier=gateway_signer_b,
        )

        # C only knows about the hop it received (hop 2)
        # This is correct for a distributed system - each federation
        # only knows its local view
        assert chain_c.get_hop_count() == 1
        assert hop2.hop_number == 2  # But the hop number is still 2

        # B knows about both hops (it participated in both)
        chain_b = await chain_store_b.get_cross_chain_by_original(original_chain_id)
        assert chain_b.get_hop_count() == 2
        assert chain_b.get_federation_path() == [
            "federation-a",
            "federation-b",
            "federation-c",
        ]

    @pytest.mark.asyncio
    async def test_validate_consent_chain_valid(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        gateway_signer_a: MockGatewaySigner,
    ) -> None:
        """Test validating a valid consent chain."""
        original_chain_id = str(uuid4())

        # Create hop
        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )
        await consent_service_b.receive_cross_federation_hop(
            hop=hop,
            source_gateway_verifier=gateway_signer_a,
        )

        # Validate in B
        validation = await consent_service_b.validate_consent_chain(original_chain_id)

        assert validation.is_valid()
        assert validation.result == ConsentValidationResult.VALID
        assert validation.valid_hops == 1

    @pytest.mark.asyncio
    async def test_validate_consent_chain_not_found(
        self,
        consent_service_a: CrossFederationConsentService,
    ) -> None:
        """Test validating a non-existent chain."""
        validation = await consent_service_a.validate_consent_chain("nonexistent")

        assert not validation.is_valid()
        assert validation.result == ConsentValidationResult.BROKEN_CHAIN

    @pytest.mark.asyncio
    async def test_validate_consent_chain_revoked(
        self,
        consent_service_a: CrossFederationConsentService,
    ) -> None:
        """Test validating a revoked chain."""
        original_chain_id = str(uuid4())

        # Create and then revoke
        await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )

        await consent_service_a.revoke_cross_federation(
            chain_id=original_chain_id,
            revoker_did="did:vkb:admin:alice",
            reason="No longer needed",
        )

        validation = await consent_service_a.validate_consent_chain(original_chain_id)

        assert not validation.is_valid()
        assert validation.result == ConsentValidationResult.REVOKED


# =============================================================================
# POLICY ENFORCEMENT TESTS
# =============================================================================


class TestPolicyEnforcement:
    """Tests for federation-level policy enforcement."""

    @pytest.mark.asyncio
    async def test_outgoing_blocked_federation(
        self,
        consent_service_a: CrossFederationConsentService,
        policy_store: InMemoryPolicyStore,
    ) -> None:
        """Test that blocked outgoing federations are rejected."""
        # Set up policy blocking federation-b
        policy = FederationConsentPolicy(
            federation_id="federation-a",
            outgoing_policy=CrossFederationPolicy.ALLOW_ALL,
            blocked_outgoing_federations=["federation-b"],
        )
        await policy_store.store_policy(policy)

        # Try to create hop to blocked federation
        with pytest.raises(PermissionError, match="blocked"):
            await consent_service_a.create_cross_federation_hop(
                original_chain_id=str(uuid4()),
                target_federation_id="federation-b",
                target_gateway_id="gateway-b",
            )

    @pytest.mark.asyncio
    async def test_outgoing_deny_all(
        self,
        consent_service_a: CrossFederationConsentService,
        policy_store: InMemoryPolicyStore,
    ) -> None:
        """Test DENY_ALL outgoing policy."""
        policy = FederationConsentPolicy(
            federation_id="federation-a",
            outgoing_policy=CrossFederationPolicy.DENY_ALL,
        )
        await policy_store.store_policy(policy)

        with pytest.raises(PermissionError, match="blocked"):
            await consent_service_a.create_cross_federation_hop(
                original_chain_id=str(uuid4()),
                target_federation_id="federation-b",
                target_gateway_id="gateway-b",
            )

    @pytest.mark.asyncio
    async def test_outgoing_allow_listed(
        self,
        consent_service_a: CrossFederationConsentService,
        policy_store: InMemoryPolicyStore,
    ) -> None:
        """Test ALLOW_LISTED outgoing policy."""
        policy = FederationConsentPolicy(
            federation_id="federation-a",
            outgoing_policy=CrossFederationPolicy.ALLOW_LISTED,
            allowed_outgoing_federations=["federation-c"],  # B not in list
        )
        await policy_store.store_policy(policy)

        # Federation-b not in allowed list
        with pytest.raises(PermissionError, match="not in allowed"):
            await consent_service_a.create_cross_federation_hop(
                original_chain_id=str(uuid4()),
                target_federation_id="federation-b",
                target_gateway_id="gateway-b",
            )

        # Federation-c is allowed
        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=str(uuid4()),
            target_federation_id="federation-c",
            target_gateway_id="gateway-c",
        )
        assert hop.to_federation_id == "federation-c"

    @pytest.mark.asyncio
    async def test_incoming_blocked_federation(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        policy_store: InMemoryPolicyStore,
    ) -> None:
        """Test that blocked incoming federations are rejected."""
        # Set up policy in B blocking A
        policy = FederationConsentPolicy(
            federation_id="federation-b",
            incoming_policy=CrossFederationPolicy.ALLOW_ALL,
            blocked_incoming_federations=["federation-a"],
        )
        await policy_store.store_policy(policy)

        # Create hop from A
        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=str(uuid4()),
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )

        # B should reject it
        with pytest.raises(PermissionError, match="blocked"):
            await consent_service_b.receive_cross_federation_hop(hop=hop)

    @pytest.mark.asyncio
    async def test_incoming_deny_all(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        policy_store: InMemoryPolicyStore,
    ) -> None:
        """Test DENY_ALL incoming policy."""
        policy = FederationConsentPolicy(
            federation_id="federation-b",
            incoming_policy=CrossFederationPolicy.DENY_ALL,
        )
        await policy_store.store_policy(policy)

        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=str(uuid4()),
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )

        with pytest.raises(PermissionError, match="blocked"):
            await consent_service_b.receive_cross_federation_hop(hop=hop)

    @pytest.mark.asyncio
    async def test_validate_blocked_federation_in_chain(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        policy_store: InMemoryPolicyStore,
        gateway_signer_a: MockGatewaySigner,
    ) -> None:
        """Test validation fails when chain traverses a blocked federation."""
        original_chain_id = str(uuid4())

        # Create hop first (before policy)
        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )
        await consent_service_b.receive_cross_federation_hop(
            hop=hop,
            source_gateway_verifier=gateway_signer_a,
        )

        # Now add blocking policy
        policy = FederationConsentPolicy(
            federation_id="federation-a",
            blocked_outgoing_federations=["federation-b"],
        )
        await policy_store.store_policy(policy)

        # Validation should fail (checked from B's perspective)
        validation = await consent_service_b.validate_consent_chain(original_chain_id)

        assert not validation.is_valid()
        assert validation.result == ConsentValidationResult.FEDERATION_BLOCKED


# =============================================================================
# REVOCATION TESTS
# =============================================================================


class TestCrossFederationRevocation:
    """Tests for revocation across federations."""

    @pytest.mark.asyncio
    async def test_revoke_chain(
        self,
        consent_service_a: CrossFederationConsentService,
        chain_store_a: InMemoryConsentChainStore,
    ) -> None:
        """Test basic chain revocation."""
        original_chain_id = str(uuid4())

        await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )

        revocation = await consent_service_a.revoke_cross_federation(
            chain_id=original_chain_id,
            revoker_did="did:vkb:admin:alice",
            reason="Security concern",
        )

        assert revocation.revoked_by == "did:vkb:admin:alice"
        assert revocation.reason == "Security concern"
        assert revocation.revoked_in_federation == "federation-a"

        # Verify chain is marked revoked in A's store
        chain = await chain_store_a.get_cross_chain_by_original(original_chain_id)
        assert chain.revoked

    @pytest.mark.asyncio
    async def test_revoke_already_revoked(
        self,
        consent_service_a: CrossFederationConsentService,
    ) -> None:
        """Test revoking an already-revoked chain."""
        original_chain_id = str(uuid4())

        await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )

        await consent_service_a.revoke_cross_federation(
            chain_id=original_chain_id,
            revoker_did="did:vkb:admin:alice",
        )

        with pytest.raises(ValueError, match="already revoked"):
            await consent_service_a.revoke_cross_federation(
                chain_id=original_chain_id,
                revoker_did="did:vkb:admin:bob",
            )

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_chain(
        self,
        consent_service_a: CrossFederationConsentService,
    ) -> None:
        """Test revoking a non-existent chain."""
        with pytest.raises(ValueError, match="not found"):
            await consent_service_a.revoke_cross_federation(
                chain_id="nonexistent",
                revoker_did="did:vkb:admin:alice",
            )

    @pytest.mark.asyncio
    async def test_revocation_scope_downstream(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        consent_service_c: CrossFederationConsentService,
        gateway_signer_a: MockGatewaySigner,
        gateway_signer_b: MockGatewaySigner,
    ) -> None:
        """Test downstream revocation scope."""
        original_chain_id = str(uuid4())

        # Create chain: A -> B -> C
        hop1 = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )
        await consent_service_b.receive_cross_federation_hop(
            hop=hop1,
            source_gateway_verifier=gateway_signer_a,
        )

        hop2 = await consent_service_b.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-c",
            target_gateway_id="gateway-c",
        )
        await consent_service_c.receive_cross_federation_hop(
            hop=hop2,
            source_gateway_verifier=gateway_signer_b,
        )

        # Revoke from B with downstream scope
        revocation = await consent_service_b.revoke_cross_federation(
            chain_id=original_chain_id,
            revoker_did="did:vkb:admin:bob",
            scope=RevocationScope.DOWNSTREAM,
        )

        # Only C should be in pending propagation (downstream of B)
        assert "federation-c" in revocation.pending_propagation
        assert "federation-a" not in revocation.pending_propagation

    @pytest.mark.asyncio
    async def test_revocation_scope_full_chain(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        consent_service_c: CrossFederationConsentService,
        gateway_signer_a: MockGatewaySigner,
        gateway_signer_b: MockGatewaySigner,
    ) -> None:
        """Test full chain revocation scope."""
        original_chain_id = str(uuid4())

        # Create chain: A -> B -> C
        hop1 = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )
        await consent_service_b.receive_cross_federation_hop(
            hop=hop1,
            source_gateway_verifier=gateway_signer_a,
        )

        hop2 = await consent_service_b.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-c",
            target_gateway_id="gateway-c",
        )
        await consent_service_c.receive_cross_federation_hop(
            hop=hop2,
            source_gateway_verifier=gateway_signer_b,
        )

        # Revoke from B with full chain scope
        revocation = await consent_service_b.revoke_cross_federation(
            chain_id=original_chain_id,
            revoker_did="did:vkb:admin:bob",
            scope=RevocationScope.FULL_CHAIN,
        )

        # Both A and C should be in pending propagation
        assert "federation-a" in revocation.pending_propagation
        assert "federation-c" in revocation.pending_propagation

    @pytest.mark.asyncio
    async def test_receive_revocation(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        gateway_signer_a: MockGatewaySigner,
        chain_store_b: InMemoryConsentChainStore,
    ) -> None:
        """Test receiving and processing a revocation."""
        original_chain_id = str(uuid4())

        # Create chain
        hop = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )
        await consent_service_b.receive_cross_federation_hop(
            hop=hop,
            source_gateway_verifier=gateway_signer_a,
        )

        # Revoke from A
        revocation = await consent_service_a.revoke_cross_federation(
            chain_id=original_chain_id,
            revoker_did="did:vkb:admin:alice",
        )

        # B receives revocation
        await consent_service_b.receive_revocation(revocation)

        # Verify B's chain is revoked in B's store
        chain = await chain_store_b.get_cross_chain_by_original(original_chain_id)
        assert chain.revoked
        assert chain.revoked_by == "did:vkb:admin:alice"

        # Verify acknowledgment
        assert "federation-b" in revocation.acknowledgments

    @pytest.mark.asyncio
    async def test_cannot_create_hop_on_revoked_chain(
        self,
        consent_service_a: CrossFederationConsentService,
    ) -> None:
        """Test that hop creation fails on revoked chain."""
        original_chain_id = str(uuid4())

        # Create initial hop
        await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )

        # Revoke
        await consent_service_a.revoke_cross_federation(
            chain_id=original_chain_id,
            revoker_did="did:vkb:admin:alice",
        )

        # Try to add another hop
        with pytest.raises(ValueError, match="revoked"):
            await consent_service_a.create_cross_federation_hop(
                original_chain_id=original_chain_id,
                target_federation_id="federation-c",
                target_gateway_id="gateway-c",
            )


# =============================================================================
# PROVENANCE TESTS
# =============================================================================


class TestProvenance:
    """Tests for provenance preservation across federations."""

    @pytest.mark.asyncio
    async def test_get_provenance(
        self,
        consent_service_a: CrossFederationConsentService,
        consent_service_b: CrossFederationConsentService,
        consent_service_c: CrossFederationConsentService,
        gateway_signer_a: MockGatewaySigner,
        gateway_signer_b: MockGatewaySigner,
    ) -> None:
        """Test retrieving provenance chain."""
        original_chain_id = str(uuid4())

        # Create chain: A -> B -> C
        hop1 = await consent_service_a.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-b",
            target_gateway_id="gateway-b",
        )
        await consent_service_b.receive_cross_federation_hop(
            hop=hop1,
            source_gateway_verifier=gateway_signer_a,
        )

        hop2 = await consent_service_b.create_cross_federation_hop(
            original_chain_id=original_chain_id,
            target_federation_id="federation-c",
            target_gateway_id="gateway-c",
        )
        await consent_service_c.receive_cross_federation_hop(
            hop=hop2,
            source_gateway_verifier=gateway_signer_b,
        )

        # Get provenance from B (B has the full picture as the middle node)
        provenance_b = await consent_service_b.get_provenance(original_chain_id)

        assert len(provenance_b) == 2
        assert provenance_b[0]["type"] == "cross_federation_hop"
        assert "federation-a" in provenance_b[0]["from"]
        assert "federation-b" in provenance_b[0]["to"]
        assert provenance_b[1]["type"] == "cross_federation_hop"
        assert "federation-b" in provenance_b[1]["from"]
        assert "federation-c" in provenance_b[1]["to"]

        # C only sees its local hop (the one it received)
        provenance_c = await consent_service_c.get_provenance(original_chain_id)
        assert len(provenance_c) == 1
        assert "federation-b" in provenance_c[0]["from"]
        assert "federation-c" in provenance_c[0]["to"]

    @pytest.mark.asyncio
    async def test_provenance_empty_for_unknown_chain(
        self,
        consent_service_a: CrossFederationConsentService,
    ) -> None:
        """Test provenance returns empty for unknown chain."""
        provenance = await consent_service_a.get_provenance("nonexistent")
        assert provenance == []


# =============================================================================
# INTEGRATION WITH LOCAL CONSENT CHAIN TESTS
# =============================================================================


class TestConsentChainIntegration:
    """Tests for integration with local ConsentChainEntry."""

    def test_consent_chain_entry_cross_federation_fields(self) -> None:
        """Test ConsentChainEntry has cross-federation fields."""
        from valence.privacy.sharing import ConsentChainEntry

        chain = ConsentChainEntry(
            id="chain-1",
            belief_id="belief-1",
            origin_sharer="did:vkb:user:alice",
            origin_timestamp=time.time(),
            origin_policy={"level": "bounded"},
            origin_signature=b"sig",
            hops=[],
            chain_hash=b"hash",
            origin_federation_id="federation-a",
            origin_gateway_id="gateway-a",
        )

        assert chain.origin_federation_id == "federation-a"
        assert chain.origin_gateway_id == "gateway-a"
        assert chain.cross_federation_hops == []
        assert not chain.crossed_federation_boundary()

    def test_consent_chain_add_cross_federation_hop(self) -> None:
        """Test adding cross-federation hop to ConsentChainEntry."""
        from valence.privacy.sharing import ConsentChainEntry
        from valence.privacy.sharing import CrossFederationHop as LocalHop

        chain = ConsentChainEntry(
            id="chain-1",
            belief_id="belief-1",
            origin_sharer="did:vkb:user:alice",
            origin_timestamp=time.time(),
            origin_policy={"level": "bounded"},
            origin_signature=b"sig",
            hops=[],
            chain_hash=b"hash",
            origin_federation_id="federation-a",
            origin_gateway_id="gateway-a",
        )

        hop = LocalHop(
            hop_id="hop-1",
            federation_id="federation-b",
            gateway_id="gateway-b",
            timestamp=time.time(),
            signature=b"hop-sig",
            from_federation_id="federation-a",
            from_gateway_id="gateway-a",
            hop_number=1,
        )

        chain.add_cross_federation_hop(hop)

        assert chain.crossed_federation_boundary()
        assert chain.get_current_federation() == "federation-b"
        assert chain.get_federation_path() == ["federation-a", "federation-b"]

    def test_local_hop_serialization(self) -> None:
        """Test CrossFederationHop serialization in sharing module."""
        from valence.privacy.sharing import CrossFederationHop as LocalHop

        hop = LocalHop(
            hop_id="hop-1",
            federation_id="federation-b",
            gateway_id="gateway-b",
            timestamp=1234567890.0,
            signature=b"test-sig",
            from_federation_id="federation-a",
            from_gateway_id="gateway-a",
            hop_number=1,
            policy_snapshot={"max_hops": 3},
        )

        data = hop.to_dict()
        restored = LocalHop.from_dict(data)

        assert restored.hop_id == hop.hop_id
        assert restored.federation_id == hop.federation_id
        assert restored.signature == hop.signature
        assert restored.policy_snapshot == hop.policy_snapshot
