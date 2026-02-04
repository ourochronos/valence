"""Tests for cross-federation domain verification (Issue #87).

Tests DNS-based verification, DID document verification, caching,
and graceful error handling.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from valence.federation.verification import (
    # Enums
    VerificationMethod,
    VerificationStatus,
    # Data classes
    VerificationResult,
    DomainClaim,
    # Cache
    VerificationCache,
    # DNS verification
    verify_dns_txt_record,
    verify_dns_txt_record_sync,
    DNS_TXT_PREFIX,
    DNS_TXT_PATTERN,
    # DID verification
    verify_did_document_claim,
    verify_did_document_claim_sync,
    DOMAIN_CLAIM_SERVICE_TYPE,
    # Main verification
    verify_cross_federation_domain,
    verify_cross_federation_domain_sync,
    verify_multiple_domains,
    get_verification_cache,
    invalidate_domain_cache,
    get_verification_cache_stats,
    # Constants
    DEFAULT_CACHE_TTL_SECONDS,
    FAILED_VERIFICATION_TTL_SECONDS,
)
from valence.federation.identity import (
    DIDDocument,
    ServiceEndpoint,
    VerificationMethod as DIDVerificationMethod,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def local_fed_did() -> str:
    """Local federation DID for testing."""
    return "did:vkb:web:local.example"


@pytest.fixture
def remote_fed_did() -> str:
    """Remote federation DID for testing."""
    return "did:vkb:web:remote.example"


@pytest.fixture
def test_domain() -> str:
    """Test domain for verification."""
    return "verified.example.com"


@pytest.fixture
def mock_did_document(remote_fed_did: str, test_domain: str) -> DIDDocument:
    """Create a mock DID document with domain claim."""
    return DIDDocument(
        id=remote_fed_did,
        services=[
            ServiceEndpoint(
                id=f"{remote_fed_did}#domain-claim-1",
                type=DOMAIN_CLAIM_SERVICE_TYPE,
                service_endpoint=test_domain,
            ),
        ],
        profile={
            "name": "Remote Federation",
            "domains": [test_domain],
        },
    )


@pytest.fixture
def verification_cache() -> VerificationCache:
    """Fresh verification cache for testing."""
    return VerificationCache(default_ttl=60, failed_ttl=10)


# =============================================================================
# DATA CLASS TESTS
# =============================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""
    
    def test_create_successful_result(self, local_fed_did: str, remote_fed_did: str, test_domain: str):
        """Test creating a successful verification result."""
        result = VerificationResult(
            local_federation=local_fed_did,
            remote_federation=remote_fed_did,
            domain=test_domain,
            verified=True,
            status=VerificationStatus.VERIFIED,
            method=VerificationMethod.DNS_TXT,
            dns_verified=True,
            dns_txt_record=f"valence-federation={remote_fed_did}",
        )
        
        assert result.verified is True
        assert result.status == VerificationStatus.VERIFIED
        assert result.method == VerificationMethod.DNS_TXT
        assert result.dns_verified is True
        assert result.error is None
    
    def test_create_failed_result(self, local_fed_did: str, remote_fed_did: str, test_domain: str):
        """Test creating a failed verification result."""
        result = VerificationResult(
            local_federation=local_fed_did,
            remote_federation=remote_fed_did,
            domain=test_domain,
            verified=False,
            status=VerificationStatus.FAILED,
            method=VerificationMethod.NONE,
            error="No matching DNS TXT record found",
        )
        
        assert result.verified is False
        assert result.status == VerificationStatus.FAILED
        assert result.error is not None
    
    def test_to_dict(self, local_fed_did: str, remote_fed_did: str, test_domain: str):
        """Test serialization to dictionary."""
        result = VerificationResult(
            local_federation=local_fed_did,
            remote_federation=remote_fed_did,
            domain=test_domain,
            verified=True,
            status=VerificationStatus.VERIFIED,
            method=VerificationMethod.BOTH,
            dns_verified=True,
            did_verified=True,
        )
        
        d = result.to_dict()
        
        assert d["local_federation"] == local_fed_did
        assert d["remote_federation"] == remote_fed_did
        assert d["domain"] == test_domain
        assert d["verified"] is True
        assert d["status"] == "verified"
        assert d["method"] == "both"


class TestDomainClaim:
    """Tests for DomainClaim dataclass."""
    
    def test_create_domain_claim(self, remote_fed_did: str, test_domain: str):
        """Test creating a domain claim."""
        claim = DomainClaim(
            federation_did=remote_fed_did,
            domain=test_domain,
            claim_type="ownership",
            verified=True,
            proof_method=VerificationMethod.DNS_TXT,
        )
        
        assert claim.federation_did == remote_fed_did
        assert claim.domain == test_domain
        assert claim.verified is True
    
    def test_to_dict(self, remote_fed_did: str, test_domain: str):
        """Test serialization to dictionary."""
        claim = DomainClaim(
            federation_did=remote_fed_did,
            domain=test_domain,
        )
        
        d = claim.to_dict()
        
        assert d["federation_did"] == remote_fed_did
        assert d["domain"] == test_domain
        assert "created_at" in d


# =============================================================================
# VERIFICATION CACHE TESTS
# =============================================================================


class TestVerificationCache:
    """Tests for VerificationCache."""
    
    def test_cache_set_and_get(
        self,
        verification_cache: VerificationCache,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
    ):
        """Test basic cache set and get."""
        result = VerificationResult(
            local_federation=local_fed_did,
            remote_federation=remote_fed_did,
            domain=test_domain,
            verified=True,
            status=VerificationStatus.VERIFIED,
            method=VerificationMethod.DNS_TXT,
        )
        
        verification_cache.set(result)
        cached = verification_cache.get(remote_fed_did, test_domain)
        
        assert cached is not None
        assert cached.verified is True
        assert cached.cached is True
        assert verification_cache.size == 1
    
    def test_cache_miss(self, verification_cache: VerificationCache, remote_fed_did: str):
        """Test cache miss."""
        cached = verification_cache.get(remote_fed_did, "nonexistent.example")
        
        assert cached is None
        assert verification_cache.misses == 1
    
    def test_cache_expiry_success(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
    ):
        """Test that successful verifications expire after TTL."""
        cache = VerificationCache(default_ttl=1, failed_ttl=1)  # 1 second TTL
        
        result = VerificationResult(
            local_federation=local_fed_did,
            remote_federation=remote_fed_did,
            domain=test_domain,
            verified=True,
            status=VerificationStatus.VERIFIED,
            method=VerificationMethod.DNS_TXT,
        )
        
        cache.set(result)
        assert cache.get(remote_fed_did, test_domain) is not None
        
        # Wait for expiry
        time.sleep(1.1)
        
        assert cache.get(remote_fed_did, test_domain) is None
    
    def test_cache_expiry_failed(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
    ):
        """Test that failed verifications have shorter TTL."""
        cache = VerificationCache(default_ttl=10, failed_ttl=1)
        
        result = VerificationResult(
            local_federation=local_fed_did,
            remote_federation=remote_fed_did,
            domain=test_domain,
            verified=False,  # Failed verification
            status=VerificationStatus.FAILED,
            method=VerificationMethod.NONE,
        )
        
        cache.set(result)
        assert cache.get(remote_fed_did, test_domain) is not None
        
        # Wait for failed TTL
        time.sleep(1.1)
        
        assert cache.get(remote_fed_did, test_domain) is None
    
    def test_cache_invalidation_by_federation(
        self,
        verification_cache: VerificationCache,
        local_fed_did: str,
        remote_fed_did: str,
    ):
        """Test invalidating cache entries by federation."""
        # Add multiple entries
        for domain in ["a.example", "b.example", "c.example"]:
            result = VerificationResult(
                local_federation=local_fed_did,
                remote_federation=remote_fed_did,
                domain=domain,
                verified=True,
                status=VerificationStatus.VERIFIED,
                method=VerificationMethod.DNS_TXT,
            )
            verification_cache.set(result)
        
        assert verification_cache.size == 3
        
        # Invalidate by federation
        count = verification_cache.invalidate(remote_federation=remote_fed_did)
        
        assert count == 3
        assert verification_cache.size == 0
    
    def test_cache_invalidation_by_domain(
        self,
        verification_cache: VerificationCache,
        local_fed_did: str,
    ):
        """Test invalidating cache entries by domain."""
        # Add entries from different federations for same domain
        for i in range(3):
            result = VerificationResult(
                local_federation=local_fed_did,
                remote_federation=f"did:vkb:web:fed{i}.example",
                domain="shared.example",
                verified=True,
                status=VerificationStatus.VERIFIED,
                method=VerificationMethod.DNS_TXT,
            )
            verification_cache.set(result)
        
        assert verification_cache.size == 3
        
        # Invalidate by domain
        count = verification_cache.invalidate(domain="shared.example")
        
        assert count == 3
        assert verification_cache.size == 0
    
    def test_cache_eviction(self, local_fed_did: str, remote_fed_did: str):
        """Test cache eviction when at capacity."""
        cache = VerificationCache(max_size=3)
        
        # Fill the cache
        for i in range(5):
            result = VerificationResult(
                local_federation=local_fed_did,
                remote_federation=remote_fed_did,
                domain=f"domain{i}.example",
                verified=True,
                status=VerificationStatus.VERIFIED,
                method=VerificationMethod.DNS_TXT,
            )
            cache.set(result)
        
        assert cache.size == 3
        assert cache.evictions == 2
    
    def test_cache_stats(
        self,
        verification_cache: VerificationCache,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
    ):
        """Test cache statistics."""
        result = VerificationResult(
            local_federation=local_fed_did,
            remote_federation=remote_fed_did,
            domain=test_domain,
            verified=True,
            status=VerificationStatus.VERIFIED,
            method=VerificationMethod.DNS_TXT,
        )
        
        verification_cache.set(result)
        verification_cache.get(remote_fed_did, test_domain)  # Hit
        verification_cache.get(remote_fed_did, "other.example")  # Miss
        
        stats = verification_cache.get_stats()
        
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# =============================================================================
# DNS VERIFICATION TESTS
# =============================================================================


class TestDNSVerification:
    """Tests for DNS TXT record verification."""
    
    @pytest.mark.asyncio
    async def test_dns_verification_success(self, remote_fed_did: str, test_domain: str):
        """Test successful DNS verification."""
        # Mock dns.resolver
        mock_rdata = MagicMock()
        mock_rdata.__str__ = lambda self: f'"{DNS_TXT_PREFIX}{remote_fed_did}"'
        
        mock_answers = [mock_rdata]
        
        with patch("dns.resolver.Resolver") as mock_resolver_class:
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.return_value = mock_answers
            
            verified, txt_record, error = await verify_dns_txt_record(test_domain, remote_fed_did)
        
        assert verified is True
        assert txt_record == f"{DNS_TXT_PREFIX}{remote_fed_did}"
        assert error is None
    
    @pytest.mark.asyncio
    async def test_dns_verification_nxdomain(self, remote_fed_did: str, test_domain: str):
        """Test DNS verification with non-existent domain."""
        import dns.resolver
        
        with patch("dns.resolver.Resolver") as mock_resolver_class:
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.side_effect = dns.resolver.NXDOMAIN()
            
            verified, txt_record, error = await verify_dns_txt_record(test_domain, remote_fed_did)
        
        assert verified is False
        assert txt_record is None
        assert "No matching DNS TXT record" in error
    
    @pytest.mark.asyncio
    async def test_dns_verification_did_mismatch(self, remote_fed_did: str, test_domain: str):
        """Test DNS verification with DID mismatch."""
        wrong_did = "did:vkb:web:wrong.example"
        
        mock_rdata = MagicMock()
        mock_rdata.__str__ = lambda self: f'"{DNS_TXT_PREFIX}{wrong_did}"'
        
        with patch("dns.resolver.Resolver") as mock_resolver_class:
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.return_value = [mock_rdata]
            
            verified, txt_record, error = await verify_dns_txt_record(test_domain, remote_fed_did)
        
        assert verified is False
    
    @pytest.mark.asyncio
    async def test_dns_verification_timeout_retry(self, remote_fed_did: str, test_domain: str):
        """Test DNS verification retries on timeout."""
        import dns.exception
        
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise dns.exception.Timeout()
            # Return valid response on third call
            mock_rdata = MagicMock()
            mock_rdata.__str__ = lambda self: f'"{DNS_TXT_PREFIX}{remote_fed_did}"'
            return [mock_rdata]
        
        with patch("dns.resolver.Resolver") as mock_resolver_class:
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.side_effect = side_effect
            
            verified, txt_record, error = await verify_dns_txt_record(
                test_domain, remote_fed_did, retries=2
            )
        
        assert call_count >= 2  # Should have retried


# =============================================================================
# DID DOCUMENT VERIFICATION TESTS
# =============================================================================


class TestDIDDocumentVerification:
    """Tests for DID document verification."""
    
    @pytest.mark.asyncio
    async def test_did_verification_via_service(
        self,
        remote_fed_did: str,
        test_domain: str,
        mock_did_document: DIDDocument,
    ):
        """Test verification via DID document service endpoint."""
        with patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            mock_resolve.return_value = mock_did_document
            
            verified, endpoint, error = await verify_did_document_claim(remote_fed_did, test_domain)
        
        assert verified is True
        assert endpoint == test_domain
        assert error is None
    
    @pytest.mark.asyncio
    async def test_did_verification_via_profile(self, remote_fed_did: str, test_domain: str):
        """Test verification via DID document profile domains."""
        did_doc = DIDDocument(
            id=remote_fed_did,
            services=[],  # No service endpoints
            profile={
                "name": "Remote Federation",
                "domains": [test_domain, "other.example"],
            },
        )
        
        with patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            mock_resolve.return_value = did_doc
            
            verified, endpoint, error = await verify_did_document_claim(remote_fed_did, test_domain)
        
        assert verified is True
        assert "profile:domains" in endpoint
    
    @pytest.mark.asyncio
    async def test_did_verification_no_claim(self, remote_fed_did: str, test_domain: str):
        """Test verification fails when no domain claim exists."""
        did_doc = DIDDocument(
            id=remote_fed_did,
            services=[],
            profile={"name": "Remote Federation"},
        )
        
        with patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            mock_resolve.return_value = did_doc
            
            verified, endpoint, error = await verify_did_document_claim(remote_fed_did, test_domain)
        
        assert verified is False
        assert endpoint is None
        assert "No matching domain claim" in error
    
    @pytest.mark.asyncio
    async def test_did_verification_resolution_failure(self, remote_fed_did: str, test_domain: str):
        """Test verification when DID resolution fails."""
        with patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            mock_resolve.return_value = None
            
            verified, endpoint, error = await verify_did_document_claim(remote_fed_did, test_domain)
        
        assert verified is False
        assert "Could not resolve DID document" in error


# =============================================================================
# CROSS-FEDERATION VERIFICATION TESTS
# =============================================================================


class TestCrossFederationVerification:
    """Tests for the main cross-federation verification function."""
    
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset global cache before each test."""
        invalidate_domain_cache()
        yield
        invalidate_domain_cache()
    
    @pytest.mark.asyncio
    async def test_verification_success_dns_only(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
    ):
        """Test successful verification via DNS only."""
        mock_rdata = MagicMock()
        mock_rdata.__str__ = lambda self: f'"{DNS_TXT_PREFIX}{remote_fed_did}"'
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.return_value = [mock_rdata]
            mock_resolve.return_value = None  # DID resolution fails
            
            result = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, test_domain, use_cache=False
            )
        
        assert result.verified is True
        assert result.status == VerificationStatus.VERIFIED
        assert result.method == VerificationMethod.DNS_TXT
        assert result.dns_verified is True
        assert result.did_verified is False
    
    @pytest.mark.asyncio
    async def test_verification_success_did_only(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
        mock_did_document: DIDDocument,
    ):
        """Test successful verification via DID document only."""
        import dns.resolver
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.side_effect = dns.resolver.NXDOMAIN()
            mock_resolve.return_value = mock_did_document
            
            result = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, test_domain, use_cache=False
            )
        
        assert result.verified is True
        assert result.status == VerificationStatus.VERIFIED
        assert result.method == VerificationMethod.DID_DOCUMENT
        assert result.dns_verified is False
        assert result.did_verified is True
    
    @pytest.mark.asyncio
    async def test_verification_success_both_methods(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
        mock_did_document: DIDDocument,
    ):
        """Test verification succeeds with both methods."""
        mock_rdata = MagicMock()
        mock_rdata.__str__ = lambda self: f'"{DNS_TXT_PREFIX}{remote_fed_did}"'
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.return_value = [mock_rdata]
            mock_resolve.return_value = mock_did_document
            
            result = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, test_domain, use_cache=False
            )
        
        assert result.verified is True
        assert result.method == VerificationMethod.BOTH
        assert result.dns_verified is True
        assert result.did_verified is True
    
    @pytest.mark.asyncio
    async def test_verification_failure_both_methods(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
    ):
        """Test verification fails when both methods fail."""
        import dns.resolver
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.side_effect = dns.resolver.NXDOMAIN()
            mock_resolve.return_value = None
            
            result = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, test_domain, use_cache=False
            )
        
        assert result.verified is False
        assert result.status == VerificationStatus.FAILED
        assert result.method == VerificationMethod.NONE
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_require_dns_verification(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
        mock_did_document: DIDDocument,
    ):
        """Test that require_dns enforces DNS verification."""
        import dns.resolver
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.side_effect = dns.resolver.NXDOMAIN()
            mock_resolve.return_value = mock_did_document  # DID succeeds
            
            result = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, test_domain,
                require_dns=True, use_cache=False
            )
        
        # Should fail because DNS failed, even though DID succeeded
        assert result.verified is False
        assert result.did_verified is True
        assert result.dns_verified is False
    
    @pytest.mark.asyncio
    async def test_require_did_verification(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
    ):
        """Test that require_did enforces DID verification."""
        mock_rdata = MagicMock()
        mock_rdata.__str__ = lambda self: f'"{DNS_TXT_PREFIX}{remote_fed_did}"'
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.return_value = [mock_rdata]  # DNS succeeds
            mock_resolve.return_value = None  # DID fails
            
            result = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, test_domain,
                require_did=True, use_cache=False
            )
        
        # Should fail because DID failed, even though DNS succeeded
        assert result.verified is False
        assert result.dns_verified is True
        assert result.did_verified is False
    
    @pytest.mark.asyncio
    async def test_require_both_methods(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
        mock_did_document: DIDDocument,
    ):
        """Test requiring both verification methods."""
        mock_rdata = MagicMock()
        mock_rdata.__str__ = lambda self: f'"{DNS_TXT_PREFIX}{remote_fed_did}"'
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.return_value = [mock_rdata]
            mock_resolve.return_value = mock_did_document
            
            result = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, test_domain,
                require_dns=True, require_did=True, use_cache=False
            )
        
        assert result.verified is True
        assert result.method == VerificationMethod.BOTH
    
    @pytest.mark.asyncio
    async def test_caching_behavior(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
        mock_did_document: DIDDocument,
    ):
        """Test that results are cached."""
        mock_rdata = MagicMock()
        mock_rdata.__str__ = lambda self: f'"{DNS_TXT_PREFIX}{remote_fed_did}"'
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.return_value = [mock_rdata]
            mock_resolve.return_value = mock_did_document
            
            # First call - not cached
            result1 = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, test_domain, use_cache=True
            )
            
            assert result1.verified is True
            assert result1.cached is False
            
            # Second call - should be cached
            result2 = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, test_domain, use_cache=True
            )
            
            assert result2.verified is True
            assert result2.cached is True
    
    @pytest.mark.asyncio
    async def test_domain_normalization(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        mock_did_document: DIDDocument,
    ):
        """Test that domains are normalized (lowercase, no trailing dot)."""
        mock_rdata = MagicMock()
        mock_rdata.__str__ = lambda self: f'"{DNS_TXT_PREFIX}{remote_fed_did}"'
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.return_value = [mock_rdata]
            mock_resolve.return_value = mock_did_document
            
            result = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, "VERIFIED.Example.COM.", use_cache=False
            )
        
        assert result.domain == "verified.example.com"


# =============================================================================
# BATCH VERIFICATION TESTS
# =============================================================================


class TestBatchVerification:
    """Tests for batch domain verification."""
    
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset global cache before each test."""
        invalidate_domain_cache()
        yield
        invalidate_domain_cache()
    
    @pytest.mark.asyncio
    async def test_verify_multiple_domains(self, local_fed_did: str):
        """Test verifying multiple domains concurrently."""
        claims = [
            ("did:vkb:web:fed1.example", "domain1.example"),
            ("did:vkb:web:fed2.example", "domain2.example"),
            ("did:vkb:web:fed3.example", "domain3.example"),
        ]
        
        # Create mock DID documents for each federation
        def create_mock_doc(did, domain):
            return DIDDocument(
                id=did,
                services=[],
                profile={"domains": [domain]},
            )
        
        import dns.resolver
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.side_effect = dns.resolver.NXDOMAIN()
            
            # Return appropriate DID doc based on the DID being resolved
            def resolve_side_effect(did):
                for fed_did, domain in claims:
                    if did == fed_did:
                        return create_mock_doc(fed_did, domain)
                return None
            
            mock_resolve.side_effect = resolve_side_effect
            
            results = await verify_multiple_domains(local_fed_did, claims, use_cache=False)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.remote_federation == claims[i][0]
            assert result.domain == claims[i][1]


# =============================================================================
# SYNC FUNCTION TESTS
# =============================================================================


class TestSyncFunctions:
    """Tests for synchronous wrapper functions."""
    
    def test_verify_dns_sync(self, remote_fed_did: str, test_domain: str):
        """Test synchronous DNS verification."""
        mock_rdata = MagicMock()
        mock_rdata.__str__ = lambda self: f'"{DNS_TXT_PREFIX}{remote_fed_did}"'
        
        with patch("dns.resolver.Resolver") as mock_resolver_class:
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.return_value = [mock_rdata]
            
            verified, txt, error = verify_dns_txt_record_sync(test_domain, remote_fed_did)
        
        assert verified is True
    
    def test_verify_did_sync(
        self,
        remote_fed_did: str,
        test_domain: str,
        mock_did_document: DIDDocument,
    ):
        """Test synchronous DID verification."""
        with patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            mock_resolve.return_value = mock_did_document
            
            verified, endpoint, error = verify_did_document_claim_sync(remote_fed_did, test_domain)
        
        assert verified is True
    
    def test_verify_cross_federation_sync(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
        mock_did_document: DIDDocument,
    ):
        """Test synchronous cross-federation verification."""
        mock_rdata = MagicMock()
        mock_rdata.__str__ = lambda self: f'"{DNS_TXT_PREFIX}{remote_fed_did}"'
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.return_value = [mock_rdata]
            mock_resolve.return_value = mock_did_document
            
            result = verify_cross_federation_domain_sync(
                local_fed_did, remote_fed_did, test_domain, use_cache=False
            )
        
        assert result.verified is True


# =============================================================================
# GLOBAL CACHE FUNCTION TESTS
# =============================================================================


class TestGlobalCacheFunctions:
    """Tests for global cache management functions."""
    
    def test_get_verification_cache(self):
        """Test getting the global cache."""
        cache = get_verification_cache()
        assert cache is not None
        assert isinstance(cache, VerificationCache)
    
    def test_invalidate_domain_cache(self, local_fed_did: str, remote_fed_did: str):
        """Test invalidating the global cache."""
        cache = get_verification_cache()
        
        # Add an entry
        result = VerificationResult(
            local_federation=local_fed_did,
            remote_federation=remote_fed_did,
            domain="test.example",
            verified=True,
            status=VerificationStatus.VERIFIED,
            method=VerificationMethod.DNS_TXT,
        )
        cache.set(result)
        
        assert cache.size > 0
        
        # Invalidate all
        count = invalidate_domain_cache()
        
        assert count > 0
        assert cache.size == 0
    
    def test_get_verification_cache_stats(self):
        """Test getting cache statistics."""
        stats = get_verification_cache_stats()
        
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for graceful error handling."""
    
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset global cache before each test."""
        invalidate_domain_cache()
        yield
        invalidate_domain_cache()
    
    @pytest.mark.asyncio
    async def test_dns_exception_handling(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
    ):
        """Test handling of DNS resolution exceptions."""
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.side_effect = Exception("Network error")
            mock_resolve.return_value = None
            
            result = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, test_domain, use_cache=False
            )
        
        # Should fail gracefully
        assert result.verified is False
        assert result.status == VerificationStatus.FAILED
        assert result.dns_error is not None
    
    @pytest.mark.asyncio
    async def test_did_exception_handling(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
    ):
        """Test handling of DID resolution exceptions."""
        import dns.resolver
        
        with patch("dns.resolver.Resolver") as mock_resolver_class, \
             patch("valence.federation.verification.resolve_did", new_callable=AsyncMock) as mock_resolve:
            
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve.side_effect = dns.resolver.NXDOMAIN()
            mock_resolve.side_effect = Exception("DID resolution error")
            
            result = await verify_cross_federation_domain(
                local_fed_did, remote_fed_did, test_domain, use_cache=False
            )
        
        # Should fail gracefully
        assert result.verified is False
        assert result.did_error is not None
    
    def test_sync_wrapper_exception_handling(
        self,
        local_fed_did: str,
        remote_fed_did: str,
        test_domain: str,
    ):
        """Test that sync wrapper handles exceptions gracefully."""
        with patch("valence.federation.verification.asyncio.run") as mock_run:
            mock_run.side_effect = RuntimeError("Event loop error")
            
            result = verify_cross_federation_domain_sync(
                local_fed_did, remote_fed_did, test_domain
            )
        
        assert result.verified is False
        assert result.status == VerificationStatus.ERROR
        assert "Event loop error" in result.error
