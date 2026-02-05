"""Cross-Federation Domain Verification for Valence Federation.

Implements domain claim verification across federation boundaries:
- DNS-based verification via TXT records
- DID document verification
- Caching of verification results with TTL
- Graceful error handling

Domain claims allow federations to assert control over specific domains,
which can be verified by other federations before trusting domain-specific
beliefs or routing requests.

Example:
    >>> result = await verify_cross_federation_domain(
    ...     local_fed=my_node_did,
    ...     remote_fed="did:vkb:web:trusted-partner.example",
    ...     domain_claim="partner.example"
    ... )
    >>> if result.verified:
    ...     print(f"Domain verified via {result.method}")
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

import aiohttp
import dns.resolver
import dns.exception

from .identity import (
    DID,
    DIDDocument,
    DIDMethod,
    parse_did,
    resolve_did,
    WELL_KNOWN_NODE_METADATA,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# DNS TXT record prefix for Valence domain verification
DNS_TXT_PREFIX = "valence-federation="

# Expected TXT record format: valence-federation=did:vkb:web:example.com
DNS_TXT_PATTERN = re.compile(r"^valence-federation=(.+)$")

# DID document service type for domain claims
DOMAIN_CLAIM_SERVICE_TYPE = "DomainClaim"

# Cache settings
DEFAULT_CACHE_TTL_SECONDS = 3600  # 1 hour
FAILED_VERIFICATION_TTL_SECONDS = 300  # 5 minutes for failed verifications
MAX_CACHE_SIZE = 10000

# DNS resolution settings
DNS_TIMEOUT_SECONDS = 10
DNS_RETRIES = 2

# DID resolution settings
DID_RESOLUTION_TIMEOUT = 15


# =============================================================================
# ENUMS
# =============================================================================


class VerificationMethod(str, Enum):
    """Method used to verify a domain claim."""
    
    DNS_TXT = "dns_txt"           # DNS TXT record verification
    DID_DOCUMENT = "did_document"  # DID document service endpoint verification
    BOTH = "both"                  # Both methods verified
    NONE = "none"                  # No verification performed


class VerificationStatus(str, Enum):
    """Status of a domain verification."""
    
    VERIFIED = "verified"      # Domain claim is verified
    FAILED = "failed"          # Verification failed
    PENDING = "pending"        # Verification in progress
    ERROR = "error"            # Error during verification
    EXPIRED = "expired"        # Cached verification expired


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class VerificationResult:
    """Result of a domain verification attempt."""
    
    local_federation: str        # DID of the verifying federation
    remote_federation: str       # DID of the federation making the claim
    domain: str                  # Domain being claimed
    verified: bool               # Whether the claim is verified
    status: VerificationStatus
    method: VerificationMethod
    
    # Details
    dns_verified: bool = False
    did_verified: bool = False
    dns_txt_record: str | None = None
    did_service_endpoint: str | None = None
    
    # Timing
    verified_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    verification_duration_ms: float = 0.0
    
    # Error handling
    error: str | None = None
    dns_error: str | None = None
    did_error: str | None = None
    
    # Cache info
    cached: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "local_federation": self.local_federation,
            "remote_federation": self.remote_federation,
            "domain": self.domain,
            "verified": self.verified,
            "status": self.status.value,
            "method": self.method.value,
            "dns_verified": self.dns_verified,
            "did_verified": self.did_verified,
            "dns_txt_record": self.dns_txt_record,
            "did_service_endpoint": self.did_service_endpoint,
            "verified_at": self.verified_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "verification_duration_ms": self.verification_duration_ms,
            "error": self.error,
            "dns_error": self.dns_error,
            "did_error": self.did_error,
            "cached": self.cached,
        }


@dataclass
class DomainClaim:
    """A domain claim made by a federation."""
    
    federation_did: str
    domain: str
    claim_type: str = "ownership"  # ownership, delegation, etc.
    proof_method: VerificationMethod = VerificationMethod.NONE
    verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "federation_did": self.federation_did,
            "domain": self.domain,
            "claim_type": self.claim_type,
            "proof_method": self.proof_method.value,
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# VERIFICATION CACHE
# =============================================================================


class VerificationCache:
    """Cache for domain verification results with TTL."""
    
    def __init__(
        self,
        default_ttl: int = DEFAULT_CACHE_TTL_SECONDS,
        failed_ttl: int = FAILED_VERIFICATION_TTL_SECONDS,
        max_size: int = MAX_CACHE_SIZE,
    ):
        """Initialize cache.
        
        Args:
            default_ttl: TTL for successful verifications in seconds
            failed_ttl: TTL for failed verifications in seconds
            max_size: Maximum cache entries
        """
        self._cache: dict[str, tuple[VerificationResult, float]] = {}
        self._default_ttl = default_ttl
        self._failed_ttl = failed_ttl
        self._max_size = max_size
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _make_key(
        self,
        remote_fed: str,
        domain: str,
    ) -> str:
        """Create cache key from federation and domain."""
        return f"{remote_fed}:{domain.lower()}"
    
    def get(
        self,
        remote_federation: str,
        domain: str,
    ) -> VerificationResult | None:
        """Get cached verification result if valid.
        
        Args:
            remote_federation: DID of the claiming federation
            domain: Domain being claimed
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._make_key(remote_federation, domain)
        
        if key not in self._cache:
            self.misses += 1
            return None
        
        result, cached_at = self._cache[key]
        ttl = self._default_ttl if result.verified else self._failed_ttl
        
        if time.time() - cached_at > ttl:
            # Expired
            del self._cache[key]
            self.misses += 1
            return None
        
        self.hits += 1
        
        # Return a copy with cached flag set
        return VerificationResult(
            local_federation=result.local_federation,
            remote_federation=result.remote_federation,
            domain=result.domain,
            verified=result.verified,
            status=result.status,
            method=result.method,
            dns_verified=result.dns_verified,
            did_verified=result.did_verified,
            dns_txt_record=result.dns_txt_record,
            did_service_endpoint=result.did_service_endpoint,
            verified_at=result.verified_at,
            expires_at=result.expires_at,
            verification_duration_ms=result.verification_duration_ms,
            error=result.error,
            dns_error=result.dns_error,
            did_error=result.did_error,
            cached=True,
        )
    
    def set(self, result: VerificationResult) -> None:
        """Cache a verification result.
        
        Args:
            result: Verification result to cache
        """
        # Evict if at capacity
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        
        key = self._make_key(result.remote_federation, result.domain)
        ttl = self._default_ttl if result.verified else self._failed_ttl
        result.expires_at = datetime.fromtimestamp(time.time() + ttl)
        self._cache[key] = (result, time.time())
    
    def invalidate(
        self,
        remote_federation: str | None = None,
        domain: str | None = None,
    ) -> int:
        """Invalidate cache entries.
        
        Args:
            remote_federation: Invalidate entries for this federation
            domain: Invalidate entries for this domain
            
        Returns:
            Number of entries invalidated
        """
        if remote_federation is None and domain is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        
        to_remove = []
        for key in self._cache:
            fed, dom = key.rsplit(":", 1)
            if remote_federation and fed == remote_federation:
                to_remove.append(key)
            elif domain and dom == domain.lower():
                to_remove.append(key)
        
        for key in to_remove:
            del self._cache[key]
        
        return len(to_remove)
    
    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry."""
        if not self._cache:
            return
        
        oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]
        self.evictions += 1
    
    @property
    def size(self) -> int:
        """Number of entries in cache."""
        return len(self._cache)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(1, total),
            "evictions": self.evictions,
        }


# =============================================================================
# DNS VERIFICATION
# =============================================================================


async def verify_dns_txt_record(
    domain: str,
    expected_did: str,
    timeout: float = DNS_TIMEOUT_SECONDS,
    retries: int = DNS_RETRIES,
) -> tuple[bool, str | None, str | None]:
    """Verify domain ownership via DNS TXT record.
    
    Checks for a TXT record at _valence-federation.<domain> or <domain>
    containing: valence-federation=<did>
    
    Args:
        domain: Domain to verify
        expected_did: DID that should be in the TXT record
        timeout: DNS query timeout in seconds
        retries: Number of retry attempts
        
    Returns:
        Tuple of (verified, txt_record_value, error_message)
    """
    # Normalize domain
    domain = domain.lower().rstrip(".")
    
    # Try both _valence-federation subdomain and root domain
    query_domains = [
        f"_valence-federation.{domain}",
        domain,
    ]
    
    for query_domain in query_domains:
        for attempt in range(retries + 1):
            try:
                resolver = dns.resolver.Resolver()
                resolver.timeout = timeout
                resolver.lifetime = timeout
                
                answers = resolver.resolve(query_domain, "TXT")
                
                for rdata in answers:
                    txt_value = str(rdata).strip('"')
                    
                    # Check for our verification record
                    match = DNS_TXT_PATTERN.match(txt_value)
                    if match:
                        found_did = match.group(1)
                        if found_did == expected_did:
                            logger.debug(f"DNS verification successful for {domain}: {txt_value}")
                            return True, txt_value, None
                        else:
                            logger.debug(f"DNS TXT found but DID mismatch: {found_did} != {expected_did}")
                
            except dns.resolver.NXDOMAIN:
                logger.debug(f"No DNS record found for {query_domain}")
                break  # No point retrying NXDOMAIN
                
            except dns.resolver.NoAnswer:
                logger.debug(f"No TXT records for {query_domain}")
                break  # No point retrying NoAnswer
                
            except dns.exception.Timeout:
                if attempt < retries:
                    logger.debug(f"DNS timeout for {query_domain}, retrying...")
                    await asyncio.sleep(0.5)
                    continue
                logger.warning(f"DNS timeout for {query_domain} after {retries + 1} attempts")
                
            except Exception as e:  # Intentionally broad: various DNS library errors possible
                if attempt < retries:
                    await asyncio.sleep(0.5)
                    continue
                return False, None, f"DNS error for {query_domain}: {str(e)}"
    
    return False, None, "No matching DNS TXT record found"


def verify_dns_txt_record_sync(
    domain: str,
    expected_did: str,
    timeout: float = DNS_TIMEOUT_SECONDS,
    retries: int = DNS_RETRIES,
) -> tuple[bool, str | None, str | None]:
    """Synchronous version of verify_dns_txt_record."""
    try:
        return asyncio.run(verify_dns_txt_record(domain, expected_did, timeout, retries))
    except Exception as e:  # Intentionally broad: asyncio.run can raise various errors
        return False, None, f"DNS verification error: {str(e)}"


# =============================================================================
# DID DOCUMENT VERIFICATION
# =============================================================================


async def verify_did_document_claim(
    remote_did: str,
    domain: str,
    timeout: float = DID_RESOLUTION_TIMEOUT,
) -> tuple[bool, str | None, str | None]:
    """Verify domain claim via DID document service endpoints.
    
    Checks if the remote federation's DID document includes a service
    endpoint declaring the domain claim.
    
    Args:
        remote_did: DID of the federation claiming the domain
        domain: Domain being claimed
        timeout: DID resolution timeout
        
    Returns:
        Tuple of (verified, service_endpoint, error_message)
    """
    try:
        # Resolve the DID document
        did_doc = await resolve_did(remote_did)
        
        if not did_doc:
            return False, None, f"Could not resolve DID document for {remote_did}"
        
        # Normalize domain for comparison
        domain_lower = domain.lower().rstrip(".")
        
        # Check services for domain claim
        for service in did_doc.services:
            if service.type == DOMAIN_CLAIM_SERVICE_TYPE:
                # Service endpoint might be the domain itself or a URL
                endpoint = service.service_endpoint.lower().rstrip("/")
                
                # Check if endpoint matches domain
                if endpoint == domain_lower or endpoint == f"https://{domain_lower}":
                    logger.debug(f"DID document verification successful for {domain}")
                    return True, service.service_endpoint, None
                
                # Check if it's a URL with the domain
                if domain_lower in endpoint:
                    logger.debug(f"DID document verification successful (URL match) for {domain}")
                    return True, service.service_endpoint, None
        
        # Also check profile for domain claims
        if did_doc.profile:
            claimed_domains = did_doc.profile.get("domains", [])
            if isinstance(claimed_domains, list):
                for claimed in claimed_domains:
                    if claimed.lower().rstrip(".") == domain_lower:
                        logger.debug(f"DID document profile verification successful for {domain}")
                        return True, f"profile:domains:{claimed}", None
        
        return False, None, "No matching domain claim in DID document"
        
    except Exception as e:  # Intentionally broad: DID resolution can fail in many ways
        logger.warning(f"DID document verification error for {remote_did}: {e}")
        return False, None, f"DID resolution error: {str(e)}"


def verify_did_document_claim_sync(
    remote_did: str,
    domain: str,
    timeout: float = DID_RESOLUTION_TIMEOUT,
) -> tuple[bool, str | None, str | None]:
    """Synchronous version of verify_did_document_claim."""
    try:
        return asyncio.run(verify_did_document_claim(remote_did, domain, timeout))
    except Exception as e:  # Intentionally broad: asyncio.run can raise various errors
        return False, None, f"DID document verification error: {str(e)}"


# =============================================================================
# CROSS-FEDERATION VERIFICATION
# =============================================================================


# Global cache instance
_verification_cache: VerificationCache | None = None


def get_verification_cache() -> VerificationCache:
    """Get the global verification cache."""
    global _verification_cache
    if _verification_cache is None:
        _verification_cache = VerificationCache()
    return _verification_cache


async def verify_cross_federation_domain(
    local_fed: str,
    remote_fed: str,
    domain_claim: str,
    require_dns: bool = False,
    require_did: bool = False,
    use_cache: bool = True,
) -> VerificationResult:
    """Verify a domain claim made by a remote federation.
    
    Performs verification using both DNS TXT records and DID document
    service endpoints. By default, verification succeeds if either
    method confirms the claim.
    
    Args:
        local_fed: DID of the local (verifying) federation
        remote_fed: DID of the remote (claiming) federation
        domain_claim: Domain being claimed (e.g., "example.com")
        require_dns: Require DNS verification to pass
        require_did: Require DID document verification to pass
        use_cache: Whether to use cached results
        
    Returns:
        VerificationResult with verification status and details
    """
    start_time = time.time()
    cache = get_verification_cache()
    
    # Normalize inputs
    domain = domain_claim.lower().rstrip(".")
    
    # Check cache
    if use_cache:
        cached = cache.get(remote_fed, domain)
        if cached:
            return cached
    
    # Initialize result
    result = VerificationResult(
        local_federation=local_fed,
        remote_federation=remote_fed,
        domain=domain,
        verified=False,
        status=VerificationStatus.PENDING,
        method=VerificationMethod.NONE,
    )
    
    # Run both verifications concurrently
    try:
        dns_task = verify_dns_txt_record(domain, remote_fed)
        did_task = verify_did_document_claim(remote_fed, domain)
        
        (dns_verified, dns_txt, dns_error), (did_verified, did_endpoint, did_error) = \
            await asyncio.gather(dns_task, did_task, return_exceptions=False)
        
        result.dns_verified = dns_verified
        result.dns_txt_record = dns_txt
        result.dns_error = dns_error
        
        result.did_verified = did_verified
        result.did_service_endpoint = did_endpoint
        result.did_error = did_error
        
    except Exception as e:  # Intentionally broad: gather may raise various async errors
        logger.exception(f"Error during domain verification for {domain}")
        result.status = VerificationStatus.ERROR
        result.error = str(e)
        result.verification_duration_ms = (time.time() - start_time) * 1000
        
        if use_cache:
            cache.set(result)
        return result
    
    # Determine verification status
    if require_dns and require_did:
        # Both must pass
        result.verified = dns_verified and did_verified
        if result.verified:
            result.method = VerificationMethod.BOTH
    elif require_dns:
        result.verified = dns_verified
        if result.verified:
            result.method = VerificationMethod.DNS_TXT
    elif require_did:
        result.verified = did_verified
        if result.verified:
            result.method = VerificationMethod.DID_DOCUMENT
    else:
        # Either can pass (default)
        result.verified = dns_verified or did_verified
        if dns_verified and did_verified:
            result.method = VerificationMethod.BOTH
        elif dns_verified:
            result.method = VerificationMethod.DNS_TXT
        elif did_verified:
            result.method = VerificationMethod.DID_DOCUMENT
    
    # Set final status
    if result.verified:
        result.status = VerificationStatus.VERIFIED
    else:
        result.status = VerificationStatus.FAILED
        if result.dns_error and result.did_error:
            result.error = f"DNS: {result.dns_error}; DID: {result.did_error}"
        elif result.dns_error:
            result.error = result.dns_error
        elif result.did_error:
            result.error = result.did_error
    
    result.verified_at = datetime.now()
    result.verification_duration_ms = (time.time() - start_time) * 1000
    
    # Cache result
    if use_cache:
        cache.set(result)
    
    logger.info(
        f"Domain verification for {domain} by {remote_fed}: "
        f"verified={result.verified}, method={result.method.value}, "
        f"duration={result.verification_duration_ms:.2f}ms"
    )
    
    return result


def verify_cross_federation_domain_sync(
    local_fed: str,
    remote_fed: str,
    domain_claim: str,
    require_dns: bool = False,
    require_did: bool = False,
    use_cache: bool = True,
) -> VerificationResult:
    """Synchronous version of verify_cross_federation_domain."""
    try:
        return asyncio.run(verify_cross_federation_domain(
            local_fed, remote_fed, domain_claim,
            require_dns=require_dns,
            require_did=require_did,
            use_cache=use_cache,
        ))
    except Exception as e:  # Intentionally broad: asyncio.run can raise various errors
        return VerificationResult(
            local_federation=local_fed,
            remote_federation=remote_fed,
            domain=domain_claim,
            verified=False,
            status=VerificationStatus.ERROR,
            method=VerificationMethod.NONE,
            error=str(e),
        )


# =============================================================================
# BATCH VERIFICATION
# =============================================================================


async def verify_multiple_domains(
    local_fed: str,
    claims: list[tuple[str, str]],  # List of (remote_fed, domain) pairs
    use_cache: bool = True,
) -> list[VerificationResult]:
    """Verify multiple domain claims concurrently.
    
    Args:
        local_fed: DID of the local federation
        claims: List of (remote_fed_did, domain) tuples to verify
        use_cache: Whether to use cached results
        
    Returns:
        List of VerificationResults in the same order as claims
    """
    tasks = [
        verify_cross_federation_domain(local_fed, remote_fed, domain, use_cache=use_cache)
        for remote_fed, domain in claims
    ]
    
    return await asyncio.gather(*tasks)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def invalidate_domain_cache(
    remote_federation: str | None = None,
    domain: str | None = None,
) -> int:
    """Invalidate cached verification results.
    
    Args:
        remote_federation: Invalidate entries for this federation
        domain: Invalidate entries for this domain
        
    Returns:
        Number of entries invalidated
    """
    return get_verification_cache().invalidate(remote_federation, domain)


def get_verification_cache_stats() -> dict[str, Any]:
    """Get verification cache statistics."""
    return get_verification_cache().get_stats()
