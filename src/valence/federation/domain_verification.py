"""Cross-Federation Domain Verification for Valence Federation.

This module provides domain claim verification across federation boundaries
using multiple verification methods:

1. **DNS-based verification**: TXT records with challenge tokens
2. **Mutual attestation**: Federations vouch for each other
3. **External authority**: Trusted third-party verification services

Example:
    >>> # Create a challenge for domain verification
    >>> challenge = await create_challenge("example.com", "did:vkb:web:my-fed")
    >>> print(f"Add TXT record: {challenge.dns_txt_value}")

    >>> # Later, check if the challenge was completed
    >>> result = await check_challenge(challenge.challenge_id)
    >>> if result.verified:
    ...     print(f"Domain verified via {result.method}")

    >>> # Or verify directly if DNS is already configured
    >>> result = await verify_domain("example.com", "did:vkb:web:remote-fed")

See: spec/components/federation/DOMAIN-VERIFICATION.md
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import aiohttp

from .verification import (
    VerificationStatus,
    get_verification_cache,
    verify_did_document_claim,
    verify_dns_txt_record,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Challenge settings
CHALLENGE_TOKEN_LENGTH = 32  # bytes
CHALLENGE_TTL_HOURS = 48  # Hours before challenge expires
CHALLENGE_PREFIX = "valence-verify="

# DNS record format for challenges
# Example: _valence-verify.example.com TXT "valence-verify=abc123..."
DNS_CHALLENGE_SUBDOMAIN = "_valence-verify"

# Attestation settings
ATTESTATION_VALIDITY_DAYS = 365  # How long attestations remain valid
MIN_ATTESTATION_TRUST = 0.6  # Minimum trust level to accept attestation
MAX_ATTESTATION_CHAIN_LENGTH = 3  # Prevent infinite attestation chains

# External authority settings
EXTERNAL_AUTHORITY_TIMEOUT = 30  # seconds
DEFAULT_AUTHORITIES = [
    "https://verify.valence-protocol.org/api/v1",
]


# =============================================================================
# ENUMS
# =============================================================================


class DomainVerificationMethod(StrEnum):
    """Methods for verifying domain ownership."""

    DNS_TXT = "dns_txt"  # DNS TXT record with challenge token
    DNS_CHALLENGE = "dns_challenge"  # Temporary DNS challenge
    MUTUAL_ATTESTATION = "mutual_attestation"  # Federation vouches for another
    EXTERNAL_AUTHORITY = "external_authority"  # Third-party verification
    DID_DOCUMENT = "did_document"  # DID document service endpoint
    COMBINED = "combined"  # Multiple methods confirmed


class ChallengeStatus(StrEnum):
    """Status of a domain verification challenge."""

    PENDING = "pending"  # Challenge created, awaiting verification
    VERIFIED = "verified"  # Challenge successfully verified
    EXPIRED = "expired"  # Challenge TTL exceeded
    FAILED = "failed"  # Verification attempted but failed
    CANCELLED = "cancelled"  # Challenge was cancelled


class AttestationType(StrEnum):
    """Types of domain attestation."""

    DIRECT = "direct"  # Direct attestation by a trusted federation
    TRANSITIVE = "transitive"  # Attestation through a chain of trust
    REVOKED = "revoked"  # Previously valid attestation now revoked


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class VerificationEvidence:
    """Evidence supporting a domain verification claim."""

    method: DomainVerificationMethod
    timestamp: datetime = field(default_factory=datetime.now)

    # DNS evidence
    dns_record: str | None = None
    dns_query_domain: str | None = None

    # Attestation evidence
    attester_did: str | None = None
    attestation_signature: str | None = None
    attestation_chain: list[str] | None = None

    # External authority evidence
    authority_url: str | None = None
    authority_response: dict[str, Any] | None = None
    authority_certificate: str | None = None

    # DID document evidence
    did_service_endpoint: str | None = None
    did_document_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method.value,
            "timestamp": self.timestamp.isoformat(),
            "dns_record": self.dns_record,
            "dns_query_domain": self.dns_query_domain,
            "attester_did": self.attester_did,
            "attestation_signature": self.attestation_signature,
            "attestation_chain": self.attestation_chain,
            "authority_url": self.authority_url,
            "authority_response": self.authority_response,
            "authority_certificate": self.authority_certificate,
            "did_service_endpoint": self.did_service_endpoint,
            "did_document_hash": self.did_document_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationEvidence:
        """Create from dictionary."""
        return cls(
            method=DomainVerificationMethod(data["method"]),
            timestamp=(datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now()),
            dns_record=data.get("dns_record"),
            dns_query_domain=data.get("dns_query_domain"),
            attester_did=data.get("attester_did"),
            attestation_signature=data.get("attestation_signature"),
            attestation_chain=data.get("attestation_chain"),
            authority_url=data.get("authority_url"),
            authority_response=data.get("authority_response"),
            authority_certificate=data.get("authority_certificate"),
            did_service_endpoint=data.get("did_service_endpoint"),
            did_document_hash=data.get("did_document_hash"),
        )


@dataclass
class DomainVerificationResult:
    """Result of a domain verification attempt.

    Contains the verification status, method used, timestamp, and evidence
    supporting the verification decision.
    """

    domain: str
    federation_did: str
    verified: bool
    status: VerificationStatus
    method: DomainVerificationMethod
    timestamp: datetime = field(default_factory=datetime.now)

    # Evidence supporting the verification
    evidence: list[VerificationEvidence] = field(default_factory=list)

    # Additional context
    expires_at: datetime | None = None
    verification_duration_ms: float = 0.0
    error: str | None = None
    cached: bool = False

    # For mutual attestation
    attestation_chain: list[str] | None = None
    trust_level: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "domain": self.domain,
            "federation_did": self.federation_did,
            "verified": self.verified,
            "status": self.status.value,
            "method": self.method.value,
            "timestamp": self.timestamp.isoformat(),
            "evidence": [e.to_dict() for e in self.evidence],
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "verification_duration_ms": self.verification_duration_ms,
            "error": self.error,
            "cached": self.cached,
            "attestation_chain": self.attestation_chain,
            "trust_level": self.trust_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainVerificationResult:
        """Create from dictionary."""
        return cls(
            domain=data["domain"],
            federation_did=data["federation_did"],
            verified=data["verified"],
            status=VerificationStatus(data["status"]),
            method=DomainVerificationMethod(data["method"]),
            timestamp=(datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now()),
            evidence=[VerificationEvidence.from_dict(e) for e in data.get("evidence", [])],
            expires_at=(datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None),
            verification_duration_ms=data.get("verification_duration_ms", 0.0),
            error=data.get("error"),
            cached=data.get("cached", False),
            attestation_chain=data.get("attestation_chain"),
            trust_level=data.get("trust_level"),
        )


@dataclass
class DomainChallenge:
    """A challenge for domain verification.

    The domain owner must add a DNS TXT record with the challenge token
    to prove ownership.
    """

    challenge_id: str
    domain: str
    federation_did: str
    token: str
    status: ChallengeStatus = ChallengeStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    verified_at: datetime | None = None

    # DNS record details
    dns_subdomain: str = DNS_CHALLENGE_SUBDOMAIN
    dns_txt_value: str = ""

    def __post_init__(self):
        """Set up DNS values after initialization."""
        if not self.dns_txt_value:
            self.dns_txt_value = f"{CHALLENGE_PREFIX}{self.token}"
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(hours=CHALLENGE_TTL_HOURS)

    @property
    def is_expired(self) -> bool:
        """Check if the challenge has expired."""
        return datetime.now() > self.expires_at if self.expires_at else False

    @property
    def full_dns_record(self) -> str:
        """Full DNS record name to create."""
        return f"{self.dns_subdomain}.{self.domain}"

    @property
    def instructions(self) -> str:
        """Human-readable instructions for completing the challenge."""
        return (
            f"To verify ownership of {self.domain}, add a DNS TXT record:\n"
            f"  Name: {self.full_dns_record}\n"
            f"  Value: {self.dns_txt_value}\n"
            f"  TTL: 300 (or any value)\n\n"
            f"Challenge expires: {self.expires_at.isoformat() if self.expires_at else 'never'}\n"
            f"Challenge ID: {self.challenge_id}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "challenge_id": self.challenge_id,
            "domain": self.domain,
            "federation_did": self.federation_did,
            "token": self.token,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "dns_subdomain": self.dns_subdomain,
            "dns_txt_value": self.dns_txt_value,
            "full_dns_record": self.full_dns_record,
            "instructions": self.instructions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainChallenge:
        """Create from dictionary."""
        challenge = cls(
            challenge_id=data["challenge_id"],
            domain=data["domain"],
            federation_did=data["federation_did"],
            token=data["token"],
            status=ChallengeStatus(data["status"]),
            created_at=(datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()),
            dns_subdomain=data.get("dns_subdomain", DNS_CHALLENGE_SUBDOMAIN),
            dns_txt_value=data.get("dns_txt_value", ""),
        )
        if data.get("expires_at"):
            challenge.expires_at = datetime.fromisoformat(data["expires_at"])
        if data.get("verified_at"):
            challenge.verified_at = datetime.fromisoformat(data["verified_at"])
        return challenge


@dataclass
class DomainAttestation:
    """An attestation that one federation vouches for another's domain claim.

    Used in mutual attestation verification where a trusted federation
    confirms that another federation controls a domain.
    """

    attestation_id: str
    domain: str
    subject_did: str  # Federation being vouched for
    attester_did: str  # Federation doing the vouching
    attestation_type: AttestationType = AttestationType.DIRECT

    # Validity
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    revoked_at: datetime | None = None

    # Cryptographic proof
    signature: str | None = None
    signature_algorithm: str = "Ed25519"

    # Chain for transitive attestations
    chain: list[str] | None = None

    def __post_init__(self):
        """Set default expiration."""
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(days=ATTESTATION_VALIDITY_DAYS)

    @property
    def is_valid(self) -> bool:
        """Check if attestation is currently valid."""
        now = datetime.now()
        if self.revoked_at and now > self.revoked_at:
            return False
        if self.expires_at and now > self.expires_at:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attestation_id": self.attestation_id,
            "domain": self.domain,
            "subject_did": self.subject_did,
            "attester_did": self.attester_did,
            "attestation_type": self.attestation_type.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "signature": self.signature,
            "signature_algorithm": self.signature_algorithm,
            "chain": self.chain,
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainAttestation:
        """Create from dictionary."""
        return cls(
            attestation_id=data["attestation_id"],
            domain=data["domain"],
            subject_did=data["subject_did"],
            attester_did=data["attester_did"],
            attestation_type=AttestationType(data.get("attestation_type", "direct")),
            created_at=(datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()),
            expires_at=(datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None),
            revoked_at=(datetime.fromisoformat(data["revoked_at"]) if data.get("revoked_at") else None),
            signature=data.get("signature"),
            signature_algorithm=data.get("signature_algorithm", "Ed25519"),
            chain=data.get("chain"),
        )


# =============================================================================
# CHALLENGE STORE
# =============================================================================


class ChallengeStore:
    """In-memory store for domain verification challenges.

    In production, this would be backed by a database.
    """

    def __init__(self):
        """Initialize the challenge store."""
        self._challenges: dict[str, DomainChallenge] = {}
        self._by_domain: dict[str, list[str]] = {}  # domain -> challenge_ids

    def add(self, challenge: DomainChallenge) -> None:
        """Add a challenge to the store."""
        self._challenges[challenge.challenge_id] = challenge
        if challenge.domain not in self._by_domain:
            self._by_domain[challenge.domain] = []
        self._by_domain[challenge.domain].append(challenge.challenge_id)

    def get(self, challenge_id: str) -> DomainChallenge | None:
        """Get a challenge by ID."""
        challenge = self._challenges.get(challenge_id)
        if challenge and challenge.is_expired and challenge.status == ChallengeStatus.PENDING:
            challenge.status = ChallengeStatus.EXPIRED
        return challenge

    def get_for_domain(self, domain: str) -> list[DomainChallenge]:
        """Get all challenges for a domain."""
        challenge_ids = self._by_domain.get(domain.lower(), [])
        result: list[DomainChallenge] = []
        for cid in challenge_ids:
            challenge = self.get(cid)
            if challenge is not None:
                result.append(challenge)
        return result

    def update(self, challenge: DomainChallenge) -> None:
        """Update a challenge in the store."""
        self._challenges[challenge.challenge_id] = challenge

    def remove(self, challenge_id: str) -> None:
        """Remove a challenge from the store."""
        if challenge_id in self._challenges:
            challenge = self._challenges[challenge_id]
            del self._challenges[challenge_id]
            if challenge.domain in self._by_domain:
                self._by_domain[challenge.domain] = [cid for cid in self._by_domain[challenge.domain] if cid != challenge_id]

    def cleanup_expired(self) -> int:
        """Remove expired challenges. Returns count removed."""
        expired = [cid for cid, c in self._challenges.items() if c.is_expired]
        for cid in expired:
            self.remove(cid)
        return len(expired)


# =============================================================================
# ATTESTATION STORE
# =============================================================================


class AttestationStore:
    """In-memory store for domain attestations.

    In production, this would be backed by a database.
    """

    def __init__(self):
        """Initialize the attestation store."""
        self._attestations: dict[str, DomainAttestation] = {}
        self._by_domain: dict[str, list[str]] = {}  # domain -> attestation_ids
        self._by_subject: dict[str, list[str]] = {}  # subject_did -> attestation_ids

    def add(self, attestation: DomainAttestation) -> None:
        """Add an attestation to the store."""
        self._attestations[attestation.attestation_id] = attestation

        domain = attestation.domain.lower()
        if domain not in self._by_domain:
            self._by_domain[domain] = []
        self._by_domain[domain].append(attestation.attestation_id)

        if attestation.subject_did not in self._by_subject:
            self._by_subject[attestation.subject_did] = []
        self._by_subject[attestation.subject_did].append(attestation.attestation_id)

    def get(self, attestation_id: str) -> DomainAttestation | None:
        """Get an attestation by ID."""
        return self._attestations.get(attestation_id)

    def get_for_domain(
        self,
        domain: str,
        subject_did: str | None = None,
        valid_only: bool = True,
    ) -> list[DomainAttestation]:
        """Get attestations for a domain."""
        attestation_ids = self._by_domain.get(domain.lower(), [])
        attestations: list[DomainAttestation] = [a for a in (self._attestations.get(aid) for aid in attestation_ids) if a is not None]

        if subject_did:
            attestations = [a for a in attestations if a.subject_did == subject_did]

        if valid_only:
            attestations = [a for a in attestations if a.is_valid]

        return attestations

    def get_for_subject(
        self,
        subject_did: str,
        valid_only: bool = True,
    ) -> list[DomainAttestation]:
        """Get all attestations for a subject federation."""
        attestation_ids = self._by_subject.get(subject_did, [])
        attestations: list[DomainAttestation] = [a for a in (self._attestations.get(aid) for aid in attestation_ids) if a is not None]

        if valid_only:
            attestations = [a for a in attestations if a.is_valid]

        return attestations

    def revoke(self, attestation_id: str) -> bool:
        """Revoke an attestation. Returns True if found and revoked."""
        attestation = self._attestations.get(attestation_id)
        if attestation:
            attestation.revoked_at = datetime.now()
            attestation.attestation_type = AttestationType.REVOKED
            return True
        return False


# =============================================================================
# EXTERNAL AUTHORITY PROTOCOL
# =============================================================================


@runtime_checkable
class ExternalAuthorityClient(Protocol):
    """Protocol for external domain verification authorities."""

    async def verify_domain(
        self,
        domain: str,
        federation_did: str,
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        """Verify a domain claim via external authority.

        Returns:
            Tuple of (verified, response_data, error_message)
        """
        ...


class DefaultExternalAuthorityClient:
    """Default implementation for external authority verification.

    Calls HTTP endpoints at trusted verification authorities.
    """

    def __init__(
        self,
        authority_urls: list[str] | None = None,
        timeout: float = EXTERNAL_AUTHORITY_TIMEOUT,
    ):
        """Initialize the client.

        Args:
            authority_urls: List of authority base URLs
            timeout: Request timeout in seconds
        """
        self.authority_urls = authority_urls or DEFAULT_AUTHORITIES
        self.timeout = timeout

    async def verify_domain(
        self,
        domain: str,
        federation_did: str,
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        """Verify a domain claim via external authority.

        Tries each authority in order until one succeeds.

        Returns:
            Tuple of (verified, response_data, error_message)
        """
        errors = []

        for authority_url in self.authority_urls:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{authority_url.rstrip('/')}/verify"
                    payload = {
                        "domain": domain,
                        "federation_did": federation_did,
                    }

                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("verified"):
                                return True, data, None
                            else:
                                errors.append(f"{authority_url}: Not verified")
                        else:
                            errors.append(f"{authority_url}: HTTP {response.status}")

            except TimeoutError:
                errors.append(f"{authority_url}: Timeout")
            except aiohttp.ClientError as e:
                errors.append(f"{authority_url}: {str(e)}")
            except Exception as e:
                errors.append(f"{authority_url}: {str(e)}")

        return False, None, "; ".join(errors)


# =============================================================================
# GLOBAL STORES
# =============================================================================


_challenge_store: ChallengeStore | None = None
_attestation_store: AttestationStore | None = None
_external_client: ExternalAuthorityClient | None = None


def get_challenge_store() -> ChallengeStore:
    """Get the global challenge store."""
    global _challenge_store
    if _challenge_store is None:
        _challenge_store = ChallengeStore()
    return _challenge_store


def get_attestation_store() -> AttestationStore:
    """Get the global attestation store."""
    global _attestation_store
    if _attestation_store is None:
        _attestation_store = AttestationStore()
    return _attestation_store


def get_external_client() -> ExternalAuthorityClient:
    """Get the global external authority client."""
    global _external_client
    if _external_client is None:
        _external_client = DefaultExternalAuthorityClient()
    return _external_client


def set_external_client(client: ExternalAuthorityClient) -> None:
    """Set a custom external authority client."""
    global _external_client
    _external_client = client


# =============================================================================
# TRUST CONFIGURATION
# =============================================================================

# Default trust score when no trust relationship exists
DEFAULT_TRUST_SCORE = 0.5

# Minimum trust score to auto-verify a domain claim (skip full verification)
# Nodes with trust >= this threshold are trusted enough that a single
# verification method suffices and attestation alone is accepted.
TRUST_AUTO_VERIFY_THRESHOLD = 0.8

# Trust score below which additional scrutiny is applied:
# - require_all methods must pass
# - attestation chains are shortened
LOW_TRUST_THRESHOLD = 0.3

# Domain used for scoping trust lookups in domain verification context
DOMAIN_VERIFICATION_TRUST_DOMAIN = "federation:domain-verification"


# =============================================================================
# TRUST LOOKUP (integration with trust system)
# =============================================================================

# Optional TrustService override (for testing or custom wiring)
_trust_service_override: Any = None


def set_trust_service(service: Any) -> None:
    """Set a custom TrustService for domain verification trust lookups.

    Args:
        service: A TrustService instance (or None to reset to default).
    """
    global _trust_service_override
    _trust_service_override = service


def _get_trust_service() -> Any:
    """Get the TrustService instance used for trust lookups.

    Returns the override if set, otherwise imports and returns the
    default singleton from the privacy.trust module.
    """
    if _trust_service_override is not None:
        return _trust_service_override
    try:
        from valence.privacy.trust.service import get_trust_service

        return get_trust_service()
    except Exception:
        logger.debug("TrustService not available, using default trust scores")
        return None


async def get_federation_trust(
    local_did: str,
    remote_did: str,
    domain: str | None = DOMAIN_VERIFICATION_TRUST_DOMAIN,
) -> float:
    """Get trust level for a remote federation using TrustService.

    Queries the trust graph for a direct or delegated trust relationship
    between the local and remote DIDs. Falls back to a default score
    if no trust relationship exists.

    Args:
        local_did: DID of the local (verifying) federation
        remote_did: DID of the remote federation being evaluated
        domain: Trust domain to scope the lookup (default: domain verification)

    Returns:
        Trust level from 0.0 to 1.0
    """
    service = _get_trust_service()
    if service is None:
        return DEFAULT_TRUST_SCORE

    try:
        # Try direct trust first
        edge = service.get_trust(local_did, remote_did, domain)
        if edge is not None:
            return edge.overall_trust

        # Try delegated (transitive) trust
        delegated = service.compute_delegated_trust(local_did, remote_did, domain)
        if delegated is not None:
            return delegated.overall_trust

        # Fall back to global trust (no domain scope)
        if domain is not None:
            edge = service.get_trust(local_did, remote_did, None)
            if edge is not None:
                return edge.overall_trust

        return DEFAULT_TRUST_SCORE
    except Exception:
        logger.debug(
            f"Error looking up trust for {remote_did}, using default",
            exc_info=True,
        )
        return DEFAULT_TRUST_SCORE


# =============================================================================
# MAIN VERIFICATION FUNCTIONS
# =============================================================================


async def create_challenge(
    domain: str,
    federation_did: str,
) -> DomainChallenge:
    """Create a new domain verification challenge.

    Creates a challenge that requires the domain owner to add a
    DNS TXT record to prove ownership.

    Args:
        domain: Domain to verify (e.g., "example.com")
        federation_did: DID of the federation claiming the domain

    Returns:
        DomainChallenge with instructions for completing verification

    Example:
        >>> challenge = await create_challenge("example.com", "did:vkb:web:my-fed")
        >>> print(challenge.instructions)
        To verify ownership of example.com, add a DNS TXT record:
          Name: _valence-verify.example.com
          Value: valence-verify=abc123...
    """
    # Generate a secure random token
    token = secrets.token_urlsafe(CHALLENGE_TOKEN_LENGTH)

    # Create the challenge
    challenge = DomainChallenge(
        challenge_id=str(uuid4()),
        domain=domain.lower().rstrip("."),
        federation_did=federation_did,
        token=token,
    )

    # Store it
    get_challenge_store().add(challenge)

    logger.info(f"Created domain verification challenge for {domain} (federation={federation_did}, expires={challenge.expires_at})")

    return challenge


async def check_challenge(
    challenge_id: str,
) -> DomainVerificationResult:
    """Check if a domain verification challenge has been completed.

    Queries DNS to see if the required TXT record has been added.

    Args:
        challenge_id: ID of the challenge to check

    Returns:
        DomainVerificationResult with verification status

    Example:
        >>> result = await check_challenge("abc-123-def")
        >>> if result.verified:
        ...     print("Domain ownership confirmed!")
    """
    start_time = time.time()
    store = get_challenge_store()

    # Get the challenge
    challenge = store.get(challenge_id)
    if not challenge:
        return DomainVerificationResult(
            domain="unknown",
            federation_did="unknown",
            verified=False,
            status=VerificationStatus.ERROR,
            method=DomainVerificationMethod.DNS_CHALLENGE,
            error=f"Challenge not found: {challenge_id}",
        )

    # Check if expired
    if challenge.is_expired:
        challenge.status = ChallengeStatus.EXPIRED
        store.update(challenge)
        return DomainVerificationResult(
            domain=challenge.domain,
            federation_did=challenge.federation_did,
            verified=False,
            status=VerificationStatus.EXPIRED,
            method=DomainVerificationMethod.DNS_CHALLENGE,
            error="Challenge has expired",
        )

    # Check if already verified
    if challenge.status == ChallengeStatus.VERIFIED:
        return DomainVerificationResult(
            domain=challenge.domain,
            federation_did=challenge.federation_did,
            verified=True,
            status=VerificationStatus.VERIFIED,
            method=DomainVerificationMethod.DNS_CHALLENGE,
            cached=True,
        )

    # Query DNS for the challenge TXT record
    try:
        import dns.exception
        import dns.resolver

        query_domain = challenge.full_dns_record
        resolver = dns.resolver.Resolver()
        resolver.timeout = 10
        resolver.lifetime = 10

        try:
            answers = resolver.resolve(query_domain, "TXT")

            for rdata in answers:
                txt_value = str(rdata).strip('"')
                if txt_value == challenge.dns_txt_value:
                    # Challenge completed successfully!
                    challenge.status = ChallengeStatus.VERIFIED
                    challenge.verified_at = datetime.now()
                    store.update(challenge)

                    evidence = VerificationEvidence(
                        method=DomainVerificationMethod.DNS_CHALLENGE,
                        dns_record=txt_value,
                        dns_query_domain=query_domain,
                    )

                    duration = (time.time() - start_time) * 1000

                    logger.info(f"Domain challenge verified for {challenge.domain}")

                    return DomainVerificationResult(
                        domain=challenge.domain,
                        federation_did=challenge.federation_did,
                        verified=True,
                        status=VerificationStatus.VERIFIED,
                        method=DomainVerificationMethod.DNS_CHALLENGE,
                        evidence=[evidence],
                        verification_duration_ms=duration,
                    )

            # TXT record found but wrong value
            challenge.status = ChallengeStatus.FAILED
            store.update(challenge)
            return DomainVerificationResult(
                domain=challenge.domain,
                federation_did=challenge.federation_did,
                verified=False,
                status=VerificationStatus.FAILED,
                method=DomainVerificationMethod.DNS_CHALLENGE,
                error="TXT record found but value does not match challenge",
                verification_duration_ms=(time.time() - start_time) * 1000,
            )

        except dns.resolver.NXDOMAIN:
            return DomainVerificationResult(
                domain=challenge.domain,
                federation_did=challenge.federation_did,
                verified=False,
                status=VerificationStatus.PENDING,
                method=DomainVerificationMethod.DNS_CHALLENGE,
                error=f"DNS record not found: {query_domain}",
                verification_duration_ms=(time.time() - start_time) * 1000,
            )

        except dns.resolver.NoAnswer:
            return DomainVerificationResult(
                domain=challenge.domain,
                federation_did=challenge.federation_did,
                verified=False,
                status=VerificationStatus.PENDING,
                method=DomainVerificationMethod.DNS_CHALLENGE,
                error=f"No TXT records found for {query_domain}",
                verification_duration_ms=(time.time() - start_time) * 1000,
            )

    except Exception as e:
        logger.exception(f"Error checking challenge {challenge_id}")
        return DomainVerificationResult(
            domain=challenge.domain,
            federation_did=challenge.federation_did,
            verified=False,
            status=VerificationStatus.ERROR,
            method=DomainVerificationMethod.DNS_CHALLENGE,
            error=str(e),
            verification_duration_ms=(time.time() - start_time) * 1000,
        )


async def verify_domain(
    domain: str,
    federation_did: str,
    local_did: str | None = None,
    methods: list[DomainVerificationMethod] | None = None,
    require_all: bool = False,
    use_cache: bool = True,
    trust_threshold: float | None = None,
) -> DomainVerificationResult:
    """Verify that a federation controls a domain.

    Attempts verification using multiple methods:
    1. DNS TXT record (permanent record asserting ownership)
    2. Mutual attestation (trusted federations vouching)
    3. External authority (third-party verification services)
    4. DID document (service endpoint in DID doc)

    Trust-weighted verification:
    - High trust (>= trust_threshold or TRUST_AUTO_VERIFY_THRESHOLD):
      A single method success is sufficient; attestation alone is accepted.
    - Low trust (< LOW_TRUST_THRESHOLD): Requires all specified methods
      to pass (overrides require_all=False), and attestation chain
      length is shortened for additional scrutiny.
    - Medium trust: Default behavior (require_all as specified).

    The trust score is included in the result's trust_level field.

    By default, verification succeeds if ANY method succeeds.
    Set require_all=True to require ALL specified methods.

    Args:
        domain: Domain to verify (e.g., "example.com")
        federation_did: DID of the federation claiming the domain
        local_did: DID of the local (verifying) federation (optional)
        methods: List of methods to try (default: all)
        require_all: If True, all methods must succeed
        use_cache: Whether to use cached results
        trust_threshold: Custom trust threshold for auto-verify
                        (default: TRUST_AUTO_VERIFY_THRESHOLD)

    Returns:
        DomainVerificationResult with status, method used, and evidence

    Example:
        >>> result = await verify_domain("example.com", "did:vkb:web:remote-fed")
        >>> if result.verified:
        ...     print(f"Verified via {result.method.value}")
        ...     for evidence in result.evidence:
        ...         print(f"  Evidence: {evidence.method.value}")
    """
    start_time = time.time()
    domain = domain.lower().rstrip(".")
    local_did = local_did or "did:vkb:local"
    auto_verify_threshold = trust_threshold if trust_threshold is not None else TRUST_AUTO_VERIFY_THRESHOLD

    # Look up trust score for the remote federation
    trust_score = await get_federation_trust(local_did, federation_did)

    # Trust-weighted adjustments:
    # - Low trust nodes require all methods to pass
    # - High trust nodes get streamlined verification
    effective_require_all = require_all
    if trust_score < LOW_TRUST_THRESHOLD:
        effective_require_all = True
        logger.info(f"Low trust ({trust_score:.2f}) for {federation_did}: requiring all verification methods")

    # Default to all methods
    if methods is None:
        methods = [
            DomainVerificationMethod.DNS_TXT,
            DomainVerificationMethod.MUTUAL_ATTESTATION,
            DomainVerificationMethod.EXTERNAL_AUTHORITY,
            DomainVerificationMethod.DID_DOCUMENT,
        ]

    # Check cache
    cache = get_verification_cache()
    if use_cache:
        cached = cache.get(federation_did, domain)
        if cached:
            return DomainVerificationResult(
                domain=domain,
                federation_did=federation_did,
                verified=cached.verified,
                status=cached.status,
                method=(
                    DomainVerificationMethod(cached.method.value)
                    if cached.method.value in [m.value for m in DomainVerificationMethod]
                    else DomainVerificationMethod.DNS_TXT
                ),
                cached=True,
                verification_duration_ms=0,
                trust_level=trust_score,
            )

    # Collect evidence from each method
    evidence_list: list[VerificationEvidence] = []
    method_results: dict[DomainVerificationMethod, bool] = {}
    errors: list[str] = []

    # Try each verification method
    for method in methods:
        try:
            if method == DomainVerificationMethod.DNS_TXT:
                verified, txt_record, error = await verify_dns_txt_record(domain, federation_did)
                method_results[method] = verified
                if verified:
                    evidence_list.append(
                        VerificationEvidence(
                            method=method,
                            dns_record=txt_record,
                            dns_query_domain=domain,
                        )
                    )
                elif error:
                    errors.append(f"DNS: {error}")

            elif method == DomainVerificationMethod.MUTUAL_ATTESTATION:
                verified, attestation = await _verify_mutual_attestation(domain, federation_did, local_did)
                method_results[method] = verified
                if verified and attestation:
                    evidence_list.append(
                        VerificationEvidence(
                            method=method,
                            attester_did=attestation.attester_did,
                            attestation_signature=attestation.signature,
                            attestation_chain=attestation.chain,
                        )
                    )
                elif not verified:
                    errors.append("Attestation: No valid attestation found")

            elif method == DomainVerificationMethod.EXTERNAL_AUTHORITY:
                verified, response, error = await _verify_external_authority(domain, federation_did)
                method_results[method] = verified
                if verified:
                    evidence_list.append(
                        VerificationEvidence(
                            method=method,
                            authority_response=response,
                        )
                    )
                elif error:
                    errors.append(f"External: {error}")

            elif method == DomainVerificationMethod.DID_DOCUMENT:
                verified, endpoint, error = await verify_did_document_claim(federation_did, domain)
                method_results[method] = verified
                if verified:
                    evidence_list.append(
                        VerificationEvidence(
                            method=method,
                            did_service_endpoint=endpoint,
                        )
                    )
                elif error:
                    errors.append(f"DID: {error}")

        except Exception as e:
            logger.warning(f"Error in {method.value} verification: {e}")
            method_results[method] = False
            errors.append(f"{method.value}: {str(e)}")

    # Determine overall result based on trust-weighted requirements
    successful_methods = [m for m, v in method_results.items() if v]

    if effective_require_all:
        verified = all(method_results.values())
    else:
        verified = any(method_results.values())

    # High-trust auto-verification: if the federation has high trust and
    # at least one method succeeded, accept even if require_all was set
    # (trust overrides strict requirements for highly trusted peers)
    if not verified and trust_score >= auto_verify_threshold and successful_methods:
        logger.info(f"High trust ({trust_score:.2f}) for {federation_did}: auto-verifying with {len(successful_methods)} method(s)")
        verified = True

    # Determine the primary method used
    if verified:
        if len(successful_methods) > 1:
            primary_method = DomainVerificationMethod.COMBINED
        else:
            primary_method = successful_methods[0] if successful_methods else methods[0]
        status = VerificationStatus.VERIFIED
    else:
        primary_method = methods[0] if methods else DomainVerificationMethod.DNS_TXT
        status = VerificationStatus.FAILED

    duration = (time.time() - start_time) * 1000

    result = DomainVerificationResult(
        domain=domain,
        federation_did=federation_did,
        verified=verified,
        status=status,
        method=primary_method,
        evidence=evidence_list,
        verification_duration_ms=duration,
        error="; ".join(errors) if errors and not verified else None,
        trust_level=trust_score,
    )

    # Cache the result (via the base verification cache)
    # We'd need to convert to VerificationResult for caching, skip for simplicity

    logger.info(
        f"Domain verification for {domain} by {federation_did}: verified={verified}, method={primary_method.value}, duration={duration:.2f}ms"
    )

    return result


# =============================================================================
# MUTUAL ATTESTATION FUNCTIONS
# =============================================================================


async def create_attestation(
    domain: str,
    subject_did: str,
    attester_did: str,
    signature: str | None = None,
) -> DomainAttestation:
    """Create a domain attestation (one federation vouching for another).

    This allows federations to vouch for each other's domain claims,
    creating a web of trust for domain verification.

    Args:
        domain: Domain being attested
        subject_did: DID of the federation being vouched for
        attester_did: DID of the federation doing the vouching
        signature: Optional cryptographic signature

    Returns:
        DomainAttestation record

    Example:
        >>> # Federation A vouches for Federation B's control of example.com
        >>> attestation = await create_attestation(
        ...     domain="example.com",
        ...     subject_did="did:vkb:web:federation-b",
        ...     attester_did="did:vkb:web:federation-a",
        ... )
    """
    attestation = DomainAttestation(
        attestation_id=str(uuid4()),
        domain=domain.lower().rstrip("."),
        subject_did=subject_did,
        attester_did=attester_did,
        signature=signature,
    )

    get_attestation_store().add(attestation)

    logger.info(f"Created domain attestation: {attester_did} vouches for {subject_did}'s control of {domain}")

    return attestation


async def revoke_attestation(attestation_id: str) -> bool:
    """Revoke a domain attestation.

    Args:
        attestation_id: ID of the attestation to revoke

    Returns:
        True if attestation was found and revoked
    """
    result = get_attestation_store().revoke(attestation_id)
    if result:
        logger.info(f"Revoked attestation: {attestation_id}")
    return result


async def _verify_mutual_attestation(
    domain: str,
    subject_did: str,
    local_did: str,
    min_trust: float = MIN_ATTESTATION_TRUST,
    max_chain_length: int = MAX_ATTESTATION_CHAIN_LENGTH,
) -> tuple[bool, DomainAttestation | None]:
    """Verify a domain via mutual attestation.

    Checks if any trusted federation has attested to the domain claim.
    Supports transitive attestation with configurable chain length.

    Trust-weighted behavior:
    - For low-trust subjects, the max attestation chain length is
      reduced (minimum 1) to increase scrutiny.
    - The min_trust threshold for attesters is unchanged; only the
      chain length is tightened.

    Returns:
        Tuple of (verified, attestation_used)
    """
    # Adjust chain length based on subject's trust level
    subject_trust = await get_federation_trust(local_did, subject_did)
    if subject_trust < LOW_TRUST_THRESHOLD:
        # Shorten chain for low-trust subjects (min 1 to allow direct attestation)
        max_chain_length = max(1, max_chain_length - 1)
        logger.debug(f"Low trust ({subject_trust:.2f}) for {subject_did}: reduced attestation chain length to {max_chain_length}")
    store = get_attestation_store()

    # Get all valid attestations for this domain + subject
    attestations = store.get_for_domain(domain, subject_did=subject_did, valid_only=True)

    if not attestations:
        return False, None

    # Check each attestation
    for attestation in attestations:
        # Get trust level for the attester
        trust = await get_federation_trust(local_did, attestation.attester_did)

        if trust >= min_trust:
            logger.debug(f"Domain {domain} verified via attestation from {attestation.attester_did} (trust={trust:.2f})")
            return True, attestation

        # Try transitive attestation (chain)
        if attestation.chain and len(attestation.chain) < max_chain_length:
            # Recursively verify through the chain
            # (simplified: just check if any chain member is trusted)
            for chain_did in attestation.chain:
                chain_trust = await get_federation_trust(local_did, chain_did)
                if chain_trust >= min_trust:
                    logger.debug(f"Domain {domain} verified via transitive attestation through {chain_did} (trust={chain_trust:.2f})")
                    return True, attestation

    return False, None


# =============================================================================
# EXTERNAL AUTHORITY FUNCTIONS
# =============================================================================


async def _verify_external_authority(
    domain: str,
    federation_did: str,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Verify a domain via external authority.

    Queries trusted third-party verification services.

    Returns:
        Tuple of (verified, response_data, error_message)
    """
    client = get_external_client()
    return await client.verify_domain(domain, federation_did)


async def register_with_authority(
    domain: str,
    federation_did: str,
    authority_url: str | None = None,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Register a domain claim with an external authority.

    This initiates a verification process with the authority.

    Args:
        domain: Domain to register
        federation_did: DID claiming the domain
        authority_url: Specific authority to use (default: first in list)

    Returns:
        Tuple of (success, response_data, error_message)
    """
    authority_url = authority_url or DEFAULT_AUTHORITIES[0] if DEFAULT_AUTHORITIES else None

    if not authority_url:
        return False, None, "No external authority configured"

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{authority_url.rstrip('/')}/register"
            payload = {
                "domain": domain,
                "federation_did": federation_did,
            }

            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=EXTERNAL_AUTHORITY_TIMEOUT),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return True, data, None
                else:
                    return False, None, f"HTTP {response.status}"

    except Exception as e:
        return False, None, str(e)


# =============================================================================
# BATCH OPERATIONS
# =============================================================================


async def verify_multiple_domains(
    claims: list[tuple[str, str]],  # List of (domain, federation_did) pairs
    local_did: str | None = None,
    methods: list[DomainVerificationMethod] | None = None,
) -> list[DomainVerificationResult]:
    """Verify multiple domain claims concurrently.

    Args:
        claims: List of (domain, federation_did) tuples to verify
        local_did: DID of the local federation
        methods: Verification methods to use

    Returns:
        List of DomainVerificationResults in the same order as claims
    """
    tasks = [
        verify_domain(
            domain=domain,
            federation_did=fed_did,
            local_did=local_did,
            methods=methods,
        )
        for domain, fed_did in claims
    ]

    return await asyncio.gather(*tasks)


# =============================================================================
# SYNCHRONOUS WRAPPERS
# =============================================================================


def create_challenge_sync(
    domain: str,
    federation_did: str,
) -> DomainChallenge:
    """Synchronous version of create_challenge."""
    return asyncio.run(create_challenge(domain, federation_did))


def check_challenge_sync(challenge_id: str) -> DomainVerificationResult:
    """Synchronous version of check_challenge."""
    return asyncio.run(check_challenge(challenge_id))


def verify_domain_sync(
    domain: str,
    federation_did: str,
    local_did: str | None = None,
    methods: list[DomainVerificationMethod] | None = None,
    require_all: bool = False,
    use_cache: bool = True,
    trust_threshold: float | None = None,
) -> DomainVerificationResult:
    """Synchronous version of verify_domain."""
    return asyncio.run(verify_domain(domain, federation_did, local_did, methods, require_all, use_cache, trust_threshold))


# =============================================================================
# CLEANUP AND MAINTENANCE
# =============================================================================


def cleanup_expired_challenges() -> int:
    """Remove expired challenges from the store.

    Returns:
        Number of challenges removed
    """
    return get_challenge_store().cleanup_expired()


def get_challenge_stats() -> dict[str, Any]:
    """Get statistics about challenges."""
    store = get_challenge_store()
    challenges = list(store._challenges.values())

    return {
        "total": len(challenges),
        "pending": len([c for c in challenges if c.status == ChallengeStatus.PENDING]),
        "verified": len([c for c in challenges if c.status == ChallengeStatus.VERIFIED]),
        "expired": len([c for c in challenges if c.status == ChallengeStatus.EXPIRED]),
        "failed": len([c for c in challenges if c.status == ChallengeStatus.FAILED]),
    }


def get_attestation_stats() -> dict[str, Any]:
    """Get statistics about attestations."""
    store = get_attestation_store()
    attestations = list(store._attestations.values())

    return {
        "total": len(attestations),
        "valid": len([a for a in attestations if a.is_valid]),
        "revoked": len([a for a in attestations if a.attestation_type == AttestationType.REVOKED]),
        "expired": len([a for a in attestations if not a.is_valid and a.attestation_type != AttestationType.REVOKED]),
    }
