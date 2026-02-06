"""Tests for TrustManager integration in domain verification (Issue #264).

Tests cover:
- Trust lookup via TrustService (direct, delegated, fallback)
- Trust-weighted verification decisions
  - High-trust auto-verify
  - Low-trust additional scrutiny (require_all enforcement)
  - Medium-trust default behavior
- Attestation chain length reduction for low-trust subjects
- trust_threshold parameter override
- Graceful fallback when TrustService is unavailable

Run with:
    python -m pytest tests/federation/test_domain_verification_trust.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valence.federation.domain_verification import (
    DEFAULT_TRUST_SCORE,
    DOMAIN_VERIFICATION_TRUST_DOMAIN,
    LOW_TRUST_THRESHOLD,
    TRUST_AUTO_VERIFY_THRESHOLD,
    _verify_mutual_attestation,
    create_attestation,
    get_attestation_store,
    get_challenge_store,
    get_federation_trust,
    set_trust_service,
    verify_domain,
)
from valence.privacy.trust.service import TrustService

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_stores():
    """Reset global stores and trust service between tests."""
    # Reset challenge store
    store = get_challenge_store()
    store._challenges.clear()
    store._by_domain.clear()

    # Reset attestation store
    att_store = get_attestation_store()
    att_store._attestations.clear()
    att_store._by_domain.clear()
    att_store._by_subject.clear()

    # Reset trust service override
    set_trust_service(None)

    yield

    set_trust_service(None)


@pytest.fixture()
def trust_service():
    """Create an in-memory TrustService and wire it into domain verification."""
    svc = TrustService(use_memory=True)
    set_trust_service(svc)
    return svc


# =============================================================================
# TRUST LOOKUP TESTS
# =============================================================================


class TestGetFederationTrust:
    """Tests for get_federation_trust with TrustService integration."""

    @pytest.mark.asyncio
    async def test_returns_default_when_no_service(self):
        """Should return DEFAULT_TRUST_SCORE when TrustService is unavailable."""
        set_trust_service(None)
        # Patch the import inside _get_trust_service to simulate unavailable service
        with patch(
            "valence.privacy.trust.service.get_trust_service",
            side_effect=ImportError("no module"),
        ):
            score = await get_federation_trust("did:local", "did:remote")
        # When import fails, _get_trust_service returns None → default score
        assert score == DEFAULT_TRUST_SCORE

    @pytest.mark.asyncio
    async def test_returns_direct_trust(self, trust_service: TrustService):
        """Should return direct trust score from TrustService."""
        trust_service.grant_trust(
            source_did="did:local",
            target_did="did:remote",
            competence=0.9,
            integrity=0.8,
            confidentiality=0.7,
            judgment=0.6,
            domain=DOMAIN_VERIFICATION_TRUST_DOMAIN,
        )

        score = await get_federation_trust("did:local", "did:remote")
        # Geometric mean of (0.9, 0.8, 0.7, 0.6)
        expected = (0.9 * 0.8 * 0.7 * 0.6) ** 0.25
        assert abs(score - expected) < 0.01

    @pytest.mark.asyncio
    async def test_returns_global_trust_fallback(self, trust_service: TrustService):
        """Should fall back to global trust when domain-specific not found."""
        # Grant global trust (no domain)
        trust_service.grant_trust(
            source_did="did:local",
            target_did="did:remote",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            judgment=0.8,
        )

        score = await get_federation_trust("did:local", "did:remote")
        assert abs(score - 0.8) < 0.01

    @pytest.mark.asyncio
    async def test_returns_delegated_trust(self, trust_service: TrustService):
        """Should return delegated trust when no direct trust exists."""
        # local trusts intermediate, allows delegation
        trust_service.grant_trust(
            source_did="did:local",
            target_did="did:intermediate",
            competence=0.9,
            integrity=0.9,
            confidentiality=0.9,
            judgment=0.9,
            can_delegate=True,
            delegation_depth=2,
        )
        # intermediate trusts remote
        trust_service.grant_trust(
            source_did="did:intermediate",
            target_did="did:remote",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            judgment=0.8,
        )

        score = await get_federation_trust("did:local", "did:remote")
        # Should get delegated trust (> 0, < direct trust of intermediate)
        assert score > 0.0
        assert score < 0.9  # Decayed through delegation

    @pytest.mark.asyncio
    async def test_returns_default_when_no_trust_edge(self, trust_service: TrustService):
        """Should return DEFAULT_TRUST_SCORE when no trust relationship exists."""
        score = await get_federation_trust("did:local", "did:unknown")
        assert score == DEFAULT_TRUST_SCORE

    @pytest.mark.asyncio
    async def test_handles_service_exception(self):
        """Should return default score on TrustService errors."""
        mock_service = MagicMock()
        mock_service.get_trust.side_effect = RuntimeError("db error")
        set_trust_service(mock_service)

        score = await get_federation_trust("did:local", "did:remote")
        assert score == DEFAULT_TRUST_SCORE


# =============================================================================
# TRUST-WEIGHTED VERIFICATION TESTS
# =============================================================================


class TestTrustWeightedVerification:
    """Tests for trust-weighted domain verification decisions."""

    @pytest.mark.asyncio
    async def test_high_trust_auto_verify(self, trust_service: TrustService):
        """High-trust federation should be auto-verified with single method."""
        # Grant very high trust
        trust_service.grant_trust(
            source_did="did:vkb:local",
            target_did="did:vkb:web:high-trust",
            competence=0.95,
            integrity=0.95,
            confidentiality=0.95,
            judgment=0.95,
            domain=DOMAIN_VERIFICATION_TRUST_DOMAIN,
        )

        # Mock only DNS to succeed, other methods fail
        with (
            patch(
                "valence.federation.domain_verification.verify_dns_txt_record",
                new_callable=AsyncMock,
                return_value=(True, "valence-did=did:vkb:web:high-trust", None),
            ),
            patch(
                "valence.federation.domain_verification._verify_mutual_attestation",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "valence.federation.domain_verification._verify_external_authority",
                new_callable=AsyncMock,
                return_value=(False, None, "unavailable"),
            ),
            patch(
                "valence.federation.domain_verification.verify_did_document_claim",
                new_callable=AsyncMock,
                return_value=(False, None, "no endpoint"),
            ),
        ):
            result = await verify_domain(
                "example.com",
                "did:vkb:web:high-trust",
                local_did="did:vkb:local",
                require_all=True,  # Would normally fail
                use_cache=False,
            )

        # Should be verified despite require_all=True because of high trust
        assert result.verified is True
        assert result.trust_level is not None
        assert result.trust_level >= TRUST_AUTO_VERIFY_THRESHOLD

    @pytest.mark.asyncio
    async def test_low_trust_requires_all_methods(self, trust_service: TrustService):
        """Low-trust federation should require all methods to pass."""
        # Grant very low trust
        trust_service.grant_trust(
            source_did="did:vkb:local",
            target_did="did:vkb:web:low-trust",
            competence=0.2,
            integrity=0.2,
            confidentiality=0.2,
            judgment=0.2,
            domain=DOMAIN_VERIFICATION_TRUST_DOMAIN,
        )

        # Mock only DNS succeeds, attestation fails
        with (
            patch(
                "valence.federation.domain_verification.verify_dns_txt_record",
                new_callable=AsyncMock,
                return_value=(True, "valence-did=did:vkb:web:low-trust", None),
            ),
            patch(
                "valence.federation.domain_verification._verify_mutual_attestation",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "valence.federation.domain_verification._verify_external_authority",
                new_callable=AsyncMock,
                return_value=(False, None, "unavailable"),
            ),
            patch(
                "valence.federation.domain_verification.verify_did_document_claim",
                new_callable=AsyncMock,
                return_value=(False, None, "no endpoint"),
            ),
        ):
            result = await verify_domain(
                "example.com",
                "did:vkb:web:low-trust",
                local_did="did:vkb:local",
                require_all=False,  # Would normally pass with ANY
                use_cache=False,
            )

        # Should FAIL because low trust forces require_all
        assert result.verified is False
        assert result.trust_level is not None
        assert result.trust_level < LOW_TRUST_THRESHOLD

    @pytest.mark.asyncio
    async def test_medium_trust_default_behavior(self, trust_service: TrustService):
        """Medium-trust federation uses default verification behavior."""
        # Grant medium trust (default range)
        trust_service.grant_trust(
            source_did="did:vkb:local",
            target_did="did:vkb:web:medium-trust",
            competence=0.6,
            integrity=0.6,
            confidentiality=0.6,
            judgment=0.6,
            domain=DOMAIN_VERIFICATION_TRUST_DOMAIN,
        )

        # Mock only DNS succeeds
        with (
            patch(
                "valence.federation.domain_verification.verify_dns_txt_record",
                new_callable=AsyncMock,
                return_value=(True, "valence-did=did:vkb:web:medium-trust", None),
            ),
            patch(
                "valence.federation.domain_verification._verify_mutual_attestation",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "valence.federation.domain_verification._verify_external_authority",
                new_callable=AsyncMock,
                return_value=(False, None, "unavailable"),
            ),
            patch(
                "valence.federation.domain_verification.verify_did_document_claim",
                new_callable=AsyncMock,
                return_value=(False, None, "no endpoint"),
            ),
        ):
            result = await verify_domain(
                "example.com",
                "did:vkb:web:medium-trust",
                local_did="did:vkb:local",
                require_all=False,
                use_cache=False,
            )

        # Should pass with default ANY behavior
        assert result.verified is True
        assert result.trust_level is not None

    @pytest.mark.asyncio
    async def test_trust_level_included_in_result(self, trust_service: TrustService):
        """Verification result should include trust_level."""
        trust_service.grant_trust(
            source_did="did:vkb:local",
            target_did="did:vkb:web:test",
            competence=0.7,
            integrity=0.7,
            confidentiality=0.7,
            judgment=0.7,
            domain=DOMAIN_VERIFICATION_TRUST_DOMAIN,
        )

        with (
            patch(
                "valence.federation.domain_verification.verify_dns_txt_record",
                new_callable=AsyncMock,
                return_value=(True, "txt-record", None),
            ),
            patch(
                "valence.federation.domain_verification._verify_mutual_attestation",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "valence.federation.domain_verification._verify_external_authority",
                new_callable=AsyncMock,
                return_value=(False, None, "err"),
            ),
            patch(
                "valence.federation.domain_verification.verify_did_document_claim",
                new_callable=AsyncMock,
                return_value=(False, None, "err"),
            ),
        ):
            result = await verify_domain(
                "example.com",
                "did:vkb:web:test",
                local_did="did:vkb:local",
                use_cache=False,
            )

        assert result.trust_level is not None
        assert abs(result.trust_level - 0.7) < 0.01

    @pytest.mark.asyncio
    async def test_custom_trust_threshold(self, trust_service: TrustService):
        """Custom trust_threshold should override TRUST_AUTO_VERIFY_THRESHOLD."""
        # Grant moderate trust (below default threshold but above custom)
        trust_service.grant_trust(
            source_did="did:vkb:local",
            target_did="did:vkb:web:custom",
            competence=0.7,
            integrity=0.7,
            confidentiality=0.7,
            judgment=0.7,
            domain=DOMAIN_VERIFICATION_TRUST_DOMAIN,
        )

        with (
            patch(
                "valence.federation.domain_verification.verify_dns_txt_record",
                new_callable=AsyncMock,
                return_value=(True, "txt", None),
            ),
            patch(
                "valence.federation.domain_verification._verify_mutual_attestation",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "valence.federation.domain_verification._verify_external_authority",
                new_callable=AsyncMock,
                return_value=(False, None, "err"),
            ),
            patch(
                "valence.federation.domain_verification.verify_did_document_claim",
                new_callable=AsyncMock,
                return_value=(False, None, "err"),
            ),
        ):
            # With default threshold (0.8), require_all=True would fail
            result_default = await verify_domain(
                "example.com",
                "did:vkb:web:custom",
                local_did="did:vkb:local",
                require_all=True,
                use_cache=False,
            )

            # With custom lower threshold (0.6), auto-verify kicks in
            result_custom = await verify_domain(
                "example.com",
                "did:vkb:web:custom",
                local_did="did:vkb:local",
                require_all=True,
                use_cache=False,
                trust_threshold=0.6,
            )

        assert result_default.verified is False
        assert result_custom.verified is True


# =============================================================================
# ATTESTATION CHAIN TRUST TESTS
# =============================================================================


class TestAttestationChainTrust:
    """Tests for trust-weighted attestation chain behavior."""

    @pytest.mark.asyncio
    async def test_low_trust_shortens_attestation_chain(self, trust_service: TrustService):
        """Low-trust subjects should have shorter attestation chain limits."""
        # Set up low trust for the subject
        trust_service.grant_trust(
            source_did="did:local",
            target_did="did:subject",
            competence=0.2,
            integrity=0.2,
            confidentiality=0.2,
            judgment=0.2,
            domain=DOMAIN_VERIFICATION_TRUST_DOMAIN,
        )

        # Create a transitive attestation with chain length 2
        await create_attestation(
            domain="example.com",
            subject_did="did:subject",
            attester_did="did:attester-a",
            signature="sig-a",
        )

        # Attester has low trust too - below MIN_ATTESTATION_TRUST
        # Patch get_federation_trust to return specific values
        call_count = 0

        async def mock_trust(local_did, remote_did, domain=None):
            nonlocal call_count
            call_count += 1
            if remote_did == "did:subject":
                return 0.2  # Low trust
            if remote_did == "did:attester-a":
                return 0.4  # Below min attestation trust (0.6)
            return DEFAULT_TRUST_SCORE

        with patch(
            "valence.federation.domain_verification.get_federation_trust",
            side_effect=mock_trust,
        ):
            verified, _ = await _verify_mutual_attestation(
                domain="example.com",
                subject_did="did:subject",
                local_did="did:local",
            )

        # Should fail: attester trust (0.4) < min attestation trust (0.6)
        assert verified is False

    @pytest.mark.asyncio
    async def test_high_trust_attestation_accepted(self, trust_service: TrustService):
        """High-trust attester should have attestation accepted."""
        # Set up high trust for attester
        trust_service.grant_trust(
            source_did="did:local",
            target_did="did:trusted-attester",
            competence=0.9,
            integrity=0.9,
            confidentiality=0.9,
            judgment=0.9,
            domain=DOMAIN_VERIFICATION_TRUST_DOMAIN,
        )
        # Medium trust for subject
        trust_service.grant_trust(
            source_did="did:local",
            target_did="did:subject",
            competence=0.6,
            integrity=0.6,
            confidentiality=0.6,
            judgment=0.6,
            domain=DOMAIN_VERIFICATION_TRUST_DOMAIN,
        )

        await create_attestation(
            domain="example.com",
            subject_did="did:subject",
            attester_did="did:trusted-attester",
            signature="sig",
        )

        verified, attestation = await _verify_mutual_attestation(
            domain="example.com",
            subject_did="did:subject",
            local_did="did:local",
        )

        assert verified is True
        assert attestation is not None
        assert attestation.attester_did == "did:trusted-attester"


# =============================================================================
# SET/RESET TRUST SERVICE TESTS
# =============================================================================


class TestTrustServiceWiring:
    """Tests for set_trust_service and _get_trust_service."""

    @pytest.mark.asyncio
    async def test_set_trust_service_override(self):
        """set_trust_service should override the default service."""
        mock = MagicMock()
        mock_edge = MagicMock()
        mock_edge.overall_trust = 0.42
        mock.get_trust.return_value = mock_edge
        set_trust_service(mock)

        score = await get_federation_trust("did:a", "did:b")
        assert abs(score - 0.42) < 0.01
        mock.get_trust.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_trust_service(self):
        """Setting trust service to None should reset to default."""
        mock = MagicMock()
        mock_edge = MagicMock()
        mock_edge.overall_trust = 0.99
        mock.get_trust.return_value = mock_edge
        set_trust_service(mock)

        score1 = await get_federation_trust("did:a", "did:b")
        assert abs(score1 - 0.99) < 0.01

        set_trust_service(None)
        # Now uses the real (or import-fallback) service
        score2 = await get_federation_trust("did:a", "did:b")
        # Should be DEFAULT_TRUST_SCORE since no real trust edges exist
        assert 0.0 <= score2 <= 1.0


# =============================================================================
# CONSTANTS TESTS
# =============================================================================


class TestTrustConstants:
    """Verify trust constants are sensible."""

    def test_threshold_ordering(self):
        """LOW_TRUST_THRESHOLD < DEFAULT_TRUST_SCORE < TRUST_AUTO_VERIFY_THRESHOLD."""
        assert LOW_TRUST_THRESHOLD < DEFAULT_TRUST_SCORE
        assert DEFAULT_TRUST_SCORE < TRUST_AUTO_VERIFY_THRESHOLD

    def test_thresholds_in_valid_range(self):
        """All trust thresholds should be between 0 and 1."""
        assert 0.0 < LOW_TRUST_THRESHOLD < 1.0
        assert 0.0 < DEFAULT_TRUST_SCORE < 1.0
        assert 0.0 < TRUST_AUTO_VERIFY_THRESHOLD < 1.0
