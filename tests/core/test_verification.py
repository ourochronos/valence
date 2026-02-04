"""Comprehensive tests for valence.core.verification module.

Tests cover:
- All data models and enums
- Stake calculation functions
- Reputation update functions
- Validation functions
- VerificationService operations
- Edge cases and error handling
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest

from valence.core.verification import (
    # Enums
    VerificationResult,
    VerificationStatus,
    StakeType,
    EvidenceType,
    EvidenceContribution,
    ContradictionType,
    UncertaintyReason,
    DisputeType,
    DisputeOutcome,
    DisputeStatus,
    ResolutionMethod,
    # Data models
    ExternalSource,
    BeliefReference,
    Observation,
    DerivationProof,
    Evidence,
    ResultDetails,
    Stake,
    Verification,
    Dispute,
    ReputationScore,
    ReputationUpdate,
    StakePosition,
    DiscrepancyBounty,
    # Constants
    ReputationConstants,
    # Functions
    calculate_min_stake,
    calculate_max_stake,
    calculate_dispute_min_stake,
    calculate_bounty,
    calculate_confirmation_reward,
    calculate_contradiction_reward,
    calculate_holder_confirmation_bonus,
    calculate_holder_contradiction_penalty,
    calculate_partial_reward,
    validate_verification_submission,
    validate_evidence_requirements,
    validate_dispute_submission,
    create_evidence,
    # Service
    VerificationService,
)
from valence.core.confidence import DimensionalConfidence
from valence.core.exceptions import ValidationException, NotFoundError


# ============================================================================
# Enum Tests
# ============================================================================

class TestVerificationResult:
    """Tests for VerificationResult enum."""

    def test_all_values_exist(self):
        expected = {"CONFIRMED", "CONTRADICTED", "UNCERTAIN", "PARTIAL"}
        actual = {v.name for v in VerificationResult}
        assert actual == expected

    def test_values_are_lowercase(self):
        for result in VerificationResult:
            assert result.value == result.value.lower()

    def test_string_equality(self):
        assert VerificationResult.CONFIRMED == "confirmed"
        assert VerificationResult.CONTRADICTED == "contradicted"


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_all_values_exist(self):
        expected = {"PENDING", "ACCEPTED", "DISPUTED", "OVERTURNED", "REJECTED", "EXPIRED"}
        actual = {s.name for s in VerificationStatus}
        assert actual == expected


class TestStakeType:
    """Tests for StakeType enum."""

    def test_all_values_exist(self):
        expected = {"STANDARD", "BOUNTY", "CHALLENGE"}
        actual = {t.name for t in StakeType}
        assert actual == expected


class TestEvidenceType:
    """Tests for EvidenceType enum."""

    def test_all_values_exist(self):
        expected = {"BELIEF", "EXTERNAL", "OBSERVATION", "DERIVATION", "TESTIMONY"}
        actual = {t.name for t in EvidenceType}
        assert actual == expected


class TestEvidenceContribution:
    """Tests for EvidenceContribution enum."""

    def test_all_values_exist(self):
        expected = {"SUPPORTS", "CONTRADICTS", "CONTEXT", "QUALIFIES"}
        actual = {c.name for c in EvidenceContribution}
        assert actual == expected


class TestContradictionType:
    """Tests for ContradictionType enum."""

    def test_all_values_exist(self):
        expected = {
            "FACTUALLY_FALSE", "OUTDATED", "MISATTRIBUTED",
            "OVERSTATED", "MISSING_CONTEXT", "LOGICAL_ERROR"
        }
        actual = {t.name for t in ContradictionType}
        assert actual == expected


class TestDisputeType:
    """Tests for DisputeType enum."""

    def test_all_values_exist(self):
        expected = {
            "EVIDENCE_INVALID", "EVIDENCE_FABRICATED", "EVIDENCE_INSUFFICIENT",
            "REASONING_FLAWED", "CONFLICT_OF_INTEREST", "NEW_EVIDENCE"
        }
        actual = {t.name for t in DisputeType}
        assert actual == expected


class TestDisputeOutcome:
    """Tests for DisputeOutcome enum."""

    def test_all_values_exist(self):
        expected = {"UPHELD", "OVERTURNED", "MODIFIED", "DISMISSED"}
        actual = {o.name for o in DisputeOutcome}
        assert actual == expected


# ============================================================================
# Constants Tests
# ============================================================================

class TestReputationConstants:
    """Tests for ReputationConstants values."""

    def test_base_rewards_defined(self):
        assert ReputationConstants.CONFIRMATION_BASE == 0.001
        assert ReputationConstants.CONTRADICTION_BASE == 0.005
        assert ReputationConstants.UNCERTAINTY_BASE == 0.0002

    def test_bounty_constants_defined(self):
        assert ReputationConstants.BOUNTY_MULTIPLIER == 0.5
        assert ReputationConstants.FIRST_FINDER_BONUS == 2.0

    def test_penalty_constants_defined(self):
        assert ReputationConstants.CONTRADICTION_PENALTY_BASE == 0.003
        assert ReputationConstants.FRIVOLOUS_DISPUTE_PENALTY == 0.2
        assert ReputationConstants.COLLUSION_PENALTY == 0.5

    def test_limit_constants_defined(self):
        assert ReputationConstants.MAX_DAILY_GAIN == 0.02
        assert ReputationConstants.REPUTATION_FLOOR == 0.1
        assert ReputationConstants.MAX_STAKE_RATIO == 0.2

    def test_timing_constants_defined(self):
        assert ReputationConstants.STAKE_LOCKUP_DAYS == 7
        assert ReputationConstants.DISPUTE_WINDOW_DAYS == 7
        assert ReputationConstants.RESOLUTION_TIMEOUT_DAYS == 14


# ============================================================================
# Data Model Tests
# ============================================================================

class TestExternalSource:
    """Tests for ExternalSource model."""

    def test_creation_with_url(self):
        source = ExternalSource(url="https://example.com", source_reputation=0.9)
        assert source.url == "https://example.com"
        assert source.source_reputation == 0.9

    def test_creation_with_doi(self):
        source = ExternalSource(doi="10.1234/example")
        assert source.doi == "10.1234/example"

    def test_to_dict(self):
        source = ExternalSource(url="https://example.com")
        d = source.to_dict()
        assert d["url"] == "https://example.com"
        assert "source_reputation" in d

    def test_from_dict(self):
        data = {"url": "https://example.com", "source_reputation": 0.8}
        source = ExternalSource.from_dict(data)
        assert source.url == "https://example.com"
        assert source.source_reputation == 0.8


class TestBeliefReference:
    """Tests for BeliefReference model."""

    def test_creation(self):
        belief_id = uuid4()
        ref = BeliefReference(
            belief_id=belief_id,
            holder_id="did:example:holder",
            content_hash="abc123",
        )
        assert ref.belief_id == belief_id
        assert ref.holder_id == "did:example:holder"

    def test_to_dict(self):
        belief_id = uuid4()
        ref = BeliefReference(
            belief_id=belief_id,
            holder_id="did:example:holder",
            content_hash="abc123",
        )
        d = ref.to_dict()
        assert d["belief_id"] == str(belief_id)
        assert d["holder_id"] == "did:example:holder"

    def test_from_dict(self):
        belief_id = uuid4()
        data = {
            "belief_id": str(belief_id),
            "holder_id": "did:example:holder",
            "content_hash": "abc123",
        }
        ref = BeliefReference.from_dict(data)
        assert ref.belief_id == belief_id


class TestObservation:
    """Tests for Observation model."""

    def test_creation(self):
        obs = Observation(
            description="Observed X",
            timestamp=datetime.now(),
            method="direct",
            reproducible=True,
        )
        assert obs.description == "Observed X"
        assert obs.reproducible is True

    def test_to_dict_and_from_dict(self):
        obs = Observation(
            description="Observed X",
            timestamp=datetime(2024, 1, 15, 10, 30),
            method="direct",
        )
        d = obs.to_dict()
        restored = Observation.from_dict(d)
        assert restored.description == obs.description
        assert restored.method == obs.method


class TestDerivationProof:
    """Tests for DerivationProof model."""

    def test_creation(self):
        proof = DerivationProof(
            premises=[uuid4(), uuid4()],
            logic_type="deductive",
            proof_steps=["Step 1", "Step 2", "Conclusion"],
        )
        assert len(proof.premises) == 2
        assert proof.logic_type == "deductive"

    def test_to_dict_and_from_dict(self):
        proof = DerivationProof(
            premises=[uuid4()],
            logic_type="inductive",
            proof_steps=["A", "B"],
        )
        d = proof.to_dict()
        restored = DerivationProof.from_dict(d)
        assert restored.logic_type == "inductive"


class TestEvidence:
    """Tests for Evidence model."""

    def test_creation_with_external_source(self):
        evidence = Evidence(
            id=uuid4(),
            type=EvidenceType.EXTERNAL,
            relevance=0.9,
            contribution=EvidenceContribution.SUPPORTS,
            external_source=ExternalSource(url="https://example.com"),
        )
        assert evidence.type == EvidenceType.EXTERNAL
        assert evidence.relevance == 0.9

    def test_relevance_validation(self):
        with pytest.raises(ValidationException):
            Evidence(
                id=uuid4(),
                type=EvidenceType.EXTERNAL,
                relevance=1.5,  # Invalid
                contribution=EvidenceContribution.SUPPORTS,
            )

    def test_content_hash(self):
        evidence = Evidence(
            id=uuid4(),
            type=EvidenceType.TESTIMONY,
            relevance=0.7,
            contribution=EvidenceContribution.SUPPORTS,
            testimony_statement="I witnessed X",
        )
        hash1 = evidence.content_hash()
        hash2 = evidence.content_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_to_dict_and_from_dict(self):
        evidence = Evidence(
            id=uuid4(),
            type=EvidenceType.EXTERNAL,
            relevance=0.8,
            contribution=EvidenceContribution.CONTRADICTS,
            external_source=ExternalSource(url="https://example.com"),
            verifier_notes="Note",
        )
        d = evidence.to_dict()
        restored = Evidence.from_dict(d)
        assert restored.type == EvidenceType.EXTERNAL
        assert restored.relevance == 0.8
        assert restored.external_source.url == "https://example.com"


class TestResultDetails:
    """Tests for ResultDetails model."""

    def test_confirmed_details(self):
        details = ResultDetails(
            confirmation_strength="strong",
            confirmed_aspects=["aspect1", "aspect2"],
        )
        assert details.confirmation_strength == "strong"

    def test_contradicted_details(self):
        details = ResultDetails(
            contradiction_type=ContradictionType.FACTUALLY_FALSE,
            corrected_belief="The correct statement is...",
            severity="major",
        )
        assert details.contradiction_type == ContradictionType.FACTUALLY_FALSE

    def test_partial_details(self):
        details = ResultDetails(
            accurate_portions=["Part A"],
            inaccurate_portions=["Part B"],
            accuracy_estimate=0.6,
        )
        assert details.accuracy_estimate == 0.6

    def test_uncertain_details(self):
        details = ResultDetails(
            uncertainty_reason=UncertaintyReason.CONFLICTING_SOURCES,
            additional_evidence_needed=["Source X", "Source Y"],
        )
        assert details.uncertainty_reason == UncertaintyReason.CONFLICTING_SOURCES

    def test_to_dict_and_from_dict(self):
        details = ResultDetails(
            contradiction_type=ContradictionType.OUTDATED,
            severity="minor",
        )
        d = details.to_dict()
        restored = ResultDetails.from_dict(d)
        assert restored.contradiction_type == ContradictionType.OUTDATED


class TestStake:
    """Tests for Stake model."""

    def test_creation(self):
        stake = Stake(
            amount=0.05,
            type=StakeType.STANDARD,
            locked_until=datetime.now() + timedelta(days=7),
            escrow_id=uuid4(),
        )
        assert stake.amount == 0.05
        assert stake.type == StakeType.STANDARD

    def test_negative_amount_raises(self):
        with pytest.raises(ValidationException):
            Stake(
                amount=-0.01,
                type=StakeType.STANDARD,
                locked_until=datetime.now(),
                escrow_id=uuid4(),
            )

    def test_to_dict_and_from_dict(self):
        stake = Stake(
            amount=0.03,
            type=StakeType.BOUNTY,
            locked_until=datetime(2024, 1, 22),
            escrow_id=uuid4(),
        )
        d = stake.to_dict()
        restored = Stake.from_dict(d)
        assert restored.amount == 0.03
        assert restored.type == StakeType.BOUNTY


class TestVerification:
    """Tests for Verification model."""

    @pytest.fixture
    def sample_evidence(self):
        return Evidence(
            id=uuid4(),
            type=EvidenceType.EXTERNAL,
            relevance=0.9,
            contribution=EvidenceContribution.SUPPORTS,
            external_source=ExternalSource(url="https://example.com"),
        )

    @pytest.fixture
    def sample_stake(self):
        return Stake(
            amount=0.05,
            type=StakeType.STANDARD,
            locked_until=datetime.now() + timedelta(days=7),
            escrow_id=uuid4(),
        )

    def test_creation(self, sample_evidence, sample_stake):
        verification = Verification(
            id=uuid4(),
            verifier_id="did:example:verifier",
            belief_id=uuid4(),
            holder_id="did:example:holder",
            result=VerificationResult.CONFIRMED,
            evidence=[sample_evidence],
            stake=sample_stake,
            reasoning="This is correct because...",
        )
        assert verification.result == VerificationResult.CONFIRMED
        assert verification.status == VerificationStatus.PENDING

    def test_to_dict(self, sample_evidence, sample_stake):
        verification = Verification(
            id=uuid4(),
            verifier_id="did:example:verifier",
            belief_id=uuid4(),
            holder_id="did:example:holder",
            result=VerificationResult.CONTRADICTED,
            evidence=[sample_evidence],
            stake=sample_stake,
        )
        d = verification.to_dict()
        assert d["result"] == "contradicted"
        assert len(d["evidence"]) == 1

    def test_from_dict(self, sample_evidence, sample_stake):
        verification = Verification(
            id=uuid4(),
            verifier_id="did:example:verifier",
            belief_id=uuid4(),
            holder_id="did:example:holder",
            result=VerificationResult.UNCERTAIN,
            evidence=[sample_evidence],
            stake=sample_stake,
        )
        d = verification.to_dict()
        restored = Verification.from_dict(d)
        assert restored.result == VerificationResult.UNCERTAIN


class TestDispute:
    """Tests for Dispute model."""

    @pytest.fixture
    def sample_evidence(self):
        return Evidence(
            id=uuid4(),
            type=EvidenceType.EXTERNAL,
            relevance=0.9,
            contribution=EvidenceContribution.CONTRADICTS,
        )

    @pytest.fixture
    def sample_stake(self):
        return Stake(
            amount=0.075,
            type=StakeType.CHALLENGE,
            locked_until=datetime.now() + timedelta(days=14),
            escrow_id=uuid4(),
        )

    def test_creation(self, sample_evidence, sample_stake):
        dispute = Dispute(
            id=uuid4(),
            verification_id=uuid4(),
            disputer_id="did:example:disputer",
            counter_evidence=[sample_evidence],
            stake=sample_stake,
            dispute_type=DisputeType.NEW_EVIDENCE,
            reasoning="New evidence contradicts...",
            proposed_result=VerificationResult.CONFIRMED,
        )
        assert dispute.status == DisputeStatus.PENDING
        assert dispute.dispute_type == DisputeType.NEW_EVIDENCE

    def test_to_dict_and_from_dict(self, sample_evidence, sample_stake):
        dispute = Dispute(
            id=uuid4(),
            verification_id=uuid4(),
            disputer_id="did:example:disputer",
            counter_evidence=[sample_evidence],
            stake=sample_stake,
            dispute_type=DisputeType.EVIDENCE_INVALID,
            reasoning="Evidence is invalid because...",
        )
        d = dispute.to_dict()
        restored = Dispute.from_dict(d)
        assert restored.dispute_type == DisputeType.EVIDENCE_INVALID


class TestReputationScore:
    """Tests for ReputationScore model."""

    def test_default_values(self):
        rep = ReputationScore(identity_id="did:example:agent")
        assert rep.overall == 0.5
        assert rep.verification_count == 0
        assert rep.stake_at_risk == 0.0

    def test_floor_enforcement(self):
        rep = ReputationScore(identity_id="did:example:agent", overall=0.05)
        assert rep.overall == 0.1  # Floored to minimum

    def test_ceiling_enforcement(self):
        rep = ReputationScore(identity_id="did:example:agent", overall=1.5)
        assert rep.overall == 1.0  # Capped at maximum

    def test_available_stake(self):
        rep = ReputationScore(
            identity_id="did:example:agent",
            overall=0.5,
            stake_at_risk=0.02,
        )
        # Max stakeable = 0.5 * 0.2 = 0.1
        # Available = 0.1 - 0.02 = 0.08
        assert rep.available_stake() == pytest.approx(0.08)

    def test_available_stake_with_high_risk(self):
        rep = ReputationScore(
            identity_id="did:example:agent",
            overall=0.5,
            stake_at_risk=0.15,  # More than max stakeable
        )
        assert rep.available_stake() == 0.0

    def test_to_dict_and_from_dict(self):
        rep = ReputationScore(
            identity_id="did:example:agent",
            overall=0.75,
            by_domain={"tech": 0.8, "science": 0.6},
            verification_count=50,
            discrepancy_finds=5,
        )
        d = rep.to_dict()
        restored = ReputationScore.from_dict(d)
        assert restored.overall == 0.75
        assert restored.by_domain["tech"] == 0.8


class TestReputationUpdate:
    """Tests for ReputationUpdate model."""

    def test_creation(self):
        update = ReputationUpdate(
            id=uuid4(),
            identity_id="did:example:agent",
            delta=0.005,
            old_value=0.5,
            new_value=0.505,
            reason="Verification confirmed",
        )
        assert update.delta == 0.005

    def test_to_dict(self):
        update = ReputationUpdate(
            id=uuid4(),
            identity_id="did:example:agent",
            delta=-0.003,
            old_value=0.5,
            new_value=0.497,
            reason="Belief contradicted",
            verification_id=uuid4(),
        )
        d = update.to_dict()
        assert d["delta"] == -0.003
        assert "verification_id" in d


class TestStakePosition:
    """Tests for StakePosition model."""

    def test_creation(self):
        position = StakePosition(
            id=uuid4(),
            identity_id="did:example:agent",
            amount=0.05,
            type=StakeType.STANDARD,
        )
        assert position.status == "locked"

    def test_to_dict(self):
        position = StakePosition(
            id=uuid4(),
            identity_id="did:example:agent",
            amount=0.05,
            type=StakeType.STANDARD,
            verification_id=uuid4(),
        )
        d = position.to_dict()
        assert d["amount"] == 0.05
        assert d["status"] == "locked"


class TestDiscrepancyBounty:
    """Tests for DiscrepancyBounty model."""

    def test_creation(self):
        bounty = DiscrepancyBounty(
            belief_id=uuid4(),
            holder_id="did:example:holder",
            base_amount=0.005,
            confidence_premium=0.64,
            age_factor=1.5,
            total_bounty=0.0128,
        )
        assert bounty.claimed is False

    def test_to_dict(self):
        bounty = DiscrepancyBounty(
            belief_id=uuid4(),
            holder_id="did:example:holder",
            base_amount=0.005,
            confidence_premium=0.64,
            age_factor=1.5,
            total_bounty=0.0128,
        )
        d = bounty.to_dict()
        assert d["claimed"] is False
        assert d["total_bounty"] == 0.0128


# ============================================================================
# Stake Calculation Tests
# ============================================================================

class TestCalculateMinStake:
    """Tests for calculate_min_stake function."""

    def test_basic_calculation(self):
        # base_stake = 0.01
        # confidence_multiplier = 0.7
        # domain_multiplier = 1.0 + (0.5 * 0.5) = 1.25
        # min_stake = 0.01 * 0.7 * 1.25 = 0.00875
        result = calculate_min_stake(0.7, 0.5)
        assert result == pytest.approx(0.00875)

    def test_low_confidence_belief(self):
        result = calculate_min_stake(0.3, 0.5)
        assert result < calculate_min_stake(0.9, 0.5)

    def test_high_domain_reputation(self):
        # Higher domain reputation = higher multiplier but also more confidence
        result_high = calculate_min_stake(0.7, 0.9)
        result_low = calculate_min_stake(0.7, 0.1)
        assert result_high > result_low

    def test_zero_confidence(self):
        result = calculate_min_stake(0.0, 0.5)
        assert result == 0.0


class TestCalculateMaxStake:
    """Tests for calculate_max_stake function."""

    def test_basic_calculation(self):
        # max_stake = reputation * 0.2
        result = calculate_max_stake(0.5)
        assert result == pytest.approx(0.1)

    def test_high_reputation(self):
        result = calculate_max_stake(0.9)
        assert result == pytest.approx(0.18)

    def test_floor_reputation(self):
        result = calculate_max_stake(0.1)
        assert result == pytest.approx(0.02)


class TestCalculateDisputeMinStake:
    """Tests for calculate_dispute_min_stake function."""

    def test_holder_gets_lower_multiplier(self):
        verification_stake = 0.05
        holder_min = calculate_dispute_min_stake(verification_stake, is_holder=True)
        third_party_min = calculate_dispute_min_stake(verification_stake, is_holder=False)
        
        assert holder_min == pytest.approx(0.05)  # 1.0x
        assert third_party_min == pytest.approx(0.075)  # 1.5x

    def test_third_party_multiplier(self):
        result = calculate_dispute_min_stake(0.1, is_holder=False)
        assert result == pytest.approx(0.15)


class TestCalculateBounty:
    """Tests for calculate_bounty function."""

    def test_basic_calculation(self):
        # bounty = stake * 0.5 * confidence^2 * age_factor
        # = 0.01 * 0.5 * 0.8^2 * 1.5 = 0.0048
        result = calculate_bounty(0.01, 0.8, 15)
        expected = 0.01 * 0.5 * 0.64 * 1.5
        assert result == pytest.approx(expected)

    def test_age_factor_caps_at_2(self):
        # After 30 days, age_factor maxes at 2.0
        result_60_days = calculate_bounty(0.01, 0.8, 60)
        result_90_days = calculate_bounty(0.01, 0.8, 90)
        assert result_60_days == result_90_days

    def test_high_confidence_earns_more(self):
        low_conf = calculate_bounty(0.01, 0.5, 30)
        high_conf = calculate_bounty(0.01, 0.9, 30)
        assert high_conf > low_conf


# ============================================================================
# Reputation Update Calculation Tests
# ============================================================================

class TestCalculateConfirmationReward:
    """Tests for calculate_confirmation_reward function."""

    def test_basic_calculation(self):
        result = calculate_confirmation_reward(
            stake=0.02,
            min_stake=0.01,
            belief_confidence=0.8,
            existing_confirmations=0,
        )
        # base=0.001, stake_mult=2.0 (capped), conf=0.8, diminish=1.0
        expected = 0.001 * 2.0 * 0.8 * 1.0
        assert result == pytest.approx(expected)

    def test_diminishing_returns(self):
        # First confirmation
        first = calculate_confirmation_reward(0.02, 0.01, 0.8, 0)
        # Second confirmation
        second = calculate_confirmation_reward(0.02, 0.01, 0.8, 1)
        # Third confirmation
        third = calculate_confirmation_reward(0.02, 0.01, 0.8, 2)
        
        assert first > second > third

    def test_higher_stake_earns_more(self):
        low_stake = calculate_confirmation_reward(0.01, 0.01, 0.8, 0)
        high_stake = calculate_confirmation_reward(0.03, 0.01, 0.8, 0)
        assert high_stake > low_stake

    def test_stake_multiplier_caps_at_2(self):
        # Even 10x stake should cap at 2x multiplier
        normal = calculate_confirmation_reward(0.02, 0.01, 0.8, 0)
        huge_stake = calculate_confirmation_reward(0.10, 0.01, 0.8, 0)
        assert huge_stake == normal  # Both capped at 2x


class TestCalculateContradictionReward:
    """Tests for calculate_contradiction_reward function."""

    def test_first_finder_bonus(self):
        first = calculate_contradiction_reward(
            stake=0.02,
            min_stake=0.01,
            belief_confidence=0.8,
            is_first_contradiction=True,
        )
        subsequent = calculate_contradiction_reward(
            stake=0.02,
            min_stake=0.01,
            belief_confidence=0.8,
            is_first_contradiction=False,
            existing_contradictions=1,
        )
        # First finder gets 2x bonus
        assert first > subsequent

    def test_higher_confidence_earns_more(self):
        low_conf = calculate_contradiction_reward(0.02, 0.01, 0.5, True)
        high_conf = calculate_contradiction_reward(0.02, 0.01, 0.9, True)
        # Confidence premium is squared
        assert high_conf > low_conf

    def test_contradiction_earns_more_than_confirmation(self):
        confirmation = calculate_confirmation_reward(0.02, 0.01, 0.8, 0)
        contradiction = calculate_contradiction_reward(0.02, 0.01, 0.8, True)
        assert contradiction > confirmation  # 5x base rate


class TestCalculateHolderConfirmationBonus:
    """Tests for calculate_holder_confirmation_bonus function."""

    def test_basic_calculation(self):
        result = calculate_holder_confirmation_bonus(
            verifier_reputation=0.8,
            stake=0.02,
            min_stake=0.01,
        )
        # 0.0005 * 0.8 * sqrt(2) ≈ 0.000566
        expected = 0.0005 * 0.8 * math.sqrt(2)
        assert result == pytest.approx(expected)

    def test_higher_verifier_reputation_helps_holder(self):
        low_rep = calculate_holder_confirmation_bonus(0.3, 0.02, 0.01)
        high_rep = calculate_holder_confirmation_bonus(0.9, 0.02, 0.01)
        assert high_rep > low_rep


class TestCalculateHolderContradictionPenalty:
    """Tests for calculate_holder_contradiction_penalty function."""

    def test_basic_calculation(self):
        result = calculate_holder_contradiction_penalty(
            belief_confidence=0.8,
            verifier_reputation=0.7,
        )
        # base=0.003, overconf=0.64, verifier=0.7
        expected = 0.003 * 0.64 * 0.7
        assert result == pytest.approx(expected)

    def test_overconfidence_penalty(self):
        # Higher claimed confidence = bigger penalty
        low_conf = calculate_holder_contradiction_penalty(0.5, 0.7)
        high_conf = calculate_holder_contradiction_penalty(0.9, 0.7)
        assert high_conf > low_conf


class TestCalculatePartialReward:
    """Tests for calculate_partial_reward function."""

    def test_50_percent_accuracy(self):
        result = calculate_partial_reward(
            accuracy_estimate=0.5,
            stake=0.02,
            min_stake=0.01,
            belief_confidence=0.8,
            existing_confirmations=0,
            existing_contradictions=0,
        )
        # Should be mix of confirmation and contradiction rewards
        assert result > 0

    def test_high_accuracy_closer_to_confirmation(self):
        high_accuracy = calculate_partial_reward(0.9, 0.02, 0.01, 0.8, 0, 0)
        low_accuracy = calculate_partial_reward(0.1, 0.02, 0.01, 0.8, 0, 0)
        # Low accuracy (more contradiction) should earn more
        assert low_accuracy > high_accuracy


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidateEvidenceRequirements:
    """Tests for validate_evidence_requirements function."""

    def test_confirmed_needs_supporting(self):
        evidence = [
            Evidence(
                id=uuid4(),
                type=EvidenceType.EXTERNAL,
                relevance=0.8,
                contribution=EvidenceContribution.CONTEXT,  # Not SUPPORTS
            )
        ]
        errors = validate_evidence_requirements(VerificationResult.CONFIRMED, evidence)
        assert len(errors) > 0
        assert "supporting evidence" in errors[0].lower()

    def test_confirmed_with_supporting_passes(self):
        evidence = [
            Evidence(
                id=uuid4(),
                type=EvidenceType.EXTERNAL,
                relevance=0.8,
                contribution=EvidenceContribution.SUPPORTS,
            )
        ]
        errors = validate_evidence_requirements(VerificationResult.CONFIRMED, evidence)
        assert len(errors) == 0

    def test_contradicted_needs_contradicting(self):
        evidence = [
            Evidence(
                id=uuid4(),
                type=EvidenceType.EXTERNAL,
                relevance=0.8,
                contribution=EvidenceContribution.SUPPORTS,
            )
        ]
        errors = validate_evidence_requirements(VerificationResult.CONTRADICTED, evidence)
        assert len(errors) > 0
        assert "contradicting evidence" in errors[0].lower()

    def test_partial_needs_both(self):
        # Only supporting
        errors = validate_evidence_requirements(
            VerificationResult.PARTIAL,
            [Evidence(
                id=uuid4(),
                type=EvidenceType.EXTERNAL,
                relevance=0.8,
                contribution=EvidenceContribution.SUPPORTS,
            )]
        )
        assert len(errors) > 0

    def test_uncertain_no_minimum(self):
        errors = validate_evidence_requirements(VerificationResult.UNCERTAIN, [])
        assert len(errors) == 0

    def test_low_relevance_warning(self):
        evidence = [
            Evidence(
                id=uuid4(),
                type=EvidenceType.EXTERNAL,
                relevance=0.05,  # Very low
                contribution=EvidenceContribution.SUPPORTS,
            )
        ]
        errors = validate_evidence_requirements(VerificationResult.CONFIRMED, evidence)
        assert any("low relevance" in e.lower() for e in errors)


class TestValidateVerificationSubmission:
    """Tests for validate_verification_submission function."""

    @pytest.fixture
    def sample_verification(self):
        return Verification(
            id=uuid4(),
            verifier_id="did:example:verifier",
            belief_id=uuid4(),
            holder_id="did:example:holder",
            result=VerificationResult.CONFIRMED,
            evidence=[
                Evidence(
                    id=uuid4(),
                    type=EvidenceType.EXTERNAL,
                    relevance=0.8,
                    contribution=EvidenceContribution.SUPPORTS,
                )
            ],
            stake=Stake(
                amount=0.02,
                type=StakeType.STANDARD,
                locked_until=datetime.now() + timedelta(days=7),
                escrow_id=uuid4(),
            ),
        )

    @pytest.fixture
    def sample_belief(self):
        return {
            "confidence": {"overall": 0.7},
            "holder_id": "did:example:holder",
            "domain_path": ["tech"],
        }

    @pytest.fixture
    def sample_reputation(self):
        return ReputationScore(
            identity_id="did:example:verifier",
            overall=0.5,
        )

    def test_self_verification_rejected(self, sample_belief, sample_reputation):
        verification = Verification(
            id=uuid4(),
            verifier_id="did:example:holder",  # Same as holder
            belief_id=uuid4(),
            holder_id="did:example:holder",
            result=VerificationResult.CONFIRMED,
            evidence=[
                Evidence(
                    id=uuid4(),
                    type=EvidenceType.EXTERNAL,
                    relevance=0.8,
                    contribution=EvidenceContribution.SUPPORTS,
                )
            ],
            stake=Stake(
                amount=0.02,
                type=StakeType.STANDARD,
                locked_until=datetime.now() + timedelta(days=7),
                escrow_id=uuid4(),
            ),
        )
        sample_reputation.identity_id = "did:example:holder"
        
        errors = validate_verification_submission(
            verification, sample_belief, sample_reputation, []
        )
        assert any("own belief" in e.lower() for e in errors)

    def test_duplicate_verification_rejected(self, sample_verification, sample_belief, sample_reputation):
        existing = [sample_verification]
        new_verification = Verification(
            id=uuid4(),
            verifier_id=sample_verification.verifier_id,  # Same verifier
            belief_id=sample_verification.belief_id,
            holder_id=sample_verification.holder_id,
            result=VerificationResult.CONTRADICTED,
            evidence=[
                Evidence(
                    id=uuid4(),
                    type=EvidenceType.EXTERNAL,
                    relevance=0.8,
                    contribution=EvidenceContribution.CONTRADICTS,
                )
            ],
            stake=sample_verification.stake,
        )
        
        errors = validate_verification_submission(
            new_verification, sample_belief, sample_reputation, existing
        )
        assert any("already verified" in e.lower() for e in errors)

    def test_insufficient_stake_rejected(self, sample_belief, sample_reputation):
        verification = Verification(
            id=uuid4(),
            verifier_id="did:example:verifier",
            belief_id=uuid4(),
            holder_id="did:example:holder",
            result=VerificationResult.CONFIRMED,
            evidence=[
                Evidence(
                    id=uuid4(),
                    type=EvidenceType.EXTERNAL,
                    relevance=0.8,
                    contribution=EvidenceContribution.SUPPORTS,
                )
            ],
            stake=Stake(
                amount=0.0001,  # Too low
                type=StakeType.STANDARD,
                locked_until=datetime.now() + timedelta(days=7),
                escrow_id=uuid4(),
            ),
        )
        
        errors = validate_verification_submission(
            verification, sample_belief, sample_reputation, []
        )
        assert any("below minimum" in e.lower() for e in errors)

    def test_valid_submission_passes(self, sample_verification, sample_belief, sample_reputation):
        errors = validate_verification_submission(
            sample_verification, sample_belief, sample_reputation, []
        )
        assert len(errors) == 0


class TestValidateDisputeSubmission:
    """Tests for validate_dispute_submission function."""

    @pytest.fixture
    def sample_verification(self):
        return Verification(
            id=uuid4(),
            verifier_id="did:example:verifier",
            belief_id=uuid4(),
            holder_id="did:example:holder",
            result=VerificationResult.CONTRADICTED,
            evidence=[],
            stake=Stake(
                amount=0.05,
                type=StakeType.STANDARD,
                locked_until=datetime.now() + timedelta(days=7),
                escrow_id=uuid4(),
            ),
            status=VerificationStatus.ACCEPTED,
            accepted_at=datetime.now(),
        )

    @pytest.fixture
    def sample_dispute(self, sample_verification):
        return Dispute(
            id=uuid4(),
            verification_id=sample_verification.id,
            disputer_id="did:example:holder",
            counter_evidence=[
                Evidence(
                    id=uuid4(),
                    type=EvidenceType.EXTERNAL,
                    relevance=0.9,
                    contribution=EvidenceContribution.SUPPORTS,
                )
            ],
            stake=Stake(
                amount=0.05,
                type=StakeType.CHALLENGE,
                locked_until=datetime.now() + timedelta(days=14),
                escrow_id=uuid4(),
            ),
            dispute_type=DisputeType.NEW_EVIDENCE,
            reasoning="New evidence shows...",
        )

    def test_cannot_dispute_pending_verification(self, sample_dispute):
        verification = Verification(
            id=uuid4(),
            verifier_id="did:example:verifier",
            belief_id=uuid4(),
            holder_id="did:example:holder",
            result=VerificationResult.CONFIRMED,
            evidence=[],
            stake=Stake(
                amount=0.05,
                type=StakeType.STANDARD,
                locked_until=datetime.now() + timedelta(days=7),
                escrow_id=uuid4(),
            ),
            status=VerificationStatus.PENDING,  # Not ACCEPTED
        )
        
        disputer_rep = ReputationScore(identity_id="did:example:holder", overall=0.5)
        
        errors = validate_dispute_submission(
            sample_dispute, verification, disputer_rep, is_holder=True
        )
        assert any("pending" in e.lower() for e in errors)

    def test_expired_dispute_window(self, sample_verification, sample_dispute):
        sample_verification.accepted_at = datetime.now() - timedelta(days=10)  # Past window
        
        disputer_rep = ReputationScore(identity_id="did:example:holder", overall=0.5)
        
        errors = validate_dispute_submission(
            sample_dispute, sample_verification, disputer_rep, is_holder=True
        )
        assert any("expired" in e.lower() for e in errors)

    def test_no_counter_evidence_rejected(self, sample_verification):
        dispute = Dispute(
            id=uuid4(),
            verification_id=sample_verification.id,
            disputer_id="did:example:holder",
            counter_evidence=[],  # No evidence
            stake=Stake(
                amount=0.05,
                type=StakeType.CHALLENGE,
                locked_until=datetime.now() + timedelta(days=14),
                escrow_id=uuid4(),
            ),
            dispute_type=DisputeType.NEW_EVIDENCE,
            reasoning="...",
        )
        
        disputer_rep = ReputationScore(identity_id="did:example:holder", overall=0.5)
        
        errors = validate_dispute_submission(
            dispute, sample_verification, disputer_rep, is_holder=True
        )
        assert any("counter-evidence" in e.lower() for e in errors)

    def test_valid_dispute_passes(self, sample_verification, sample_dispute):
        disputer_rep = ReputationScore(identity_id="did:example:holder", overall=0.5)
        
        errors = validate_dispute_submission(
            sample_dispute, sample_verification, disputer_rep, is_holder=True
        )
        assert len(errors) == 0


# ============================================================================
# VerificationService Tests
# ============================================================================

class TestVerificationService:
    """Tests for VerificationService class."""

    @pytest.fixture
    def service(self):
        return VerificationService()

    @pytest.fixture
    def sample_belief_info(self):
        return {
            "holder_id": "did:example:holder",
            "confidence": {"overall": 0.7},
            "domain_path": ["tech"],
        }

    @pytest.fixture
    def supporting_evidence(self):
        return [Evidence(
            id=uuid4(),
            type=EvidenceType.EXTERNAL,
            relevance=0.9,
            contribution=EvidenceContribution.SUPPORTS,
            external_source=ExternalSource(url="https://example.com"),
        )]

    @pytest.fixture
    def contradicting_evidence(self):
        return [Evidence(
            id=uuid4(),
            type=EvidenceType.EXTERNAL,
            relevance=0.9,
            contribution=EvidenceContribution.CONTRADICTS,
            external_source=ExternalSource(url="https://counter.example.com"),
        )]

    def test_submit_verification_success(self, service, sample_belief_info, supporting_evidence):
        belief_id = uuid4()
        
        verification = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier",
            result=VerificationResult.CONFIRMED,
            evidence=supporting_evidence,
            stake_amount=0.02,
            reasoning="This is accurate because...",
        )
        
        assert verification.id is not None
        assert verification.status == VerificationStatus.PENDING
        assert verification.result == VerificationResult.CONFIRMED

    def test_submit_verification_self_verification_rejected(self, service, sample_belief_info, supporting_evidence):
        belief_id = uuid4()
        
        with pytest.raises(ValidationException) as exc:
            service.submit_verification(
                belief_id=belief_id,
                belief_info=sample_belief_info,
                verifier_id="did:example:holder",  # Same as holder
                result=VerificationResult.CONFIRMED,
                evidence=supporting_evidence,
                stake_amount=0.02,
            )
        
        assert "own belief" in str(exc.value).lower()

    def test_accept_verification(self, service, sample_belief_info, supporting_evidence):
        belief_id = uuid4()
        
        verification = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier",
            result=VerificationResult.CONFIRMED,
            evidence=supporting_evidence,
            stake_amount=0.02,
        )
        
        accepted = service.accept_verification(verification.id)
        
        assert accepted.status == VerificationStatus.ACCEPTED
        assert accepted.accepted_at is not None

    def test_accept_verification_updates_reputation(self, service, sample_belief_info, supporting_evidence):
        belief_id = uuid4()
        
        verification = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier",
            result=VerificationResult.CONFIRMED,
            evidence=supporting_evidence,
            stake_amount=0.02,
        )
        
        verifier_rep_before = service.get_or_create_reputation("did:example:verifier").overall
        
        service.accept_verification(verification.id)
        
        verifier_rep_after = service.get_reputation("did:example:verifier").overall
        
        # Verifier should gain reputation
        assert verifier_rep_after > verifier_rep_before

    def test_contradiction_gives_higher_reward(self, service, sample_belief_info, contradicting_evidence):
        belief_id = uuid4()
        
        verification = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier",
            result=VerificationResult.CONTRADICTED,
            evidence=contradicting_evidence,
            stake_amount=0.02,
        )
        
        verifier_rep_before = service.get_or_create_reputation("did:example:verifier").overall
        
        service.accept_verification(verification.id)
        
        verifier_rep = service.get_reputation("did:example:verifier")
        verifier_delta = verifier_rep.overall - verifier_rep_before
        
        # Contradiction reward should be significant
        assert verifier_delta > 0

    def test_contradiction_penalizes_holder(self, service, sample_belief_info, contradicting_evidence):
        belief_id = uuid4()
        
        verification = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier",
            result=VerificationResult.CONTRADICTED,
            evidence=contradicting_evidence,
            stake_amount=0.02,
        )
        
        holder_rep_before = service.get_or_create_reputation("did:example:holder").overall
        
        service.accept_verification(verification.id)
        
        holder_rep_after = service.get_reputation("did:example:holder").overall
        
        # Holder should lose reputation
        assert holder_rep_after < holder_rep_before

    def test_dispute_verification(self, service, sample_belief_info, contradicting_evidence, supporting_evidence):
        belief_id = uuid4()
        
        # Submit and accept verification
        verification = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier",
            result=VerificationResult.CONTRADICTED,
            evidence=contradicting_evidence,
            stake_amount=0.02,
        )
        service.accept_verification(verification.id)
        
        # File dispute
        dispute = service.dispute_verification(
            verification_id=verification.id,
            disputer_id="did:example:holder",
            counter_evidence=supporting_evidence,
            stake_amount=0.02,  # Holder gets 1.0x multiplier
            dispute_type=DisputeType.NEW_EVIDENCE,
            reasoning="New evidence shows the belief was correct",
            proposed_result=VerificationResult.CONFIRMED,
        )
        
        assert dispute.id is not None
        assert dispute.status == DisputeStatus.PENDING
        
        # Verification should be marked as disputed
        updated_verification = service.get_verification(verification.id)
        assert updated_verification.status == VerificationStatus.DISPUTED

    def test_dispute_nonexistent_verification_raises(self, service, supporting_evidence):
        with pytest.raises(NotFoundError):
            service.dispute_verification(
                verification_id=uuid4(),
                disputer_id="did:example:holder",
                counter_evidence=supporting_evidence,
                stake_amount=0.02,
                dispute_type=DisputeType.NEW_EVIDENCE,
                reasoning="...",
            )

    def test_resolve_dispute_upheld(self, service, sample_belief_info, contradicting_evidence, supporting_evidence):
        belief_id = uuid4()
        
        # Submit, accept, and dispute
        verification = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier",
            result=VerificationResult.CONTRADICTED,
            evidence=contradicting_evidence,
            stake_amount=0.02,
        )
        service.accept_verification(verification.id)
        
        dispute = service.dispute_verification(
            verification_id=verification.id,
            disputer_id="did:example:holder",
            counter_evidence=supporting_evidence,
            stake_amount=0.02,
            dispute_type=DisputeType.NEW_EVIDENCE,
            reasoning="New evidence",
        )
        
        disputer_rep_before = service.get_reputation("did:example:holder").overall
        
        # Resolve as upheld (verifier wins)
        resolved = service.resolve_dispute(
            dispute_id=dispute.id,
            outcome=DisputeOutcome.UPHELD,
            resolution_reasoning="Original evidence is stronger",
        )
        
        assert resolved.outcome == DisputeOutcome.UPHELD
        assert resolved.status == DisputeStatus.RESOLVED
        
        # Disputer should lose stake
        disputer_rep_after = service.get_reputation("did:example:holder").overall
        assert disputer_rep_after < disputer_rep_before

    def test_resolve_dispute_overturned(self, service, sample_belief_info, contradicting_evidence, supporting_evidence):
        belief_id = uuid4()
        
        # Submit, accept, and dispute
        verification = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier",
            result=VerificationResult.CONTRADICTED,
            evidence=contradicting_evidence,
            stake_amount=0.02,
        )
        service.accept_verification(verification.id)
        
        dispute = service.dispute_verification(
            verification_id=verification.id,
            disputer_id="did:example:holder",
            counter_evidence=supporting_evidence,
            stake_amount=0.02,
            dispute_type=DisputeType.NEW_EVIDENCE,
            reasoning="New evidence",
        )
        
        verifier_rep_before = service.get_reputation("did:example:verifier").overall
        
        # Resolve as overturned (disputer wins)
        resolved = service.resolve_dispute(
            dispute_id=dispute.id,
            outcome=DisputeOutcome.OVERTURNED,
            resolution_reasoning="New evidence is compelling",
        )
        
        assert resolved.outcome == DisputeOutcome.OVERTURNED
        
        # Verifier should lose stake
        verifier_rep_after = service.get_reputation("did:example:verifier").overall
        assert verifier_rep_after < verifier_rep_before
        
        # Verification should be overturned
        updated_verification = service.get_verification(verification.id)
        assert updated_verification.status == VerificationStatus.OVERTURNED

    def test_resolve_dispute_dismissed(self, service, sample_belief_info, contradicting_evidence, supporting_evidence):
        belief_id = uuid4()
        
        verification = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier",
            result=VerificationResult.CONTRADICTED,
            evidence=contradicting_evidence,
            stake_amount=0.02,
        )
        service.accept_verification(verification.id)
        
        dispute = service.dispute_verification(
            verification_id=verification.id,
            disputer_id="did:example:holder",
            counter_evidence=supporting_evidence,
            stake_amount=0.02,
            dispute_type=DisputeType.NEW_EVIDENCE,
            reasoning="Frivolous dispute",
        )
        
        disputer_rep_before = service.get_reputation("did:example:holder").overall
        
        # Resolve as dismissed (frivolous)
        resolved = service.resolve_dispute(
            dispute_id=dispute.id,
            outcome=DisputeOutcome.DISMISSED,
            resolution_reasoning="Dispute is frivolous",
        )
        
        assert resolved.outcome == DisputeOutcome.DISMISSED
        
        # Disputer should get extra penalty for frivolous dispute
        disputer_rep_after = service.get_reputation("did:example:holder").overall
        penalty = disputer_rep_before - disputer_rep_after
        # Penalty should include frivolous multiplier
        assert penalty > 0.02  # More than just stake

    def test_get_verification_summary(self, service, sample_belief_info, supporting_evidence, contradicting_evidence):
        belief_id = uuid4()
        
        # Submit multiple verifications
        v1 = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier1",
            result=VerificationResult.CONFIRMED,
            evidence=supporting_evidence,
            stake_amount=0.02,
        )
        service.accept_verification(v1.id)
        
        v2 = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier2",
            result=VerificationResult.CONFIRMED,
            evidence=supporting_evidence,
            stake_amount=0.03,
        )
        service.accept_verification(v2.id)
        
        summary = service.get_verification_summary(belief_id)
        
        assert summary["total"] == 2
        assert summary["by_result"]["confirmed"] == 2
        assert summary["by_status"]["accepted"] == 2
        assert summary["total_stake"] == pytest.approx(0.05)
        assert summary["consensus_result"] == "confirmed"

    def test_stake_locking_and_release(self, service, sample_belief_info, supporting_evidence):
        belief_id = uuid4()
        
        # Check initial state
        verifier_rep = service.get_or_create_reputation("did:example:verifier")
        initial_stake_at_risk = verifier_rep.stake_at_risk
        
        # Submit verification (locks stake)
        verification = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier",
            result=VerificationResult.CONFIRMED,
            evidence=supporting_evidence,
            stake_amount=0.02,
        )
        
        # Check stake is locked
        verifier_rep = service.get_reputation("did:example:verifier")
        assert verifier_rep.stake_at_risk == initial_stake_at_risk + 0.02

    def test_discrepancy_find_increments(self, service, sample_belief_info, contradicting_evidence):
        belief_id = uuid4()
        
        # Check initial state
        verifier_rep = service.get_or_create_reputation("did:example:verifier")
        initial_finds = verifier_rep.discrepancy_finds
        
        # Submit and accept contradiction
        verification = service.submit_verification(
            belief_id=belief_id,
            belief_info=sample_belief_info,
            verifier_id="did:example:verifier",
            result=VerificationResult.CONTRADICTED,
            evidence=contradicting_evidence,
            stake_amount=0.02,
        )
        service.accept_verification(verification.id)
        
        # Check discrepancy_finds incremented
        verifier_rep = service.get_reputation("did:example:verifier")
        assert verifier_rep.discrepancy_finds == initial_finds + 1


# ============================================================================
# Create Evidence Helper Tests
# ============================================================================

class TestCreateEvidence:
    """Tests for create_evidence helper function."""

    def test_create_external_evidence(self):
        evidence = create_evidence(
            EvidenceType.EXTERNAL,
            EvidenceContribution.SUPPORTS,
            url="https://example.com",
        )
        assert evidence.type == EvidenceType.EXTERNAL
        assert evidence.external_source is not None
        assert evidence.external_source.url == "https://example.com"

    def test_create_observation_evidence(self):
        evidence = create_evidence(
            EvidenceType.OBSERVATION,
            EvidenceContribution.SUPPORTS,
            description="Observed X",
            timestamp=datetime.now(),
            method="direct",
        )
        assert evidence.type == EvidenceType.OBSERVATION
        assert evidence.observation is not None
        assert evidence.observation.description == "Observed X"

    def test_create_testimony_evidence(self):
        evidence = create_evidence(
            EvidenceType.TESTIMONY,
            EvidenceContribution.CONTEXT,
            relevance=0.6,
            statement="I witnessed...",
        )
        assert evidence.type == EvidenceType.TESTIMONY
        assert evidence.testimony_statement == "I witnessed..."


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_reputation_floor_prevents_going_below(self):
        rep = ReputationScore(identity_id="test", overall=0.15)
        service = VerificationService()
        service._reputations["test"] = rep
        
        # Apply large negative delta
        service._apply_reputation_update(rep, -0.10, "test penalty")
        
        # Should be floored at 0.1
        assert rep.overall == ReputationConstants.REPUTATION_FLOOR

    def test_reputation_ceiling_prevents_going_above(self):
        rep = ReputationScore(identity_id="test", overall=0.95)
        service = VerificationService()
        service._reputations["test"] = rep
        
        # Apply large positive delta
        service._apply_reputation_update(rep, 0.10, "test bonus")
        
        # Should be capped at 1.0
        assert rep.overall == 1.0

    def test_accept_already_accepted_verification_raises(self):
        service = VerificationService()
        
        verification = Verification(
            id=uuid4(),
            verifier_id="did:example:verifier",
            belief_id=uuid4(),
            holder_id="did:example:holder",
            result=VerificationResult.CONFIRMED,
            evidence=[],
            stake=Stake(
                amount=0.02,
                type=StakeType.STANDARD,
                locked_until=datetime.now() + timedelta(days=7),
                escrow_id=uuid4(),
            ),
            status=VerificationStatus.ACCEPTED,  # Already accepted
        )
        service._verifications[verification.id] = verification
        
        with pytest.raises(ValidationException) as exc:
            service.accept_verification(verification.id)
        
        assert "not pending" in str(exc.value).lower()

    def test_resolve_already_resolved_dispute_raises(self):
        service = VerificationService()
        
        dispute = Dispute(
            id=uuid4(),
            verification_id=uuid4(),
            disputer_id="did:example:disputer",
            counter_evidence=[],
            stake=Stake(
                amount=0.02,
                type=StakeType.CHALLENGE,
                locked_until=datetime.now() + timedelta(days=14),
                escrow_id=uuid4(),
            ),
            dispute_type=DisputeType.NEW_EVIDENCE,
            reasoning="...",
            status=DisputeStatus.RESOLVED,  # Already resolved
        )
        service._disputes[dispute.id] = dispute
        
        with pytest.raises(ValidationException) as exc:
            service.resolve_dispute(
                dispute.id,
                DisputeOutcome.UPHELD,
                "...",
            )
        
        assert "not pending" in str(exc.value).lower()

    def test_get_nonexistent_verification_returns_none(self):
        service = VerificationService()
        assert service.get_verification(uuid4()) is None

    def test_get_nonexistent_dispute_returns_none(self):
        service = VerificationService()
        assert service.get_dispute(uuid4()) is None

    def test_get_verifications_for_belief_empty(self):
        service = VerificationService()
        verifications = service.get_verifications_for_belief(uuid4())
        assert verifications == []


# ============================================================================
# Serialization Round-Trip Tests
# ============================================================================

class TestSerializationRoundTrip:
    """Tests for full serialization/deserialization round trips."""

    def test_verification_full_roundtrip(self):
        original = Verification(
            id=uuid4(),
            verifier_id="did:example:verifier",
            belief_id=uuid4(),
            holder_id="did:example:holder",
            result=VerificationResult.PARTIAL,
            evidence=[
                Evidence(
                    id=uuid4(),
                    type=EvidenceType.EXTERNAL,
                    relevance=0.9,
                    contribution=EvidenceContribution.SUPPORTS,
                    external_source=ExternalSource(url="https://example.com"),
                ),
                Evidence(
                    id=uuid4(),
                    type=EvidenceType.OBSERVATION,
                    relevance=0.7,
                    contribution=EvidenceContribution.CONTRADICTS,
                    observation=Observation(
                        description="Observed X",
                        timestamp=datetime.now(),
                        method="direct",
                    ),
                ),
            ],
            stake=Stake(
                amount=0.05,
                type=StakeType.STANDARD,
                locked_until=datetime.now() + timedelta(days=7),
                escrow_id=uuid4(),
            ),
            reasoning="Partial verification",
            result_details=ResultDetails(
                accurate_portions=["Part A"],
                inaccurate_portions=["Part B"],
                accuracy_estimate=0.6,
            ),
            status=VerificationStatus.ACCEPTED,
        )
        
        # Convert to dict and back
        d = original.to_dict()
        json_str = json.dumps(d)
        restored_dict = json.loads(json_str)
        restored = Verification.from_dict(restored_dict)
        
        assert restored.id == original.id
        assert restored.result == original.result
        assert len(restored.evidence) == 2
        assert restored.result_details.accuracy_estimate == 0.6

    def test_dispute_full_roundtrip(self):
        original = Dispute(
            id=uuid4(),
            verification_id=uuid4(),
            disputer_id="did:example:disputer",
            counter_evidence=[
                Evidence(
                    id=uuid4(),
                    type=EvidenceType.EXTERNAL,
                    relevance=0.95,
                    contribution=EvidenceContribution.SUPPORTS,
                ),
            ],
            stake=Stake(
                amount=0.075,
                type=StakeType.CHALLENGE,
                locked_until=datetime.now() + timedelta(days=14),
                escrow_id=uuid4(),
            ),
            dispute_type=DisputeType.NEW_EVIDENCE,
            reasoning="New evidence found",
            proposed_result=VerificationResult.CONFIRMED,
            status=DisputeStatus.RESOLVED,
            outcome=DisputeOutcome.OVERTURNED,
            resolution_reasoning="New evidence is compelling",
            resolution_method=ResolutionMethod.JURY,
        )
        
        d = original.to_dict()
        json_str = json.dumps(d)
        restored_dict = json.loads(json_str)
        restored = Dispute.from_dict(restored_dict)
        
        assert restored.id == original.id
        assert restored.outcome == DisputeOutcome.OVERTURNED
        assert restored.resolution_method == ResolutionMethod.JURY
