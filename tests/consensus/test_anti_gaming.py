"""Tests for anti-gaming measures.

Tests cover:
- Tenure penalty calculation
- Diversity scoring
- Collusion detection
- Stake manipulation detection
- Anti-gaming engine
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from valence.consensus.models import (
    Validator,
    ValidatorSet,
    ValidatorTier,
    ValidatorStatus,
    ValidatorPerformance,
    StakeRegistration,
    StakeStatus,
    IdentityAttestation,
    AttestationType,
    DiversityConstraints,
    SlashingEvidence,
)
from valence.consensus.anti_gaming import (
    AntiGamingEngine,
    CollusionAlert,
    CollusionIndicator,
    SeverityLevel,
    VotingRecord,
    compute_tenure_penalty,
    compute_diversity_score,
    detect_collusion_patterns,
    MAX_CONSECUTIVE_EPOCHS_BEFORE_PENALTY,
    TENURE_PENALTY_FACTOR,
    VOTING_CORRELATION_THRESHOLD,
    MIN_VOTES_FOR_CORRELATION,
    STAKE_TIMING_WINDOW_HOURS,
    MIN_CORRELATED_VALIDATORS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_validator():
    """Create a sample validator."""
    return Validator(
        id=uuid4(),
        agent_id="did:vkb:key:z6MkTestValidator",
        staked_reputation=0.20,
        tier=ValidatorTier.STANDARD,
        stake_lock_until=datetime.now() + timedelta(days=21),
        selection_weight=1.0,
        selection_ticket=b"t" * 32,
        public_key=b"p" * 32,
        attestations=[],
        federation_membership=["federation_a"],
        tenure_epochs=1,
        status=ValidatorStatus.ACTIVE,
    )


@pytest.fixture
def diverse_validator_set():
    """Create a diverse validator set."""
    validators = []
    for i in range(31):
        validator = Validator(
            id=uuid4(),
            agent_id=f"did:vkb:key:z6MkValidator{i:03d}",
            staked_reputation=0.10 + (i % 3) * 0.15,
            tier=[ValidatorTier.STANDARD, ValidatorTier.ENHANCED, ValidatorTier.GUARDIAN][i % 3],
            stake_lock_until=datetime.now() + timedelta(days=21),
            selection_weight=1.0,
            selection_ticket=hashlib.sha256(f"ticket_{i}".encode()).digest(),
            public_key=hashlib.sha256(f"pubkey_{i}".encode()).digest(),
            attestations=[],
            federation_membership=[f"federation_{i % 6}"],  # 6 different federations
            tenure_epochs=i % 4,  # 0-3 tenure
            status=ValidatorStatus.ACTIVE,
        )
        validators.append(validator)
    
    return ValidatorSet(
        epoch=42,
        epoch_start=datetime.now(),
        epoch_end=datetime.now() + timedelta(days=7),
        validators=validators,
        selection_seed=b"s" * 32,
    )


@pytest.fixture
def concentrated_validator_set():
    """Create a validator set with concentration issues."""
    validators = []
    for i in range(31):
        # 15 validators from same federation (nearly 50%)
        federation = "dominant_fed" if i < 15 else f"other_fed_{i}"
        # Most are enhanced tier
        tier = ValidatorTier.ENHANCED if i < 25 else ValidatorTier.STANDARD
        # High tenure for many
        tenure = 8 if i < 20 else 1
        
        validator = Validator(
            id=uuid4(),
            agent_id=f"did:vkb:key:z6MkValidator{i:03d}",
            staked_reputation=0.35,
            tier=tier,
            stake_lock_until=datetime.now() + timedelta(days=21),
            selection_weight=1.0,
            selection_ticket=hashlib.sha256(f"ticket_{i}".encode()).digest(),
            public_key=hashlib.sha256(f"pubkey_{i}".encode()).digest(),
            attestations=[],
            federation_membership=[federation],
            tenure_epochs=tenure,
            status=ValidatorStatus.ACTIVE,
        )
        validators.append(validator)
    
    return ValidatorSet(
        epoch=42,
        epoch_start=datetime.now(),
        epoch_end=datetime.now() + timedelta(days=7),
        validators=validators,
        selection_seed=b"s" * 32,
    )


# =============================================================================
# TENURE PENALTY TESTS
# =============================================================================


class TestTenurePenalty:
    """Tests for tenure penalty calculation."""
    
    def test_no_penalty_for_new_validators(self):
        """New validators should have no penalty."""
        assert compute_tenure_penalty(0) == 1.0
        assert compute_tenure_penalty(1) == 1.0
    
    def test_no_penalty_at_threshold(self):
        """Penalty should not apply at threshold."""
        assert compute_tenure_penalty(MAX_CONSECUTIVE_EPOCHS_BEFORE_PENALTY) == 1.0
    
    def test_penalty_starts_after_threshold(self):
        """Penalty should start after threshold."""
        penalty = compute_tenure_penalty(MAX_CONSECUTIVE_EPOCHS_BEFORE_PENALTY + 1)
        assert penalty == TENURE_PENALTY_FACTOR  # 0.9
    
    def test_penalty_compounds(self):
        """Penalty should compound with each additional epoch."""
        penalty_5 = compute_tenure_penalty(5)
        penalty_6 = compute_tenure_penalty(6)
        penalty_7 = compute_tenure_penalty(7)
        
        assert penalty_6 == penalty_5 * TENURE_PENALTY_FACTOR
        assert penalty_7 == penalty_6 * TENURE_PENALTY_FACTOR
    
    def test_severe_penalty_for_very_long_tenure(self):
        """Very long tenure should result in severe penalty."""
        # 12 epochs = 8 beyond threshold
        penalty = compute_tenure_penalty(12)
        expected = TENURE_PENALTY_FACTOR ** 8  # 0.9^8 ≈ 0.43
        assert abs(penalty - expected) < 0.01
    
    def test_penalty_approaches_zero(self):
        """Penalty should approach zero for extreme tenure."""
        penalty_20 = compute_tenure_penalty(20)
        penalty_30 = compute_tenure_penalty(30)
        
        assert penalty_20 < 0.2
        assert penalty_30 < 0.1  # Relaxed threshold


# =============================================================================
# DIVERSITY SCORING TESTS
# =============================================================================


class TestDiversityScoring:
    """Tests for diversity score calculation."""
    
    def test_diverse_set_scores_high(self, diverse_validator_set):
        """A diverse validator set should score high."""
        scores = compute_diversity_score(diverse_validator_set)
        
        assert scores["overall_score"] > 0.5
        assert scores["federation_gini"] < 0.5  # Lower is better
        assert scores["new_validator_ratio"] > 0
    
    def test_concentrated_set_scores_low(self, concentrated_validator_set):
        """A concentrated validator set should score lower."""
        diverse_scores = compute_diversity_score(
            ValidatorSet(
                epoch=1,
                epoch_start=datetime.now(),
                epoch_end=datetime.now() + timedelta(days=7),
                validators=[
                    Validator(
                        id=uuid4(),
                        agent_id=f"did:vkb:key:v{i}",
                        staked_reputation=0.15,
                        tier=[ValidatorTier.STANDARD, ValidatorTier.ENHANCED, ValidatorTier.GUARDIAN][i % 3],
                        stake_lock_until=datetime.now() + timedelta(days=21),
                        selection_weight=1.0,
                        selection_ticket=b"t" * 32,
                        public_key=b"p" * 32,
                        federation_membership=[f"fed_{i}"],  # All different
                        tenure_epochs=1,
                        status=ValidatorStatus.ACTIVE,
                    )
                    for i in range(10)
                ],
            )
        )
        
        concentrated_scores = compute_diversity_score(concentrated_validator_set)
        
        # Concentrated set should have lower overall score
        assert concentrated_scores["overall_score"] < diverse_scores["overall_score"]
    
    def test_empty_set_returns_zero(self):
        """Empty validator set should return zero scores."""
        empty_set = ValidatorSet(
            epoch=1,
            epoch_start=datetime.now(),
            epoch_end=datetime.now() + timedelta(days=7),
            validators=[],
        )
        
        scores = compute_diversity_score(empty_set)
        assert scores["overall_score"] == 0.0
        assert scores["federation_gini"] == 1.0  # Worst inequality
    
    def test_scores_include_all_metrics(self, diverse_validator_set):
        """Score output should include all expected metrics."""
        scores = compute_diversity_score(diverse_validator_set)
        
        assert "overall_score" in scores
        assert "federation_gini" in scores
        assert "tier_entropy" in scores
        assert "tenure_variance" in scores
        assert "new_validator_ratio" in scores
        assert "validators_analyzed" in scores
    
    def test_new_validator_ratio_calculation(self):
        """New validator ratio should be correctly calculated."""
        validators = []
        for i in range(10):
            validators.append(Validator(
                id=uuid4(),
                agent_id=f"did:vkb:key:v{i}",
                staked_reputation=0.15,
                tier=ValidatorTier.STANDARD,
                stake_lock_until=datetime.now() + timedelta(days=21),
                selection_weight=1.0,
                selection_ticket=b"t" * 32,
                public_key=b"p" * 32,
                federation_membership=["fed"],
                tenure_epochs=0 if i < 5 else 3,  # 5 new, 5 returning
                status=ValidatorStatus.ACTIVE,
            ))
        
        validator_set = ValidatorSet(
            epoch=1,
            epoch_start=datetime.now(),
            epoch_end=datetime.now() + timedelta(days=7),
            validators=validators,
        )
        
        scores = compute_diversity_score(validator_set)
        assert scores["new_validator_ratio"] == 0.5  # 5/10


# =============================================================================
# COLLUSION DETECTION TESTS
# =============================================================================


class TestCollusionDetection:
    """Tests for collusion pattern detection."""
    
    def test_no_alerts_for_normal_voting(self, diverse_validator_set):
        """Normal voting patterns should not trigger alerts."""
        # Create diverse voting records
        voting_records = []
        for i, validator in enumerate(diverse_validator_set.validators):
            for p in range(25):
                voting_records.append(VotingRecord(
                    validator_id=validator.agent_id,
                    proposal_id=uuid4(),
                    vote=["approve", "reject", "abstain"][p % 3],
                    voted_at=datetime.now() - timedelta(hours=p),
                ))
        
        alerts = detect_collusion_patterns(
            voting_records=voting_records,
            stake_registrations=[],
            validator_set=diverse_validator_set,
        )
        
        # Should not have voting correlation alerts
        correlation_alerts = [
            a for a in alerts
            if a.indicator == CollusionIndicator.VOTING_CORRELATION
        ]
        assert len(correlation_alerts) == 0
    
    def test_detects_voting_correlation(self, diverse_validator_set):
        """Should detect highly correlated voting."""
        # Create identical voting patterns for first 5 validators
        voting_records = []
        proposal_ids = [uuid4() for _ in range(25)]
        votes = ["approve", "reject", "approve", "approve", "reject"] * 5
        
        # Colluding validators vote identically
        for i in range(5):
            validator = diverse_validator_set.validators[i]
            for j, proposal_id in enumerate(proposal_ids):
                voting_records.append(VotingRecord(
                    validator_id=validator.agent_id,
                    proposal_id=proposal_id,
                    vote=votes[j],
                    voted_at=datetime.now() - timedelta(hours=j),
                ))
        
        # Other validators vote differently
        for i in range(5, 20):
            validator = diverse_validator_set.validators[i]
            for j, proposal_id in enumerate(proposal_ids):
                voting_records.append(VotingRecord(
                    validator_id=validator.agent_id,
                    proposal_id=proposal_id,
                    vote=["approve", "reject"][(i + j) % 2],
                    voted_at=datetime.now() - timedelta(hours=j),
                ))
        
        alerts = detect_collusion_patterns(
            voting_records=voting_records,
            stake_registrations=[],
            validator_set=diverse_validator_set,
        )
        
        correlation_alerts = [
            a for a in alerts
            if a.indicator == CollusionIndicator.VOTING_CORRELATION
        ]
        
        # Should detect the correlated group
        assert len(correlation_alerts) > 0
        assert correlation_alerts[0].severity in (SeverityLevel.WARNING, SeverityLevel.HIGH)
    
    def test_detects_stake_timing(self, diverse_validator_set):
        """Should detect suspicious stake timing."""
        # Create stake registrations within same window
        base_time = datetime.now() - timedelta(days=30)
        stake_registrations = [
            (v.agent_id, base_time + timedelta(hours=i % 3))  # Within 3 hours
            for i, v in enumerate(diverse_validator_set.validators[:6])
        ]
        
        alerts = detect_collusion_patterns(
            voting_records=[],
            stake_registrations=stake_registrations,
            validator_set=diverse_validator_set,
        )
        
        timing_alerts = [
            a for a in alerts
            if a.indicator == CollusionIndicator.STAKE_TIMING
        ]
        
        assert len(timing_alerts) > 0
    
    def test_detects_federation_clustering(self, concentrated_validator_set):
        """Should detect over-represented federations."""
        alerts = detect_collusion_patterns(
            voting_records=[],
            stake_registrations=[],
            validator_set=concentrated_validator_set,
        )
        
        clustering_alerts = [
            a for a in alerts
            if a.indicator == CollusionIndicator.FEDERATION_CLUSTERING
        ]
        
        assert len(clustering_alerts) > 0
        assert "dominant_fed" in clustering_alerts[0].description


# =============================================================================
# ANTI-GAMING ENGINE TESTS
# =============================================================================


class TestAntiGamingEngine:
    """Tests for the AntiGamingEngine class."""
    
    def test_compute_tenure_penalty(self):
        """Engine should compute tenure penalty correctly."""
        engine = AntiGamingEngine()
        
        assert engine.compute_tenure_penalty(0) == 1.0
        assert engine.compute_tenure_penalty(5) == 0.9
    
    def test_compute_diversity_score(self, diverse_validator_set):
        """Engine should compute diversity score."""
        engine = AntiGamingEngine()
        scores = engine.compute_diversity_score(diverse_validator_set)
        
        assert "overall_score" in scores
        assert scores["validators_analyzed"] == 31
    
    def test_analyze_validator_set(self, diverse_validator_set):
        """analyze_validator_set should return comprehensive report."""
        engine = AntiGamingEngine()
        analysis = engine.analyze_validator_set(diverse_validator_set)
        
        assert "validator_count" in analysis
        assert "epoch" in analysis
        assert "diversity" in analysis
        assert "tenure_stats" in analysis
        assert "alerts" in analysis
        assert "health_score" in analysis
        
        assert analysis["validator_count"] == 31
        assert analysis["epoch"] == 42
        assert 0.0 <= analysis["health_score"] <= 1.0
    
    def test_health_score_decreases_with_issues(self, concentrated_validator_set):
        """Health score should be lower for problematic sets."""
        engine = AntiGamingEngine()
        
        diverse_set = ValidatorSet(
            epoch=1,
            epoch_start=datetime.now(),
            epoch_end=datetime.now() + timedelta(days=7),
            validators=[
                Validator(
                    id=uuid4(),
                    agent_id=f"did:vkb:key:v{i}",
                    staked_reputation=0.15,
                    tier=[ValidatorTier.STANDARD, ValidatorTier.ENHANCED, ValidatorTier.GUARDIAN][i % 3],
                    stake_lock_until=datetime.now() + timedelta(days=21),
                    selection_weight=1.0,
                    selection_ticket=b"t" * 32,
                    public_key=b"p" * 32,
                    federation_membership=[f"fed_{i}"],
                    tenure_epochs=1,
                    status=ValidatorStatus.ACTIVE,
                )
                for i in range(31)
            ],
        )
        
        diverse_analysis = engine.analyze_validator_set(diverse_set)
        concentrated_analysis = engine.analyze_validator_set(concentrated_validator_set)
        
        assert concentrated_analysis["health_score"] < diverse_analysis["health_score"]
    
    def test_generate_slashing_evidence_high_severity(self):
        """Should generate evidence for HIGH severity alerts."""
        engine = AntiGamingEngine()
        
        alert = CollusionAlert(
            id=uuid4(),
            indicator=CollusionIndicator.VOTING_CORRELATION,
            severity=SeverityLevel.HIGH,
            validators=["did:vkb:key:v1", "did:vkb:key:v2"],
            description="Test alert",
            evidence_data={"test": "data"},
            epoch=42,
        )
        
        evidence = engine.generate_slashing_evidence(alert)
        
        assert evidence is not None
        assert isinstance(evidence, SlashingEvidence)
        assert "voting_correlation" in evidence.evidence_type
    
    def test_no_evidence_for_low_severity(self):
        """Should not generate evidence for low severity alerts."""
        engine = AntiGamingEngine()
        
        alert = CollusionAlert(
            id=uuid4(),
            indicator=CollusionIndicator.STAKE_TIMING,
            severity=SeverityLevel.INFO,
            validators=["did:vkb:key:v1"],
            description="Test alert",
            epoch=42,
        )
        
        evidence = engine.generate_slashing_evidence(alert)
        assert evidence is None


# =============================================================================
# COLLUSION ALERT TESTS
# =============================================================================


class TestCollusionAlert:
    """Tests for CollusionAlert dataclass."""
    
    def test_to_dict(self):
        """Alert should serialize to dict correctly."""
        alert = CollusionAlert(
            id=uuid4(),
            indicator=CollusionIndicator.VOTING_CORRELATION,
            severity=SeverityLevel.HIGH,
            validators=["did:vkb:key:v1", "did:vkb:key:v2"],
            description="Test alert",
            evidence_data={"correlation": 0.98},
            epoch=42,
        )
        
        data = alert.to_dict()
        
        assert data["indicator"] == "voting_correlation"
        assert data["severity"] == "high"
        assert len(data["validators"]) == 2
        assert data["epoch"] == 42
        assert data["evidence_data"]["correlation"] == 0.98


# =============================================================================
# VOTING RECORD TESTS
# =============================================================================


class TestVotingRecord:
    """Tests for VotingRecord dataclass."""
    
    def test_voting_record_creation(self):
        """Should create voting record correctly."""
        record = VotingRecord(
            validator_id="did:vkb:key:v1",
            proposal_id=uuid4(),
            vote="approve",
            voted_at=datetime.now(),
        )
        
        assert record.validator_id == "did:vkb:key:v1"
        assert record.vote == "approve"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAntiGamingIntegration:
    """Integration tests for anti-gaming measures."""
    
    def test_full_analysis_pipeline(self, diverse_validator_set):
        """Test full analysis pipeline with voting records."""
        engine = AntiGamingEngine()
        
        # Create some voting records
        voting_records = []
        proposal_ids = [uuid4() for _ in range(30)]
        
        for validator in diverse_validator_set.validators:
            for proposal_id in proposal_ids:
                voting_records.append(VotingRecord(
                    validator_id=validator.agent_id,
                    proposal_id=proposal_id,
                    vote=["approve", "reject"][hash(validator.agent_id + str(proposal_id)) % 2],
                    voted_at=datetime.now() - timedelta(hours=1),
                ))
        
        # Create stake registrations
        stake_registrations = [
            (v.agent_id, datetime.now() - timedelta(days=30 + i))
            for i, v in enumerate(diverse_validator_set.validators)
        ]
        
        # Run full analysis
        analysis = engine.analyze_validator_set(
            validator_set=diverse_validator_set,
            voting_records=voting_records,
            stake_registrations=stake_registrations,
        )
        
        assert "diversity" in analysis
        assert "tenure_stats" in analysis
        assert "alerts" in analysis
        assert "health_score" in analysis
        assert analysis["validator_count"] == 31
    
    def test_attack_detection_scenario(self):
        """Test detection of a simulated attack scenario."""
        # Create a validator set where 10 validators collude
        validators = []
        
        # Colluding validators - same federation, coordinated
        for i in range(10):
            validators.append(Validator(
                id=uuid4(),
                agent_id=f"did:vkb:key:colluder_{i}",
                staked_reputation=0.50,
                tier=ValidatorTier.GUARDIAN,
                stake_lock_until=datetime.now() + timedelta(days=21),
                selection_weight=2.0,
                selection_ticket=b"t" * 32,
                public_key=b"p" * 32,
                federation_membership=["colluder_fed"],
                tenure_epochs=8,  # Long tenure
                status=ValidatorStatus.ACTIVE,
            ))
        
        # Honest validators
        for i in range(21):
            validators.append(Validator(
                id=uuid4(),
                agent_id=f"did:vkb:key:honest_{i}",
                staked_reputation=0.15,
                tier=ValidatorTier.STANDARD,
                stake_lock_until=datetime.now() + timedelta(days=21),
                selection_weight=1.0,
                selection_ticket=b"t" * 32,
                public_key=b"p" * 32,
                federation_membership=[f"honest_fed_{i % 5}"],
                tenure_epochs=1,
                status=ValidatorStatus.ACTIVE,
            ))
        
        validator_set = ValidatorSet(
            epoch=42,
            epoch_start=datetime.now(),
            epoch_end=datetime.now() + timedelta(days=7),
            validators=validators,
        )
        
        # Create coordinated voting from colluders
        voting_records = []
        proposal_ids = [uuid4() for _ in range(30)]
        colluder_votes = ["approve"] * 20 + ["reject"] * 10
        
        for validator in validators[:10]:  # Colluders vote identically
            for j, proposal_id in enumerate(proposal_ids):
                voting_records.append(VotingRecord(
                    validator_id=validator.agent_id,
                    proposal_id=proposal_id,
                    vote=colluder_votes[j],
                    voted_at=datetime.now() - timedelta(hours=j),
                ))
        
        for validator in validators[10:]:  # Honest validators vote independently
            for j, proposal_id in enumerate(proposal_ids):
                voting_records.append(VotingRecord(
                    validator_id=validator.agent_id,
                    proposal_id=proposal_id,
                    vote=["approve", "reject"][hash(validator.agent_id) % 2],
                    voted_at=datetime.now() - timedelta(hours=j),
                ))
        
        # Coordinated stake timing
        base_time = datetime.now() - timedelta(days=30)
        stake_registrations = [
            (v.agent_id, base_time + timedelta(hours=i % 2))
            for i, v in enumerate(validators[:10])
        ]
        
        engine = AntiGamingEngine()
        analysis = engine.analyze_validator_set(
            validator_set=validator_set,
            voting_records=voting_records,
            stake_registrations=stake_registrations,
        )
        
        # Should detect multiple issues
        assert analysis["alert_count"] > 0
        assert analysis["health_score"] < 0.8
        
        # Check for specific alerts
        alert_types = {a["indicator"] for a in analysis["alerts"]}
        assert CollusionIndicator.FEDERATION_CLUSTERING.value in alert_types
