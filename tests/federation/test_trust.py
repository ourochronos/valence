"""Tests for trust computation and management.

Tests cover:
- TrustManager class initialization and delegation
- Trust signal processing (corroboration, dispute, endorsement)
- Signal weight application
- Process convenience functions (module-level)
- Trust manager singleton pattern
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import MagicMock, patch
from uuid import uuid4, UUID

import pytest

from valence.federation.trust import (
    TrustManager,
    TrustSignal,
    SIGNAL_WEIGHTS,
    get_trust_manager,
    get_effective_trust,
    process_corroboration,
    process_dispute,
    assess_and_respond_to_threat,
)
from valence.federation.models import (
    NodeTrust,
    TrustPhase,
    TrustPreference,
    ThreatLevel,
    AnnotationType,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_cursor():
    """Mock database cursor."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Mock the get_cursor context manager."""
    @contextmanager
    def _mock_get_cursor(dict_cursor: bool = True) -> Generator:
        yield mock_cursor

    with patch("valence.federation.trust.get_cursor", _mock_get_cursor):
        yield mock_cursor


@pytest.fixture
def sample_node_trust():
    """Create a sample NodeTrust mock."""
    def _factory(**kwargs):
        trust = MagicMock(spec=NodeTrust)
        trust.id = kwargs.get("id", uuid4())
        trust.node_id = kwargs.get("node_id", uuid4())
        trust.overall = kwargs.get("overall", 0.5)
        trust.belief_accuracy = kwargs.get("belief_accuracy", 0.5)
        trust.endorsement_strength = kwargs.get("endorsement_strength", 0.0)
        trust.uptime_reliability = kwargs.get("uptime_reliability", 0.7)
        trust.contribution_consistency = kwargs.get("contribution_consistency", 0.5)
        trust.domain_expertise = kwargs.get("domain_expertise", {})
        trust.beliefs_received = kwargs.get("beliefs_received", 10)
        trust.beliefs_corroborated = kwargs.get("beliefs_corroborated", 5)
        trust.beliefs_disputed = kwargs.get("beliefs_disputed", 1)
        trust.sync_requests_served = kwargs.get("sync_requests_served", 20)
        trust.aggregation_participations = kwargs.get("aggregation_participations", 3)
        trust.endorsements_received = kwargs.get("endorsements_received", 0)
        trust.manual_trust_adjustment = kwargs.get("manual_trust_adjustment", 0.0)
        trust.relationship_started_at = kwargs.get(
            "relationship_started_at",
            datetime.now() - timedelta(days=30)
        )
        trust.last_interaction_at = kwargs.get("last_interaction_at", datetime.now())
        return trust
    return _factory


@pytest.fixture
def mock_registry():
    """Create a mock TrustRegistry."""
    registry = MagicMock()
    return registry


@pytest.fixture
def mock_threat_detector():
    """Create a mock ThreatDetector."""
    detector = MagicMock()
    return detector


@pytest.fixture
def mock_policy():
    """Create a mock TrustPolicy."""
    policy = MagicMock()
    return policy


# =============================================================================
# TRUST SIGNAL TESTS
# =============================================================================


class TestTrustSignal:
    """Tests for TrustSignal dataclass."""

    def test_trust_signal_creation(self):
        """Test creating a trust signal."""
        node_id = uuid4()
        signal = TrustSignal(
            node_id=node_id,
            signal_type="corroboration",
            value=1.0,
            domain="science",
        )

        assert signal.node_id == node_id
        assert signal.signal_type == "corroboration"
        assert signal.value == 1.0
        assert signal.domain == "science"
        assert signal.timestamp is not None

    def test_trust_signal_defaults(self):
        """Test trust signal default values."""
        node_id = uuid4()
        signal = TrustSignal(node_id=node_id, signal_type="dispute")

        assert signal.value == 1.0
        assert signal.domain is None
        assert signal.source_node_id is None
        assert signal.belief_id is None

    def test_trust_signal_with_belief(self):
        """Test trust signal with belief reference."""
        node_id = uuid4()
        belief_id = uuid4()
        signal = TrustSignal(
            node_id=node_id,
            signal_type="corroboration",
            belief_id=belief_id,
        )

        assert signal.belief_id == belief_id


class TestSignalWeights:
    """Tests for signal weight constants."""

    def test_signal_weights_defined(self):
        """Test that all signal weights are defined."""
        expected_signals = [
            "corroboration",
            "dispute",
            "endorsement",
            "sync_success",
            "sync_failure",
            "aggregation_participation",
        ]
        
        for signal in expected_signals:
            assert signal in SIGNAL_WEIGHTS

    def test_corroboration_positive(self):
        """Test that corroboration is positive."""
        assert SIGNAL_WEIGHTS["corroboration"] > 0

    def test_dispute_negative(self):
        """Test that dispute is negative."""
        assert SIGNAL_WEIGHTS["dispute"] < 0

    def test_sync_failure_negative(self):
        """Test that sync failure is negative."""
        assert SIGNAL_WEIGHTS["sync_failure"] < 0


# =============================================================================
# TRUST MANAGER TESTS
# =============================================================================


class TestTrustManagerInit:
    """Tests for TrustManager initialization."""

    def test_trust_manager_init_default(self):
        """Test TrustManager with default parameters."""
        with patch("valence.federation.trust.TrustRegistry") as MockRegistry, \
             patch("valence.federation.trust.ThreatDetector") as MockDetector, \
             patch("valence.federation.trust.TrustPolicy") as MockPolicy:
            
            manager = TrustManager()

            MockRegistry.assert_called_once()
            MockDetector.assert_called_once()
            MockPolicy.assert_called_once()
            assert manager.decay_half_life_days == 30  # Default
            assert manager.decay_min_threshold == 0.1  # Default

    def test_trust_manager_init_custom(self):
        """Test TrustManager with custom parameters."""
        with patch("valence.federation.trust.TrustRegistry"), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager(
                decay_half_life_days=60,
                decay_min_threshold=0.05,
            )

            assert manager.decay_half_life_days == 60
            assert manager.decay_min_threshold == 0.05


class TestTrustManagerDelegation:
    """Tests for TrustManager method delegation."""

    def test_get_node_trust_delegates(self, mock_registry, sample_node_trust):
        """Test that get_node_trust delegates to registry."""
        node_id = uuid4()
        expected_trust = sample_node_trust(node_id=node_id)
        mock_registry.get_node_trust.return_value = expected_trust

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.get_node_trust(node_id)

            mock_registry.get_node_trust.assert_called_once_with(node_id)
            assert result == expected_trust

    def test_get_user_trust_preference_delegates(self, mock_registry):
        """Test that get_user_trust_preference delegates to registry."""
        node_id = uuid4()
        mock_registry.get_user_trust_preference.return_value = None

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.get_user_trust_preference(node_id)

            mock_registry.get_user_trust_preference.assert_called_once_with(node_id)
            assert result is None

    def test_get_effective_trust_delegates(self, mock_policy):
        """Test that get_effective_trust delegates to policy."""
        node_id = uuid4()
        mock_policy.get_effective_trust.return_value = 0.75

        with patch("valence.federation.trust.TrustRegistry"), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy", return_value=mock_policy):
            
            manager = TrustManager()
            result = manager.get_effective_trust(node_id, domain="science")

            mock_policy.get_effective_trust.assert_called_once_with(node_id, "science", True)
            assert result == 0.75

    def test_assess_threat_level_delegates(self, mock_threat_detector):
        """Test that assess_threat_level delegates to threat detector."""
        node_id = uuid4()
        mock_threat_detector.assess_threat_level.return_value = (
            ThreatLevel.LOW,
            {"threat_score": 0.15, "signals": []}
        )

        with patch("valence.federation.trust.TrustRegistry"), \
             patch("valence.federation.trust.ThreatDetector", return_value=mock_threat_detector), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            level, assessment = manager.assess_threat_level(node_id)

            mock_threat_detector.assess_threat_level.assert_called_once_with(node_id)
            assert level == ThreatLevel.LOW

    def test_apply_threat_response_delegates(self, mock_threat_detector):
        """Test that apply_threat_response delegates to threat detector."""
        node_id = uuid4()
        mock_threat_detector.apply_threat_response.return_value = True

        with patch("valence.federation.trust.TrustRegistry"), \
             patch("valence.federation.trust.ThreatDetector", return_value=mock_threat_detector), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.apply_threat_response(
                node_id,
                ThreatLevel.MEDIUM,
                {"threat_score": 0.45}
            )

            mock_threat_detector.apply_threat_response.assert_called_once()
            assert result is True

    def test_check_phase_transition_delegates(self, mock_policy):
        """Test that check_phase_transition delegates to policy."""
        node_id = uuid4()
        mock_policy.check_phase_transition.return_value = TrustPhase.CONTRIBUTOR

        with patch("valence.federation.trust.TrustRegistry"), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy", return_value=mock_policy):
            
            manager = TrustManager()
            result = manager.check_phase_transition(node_id)

            mock_policy.check_phase_transition.assert_called_once_with(node_id)
            assert result == TrustPhase.CONTRIBUTOR


# =============================================================================
# SIGNAL PROCESSING TESTS
# =============================================================================


class TestProcessSignal:
    """Tests for TrustManager.process_signal."""

    def test_process_corroboration_signal(self, mock_registry, sample_node_trust):
        """Test processing a corroboration signal."""
        node_id = uuid4()
        trust = sample_node_trust(node_id=node_id, beliefs_corroborated=5, belief_accuracy=0.5)
        mock_registry.get_node_trust.return_value = trust
        mock_registry.save_node_trust.return_value = trust

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            signal = TrustSignal(node_id=node_id, signal_type="corroboration")
            result = manager.process_signal(signal)

            assert result is not None
            # beliefs_corroborated should be incremented
            assert trust.beliefs_corroborated == 6

    def test_process_dispute_signal(self, mock_registry, sample_node_trust):
        """Test processing a dispute signal."""
        node_id = uuid4()
        trust = sample_node_trust(node_id=node_id, beliefs_disputed=1, belief_accuracy=0.8)
        mock_registry.get_node_trust.return_value = trust
        mock_registry.save_node_trust.return_value = trust

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            signal = TrustSignal(node_id=node_id, signal_type="dispute", value=1.0)
            result = manager.process_signal(signal)

            assert result is not None
            assert trust.beliefs_disputed == 2

    def test_process_endorsement_signal(self, mock_registry, sample_node_trust):
        """Test processing an endorsement signal."""
        node_id = uuid4()
        trust = sample_node_trust(node_id=node_id, endorsements_received=0)
        mock_registry.get_node_trust.return_value = trust
        mock_registry.save_node_trust.return_value = trust

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            signal = TrustSignal(node_id=node_id, signal_type="endorsement")
            result = manager.process_signal(signal)

            assert result is not None
            assert trust.endorsements_received == 1

    def test_process_sync_success_signal(self, mock_registry, sample_node_trust):
        """Test processing a sync success signal."""
        node_id = uuid4()
        trust = sample_node_trust(node_id=node_id, sync_requests_served=20)
        mock_registry.get_node_trust.return_value = trust
        mock_registry.save_node_trust.return_value = trust

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            signal = TrustSignal(node_id=node_id, signal_type="sync_success")
            result = manager.process_signal(signal)

            assert result is not None
            assert trust.sync_requests_served == 21

    def test_process_signal_no_trust_record(self, mock_registry):
        """Test processing signal when no trust record exists."""
        node_id = uuid4()
        mock_registry.get_node_trust.return_value = None

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            signal = TrustSignal(node_id=node_id, signal_type="corroboration")
            result = manager.process_signal(signal)

            assert result is None

    def test_process_signal_with_domain(self, mock_registry, sample_node_trust):
        """Test processing signal with domain-specific update."""
        node_id = uuid4()
        trust = sample_node_trust(node_id=node_id, domain_expertise={"science": 0.5})
        mock_registry.get_node_trust.return_value = trust
        mock_registry.save_node_trust.return_value = trust

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            signal = TrustSignal(
                node_id=node_id,
                signal_type="corroboration",
                domain="science"
            )
            result = manager.process_signal(signal)

            assert result is not None
            # Domain expertise should be updated
            # The mock allows attribute assignment, check it was accessed


class TestProcessCorroboration:
    """Tests for TrustManager.process_corroboration."""

    def test_process_corroboration_basic(self, mock_registry, sample_node_trust):
        """Test basic corroboration processing."""
        node_id = uuid4()
        trust = sample_node_trust(node_id=node_id)
        mock_registry.get_node_trust.return_value = trust
        mock_registry.save_node_trust.return_value = trust

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.process_corroboration(node_id)

            assert result is not None

    def test_process_corroboration_with_belief(self, mock_registry, sample_node_trust):
        """Test corroboration processing with belief reference."""
        node_id = uuid4()
        belief_id = uuid4()
        trust = sample_node_trust(node_id=node_id)
        mock_registry.get_node_trust.return_value = trust
        mock_registry.save_node_trust.return_value = trust

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.process_corroboration(node_id, belief_id=belief_id)

            assert result is not None


class TestProcessDispute:
    """Tests for TrustManager.process_dispute."""

    def test_process_dispute_basic(self, mock_registry, sample_node_trust):
        """Test basic dispute processing."""
        node_id = uuid4()
        trust = sample_node_trust(node_id=node_id)
        mock_registry.get_node_trust.return_value = trust
        mock_registry.save_node_trust.return_value = trust

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.process_dispute(node_id)

            assert result is not None

    def test_process_dispute_with_severity(self, mock_registry, sample_node_trust):
        """Test dispute processing with custom severity."""
        node_id = uuid4()
        trust = sample_node_trust(node_id=node_id)
        mock_registry.get_node_trust.return_value = trust
        mock_registry.save_node_trust.return_value = trust

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.process_dispute(node_id, severity=2.0)

            assert result is not None


class TestProcessEndorsement:
    """Tests for TrustManager.process_endorsement."""

    def test_process_endorsement_basic(self, mock_registry, mock_policy, sample_node_trust):
        """Test basic endorsement processing."""
        subject_id = uuid4()
        endorser_id = uuid4()
        trust = sample_node_trust(node_id=subject_id)
        mock_registry.get_node_trust.return_value = trust
        mock_registry.save_node_trust.return_value = trust
        mock_policy.get_effective_trust.return_value = 0.6

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy", return_value=mock_policy):
            
            manager = TrustManager()
            result = manager.process_endorsement(subject_id, endorser_id)

            assert result is not None


# =============================================================================
# USER PREFERENCE TESTS
# =============================================================================


class TestUserPreferences:
    """Tests for user trust preference methods."""

    def test_set_user_preference(self, mock_registry):
        """Test setting user preference."""
        node_id = uuid4()
        mock_pref = MagicMock()
        mock_registry.set_user_preference.return_value = mock_pref

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.set_user_preference(
                node_id,
                TrustPreference.ELEVATED,
                reason="Trusted source"
            )

            mock_registry.set_user_preference.assert_called_once()
            assert result == mock_pref

    def test_block_node(self, mock_registry):
        """Test blocking a node."""
        node_id = uuid4()
        mock_pref = MagicMock()
        mock_registry.set_user_preference.return_value = mock_pref

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.block_node(node_id, reason="Suspicious")

            # Should call set_user_preference with BLOCKED
            call_args = mock_registry.set_user_preference.call_args
            assert call_args[1]["preference"] == TrustPreference.BLOCKED

    def test_unblock_node(self, mock_registry):
        """Test unblocking a node."""
        node_id = uuid4()
        mock_pref = MagicMock()
        mock_registry.set_user_preference.return_value = mock_pref

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.unblock_node(node_id)

            call_args = mock_registry.set_user_preference.call_args
            assert call_args[1]["preference"] == TrustPreference.AUTOMATIC


# =============================================================================
# BELIEF ANNOTATION TESTS
# =============================================================================


class TestBeliefAnnotations:
    """Tests for belief annotation methods."""

    def test_annotate_belief(self, mock_registry):
        """Test annotating a belief."""
        belief_id = uuid4()
        mock_annotation = MagicMock()
        mock_registry.annotate_belief.return_value = mock_annotation

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.annotate_belief(
                belief_id,
                AnnotationType.CORROBORATION,
                confidence_delta=0.05
            )

            mock_registry.annotate_belief.assert_called_once()
            assert result == mock_annotation

    def test_get_belief_trust_adjustments(self, mock_registry):
        """Test getting belief trust adjustments."""
        belief_id = uuid4()
        mock_registry.get_belief_trust_adjustments.return_value = 0.15

        with patch("valence.federation.trust.TrustRegistry", return_value=mock_registry), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = TrustManager()
            result = manager.get_belief_trust_adjustments(belief_id)

            assert result == 0.15


# =============================================================================
# MODULE-LEVEL FUNCTION TESTS
# =============================================================================


class TestGetTrustManager:
    """Tests for get_trust_manager singleton."""

    def test_get_trust_manager_returns_instance(self):
        """Test that get_trust_manager returns a TrustManager."""
        # Reset the global manager
        import valence.federation.trust
        valence.federation.trust._default_manager = None

        with patch("valence.federation.trust.TrustRegistry"), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager = get_trust_manager()
            
            assert manager is not None
            assert isinstance(manager, TrustManager)

    def test_get_trust_manager_singleton(self):
        """Test that get_trust_manager returns same instance."""
        import valence.federation.trust
        valence.federation.trust._default_manager = None

        with patch("valence.federation.trust.TrustRegistry"), \
             patch("valence.federation.trust.ThreatDetector"), \
             patch("valence.federation.trust.TrustPolicy"):
            
            manager1 = get_trust_manager()
            manager2 = get_trust_manager()
            
            assert manager1 is manager2


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_effective_trust_function(self):
        """Test module-level get_effective_trust."""
        node_id = uuid4()
        
        with patch("valence.federation.trust.get_trust_manager") as mock_get_mgr:
            mock_mgr = MagicMock()
            mock_mgr.get_effective_trust.return_value = 0.7
            mock_get_mgr.return_value = mock_mgr

            result = get_effective_trust(node_id, domain="science")

            mock_mgr.get_effective_trust.assert_called_once_with(node_id, "science")
            assert result == 0.7

    def test_process_corroboration_function(self):
        """Test module-level process_corroboration."""
        node_id = uuid4()
        
        with patch("valence.federation.trust.get_trust_manager") as mock_get_mgr:
            mock_mgr = MagicMock()
            mock_trust = MagicMock()
            mock_mgr.process_corroboration.return_value = mock_trust
            mock_get_mgr.return_value = mock_mgr

            result = process_corroboration(node_id)

            mock_mgr.process_corroboration.assert_called_once_with(node_id, None, None)
            assert result == mock_trust

    def test_process_dispute_function(self):
        """Test module-level process_dispute."""
        node_id = uuid4()
        
        with patch("valence.federation.trust.get_trust_manager") as mock_get_mgr:
            mock_mgr = MagicMock()
            mock_trust = MagicMock()
            mock_mgr.process_dispute.return_value = mock_trust
            mock_get_mgr.return_value = mock_mgr

            result = process_dispute(node_id, severity=1.5)

            mock_mgr.process_dispute.assert_called_once_with(node_id, None, None, 1.5)
            assert result == mock_trust

    def test_assess_and_respond_to_threat_function(self):
        """Test module-level assess_and_respond_to_threat."""
        node_id = uuid4()
        
        with patch("valence.federation.trust.get_trust_manager") as mock_get_mgr:
            mock_mgr = MagicMock()
            mock_mgr.assess_threat_level.return_value = (
                ThreatLevel.MEDIUM,
                {"threat_score": 0.45}
            )
            mock_mgr.apply_threat_response.return_value = True
            mock_get_mgr.return_value = mock_mgr

            level, assessment = assess_and_respond_to_threat(node_id)

            mock_mgr.assess_threat_level.assert_called_once_with(node_id)
            mock_mgr.apply_threat_response.assert_called_once()
            assert level == ThreatLevel.MEDIUM
