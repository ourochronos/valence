"""Tests for trust decay functionality.

Tests the time-based trust decay implementation added in Issue #61.
"""

import math
import pytest
from datetime import datetime, timedelta, timezone

from valence.privacy.trust import TrustEdge, DecayModel


class TestDecayModel:
    """Tests for DecayModel enum."""
    
    def test_from_string_valid(self):
        """Test converting valid strings to DecayModel."""
        assert DecayModel.from_string("none") == DecayModel.NONE
        assert DecayModel.from_string("linear") == DecayModel.LINEAR
        assert DecayModel.from_string("exponential") == DecayModel.EXPONENTIAL
        assert DecayModel.from_string("EXPONENTIAL") == DecayModel.EXPONENTIAL
        assert DecayModel.from_string("Linear") == DecayModel.LINEAR
    
    def test_from_string_invalid_defaults_to_exponential(self):
        """Test that invalid strings default to EXPONENTIAL."""
        assert DecayModel.from_string("invalid") == DecayModel.EXPONENTIAL
        assert DecayModel.from_string("") == DecayModel.EXPONENTIAL


class TestTrustEdgeDecay:
    """Tests for TrustEdge decay functionality."""
    
    def test_create_edge_with_decay(self):
        """Test creating edge with decay settings."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            integrity=0.8,
            decay_rate=0.1,
            decay_model=DecayModel.EXPONENTIAL,
        )
        
        assert edge.decay_rate == 0.1
        assert edge.decay_model == DecayModel.EXPONENTIAL
        assert edge.last_refreshed is not None
    
    def test_default_no_decay(self):
        """Test that default decay_rate is 0 (no decay)."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
        )
        
        assert edge.decay_rate == 0.0
    
    def test_invalid_decay_rate(self):
        """Test that invalid decay rates are rejected."""
        with pytest.raises(ValueError, match="decay_rate must be between"):
            TrustEdge(
                source_did="did:key:alice",
                target_did="did:key:bob",
                decay_rate=1.5,
            )
        
        with pytest.raises(ValueError, match="decay_rate must be between"):
            TrustEdge(
                source_did="did:key:alice",
                target_did="did:key:bob",
                decay_rate=-0.1,
            )
    
    def test_effective_trust_no_decay(self):
        """Test effective_trust with no decay returns base values."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            integrity=0.7,
            confidentiality=0.6,
            judgment=0.5,
            decay_rate=0.0,
        )
        
        effective = edge.effective_trust()
        assert effective["competence"] == 0.8
        assert effective["integrity"] == 0.7
        assert effective["confidentiality"] == 0.6
        assert effective["judgment"] == 0.5
    
    def test_effective_trust_exponential_decay(self):
        """Test exponential decay calculation."""
        # Create edge refreshed 7 days ago
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            judgment=0.8,
            decay_rate=0.1,  # 10% decay per day -> 90% retention
            decay_model=DecayModel.EXPONENTIAL,
            last_refreshed=seven_days_ago,
        )
        
        effective = edge.effective_trust()
        # Expected: 0.8 * (0.9 ^ 7) = 0.8 * 0.4783 = 0.3826
        expected = 0.8 * (0.9 ** 7)
        assert effective["competence"] == pytest.approx(expected, rel=0.01)
    
    def test_effective_trust_linear_decay(self):
        """Test linear decay calculation."""
        # Create edge refreshed 3 days ago
        three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
        
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            judgment=0.8,
            decay_rate=0.1,  # Lose 0.1 per day
            decay_model=DecayModel.LINEAR,
            last_refreshed=three_days_ago,
        )
        
        effective = edge.effective_trust()
        # Expected: 0.8 - (0.1 * 3) = 0.5
        expected = 0.8 - (0.1 * 3)
        assert effective["competence"] == pytest.approx(expected, rel=0.01)
    
    def test_effective_trust_floors_at_zero(self):
        """Test that effective trust doesn't go negative."""
        # Create edge refreshed 10 days ago with high decay
        ten_days_ago = datetime.now(timezone.utc) - timedelta(days=10)
        
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.5,
            decay_rate=0.1,  # Would be 0.5 - 1.0 = -0.5
            decay_model=DecayModel.LINEAR,
            last_refreshed=ten_days_ago,
        )
        
        effective = edge.effective_trust()
        assert effective["competence"] == 0.0
    
    def test_refresh_trust_resets_decay_clock(self):
        """Test that refresh_trust resets the decay clock."""
        old_time = datetime.now(timezone.utc) - timedelta(days=5)
        
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            judgment=0.8,
            decay_rate=0.1,
            decay_model=DecayModel.EXPONENTIAL,
            last_refreshed=old_time,
        )
        
        # Before refresh, trust is decayed
        decayed = edge.effective_trust()
        assert decayed["competence"] < 0.8
        
        # Refresh
        edge.refresh_trust()
        
        # After refresh, trust should be back to base values
        refreshed = edge.effective_trust()
        assert refreshed["competence"] == pytest.approx(0.8, rel=0.01)
    
    def test_refresh_trust_with_new_values(self):
        """Test refreshing with updated trust values."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.5,
        )
        
        edge.refresh_trust(new_values={"competence": 0.9, "integrity": 0.85})
        
        assert edge.competence == 0.9
        assert edge.integrity == 0.85
    
    def test_refresh_trust_validates_new_values(self):
        """Test that invalid new values are rejected."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
        )
        
        with pytest.raises(ValueError, match="competence must be between"):
            edge.refresh_trust(new_values={"competence": 1.5})
    
    def test_with_decay_creates_copy(self):
        """Test that with_decay creates a new edge."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
        )
        
        decaying = edge.with_decay(decay_rate=0.1, decay_model=DecayModel.LINEAR)
        
        assert decaying is not edge
        assert decaying.decay_rate == 0.1
        assert decaying.decay_model == DecayModel.LINEAR
        assert edge.decay_rate == 0.0  # Original unchanged
    
    def test_is_stale(self):
        """Test staleness detection."""
        # Fresh edge is not stale
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            judgment=0.8,
            decay_rate=0.1,
        )
        assert not edge.is_stale()
        
        # Old edge with decay becomes stale
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        edge.last_refreshed = old_time
        assert edge.is_stale(min_effective_trust=0.1)
    
    def test_time_until_stale_no_decay(self):
        """Test time_until_stale with no decay returns None."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            decay_rate=0.0,
        )
        
        assert edge.time_until_stale() is None
    
    def test_time_until_stale_exponential(self):
        """Test time_until_stale calculation for exponential decay."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            judgment=0.8,
            decay_rate=0.1,
            decay_model=DecayModel.EXPONENTIAL,
        )
        
        # overall_trust = geometric mean = 0.8
        # 0.8 * 0.9^days = 0.1
        # days = log(0.1/0.8) / log(0.9)
        expected = math.log(0.1 / 0.8) / math.log(0.9)
        
        time_until = edge.time_until_stale(min_effective_trust=0.1)
        assert time_until == pytest.approx(expected, rel=0.01)
    
    def test_days_since_refresh(self):
        """Test days_since_refresh property."""
        five_days_ago = datetime.now(timezone.utc) - timedelta(days=5)
        
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            last_refreshed=five_days_ago,
        )
        
        assert edge.days_since_refresh == pytest.approx(5.0, rel=0.1)
    
    def test_to_dict_includes_decay_fields(self):
        """Test that to_dict includes decay fields."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            decay_rate=0.15,
            decay_model=DecayModel.LINEAR,
        )
        
        data = edge.to_dict()
        assert data["decay_rate"] == 0.15
        assert data["decay_model"] == "linear"
        assert "last_refreshed" in data
        assert "effective_trust" in data
    
    def test_from_dict_parses_decay_fields(self):
        """Test that from_dict parses decay fields."""
        data = {
            "source_did": "did:key:alice",
            "target_did": "did:key:bob",
            "competence": 0.8,
            "decay_rate": 0.2,
            "decay_model": "linear",
            "last_refreshed": "2025-01-01T12:00:00+00:00",
        }
        
        edge = TrustEdge.from_dict(data)
        
        assert edge.decay_rate == 0.2
        assert edge.decay_model == DecayModel.LINEAR
    
    def test_roundtrip_serialization_with_decay(self):
        """Test serialization roundtrip preserves decay fields."""
        original = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            decay_rate=0.1,
            decay_model=DecayModel.EXPONENTIAL,
        )
        
        restored = TrustEdge.from_dict(original.to_dict())
        
        assert restored.decay_rate == original.decay_rate
        assert restored.decay_model == original.decay_model
    
    def test_string_decay_model_conversion(self):
        """Test that string decay model is converted to enum."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            decay_model="linear",  # type: ignore
        )
        
        assert edge.decay_model == DecayModel.LINEAR
    
    def test_decay_model_none_bypasses_decay(self):
        """Test that DecayModel.NONE bypasses decay even with decay_rate set."""
        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.8,
            decay_rate=0.5,  # High decay rate
            decay_model=DecayModel.NONE,  # But model is NONE
            last_refreshed=old_time,
        )
        
        effective = edge.effective_trust()
        assert effective["competence"] == 0.8  # No decay applied
