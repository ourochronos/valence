"""Tests for anomaly detection - AnomalyDetector and rules."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from valence.privacy.anomaly import (
    AnomalyType,
    AnomalyAlert,
    RuleConfig,
    AnomalyDetector,
    get_anomaly_detector,
    set_anomaly_detector,
)


class TestAnomalyType:
    """Tests for AnomalyType enum."""
    
    def test_anomaly_types_exist(self):
        """Verify all expected anomaly types are defined."""
        assert AnomalyType.RAPID_SHARING.value == "rapid_sharing"
        assert AnomalyType.BULK_ACCESS.value == "bulk_access"
        assert AnomalyType.UNUSUAL_HOURS.value == "unusual_hours"
        assert AnomalyType.FAILED_AUTH_SPIKE.value == "failed_auth_spike"
        assert AnomalyType.MASS_REVOCATION.value == "mass_revocation"
        assert AnomalyType.TRUST_ABUSE.value == "trust_abuse"
    
    def test_type_from_string(self):
        """Test creating type from string value."""
        assert AnomalyType("rapid_sharing") == AnomalyType.RAPID_SHARING
        assert AnomalyType("failed_auth_spike") == AnomalyType.FAILED_AUTH_SPIKE


class TestAnomalyAlert:
    """Tests for AnomalyAlert dataclass."""
    
    def test_create_alert(self):
        """Test creating an anomaly alert."""
        now = datetime.now(timezone.utc)
        alert = AnomalyAlert(
            anomaly_type=AnomalyType.RAPID_SHARING,
            actor_did="did:key:alice",
            timestamp=now,
            details={"target_did": "did:key:bob"},
            severity=3,
            event_count=15,
            window_seconds=3600.0,
        )
        
        assert alert.anomaly_type == AnomalyType.RAPID_SHARING
        assert alert.actor_did == "did:key:alice"
        assert alert.timestamp == now
        assert alert.details["target_did"] == "did:key:bob"
        assert alert.severity == 3
        assert alert.event_count == 15
        assert alert.window_seconds == 3600.0
    
    def test_default_values(self):
        """Test default values for optional fields."""
        now = datetime.now(timezone.utc)
        alert = AnomalyAlert(
            anomaly_type=AnomalyType.BULK_ACCESS,
            actor_did="did:key:alice",
            timestamp=now,
            details={},
        )
        
        assert alert.severity == 3
        assert alert.event_count == 0
        assert alert.window_seconds == 0.0
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now(timezone.utc)
        alert = AnomalyAlert(
            anomaly_type=AnomalyType.FAILED_AUTH_SPIKE,
            actor_did="did:key:alice",
            timestamp=now,
            details={"reason": "invalid_signature"},
            severity=4,
            event_count=5,
            window_seconds=300.0,
        )
        
        data = alert.to_dict()
        assert data["anomaly_type"] == "failed_auth_spike"
        assert data["actor_did"] == "did:key:alice"
        assert data["timestamp"] == now.isoformat()
        assert data["details"]["reason"] == "invalid_signature"
        assert data["severity"] == 4
        assert data["event_count"] == 5
        assert data["window_seconds"] == 300.0


class TestRuleConfig:
    """Tests for RuleConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = RuleConfig()
        
        assert config.enabled is True
        assert config.threshold == 10
        assert config.window_seconds == 3600.0
        assert config.cooldown_seconds == 300.0
        assert config.severity == 3
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = RuleConfig(
            enabled=False,
            threshold=50,
            window_seconds=600.0,
            cooldown_seconds=60.0,
            severity=5,
        )
        
        assert config.enabled is False
        assert config.threshold == 50
        assert config.window_seconds == 600.0
        assert config.cooldown_seconds == 60.0
        assert config.severity == 5


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a fresh detector for each test."""
        return AnomalyDetector()
    
    @pytest.fixture
    def alerts(self):
        """List to collect alerts during tests."""
        return []
    
    @pytest.fixture
    def detector_with_callback(self, detector, alerts):
        """Detector with callback that collects alerts."""
        detector.set_callback(lambda alert: alerts.append(alert))
        return detector
    
    # Rule configuration tests
    
    def test_default_rule_configs(self, detector):
        """Test that default rule configs are sensible."""
        rapid = detector.get_rule_config(AnomalyType.RAPID_SHARING)
        assert rapid.enabled is True
        assert rapid.threshold == 10
        assert rapid.window_seconds == 3600.0  # 1 hour
        
        bulk = detector.get_rule_config(AnomalyType.BULK_ACCESS)
        assert bulk.threshold == 100
        assert bulk.window_seconds == 60.0  # 1 minute
        
        failed = detector.get_rule_config(AnomalyType.FAILED_AUTH_SPIKE)
        assert failed.threshold == 5
        assert failed.severity == 4  # Higher severity
    
    def test_configure_rule(self, detector):
        """Test modifying rule configuration."""
        detector.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=20,
            window_seconds=1800.0,
            severity=4,
        )
        
        config = detector.get_rule_config(AnomalyType.RAPID_SHARING)
        assert config.threshold == 20
        assert config.window_seconds == 1800.0
        assert config.severity == 4
        assert config.enabled is True  # Unchanged
    
    def test_enable_disable_rule(self, detector):
        """Test enabling and disabling rules."""
        detector.disable_rule(AnomalyType.UNUSUAL_HOURS)
        assert detector.get_rule_config(AnomalyType.UNUSUAL_HOURS).enabled is False
        
        detector.enable_rule(AnomalyType.UNUSUAL_HOURS)
        assert detector.get_rule_config(AnomalyType.UNUSUAL_HOURS).enabled is True
    
    def test_severity_clamped(self, detector):
        """Test that severity is clamped to 1-5."""
        detector.configure_rule(AnomalyType.RAPID_SHARING, severity=10)
        assert detector.get_rule_config(AnomalyType.RAPID_SHARING).severity == 5
        
        detector.configure_rule(AnomalyType.RAPID_SHARING, severity=0)
        assert detector.get_rule_config(AnomalyType.RAPID_SHARING).severity == 1
    
    # Rapid sharing tests
    
    def test_rapid_sharing_triggers_alert(self, detector_with_callback, alerts):
        """Test that rapid sharing triggers an alert at threshold."""
        detector_with_callback.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=5,
            window_seconds=3600.0,
        )
        
        # Record shares below threshold
        for i in range(4):
            result = detector_with_callback.record_share("did:key:alice")
            assert result is None
        
        assert len(alerts) == 0
        
        # This should trigger
        result = detector_with_callback.record_share(
            "did:key:alice",
            target_did="did:key:bob",
        )
        
        assert result is not None
        assert result.anomaly_type == AnomalyType.RAPID_SHARING
        assert result.actor_did == "did:key:alice"
        assert result.event_count == 5
        assert len(alerts) == 1
    
    def test_rapid_sharing_per_actor(self, detector_with_callback, alerts):
        """Test that rapid sharing is tracked per actor."""
        detector_with_callback.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=3,
        )
        
        # Alice shares 3 times - should trigger
        for _ in range(3):
            detector_with_callback.record_share("did:key:alice")
        
        assert len(alerts) == 1
        assert alerts[0].actor_did == "did:key:alice"
        
        # Bob shares 2 times - should not trigger
        for _ in range(2):
            detector_with_callback.record_share("did:key:bob")
        
        assert len(alerts) == 1  # Still just Alice's alert
    
    # Bulk access tests
    
    def test_bulk_access_triggers_alert(self, detector_with_callback, alerts):
        """Test that bulk access triggers an alert."""
        detector_with_callback.configure_rule(
            AnomalyType.BULK_ACCESS,
            threshold=10,
            window_seconds=60.0,
        )
        
        for i in range(10):
            detector_with_callback.record_access(
                "did:key:alice",
                resource=f"belief:{i}",
            )
        
        assert len(alerts) == 1
        assert alerts[0].anomaly_type == AnomalyType.BULK_ACCESS
        assert alerts[0].event_count == 10
    
    # Failed auth tests
    
    def test_failed_auth_spike(self, detector_with_callback, alerts):
        """Test failed auth spike detection."""
        detector_with_callback.configure_rule(
            AnomalyType.FAILED_AUTH_SPIKE,
            threshold=3,
            window_seconds=300.0,
        )
        
        for _ in range(3):
            detector_with_callback.record_failed_auth(
                "did:key:attacker",
                reason="invalid_signature",
            )
        
        assert len(alerts) == 1
        assert alerts[0].anomaly_type == AnomalyType.FAILED_AUTH_SPIKE
        assert alerts[0].severity == 4  # High severity
    
    # Unusual hours tests
    
    def test_unusual_hours_detection(self, detector_with_callback, alerts):
        """Test unusual hours detection."""
        detector_with_callback.set_normal_hours(9, 17)  # 9 AM - 5 PM
        
        # Activity at 3 AM should trigger
        late_night = datetime(2026, 2, 4, 3, 0, 0, tzinfo=timezone.utc)
        result = detector_with_callback.record_activity(
            "did:key:alice",
            timestamp=late_night,
            action="access_belief",
        )
        
        assert result is not None
        assert result.anomaly_type == AnomalyType.UNUSUAL_HOURS
        assert result.details["hour"] == 3
        assert len(alerts) == 1
    
    def test_normal_hours_no_alert(self, detector_with_callback, alerts):
        """Test that activity during normal hours doesn't alert."""
        detector_with_callback.set_normal_hours(9, 17)
        
        # Activity at 2 PM should not trigger
        normal_time = datetime(2026, 2, 4, 14, 0, 0, tzinfo=timezone.utc)
        result = detector_with_callback.record_activity(
            "did:key:alice",
            timestamp=normal_time,
        )
        
        assert result is None
        assert len(alerts) == 0
    
    # Mass revocation tests
    
    def test_mass_revocation(self, detector_with_callback, alerts):
        """Test mass revocation detection."""
        detector_with_callback.configure_rule(
            AnomalyType.MASS_REVOCATION,
            threshold=5,
        )
        
        for i in range(5):
            detector_with_callback.record_revocation(
                "did:key:alice",
                target_did=f"did:key:user{i}",
            )
        
        assert len(alerts) == 1
        assert alerts[0].anomaly_type == AnomalyType.MASS_REVOCATION
    
    # Trust abuse tests
    
    def test_trust_abuse(self, detector_with_callback, alerts):
        """Test trust abuse pattern detection."""
        detector_with_callback.configure_rule(
            AnomalyType.TRUST_ABUSE,
            threshold=5,
        )
        
        # Rapid grant/revoke cycles
        for i in range(5):
            detector_with_callback.record_trust_change(
                "did:key:alice",
                target_did="did:key:bob",
                change_type="grant" if i % 2 == 0 else "revoke",
            )
        
        assert len(alerts) == 1
        assert alerts[0].anomaly_type == AnomalyType.TRUST_ABUSE
    
    # Cooldown tests
    
    def test_cooldown_prevents_flooding(self, detector_with_callback, alerts):
        """Test that cooldown prevents alert flooding."""
        detector_with_callback.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=3,
            cooldown_seconds=60.0,
        )
        
        now = datetime.now(timezone.utc)
        
        # First burst - should alert
        for i in range(3):
            detector_with_callback.record_share(
                "did:key:alice",
                timestamp=now + timedelta(seconds=i),
            )
        
        assert len(alerts) == 1
        
        # More shares within cooldown - should not alert again
        for i in range(5):
            detector_with_callback.record_share(
                "did:key:alice",
                timestamp=now + timedelta(seconds=10 + i),
            )
        
        assert len(alerts) == 1  # Still just one alert
    
    def test_cooldown_expires(self, detector_with_callback, alerts):
        """Test that alerts resume after cooldown expires."""
        detector_with_callback.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=3,
            cooldown_seconds=60.0,
        )
        
        now = datetime.now(timezone.utc)
        
        # First burst
        for i in range(3):
            detector_with_callback.record_share(
                "did:key:alice",
                timestamp=now + timedelta(seconds=i),
            )
        
        assert len(alerts) == 1
        
        # After cooldown expires
        after_cooldown = now + timedelta(seconds=120)
        for i in range(3):
            detector_with_callback.record_share(
                "did:key:alice",
                timestamp=after_cooldown + timedelta(seconds=i),
            )
        
        assert len(alerts) == 2  # Second alert
    
    def test_reset_cooldown(self, detector_with_callback, alerts):
        """Test manual cooldown reset."""
        detector_with_callback.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=3,
            cooldown_seconds=3600.0,  # Long cooldown
        )
        
        # Trigger first alert
        for _ in range(3):
            detector_with_callback.record_share("did:key:alice")
        
        assert len(alerts) == 1
        
        # More shares - blocked by cooldown
        for _ in range(3):
            detector_with_callback.record_share("did:key:alice")
        
        assert len(alerts) == 1
        
        # Reset cooldown
        detector_with_callback.reset_cooldown(
            AnomalyType.RAPID_SHARING,
            "did:key:alice",
        )
        
        # Now should trigger again
        for _ in range(3):
            detector_with_callback.record_share("did:key:alice")
        
        assert len(alerts) == 2
    
    # Time window tests
    
    def test_events_outside_window_not_counted(self, detector_with_callback, alerts):
        """Test that old events are pruned from count."""
        detector_with_callback.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=5,
            window_seconds=60.0,  # 1 minute window
        )
        
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(seconds=120)  # 2 minutes ago
        
        # 3 old events (outside window)
        for i in range(3):
            detector_with_callback.record_share(
                "did:key:alice",
                timestamp=old_time + timedelta(seconds=i),
            )
        
        # 3 recent events (inside window)
        for i in range(3):
            detector_with_callback.record_share(
                "did:key:alice",
                timestamp=now + timedelta(seconds=i),
            )
        
        # Should not trigger (only 3 events in window, threshold is 5)
        assert len(alerts) == 0
    
    # Disabled rule tests
    
    def test_disabled_rule_no_alert(self, detector_with_callback, alerts):
        """Test that disabled rules don't generate alerts."""
        detector_with_callback.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=3,
            enabled=False,
        )
        
        for _ in range(10):
            detector_with_callback.record_share("did:key:alice")
        
        assert len(alerts) == 0
    
    # Event count and clear tests
    
    def test_get_event_count(self, detector):
        """Test getting current event count."""
        detector.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=100,  # High threshold
            window_seconds=3600.0,
        )
        
        for _ in range(5):
            detector.record_share("did:key:alice")
        
        count = detector.get_event_count(AnomalyType.RAPID_SHARING, "did:key:alice")
        assert count == 5
        
        # Different actor has 0
        count = detector.get_event_count(AnomalyType.RAPID_SHARING, "did:key:bob")
        assert count == 0
    
    def test_clear_events(self, detector):
        """Test clearing events."""
        detector.configure_rule(AnomalyType.RAPID_SHARING, threshold=100)
        
        for _ in range(5):
            detector.record_share("did:key:alice")
            detector.record_share("did:key:bob")
        
        # Clear Alice's events
        detector.clear_events("did:key:alice")
        
        assert detector.get_event_count(AnomalyType.RAPID_SHARING, "did:key:alice") == 0
        assert detector.get_event_count(AnomalyType.RAPID_SHARING, "did:key:bob") == 5
        
        # Clear all events
        detector.clear_events()
        assert detector.get_event_count(AnomalyType.RAPID_SHARING, "did:key:bob") == 0
    
    # Callback tests
    
    def test_no_callback_still_returns_alert(self, detector):
        """Test that alerts are returned even without callback."""
        detector.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=3,
        )
        
        for _ in range(2):
            result = detector.record_share("did:key:alice")
            assert result is None
        
        result = detector.record_share("did:key:alice")
        assert result is not None
        assert result.anomaly_type == AnomalyType.RAPID_SHARING
    
    def test_callback_receives_alert(self, detector):
        """Test that callback receives alert with full details."""
        received_alerts: List[AnomalyAlert] = []
        
        def callback(alert: AnomalyAlert):
            received_alerts.append(alert)
        
        detector.set_callback(callback)
        detector.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=2,
            severity=4,
        )
        
        detector.record_share("did:key:alice", target_did="did:key:bob")
        detector.record_share("did:key:alice", target_did="did:key:charlie")
        
        assert len(received_alerts) == 1
        alert = received_alerts[0]
        assert alert.actor_did == "did:key:alice"
        assert alert.severity == 4
        assert alert.event_count == 2


class TestSingleton:
    """Tests for module-level singleton functions."""
    
    def test_get_anomaly_detector(self):
        """Test getting the singleton detector."""
        detector1 = get_anomaly_detector()
        detector2 = get_anomaly_detector()
        
        assert detector1 is detector2
    
    def test_set_anomaly_detector(self):
        """Test setting a custom detector."""
        original = get_anomaly_detector()
        
        custom = AnomalyDetector()
        custom.configure_rule(AnomalyType.RAPID_SHARING, threshold=999)
        
        set_anomaly_detector(custom)
        
        assert get_anomaly_detector() is custom
        assert get_anomaly_detector().get_rule_config(
            AnomalyType.RAPID_SHARING
        ).threshold == 999
        
        # Restore original for other tests
        set_anomaly_detector(original)


class TestThreadSafety:
    """Tests for thread safety."""
    
    def test_concurrent_events(self):
        """Test that concurrent event recording is thread-safe."""
        import threading
        
        detector = AnomalyDetector()
        detector.configure_rule(
            AnomalyType.RAPID_SHARING,
            threshold=1000,  # High threshold
            window_seconds=3600.0,
        )
        
        errors = []
        
        def record_events():
            try:
                for _ in range(100):
                    detector.record_share("did:key:alice")
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=record_events) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        # Should have recorded ~1000 events
        count = detector.get_event_count(AnomalyType.RAPID_SHARING, "did:key:alice")
        assert count == 1000
