"""Anomaly detection for Valence - detect suspicious activity patterns.

Implements rule-based anomaly detection with configurable thresholds,
cooldowns to prevent alert flooding, and callback-based alerting.

Issue #85: Basic anomaly detection rules.
"""

from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import threading


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    
    RAPID_SHARING = "rapid_sharing"          # >N shares per hour
    BULK_ACCESS = "bulk_access"              # >N accesses in short time
    UNUSUAL_HOURS = "unusual_hours"          # Activity outside normal hours
    FAILED_AUTH_SPIKE = "failed_auth_spike"  # >N failed auths in window
    MASS_REVOCATION = "mass_revocation"      # >N revocations in short time
    TRUST_ABUSE = "trust_abuse"              # Rapid trust/revoke cycles


@dataclass
class AnomalyAlert:
    """Alert generated when an anomaly is detected.
    
    Attributes:
        anomaly_type: Type of anomaly detected
        actor_did: DID of the entity exhibiting suspicious behavior
        timestamp: When the anomaly was detected
        details: Additional context about the anomaly
        severity: Severity level (1-5, 5 being most severe)
        event_count: Number of events that triggered this alert
        window_seconds: Time window over which events were counted
    """
    
    anomaly_type: AnomalyType
    actor_did: str
    timestamp: datetime
    details: Dict[str, Any]
    severity: int = 3
    event_count: int = 0
    window_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "actor_did": self.actor_did,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "severity": self.severity,
            "event_count": self.event_count,
            "window_seconds": self.window_seconds,
        }


@dataclass
class RuleConfig:
    """Configuration for a single anomaly detection rule.
    
    Attributes:
        enabled: Whether this rule is active
        threshold: Count threshold to trigger (e.g., N events)
        window_seconds: Time window for counting events
        cooldown_seconds: Minimum time between alerts of this type per actor
        severity: Alert severity when triggered (1-5)
    """
    
    enabled: bool = True
    threshold: int = 10
    window_seconds: float = 3600.0  # 1 hour default
    cooldown_seconds: float = 300.0  # 5 minutes default
    severity: int = 3


class AlertCallback(Protocol):
    """Protocol for alert callback functions."""
    
    def __call__(self, alert: AnomalyAlert) -> None:
        """Called when an anomaly is detected."""
        ...


class AnomalyDetector:
    """Rule-based anomaly detector for suspicious activity patterns.
    
    Monitors event streams and triggers alerts when configurable
    thresholds are exceeded. Includes cooldown logic to prevent
    alert flooding.
    
    Usage:
        detector = AnomalyDetector()
        detector.set_callback(my_alert_handler)
        detector.configure_rule(AnomalyType.RAPID_SHARING, threshold=20)
        
        # Feed events
        detector.record_share("did:key:alice")
        detector.record_share("did:key:alice")  # ... repeated
        # Alert triggered when threshold exceeded
    """
    
    def __init__(self):
        """Initialize detector with default rule configurations."""
        self._lock = threading.Lock()
        self._callback: Optional[AlertCallback] = None
        
        # Rule configurations
        self._rules: Dict[AnomalyType, RuleConfig] = {
            AnomalyType.RAPID_SHARING: RuleConfig(
                threshold=10,
                window_seconds=3600.0,  # 1 hour
                cooldown_seconds=300.0,
                severity=3,
            ),
            AnomalyType.BULK_ACCESS: RuleConfig(
                threshold=100,
                window_seconds=60.0,  # 1 minute
                cooldown_seconds=300.0,
                severity=3,
            ),
            AnomalyType.UNUSUAL_HOURS: RuleConfig(
                threshold=1,  # Any activity
                window_seconds=3600.0,
                cooldown_seconds=3600.0,  # 1 hour cooldown
                severity=2,
            ),
            AnomalyType.FAILED_AUTH_SPIKE: RuleConfig(
                threshold=5,
                window_seconds=300.0,  # 5 minutes
                cooldown_seconds=600.0,
                severity=4,
            ),
            AnomalyType.MASS_REVOCATION: RuleConfig(
                threshold=10,
                window_seconds=300.0,  # 5 minutes
                cooldown_seconds=300.0,
                severity=4,
            ),
            AnomalyType.TRUST_ABUSE: RuleConfig(
                threshold=5,
                window_seconds=3600.0,  # 1 hour
                cooldown_seconds=600.0,
                severity=4,
            ),
        }
        
        # Event history per actor per anomaly type
        # Dict[AnomalyType, Dict[actor_did, List[timestamp]]]
        self._events: Dict[AnomalyType, Dict[str, List[datetime]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Last alert time per actor per anomaly type (for cooldown)
        # Dict[AnomalyType, Dict[actor_did, datetime]]
        self._last_alerts: Dict[AnomalyType, Dict[str, datetime]] = defaultdict(dict)
        
        # Configurable normal hours (default 6 AM - 10 PM)
        self._normal_hours_start: int = 6   # 6 AM
        self._normal_hours_end: int = 22    # 10 PM
    
    def set_callback(self, callback: AlertCallback) -> None:
        """Set the callback function for alerts.
        
        Args:
            callback: Function to call when an anomaly is detected
        """
        with self._lock:
            self._callback = callback
    
    def configure_rule(
        self,
        anomaly_type: AnomalyType,
        *,
        enabled: Optional[bool] = None,
        threshold: Optional[int] = None,
        window_seconds: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
        severity: Optional[int] = None,
    ) -> None:
        """Configure a specific anomaly detection rule.
        
        Args:
            anomaly_type: The rule to configure
            enabled: Whether the rule is active (None = no change)
            threshold: Event count threshold (None = no change)
            window_seconds: Time window for counting (None = no change)
            cooldown_seconds: Cooldown between alerts (None = no change)
            severity: Alert severity 1-5 (None = no change)
        """
        with self._lock:
            rule = self._rules[anomaly_type]
            if enabled is not None:
                rule.enabled = enabled
            if threshold is not None:
                rule.threshold = threshold
            if window_seconds is not None:
                rule.window_seconds = window_seconds
            if cooldown_seconds is not None:
                rule.cooldown_seconds = cooldown_seconds
            if severity is not None:
                rule.severity = max(1, min(5, severity))
    
    def get_rule_config(self, anomaly_type: AnomalyType) -> RuleConfig:
        """Get the current configuration for a rule.
        
        Returns a copy to prevent external modification.
        """
        with self._lock:
            rule = self._rules[anomaly_type]
            return RuleConfig(
                enabled=rule.enabled,
                threshold=rule.threshold,
                window_seconds=rule.window_seconds,
                cooldown_seconds=rule.cooldown_seconds,
                severity=rule.severity,
            )
    
    def enable_rule(self, anomaly_type: AnomalyType) -> None:
        """Enable a specific rule."""
        self.configure_rule(anomaly_type, enabled=True)
    
    def disable_rule(self, anomaly_type: AnomalyType) -> None:
        """Disable a specific rule."""
        self.configure_rule(anomaly_type, enabled=False)
    
    def set_normal_hours(self, start_hour: int, end_hour: int) -> None:
        """Set the normal operating hours (for unusual hours detection).
        
        Args:
            start_hour: Start of normal hours (0-23)
            end_hour: End of normal hours (0-23)
        """
        with self._lock:
            self._normal_hours_start = max(0, min(23, start_hour))
            self._normal_hours_end = max(0, min(23, end_hour))
    
    def _is_unusual_hour(self, timestamp: datetime) -> bool:
        """Check if a timestamp falls outside normal hours."""
        hour = timestamp.hour
        if self._normal_hours_start <= self._normal_hours_end:
            # Normal case: e.g., 6 AM to 10 PM
            return hour < self._normal_hours_start or hour >= self._normal_hours_end
        else:
            # Overnight case: e.g., 10 PM to 6 AM is normal
            return self._normal_hours_end <= hour < self._normal_hours_start
    
    def _prune_old_events(
        self,
        anomaly_type: AnomalyType,
        actor_did: str,
        window_seconds: float,
        now: datetime,
    ) -> None:
        """Remove events outside the time window."""
        cutoff = now - timedelta(seconds=window_seconds)
        events = self._events[anomaly_type][actor_did]
        self._events[anomaly_type][actor_did] = [
            ts for ts in events if ts > cutoff
        ]
    
    def _check_cooldown(
        self,
        anomaly_type: AnomalyType,
        actor_did: str,
        cooldown_seconds: float,
        now: datetime,
    ) -> bool:
        """Check if we're still in cooldown period for this actor/type.
        
        Returns True if in cooldown (should NOT alert).
        """
        last_alert = self._last_alerts[anomaly_type].get(actor_did)
        if last_alert is None:
            return False
        return (now - last_alert).total_seconds() < cooldown_seconds
    
    def _trigger_alert(
        self,
        anomaly_type: AnomalyType,
        actor_did: str,
        event_count: int,
        window_seconds: float,
        severity: int,
        details: Dict[str, Any],
        now: datetime,
    ) -> Optional[AnomalyAlert]:
        """Create and dispatch an alert."""
        alert = AnomalyAlert(
            anomaly_type=anomaly_type,
            actor_did=actor_did,
            timestamp=now,
            details=details,
            severity=severity,
            event_count=event_count,
            window_seconds=window_seconds,
        )
        
        # Record alert time for cooldown
        self._last_alerts[anomaly_type][actor_did] = now
        
        # Call callback if set
        if self._callback:
            self._callback(alert)
        
        return alert
    
    def _record_event(
        self,
        anomaly_type: AnomalyType,
        actor_did: str,
        timestamp: Optional[datetime] = None,
        extra_details: Optional[Dict[str, Any]] = None,
    ) -> Optional[AnomalyAlert]:
        """Record an event and check if it triggers an anomaly.
        
        Returns an AnomalyAlert if threshold exceeded, None otherwise.
        """
        with self._lock:
            rule = self._rules[anomaly_type]
            if not rule.enabled:
                return None
            
            now = timestamp or datetime.now(timezone.utc)
            
            # Prune old events
            self._prune_old_events(anomaly_type, actor_did, rule.window_seconds, now)
            
            # Add new event
            self._events[anomaly_type][actor_did].append(now)
            
            # Check threshold
            event_count = len(self._events[anomaly_type][actor_did])
            if event_count < rule.threshold:
                return None
            
            # Check cooldown
            if self._check_cooldown(anomaly_type, actor_did, rule.cooldown_seconds, now):
                return None
            
            # Trigger alert
            details = extra_details or {}
            details["rule_threshold"] = rule.threshold
            details["rule_window_seconds"] = rule.window_seconds
            
            return self._trigger_alert(
                anomaly_type=anomaly_type,
                actor_did=actor_did,
                event_count=event_count,
                window_seconds=rule.window_seconds,
                severity=rule.severity,
                details=details,
                now=now,
            )
    
    # Public event recording methods
    
    def record_share(
        self,
        actor_did: str,
        timestamp: Optional[datetime] = None,
        target_did: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> Optional[AnomalyAlert]:
        """Record a share event.
        
        Args:
            actor_did: DID of the entity sharing
            timestamp: When the share occurred (defaults to now)
            target_did: DID of the share recipient
            resource: Resource being shared
            
        Returns:
            AnomalyAlert if rapid sharing detected, None otherwise
        """
        details = {}
        if target_did:
            details["target_did"] = target_did
        if resource:
            details["resource"] = resource
        
        return self._record_event(
            AnomalyType.RAPID_SHARING,
            actor_did,
            timestamp,
            details,
        )
    
    def record_access(
        self,
        actor_did: str,
        timestamp: Optional[datetime] = None,
        resource: Optional[str] = None,
    ) -> Optional[AnomalyAlert]:
        """Record an access event.
        
        Args:
            actor_did: DID of the entity accessing
            timestamp: When the access occurred
            resource: Resource being accessed
            
        Returns:
            AnomalyAlert if bulk access detected, None otherwise
        """
        details = {}
        if resource:
            details["resource"] = resource
        
        return self._record_event(
            AnomalyType.BULK_ACCESS,
            actor_did,
            timestamp,
            details,
        )
    
    def record_activity(
        self,
        actor_did: str,
        timestamp: Optional[datetime] = None,
        action: Optional[str] = None,
    ) -> Optional[AnomalyAlert]:
        """Record general activity for unusual hours detection.
        
        Args:
            actor_did: DID of the entity
            timestamp: When the activity occurred
            action: Description of the activity
            
        Returns:
            AnomalyAlert if unusual hours detected, None otherwise
        """
        now = timestamp or datetime.now(timezone.utc)
        
        with self._lock:
            rule = self._rules[AnomalyType.UNUSUAL_HOURS]
            if not rule.enabled:
                return None
            
            # Check if this is an unusual hour
            if not self._is_unusual_hour(now):
                return None
            
            # Check cooldown
            if self._check_cooldown(
                AnomalyType.UNUSUAL_HOURS,
                actor_did,
                rule.cooldown_seconds,
                now,
            ):
                return None
            
            details = {"hour": now.hour}
            if action:
                details["action"] = action
            details["normal_hours"] = f"{self._normal_hours_start}:00-{self._normal_hours_end}:00"
            
            return self._trigger_alert(
                anomaly_type=AnomalyType.UNUSUAL_HOURS,
                actor_did=actor_did,
                event_count=1,
                window_seconds=0,
                severity=rule.severity,
                details=details,
                now=now,
            )
    
    def record_failed_auth(
        self,
        actor_did: str,
        timestamp: Optional[datetime] = None,
        reason: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> Optional[AnomalyAlert]:
        """Record a failed authentication attempt.
        
        Args:
            actor_did: DID of the entity attempting auth
            timestamp: When the failure occurred
            reason: Reason for the failure
            resource: Resource being accessed
            
        Returns:
            AnomalyAlert if failed auth spike detected, None otherwise
        """
        details = {}
        if reason:
            details["reason"] = reason
        if resource:
            details["resource"] = resource
        
        return self._record_event(
            AnomalyType.FAILED_AUTH_SPIKE,
            actor_did,
            timestamp,
            details,
        )
    
    def record_revocation(
        self,
        actor_did: str,
        timestamp: Optional[datetime] = None,
        target_did: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> Optional[AnomalyAlert]:
        """Record a share revocation event.
        
        Args:
            actor_did: DID of the entity revoking
            timestamp: When the revocation occurred
            target_did: DID whose access was revoked
            resource: Resource being revoked
            
        Returns:
            AnomalyAlert if mass revocation detected, None otherwise
        """
        details = {}
        if target_did:
            details["target_did"] = target_did
        if resource:
            details["resource"] = resource
        
        return self._record_event(
            AnomalyType.MASS_REVOCATION,
            actor_did,
            timestamp,
            details,
        )
    
    def record_trust_change(
        self,
        actor_did: str,
        timestamp: Optional[datetime] = None,
        target_did: Optional[str] = None,
        change_type: Optional[str] = None,
    ) -> Optional[AnomalyAlert]:
        """Record a trust grant or revocation.
        
        Args:
            actor_did: DID of the entity changing trust
            timestamp: When the change occurred
            target_did: DID whose trust was changed
            change_type: "grant" or "revoke"
            
        Returns:
            AnomalyAlert if trust abuse pattern detected, None otherwise
        """
        details = {}
        if target_did:
            details["target_did"] = target_did
        if change_type:
            details["change_type"] = change_type
        
        return self._record_event(
            AnomalyType.TRUST_ABUSE,
            actor_did,
            timestamp,
            details,
        )
    
    def get_event_count(
        self,
        anomaly_type: AnomalyType,
        actor_did: str,
    ) -> int:
        """Get the current event count for an actor within the rule's window.
        
        Useful for monitoring and debugging.
        """
        with self._lock:
            rule = self._rules[anomaly_type]
            now = datetime.now(timezone.utc)
            self._prune_old_events(anomaly_type, actor_did, rule.window_seconds, now)
            return len(self._events[anomaly_type][actor_did])
    
    def clear_events(self, actor_did: Optional[str] = None) -> None:
        """Clear recorded events.
        
        Args:
            actor_did: Clear only for this actor, or all if None
        """
        with self._lock:
            if actor_did is None:
                self._events.clear()
                self._events = defaultdict(lambda: defaultdict(list))
            else:
                for anomaly_type in AnomalyType:
                    if actor_did in self._events[anomaly_type]:
                        del self._events[anomaly_type][actor_did]
    
    def reset_cooldown(
        self,
        anomaly_type: Optional[AnomalyType] = None,
        actor_did: Optional[str] = None,
    ) -> None:
        """Reset cooldown timers.
        
        Args:
            anomaly_type: Reset for this type only, or all if None
            actor_did: Reset for this actor only, or all if None
        """
        with self._lock:
            if anomaly_type is None and actor_did is None:
                self._last_alerts.clear()
                self._last_alerts = defaultdict(dict)
            elif anomaly_type is not None and actor_did is not None:
                if actor_did in self._last_alerts[anomaly_type]:
                    del self._last_alerts[anomaly_type][actor_did]
            elif anomaly_type is not None:
                self._last_alerts[anomaly_type].clear()
            else:  # actor_did is not None
                for atype in AnomalyType:
                    if actor_did in self._last_alerts[atype]:
                        del self._last_alerts[atype][actor_did]


# Module-level singleton for convenience
_default_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    """Get the default anomaly detector singleton."""
    global _default_detector
    if _default_detector is None:
        _default_detector = AnomalyDetector()
    return _default_detector


def set_anomaly_detector(detector: AnomalyDetector) -> None:
    """Set the default anomaly detector singleton."""
    global _default_detector
    _default_detector = detector
