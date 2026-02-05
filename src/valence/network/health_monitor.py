"""
Health Monitor - Handles health monitoring for NodeClient.

This module manages:
- Health gossip (sending and receiving observations)
- Router health observation aggregation
- Keepalive pings
- Misbehavior detection
- Eclipse attack anomaly detection

Extracted from NodeClient as part of god class decomposition (Issue #128).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .discovery import DiscoveryClient
    from .node import RouterConnection
    from .messages import (
        HealthGossip,
        RouterHealthObservation,
        MisbehaviorType,
        RouterBehaviorMetrics,
        MisbehaviorEvidence,
        MisbehaviorReport,
        NetworkBaseline,
    )

logger = logging.getLogger(__name__)


@dataclass
class HealthMonitorConfig:
    """Configuration for HealthMonitor."""
    
    # Gossip timing
    gossip_interval: float = 30.0
    gossip_ttl: int = 2
    
    # Observation settings
    own_observation_weight: float = 0.7
    peer_observation_weight: float = 0.3
    max_observations_per_gossip: int = 10
    max_peer_observations: int = 100
    observation_max_age: float = 300.0  # 5 minutes
    
    # Keepalive
    keepalive_interval: float = 2.0
    ping_timeout: float = 3.0
    missed_pings_threshold: int = 2
    
    # Misbehavior detection
    misbehavior_detection_enabled: bool = True
    min_messages_for_detection: int = 20
    delivery_rate_threshold: float = 0.75
    latency_threshold_stddevs: float = 2.0
    ack_failure_threshold: float = 0.30
    mild_severity_threshold: float = 0.3
    severe_severity_threshold: float = 0.7
    auto_avoid_flagged_routers: bool = True
    flagged_router_penalty: float = 0.1
    report_cooldown_seconds: float = 300.0
    max_evidence_per_report: int = 10
    baseline_min_samples: int = 3
    
    # Eclipse anomaly detection
    anomaly_detection_enabled: bool = True
    anomaly_window: float = 60.0
    anomaly_threshold: int = 3


class HealthMonitor:
    """
    Monitors health of routers and network.
    
    Responsible for:
    - Health gossip protocol
    - Router health observation tracking
    - Keepalive pings
    - Misbehavior detection
    - Eclipse attack anomaly detection
    """
    
    def __init__(
        self,
        node_id: str,
        discovery: "DiscoveryClient",
        config: Optional[HealthMonitorConfig] = None,
        on_router_flagged: Optional[Callable[[str, Any], None]] = None,
    ):
        """
        Initialize the HealthMonitor.
        
        Args:
            node_id: This node's ID
            discovery: DiscoveryClient for reporting
            config: Health monitor configuration
            on_router_flagged: Callback when router is flagged for misbehavior
        """
        self.node_id = node_id
        self.discovery = discovery
        self.config = config or HealthMonitorConfig()
        self.on_router_flagged = on_router_flagged
        
        # Health observations
        self._own_observations: Dict[str, "RouterHealthObservation"] = {}
        self._peer_observations: Dict[str, Dict[str, Any]] = {}
        
        # Misbehavior tracking
        self._router_behavior_metrics: Dict[str, "RouterBehaviorMetrics"] = {}
        self._network_baseline: Optional["NetworkBaseline"] = None
        self._flagged_routers: Dict[str, "MisbehaviorReport"] = {}
        self._last_misbehavior_reports: Dict[str, float] = {}
        
        # Eclipse anomaly detection
        self._failure_events: List[Dict[str, Any]] = []
        self._anomaly_alerts: List[Dict[str, Any]] = []
        
        # Keepalive state
        self._missed_pings: Dict[str, int] = {}
        
        # Statistics
        self._stats: Dict[str, int] = {
            "gossip_sent": 0,
            "gossip_received": 0,
            "anomalies_detected": 0,
            "routers_flagged": 0,
            "misbehavior_ack_success": 0,
            "misbehavior_ack_failure": 0,
            "misbehavior_reports_sent": 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get health monitor statistics."""
        return {
            **self._stats,
            "own_observations": len(self._own_observations),
            "peer_observation_sources": len(self._peer_observations),
            "flagged_routers": len(self._flagged_routers),
            "recent_anomalies": len(self._anomaly_alerts),
        }
    
    # -------------------------------------------------------------------------
    # HEALTH OBSERVATIONS
    # -------------------------------------------------------------------------
    
    def update_observation(self, router_id: str, conn: "RouterConnection") -> None:
        """
        Update our own health observation for a router.
        
        Called after receiving pong, after ACK success/failure, etc.
        """
        from .messages import RouterHealthObservation
        
        now = time.time()
        
        obs = RouterHealthObservation(
            router_id=router_id,
            latency_ms=conn.ping_latency_ms,
            success_rate=conn.ack_success_rate,
            failure_count=conn.ack_failure,
            success_count=conn.ack_success,
            last_seen=now,
            load_pct=conn.router.capacity.get("current_load_pct", 0),
        )
        
        self._own_observations[router_id] = obs
    
    def get_aggregated_health(self, router_id: str) -> float:
        """
        Get aggregated health score combining own and peer observations.
        
        Own observations are weighted higher (default 0.7 vs 0.3 for peers).
        
        Returns:
            Health score from 0.0 to 1.0
        """
        own_obs = self._own_observations.get(router_id)
        
        # Collect peer observations
        peer_obs_list = []
        now = time.time()
        
        for peer_id, peer_data in self._peer_observations.items():
            obs = peer_data.get(router_id)
            if obs:
                obs_age = now - obs.last_seen if hasattr(obs, 'last_seen') else float('inf')
                if obs_age <= self.config.observation_max_age:
                    peer_obs_list.append(obs)
        
        # Calculate scores
        if own_obs:
            own_score = self._calculate_observation_score(own_obs)
        else:
            own_score = None
        
        if peer_obs_list:
            peer_scores = [self._calculate_observation_score(obs) for obs in peer_obs_list]
            peer_score = sum(peer_scores) / len(peer_scores)
        else:
            peer_score = None
        
        # Combine with weights
        if own_score is not None and peer_score is not None:
            return (own_score * self.config.own_observation_weight + 
                    peer_score * self.config.peer_observation_weight)
        elif own_score is not None:
            return own_score
        elif peer_score is not None:
            return peer_score * 0.8  # Penalty for no direct observation
        else:
            return 0.5  # No data
    
    def _calculate_observation_score(self, obs: "RouterHealthObservation") -> float:
        """Calculate health score from a single observation."""
        success_score = obs.success_rate * 0.5
        latency_score = max(0, 1.0 - (obs.latency_ms / 500)) * 0.3
        load_score = (1.0 - (obs.load_pct / 100)) * 0.2
        return success_score + latency_score + load_score
    
    # -------------------------------------------------------------------------
    # HEALTH GOSSIP
    # -------------------------------------------------------------------------
    
    def create_gossip_message(self) -> "HealthGossip":
        """Create a health gossip message with our observations."""
        from .messages import HealthGossip
        
        observations = self._sample_observations_for_gossip()
        
        return HealthGossip(
            source_node_id=self.node_id,
            timestamp=time.time(),
            observations=observations,
            ttl=self.config.gossip_ttl,
        )
    
    def _sample_observations_for_gossip(self) -> List["RouterHealthObservation"]:
        """Sample observations to include in gossip message."""
        now = time.time()
        
        recent_obs = []
        for router_id, obs in self._own_observations.items():
            obs_age = now - obs.last_seen if obs.last_seen > 0 else float('inf')
            if obs_age <= self.config.observation_max_age:
                recent_obs.append((obs_age, obs))
        
        recent_obs.sort(key=lambda x: x[0])
        return [obs for _, obs in recent_obs[:self.config.max_observations_per_gossip]]
    
    def handle_gossip(self, gossip: "HealthGossip") -> None:
        """Handle incoming health gossip from a peer."""
        source = gossip.source_node_id
        
        if source == self.node_id:
            return
        
        if gossip.ttl <= 0:
            return
        
        self._stats["gossip_received"] += 1
        
        if source not in self._peer_observations:
            self._peer_observations[source] = {}
        
        for obs in gossip.observations:
            self._peer_observations[source][obs.router_id] = obs
        
        self._prune_peer_observations()
        
        logger.debug(
            f"Received gossip from {source[:16]}... with "
            f"{len(gossip.observations)} observations"
        )
    
    def _prune_peer_observations(self) -> None:
        """Prune old or excess peer observations."""
        now = time.time()
        total_obs = 0
        
        # Remove old observations
        for peer_id in list(self._peer_observations.keys()):
            peer_data = self._peer_observations[peer_id]
            for router_id in list(peer_data.keys()):
                obs = peer_data[router_id]
                obs_age = now - obs.last_seen if obs.last_seen > 0 else float('inf')
                if obs_age > self.config.observation_max_age:
                    del peer_data[router_id]
            
            if not peer_data:
                del self._peer_observations[peer_id]
            else:
                total_obs += len(peer_data)
        
        # Limit total observations
        if total_obs > self.config.max_peer_observations:
            all_obs = []
            for peer_id, peer_data in self._peer_observations.items():
                for router_id, obs in peer_data.items():
                    all_obs.append((peer_id, router_id, obs.last_seen))
            
            all_obs.sort(key=lambda x: x[2])
            
            to_remove = len(all_obs) - self.config.max_peer_observations
            for i in range(to_remove):
                peer_id, router_id, _ = all_obs[i]
                if peer_id in self._peer_observations:
                    self._peer_observations[peer_id].pop(router_id, None)
    
    def get_health_observations(self) -> Dict[str, Any]:
        """Get current health observation state."""
        aggregated_scores = {}
        for router_id in self._own_observations.keys():
            aggregated_scores[router_id] = round(self.get_aggregated_health(router_id), 3)
        
        return {
            "own_observations": {
                router_id: obs.to_dict()
                for router_id, obs in self._own_observations.items()
            },
            "peer_observation_count": sum(
                len(peer_data) for peer_data in self._peer_observations.values()
            ),
            "peers_with_observations": len(self._peer_observations),
            "aggregated_health_scores": aggregated_scores,
        }
    
    # -------------------------------------------------------------------------
    # MISBEHAVIOR DETECTION
    # -------------------------------------------------------------------------
    
    def _get_router_metrics(self, router_id: str) -> "RouterBehaviorMetrics":
        """Get or create behavior metrics for a router."""
        from .messages import RouterBehaviorMetrics
        
        if router_id not in self._router_behavior_metrics:
            self._router_behavior_metrics[router_id] = RouterBehaviorMetrics(
                router_id=router_id,
                first_seen=time.time(),
            )
        return self._router_behavior_metrics[router_id]
    
    def record_delivery_outcome(
        self,
        router_id: str,
        message_id: str,
        delivered: bool,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Record the outcome of a message delivery attempt."""
        if not self.config.misbehavior_detection_enabled:
            return
        
        metrics = self._get_router_metrics(router_id)
        metrics.record_delivery(delivered)
        
        if delivered and latency_ms is not None:
            metrics.record_latency(latency_ms)
        
        self._check_router_behavior(router_id)
    
    def record_ack_outcome(self, router_id: str, success: bool) -> None:
        """Record an ACK success or failure for a router."""
        if not self.config.misbehavior_detection_enabled:
            return
        
        metrics = self._get_router_metrics(router_id)
        metrics.record_ack(success)
        
        if success:
            self._stats["misbehavior_ack_success"] += 1
        else:
            self._stats["misbehavior_ack_failure"] += 1
        
        self._check_router_behavior(router_id)
    
    def _update_network_baseline(self) -> None:
        """Update the network baseline from all router metrics."""
        from .messages import NetworkBaseline
        
        if not self._router_behavior_metrics:
            return
        
        delivery_rates = []
        latencies = []
        ack_rates = []
        
        for router_id, metrics in self._router_behavior_metrics.items():
            if metrics.flagged:
                continue
            
            if metrics.messages_sent >= self.config.min_messages_for_detection:
                delivery_rates.append(metrics.delivery_rate)
                ack_rates.append(metrics.ack_success_rate)
                
                if metrics.latency_samples > 0:
                    latencies.append(metrics.avg_latency_ms)
        
        if len(delivery_rates) < self.config.baseline_min_samples:
            if self._network_baseline is None:
                self._network_baseline = NetworkBaseline()
            return
        
        avg_delivery = sum(delivery_rates) / len(delivery_rates)
        avg_ack = sum(ack_rates) / len(ack_rates)
        
        if len(delivery_rates) > 1:
            variance = sum((r - avg_delivery) ** 2 for r in delivery_rates) / len(delivery_rates)
            delivery_stddev = variance ** 0.5
        else:
            delivery_stddev = 0.05
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            if len(latencies) > 1:
                lat_variance = sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)
                latency_stddev = lat_variance ** 0.5
            else:
                latency_stddev = 50.0
        else:
            avg_latency = 100.0
            latency_stddev = 50.0
        
        self._network_baseline = NetworkBaseline(
            avg_delivery_rate=avg_delivery,
            avg_latency_ms=avg_latency,
            avg_ack_success_rate=avg_ack,
            sample_count=len(delivery_rates),
            last_updated=time.time(),
            delivery_rate_stddev=max(delivery_stddev, 0.01),
            latency_stddev_ms=max(latency_stddev, 10.0),
        )
    
    def _check_router_behavior(self, router_id: str) -> Optional["MisbehaviorReport"]:
        """Check if a router's behavior is anomalous."""
        from .messages import MisbehaviorType, MisbehaviorEvidence, MisbehaviorReport
        
        if not self.config.misbehavior_detection_enabled:
            return None
        
        metrics = self._get_router_metrics(router_id)
        
        if metrics.messages_sent < self.config.min_messages_for_detection:
            return None
        
        if metrics.flagged:
            return None
        
        baseline = self._network_baseline
        if baseline is None:
            self._update_network_baseline()
            baseline = self._network_baseline
        
        if baseline is None:
            from .messages import NetworkBaseline
            baseline = NetworkBaseline()
        
        evidence_list: List[MisbehaviorEvidence] = []
        anomaly_score = 0.0
        misbehavior_type = ""
        
        # Check delivery rate
        delivery_rate = metrics.delivery_rate
        if delivery_rate < self.config.delivery_rate_threshold:
            if baseline.is_delivery_rate_anomalous(
                delivery_rate, self.config.latency_threshold_stddevs
            ):
                evidence_list.append(MisbehaviorEvidence(
                    misbehavior_type=MisbehaviorType.MESSAGE_DROP,
                    delivery_rate_baseline=baseline.avg_delivery_rate,
                    delivery_rate_observed=delivery_rate,
                    description=f"Delivery rate {delivery_rate:.1%} below threshold"
                ))
                anomaly_score += 0.4
                misbehavior_type = MisbehaviorType.MESSAGE_DROP
        
        # Check ACK failure rate
        ack_failure_rate = 1.0 - metrics.ack_success_rate
        if ack_failure_rate > self.config.ack_failure_threshold:
            evidence_list.append(MisbehaviorEvidence(
                misbehavior_type=MisbehaviorType.ACK_FAILURE,
                description=f"ACK failure rate {ack_failure_rate:.1%} exceeds threshold"
            ))
            anomaly_score += 0.3
            if not misbehavior_type:
                misbehavior_type = MisbehaviorType.ACK_FAILURE
        
        # Check latency
        if metrics.latency_samples > 0 and metrics.avg_latency_ms > 0:
            if baseline.is_latency_anomalous(
                metrics.avg_latency_ms, self.config.latency_threshold_stddevs
            ):
                evidence_list.append(MisbehaviorEvidence(
                    misbehavior_type=MisbehaviorType.MESSAGE_DELAY,
                    expected_latency_ms=baseline.avg_latency_ms,
                    actual_latency_ms=metrics.avg_latency_ms,
                    description=f"Latency {metrics.avg_latency_ms:.1f}ms exceeds baseline"
                ))
                anomaly_score += 0.3
                if not misbehavior_type:
                    misbehavior_type = MisbehaviorType.MESSAGE_DELAY
        
        if not evidence_list:
            return None
        
        severity = min(1.0, anomaly_score)
        metrics.anomaly_score = severity
        
        if severity < self.config.mild_severity_threshold:
            return None
        
        # Flag the router
        metrics.flagged = True
        metrics.flag_reason = misbehavior_type
        
        report = MisbehaviorReport(
            reporter_id=self.node_id,
            router_id=router_id,
            misbehavior_type=misbehavior_type,
            evidence=evidence_list[:self.config.max_evidence_per_report],
            metrics=metrics,
            severity=severity,
        )
        
        self._flagged_routers[router_id] = report
        self._stats["routers_flagged"] += 1
        
        logger.warning(
            f"FLAGGED ROUTER {router_id[:16]}... for {misbehavior_type}: "
            f"severity={severity:.2f}"
        )
        
        if self.on_router_flagged:
            try:
                self.on_router_flagged(router_id, report)
            except Exception as e:
                logger.warning(f"on_router_flagged callback error: {e}")
        
        return report
    
    def is_router_flagged(self, router_id: str) -> bool:
        """Check if a router has been flagged for misbehavior."""
        metrics = self._router_behavior_metrics.get(router_id)
        if metrics and metrics.flagged:
            return True
        return router_id in self._flagged_routers
    
    def get_flagged_routers(self) -> Dict[str, Dict[str, Any]]:
        """Get all flagged routers with their reports."""
        result = {}
        for router_id, report in self._flagged_routers.items():
            result[router_id] = {
                "misbehavior_type": report.misbehavior_type,
                "severity": report.severity,
                "flagged_at": report.timestamp,
                "evidence_count": len(report.evidence),
            }
        return result
    
    def clear_router_flag(self, router_id: str) -> bool:
        """Clear the misbehavior flag for a router."""
        cleared = False
        
        if router_id in self._flagged_routers:
            del self._flagged_routers[router_id]
            cleared = True
        
        metrics = self._router_behavior_metrics.get(router_id)
        if metrics and metrics.flagged:
            metrics.flagged = False
            metrics.flag_reason = ""
            metrics.anomaly_score = 0.0
            cleared = True
        
        if cleared:
            logger.info(f"Cleared misbehavior flag for router {router_id[:16]}...")
        
        return cleared
    
    def get_misbehavior_detection_stats(self) -> Dict[str, Any]:
        """Get statistics for misbehavior detection."""
        return {
            "enabled": self.config.misbehavior_detection_enabled,
            "routers_tracked": len(self._router_behavior_metrics),
            "routers_flagged": len(self._flagged_routers),
            "reports_sent": self._stats.get("misbehavior_reports_sent", 0),
            "ack_success_total": self._stats.get("misbehavior_ack_success", 0),
            "ack_failure_total": self._stats.get("misbehavior_ack_failure", 0),
            "thresholds": {
                "delivery_rate": self.config.delivery_rate_threshold,
                "ack_failure": self.config.ack_failure_threshold,
                "latency_stddevs": self.config.latency_threshold_stddevs,
                "min_messages": self.config.min_messages_for_detection,
            },
            "flagged_routers": list(self._flagged_routers.keys()),
        }
    
    # -------------------------------------------------------------------------
    # ECLIPSE ANOMALY DETECTION
    # -------------------------------------------------------------------------
    
    def record_failure_event(
        self,
        router_id: str,
        failure_type: str,
        error_code: Optional[str] = None,
    ) -> None:
        """Record a router failure event for anomaly detection."""
        if not self.config.anomaly_detection_enabled:
            return
        
        now = time.time()
        
        event = {
            "router_id": router_id,
            "failure_type": failure_type,
            "error_code": error_code,
            "timestamp": now,
        }
        
        self._failure_events.append(event)
        
        # Prune old events
        cutoff = now - self.config.anomaly_window
        self._failure_events = [
            e for e in self._failure_events
            if e["timestamp"] >= cutoff
        ]
        
        self._detect_anomalies()
    
    def _detect_anomalies(self) -> Optional[Dict[str, Any]]:
        """Detect anomalous patterns in router failures."""
        now = time.time()
        cutoff = now - self.config.anomaly_window
        
        recent_failures = [
            e for e in self._failure_events
            if e["timestamp"] >= cutoff
        ]
        
        if len(recent_failures) < self.config.anomaly_threshold:
            return None
        
        # Check for same failure type across multiple routers
        failure_type_counts: Dict[str, List[str]] = {}
        for event in recent_failures:
            ft = event["failure_type"]
            if ft not in failure_type_counts:
                failure_type_counts[ft] = []
            failure_type_counts[ft].append(event["router_id"])
        
        for failure_type, router_ids in failure_type_counts.items():
            unique_routers = set(router_ids)
            if len(unique_routers) >= self.config.anomaly_threshold:
                anomaly = {
                    "type": "correlated_failures",
                    "failure_type": failure_type,
                    "affected_routers": list(unique_routers),
                    "count": len(unique_routers),
                    "window_seconds": self.config.anomaly_window,
                    "detected_at": now,
                }
                
                self._anomaly_alerts.append(anomaly)
                self._stats["anomalies_detected"] += 1
                
                # Keep only recent anomalies
                one_hour_ago = now - 3600
                self._anomaly_alerts = [
                    a for a in self._anomaly_alerts
                    if a["detected_at"] >= one_hour_ago
                ]
                
                logger.warning(
                    f"ECLIPSE ANOMALY DETECTED: {len(unique_routers)} routers "
                    f"experienced '{failure_type}' within {self.config.anomaly_window}s"
                )
                
                return anomaly
        
        return None
    
    def get_anomaly_alerts(self) -> List[Dict[str, Any]]:
        """Get recent anomaly alerts."""
        return list(self._anomaly_alerts)
    
    def clear_anomaly_alerts(self) -> int:
        """Clear anomaly alerts."""
        count = len(self._anomaly_alerts)
        self._anomaly_alerts.clear()
        return count
    
    # -------------------------------------------------------------------------
    # KEEPALIVE
    # -------------------------------------------------------------------------
    
    def track_ping_response(self, router_id: str, success: bool) -> bool:
        """
        Track ping response for a router.
        
        Returns:
            True if router should be considered failed (too many missed pings)
        """
        if success:
            self._missed_pings[router_id] = 0
            return False
        
        missed = self._missed_pings.get(router_id, 0) + 1
        self._missed_pings[router_id] = missed
        
        logger.warning(
            f"Ping timeout for router {router_id[:16]}... "
            f"({missed}/{self.config.missed_pings_threshold} missed)"
        )
        
        return missed >= self.config.missed_pings_threshold
    
    def clear_ping_state(self, router_id: str) -> None:
        """Clear ping state for a router."""
        self._missed_pings.pop(router_id, None)
