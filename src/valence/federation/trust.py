"""Trust Computation and Management for Valence Federation.

This module implements:
- TrustManager class for computing and updating trust
- Trust signal processing (corroboration, disputes, etc.)
- Trust phase transitions (observer → contributor → participant)
- Trust decay over time
- Effective trust calculation with user overrides
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from ..core.db import get_cursor
from .models import (
    FederationNode,
    NodeTrust,
    UserNodeTrust,
    TrustPhase,
    TrustPreference,
    ThreatLevel,
    AnnotationType,
    BeliefTrustAnnotation,
    TrustAttestation,
    TRUST_WEIGHTS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


# Trust signal impact weights
SIGNAL_WEIGHTS = {
    "corroboration": 0.02,      # Per corroborated belief
    "dispute": -0.05,           # Per disputed belief
    "endorsement": 0.10,        # Per endorsement received
    "sync_success": 0.005,      # Per successful sync
    "sync_failure": -0.01,      # Per failed sync
    "aggregation_participation": 0.01,  # Per aggregation contribution
}

# Trust decay parameters
DECAY_HALF_LIFE_DAYS = 30  # Trust decays by half over this period without interaction
DECAY_MIN_THRESHOLD = 0.1  # Minimum trust after decay

# Phase transition thresholds
PHASE_TRANSITION = {
    TrustPhase.OBSERVER: {
        "min_days": 7,
        "min_trust": 0.0,  # Observer is the starting phase
        "min_interactions": 0,
    },
    TrustPhase.CONTRIBUTOR: {
        "min_days": 7,
        "min_trust": 0.15,
        "min_interactions": 5,
    },
    TrustPhase.PARTICIPANT: {
        "min_days": 30,
        "min_trust": 0.4,
        "min_interactions": 20,
    },
    TrustPhase.ANCHOR: {
        "min_days": 90,
        "min_trust": 0.8,
        "min_interactions": 100,
        "min_endorsements": 3,
    },
}

# Threat level thresholds
THREAT_THRESHOLDS = {
    ThreatLevel.NONE: 0.0,
    ThreatLevel.LOW: 0.2,
    ThreatLevel.MEDIUM: 0.4,
    ThreatLevel.HIGH: 0.6,
    ThreatLevel.CRITICAL: 0.8,
}

# User preference multipliers
PREFERENCE_MULTIPLIERS = {
    TrustPreference.BLOCKED: 0.0,
    TrustPreference.REDUCED: 0.5,
    TrustPreference.AUTOMATIC: 1.0,
    TrustPreference.ELEVATED: 1.2,
    TrustPreference.ANCHOR: 1.5,
}


# =============================================================================
# TRUST SIGNAL
# =============================================================================


@dataclass
class TrustSignal:
    """A signal that affects node trust."""

    node_id: UUID
    signal_type: str  # corroboration, dispute, endorsement, etc.
    value: float = 1.0  # Magnitude (positive or negative based on type)
    domain: str | None = None  # If domain-specific
    source_node_id: UUID | None = None  # If from another node
    belief_id: UUID | None = None  # If related to a specific belief
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


# =============================================================================
# TRUST MANAGER
# =============================================================================


class TrustManager:
    """Manages trust computation and updates for federation nodes.

    This class is responsible for:
    - Processing trust signals (corroborations, disputes, endorsements)
    - Computing effective trust with user overrides
    - Managing trust phase transitions
    - Applying trust decay over time
    - Handling threat assessment
    """

    def __init__(
        self,
        decay_half_life_days: int = DECAY_HALF_LIFE_DAYS,
        decay_min_threshold: float = DECAY_MIN_THRESHOLD,
    ):
        """Initialize the TrustManager.

        Args:
            decay_half_life_days: Days for trust to decay by half
            decay_min_threshold: Minimum trust after decay
        """
        self.decay_half_life_days = decay_half_life_days
        self.decay_min_threshold = decay_min_threshold

    # -------------------------------------------------------------------------
    # TRUST RETRIEVAL
    # -------------------------------------------------------------------------

    def get_node_trust(self, node_id: UUID) -> NodeTrust | None:
        """Get trust record for a node.

        Args:
            node_id: The node's UUID

        Returns:
            NodeTrust if found, None otherwise
        """
        try:
            with get_cursor() as cur:
                cur.execute("SELECT * FROM node_trust WHERE node_id = %s", (node_id,))
                row = cur.fetchone()
                if row:
                    return NodeTrust.from_row(row)
                return None
        except Exception as e:
            logger.warning(f"Error getting trust for node {node_id}: {e}")
            return None

    def get_user_trust_preference(self, node_id: UUID) -> UserNodeTrust | None:
        """Get user's trust preference for a node.

        Args:
            node_id: The node's UUID

        Returns:
            UserNodeTrust if found, None otherwise
        """
        try:
            with get_cursor() as cur:
                cur.execute("SELECT * FROM user_node_trust WHERE node_id = %s", (node_id,))
                row = cur.fetchone()
                if row:
                    return UserNodeTrust.from_row(row)
                return None
        except Exception as e:
            logger.warning(f"Error getting user trust preference for node {node_id}: {e}")
            return None

    def get_effective_trust(
        self,
        node_id: UUID,
        domain: str | None = None,
        apply_decay: bool = True,
    ) -> float:
        """Get effective trust score for a node.

        Combines computed trust with user overrides and decay.

        Args:
            node_id: The node's UUID
            domain: Optional domain for domain-specific trust
            apply_decay: Whether to apply time-based decay

        Returns:
            Effective trust score (0.0 to 1.0)
        """
        node_trust = self.get_node_trust(node_id)
        if not node_trust:
            return 0.1  # Default for unknown nodes

        # Get base trust (domain-specific if applicable)
        base_trust = node_trust.get_domain_trust(domain) if domain else node_trust.overall

        # Apply decay based on last interaction
        if apply_decay and node_trust.last_interaction_at:
            base_trust = self._apply_decay(base_trust, node_trust.last_interaction_at)

        # Get user preference
        user_pref = self.get_user_trust_preference(node_id)
        if user_pref:
            # Get preference (possibly domain-specific)
            pref = user_pref.get_effective_preference(domain)

            # Check for manual trust score override
            if user_pref.manual_trust_score is not None:
                return user_pref.manual_trust_score

            # Apply preference multiplier
            multiplier = PREFERENCE_MULTIPLIERS.get(pref, 1.0)
            base_trust *= multiplier

        return max(0.0, min(1.0, base_trust))

    # -------------------------------------------------------------------------
    # TRUST SIGNALS
    # -------------------------------------------------------------------------

    def process_signal(self, signal: TrustSignal) -> NodeTrust | None:
        """Process a trust signal and update node trust.

        Args:
            signal: The trust signal to process

        Returns:
            Updated NodeTrust if successful, None on error
        """
        node_trust = self.get_node_trust(signal.node_id)
        if not node_trust:
            logger.warning(f"No trust record for node {signal.node_id}")
            return None

        # Get signal weight
        weight = SIGNAL_WEIGHTS.get(signal.signal_type, 0.0) * signal.value

        # Update appropriate dimension
        if signal.signal_type == "corroboration":
            node_trust.beliefs_corroborated += 1
            if node_trust.belief_accuracy is not None:
                node_trust.belief_accuracy = min(1.0, node_trust.belief_accuracy + weight)
            else:
                node_trust.belief_accuracy = 0.5 + weight

        elif signal.signal_type == "dispute":
            node_trust.beliefs_disputed += 1
            if node_trust.belief_accuracy is not None:
                node_trust.belief_accuracy = max(0.0, node_trust.belief_accuracy + weight)
            else:
                node_trust.belief_accuracy = max(0.0, 0.5 + weight)

        elif signal.signal_type == "endorsement":
            node_trust.endorsements_received += 1
            if node_trust.endorsement_strength is not None:
                node_trust.endorsement_strength = min(1.0, node_trust.endorsement_strength + weight)
            else:
                node_trust.endorsement_strength = weight

        elif signal.signal_type in ("sync_success", "sync_failure"):
            node_trust.sync_requests_served += 1
            if node_trust.uptime_reliability is not None:
                node_trust.uptime_reliability = max(0.0, min(1.0, node_trust.uptime_reliability + weight))
            else:
                base = 0.7 if signal.signal_type == "sync_success" else 0.3
                node_trust.uptime_reliability = base

        elif signal.signal_type == "aggregation_participation":
            node_trust.aggregation_participations += 1
            if node_trust.contribution_consistency is not None:
                node_trust.contribution_consistency = min(1.0, node_trust.contribution_consistency + weight)
            else:
                node_trust.contribution_consistency = 0.5 + weight

        # Update domain-specific trust if applicable
        if signal.domain:
            current_domain_trust = node_trust.domain_expertise.get(signal.domain, 0.5)
            new_domain_trust = max(0.0, min(1.0, current_domain_trust + weight))
            node_trust.domain_expertise[signal.domain] = new_domain_trust

        # Update last interaction time
        node_trust.last_interaction_at = datetime.now()

        # Recalculate overall trust
        node_trust.recalculate_overall()

        # Persist changes
        return self._save_node_trust(node_trust)

    def process_corroboration(
        self,
        node_id: UUID,
        belief_id: UUID | None = None,
        domain: str | None = None,
    ) -> NodeTrust | None:
        """Process a corroboration signal.

        Args:
            node_id: The node that provided the corroborated belief
            belief_id: Optional related belief ID
            domain: Optional domain for domain-specific trust

        Returns:
            Updated NodeTrust if successful
        """
        signal = TrustSignal(
            node_id=node_id,
            signal_type="corroboration",
            belief_id=belief_id,
            domain=domain,
        )
        return self.process_signal(signal)

    def process_dispute(
        self,
        node_id: UUID,
        belief_id: UUID | None = None,
        domain: str | None = None,
        severity: float = 1.0,
    ) -> NodeTrust | None:
        """Process a dispute signal.

        Args:
            node_id: The node that provided the disputed belief
            belief_id: Optional related belief ID
            domain: Optional domain for domain-specific trust
            severity: Severity multiplier (1.0 = normal)

        Returns:
            Updated NodeTrust if successful
        """
        signal = TrustSignal(
            node_id=node_id,
            signal_type="dispute",
            value=severity,
            belief_id=belief_id,
            domain=domain,
        )
        return self.process_signal(signal)

    def process_endorsement(
        self,
        subject_node_id: UUID,
        endorser_node_id: UUID,
        attestation: TrustAttestation | None = None,
    ) -> NodeTrust | None:
        """Process an endorsement from another node.

        Args:
            subject_node_id: The node being endorsed
            endorser_node_id: The node giving the endorsement
            attestation: Optional attestation with specific dimensions

        Returns:
            Updated NodeTrust if successful
        """
        # Get endorser's trust to weight the endorsement
        endorser_trust = self.get_effective_trust(endorser_node_id)
        weight = endorser_trust  # Weight endorsement by endorser's trust

        signal = TrustSignal(
            node_id=subject_node_id,
            signal_type="endorsement",
            value=weight,
            source_node_id=endorser_node_id,
        )

        node_trust = self.process_signal(signal)

        # If attestation includes specific dimensions, apply those too
        if attestation and node_trust:
            for dim, value in attestation.attested_dimensions.items():
                if hasattr(node_trust, dim):
                    current = getattr(node_trust, dim) or 0.5
                    # Weighted average with endorser's trust as weight
                    new_value = current * (1 - weight) + value * weight
                    setattr(node_trust, dim, new_value)

            # Handle domain-specific endorsements
            if attestation.domains:
                for domain in attestation.domains:
                    current = node_trust.domain_expertise.get(domain, 0.5)
                    boost = SIGNAL_WEIGHTS["endorsement"] * weight
                    node_trust.domain_expertise[domain] = min(1.0, current + boost)

            node_trust.recalculate_overall()
            node_trust = self._save_node_trust(node_trust)

        return node_trust

    # -------------------------------------------------------------------------
    # TRUST DECAY
    # -------------------------------------------------------------------------

    def _apply_decay(
        self,
        trust: float,
        last_interaction: datetime,
    ) -> float:
        """Apply time-based decay to trust.

        Uses exponential decay with configurable half-life.

        Args:
            trust: Current trust value
            last_interaction: Timestamp of last interaction

        Returns:
            Decayed trust value
        """
        days_since_interaction = (datetime.now() - last_interaction).days
        if days_since_interaction <= 0:
            return trust

        # Exponential decay: trust * 0.5^(days / half_life)
        decay_factor = 0.5 ** (days_since_interaction / self.decay_half_life_days)
        decayed = trust * decay_factor

        # Don't decay below minimum threshold
        return max(self.decay_min_threshold, decayed)

    def apply_decay_to_all_nodes(self) -> int:
        """Apply trust decay to all nodes that haven't interacted recently.

        Returns:
            Number of nodes updated
        """
        count = 0
        try:
            with get_cursor() as cur:
                # Find nodes that haven't interacted recently
                threshold = datetime.now() - timedelta(days=7)
                cur.execute("""
                    SELECT nt.* FROM node_trust nt
                    WHERE nt.last_interaction_at < %s
                    OR nt.last_interaction_at IS NULL
                """, (threshold,))
                rows = cur.fetchall()

                for row in rows:
                    node_trust = NodeTrust.from_row(row)
                    if node_trust.last_interaction_at:
                        old_trust = node_trust.overall
                        decayed = self._apply_decay(old_trust, node_trust.last_interaction_at)
                        if decayed != old_trust:
                            node_trust.overall = decayed
                            self._save_node_trust(node_trust)
                            count += 1

        except Exception as e:
            logger.exception(f"Error applying trust decay: {e}")

        return count

    # -------------------------------------------------------------------------
    # PHASE TRANSITIONS
    # -------------------------------------------------------------------------

    def check_phase_transition(
        self,
        node_id: UUID,
    ) -> TrustPhase | None:
        """Check if a node qualifies for a phase transition.

        Args:
            node_id: The node's UUID

        Returns:
            New phase if transition is warranted, None otherwise
        """
        # Get node and trust
        node = self._get_node(node_id)
        if not node:
            return None

        node_trust = self.get_node_trust(node_id)
        if not node_trust:
            return None

        current_phase = node.trust_phase
        days_in_phase = (datetime.now() - node.phase_started_at).days

        # Determine total interactions
        total_interactions = (
            node_trust.beliefs_received +
            node_trust.sync_requests_served +
            node_trust.aggregation_participations
        )

        # Check for demotion first (trust fell too low)
        if current_phase != TrustPhase.OBSERVER:
            prev_phases = [TrustPhase.OBSERVER, TrustPhase.CONTRIBUTOR, TrustPhase.PARTICIPANT]
            current_idx = prev_phases.index(current_phase) if current_phase in prev_phases else len(prev_phases)

            for i in range(current_idx - 1, -1, -1):
                phase = prev_phases[i]
                req = PHASE_TRANSITION[prev_phases[i + 1] if i + 1 < len(prev_phases) else TrustPhase.ANCHOR]
                if node_trust.overall < req["min_trust"] * 0.8:  # 20% below threshold
                    return phase

        # Check for promotion
        next_phase_map = {
            TrustPhase.OBSERVER: TrustPhase.CONTRIBUTOR,
            TrustPhase.CONTRIBUTOR: TrustPhase.PARTICIPANT,
            TrustPhase.PARTICIPANT: TrustPhase.ANCHOR,
        }

        next_phase = next_phase_map.get(current_phase)
        if not next_phase:
            return None  # Already at ANCHOR

        req = PHASE_TRANSITION[next_phase]

        # Check requirements
        if days_in_phase < req["min_days"]:
            return None
        if node_trust.overall < req["min_trust"]:
            return None
        if total_interactions < req["min_interactions"]:
            return None
        if "min_endorsements" in req and node_trust.endorsements_received < req["min_endorsements"]:
            return None

        return next_phase

    def transition_phase(
        self,
        node_id: UUID,
        new_phase: TrustPhase,
        reason: str | None = None,
    ) -> bool:
        """Transition a node to a new trust phase.

        Args:
            node_id: The node's UUID
            new_phase: The new trust phase
            reason: Optional reason for transition

        Returns:
            True if successful
        """
        try:
            with get_cursor() as cur:
                cur.execute("""
                    UPDATE federation_nodes
                    SET trust_phase = %s,
                        phase_started_at = NOW(),
                        metadata = jsonb_set(
                            COALESCE(metadata, '{}'),
                            '{phase_transition_reason}',
                            %s::jsonb
                        )
                    WHERE id = %s
                """, (new_phase.value, f'"{reason}"' if reason else 'null', node_id))

                logger.info(f"Node {node_id} transitioned to phase {new_phase.value}: {reason}")
                return True

        except Exception as e:
            logger.exception(f"Error transitioning node {node_id} to phase {new_phase.value}")
            return False

    def check_and_apply_transitions(self) -> list[tuple[UUID, TrustPhase, TrustPhase]]:
        """Check all nodes for phase transitions and apply them.

        Returns:
            List of (node_id, old_phase, new_phase) for transitions that occurred
        """
        transitions = []

        try:
            with get_cursor() as cur:
                cur.execute("SELECT id, trust_phase FROM federation_nodes WHERE status != 'unreachable'")
                rows = cur.fetchall()

            for row in rows:
                node_id = row["id"]
                old_phase = TrustPhase(row["trust_phase"])

                new_phase = self.check_phase_transition(node_id)
                if new_phase and new_phase != old_phase:
                    direction = "promoted" if new_phase.value > old_phase.value else "demoted"
                    reason = f"Automatically {direction} based on trust metrics"

                    if self.transition_phase(node_id, new_phase, reason):
                        transitions.append((node_id, old_phase, new_phase))

        except Exception as e:
            logger.exception(f"Error checking phase transitions: {e}")

        return transitions

    # -------------------------------------------------------------------------
    # THREAT ASSESSMENT
    # -------------------------------------------------------------------------

    def assess_threat_level(
        self,
        node_id: UUID,
    ) -> tuple[ThreatLevel, dict[str, Any]]:
        """Assess the threat level of a node based on behavior signals.

        Args:
            node_id: The node's UUID

        Returns:
            Tuple of (threat level, assessment details)
        """
        node_trust = self.get_node_trust(node_id)
        if not node_trust:
            return ThreatLevel.NONE, {"reason": "No trust record"}

        assessment = {
            "signals": [],
            "threat_score": 0.0,
        }

        # Signal 1: High dispute ratio
        if node_trust.beliefs_received > 10:
            dispute_ratio = node_trust.beliefs_disputed / node_trust.beliefs_received
            if dispute_ratio > 0.3:
                assessment["signals"].append({
                    "type": "high_dispute_ratio",
                    "value": dispute_ratio,
                    "contribution": min(0.3, dispute_ratio),
                })
                assessment["threat_score"] += min(0.3, dispute_ratio)

        # Signal 2: Very low trust after significant interaction
        total_interactions = (
            node_trust.beliefs_received +
            node_trust.sync_requests_served +
            node_trust.aggregation_participations
        )
        if total_interactions > 20 and node_trust.overall < 0.2:
            assessment["signals"].append({
                "type": "persistently_low_trust",
                "value": node_trust.overall,
                "contribution": 0.2,
            })
            assessment["threat_score"] += 0.2

        # Signal 3: Trust declined rapidly
        # This would require historical tracking; for now use a heuristic
        if node_trust.beliefs_corroborated > 0:
            corroboration_ratio = node_trust.beliefs_corroborated / max(1, node_trust.beliefs_received)
            if corroboration_ratio < 0.1:
                assessment["signals"].append({
                    "type": "low_corroboration",
                    "value": corroboration_ratio,
                    "contribution": 0.15,
                })
                assessment["threat_score"] += 0.15

        # Signal 4: Rapid volume (potential spam/Sybil)
        if node_trust.beliefs_received > 100:
            days_active = max(1, (datetime.now() - node_trust.relationship_started_at).days)
            daily_rate = node_trust.beliefs_received / days_active
            if daily_rate > 50:  # More than 50 beliefs per day
                assessment["signals"].append({
                    "type": "high_volume",
                    "value": daily_rate,
                    "contribution": min(0.2, (daily_rate - 50) / 200),
                })
                assessment["threat_score"] += min(0.2, (daily_rate - 50) / 200)

        # Determine threat level from score
        threat_level = ThreatLevel.NONE
        for level, threshold in sorted(THREAT_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if assessment["threat_score"] >= threshold:
                threat_level = level
                break

        assessment["level"] = threat_level.value
        return threat_level, assessment

    def apply_threat_response(
        self,
        node_id: UUID,
        threat_level: ThreatLevel,
        assessment: dict[str, Any],
    ) -> bool:
        """Apply graduated response based on threat level.

        Per PRINCIPLES: "reduced access rather than exile"

        Args:
            node_id: The node's UUID
            threat_level: Assessed threat level
            assessment: Assessment details

        Returns:
            True if response was applied
        """
        node_trust = self.get_node_trust(node_id)
        if not node_trust:
            return False

        if threat_level == ThreatLevel.NONE:
            return True  # No action needed

        # Level 1: Low - Increased scrutiny (log, no penalty)
        if threat_level == ThreatLevel.LOW:
            logger.warning(
                f"Node {node_id} at LOW threat level: {assessment['signals']}"
            )
            return True

        # Level 2: Medium - Reduce trust
        if threat_level == ThreatLevel.MEDIUM:
            penalty = -0.1
            node_trust.manual_trust_adjustment += penalty
            node_trust.adjustment_reason = f"Automated penalty: {threat_level.value}"
            node_trust.recalculate_overall()
            self._save_node_trust(node_trust)
            logger.warning(
                f"Node {node_id} at MEDIUM threat level, applied trust penalty: {penalty}"
            )
            return True

        # Level 3: High - Quarantine from sensitive operations
        if threat_level == ThreatLevel.HIGH:
            penalty = -0.3
            node_trust.manual_trust_adjustment += penalty
            node_trust.adjustment_reason = f"Quarantine: {threat_level.value}"
            node_trust.recalculate_overall()
            self._save_node_trust(node_trust)

            # Mark in metadata for exclusion from consensus
            try:
                with get_cursor() as cur:
                    cur.execute("""
                        UPDATE federation_nodes
                        SET metadata = jsonb_set(
                            COALESCE(metadata, '{}'),
                            '{quarantine_until}',
                            to_jsonb((NOW() + INTERVAL '7 days')::TEXT)
                        )
                        WHERE id = %s
                    """, (node_id,))
            except Exception as e:
                logger.exception(f"Error setting quarantine: {e}")

            logger.warning(
                f"Node {node_id} at HIGH threat level, quarantined for 7 days"
            )
            return True

        # Level 4: Critical - Functional isolation (read-only)
        if threat_level == ThreatLevel.CRITICAL:
            penalty = -0.5
            node_trust.manual_trust_adjustment = max(-0.9, node_trust.manual_trust_adjustment + penalty)
            node_trust.adjustment_reason = f"Isolation: {threat_level.value}"
            node_trust.recalculate_overall()
            self._save_node_trust(node_trust)

            # Mark as read-only (can still receive, not contribute)
            try:
                with get_cursor() as cur:
                    cur.execute("""
                        UPDATE federation_nodes
                        SET metadata = jsonb_set(
                            jsonb_set(
                                COALESCE(metadata, '{}'),
                                '{read_only}',
                                'true'
                            ),
                            '{isolation_reason}',
                            %s::jsonb
                        ),
                        status = 'suspended'
                        WHERE id = %s
                    """, (f'"{assessment}"', node_id))
            except Exception as e:
                logger.exception(f"Error setting isolation: {e}")

            logger.error(
                f"Node {node_id} at CRITICAL threat level, isolated"
            )
            return True

        return False

    # -------------------------------------------------------------------------
    # USER TRUST PREFERENCES
    # -------------------------------------------------------------------------

    def set_user_preference(
        self,
        node_id: UUID,
        preference: TrustPreference,
        manual_score: float | None = None,
        reason: str | None = None,
        domain_overrides: dict[str, str] | None = None,
    ) -> UserNodeTrust | None:
        """Set user's trust preference for a node.

        Args:
            node_id: The node's UUID
            preference: Trust preference level
            manual_score: Optional manual trust score override
            reason: Reason for the preference
            domain_overrides: Domain-specific preference overrides

        Returns:
            UserNodeTrust if successful
        """
        try:
            with get_cursor() as cur:
                cur.execute("""
                    INSERT INTO user_node_trust (node_id, trust_preference, manual_trust_score, reason, domain_overrides)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (node_id) DO UPDATE SET
                        trust_preference = EXCLUDED.trust_preference,
                        manual_trust_score = EXCLUDED.manual_trust_score,
                        reason = EXCLUDED.reason,
                        domain_overrides = COALESCE(user_node_trust.domain_overrides, '{}'::jsonb) || COALESCE(EXCLUDED.domain_overrides, '{}'::jsonb),
                        modified_at = NOW()
                    RETURNING *
                """, (
                    node_id,
                    preference.value,
                    manual_score,
                    reason,
                    domain_overrides or {},
                ))
                row = cur.fetchone()
                if row:
                    return UserNodeTrust.from_row(row)
                return None

        except Exception as e:
            logger.exception(f"Error setting user preference for node {node_id}")
            return None

    def block_node(
        self,
        node_id: UUID,
        reason: str | None = None,
    ) -> UserNodeTrust | None:
        """Block a node (user-level, not federation-level).

        Args:
            node_id: The node's UUID
            reason: Reason for blocking

        Returns:
            UserNodeTrust if successful
        """
        return self.set_user_preference(
            node_id=node_id,
            preference=TrustPreference.BLOCKED,
            reason=reason or "Manually blocked by user",
        )

    def unblock_node(
        self,
        node_id: UUID,
    ) -> UserNodeTrust | None:
        """Unblock a previously blocked node.

        Args:
            node_id: The node's UUID

        Returns:
            UserNodeTrust if successful
        """
        return self.set_user_preference(
            node_id=node_id,
            preference=TrustPreference.AUTOMATIC,
            reason="Unblocked by user",
        )

    # -------------------------------------------------------------------------
    # BELIEF ANNOTATIONS
    # -------------------------------------------------------------------------

    def annotate_belief(
        self,
        belief_id: UUID,
        annotation_type: AnnotationType,
        source_node_id: UUID | None = None,
        confidence_delta: float = 0.0,
        attestation: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
    ) -> BeliefTrustAnnotation | None:
        """Add a trust annotation to a belief.

        Args:
            belief_id: The belief's UUID
            annotation_type: Type of annotation
            source_node_id: Source node for the annotation
            confidence_delta: Change to apply to belief confidence
            attestation: Corroboration attestation data
            expires_at: When this annotation expires

        Returns:
            BeliefTrustAnnotation if successful
        """
        try:
            with get_cursor() as cur:
                cur.execute("""
                    INSERT INTO belief_trust_annotations
                    (belief_id, type, source_node_id, confidence_delta, corroboration_attestation, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING *
                """, (
                    belief_id,
                    annotation_type.value,
                    source_node_id,
                    confidence_delta,
                    attestation,
                    expires_at,
                ))
                row = cur.fetchone()
                if row:
                    return BeliefTrustAnnotation.from_row(row)
                return None

        except Exception as e:
            logger.exception(f"Error annotating belief {belief_id}")
            return None

    def get_belief_trust_adjustments(
        self,
        belief_id: UUID,
    ) -> float:
        """Get total trust adjustment for a belief from all annotations.

        Args:
            belief_id: The belief's UUID

        Returns:
            Total confidence delta from annotations
        """
        try:
            with get_cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(SUM(confidence_delta), 0) as total_delta
                    FROM belief_trust_annotations
                    WHERE belief_id = %s
                    AND (expires_at IS NULL OR expires_at > NOW())
                """, (belief_id,))
                row = cur.fetchone()
                return float(row["total_delta"]) if row else 0.0

        except Exception as e:
            logger.warning(f"Error getting belief trust adjustments: {e}")
            return 0.0

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _get_node(self, node_id: UUID) -> FederationNode | None:
        """Get a federation node by ID."""
        try:
            with get_cursor() as cur:
                cur.execute("SELECT * FROM federation_nodes WHERE id = %s", (node_id,))
                row = cur.fetchone()
                if row:
                    return FederationNode.from_row(row)
                return None
        except Exception as e:
            logger.warning(f"Error getting node {node_id}: {e}")
            return None

    def _save_node_trust(self, node_trust: NodeTrust) -> NodeTrust | None:
        """Persist node trust changes to database.

        Args:
            node_trust: The NodeTrust to save

        Returns:
            Updated NodeTrust if successful
        """
        try:
            with get_cursor() as cur:
                cur.execute("""
                    UPDATE node_trust SET
                        trust = %s,
                        beliefs_received = %s,
                        beliefs_corroborated = %s,
                        beliefs_disputed = %s,
                        sync_requests_served = %s,
                        aggregation_participations = %s,
                        endorsements_received = %s,
                        endorsements_given = %s,
                        last_interaction_at = %s,
                        manual_trust_adjustment = %s,
                        adjustment_reason = %s,
                        modified_at = NOW()
                    WHERE node_id = %s
                    RETURNING *
                """, (
                    node_trust.to_trust_dict(),
                    node_trust.beliefs_received,
                    node_trust.beliefs_corroborated,
                    node_trust.beliefs_disputed,
                    node_trust.sync_requests_served,
                    node_trust.aggregation_participations,
                    node_trust.endorsements_received,
                    node_trust.endorsements_given,
                    node_trust.last_interaction_at,
                    node_trust.manual_trust_adjustment,
                    node_trust.adjustment_reason,
                    node_trust.node_id,
                ))
                row = cur.fetchone()
                if row:
                    return NodeTrust.from_row(row)
                return None

        except Exception as e:
            logger.exception(f"Error saving node trust for {node_trust.node_id}")
            return None


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================


# Default manager instance
_default_manager: TrustManager | None = None


def get_trust_manager() -> TrustManager:
    """Get the default TrustManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = TrustManager()
    return _default_manager


def get_effective_trust(
    node_id: UUID,
    domain: str | None = None,
) -> float:
    """Get effective trust for a node (convenience function).

    Args:
        node_id: The node's UUID
        domain: Optional domain for domain-specific trust

    Returns:
        Effective trust score
    """
    return get_trust_manager().get_effective_trust(node_id, domain)


def process_corroboration(
    node_id: UUID,
    belief_id: UUID | None = None,
    domain: str | None = None,
) -> NodeTrust | None:
    """Process a corroboration signal (convenience function)."""
    return get_trust_manager().process_corroboration(node_id, belief_id, domain)


def process_dispute(
    node_id: UUID,
    belief_id: UUID | None = None,
    domain: str | None = None,
    severity: float = 1.0,
) -> NodeTrust | None:
    """Process a dispute signal (convenience function)."""
    return get_trust_manager().process_dispute(node_id, belief_id, domain, severity)


def assess_and_respond_to_threat(
    node_id: UUID,
) -> tuple[ThreatLevel, dict[str, Any]]:
    """Assess threat level and apply appropriate response.

    Args:
        node_id: The node's UUID

    Returns:
        Tuple of (threat level, assessment details)
    """
    manager = get_trust_manager()
    level, assessment = manager.assess_threat_level(node_id)
    manager.apply_threat_response(node_id, level, assessment)
    return level, assessment
