"""Trust Computation and Management for Valence Federation.

This module implements:
- TrustManager class for computing and updating trust
- Trust signal processing (corroboration, disputes, etc.)
- Trust phase transitions (observer → contributor → participant)
- Trust decay over time
- Effective trust calculation with user overrides

Refactored in Issue #31 to delegate to specialized components:
- TrustRegistry: Basic CRUD operations
- ThreatDetector: Signal analysis and threat assessment
- TrustPolicy: Phase transitions and decay
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
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
from .trust_registry import TrustRegistry
from .threat_detector import ThreatDetector, THREAT_THRESHOLDS
from .trust_policy import (
    TrustPolicy,
    DECAY_HALF_LIFE_DAYS,
    DECAY_MIN_THRESHOLD,
    PHASE_TRANSITION,
    PREFERENCE_MULTIPLIERS,
    CONCENTRATION_THRESHOLDS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS (re-exported for backward compatibility)
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
    
    Internally delegates to:
    - TrustRegistry: CRUD operations
    - ThreatDetector: Threat analysis
    - TrustPolicy: Phase transitions and decay
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
        
        # Initialize delegate components
        self._registry = TrustRegistry()
        self._threat_detector = ThreatDetector(self._registry)
        self._policy = TrustPolicy(
            self._registry,
            decay_half_life_days=decay_half_life_days,
            decay_min_threshold=decay_min_threshold,
        )

    # -------------------------------------------------------------------------
    # TRUST RETRIEVAL (delegated to registry)
    # -------------------------------------------------------------------------

    def get_node_trust(self, node_id: UUID) -> NodeTrust | None:
        """Get trust record for a node."""
        return self._registry.get_node_trust(node_id)

    def get_user_trust_preference(self, node_id: UUID) -> UserNodeTrust | None:
        """Get user's trust preference for a node."""
        return self._registry.get_user_trust_preference(node_id)

    def get_effective_trust(
        self,
        node_id: UUID,
        domain: str | None = None,
        apply_decay: bool = True,
    ) -> float:
        """Get effective trust score for a node."""
        return self._policy.get_effective_trust(node_id, domain, apply_decay)

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
        node_trust = self._registry.get_node_trust(signal.node_id)
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
        return self._registry.save_node_trust(node_trust)

    def process_corroboration(
        self,
        node_id: UUID,
        belief_id: UUID | None = None,
        domain: str | None = None,
    ) -> NodeTrust | None:
        """Process a corroboration signal."""
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
        """Process a dispute signal."""
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
        """Process an endorsement from another node."""
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
            node_trust = self._registry.save_node_trust(node_trust)

        return node_trust

    # -------------------------------------------------------------------------
    # TRUST DECAY (delegated to policy)
    # -------------------------------------------------------------------------

    def _apply_decay(self, trust: float, last_interaction: datetime) -> float:
        """Apply time-based decay to trust."""
        return self._policy._apply_decay(trust, last_interaction)

    def apply_decay_to_all_nodes(self) -> int:
        """Apply trust decay to all nodes that haven't interacted recently."""
        return self._policy.apply_decay_to_all_nodes()

    # -------------------------------------------------------------------------
    # PHASE TRANSITIONS (delegated to policy)
    # -------------------------------------------------------------------------

    def check_phase_transition(self, node_id: UUID) -> TrustPhase | None:
        """Check if a node qualifies for a phase transition."""
        return self._policy.check_phase_transition(node_id)

    def transition_phase(
        self,
        node_id: UUID,
        new_phase: TrustPhase,
        reason: str | None = None,
    ) -> bool:
        """Transition a node to a new trust phase."""
        return self._policy.transition_phase(node_id, new_phase, reason)

    def check_and_apply_transitions(self) -> list[tuple[UUID, TrustPhase, TrustPhase]]:
        """Check all nodes for phase transitions and apply them."""
        return self._policy.check_and_apply_transitions()

    # -------------------------------------------------------------------------
    # TRUST CONCENTRATION (delegated to policy)
    # -------------------------------------------------------------------------

    def check_trust_concentration(
        self,
        thresholds: dict[str, float] | None = None,
    ):
        """Check for trust concentration issues in the network.
        
        Detects when trust is too concentrated in few nodes.
        
        Args:
            thresholds: Optional custom thresholds
            
        Returns:
            TrustConcentrationReport with warnings and metrics
        """
        return self._policy.check_trust_concentration(thresholds)

    # -------------------------------------------------------------------------
    # THREAT ASSESSMENT (delegated to threat detector)
    # -------------------------------------------------------------------------

    def assess_threat_level(self, node_id: UUID) -> tuple[ThreatLevel, dict[str, Any]]:
        """Assess the threat level of a node based on behavior signals."""
        return self._threat_detector.assess_threat_level(node_id)

    def apply_threat_response(
        self,
        node_id: UUID,
        threat_level: ThreatLevel,
        assessment: dict[str, Any],
    ) -> bool:
        """Apply graduated response based on threat level."""
        return self._threat_detector.apply_threat_response(node_id, threat_level, assessment)

    # -------------------------------------------------------------------------
    # USER TRUST PREFERENCES (delegated to registry)
    # -------------------------------------------------------------------------

    def set_user_preference(
        self,
        node_id: UUID,
        preference: TrustPreference,
        manual_score: float | None = None,
        reason: str | None = None,
        domain_overrides: dict[str, str] | None = None,
    ) -> UserNodeTrust | None:
        """Set user's trust preference for a node."""
        return self._registry.set_user_preference(
            node_id=node_id,
            preference=preference,
            manual_score=manual_score,
            reason=reason,
            domain_overrides=domain_overrides,
        )

    def block_node(self, node_id: UUID, reason: str | None = None) -> UserNodeTrust | None:
        """Block a node (user-level, not federation-level)."""
        return self.set_user_preference(
            node_id=node_id,
            preference=TrustPreference.BLOCKED,
            reason=reason or "Manually blocked by user",
        )

    def unblock_node(self, node_id: UUID) -> UserNodeTrust | None:
        """Unblock a previously blocked node."""
        return self.set_user_preference(
            node_id=node_id,
            preference=TrustPreference.AUTOMATIC,
            reason="Unblocked by user",
        )

    # -------------------------------------------------------------------------
    # BELIEF ANNOTATIONS (delegated to registry)
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
        """Add a trust annotation to a belief."""
        return self._registry.annotate_belief(
            belief_id=belief_id,
            annotation_type=annotation_type,
            source_node_id=source_node_id,
            confidence_delta=confidence_delta,
            attestation=attestation,
            expires_at=expires_at,
        )

    def get_belief_trust_adjustments(self, belief_id: UUID) -> float:
        """Get total trust adjustment for a belief from all annotations."""
        return self._registry.get_belief_trust_adjustments(belief_id)

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS (for backward compatibility)
    # -------------------------------------------------------------------------

    def _get_node(self, node_id: UUID) -> FederationNode | None:
        """Get a federation node by ID."""
        return self._registry.get_node(node_id)

    def _save_node_trust(self, node_trust: NodeTrust) -> NodeTrust | None:
        """Persist node trust changes to database."""
        return self._registry.save_node_trust(node_trust)


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
    """Get effective trust for a node (convenience function)."""
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
    """Assess threat level and apply appropriate response."""
    manager = get_trust_manager()
    level, assessment = manager.assess_threat_level(node_id)
    manager.apply_threat_response(node_id, level, assessment)
    return level, assessment


def check_trust_concentration(
    thresholds: dict[str, float] | None = None,
):
    """Check for trust concentration issues in the network (convenience function).
    
    Args:
        thresholds: Optional custom thresholds
        
    Returns:
        TrustConcentrationReport with warnings and metrics
    """
    return get_trust_manager().check_trust_concentration(thresholds)
