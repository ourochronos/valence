"""Trust Registry - Basic CRUD operations for trust data.

Part of the TrustManager refactor (Issue #31). This module handles:
- Getting/setting node trust records
- User trust preferences
- Belief annotations
- Persistence to database
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg2

from ..core.db import get_cursor
from .models import (
    FederationNode,
    NodeTrust,
    UserNodeTrust,
    TrustPreference,
    AnnotationType,
    BeliefTrustAnnotation,
)

logger = logging.getLogger(__name__)


class TrustRegistry:
    """Basic CRUD operations for trust data.
    
    Responsible for:
    - Retrieving node trust records
    - Persisting trust updates
    - Managing user trust preferences
    - Handling belief annotations
    """

    # -------------------------------------------------------------------------
    # NODE TRUST RETRIEVAL
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
        except psycopg2.Error as e:
            logger.warning(f"Database error getting trust for node {node_id}: {e}")
            return None

    def get_node(self, node_id: UUID) -> FederationNode | None:
        """Get a federation node by ID."""
        try:
            with get_cursor() as cur:
                cur.execute("SELECT * FROM federation_nodes WHERE id = %s", (node_id,))
                row = cur.fetchone()
                if row:
                    return FederationNode.from_row(row)
                return None
        except psycopg2.Error as e:
            logger.warning(f"Database error getting node {node_id}: {e}")
            return None

    def save_node_trust(self, node_trust: NodeTrust) -> NodeTrust | None:
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

        except psycopg2.Error as e:
            logger.exception(f"Database error saving node trust for {node_trust.node_id}")
            return None

    # -------------------------------------------------------------------------
    # USER TRUST PREFERENCES
    # -------------------------------------------------------------------------

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
        except psycopg2.Error as e:
            logger.warning(f"Database error getting user trust preference for node {node_id}: {e}")
            return None

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

        except psycopg2.Error as e:
            logger.exception(f"Database error setting user preference for node {node_id}")
            return None

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

        except psycopg2.Error as e:
            logger.exception(f"Database error annotating belief {belief_id}")
            return None

    def get_belief_trust_adjustments(self, belief_id: UUID) -> float:
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

        except psycopg2.Error as e:
            logger.warning(f"Database error getting belief trust adjustments: {e}")
            return 0.0
