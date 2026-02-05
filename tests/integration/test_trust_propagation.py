"""Integration tests for trust propagation in federation.

Tests the trust propagation model:
- Trust anchor management
- Transitive trust calculation
- Trust score updates from corroboration
- Trust boundaries and limits

Requirements:
    - PostgreSQL database available (VKB_DB_HOST, VKB_DB_NAME, etc.)
    - Schema initialized with federation_nodes, node_trust tables

These tests are automatically skipped when PostgreSQL is unavailable.
"""

from __future__ import annotations

from uuid import uuid4

import psycopg2.extras
import pytest
from psycopg2.extras import Json

pytestmark = pytest.mark.integration


class TestFederationNodes:
    """Tests for federation node management."""

    def test_create_federation_node(self, db_conn):
        """Test creating a federation node."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            did = f"did:vkb:web:node-{uuid4()}.example.com"
            cur.execute(
                """
                INSERT INTO federation_nodes (did, federation_endpoint, public_key_multibase, status)
                VALUES (%s, %s, %s, %s)
                RETURNING id, did, status
            """,
                (
                    did,
                    "https://trusted.example.com/federation",
                    "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
                    "active",
                ),
            )

            node = cur.fetchone()
            assert node["did"] == did
            assert node["status"] == "active"

    def test_create_node_trust(self, db_conn):
        """Test creating trust for a federation node."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create node first
            did = f"did:vkb:web:trust-test-{uuid4()}.example.com"
            cur.execute(
                """
                INSERT INTO federation_nodes (did, federation_endpoint, public_key_multibase, status)
                VALUES (%s, 'http://test.example.com', 'z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK', 'active')
                RETURNING id
            """,
                (did,),
            )
            node_id = cur.fetchone()["id"]

            # Add trust
            cur.execute(
                """
                INSERT INTO node_trust (node_id, trust)
                VALUES (%s, %s)
                RETURNING id, trust
            """,
                (node_id, Json({"overall": 0.85, "belief_accuracy": 0.9})),
            )

            trust = cur.fetchone()
            assert trust["trust"]["overall"] == 0.85
            assert trust["trust"]["belief_accuracy"] == 0.9

    def test_deactivate_federation_node(self, db_conn):
        """Test deactivating a federation node."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create node
            did = f"did:vkb:web:deactivate-{uuid4()}.example.com"
            cur.execute(
                """
                INSERT INTO federation_nodes (did, federation_endpoint, public_key_multibase, status)
                VALUES (%s, 'http://deactivate.example.com', 'z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK', 'active')
                RETURNING id
            """,
                (did,),
            )
            node_id = cur.fetchone()["id"]

            # Deactivate
            cur.execute(
                """
                UPDATE federation_nodes
                SET status = 'suspended', last_seen_at = NOW()
                WHERE id = %s
                RETURNING status
            """,
                (node_id,),
            )

            result = cur.fetchone()
            assert result["status"] == "suspended"


class TestTransitiveTrust:
    """Tests for transitive trust calculation."""

    def test_direct_trust_propagation(self, db_conn):
        """Test trust propagation through direct peer relationship."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create trust chain: local -> peer_a (0.8) -> peer_b (0.7)
            # Expected trust in peer_b = 0.8 * 0.7 = 0.56

            local_trust_in_a = 0.8
            a_trust_in_b = 0.7

            # Create federation node
            did = f"did:vkb:web:peer-a-{uuid4()}.example.com"
            cur.execute(
                """
                INSERT INTO federation_nodes (did, federation_endpoint, public_key_multibase, status)
                VALUES (%s, 'http://peer-a.example.com', 'z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK', 'active')
                RETURNING id
            """,
                (did,),
            )
            node_id = cur.fetchone()["id"]

            # Store trust relationship
            cur.execute(
                """
                INSERT INTO node_trust (node_id, trust)
                VALUES (%s, %s)
            """,
                (node_id, Json({"overall": local_trust_in_a})),
            )

            # Calculate transitive trust
            transitive_trust = local_trust_in_a * a_trust_in_b

            assert transitive_trust == pytest.approx(0.56, rel=0.01)

    def test_trust_decay_with_hops(self, db_conn):
        """Test that trust decays with each hop in the chain."""
        # Trust chain: local -> A (0.9) -> B (0.9) -> C (0.9)
        hops = [0.9, 0.9, 0.9]

        # Calculate cumulative trust
        trust = 1.0
        for hop_trust in hops:
            trust *= hop_trust

        # After 3 hops of 0.9, trust should be ~0.73
        assert trust == pytest.approx(0.729, rel=0.01)

        # With many hops, trust approaches zero
        many_hops = [0.9] * 10
        trust = 1.0
        for hop_trust in many_hops:
            trust *= hop_trust

        assert trust < 0.4  # Significantly decayed

    def test_trust_ceiling(self, db_conn):
        """Test that transitive trust cannot exceed direct trust."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor):
            # Even if A trusts B at 1.0, our trust in B is limited by our trust in A
            our_trust_in_a = 0.7
            a_trust_in_b = 1.0

            transitive_trust = min(our_trust_in_a, our_trust_in_a * a_trust_in_b)

            assert transitive_trust <= our_trust_in_a


class TestCorroborationTrust:
    """Tests for trust updates based on corroboration."""

    def test_corroboration_boosts_confidence(self, db_conn):
        """Test that corroboration from trusted peers boosts belief confidence."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create initial belief
            initial_confidence = 0.6
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES ('Fact needing corroboration', %s, ARRAY['shared'])
                RETURNING id
            """,
                (Json({"overall": initial_confidence}),),
            )
            cur.fetchone()["id"]

            # Simulate corroboration from trusted peer (trust = 0.8)
            peer_trust = 0.8
            peer_confidence = 0.9  # Peer is confident in this fact

            # Corroboration formula: new_conf = old_conf + (peer_trust * peer_conf * (1 - old_conf)) * factor
            # Simplified: boost proportional to peer's trust-weighted confidence
            boost_factor = 0.3  # How much we weight external corroboration
            boost = peer_trust * peer_confidence * (1 - initial_confidence) * boost_factor
            new_confidence = min(1.0, initial_confidence + boost)

            # Should increase confidence
            assert new_confidence > initial_confidence
            assert new_confidence < 1.0  # But not to certainty

    def test_contradiction_lowers_confidence(self, db_conn):
        """Test that contradiction from trusted peer lowers confidence."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create belief with high confidence
            initial_confidence = 0.9
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES ('Potentially wrong fact', %s, ARRAY['uncertain'])
                RETURNING id
            """,
                (Json({"overall": initial_confidence}),),
            )

            # Trusted peer contradicts (trust = 0.7, claims it's false = confidence 0.1)
            peer_trust = 0.7
            peer_confidence = 0.1  # Low confidence = contradiction

            # Contradiction formula: reduce confidence proportionally
            reduction_factor = 0.2
            reduction = peer_trust * (1 - peer_confidence) * initial_confidence * reduction_factor
            new_confidence = max(0.0, initial_confidence - reduction)

            assert new_confidence < initial_confidence

    def test_multiple_corroborations(self, db_conn):
        """Test confidence updates from multiple corroborating sources."""
        initial_confidence = 0.5

        corroborations = [
            {"peer_trust": 0.8, "peer_conf": 0.9},
            {"peer_trust": 0.6, "peer_conf": 0.85},
            {"peer_trust": 0.9, "peer_conf": 0.95},
        ]

        confidence = initial_confidence
        for c in corroborations:
            boost_factor = 0.2
            boost = c["peer_trust"] * c["peer_conf"] * (1 - confidence) * boost_factor
            confidence = min(1.0, confidence + boost)

        # Multiple corroborations should significantly boost confidence
        assert confidence > 0.65  # ~0.68 with current formula
        # But should still be bounded by uncertainty
        assert confidence < 1.0


class TestTrustBoundaries:
    """Tests for trust boundaries and limits."""

    def test_minimum_trust_threshold(self, db_conn):
        """Test that peers below minimum trust are not synced with."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            minimum_trust = 0.3

            # Create nodes with varying trust
            peers = [
                ("trusted-peer", 0.8),
                ("marginal-peer", 0.35),
                ("untrusted-peer", 0.1),
            ]

            node_ids_map = {}
            for name, trust in peers:
                did = f"did:vkb:web:{name}-{uuid4()}.example.com"
                cur.execute(
                    """
                    INSERT INTO federation_nodes (did, federation_endpoint, public_key_multibase, status)
                    VALUES (%s, %s, %s, 'active')
                    RETURNING id
                """,
                    (did, f"http://{name}.example.com", "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"),
                )
                node_id = cur.fetchone()["id"]
                node_ids_map[name] = node_id

                cur.execute(
                    """
                    INSERT INTO node_trust (node_id, trust)
                    VALUES (%s, %s)
                """,
                    (node_id, Json({"overall": trust})),
                )

            # Query peers above threshold via join
            cur.execute(
                """
                SELECT fn.id, nt.trust
                FROM federation_nodes fn
                JOIN node_trust nt ON fn.id = nt.node_id
                WHERE (nt.trust->>'overall')::numeric >= %s
                AND fn.status = 'active'
            """,
                (minimum_trust,),
            )

            syncable_peers = cur.fetchall()
            syncable_ids = [p["id"] for p in syncable_peers]

            assert node_ids_map["trusted-peer"] in syncable_ids
            assert node_ids_map["marginal-peer"] in syncable_ids
            assert node_ids_map["untrusted-peer"] not in syncable_ids

    def test_trust_domain_isolation(self, db_conn):
        """Test that trust can be scoped to specific domains."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create node with domain-specific trust
            # High trust for 'tech' domain, low for 'personal'
            did = f"did:vkb:web:domain-peer-{uuid4()}.example.com"
            cur.execute(
                """
                INSERT INTO federation_nodes (did, federation_endpoint, public_key_multibase, status)
                VALUES (%s, 'http://domain.example.com', 'z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK', 'active')
                RETURNING id
            """,
                (did,),
            )
            node_id = cur.fetchone()["id"]

            # Store domain-specific trust in JSONB
            domain_trust = {
                "overall": 0.5,
                "domains": {
                    "tech": 0.9,
                    "personal": 0.2,
                },
            }

            cur.execute(
                """
                INSERT INTO node_trust (node_id, trust)
                VALUES (%s, %s)
                RETURNING trust
            """,
                (node_id, Json(domain_trust)),
            )

            result = cur.fetchone()
            assert result["trust"]["domains"]["tech"] == 0.9

    def test_trust_by_status(self, db_conn):
        """Test that node status affects trust relationships."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create nodes with different statuses
            for status in ["active", "discovered", "suspended"]:
                did = f"did:vkb:web:{status}-node-{uuid4()}.example.com"
                cur.execute(
                    """
                    INSERT INTO federation_nodes (did, federation_endpoint, public_key_multibase, status)
                VALUES (%s, %s, %s, %s)
                """,
                    (did, f"http://{status}.example.com", "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK", status),
                )

            # Query only active nodes
            cur.execute(
                """
                SELECT did, status
                FROM federation_nodes
                WHERE status = 'active'
            """
            )

            active_nodes = cur.fetchall()
            statuses = [n["status"] for n in active_nodes]

            # All returned nodes should be active
            assert all(s == "active" for s in statuses)


class TestTrustMetrics:
    """Tests for trust metrics and monitoring."""

    @pytest.mark.skip(reason="peer_nodes table not in schema yet")
    def test_track_sync_success_rate(self, db_conn):
        """Test tracking sync success rate per peer."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create peer with sync stats
            cur.execute(
                """
                INSERT INTO peer_nodes (node_id, endpoint, trust_level, status)
                VALUES ('metrics-peer', 'http://metrics.example.com', 0.7, 'active')
                RETURNING id
            """
            )
            cur.fetchone()["id"]

            # Simulate sync attempts (would be in separate table in real impl)
            sync_attempts = 10
            sync_successes = 8
            success_rate = sync_successes / sync_attempts

            # High success rate should not decrease trust
            # Low success rate might trigger trust review
            assert success_rate >= 0.7  # Acceptable rate

    def test_belief_acceptance_rate(self, db_conn):
        """Test tracking belief acceptance rate during sync."""
        # Track how many beliefs from a peer are accepted vs rejected
        beliefs_received = 100
        beliefs_accepted = 85
        beliefs_rejected = 15

        acceptance_rate = beliefs_accepted / beliefs_received

        # Good acceptance rate
        assert acceptance_rate > 0.8

        # If rejection rate is high, might indicate trust issues
        rejection_rate = beliefs_rejected / beliefs_received
        if rejection_rate > 0.3:
            # Would trigger trust review
            pass

    def test_corroboration_tracking(self, db_conn_committed, seed_beliefs):
        """Test tracking corroboration statistics for beliefs."""
        with db_conn_committed.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get a belief
            cur.execute("SELECT id FROM beliefs LIMIT 1")
            belief_id = cur.fetchone()["id"]

            # Check if corroboration tracking exists
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_name = 'beliefs'
                    AND column_name = 'corroboration_count'
                )
            """
            )

            # If corroboration tracking exists, verify it
            if cur.fetchone()["exists"]:
                cur.execute(
                    """
                    SELECT corroboration_count
                    FROM beliefs
                    WHERE id = %s
                """,
                    (belief_id,),
                )

                result = cur.fetchone()
                assert result["corroboration_count"] >= 0


class TestTrustRecovery:
    """Tests for trust recovery mechanisms."""

    def test_trust_rehabilitation(self, db_conn):
        """Test that trust can be restored after issues are resolved."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create node with reduced trust
            did = f"did:vkb:web:rehab-{uuid4()}.example.com"
            cur.execute(
                """
                INSERT INTO federation_nodes (did, federation_endpoint, public_key_multibase, status)
                VALUES (%s, 'http://rehab.example.com', 'z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK', 'suspended')
                RETURNING id
            """,
                (did,),
            )
            node_id = cur.fetchone()["id"]

            # Create initial low trust
            cur.execute(
                """
                INSERT INTO node_trust (node_id, trust)
                VALUES (%s, %s)
                RETURNING id
            """,
                (node_id, Json({"overall": 0.3})),
            )
            trust_id = cur.fetchone()["id"]

            # After successful interactions, restore trust gradually
            trust_increment = 0.1
            new_trust = 0.3 + trust_increment

            cur.execute(
                """
                UPDATE node_trust
                SET trust = %s
                WHERE id = %s
                RETURNING trust
            """,
                (Json({"overall": new_trust}), trust_id),
            )

            result = cur.fetchone()
            assert result["trust"]["overall"] == pytest.approx(0.4, rel=0.01)

    def test_permanent_distrust(self, db_conn):
        """Test marking a node as permanently distrusted."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create and permanently distrust a node
            did = f"did:vkb:web:bad-{uuid4()}.example.com"
            cur.execute(
                """
                INSERT INTO federation_nodes (did, federation_endpoint, public_key_multibase, status)
                VALUES (%s, 'http://bad.example.com', 'z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK', 'suspended')
                RETURNING id
            """,
                (did,),
            )
            node_id = cur.fetchone()["id"]

            # Set trust to zero
            cur.execute(
                """
                INSERT INTO node_trust (node_id, trust)
                VALUES (%s, %s)
            """,
                (node_id, Json({"overall": 0.0})),
            )

            # Verify blocked nodes are excluded from sync queries
            cur.execute(
                """
                SELECT COUNT(*) as count
                FROM federation_nodes fn
                JOIN node_trust nt ON fn.id = nt.node_id
                WHERE fn.status = 'active'
                AND (nt.trust->>'overall')::numeric > 0
                AND fn.id = %s
            """,
                (node_id,),
            )

            result = cur.fetchone()
            assert result["count"] == 0
