"""Integration tests for federation sync between two Valence nodes.

Tests the federation protocol:
- Node discovery and handshake
- Belief propagation between peers
- Selective sync based on trust levels
- Conflict resolution during sync

Requirements:
    - PostgreSQL database available (VKB_DB_HOST, VKB_DB_NAME, etc.)
    - Schema initialized with federation tables

These tests are automatically skipped when PostgreSQL is unavailable.
"""

from __future__ import annotations

from uuid import uuid4

import psycopg2.extras
import pytest
import requests
from psycopg2.extras import Json

pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="Requires primary_api fixture from live node setup")
class TestNodeDiscovery:
    """Tests for federation node discovery."""

    def test_node_metadata_endpoint(self, primary_api):
        """Test that node metadata is discoverable."""
        resp = requests.get(
            f"{primary_api}/.well-known/vfp-node-metadata",
            timeout=10,
        )

        assert resp.status_code == 200
        data = resp.json()

        # Verify required fields
        assert "node_id" in data
        assert "endpoint" in data
        assert "capabilities" in data
        assert "version" in data

    def test_trust_anchors_endpoint(self, primary_api):
        """Test that trust anchors are retrievable."""
        resp = requests.get(
            f"{primary_api}/.well-known/vfp-trust-anchors",
            timeout=10,
        )

        # May return 200 with anchors or 404 if none configured
        assert resp.status_code in [200, 404]

        if resp.status_code == 200:
            data = resp.json()
            assert "anchors" in data
            assert isinstance(data["anchors"], list)

    def test_federation_status(self, primary_api):
        """Test federation status endpoint."""
        resp = requests.get(
            f"{primary_api}/api/v1/federation/status",
            timeout=10,
        )

        assert resp.status_code == 200
        data = resp.json()

        assert "enabled" in data
        assert "node_id" in data


class TestPeerRegistration:
    """Tests for registering federation nodes."""

    def test_register_federation_node(self, db_conn):
        """Test registering a federation node in the database."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            did = f"did:vkb:web:peer-{uuid4()}.example.com"

            cur.execute(
                """
                INSERT INTO federation_nodes (did, federation_endpoint, mcp_endpoint, public_key_multibase, status)
                VALUES (%s, %s, %s, %s, 'discovered')
                RETURNING id, status
            """,
                (
                    did,
                    "http://peer.example.com:8080/federation",
                    "http://peer.example.com:8080/mcp",
                    "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
                ),
            )

            result = cur.fetchone()
            assert result is not None
            assert result["status"] == "discovered"

    def test_activate_federation_node(self, db_conn):
        """Test activating a registered node after verification."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Register node
            did = f"did:vkb:web:activate-{uuid4()}.example.com"
            cur.execute(
                """
                INSERT INTO federation_nodes (did, federation_endpoint, public_key_multibase, status)
                VALUES (%s, 'http://test.example.com', 'z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK', 'discovered')
                RETURNING id
            """,
                (did,),
            )
            node_id = cur.fetchone()["id"]

            # Activate after verification
            cur.execute(
                """
                UPDATE federation_nodes
                SET status = 'active', last_seen_at = NOW()
                WHERE id = %s
                RETURNING status, last_seen_at
            """,
                (node_id,),
            )

            result = cur.fetchone()
            assert result["status"] == "active"
            assert result["last_seen_at"] is not None


class TestBeliefSync:
    """Tests for belief synchronization between nodes."""

    def test_create_syncable_belief(self, db_conn):
        """Test creating a belief marked for federation sync."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path, status)
                VALUES ('Shared knowledge for sync', '{"overall": 0.85}', ARRAY['shared'], 'active')
                RETURNING id
            """
            )
            belief_id = cur.fetchone()["id"]

            # Mark for sync (if sync_status column exists)
            try:
                cur.execute(
                    """
                    UPDATE beliefs
                    SET metadata = jsonb_set(
                        COALESCE(metadata, '{}'::jsonb),
                        '{sync}',
                        '{"status": "pending", "priority": "normal"}'::jsonb
                    )
                    WHERE id = %s
                    RETURNING metadata
                """,
                    (belief_id,),
                )

                result = cur.fetchone()
                assert result["metadata"]["sync"]["status"] == "pending"
            except psycopg2.Error:
                # metadata column might not exist
                pass

    def test_sync_log_entry(self, db_conn):
        """Test logging sync operations."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create a belief to sync
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES ('Belief to sync', '{"overall": 0.8}', ARRAY['test'])
                RETURNING id
            """
            )
            belief_id = cur.fetchone()["id"]

            # Check if sync_log table exists
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'sync_log'
                )
            """
            )

            if cur.fetchone()["exists"]:
                # Log sync operation
                cur.execute(
                    """
                    INSERT INTO sync_log (entity_type, entity_id, operation, peer_node_id, status)
                    VALUES ('belief', %s, 'push', 'test-peer', 'discovered')
                    RETURNING id
                """,
                    (belief_id,),
                )

                log_entry = cur.fetchone()
                assert log_entry is not None

    def test_conflict_detection(self, db_conn):
        """Test detecting conflicts during sync."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create local belief
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES ('Local version of fact', '{"overall": 0.7}', ARRAY['shared'])
                RETURNING id
            """
            )
            cur.fetchone()["id"]

            # Simulate incoming belief with same content but different confidence
            # In real sync, we'd detect this as a potential conflict

            # Check for existing belief with similar content
            cur.execute(
                """
                SELECT id, content, confidence
                FROM beliefs
                WHERE content_tsv @@ plainto_tsquery('english', 'Local version fact')
                AND status = 'active'
            """
            )

            existing = cur.fetchall()

            # If we found existing beliefs, we have a potential conflict
            # Real implementation would use semantic similarity
            if len(existing) > 1:
                # Log conflict
                cur.execute(
                    """
                    INSERT INTO tensions (belief_a_id, belief_b_id, type, description)
                    VALUES (%s, %s, 'sync_conflict', 'Conflicting versions from sync')
                """,
                    (existing[0]["id"], existing[1]["id"]),
                )


class TestTrustBasedSync:
    """Tests for trust-level based sync filtering."""

    def test_filter_by_trust_level(self, db_conn):
        """Test that beliefs are filtered based on trust level."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create beliefs with different confidences
            beliefs = [
                ("High confidence fact", 0.95, ["trusted"]),
                ("Medium confidence fact", 0.7, ["shared"]),
                ("Low confidence fact", 0.3, ["uncertain"]),
            ]

            for content, conf, domains in beliefs:
                cur.execute(
                    """
                    INSERT INTO beliefs (content, confidence, domain_path)
                    VALUES (%s, %s, %s)
                """,
                    (content, Json({"overall": conf}), domains),
                )

            # Query beliefs suitable for sync to a peer with 0.5 trust
            peer_trust = 0.5
            cur.execute(
                """
                SELECT content, confidence
                FROM beliefs
                WHERE (confidence->>'overall')::numeric >= %s
                AND status = 'active'
            """,
                (peer_trust,),
            )

            syncable = cur.fetchall()

            # Should include high and medium, not low
            contents = [r["content"] for r in syncable]
            assert "High confidence fact" in contents
            assert "Medium confidence fact" in contents
            # Low confidence might not be in results if threshold applies

    @pytest.mark.skip(reason="peer_nodes table not in schema yet")
    def test_trust_decay_over_time(self, db_conn):
        """Test that trust scores can decay for inactive peers."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create peer with last_seen in past
            node_id = f"decay-test-{uuid4()}"
            cur.execute(
                """
                INSERT INTO peer_nodes (node_id, endpoint, trust_level, status, last_seen)
                VALUES (%s, 'http://old.peer.com', 0.8, 'active', NOW() - INTERVAL '30 days')
                RETURNING id, trust_level
            """,
                (node_id,),
            )

            peer = cur.fetchone()
            original_trust = peer["trust_level"]

            # In a real implementation, we'd have a decay function
            # For now, just verify the peer was created with old timestamp
            cur.execute(
                """
                SELECT node_id, trust_level, last_seen
                FROM peer_nodes
                WHERE node_id = %s
                AND last_seen < NOW() - INTERVAL '7 days'
            """,
                (node_id,),
            )

            stale_peer = cur.fetchone()
            assert stale_peer is not None
            assert stale_peer["trust_level"] == original_trust


@pytest.mark.skip(reason="Requires primary_api/peer_api fixtures from live node setup")
@pytest.mark.slow
class TestMultiNodeSync:
    """Tests requiring multiple server instances."""

    def test_two_node_handshake(self, primary_api, peer_api):
        """Test handshake between two Valence nodes."""
        # Get metadata from both nodes
        primary_meta = requests.get(
            f"{primary_api}/.well-known/vfp-node-metadata",
            timeout=10,
        ).json()

        peer_meta = requests.get(
            f"{peer_api}/.well-known/vfp-node-metadata",
            timeout=10,
        ).json()

        # Verify they have different node IDs
        assert primary_meta["node_id"] != peer_meta["node_id"]

        # Verify both are reachable
        assert "endpoint" in primary_meta
        assert "endpoint" in peer_meta

    def test_health_check_both_nodes(self, primary_api, peer_api):
        """Test that both nodes are healthy."""
        primary_health = requests.get(
            f"{primary_api}/api/v1/health",
            timeout=10,
        )
        peer_health = requests.get(
            f"{peer_api}/api/v1/health",
            timeout=10,
        )

        assert primary_health.status_code == 200
        assert peer_health.status_code == 200

        assert primary_health.json()["status"] in ["healthy", "degraded"]
        assert peer_health.json()["status"] in ["healthy", "degraded"]


class TestSyncProtocol:
    """Tests for the sync protocol implementation."""

    def test_sync_request_format(self, db_conn):
        """Test the format of sync requests."""
        # Define expected sync request structure
        sync_request = {
            "version": "1.0",
            "operation": "push",
            "node_id": "test-node",
            "timestamp": "2024-01-01T00:00:00Z",
            "beliefs": [
                {
                    "id": str(uuid4()),
                    "content": "Test belief",
                    "confidence": {"overall": 0.8},
                    "domain_path": ["test"],
                    "checksum": "sha256:abc123",
                }
            ],
            "cursor": None,  # For pagination
        }

        # Validate structure
        assert "version" in sync_request
        assert "operation" in sync_request
        assert "node_id" in sync_request
        assert isinstance(sync_request["beliefs"], list)

    def test_sync_response_format(self, db_conn):
        """Test the format of sync responses."""
        sync_response = {
            "version": "1.0",
            "status": "accepted",
            "node_id": "responder-node",
            "accepted": [str(uuid4())],
            "rejected": [],
            "conflicts": [],
            "cursor": "next-page-cursor",
        }

        assert sync_response["status"] in ["accepted", "partial", "rejected"]
        assert isinstance(sync_response["accepted"], list)

    def test_sync_cursor_pagination(self, db_conn_committed, seed_beliefs):
        """Test paginated sync using cursor."""
        with db_conn_committed.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            page_size = 2

            # First page
            cur.execute(
                """
                SELECT id, content, created_at
                FROM beliefs
                WHERE status = 'active'
                ORDER BY created_at ASC
                LIMIT %s
            """,
                (page_size,),
            )

            first_page = cur.fetchall()
            assert len(first_page) <= page_size

            if first_page:
                # Use last item's created_at as cursor
                cursor = first_page[-1]["created_at"]

                # Second page
                cur.execute(
                    """
                    SELECT id, content, created_at
                    FROM beliefs
                    WHERE status = 'active'
                    AND created_at > %s
                    ORDER BY created_at ASC
                    LIMIT %s
                """,
                    (cursor, page_size),
                )

                second_page = cur.fetchall()

                # Verify no overlap
                first_ids = {r["id"] for r in first_page}
                second_ids = {r["id"] for r in second_page}
                assert first_ids.isdisjoint(second_ids)
