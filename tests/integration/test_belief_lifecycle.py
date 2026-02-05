"""Integration tests for full belief lifecycle.

Tests the complete lifecycle of beliefs:
- Creation with confidence and domains
- Querying with semantic search
- Updates and supersession
- Deletion and archival

Requirements:
    - PostgreSQL database available (VKB_DB_HOST, VKB_DB_NAME, etc.)
    - Schema initialized with beliefs, entities tables

These tests are automatically skipped when PostgreSQL is unavailable.
"""

from __future__ import annotations

import psycopg2.extras
import pytest
from psycopg2.extras import Json

pytestmark = pytest.mark.integration


class TestBeliefCreation:
    """Tests for belief creation."""

    def test_create_belief_basic(self, db_conn):
        """Test creating a basic belief."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            content = "Test belief for lifecycle testing"
            confidence = {"overall": 0.8}
            domains = ["test", "lifecycle"]

            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES (%s, %s, %s)
                RETURNING id, content, confidence, domain_path, status, created_at
            """,
                (content, Json(confidence), domains),
            )

            result = cur.fetchone()

            assert result is not None
            assert result["content"] == content
            assert result["status"] == "active"
            assert result["domain_path"] == domains
            assert result["created_at"] is not None

    def test_create_belief_with_6d_confidence(self, db_conn):
        """Test creating belief with full 6D confidence vector."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            confidence = {
                "overall": 0.85,
                "source_reliability": 0.9,
                "method_quality": 0.8,
                "internal_consistency": 0.95,
                "temporal_freshness": 1.0,
                "corroboration": 0.6,
                "domain_applicability": 0.85,
            }

            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES (%s, %s, %s)
                RETURNING id, confidence
            """,
                ("High-quality fact", Json(confidence), ["facts"]),
            )

            result = cur.fetchone()
            stored_confidence = result["confidence"]

            # Verify all dimensions stored
            assert stored_confidence["source_reliability"] == 0.9
            assert stored_confidence["temporal_freshness"] == 1.0

    def test_create_belief_with_source(self, db_conn):
        """Test creating belief linked to a source."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create source first
            cur.execute(
                """
                INSERT INTO sources (type, title)
                VALUES ('document', 'Test Document')
                RETURNING id
            """
            )
            source_id = cur.fetchone()["id"]

            # Create belief with source
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path, source_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id, source_id
            """,
                ("Fact from document", '{"overall": 0.7}', ["facts"], source_id),
            )

            result = cur.fetchone()
            assert result["source_id"] == source_id


class TestBeliefQuerying:
    """Tests for belief querying."""

    def test_query_by_domain(self, db_conn_committed, seed_beliefs):
        """Test querying beliefs by domain path."""
        with db_conn_committed.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, content, domain_path
                FROM beliefs
                WHERE domain_path @> ARRAY['tech']
            """
            )

            results = cur.fetchall()

            # Should find beliefs in tech domain
            assert len(results) >= 3  # Python, Claude, PostgreSQL
            for row in results:
                assert "tech" in row["domain_path"]

    def test_query_by_subdomain(self, db_conn_committed, seed_beliefs):
        """Test querying beliefs by specific subdomain."""
        with db_conn_committed.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT content
                FROM beliefs
                WHERE domain_path @> ARRAY['tech', 'ai']
            """
            )

            results = cur.fetchall()

            assert len(results) >= 1
            assert any("Claude" in r["content"] for r in results)

    def test_query_by_confidence_threshold(self, db_conn_committed, seed_beliefs):
        """Test filtering beliefs by minimum confidence."""
        with db_conn_committed.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT content, confidence
                FROM beliefs
                WHERE (confidence->>'overall')::numeric >= 0.85
            """
            )

            results = cur.fetchall()

            for row in results:
                assert float(row["confidence"]["overall"]) >= 0.85

    def test_full_text_search(self, db_conn_committed, seed_beliefs):
        """Test full-text search on belief content."""
        with db_conn_committed.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT content
                FROM beliefs
                WHERE content_tsv @@ plainto_tsquery('english', 'programming language')
            """
            )

            results = cur.fetchall()

            assert len(results) >= 1
            assert any("Python" in r["content"] for r in results)


class TestBeliefSupersession:
    """Tests for belief supersession chain."""

    def test_supersede_belief(self, db_conn):
        """Test superseding a belief with updated information."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create original belief
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES ('Python 3.11 is the latest version', '{"overall": 0.9}', ARRAY['tech'])
                RETURNING id
            """
            )
            original_id = cur.fetchone()["id"]

            # Create new belief that supersedes it
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path, supersedes_id)
                VALUES ('Python 3.12 is the latest version', '{"overall": 0.95}', ARRAY['tech'], %s)
                RETURNING id
            """,
                (original_id,),
            )
            new_id = cur.fetchone()["id"]

            # Update original to mark as superseded
            cur.execute(
                """
                UPDATE beliefs
                SET superseded_by_id = %s, status = 'superseded'
                WHERE id = %s
            """,
                (new_id, original_id),
            )

            # Verify chain
            cur.execute(
                """
                SELECT id, status, superseded_by_id
                FROM beliefs
                WHERE id = %s
            """,
                (original_id,),
            )

            original = cur.fetchone()
            assert original["status"] == "superseded"
            assert original["superseded_by_id"] == new_id

    def test_query_active_only(self, db_conn):
        """Test querying excludes superseded beliefs by default."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create active and superseded beliefs
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path, status)
                VALUES
                    ('Active belief', '{"overall": 0.8}', ARRAY['test'], 'active'),
                    ('Superseded belief', '{"overall": 0.7}', ARRAY['test'], 'superseded')
            """
            )

            # Query active only
            cur.execute(
                """
                SELECT content FROM beliefs
                WHERE domain_path @> ARRAY['test']
                AND status = 'active'
            """
            )

            results = cur.fetchall()
            contents = [r["content"] for r in results]

            assert "Active belief" in contents
            assert "Superseded belief" not in contents


class TestBeliefTensions:
    """Tests for handling contradictory beliefs."""

    def test_detect_tension(self, db_conn):
        """Test creating a tension between conflicting beliefs."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create two conflicting beliefs
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES ('Feature X should be implemented', '{"overall": 0.7}', ARRAY['decisions'])
                RETURNING id
            """
            )
            belief_a = cur.fetchone()["id"]

            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES ('Feature X should not be implemented', '{"overall": 0.6}', ARRAY['decisions'])
                RETURNING id
            """
            )
            belief_b = cur.fetchone()["id"]

            # Record tension
            cur.execute(
                """
                INSERT INTO tensions (belief_a_id, belief_b_id, type, description, severity)
                VALUES (%s, %s, 'contradiction', 'Conflicting decisions about Feature X', 'medium')
                RETURNING id, status
            """,
                (belief_a, belief_b),
            )

            tension = cur.fetchone()
            assert tension["status"] == "detected"

    def test_resolve_tension(self, db_conn):
        """Test resolving a tension."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create beliefs and tension
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES
                    ('Old decision', '{"overall": 0.5}', ARRAY['decisions']),
                    ('New decision', '{"overall": 0.9}', ARRAY['decisions'])
            """
            )

            cur.execute("SELECT id FROM beliefs WHERE content = 'Old decision'")
            old_id = cur.fetchone()["id"]
            cur.execute("SELECT id FROM beliefs WHERE content = 'New decision'")
            new_id = cur.fetchone()["id"]

            cur.execute(
                """
                INSERT INTO tensions (belief_a_id, belief_b_id, type, description)
                VALUES (%s, %s, 'temporal_conflict', 'Outdated decision')
                RETURNING id
            """,
                (old_id, new_id),
            )
            tension_id = cur.fetchone()["id"]

            # Resolve tension
            cur.execute(
                """
                UPDATE tensions
                SET status = 'resolved',
                    resolution = 'Newer decision takes precedence',
                    resolved_at = NOW()
                WHERE id = %s
                RETURNING status
            """,
                (tension_id,),
            )

            result = cur.fetchone()
            assert result["status"] == "resolved"


class TestBeliefEntityLinks:
    """Tests for belief-entity relationships."""

    def test_link_belief_to_entity(self, db_conn_committed, seed_beliefs, seed_entities):
        """Test linking a belief to an entity."""
        # Get a belief and entity
        with db_conn_committed.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id FROM beliefs WHERE content LIKE '%Python%' LIMIT 1")
            belief_id = cur.fetchone()["id"]

            cur.execute("SELECT id FROM entities WHERE name = 'Python'")
            entity_id = cur.fetchone()["id"]

            # Create link
            cur.execute(
                """
                INSERT INTO belief_entities (belief_id, entity_id, role)
                VALUES (%s, %s, 'subject')
                RETURNING belief_id, entity_id
            """,
                (belief_id, entity_id),
            )

            link = cur.fetchone()
            assert link is not None

    def test_query_beliefs_by_entity(self, db_conn_committed, seed_beliefs, seed_entities):
        """Test querying beliefs related to an entity."""
        with db_conn_committed.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Link some beliefs to entities first
            cur.execute("SELECT id FROM beliefs WHERE content LIKE '%Python%' LIMIT 1")
            belief_id = cur.fetchone()["id"]

            cur.execute("SELECT id FROM entities WHERE name = 'Python'")
            entity_id = cur.fetchone()["id"]

            cur.execute(
                """
                INSERT INTO belief_entities (belief_id, entity_id, role)
                VALUES (%s, %s, 'subject')
            """,
                (belief_id, entity_id),
            )
            db_conn_committed.commit()

            # Query beliefs by entity
            cur.execute(
                """
                SELECT b.content
                FROM beliefs b
                JOIN belief_entities be ON b.id = be.belief_id
                JOIN entities e ON be.entity_id = e.id
                WHERE e.name = 'Python'
            """
            )

            results = cur.fetchall()
            assert len(results) >= 1


class TestBeliefDeletion:
    """Tests for belief deletion and archival."""

    def test_archive_belief(self, db_conn):
        """Test archiving a belief (soft delete)."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create belief
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES ('To be archived', '{"overall": 0.5}', ARRAY['test'])
                RETURNING id
            """
            )
            belief_id = cur.fetchone()["id"]

            # Archive it
            cur.execute(
                """
                UPDATE beliefs
                SET status = 'archived'
                WHERE id = %s
                RETURNING status
            """,
                (belief_id,),
            )

            result = cur.fetchone()
            assert result["status"] == "archived"

    def test_hard_delete_belief(self, db_conn):
        """Test hard deleting a belief."""
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Create belief
            cur.execute(
                """
                INSERT INTO beliefs (content, confidence, domain_path)
                VALUES ('To be deleted', '{"overall": 0.5}', ARRAY['test'])
                RETURNING id
            """
            )
            belief_id = cur.fetchone()["id"]

            # Delete belief (clean up any links first)
            cur.execute("DELETE FROM belief_entities WHERE belief_id = %s", (belief_id,))
            cur.execute(
                "DELETE FROM tensions WHERE belief_a_id = %s OR belief_b_id = %s",
                (belief_id, belief_id),
            )
            cur.execute("DELETE FROM beliefs WHERE id = %s", (belief_id,))

            # Verify deleted
            cur.execute("SELECT id FROM beliefs WHERE id = %s", (belief_id,))
            assert cur.fetchone() is None
