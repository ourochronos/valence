"""
Integration tests for Valence Pod deployment.

These tests verify that a deployed pod is functioning correctly.
They can be run against a local development database or a remote pod.

Requirements:
    - PostgreSQL database available (VKB_DB_HOST, VKB_DB_NAME, etc.)
    - For remote tests: VALENCE_DOMAIN environment variable

These tests are automatically skipped when PostgreSQL is unavailable.

Usage:
    # Test local database
    pytest tests/integration/test_deployment.py -v

    # Test remote pod
    VALENCE_POD_IP=x.x.x.x VALENCE_DOMAIN=pod.example.com \
        pytest tests/integration/test_deployment.py -v

    # Skip slow tests
    pytest tests/integration/test_deployment.py -v -m "not slow"

    # Skip integration tests entirely (no DB)
    pytest -m "not integration"
"""

import os
import subprocess
import pytest
import psycopg2
from typing import Optional

# Configuration from environment
VALENCE_POD_IP = os.environ.get("VALENCE_POD_IP")
VALENCE_DOMAIN = os.environ.get("VALENCE_DOMAIN")
VALENCE_DB_PASSWORD = os.environ.get("VALENCE_DB_PASSWORD", "")
DB_HOST = os.environ.get("VKB_DB_HOST", "localhost")
DB_NAME = os.environ.get("VKB_DB_NAME", "valence")
DB_USER = os.environ.get("VKB_DB_USER", "valence")


def _check_db_available():
    """Check if the database is available."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=VALENCE_DB_PASSWORD,
            connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


# Check once at module load
_DB_AVAILABLE = _check_db_available()


def get_db_connection():
    """Get a database connection, either local or via SSH tunnel."""
    if not _DB_AVAILABLE:
        pytest.skip("Database not available (integration test)")

    if VALENCE_POD_IP and DB_HOST == "localhost":
        # Remote pod - would need SSH tunnel
        pytest.skip("Remote database testing requires SSH tunnel setup")

    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=VALENCE_DB_PASSWORD,
    )


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestDatabaseSchema:
    """Tests for database schema verification."""

    def test_database_connection(self):
        """Verify we can connect to the database."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        conn.close()

    def test_pgvector_extension(self):
        """Verify pgvector extension is installed."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        result = cursor.fetchone()
        assert result is not None, "pgvector extension not installed"
        conn.close()

    def test_core_tables_exist(self):
        """Verify core tables exist in the schema."""
        required_tables = [
            "beliefs",
            "entities",
            "sessions",
            "exchanges",
            "patterns",
            "tensions",
        ]

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        existing_tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        missing = set(required_tables) - existing_tables
        assert not missing, f"Missing tables: {missing}"

    def test_beliefs_table_columns(self):
        """Verify beliefs table has expected columns."""
        expected_columns = {
            "id",
            "content",
            "confidence",
            "domain_path",
            "created_at",
            "embedding",
        }

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'beliefs'
        """)
        actual_columns = {row[0] for row in cursor.fetchall()}
        conn.close()

        missing = expected_columns - actual_columns
        assert not missing, f"Beliefs table missing columns: {missing}"

    def test_sessions_table_columns(self):
        """Verify sessions table has expected columns."""
        expected_columns = {
            "id",
            "external_room_id",
            "status",
            "started_at",
        }

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'sessions'
        """)
        actual_columns = {row[0] for row in cursor.fetchall()}
        conn.close()

        missing = expected_columns - actual_columns
        assert not missing, f"Sessions table missing columns: {missing}"


class TestBeliefOperations:
    """Tests for belief CRUD operations."""

    @pytest.fixture
    def db_conn(self):
        """Provide a database connection with cleanup."""
        conn = get_db_connection()
        yield conn
        conn.rollback()  # Rollback any test changes
        conn.close()

    def test_create_belief(self, db_conn):
        """Test creating a belief."""
        cursor = db_conn.cursor()

        cursor.execute("""
            INSERT INTO beliefs (content, confidence, domain_path)
            VALUES ('Test belief from integration test', 0.8, ARRAY['test'])
            RETURNING id
        """)
        belief_id = cursor.fetchone()[0]
        assert belief_id is not None

        # Verify it exists
        cursor.execute("SELECT content FROM beliefs WHERE id = %s", (belief_id,))
        result = cursor.fetchone()
        assert result[0] == "Test belief from integration test"

    def test_query_beliefs(self, db_conn):
        """Test querying beliefs."""
        cursor = db_conn.cursor()

        # Insert test data
        cursor.execute("""
            INSERT INTO beliefs (content, confidence, domain_path)
            VALUES ('Query test belief', 0.7, ARRAY['test', 'query'])
            RETURNING id
        """)
        belief_id = cursor.fetchone()[0]

        # Query by domain path
        cursor.execute("""
            SELECT id, content FROM beliefs
            WHERE domain_path @> ARRAY['test']
        """)
        results = cursor.fetchall()
        assert any(r[0] == belief_id for r in results)


class TestSessionOperations:
    """Tests for session operations."""

    @pytest.fixture
    def db_conn(self):
        """Provide a database connection with cleanup."""
        conn = get_db_connection()
        yield conn
        conn.rollback()
        conn.close()

    def test_create_session(self, db_conn):
        """Test creating a session."""
        cursor = db_conn.cursor()

        cursor.execute("""
            INSERT INTO vkb_sessions (external_room_id, status, platform)
            VALUES ('!test_room:example.com', 'active', 'slack')
            RETURNING id
        """)
        session_id = cursor.fetchone()[0]
        assert session_id is not None

    def test_add_exchange(self, db_conn):
        """Test adding an exchange to a session."""
        cursor = db_conn.cursor()

        # Create session
        cursor.execute("""
            INSERT INTO vkb_sessions (external_room_id, status, platform)
            VALUES ('!exchange_test:example.com', 'active', 'slack')
            RETURNING id
        """)
        session_id = cursor.fetchone()[0]

        # Add exchange
        cursor.execute("""
            INSERT INTO vkb_exchanges (session_id, sequence, role, content)
            VALUES (%s, 1, 'user', 'Test message')
            RETURNING id
        """, (session_id,))
        exchange_id = cursor.fetchone()[0]
        assert exchange_id is not None


class TestEntityOperations:
    """Tests for entity operations."""

    @pytest.fixture
    def db_conn(self):
        """Provide a database connection with cleanup."""
        conn = get_db_connection()
        yield conn
        conn.rollback()
        conn.close()

    def test_create_entity(self, db_conn):
        """Test creating an entity."""
        cursor = db_conn.cursor()

        cursor.execute("""
            INSERT INTO entities (name, entity_type, aliases)
            VALUES ('Test Entity', 'concept', ARRAY['test', 'testing'])
            RETURNING id
        """)
        entity_id = cursor.fetchone()[0]
        assert entity_id is not None

    def test_entity_aliases(self, db_conn):
        """Test entity alias lookup."""
        cursor = db_conn.cursor()

        cursor.execute("""
            INSERT INTO entities (name, entity_type, aliases)
            VALUES ('Alias Test', 'tool', ARRAY['at', 'alias-test'])
            RETURNING id
        """)
        entity_id = cursor.fetchone()[0]

        # Query by alias
        cursor.execute("""
            SELECT id FROM entities WHERE 'at' = ANY(aliases)
        """)
        result = cursor.fetchone()
        assert result[0] == entity_id


@pytest.mark.slow
class TestRemoteEndpoints:
    """Tests for remote HTTP endpoints (requires VALENCE_DOMAIN)."""

    @pytest.fixture(autouse=True)
    def require_domain(self):
        """Skip these tests if domain is not set."""
        if not VALENCE_DOMAIN:
            pytest.skip("VALENCE_DOMAIN not set")

    def test_matrix_wellknown_server(self):
        """Test Matrix server well-known endpoint."""
        import requests

        response = requests.get(
            f"https://{VALENCE_DOMAIN}/.well-known/matrix/server",
            timeout=10,
        )
        assert response.status_code == 200
        data = response.json()
        assert "m.server" in data

    def test_matrix_wellknown_client(self):
        """Test Matrix client well-known endpoint."""
        import requests

        response = requests.get(
            f"https://{VALENCE_DOMAIN}/.well-known/matrix/client",
            timeout=10,
        )
        assert response.status_code == 200
        data = response.json()
        assert "m.homeserver" in data

    def test_matrix_versions(self):
        """Test Matrix versions endpoint."""
        import requests

        response = requests.get(
            f"https://{VALENCE_DOMAIN}/_matrix/client/versions",
            timeout=10,
        )
        assert response.status_code == 200
        data = response.json()
        assert "versions" in data


class TestIdempotency:
    """Tests for deployment idempotency."""

    @pytest.fixture
    def db_conn(self):
        """Provide a database connection."""
        conn = get_db_connection()
        yield conn
        conn.close()

    def test_schema_reapply_safe(self, db_conn):
        """Verify schema can be reapplied without errors.

        This tests that IF NOT EXISTS patterns work correctly.
        """
        cursor = db_conn.cursor()

        # Try creating tables that should already exist
        # This should not raise errors if schema is idempotent

        # Test a typical IF NOT EXISTS pattern
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS beliefs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid()
            )
        """)
        # Should not raise error

        db_conn.rollback()

    def test_extension_reapply_safe(self, db_conn):
        """Verify pgvector extension can be created again."""
        cursor = db_conn.cursor()

        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # Should not raise error

        db_conn.rollback()


# CLI for running verification outside pytest
if __name__ == "__main__":
    print("Running deployment verification...")
    print(f"Target: {VALENCE_DOMAIN or 'local'}")
    print()

    # Quick connectivity check
    try:
        conn = get_db_connection()
        print("✓ Database connection successful")
        conn.close()
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        exit(1)

    print()
    print("Run full tests with: pytest tests/integration/test_deployment.py -v")
