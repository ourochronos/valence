"""
Integration tests for Valence v2 deployment.

Verify that a deployed database has the correct schema and basic operations work.

Requirements:
    - PostgreSQL with pgvector available (VALENCE_DB_HOST, etc.)

Usage:
    pytest tests/integration/test_deployment.py -v
    pytest -m "not integration"  # skip
"""

import os

import psycopg2
import pytest

DB_HOST = os.environ.get("VALENCE_DB_HOST", "localhost")
DB_PORT = int(os.environ.get("VALENCE_DB_PORT", "5432"))
DB_NAME = os.environ.get("VALENCE_DB_NAME", "valence_test")
DB_USER = os.environ.get("VALENCE_DB_USER", "valence")
DB_PASS = os.environ.get("VALENCE_DB_PASSWORD", "")


def _check_db_available():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


_DB_AVAILABLE = _check_db_available()


def get_db_connection():
    if not _DB_AVAILABLE:
        pytest.skip("Database not available (integration test)")
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )


pytestmark = pytest.mark.integration

# The 10 v2 tables
V2_TABLES = {
    "sources",
    "articles",
    "article_sources",
    "usage_traces",
    "contentions",
    "entities",
    "article_entities",
    "system_config",
    "article_mutations",
    "mutation_queue",
}


class TestDatabaseSchema:
    def test_database_connection(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1
        conn.close()

    def test_pgvector_extension(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        assert cursor.fetchone() is not None, "pgvector extension not installed"
        conn.close()

    def test_v2_tables_exist(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        existing = {row[0] for row in cursor.fetchall()}
        conn.close()
        missing = V2_TABLES - existing
        assert not missing, f"Missing tables: {missing}"

    def test_sources_table_columns(self):
        expected = {
            "id",
            "type",
            "title",
            "url",
            "content",
            "fingerprint",
            "reliability",
            "content_hash",
            "metadata",
            "created_at",
            "embedding",
            "content_tsv",
            "supersedes_id",
        }
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'sources'
        """)
        actual = {row[0] for row in cursor.fetchall()}
        conn.close()
        missing = expected - actual
        assert not missing, f"Sources table missing columns: {missing}"

    def test_articles_table_columns(self):
        expected = {
            "id",
            "title",
            "content",
            "status",
            "version",
            "confidence",
            "domain_path",
            "created_at",
            "modified_at",
            "embedding",
            "content_tsv",
            "usage_score",
            "pinned",
        }
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'articles'
        """)
        actual = {row[0] for row in cursor.fetchall()}
        conn.close()
        missing = expected - actual
        assert not missing, f"Articles table missing columns: {missing}"


class TestSourceOperations:
    @pytest.fixture
    def db_conn(self):
        conn = get_db_connection()
        yield conn
        conn.rollback()
        conn.close()

    def test_insert_source(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute("""
            INSERT INTO sources (type, title, content, fingerprint, reliability, content_hash)
            VALUES ('observation', 'Test', 'Test content', 'abc123', 0.8, 'abc123')
            RETURNING id
        """)
        source_id = cursor.fetchone()[0]
        assert source_id is not None

    def test_supersession(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute("""
            INSERT INTO sources (type, content, fingerprint, reliability, content_hash)
            VALUES ('observation', 'Original', 'orig1', 0.8, 'orig1')
            RETURNING id
        """)
        original_id = cursor.fetchone()[0]

        cursor.execute(
            """
            INSERT INTO sources (type, content, fingerprint, reliability, content_hash, supersedes_id)
            VALUES ('observation', 'Updated', 'upd1', 0.9, 'upd1', %s)
            RETURNING id
        """,
            (original_id,),
        )
        new_id = cursor.fetchone()[0]
        assert new_id is not None

        cursor.execute("SELECT supersedes_id FROM sources WHERE id = %s", (new_id,))
        assert cursor.fetchone()[0] == original_id


class TestEntityOperations:
    @pytest.fixture
    def db_conn(self):
        conn = get_db_connection()
        yield conn
        conn.rollback()
        conn.close()

    def test_create_entity(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute("""
            INSERT INTO entities (name, type, aliases)
            VALUES ('Test Entity', 'concept', ARRAY['test'])
            RETURNING id
        """)
        assert cursor.fetchone()[0] is not None

    def test_entity_alias_lookup(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute("""
            INSERT INTO entities (name, type, aliases)
            VALUES ('Alias Test', 'tool', ARRAY['at', 'alias-test'])
            RETURNING id
        """)
        entity_id = cursor.fetchone()[0]
        cursor.execute("SELECT id FROM entities WHERE 'at' = ANY(aliases)")
        assert cursor.fetchone()[0] == entity_id


class TestIdempotency:
    @pytest.fixture
    def db_conn(self):
        conn = get_db_connection()
        yield conn
        conn.close()

    def test_extension_reapply_safe(self, db_conn):
        cursor = db_conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        db_conn.rollback()
