#!/usr/bin/env python3
"""
Valence SessionEnd Hook
Closes VKB session and auto-captures beliefs from session summary/themes.

Auto-capture creates beliefs from:
1. Session summary (if substantive) at confidence 0.50
2. Individual themes at confidence 0.45
3. Links beliefs to session via vkb_session_insights

Uses parameterized queries to prevent SQL injection.
"""

import hashlib
import json
import os
import sys
import uuid

# Try to import psycopg2, but gracefully handle if not available
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# Curation constants (duplicated from core/curation.py — hooks are standalone)
SUMMARY_CONFIDENCE = 0.50  # lowered: corroboration will raise it
THEME_CONFIDENCE = 0.45    # lowered: corroboration will raise it
MAX_AUTO_BELIEFS_PER_SESSION = 10
MIN_SUMMARY_LENGTH = 20
MIN_THEME_LENGTH = 10


def get_db_connection():
    """Create database connection using environment variables."""
    if not HAS_PSYCOPG2:
        return None

    try:
        return psycopg2.connect(
            host=os.environ.get("VALENCE_DB_HOST", "localhost"),
            dbname=os.environ.get("VALENCE_DB_NAME", "valence"),
            user=os.environ.get("VALENCE_DB_USER", "valence"),
            password=os.environ.get("VALENCE_DB_PASSWORD", ""),
        )
    except psycopg2.Error:
        return None


def get_session(conn, session_id: str) -> dict | None:
    """Fetch session summary and themes."""
    if not conn or not session_id:
        return None

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT summary, themes, project_context
                FROM vkb_sessions
                WHERE id = %s
                """,
                (session_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except psycopg2.Error:
        return None


def create_belief(conn, content: str, confidence: float, domain_path: list[str] | None = None) -> str | None:
    """Create a belief and return its ID. Deduplicates by content hash.

    If an exact duplicate exists, reinforces the existing belief instead
    of creating a new one.
    """
    content_hash = hashlib.sha256(content.strip().lower().encode()).hexdigest()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Check for exact duplicate
            cur.execute(
                "SELECT id, confidence FROM beliefs WHERE content_hash = %s AND status = 'active' AND superseded_by_id IS NULL",
                (content_hash,),
            )
            existing = cur.fetchone()
            if existing:
                # Reinforce: record corroboration and bump confidence
                existing_id = str(existing["id"])
                cur.execute("SELECT COUNT(*) as cnt FROM belief_corroborations WHERE belief_id = %s", (existing_id,))
                count = cur.fetchone()["cnt"]
                cur.execute(
                    "INSERT INTO belief_corroborations (belief_id, source_type) VALUES (%s, 'session')",
                    (existing_id,),
                )
                # Simple escalation: 0-1→0.50, 2→0.65, 3+→0.80
                new_count = count + 1
                new_overall = 0.50 if new_count <= 1 else (0.65 if new_count == 2 else 0.80)
                existing_conf = existing["confidence"] if isinstance(existing["confidence"], dict) else json.loads(existing["confidence"])
                existing_conf["overall"] = max(existing_conf.get("overall", 0.5), new_overall)
                existing_conf["corroboration"] = new_overall
                cur.execute("UPDATE beliefs SET confidence = %s, modified_at = NOW() WHERE id = %s", (json.dumps(existing_conf), existing_id))
                return existing_id

            # No duplicate — create new
            belief_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO beliefs (id, content, confidence, domain_path, extraction_method, content_hash, status)
                VALUES (%s, %s, %s, %s, 'auto', %s, 'active')
                """,
                (belief_id, content, json.dumps({"overall": confidence}), domain_path or [], content_hash),
            )
            return belief_id
    except psycopg2.Error:
        return None


def link_belief_to_session(conn, session_id: str, belief_id: str) -> bool:
    """Create a vkb_session_insights link."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vkb_session_insights (session_id, belief_id, extraction_method)
                VALUES (%s, %s, 'auto')
                ON CONFLICT (session_id, belief_id) DO NOTHING
                """,
                (session_id, belief_id)
            )
            return True
    except psycopg2.Error:
        return False


def generate_embedding(conn, belief_id: str, content: str) -> bool:
    """Try to generate and store an embedding for the belief."""
    try:
        from our_embeddings.service import generate_embedding as gen_emb, vector_to_pgvector
        embedding = gen_emb(content)
        if embedding:
            emb_str = vector_to_pgvector(embedding)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE beliefs SET embedding = %s::vector WHERE id = %s",
                    (emb_str, belief_id)
                )
            return True
    except Exception:
        pass  # Embedding failure is non-fatal
    return False


def auto_capture_beliefs(conn, session_id: str, session_data: dict) -> int:
    """Extract beliefs from session summary and themes.

    Returns number of beliefs created.
    """
    summary = session_data.get("summary") or ""
    themes = session_data.get("themes") or []
    project_context = session_data.get("project_context") or ""
    domain_path = [project_context] if project_context else []

    beliefs_created = 0

    # Capture summary as a belief
    if summary and len(summary.strip()) >= MIN_SUMMARY_LENGTH:
        belief_id = create_belief(conn, summary, SUMMARY_CONFIDENCE, domain_path)
        if belief_id:
            link_belief_to_session(conn, session_id, belief_id)
            generate_embedding(conn, belief_id, summary)
            beliefs_created += 1

    # Capture each theme as an individual belief
    for theme in themes:
        if beliefs_created >= MAX_AUTO_BELIEFS_PER_SESSION:
            break

        theme = theme.strip() if isinstance(theme, str) else ""
        if len(theme) >= MIN_THEME_LENGTH:
            belief_id = create_belief(conn, theme, THEME_CONFIDENCE, domain_path)
            if belief_id:
                link_belief_to_session(conn, session_id, belief_id)
                generate_embedding(conn, belief_id, theme)
                beliefs_created += 1

    return beliefs_created


def close_session(conn, session_id: str) -> bool:
    """Close session using parameterized query to prevent SQL injection."""
    if not conn or not session_id:
        return False

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE vkb_sessions
                SET status = 'completed', ended_at = NOW()
                WHERE id = %s
                """,
                (session_id,)
            )
            return cur.rowcount > 0
    except psycopg2.Error:
        return False


def main():
    """Main entry point for session end hook."""
    session_id = os.environ.get("VALENCE_SESSION_ID", "")

    if session_id:
        conn = get_db_connection()
        if conn:
            try:
                # Auto-capture beliefs from session summary/themes
                session_data = get_session(conn, session_id)
                if session_data:
                    auto_capture_beliefs(conn, session_id, session_data)

                # Close the session
                close_session(conn, session_id)

                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                conn.close()

    # Return empty object (no modifications needed)
    print("{}")


if __name__ == "__main__":
    main()
