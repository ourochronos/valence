#!/usr/bin/env python3
"""
Valence SessionEnd Hook
Closes VKB session when Claude session ends.

Uses parameterized queries to prevent SQL injection.
"""

import json
import os
import sys

# Try to import psycopg2, but gracefully handle if not available
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


def get_db_connection():
    """Create database connection using environment variables."""
    if not HAS_PSYCOPG2:
        return None

    try:
        return psycopg2.connect(
            host=os.environ.get("VKB_DB_HOST", "localhost"),
            dbname=os.environ.get("VKB_DB_NAME", "valence"),
            user=os.environ.get("VKB_DB_USER", "valence"),
            password=os.environ.get("VKB_DB_PASSWORD", ""),
        )
    except psycopg2.Error:
        return None


def close_session(conn, session_id: str) -> bool:
    """Close session using parameterized query to prevent SQL injection."""
    if not conn or not session_id:
        return False

    try:
        with conn.cursor() as cur:
            # Use parameterized query - session_id is passed as parameter, not interpolated
            cur.execute(
                """
                UPDATE sessions
                SET status = 'completed', ended_at = NOW()
                WHERE id = %s
                """,
                (session_id,)
            )
            conn.commit()
            return cur.rowcount > 0
    except psycopg2.Error:
        try:
            conn.rollback()
        except psycopg2.Error:
            pass
        return False


def main():
    """Main entry point for session end hook."""
    session_id = os.environ.get("VALENCE_SESSION_ID", "")

    if session_id:
        conn = get_db_connection()
        if conn:
            try:
                close_session(conn, session_id)
            finally:
                conn.close()

    # Return empty object (no modifications needed)
    print("{}")


if __name__ == "__main__":
    main()
