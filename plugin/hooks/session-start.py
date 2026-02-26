#!/usr/bin/env python3
"""
Valence SessionStart Hook
Injects relevant context and behavioral conditioning at session start.

Uses parameterized queries to prevent SQL injection.
"""

import json
import os
import sys
from typing import Any

# Try to import psycopg2, but gracefully handle if not available
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


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


def query_beliefs(conn, project_context: str = "") -> list[dict[str, Any]]:
    """Query recent active beliefs, preferring project-relevant ones."""
    if not conn:
        return []

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if project_context:
                # Try project-relevant beliefs first, fall back to recent
                cur.execute("""
                    (SELECT content, confidence->>'overall' as confidence, created_at
                    FROM beliefs
                    WHERE status = 'active'
                    AND superseded_by_id IS NULL
                    AND domain_path && %s
                    ORDER BY created_at DESC
                    LIMIT 5)
                    UNION ALL
                    (SELECT content, confidence->>'overall' as confidence, created_at
                    FROM beliefs
                    WHERE status = 'active'
                    AND superseded_by_id IS NULL
                    AND NOT (domain_path && %s)
                    ORDER BY created_at DESC
                    LIMIT 5)
                    LIMIT 5
                """, ([project_context], [project_context]))
            else:
                cur.execute("""
                    SELECT content, confidence->>'overall' as confidence, created_at
                    FROM beliefs
                    WHERE status = 'active'
                    AND superseded_by_id IS NULL
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
            return [dict(row) for row in cur.fetchall()]
    except psycopg2.Error:
        return []


def query_patterns(conn) -> list[dict[str, Any]]:
    """Query established patterns using parameterized query."""
    if not conn:
        return []

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT type, description, confidence
                FROM vkb_patterns
                WHERE status = 'established'
                ORDER BY confidence DESC
                LIMIT 5
            """)
            return [dict(row) for row in cur.fetchall()]
    except psycopg2.Error:
        return []


def create_session(conn, project_context: str) -> str | None:
    """Create new VKB session using parameterized query."""
    if not conn:
        return None

    try:
        with conn.cursor() as cur:
            # Use parameterized query to prevent SQL injection
            cur.execute(
                """
                INSERT INTO vkb_sessions (platform, project_context, status)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                ("claude-code", project_context, "active")
            )
            result = cur.fetchone()
            conn.commit()
            return str(result[0]) if result else None
    except psycopg2.Error:
        try:
            conn.rollback()
        except psycopg2.Error:
            pass
        return None


def format_beliefs(beliefs: list[dict[str, Any]]) -> str:
    """Format beliefs for display."""
    if not beliefs:
        return "No beliefs loaded."

    lines = []
    for b in beliefs[:5]:
        try:
            conf = float(b.get("confidence", 0.7))
        except (ValueError, TypeError):
            conf = 0.7
        content = str(b.get("content", ""))[:100]
        lines.append(f"- [{conf:.1f}] {content}")

    return "\n".join(lines) if lines else "No beliefs loaded."


def format_patterns(patterns: list[dict[str, Any]]) -> str:
    """Format patterns for display."""
    if not patterns:
        return "No patterns loaded."

    lines = []
    for p in patterns[:5]:
        ptype = p.get("type", "unknown")
        desc = str(p.get("description", ""))[:80]
        lines.append(f"- [{ptype}] {desc}")

    return "\n".join(lines) if lines else "No patterns loaded."


def build_context(
    project_context: str,
    session_id: str | None,
    beliefs: list[dict[str, Any]],
    patterns: list[dict[str, Any]]
) -> str:
    """Build the context injection string."""
    formatted_beliefs = format_beliefs(beliefs)
    formatted_patterns = format_patterns(patterns)

    context_parts = [
        "<VALENCE_KNOWLEDGE_SUBSTRATE>",
        "You have access to the Valence knowledge substrate through MCP tools.",
        "",
        "CRITICAL BEHAVIOR: Before answering questions about:",
        "- Past decisions or discussions",
        "- User preferences or values",
        "- Technical approaches previously explored",
        "- Any factual claims that might be in the knowledge base",
        "",
        "You MUST query the knowledge base first using:",
        "- mcp__valence__belief_query - Search for relevant beliefs",
        "- mcp__valence__pattern_list - Get established behavioral patterns",
        "",
        "This is not optional. Query first, then respond with grounded information.",
        "",
        f"Current project context: {project_context}",
    ]

    if session_id:
        context_parts.append(f"VKB session ID: {session_id}")

    context_parts.extend([
        "",
        "Recent beliefs:",
        formatted_beliefs,
        "",
        "Established patterns:",
        formatted_patterns,
        "",
        "Available skills:",
        "- /valence:using-valence - Learn about the knowledge substrate",
        "- /valence:query-knowledge - Search the knowledge base",
        "- /valence:capture-insight - Store important information",
        "- /valence:ingest-document - Add documents to the substrate",
        "- /valence:review-tensions - Review and resolve contradictions",
        "</VALENCE_KNOWLEDGE_SUBSTRATE>",
    ])

    return "\n".join(context_parts)


def main():
    """Main entry point for session start hook."""
    # Get project context from environment or current directory
    project_context = os.environ.get("CLAUDE_PROJECT") or os.path.basename(os.getcwd())

    # Initialize with defaults
    beliefs = []
    patterns = []
    session_id = None

    # Try to connect and query database
    conn = get_db_connection()
    if conn:
        try:
            beliefs = query_beliefs(conn, project_context)
            patterns = query_patterns(conn)
            session_id = create_session(conn, project_context)
        finally:
            conn.close()

    # Build context
    context = build_context(project_context, session_id, beliefs, patterns)

    # Build output
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context
        },
        "environmentVariables": {
            "VALENCE_SESSION_ID": session_id or "",
            "VALENCE_PROJECT": project_context
        }
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
