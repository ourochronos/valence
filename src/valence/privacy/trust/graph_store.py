"""Database storage for trust graph edges.

This module contains the TrustGraphStore class for persisting trust edges
to the database, along with the singleton accessor function.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

from .edges import TrustEdge

# Singleton store instance
_default_store: TrustGraphStore | None = None
_default_store_lock = threading.Lock()


class TrustGraphStore:
    """Database storage for trust graph edges.

    Provides CRUD operations for TrustEdge persistence.
    """

    def __init__(self) -> None:
        """Initialize the store."""
        pass

    def add_edge(self, edge: TrustEdge) -> TrustEdge:
        """Add or update a trust edge.

        Uses upsert semantics - updates if edge exists, inserts otherwise.

        Args:
            edge: The trust edge to store

        Returns:
            The stored edge with ID and timestamps populated
        """

        from valence.core.db import get_cursor

        now = datetime.now(UTC)

        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO trust_edges (
                    id, source_did, target_did,
                    competence, integrity, confidentiality, judgment,
                    domain, can_delegate, delegation_depth,
                    created_at, updated_at, expires_at
                )
                VALUES (
                    COALESCE(%s, gen_random_uuid()), %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (source_did, target_did, COALESCE(domain, ''))
                DO UPDATE SET
                    competence = EXCLUDED.competence,
                    integrity = EXCLUDED.integrity,
                    confidentiality = EXCLUDED.confidentiality,
                    judgment = EXCLUDED.judgment,
                    can_delegate = EXCLUDED.can_delegate,
                    delegation_depth = EXCLUDED.delegation_depth,
                    updated_at = EXCLUDED.updated_at,
                    expires_at = EXCLUDED.expires_at
                RETURNING id, created_at, updated_at
                """,
                (
                    str(edge.id) if edge.id else None,
                    edge.source_did,
                    edge.target_did,
                    edge.competence,
                    edge.integrity,
                    edge.confidentiality,
                    edge.judgment,
                    edge.domain,
                    edge.can_delegate,
                    edge.delegation_depth,
                    edge.created_at or now,
                    now,
                    edge.expires_at,
                ),
            )
            row = cur.fetchone()

            edge.id = row["id"]
            edge.created_at = row["created_at"]
            edge.updated_at = row["updated_at"]

        return edge

    def get_edge(
        self,
        source_did: str,
        target_did: str,
        domain: str | None = None,
    ) -> TrustEdge | None:
        """Get a specific trust edge.

        Args:
            source_did: The trusting DID
            target_did: The trusted DID
            domain: Optional domain filter

        Returns:
            The trust edge if found, None otherwise
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            if domain is not None:
                cur.execute(
                    """
                    SELECT id, source_did, target_did,
                           competence, integrity, confidentiality, judgment,
                           domain, can_delegate, delegation_depth,
                           created_at, updated_at, expires_at
                    FROM trust_edges
                    WHERE source_did = %s AND target_did = %s AND domain = %s
                    """,
                    (source_did, target_did, domain),
                )
            else:
                cur.execute(
                    """
                    SELECT id, source_did, target_did,
                           competence, integrity, confidentiality, judgment,
                           domain, can_delegate, delegation_depth,
                           created_at, updated_at, expires_at
                    FROM trust_edges
                    WHERE source_did = %s AND target_did = %s AND domain IS NULL
                    """,
                    (source_did, target_did),
                )

            row = cur.fetchone()
            if row is None:
                return None

            return TrustEdge(
                id=row["id"],
                source_did=row["source_did"],
                target_did=row["target_did"],
                competence=row["competence"],
                integrity=row["integrity"],
                confidentiality=row["confidentiality"],
                judgment=row["judgment"],
                domain=row["domain"],
                can_delegate=row.get("can_delegate", False),
                delegation_depth=row.get("delegation_depth", 0),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                expires_at=row["expires_at"],
            )

    def get_edges_from(
        self,
        source_did: str,
        domain: str | None = None,
        include_expired: bool = False,
    ) -> list[TrustEdge]:
        """Get all trust edges from a DID.

        Args:
            source_did: The trusting DID
            domain: Optional domain filter
            include_expired: Whether to include expired edges

        Returns:
            List of trust edges
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            query = """
                SELECT id, source_did, target_did,
                       competence, integrity, confidentiality, judgment,
                       domain, can_delegate, delegation_depth,
                       created_at, updated_at, expires_at
                FROM trust_edges
                WHERE source_did = %s
            """
            params: list[Any] = [source_did]

            if domain is not None:
                query += " AND domain = %s"
                params.append(domain)

            if not include_expired:
                query += " AND (expires_at IS NULL OR expires_at > NOW())"

            cur.execute(query, params)
            rows = cur.fetchall()

            return [
                TrustEdge(
                    id=row["id"],
                    source_did=row["source_did"],
                    target_did=row["target_did"],
                    competence=row["competence"],
                    integrity=row["integrity"],
                    confidentiality=row["confidentiality"],
                    judgment=row["judgment"],
                    domain=row["domain"],
                    can_delegate=row.get("can_delegate", False),
                    delegation_depth=row.get("delegation_depth", 0),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    expires_at=row["expires_at"],
                )
                for row in rows
            ]

    def get_edges_to(
        self,
        target_did: str,
        domain: str | None = None,
        include_expired: bool = False,
    ) -> list[TrustEdge]:
        """Get all trust edges to a DID.

        Args:
            target_did: The trusted DID
            domain: Optional domain filter
            include_expired: Whether to include expired edges

        Returns:
            List of trust edges
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            query = """
                SELECT id, source_did, target_did,
                       competence, integrity, confidentiality, judgment,
                       domain, can_delegate, delegation_depth,
                       created_at, updated_at, expires_at
                FROM trust_edges
                WHERE target_did = %s
            """
            params: list[Any] = [target_did]

            if domain is not None:
                query += " AND domain = %s"
                params.append(domain)

            if not include_expired:
                query += " AND (expires_at IS NULL OR expires_at > NOW())"

            cur.execute(query, params)
            rows = cur.fetchall()

            return [
                TrustEdge(
                    id=row["id"],
                    source_did=row["source_did"],
                    target_did=row["target_did"],
                    competence=row["competence"],
                    integrity=row["integrity"],
                    confidentiality=row["confidentiality"],
                    judgment=row["judgment"],
                    domain=row["domain"],
                    can_delegate=row.get("can_delegate", False),
                    delegation_depth=row.get("delegation_depth", 0),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    expires_at=row["expires_at"],
                )
                for row in rows
            ]

    def delete_edge(
        self,
        source_did: str,
        target_did: str,
        domain: str | None = None,
    ) -> bool:
        """Delete a trust edge.

        Args:
            source_did: The trusting DID
            target_did: The trusted DID
            domain: Optional domain filter

        Returns:
            True if an edge was deleted, False otherwise
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            if domain is not None:
                cur.execute(
                    """
                    DELETE FROM trust_edges
                    WHERE source_did = %s AND target_did = %s AND domain = %s
                    RETURNING id
                    """,
                    (source_did, target_did, domain),
                )
            else:
                cur.execute(
                    """
                    DELETE FROM trust_edges
                    WHERE source_did = %s AND target_did = %s AND domain IS NULL
                    RETURNING id
                    """,
                    (source_did, target_did),
                )

            row = cur.fetchone()
            return row is not None

    def delete_edges_from(self, source_did: str) -> int:
        """Delete all trust edges from a DID.

        Args:
            source_did: The trusting DID

        Returns:
            Number of edges deleted
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute(
                """
                DELETE FROM trust_edges
                WHERE source_did = %s
                RETURNING id
                """,
                (source_did,),
            )
            rows = cur.fetchall()
            return len(rows)

    def delete_edges_to(self, target_did: str) -> int:
        """Delete all trust edges to a DID.

        Args:
            target_did: The trusted DID

        Returns:
            Number of edges deleted
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute(
                """
                DELETE FROM trust_edges
                WHERE target_did = %s
                RETURNING id
                """,
                (target_did,),
            )
            rows = cur.fetchall()
            return len(rows)

    def cleanup_expired(self) -> int:
        """Delete all expired trust edges.

        Returns:
            Number of edges deleted
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute(
                """
                DELETE FROM trust_edges
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                RETURNING id
                """,
            )
            rows = cur.fetchall()
            return len(rows)

    def count_edges(
        self,
        source_did: str | None = None,
        target_did: str | None = None,
        domain: str | None = None,
    ) -> int:
        """Count trust edges with optional filters.

        Args:
            source_did: Optional source filter
            target_did: Optional target filter
            domain: Optional domain filter

        Returns:
            Number of matching edges
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            query = "SELECT COUNT(*) as count FROM trust_edges WHERE 1=1"
            params: list[Any] = []

            if source_did is not None:
                query += " AND source_did = %s"
                params.append(source_did)

            if target_did is not None:
                query += " AND target_did = %s"
                params.append(target_did)

            if domain is not None:
                query += " AND domain = %s"
                params.append(domain)

            cur.execute(query, params)
            row = cur.fetchone()
            return row["count"]


def get_trust_graph_store() -> TrustGraphStore:
    """Get the singleton TrustGraphStore instance.

    Thread-safe initialization using double-checked locking pattern.

    Returns:
        The shared TrustGraphStore instance
    """
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            # Double-check after acquiring lock
            if _default_store is None:
                _default_store = TrustGraphStore()
    return _default_store
