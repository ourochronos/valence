"""Supersession chain integrity verification (#354).

Walks all supersession chains in the beliefs table and checks for:
1. Bidirectional consistency: supersedes_id ↔ superseded_by_id match
2. Monotonic timestamps: created_at strictly increases along chain
3. Cycle detection: no belief appears twice in a chain
4. Terminal status: chain head is 'active', all others 'superseded'
5. Orphan detection: superseded beliefs without a superseded_by_id link
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChainIssue:
    """A single integrity issue found in a supersession chain."""

    chain_id: str  # UUID of the chain head (or root)
    issue_type: str  # bidirectional_mismatch, non_monotonic, cycle, bad_terminal, orphan
    description: str
    belief_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "issue_type": self.issue_type,
            "description": self.description,
            "belief_ids": self.belief_ids,
        }


@dataclass
class ChainReport:
    """Full chain verification report."""

    total_chains: int = 0
    total_beliefs_in_chains: int = 0
    healthy_chains: int = 0
    issues: list[ChainIssue] = field(default_factory=list)

    @property
    def unhealthy_chains(self) -> int:
        chain_ids = {i.chain_id for i in self.issues}
        return len(chain_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_chains": self.total_chains,
            "total_beliefs_in_chains": self.total_beliefs_in_chains,
            "healthy_chains": self.healthy_chains,
            "unhealthy_chains": self.unhealthy_chains,
            "issues": [i.to_dict() for i in self.issues],
        }

    def __str__(self) -> str:
        lines = [
            "Chain Verification Report",
            f"  Total chains: {self.total_chains}",
            f"  Beliefs in chains: {self.total_beliefs_in_chains}",
            f"  Healthy: {self.healthy_chains}",
            f"  Unhealthy: {self.unhealthy_chains}",
            f"  Issues: {len(self.issues)}",
        ]
        for issue in self.issues:
            lines.append(f"    [{issue.issue_type}] {issue.description}")
        return "\n".join(lines)


def verify_chains(cur, limit: int | None = None) -> ChainReport:
    """Verify all supersession chains in the database.

    Finds chain heads (beliefs that have supersedes_id but no superseded_by_id,
    i.e. the newest in each chain), then walks backward checking integrity.

    Args:
        cur: Database cursor
        limit: Max chains to check (None = all)

    Returns:
        ChainReport with all findings
    """
    report = ChainReport()

    # Find all chain heads: beliefs that are part of a chain (have supersedes_id
    # OR have superseded_by_id) but are the terminal node (superseded_by_id IS NULL).
    # These are the "current" versions.
    query = """
        SELECT id, content, status, supersedes_id, superseded_by_id, created_at
        FROM articles
        WHERE superseded_by_id IS NULL
          AND supersedes_id IS NOT NULL
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    cur.execute(query)
    chain_heads = cur.fetchall()

    # Also find orphans: superseded beliefs whose superseded_by_id points to a non-existent belief
    cur.execute("""
        SELECT b.id, b.superseded_by_id
        FROM articles b
        LEFT JOIN beliefs b2 ON b.superseded_by_id = b2.id
        WHERE b.superseded_by_id IS NOT NULL
          AND b2.id IS NULL
    """)
    orphan_rows = cur.fetchall()
    for row in orphan_rows:
        report.issues.append(ChainIssue(
            chain_id=str(row["id"]),
            issue_type="orphan",
            description=f"Belief {str(row['id'])[:8]} has superseded_by_id={str(row['superseded_by_id'])[:8]} but that belief does not exist",
            belief_ids=[str(row["id"])],
        ))

    beliefs_in_chains = set()

    for head in chain_heads:
        head_id = str(head["id"])
        chain_beliefs = []
        visited = set()
        current_id = head_id
        chain_ok = True

        # Walk backward through the chain
        while current_id:
            if current_id in visited:
                report.issues.append(ChainIssue(
                    chain_id=head_id,
                    issue_type="cycle",
                    description=f"Cycle detected: belief {current_id[:8]} appears twice in chain",
                    belief_ids=list(visited),
                ))
                chain_ok = False
                break

            visited.add(current_id)
            cur.execute(
                "SELECT id, content, status, supersedes_id, superseded_by_id, created_at FROM articles WHERE id = %s",
                (current_id,),
            )
            row = cur.fetchone()
            if not row:
                report.issues.append(ChainIssue(
                    chain_id=head_id,
                    issue_type="broken_link",
                    description=f"Belief {current_id[:8]} referenced in chain but does not exist",
                    belief_ids=list(visited),
                ))
                chain_ok = False
                break

            chain_beliefs.append(row)

            # Check bidirectional consistency
            supersedes_id = str(row["supersedes_id"]) if row["supersedes_id"] else None
            str(row["superseded_by_id"]) if row["superseded_by_id"] else None

            if supersedes_id:
                # The older belief should point back to us via superseded_by_id
                cur.execute("SELECT superseded_by_id FROM articles WHERE id = %s", (supersedes_id,))
                older = cur.fetchone()
                if older:
                    older_superseded_by = str(older["superseded_by_id"]) if older["superseded_by_id"] else None
                    if older_superseded_by != current_id:
                        report.issues.append(ChainIssue(
                            chain_id=head_id,
                            issue_type="bidirectional_mismatch",
                            description=(
                                f"Belief {current_id[:8]} supersedes {supersedes_id[:8]}, "
                                f"but {supersedes_id[:8]}.superseded_by_id={older_superseded_by[:8] if older_superseded_by else 'NULL'} "
                                f"(expected {current_id[:8]})"
                            ),
                            belief_ids=[current_id, supersedes_id],
                        ))
                        chain_ok = False

            current_id = supersedes_id

        # Check monotonic timestamps (chain_beliefs is head-first, so created_at should decrease)
        for i in range(len(chain_beliefs) - 1):
            newer = chain_beliefs[i]
            older = chain_beliefs[i + 1]
            if newer["created_at"] <= older["created_at"]:
                report.issues.append(ChainIssue(
                    chain_id=head_id,
                    issue_type="non_monotonic",
                    description=(
                        f"Timestamp not monotonic: {str(newer['id'])[:8]} ({newer['created_at'].isoformat()}) "
                        f"<= {str(older['id'])[:8]} ({older['created_at'].isoformat()})"
                    ),
                    belief_ids=[str(newer["id"]), str(older["id"])],
                ))
                chain_ok = False

        # Check terminal status: head should be 'active', others should be 'superseded'
        if chain_beliefs:
            head_belief = chain_beliefs[0]
            if head_belief["status"] != "active":
                report.issues.append(ChainIssue(
                    chain_id=head_id,
                    issue_type="bad_terminal",
                    description=f"Chain head {head_id[:8]} has status='{head_belief['status']}' (expected 'active')",
                    belief_ids=[head_id],
                ))
                chain_ok = False

            for belief in chain_beliefs[1:]:
                if belief["status"] != "superseded":
                    report.issues.append(ChainIssue(
                        chain_id=head_id,
                        issue_type="bad_terminal",
                        description=(
                            f"Non-head belief {str(belief['id'])[:8]} has status='{belief['status']}' (expected 'superseded')"
                        ),
                        belief_ids=[str(belief["id"])],
                    ))
                    chain_ok = False

        beliefs_in_chains.update(visited)
        report.total_chains += 1
        if chain_ok:
            report.healthy_chains += 1

    report.total_beliefs_in_chains = len(beliefs_in_chains)
    return report
