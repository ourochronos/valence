"""MLS group key management for work_with_me sharing (#342).

Manages MLS-backed sharing groups for the work_with_me intent.
Each group maintains forward secrecy and post-compromise security
via epoch-based key ratcheting.

Groups are linked to sharing intents and members are tracked
by DID with their MLS key packages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from our_crypto import create_mls_backend

logger = logging.getLogger(__name__)

DEFAULT_MLS_BACKEND = "hkdf"


@dataclass
class SharingGroup:
    """A sharing group backed by MLS group encryption."""

    id: str
    name: str
    creator_did: str
    mls_group_id: str
    intent: str = "work_with_me"
    members: list[str] = field(default_factory=list)
    epoch: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "creator_did": self.creator_did,
            "mls_group_id": self.mls_group_id,
            "intent": self.intent,
            "members": self.members,
            "epoch": self.epoch,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class GroupMembership:
    """A member's status in a sharing group."""

    group_id: str
    member_did: str
    role: str = "member"  # creator, admin, member
    joined_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "member_did": self.member_did,
            "role": self.role,
            "joined_at": self.joined_at.isoformat(),
        }


def _get_mls_backend():
    """Get the MLS backend. Separate function for test patching."""
    return create_mls_backend(DEFAULT_MLS_BACKEND)


def create_sharing_group(
    cur,
    name: str,
    creator_did: str,
    initial_members: list[str] | None = None,
) -> SharingGroup:
    """Create a new MLS-backed sharing group.

    Args:
        cur: Database cursor.
        name: Human-readable group name.
        creator_did: DID of the group creator.
        initial_members: Optional list of DIDs to add immediately.

    Returns:
        SharingGroup with MLS group initialized.
    """
    group_id = str(uuid4())
    mls_group_id = f"mls-{group_id}"

    backend = _get_mls_backend()
    backend.create_group(
        group_id=mls_group_id,
        creator_id=creator_did,
        credential=creator_did.encode(),
    )

    members = [creator_did]

    if initial_members:
        for member_did in initial_members:
            backend.add_member(
                group_id=mls_group_id,
                member_id=member_did,
                key_package=member_did.encode(),
                credential=member_did.encode(),
            )
            members.append(member_did)

    group = SharingGroup(
        id=group_id,
        name=name,
        creator_did=creator_did,
        mls_group_id=mls_group_id,
        members=members,
        epoch=0,
    )

    cur.execute(
        """
        INSERT INTO sharing_groups (id, name, creator_did, mls_group_id, intent, epoch, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (group.id, group.name, group.creator_did, group.mls_group_id,
         group.intent, group.epoch, group.created_at),
    )

    # Insert creator membership
    cur.execute(
        """
        INSERT INTO sharing_group_members (group_id, member_did, role, joined_at)
        VALUES (%s, %s, %s, %s)
        """,
        (group.id, creator_did, "creator", group.created_at),
    )

    # Insert initial members
    if initial_members:
        for member_did in initial_members:
            cur.execute(
                """
                INSERT INTO sharing_group_members (group_id, member_did, role, joined_at)
                VALUES (%s, %s, %s, %s)
                """,
                (group.id, member_did, "member", group.created_at),
            )

    return group


def add_group_member(
    cur,
    group_id: str,
    member_did: str,
    added_by: str,
) -> GroupMembership | None:
    """Add a member to a sharing group.

    Args:
        cur: Database cursor.
        group_id: UUID of the sharing group.
        member_did: DID of the member to add.
        added_by: DID of the user adding the member.

    Returns:
        GroupMembership if added, None if group not found or already a member.
    """
    cur.execute("SELECT * FROM sharing_groups WHERE id = %s", (group_id,))
    group_row = cur.fetchone()
    if not group_row:
        return None

    # Check not already a member
    cur.execute(
        "SELECT 1 FROM sharing_group_members WHERE group_id = %s AND member_did = %s",
        (group_id, member_did),
    )
    if cur.fetchone():
        return None  # Already a member

    mls_group_id = group_row["mls_group_id"]
    backend = _get_mls_backend()
    backend.add_member(
        group_id=mls_group_id,
        member_id=member_did,
        key_package=member_did.encode(),
        credential=member_did.encode(),
    )

    # Advance epoch
    new_epoch = group_row["epoch"] + 1
    cur.execute(
        "UPDATE sharing_groups SET epoch = %s WHERE id = %s",
        (new_epoch, group_id),
    )

    now = datetime.now(UTC)
    membership = GroupMembership(
        group_id=group_id,
        member_did=member_did,
        joined_at=now,
    )

    cur.execute(
        """
        INSERT INTO sharing_group_members (group_id, member_did, role, joined_at)
        VALUES (%s, %s, %s, %s)
        """,
        (group_id, member_did, "member", now),
    )

    return membership


def remove_group_member(
    cur,
    group_id: str,
    member_did: str,
    removed_by: str,
) -> bool:
    """Remove a member from a sharing group.

    Args:
        cur: Database cursor.
        group_id: UUID of the sharing group.
        member_did: DID of the member to remove.
        removed_by: DID of the user removing the member.

    Returns:
        True if removed, False if not found.
    """
    cur.execute("SELECT * FROM sharing_groups WHERE id = %s", (group_id,))
    group_row = cur.fetchone()
    if not group_row:
        return False

    mls_group_id = group_row["mls_group_id"]
    backend = _get_mls_backend()
    backend.remove_member(
        group_id=mls_group_id,
        member_id=member_did,
        remover_id=removed_by,
    )

    # Advance epoch (key rotation on removal)
    new_epoch = group_row["epoch"] + 1
    cur.execute(
        "UPDATE sharing_groups SET epoch = %s WHERE id = %s",
        (new_epoch, group_id),
    )

    cur.execute(
        "DELETE FROM sharing_group_members WHERE group_id = %s AND member_did = %s",
        (group_id, member_did),
    )

    return True


def rotate_group_keys(
    cur,
    group_id: str,
    rotated_by: str,
) -> int:
    """Rotate keys for a sharing group (advance epoch).

    Args:
        cur: Database cursor.
        group_id: UUID of the sharing group.
        rotated_by: DID of the member requesting rotation.

    Returns:
        New epoch number, or -1 if group not found.
    """
    cur.execute("SELECT * FROM sharing_groups WHERE id = %s", (group_id,))
    group_row = cur.fetchone()
    if not group_row:
        return -1

    mls_group_id = group_row["mls_group_id"]
    backend = _get_mls_backend()
    backend.update_keys(
        group_id=mls_group_id,
        member_id=rotated_by,
    )

    new_epoch = group_row["epoch"] + 1
    cur.execute(
        "UPDATE sharing_groups SET epoch = %s WHERE id = %s",
        (new_epoch, group_id),
    )

    return new_epoch


def get_group(cur, group_id: str) -> SharingGroup | None:
    """Get a sharing group by ID.

    Args:
        cur: Database cursor.
        group_id: UUID of the sharing group.

    Returns:
        SharingGroup or None if not found.
    """
    cur.execute("SELECT * FROM sharing_groups WHERE id = %s", (group_id,))
    row = cur.fetchone()
    if not row:
        return None

    cur.execute(
        "SELECT member_did FROM sharing_group_members WHERE group_id = %s",
        (group_id,),
    )
    members = [r["member_did"] for r in cur.fetchall()]

    return SharingGroup(
        id=row["id"],
        name=row["name"],
        creator_did=row["creator_did"],
        mls_group_id=row["mls_group_id"],
        intent=row.get("intent", "work_with_me"),
        members=members,
        epoch=row["epoch"],
        created_at=row["created_at"],
    )


def list_groups_for_member(cur, member_did: str) -> list[SharingGroup]:
    """List all sharing groups a member belongs to.

    Args:
        cur: Database cursor.
        member_did: DID to look up.

    Returns:
        List of SharingGroup objects.
    """
    cur.execute(
        """
        SELECT sg.* FROM sharing_groups sg
        JOIN sharing_group_members sgm ON sg.id = sgm.group_id
        WHERE sgm.member_did = %s
        ORDER BY sg.created_at DESC
        """,
        (member_did,),
    )
    rows = cur.fetchall()

    groups = []
    for row in rows:
        cur.execute(
            "SELECT member_did FROM sharing_group_members WHERE group_id = %s",
            (row["id"],),
        )
        members = [r["member_did"] for r in cur.fetchall()]
        groups.append(SharingGroup(
            id=row["id"],
            name=row["name"],
            creator_did=row["creator_did"],
            mls_group_id=row["mls_group_id"],
            intent=row.get("intent", "work_with_me"),
            members=members,
            epoch=row["epoch"],
            created_at=row["created_at"],
        ))

    return groups
