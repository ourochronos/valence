"""Tests for MLS group sharing (#342).

Tests cover:
1. create_sharing_group basic
2. create_sharing_group with initial members
3. create_sharing_group persists to DB
4. add_group_member
5. add_group_member duplicate rejected
6. add_group_member group not found
7. remove_group_member
8. remove_group_member group not found
9. rotate_group_keys
10. rotate_group_keys group not found
11. get_group
12. get_group not found
13. list_groups_for_member
14. SharingGroup serialization
15. GroupMembership serialization
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from valence.core.group_sharing import (
    GroupMembership,
    SharingGroup,
    add_group_member,
    create_sharing_group,
    get_group,
    list_groups_for_member,
    remove_group_member,
    rotate_group_keys,
)


@pytest.fixture
def mock_cur():
    return MagicMock()


@pytest.fixture
def mock_mls():
    backend = MagicMock()
    with patch("valence.core.group_sharing._get_mls_backend", return_value=backend):
        yield backend


class TestCreateSharingGroup:
    """Test group creation."""

    def test_basic_creation(self, mock_cur, mock_mls):
        group = create_sharing_group(mock_cur, "Test Group", "did:valence:alice")
        assert group.name == "Test Group"
        assert group.creator_did == "did:valence:alice"
        assert group.intent == "work_with_me"
        assert "did:valence:alice" in group.members
        assert group.epoch == 0
        mock_mls.create_group.assert_called_once()

    def test_with_initial_members(self, mock_cur, mock_mls):
        group = create_sharing_group(
            mock_cur, "Team", "did:alice",
            initial_members=["did:bob", "did:carol"],
        )
        assert len(group.members) == 3
        assert "did:bob" in group.members
        assert "did:carol" in group.members
        assert mock_mls.add_member.call_count == 2

    def test_persists_to_db(self, mock_cur, mock_mls):
        create_sharing_group(mock_cur, "Group", "did:alice")
        # Should INSERT into sharing_groups + sharing_group_members (creator)
        calls = mock_cur.execute.call_args_list
        sql_calls = [c[0][0] for c in calls]
        assert any("INSERT INTO sharing_groups" in s for s in sql_calls)
        assert any("INSERT INTO sharing_group_members" in s for s in sql_calls)

    def test_with_members_persists_all(self, mock_cur, mock_mls):
        create_sharing_group(
            mock_cur, "Group", "did:alice",
            initial_members=["did:bob"],
        )
        # Should have: 1 group INSERT + 1 creator member INSERT + 1 member INSERT = 3
        member_inserts = [
            c for c in mock_cur.execute.call_args_list
            if "INSERT INTO sharing_group_members" in c[0][0]
        ]
        assert len(member_inserts) == 2  # creator + bob


class TestAddGroupMember:
    """Test member addition."""

    def test_adds_member(self, mock_cur, mock_mls):
        mock_cur.fetchone.side_effect = [
            {"id": "g1", "mls_group_id": "mls-g1", "epoch": 0},  # group lookup
            None,  # not already member
        ]
        result = add_group_member(mock_cur, "g1", "did:bob", "did:alice")
        assert result is not None
        assert result.member_did == "did:bob"
        assert result.role == "member"
        mock_mls.add_member.assert_called_once()

    def test_rejects_duplicate(self, mock_cur, mock_mls):
        mock_cur.fetchone.side_effect = [
            {"id": "g1", "mls_group_id": "mls-g1", "epoch": 0},  # group
            {"exists": True},  # already member
        ]
        result = add_group_member(mock_cur, "g1", "did:bob", "did:alice")
        assert result is None

    def test_group_not_found(self, mock_cur, mock_mls):
        mock_cur.fetchone.return_value = None
        result = add_group_member(mock_cur, "missing", "did:bob", "did:alice")
        assert result is None

    def test_advances_epoch(self, mock_cur, mock_mls):
        mock_cur.fetchone.side_effect = [
            {"id": "g1", "mls_group_id": "mls-g1", "epoch": 2},
            None,  # not already member
        ]
        add_group_member(mock_cur, "g1", "did:bob", "did:alice")
        # Check epoch was updated to 3
        update_calls = [
            c for c in mock_cur.execute.call_args_list
            if "UPDATE sharing_groups SET epoch" in c[0][0]
        ]
        assert len(update_calls) == 1
        assert update_calls[0][0][1][0] == 3  # new epoch


class TestRemoveGroupMember:
    """Test member removal."""

    def test_removes_member(self, mock_cur, mock_mls):
        mock_cur.fetchone.return_value = {
            "id": "g1", "mls_group_id": "mls-g1", "epoch": 1,
        }
        result = remove_group_member(mock_cur, "g1", "did:bob", "did:alice")
        assert result is True
        mock_mls.remove_member.assert_called_once()

    def test_group_not_found(self, mock_cur, mock_mls):
        mock_cur.fetchone.return_value = None
        result = remove_group_member(mock_cur, "missing", "did:bob", "did:alice")
        assert result is False

    def test_advances_epoch_on_removal(self, mock_cur, mock_mls):
        mock_cur.fetchone.return_value = {
            "id": "g1", "mls_group_id": "mls-g1", "epoch": 5,
        }
        remove_group_member(mock_cur, "g1", "did:bob", "did:alice")
        update_calls = [
            c for c in mock_cur.execute.call_args_list
            if "UPDATE sharing_groups SET epoch" in c[0][0]
        ]
        assert len(update_calls) == 1
        assert update_calls[0][0][1][0] == 6


class TestRotateGroupKeys:
    """Test key rotation."""

    def test_rotates_keys(self, mock_cur, mock_mls):
        mock_cur.fetchone.return_value = {
            "id": "g1", "mls_group_id": "mls-g1", "epoch": 3,
        }
        new_epoch = rotate_group_keys(mock_cur, "g1", "did:alice")
        assert new_epoch == 4
        mock_mls.update_keys.assert_called_once()

    def test_group_not_found(self, mock_cur, mock_mls):
        mock_cur.fetchone.return_value = None
        result = rotate_group_keys(mock_cur, "missing", "did:alice")
        assert result == -1


class TestGetGroup:
    """Test group retrieval."""

    def test_returns_group(self, mock_cur, mock_mls):
        now = datetime.now(timezone.utc)
        mock_cur.fetchone.return_value = {
            "id": "g1", "name": "Team", "creator_did": "did:alice",
            "mls_group_id": "mls-g1", "intent": "work_with_me",
            "epoch": 2, "created_at": now,
        }
        mock_cur.fetchall.return_value = [
            {"member_did": "did:alice"},
            {"member_did": "did:bob"},
        ]
        group = get_group(mock_cur, "g1")
        assert group is not None
        assert group.name == "Team"
        assert len(group.members) == 2

    def test_not_found(self, mock_cur, mock_mls):
        mock_cur.fetchone.return_value = None
        assert get_group(mock_cur, "missing") is None


class TestListGroupsForMember:
    """Test member group listing."""

    def test_returns_groups(self, mock_cur, mock_mls):
        now = datetime.now(timezone.utc)
        mock_cur.fetchall.side_effect = [
            # First fetchall: groups
            [
                {"id": "g1", "name": "Team A", "creator_did": "did:alice",
                 "mls_group_id": "mls-g1", "intent": "work_with_me",
                 "epoch": 0, "created_at": now},
            ],
            # Second fetchall: members of g1
            [{"member_did": "did:alice"}, {"member_did": "did:bob"}],
        ]
        groups = list_groups_for_member(mock_cur, "did:alice")
        assert len(groups) == 1
        assert groups[0].name == "Team A"


class TestSerialization:
    """Test dataclass serialization."""

    def test_sharing_group_to_dict(self):
        group = SharingGroup(
            id="g1", name="Test", creator_did="did:alice",
            mls_group_id="mls-g1", members=["did:alice"],
        )
        d = group.to_dict()
        assert d["name"] == "Test"
        assert d["members"] == ["did:alice"]

    def test_group_membership_to_dict(self):
        m = GroupMembership(group_id="g1", member_did="did:bob", role="member")
        d = m.to_dict()
        assert d["member_did"] == "did:bob"
        assert d["role"] == "member"
