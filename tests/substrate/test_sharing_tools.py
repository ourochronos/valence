"""Tests for sharing tool handlers (belief_share, belief_shares_list, belief_share_revoke).

Also tests belief_create updates for visibility and sharing_intent params.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.substrate.tools import (
    SUBSTRATE_HANDLERS,
    SUBSTRATE_TOOLS,
    belief_create,
    belief_share,
    belief_share_revoke,
    belief_shares_list,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_cursor():
    """Mock database cursor."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Mock the get_cursor context manager."""

    @contextmanager
    def _mock_get_cursor(dict_cursor: bool = True) -> Generator:
        yield mock_cursor

    with patch("valence.substrate.tools._common.get_cursor", _mock_get_cursor):
        yield mock_cursor


@pytest.fixture
def sample_belief_row():
    """Create a sample belief row."""

    def _factory(id=None, content="Test belief content", **kwargs):
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "content": content,
            "confidence": json.dumps(kwargs.get("confidence", {"overall": 0.7})),
            "domain_path": kwargs.get("domain_path", ["test", "domain"]),
            "valid_from": kwargs.get("valid_from"),
            "valid_until": kwargs.get("valid_until"),
            "created_at": kwargs.get("created_at", now),
            "modified_at": kwargs.get("modified_at", now),
            "source_id": kwargs.get("source_id"),
            "extraction_method": kwargs.get("extraction_method"),
            "supersedes_id": kwargs.get("supersedes_id"),
            "superseded_by_id": kwargs.get("superseded_by_id"),
            "status": kwargs.get("status", "active"),
            "opt_out_federation": kwargs.get("opt_out_federation", False),
        }

    return _factory


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================


class TestSharingToolDefinitions:
    """Tests for sharing tool definitions."""

    def test_sharing_tools_in_substrate_tools(self):
        """Sharing tools are registered in SUBSTRATE_TOOLS."""
        tool_names = [t.name for t in SUBSTRATE_TOOLS]
        assert "belief_share" in tool_names
        assert "belief_shares_list" in tool_names
        assert "belief_share_revoke" in tool_names

    def test_sharing_tools_have_handlers(self):
        """All sharing tools have corresponding handlers."""
        assert "belief_share" in SUBSTRATE_HANDLERS
        assert "belief_shares_list" in SUBSTRATE_HANDLERS
        assert "belief_share_revoke" in SUBSTRATE_HANDLERS

    def test_belief_create_has_visibility_param(self):
        """belief_create tool definition includes visibility parameter."""
        tool = next(t for t in SUBSTRATE_TOOLS if t.name == "belief_create")
        props = tool.inputSchema["properties"]
        assert "visibility" in props
        assert props["visibility"]["enum"] == ["private", "federated", "public"]

    def test_belief_create_has_sharing_intent_param(self):
        """belief_create tool definition includes sharing_intent parameter."""
        tool = next(t for t in SUBSTRATE_TOOLS if t.name == "belief_create")
        props = tool.inputSchema["properties"]
        assert "sharing_intent" in props
        assert "know_me" in props["sharing_intent"]["enum"]

    def test_belief_share_required_params(self):
        """belief_share requires belief_id and recipient_did."""
        tool = next(t for t in SUBSTRATE_TOOLS if t.name == "belief_share")
        assert "belief_id" in tool.inputSchema["required"]
        assert "recipient_did" in tool.inputSchema["required"]


# =============================================================================
# BELIEF_SHARE
# =============================================================================


class TestBeliefShare:
    """Tests for belief_share handler."""

    def test_share_creates_consent_chain_and_share(self, mock_get_cursor, sample_belief_row):
        """belief_share creates consent_chain + share records and returns IDs."""
        belief_id = uuid4()
        consent_chain_id = uuid4()
        share_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            # 1. Belief lookup
            {"id": belief_id, "content": "Test belief"},
            # 2. consent_chain INSERT RETURNING id
            {"id": consent_chain_id},
            # 3. share INSERT RETURNING id
            {"id": share_id},
        ]

        result = belief_share(
            belief_id=str(belief_id),
            recipient_did="did:key:alice",
            intent="know_me",
        )

        assert result["success"] is True
        assert result["share_id"] == str(share_id)
        assert result["consent_chain_id"] == str(consent_chain_id)
        assert result["recipient"] == "did:key:alice"
        assert result["intent"] == "know_me"

    def test_share_nonexistent_belief(self, mock_get_cursor):
        """belief_share returns error for nonexistent belief."""
        mock_get_cursor.fetchone.return_value = None

        result = belief_share(belief_id=str(uuid4()), recipient_did="did:key:alice")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_share_with_know_me_intent(self, mock_get_cursor):
        """know_me intent produces DIRECT + CRYPTOGRAPHIC policy."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Test"},
            {"id": uuid4()},
            {"id": uuid4()},
        ]

        result = belief_share(
            belief_id=str(belief_id),
            recipient_did="did:key:alice",
            intent="know_me",
        )

        assert result["success"] is True
        assert result["policy"]["level"] == "direct"
        assert result["policy"]["enforcement"] == "cryptographic"

    def test_share_with_work_with_me_intent(self, mock_get_cursor):
        """work_with_me intent produces BOUNDED + POLICY."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Test"},
            {"id": uuid4()},
            {"id": uuid4()},
        ]

        result = belief_share(
            belief_id=str(belief_id),
            recipient_did="did:key:alice",
            intent="work_with_me",
        )

        assert result["success"] is True
        assert result["policy"]["level"] == "bounded"
        assert result["policy"]["enforcement"] == "policy"

    def test_share_with_learn_from_me_intent(self, mock_get_cursor):
        """learn_from_me intent produces CASCADING + POLICY."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Test"},
            {"id": uuid4()},
            {"id": uuid4()},
        ]

        result = belief_share(
            belief_id=str(belief_id),
            recipient_did="did:key:alice",
            intent="learn_from_me",
        )

        assert result["success"] is True
        assert result["policy"]["level"] == "cascading"

    def test_share_with_use_this_intent(self, mock_get_cursor):
        """use_this intent produces PUBLIC + HONOR."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Test"},
            {"id": uuid4()},
            {"id": uuid4()},
        ]

        result = belief_share(
            belief_id=str(belief_id),
            recipient_did="did:key:alice",
            intent="use_this",
        )

        assert result["success"] is True
        assert result["policy"]["level"] == "public"
        assert result["policy"]["enforcement"] == "honor"

    def test_share_inserts_consent_chain_with_policy(self, mock_get_cursor):
        """Consent chain INSERT includes the IntentConfig as origin_policy."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Test"},
            {"id": uuid4()},
            {"id": uuid4()},
        ]

        belief_share(
            belief_id=str(belief_id),
            recipient_did="did:key:alice",
            intent="know_me",
        )

        # Check the consent_chains INSERT call
        calls = mock_get_cursor.execute.call_args_list
        consent_chain_call = calls[1]  # Second execute call (after belief lookup)
        sql = consent_chain_call[0][0]
        params = consent_chain_call[0][1]

        assert "consent_chains" in sql
        # origin_policy param should be JSON with intent info
        policy_json = json.loads(params[2])
        assert policy_json["intent"] == "know_me"

    def test_share_updates_belief_share_policy(self, mock_get_cursor):
        """belief_share updates the belief's share_policy column."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Test"},
            {"id": uuid4()},
            {"id": uuid4()},
        ]

        belief_share(
            belief_id=str(belief_id),
            recipient_did="did:key:alice",
            intent="know_me",
        )

        # Last execute call should be the UPDATE beliefs
        calls = mock_get_cursor.execute.call_args_list
        update_call = calls[-1]
        sql = update_call[0][0]
        assert "UPDATE beliefs" in sql
        assert "share_policy" in sql

    def test_share_with_expiry(self, mock_get_cursor):
        """belief_share accepts expires_at parameter."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Test"},
            {"id": uuid4()},
            {"id": uuid4()},
        ]

        result = belief_share(
            belief_id=str(belief_id),
            recipient_did="did:key:alice",
            intent="work_with_me",
            expires_at="2026-12-31T23:59:59",
        )

        assert result["success"] is True


# =============================================================================
# BELIEF_SHARES_LIST
# =============================================================================


class TestBeliefSharesList:
    """Tests for belief_shares_list handler."""

    def test_list_outgoing_shares(self, mock_get_cursor):
        """Returns outgoing shares for the local DID."""
        share_id = uuid4()
        belief_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchall.return_value = [
            {
                "share_id": share_id,
                "recipient_did": "did:key:alice",
                "intent": "know_me",
                "belief_id": belief_id,
                "created_at": now,
                "access_count": 0,
                "origin_sharer": "did:valence:local",
                "revoked": False,
                "revoked_at": None,
                "revocation_reason": None,
                "belief_content": "Test belief",
            }
        ]

        result = belief_shares_list(direction="outgoing")

        assert result["success"] is True
        assert result["direction"] == "outgoing"
        assert len(result["shares"]) == 1
        assert result["shares"][0]["share_id"] == str(share_id)
        assert result["shares"][0]["recipient_did"] == "did:key:alice"
        assert result["shares"][0]["intent"] == "know_me"

    def test_list_incoming_shares(self, mock_get_cursor):
        """Returns incoming shares for the local DID."""
        mock_get_cursor.fetchall.return_value = []

        result = belief_shares_list(direction="incoming")

        assert result["success"] is True
        assert result["direction"] == "incoming"

        # Check that SQL queries by recipient_did for incoming
        sql = mock_get_cursor.execute.call_args[0][0]
        assert "recipient_did" in sql

    def test_filters_revoked_by_default(self, mock_get_cursor):
        """Revoked shares are excluded by default."""
        mock_get_cursor.fetchall.return_value = []

        belief_shares_list(direction="outgoing", include_revoked=False)

        sql = mock_get_cursor.execute.call_args[0][0]
        assert "revoked = false" in sql

    def test_includes_revoked_when_requested(self, mock_get_cursor):
        """Revoked shares included when include_revoked=True."""
        mock_get_cursor.fetchall.return_value = []

        belief_shares_list(direction="outgoing", include_revoked=True)

        sql = mock_get_cursor.execute.call_args[0][0]
        assert "revoked = false" not in sql

    def test_empty_list(self, mock_get_cursor):
        """Returns empty list when no shares exist."""
        mock_get_cursor.fetchall.return_value = []

        result = belief_shares_list()

        assert result["success"] is True
        assert result["shares"] == []
        assert result["total_count"] == 0


# =============================================================================
# BELIEF_SHARE_REVOKE
# =============================================================================


class TestBeliefShareRevoke:
    """Tests for belief_share_revoke handler."""

    def test_revoke_share(self, mock_get_cursor):
        """Revokes a share by marking the consent chain."""
        share_id = uuid4()
        consent_chain_id = uuid4()

        mock_get_cursor.fetchone.return_value = {
            "id": share_id,
            "consent_chain_id": consent_chain_id,
            "origin_sharer": "did:valence:local",
            "revoked": False,
        }

        result = belief_share_revoke(share_id=str(share_id), reason="No longer relevant")

        assert result["success"] is True
        assert result["share_id"] == str(share_id)
        assert result["revoked"] is True
        assert result["reason"] == "No longer relevant"

        # Verify the UPDATE call
        calls = mock_get_cursor.execute.call_args_list
        update_call = calls[-1]
        sql = update_call[0][0]
        assert "UPDATE consent_chains" in sql
        assert "revoked = true" in sql

    def test_revoke_nonexistent_share(self, mock_get_cursor):
        """Returns error for nonexistent share."""
        mock_get_cursor.fetchone.return_value = None

        result = belief_share_revoke(share_id=str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_revoke_already_revoked(self, mock_get_cursor):
        """Returns error when share is already revoked."""
        share_id = uuid4()
        mock_get_cursor.fetchone.return_value = {
            "id": share_id,
            "consent_chain_id": uuid4(),
            "origin_sharer": "did:valence:local",
            "revoked": True,
        }

        result = belief_share_revoke(share_id=str(share_id))

        assert result["success"] is False
        assert "already revoked" in result["error"]

    def test_revoke_not_owner(self, mock_get_cursor):
        """Returns error when trying to revoke someone else's share."""
        share_id = uuid4()
        mock_get_cursor.fetchone.return_value = {
            "id": share_id,
            "consent_chain_id": uuid4(),
            "origin_sharer": "did:key:someone_else",
            "revoked": False,
        }

        result = belief_share_revoke(share_id=str(share_id))

        assert result["success"] is False
        assert "did not create" in result["error"]

    def test_revoke_without_reason(self, mock_get_cursor):
        """Revocation works without a reason."""
        share_id = uuid4()
        mock_get_cursor.fetchone.return_value = {
            "id": share_id,
            "consent_chain_id": uuid4(),
            "origin_sharer": "did:valence:local",
            "revoked": False,
        }

        result = belief_share_revoke(share_id=str(share_id))

        assert result["success"] is True
        assert result["reason"] is None


# =============================================================================
# BELIEF_CREATE — VISIBILITY & SHARING_INTENT
# =============================================================================


class TestBeliefCreateVisibility:
    """Tests for belief_create visibility and sharing_intent params."""

    def test_create_with_visibility(self, mock_get_cursor, sample_belief_row):
        """belief_create with visibility stores it in the INSERT."""
        belief_id = uuid4()
        row = sample_belief_row(id=belief_id)
        mock_get_cursor.fetchone.side_effect = [
            None,  # No exact hash match
            None,  # No fuzzy semantic match
            row,   # INSERT RETURNING
        ]

        result = belief_create(content="Test belief", visibility="federated")

        assert result["success"] is True

        # Check that the INSERT includes visibility
        calls = mock_get_cursor.execute.call_args_list
        # Find the INSERT INTO beliefs call
        insert_call = None
        for call in calls:
            sql = call[0][0]
            if "INSERT INTO beliefs" in sql:
                insert_call = call
                break
        assert insert_call is not None
        sql = insert_call[0][0]
        assert "visibility" in sql
        # Check that "federated" is in the params
        params = insert_call[0][1]
        assert "federated" in params

    def test_create_with_sharing_intent(self, mock_get_cursor, sample_belief_row):
        """belief_create with sharing_intent generates and stores SharePolicy."""
        belief_id = uuid4()
        row = sample_belief_row(id=belief_id)
        mock_get_cursor.fetchone.side_effect = [
            None,  # No exact hash match
            None,  # No fuzzy semantic match
            row,   # INSERT RETURNING
        ]

        result = belief_create(content="Public knowledge", sharing_intent="use_this")

        assert result["success"] is True

        # Check that the INSERT includes share_policy
        calls = mock_get_cursor.execute.call_args_list
        insert_call = None
        for call in calls:
            sql = call[0][0]
            if "INSERT INTO beliefs" in sql:
                insert_call = call
                break
        assert insert_call is not None
        sql = insert_call[0][0]
        assert "share_policy" in sql

        # The share_policy param should be JSON with intent info
        params = insert_call[0][1]
        # share_policy is second-to-last param (embedding is last)
        share_policy_param = params[-2]
        assert share_policy_param is not None
        policy_data = json.loads(share_policy_param)
        assert policy_data["intent"] == "use_this"
        assert policy_data["policy"]["level"] == "public"

    def test_create_default_visibility(self, mock_get_cursor, sample_belief_row):
        """belief_create defaults to private visibility."""
        belief_id = uuid4()
        row = sample_belief_row(id=belief_id)
        mock_get_cursor.fetchone.side_effect = [
            None,  # No exact hash match
            None,  # No fuzzy semantic match
            row,   # INSERT RETURNING
        ]

        belief_create(content="Default visibility test")

        calls = mock_get_cursor.execute.call_args_list
        insert_call = None
        for call in calls:
            sql = call[0][0]
            if "INSERT INTO beliefs" in sql:
                insert_call = call
                break
        assert insert_call is not None
        params = insert_call[0][1]
        # visibility param should be "private"
        assert "private" in params

    def test_create_without_sharing_intent(self, mock_get_cursor, sample_belief_row):
        """belief_create without sharing_intent stores NULL share_policy."""
        belief_id = uuid4()
        row = sample_belief_row(id=belief_id)
        mock_get_cursor.fetchone.side_effect = [
            None,  # No exact hash match
            None,  # No fuzzy semantic match
            row,   # INSERT RETURNING
        ]

        belief_create(content="No intent test")

        calls = mock_get_cursor.execute.call_args_list
        insert_call = None
        for call in calls:
            sql = call[0][0]
            if "INSERT INTO beliefs" in sql:
                insert_call = call
                break
        assert insert_call is not None
        params = insert_call[0][1]
        # share_policy param should be None (second-to-last, embedding is last)
        assert params[-2] is None


# =============================================================================
# BELIEF_CREATE — VALIDATION (#333, #334)
# =============================================================================


class TestBeliefCreateValidation:
    """Tests for belief_create input validation (hardening)."""

    def test_create_rejects_know_me_intent(self, mock_get_cursor):
        """belief_create with know_me returns clean error, not crash (#333)."""
        result = belief_create(content="Test", sharing_intent="know_me")

        assert result["success"] is False
        assert "know_me" in result["error"]
        assert "belief_share" in result["error"]
        # Should NOT have called the database at all
        mock_get_cursor.execute.assert_not_called()

    def test_create_rejects_invalid_visibility(self):
        """belief_create with invalid visibility returns clean error (#334)."""
        result = belief_create(content="Test", visibility="secret")

        assert result["success"] is False
        assert "Invalid visibility" in result["error"]
        assert "secret" in result["error"]

    def test_create_rejects_invalid_sharing_intent(self):
        """belief_create with invalid sharing_intent returns clean error (#334)."""
        result = belief_create(content="Test", sharing_intent="yolo")

        assert result["success"] is False
        assert "Invalid sharing_intent" in result["error"]
        assert "yolo" in result["error"]


# =============================================================================
# BELIEF_SHARE — VALIDATION (#334)
# =============================================================================


class TestBeliefShareValidation:
    """Tests for belief_share input validation (hardening)."""

    def test_share_rejects_invalid_intent(self, mock_get_cursor):
        """belief_share with invalid intent returns clean error (#334)."""
        result = belief_share(belief_id=str(uuid4()), recipient_did="did:key:alice", intent="yolo")

        assert result["success"] is False
        assert "Invalid intent" in result["error"]
        assert "yolo" in result["error"]
        # Should NOT have called the database
        mock_get_cursor.execute.assert_not_called()

    def test_share_rejects_invalid_expires_at(self, mock_get_cursor):
        """belief_share with invalid expires_at returns clean error (#334)."""
        result = belief_share(
            belief_id=str(uuid4()),
            recipient_did="did:key:alice",
            expires_at="not-a-date",
        )

        assert result["success"] is False
        assert "Invalid expires_at" in result["error"]

    def test_share_duplicate_recipient(self, mock_get_cursor):
        """belief_share for duplicate recipient returns 'already shared' error (#334)."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Test", "share_policy": None},  # belief lookup
            {"id": uuid4()},  # consent chain INSERT
        ]

        # Simulate UniqueViolation on shares INSERT
        class UniqueViolationError(Exception):
            pass

        UniqueViolationError.__name__ = "UniqueViolation"

        # The third execute call (shares INSERT) should raise
        call_count = [0]
        original_execute = mock_get_cursor.execute

        def counting_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:  # shares INSERT
                raise UniqueViolationError("duplicate key value violates unique constraint")
            return original_execute(*args, **kwargs)

        mock_get_cursor.execute = counting_execute

        result = belief_share(belief_id=str(belief_id), recipient_did="did:key:alice")

        assert result["success"] is False
        assert "already shared" in result["error"].lower()


# =============================================================================
# BELIEF_SHARE — RESHARE POLICY CHECK (#337)
# =============================================================================


class TestBeliefShareReshareCheck:
    """Tests for belief_share reshare policy enforcement (#337)."""

    def _make_policy_json(self, level, enforcement="policy", recipients=None, expires_at=None):
        """Helper to build IntentConfig-style JSON stored in share_policy column."""
        policy = {
            "level": level,
            "enforcement": enforcement,
        }
        if recipients:
            policy["recipients"] = recipients
        if expires_at:
            policy["propagation"] = {"max_hops": 0, "expires_at": expires_at}
        return json.dumps({"intent": "know_me", "policy": policy})

    def test_share_blocked_by_private_policy(self, mock_get_cursor):
        """PRIVATE share_policy blocks sharing."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.return_value = {
            "id": belief_id,
            "content": "Secret",
            "share_policy": self._make_policy_json("private"),
        }

        result = belief_share(belief_id=str(belief_id), recipient_did="did:key:bob")

        assert result["success"] is False
        assert "does not allow" in result["error"]

    def test_share_blocked_by_direct_wrong_recipient(self, mock_get_cursor):
        """DIRECT policy to Alice blocks sharing to Bob."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.return_value = {
            "id": belief_id,
            "content": "For Alice only",
            "share_policy": self._make_policy_json("direct", "cryptographic", recipients=["did:key:alice"]),
        }

        result = belief_share(belief_id=str(belief_id), recipient_did="did:key:bob")

        assert result["success"] is False
        assert "does not allow" in result["error"]

    def test_share_allowed_by_direct_correct_recipient(self, mock_get_cursor):
        """DIRECT policy to Alice allows sharing to Alice."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {
                "id": belief_id,
                "content": "For Alice",
                "share_policy": self._make_policy_json("direct", "cryptographic", recipients=["did:key:alice"]),
            },
            {"id": uuid4()},  # consent chain
            {"id": uuid4()},  # share
        ]

        result = belief_share(belief_id=str(belief_id), recipient_did="did:key:alice")

        assert result["success"] is True

    def test_share_allowed_when_no_policy(self, mock_get_cursor):
        """NULL share_policy permits sharing (permissive default)."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Open belief", "share_policy": None},
            {"id": uuid4()},  # consent chain
            {"id": uuid4()},  # share
        ]

        result = belief_share(belief_id=str(belief_id), recipient_did="did:key:anyone")

        assert result["success"] is True

    def test_share_blocked_by_expired_policy(self, mock_get_cursor):
        """Expired propagation blocks sharing."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.return_value = {
            "id": belief_id,
            "content": "Expired",
            "share_policy": self._make_policy_json("bounded", expires_at="2020-01-01T00:00:00"),
        }

        result = belief_share(belief_id=str(belief_id), recipient_did="did:key:bob")

        assert result["success"] is False
        assert "expired" in result["error"].lower()

    def test_share_allowed_by_public_policy(self, mock_get_cursor):
        """PUBLIC policy permits sharing to anyone."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {
                "id": belief_id,
                "content": "Public knowledge",
                "share_policy": self._make_policy_json("public", "honor"),
            },
            {"id": uuid4()},  # consent chain
            {"id": uuid4()},  # share
        ]

        result = belief_share(belief_id=str(belief_id), recipient_did="did:key:anyone")

        assert result["success"] is True


# =============================================================================
# BELIEF_SHARE — RESPONSE CONTENT (#336)
# =============================================================================


class TestBeliefShareResponseContent:
    """Tests for belief_share response including belief content (#336)."""

    def test_share_response_includes_content(self, mock_get_cursor):
        """belief_share response includes belief_content."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Important knowledge here", "share_policy": None},
            {"id": uuid4()},  # consent chain
            {"id": uuid4()},  # share
        ]

        result = belief_share(belief_id=str(belief_id), recipient_did="did:key:alice")

        assert result["success"] is True
        assert result["belief_content"] == "Important knowledge here"


# =============================================================================
# BELIEF_SHARES_LIST — VALIDATION & FILTER (#334, #335)
# =============================================================================


class TestBeliefSharesListValidation:
    """Tests for belief_shares_list validation and belief_id filter (#334, #335)."""

    def test_list_rejects_invalid_direction(self):
        """belief_shares_list with invalid direction returns clean error (#334)."""
        result = belief_shares_list(direction="sideways")

        assert result["success"] is False
        assert "Invalid direction" in result["error"]
        assert "sideways" in result["error"]

    def test_list_filters_by_belief_id(self, mock_get_cursor):
        """belief_shares_list filters by belief_id when provided (#335)."""
        target_belief_id = str(uuid4())
        mock_get_cursor.fetchall.return_value = []

        belief_shares_list(direction="outgoing", belief_id=target_belief_id)

        sql = mock_get_cursor.execute.call_args[0][0]
        params = mock_get_cursor.execute.call_args[0][1]
        assert "s.belief_id = %s" in sql
        assert target_belief_id in params

    def test_list_no_belief_id_filter(self, mock_get_cursor):
        """belief_shares_list returns all when belief_id not provided."""
        mock_get_cursor.fetchall.return_value = []

        belief_shares_list(direction="outgoing")

        sql = mock_get_cursor.execute.call_args[0][0]
        assert "s.belief_id = %s" not in sql

    def test_list_tool_definition_includes_belief_id(self):
        """belief_shares_list tool definition has belief_id property (#335)."""
        tool = next(t for t in SUBSTRATE_TOOLS if t.name == "belief_shares_list")
        props = tool.inputSchema["properties"]
        assert "belief_id" in props
        assert props["belief_id"]["type"] == "string"


# =============================================================================
# BELIEF_SHARE — POLICY PRESERVATION
# =============================================================================


class TestBeliefSharePolicyPreservation:
    """Tests for belief_share COALESCE behavior on share_policy."""

    def test_share_sets_policy_on_first_share(self, mock_get_cursor):
        """First share sets share_policy (NULL → policy)."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Test", "share_policy": None},
            {"id": uuid4()},  # consent chain
            {"id": uuid4()},  # share
        ]

        belief_share(belief_id=str(belief_id), recipient_did="did:key:alice")

        # Find the UPDATE beliefs call
        calls = mock_get_cursor.execute.call_args_list
        update_call = calls[-1]
        sql = update_call[0][0]
        assert "COALESCE" in sql
        assert "share_policy" in sql

    def test_share_preserves_existing_policy(self, mock_get_cursor):
        """Existing share_policy is not overwritten (COALESCE)."""
        belief_id = uuid4()
        existing_policy = json.dumps({"intent": "know_me", "policy": {"level": "direct", "enforcement": "cryptographic", "recipients": ["did:key:alice"]}})
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Test", "share_policy": existing_policy},
            {"id": uuid4()},  # consent chain
            {"id": uuid4()},  # share
        ]

        # Share to Alice (allowed by existing DIRECT policy)
        belief_share(belief_id=str(belief_id), recipient_did="did:key:alice")

        # The UPDATE uses COALESCE, so existing policy is preserved
        calls = mock_get_cursor.execute.call_args_list
        update_call = calls[-1]
        sql = update_call[0][0]
        assert "COALESCE(share_policy," in sql


# =============================================================================
# BELIEF_SHARE — OWNERSHIP CHECK (#337)
# =============================================================================


class TestBeliefShareOwnership:
    """Tests for belief_share ownership enforcement (#337)."""

    def test_share_allowed_when_holder_id_null(self, mock_get_cursor):
        """NULL holder_id = locally created, sharing is permissive (backward compat)."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "My belief", "share_policy": None, "holder_id": None},
            {"id": uuid4()},  # consent chain
            {"id": uuid4()},  # share
        ]

        result = belief_share(belief_id=str(belief_id), recipient_did="did:key:alice")
        assert result["success"] is True

    def test_share_allowed_when_holder_id_missing_from_row(self, mock_get_cursor):
        """Missing holder_id key (legacy row) treated as NULL — permissive."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Legacy belief", "share_policy": None},  # no holder_id key
            {"id": uuid4()},  # consent chain
            {"id": uuid4()},  # share
        ]

        result = belief_share(belief_id=str(belief_id), recipient_did="did:key:alice")
        assert result["success"] is True

    def test_share_blocked_by_nonmatching_holder(self, mock_get_cursor):
        """Non-NULL holder_id that doesn't match local entity blocks sharing."""
        belief_id = uuid4()
        other_entity_id = uuid4()
        mock_get_cursor.fetchone.return_value = {
            "id": belief_id,
            "content": "Their belief",
            "share_policy": None,
            "holder_id": other_entity_id,
        }

        # No VALENCE_LOCAL_ENTITY_ID set → can't prove ownership
        with patch.dict("os.environ", {}, clear=False):
            # Ensure env var is not set
            import os

            os.environ.pop("VALENCE_LOCAL_ENTITY_ID", None)
            result = belief_share(belief_id=str(belief_id), recipient_did="did:key:alice")

        assert result["success"] is False
        assert "not the holder" in result["error"]

    def test_share_blocked_when_holder_id_mismatches_local_entity(self, mock_get_cursor):
        """holder_id doesn't match VALENCE_LOCAL_ENTITY_ID → blocked."""
        belief_id = uuid4()
        other_entity_id = uuid4()
        local_entity_id = uuid4()
        mock_get_cursor.fetchone.return_value = {
            "id": belief_id,
            "content": "Their belief",
            "share_policy": None,
            "holder_id": other_entity_id,
        }

        with patch.dict("os.environ", {"VALENCE_LOCAL_ENTITY_ID": str(local_entity_id)}):
            result = belief_share(belief_id=str(belief_id), recipient_did="did:key:alice")

        assert result["success"] is False
        assert "not the holder" in result["error"]

    def test_share_allowed_when_holder_id_matches_local_entity(self, mock_get_cursor):
        """holder_id matches VALENCE_LOCAL_ENTITY_ID → allowed."""
        belief_id = uuid4()
        local_entity_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {
                "id": belief_id,
                "content": "My belief",
                "share_policy": None,
                "holder_id": local_entity_id,
            },
            {"id": uuid4()},  # consent chain
            {"id": uuid4()},  # share
        ]

        with patch.dict("os.environ", {"VALENCE_LOCAL_ENTITY_ID": str(local_entity_id)}):
            result = belief_share(belief_id=str(belief_id), recipient_did="did:key:alice")

        assert result["success"] is True
        assert result["recipient"] == "did:key:alice"


# =============================================================================
# EDGE CASES (#338)
# =============================================================================


class TestBeliefShareEdgeCases:
    """Additional edge cases for sharing tools (#338)."""

    def test_share_with_max_hops_zero(self, mock_get_cursor):
        """max_hops=0 should create a non-propagatable share."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "No propagation"},
            {"id": uuid4()},  # consent chain
            {"id": uuid4()},  # share
        ]

        result = belief_share(
            belief_id=str(belief_id),
            recipient_did="did:key:alice",
            max_hops=0,
        )

        assert result["success"] is True
        # Policy should reflect max_hops=0
        assert result["policy"]["propagation"]["max_hops"] == 0

    def test_share_with_past_expires_at(self, mock_get_cursor):
        """Sharing with already-expired expires_at still creates the share.

        The expires_at is stored but enforcement happens at access time, not creation.
        """
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Expired share"},
            {"id": uuid4()},  # consent chain
            {"id": uuid4()},  # share
        ]

        result = belief_share(
            belief_id=str(belief_id),
            recipient_did="did:key:alice",
            expires_at="2020-01-01T00:00:00",
        )

        assert result["success"] is True

    def test_list_shares_with_belief_filter(self, mock_get_cursor):
        """belief_shares_list can filter by belief_id."""
        belief_id = uuid4()
        mock_get_cursor.fetchall.return_value = []

        result = belief_shares_list(direction="outgoing", belief_id=str(belief_id))

        assert result["success"] is True
        # Check that SQL includes the belief_id filter
        sql = mock_get_cursor.execute.call_args[0][0]
        assert "s.belief_id = %s" in sql

    def test_list_shares_invalid_direction(self, mock_get_cursor):
        """Invalid direction returns clean error."""
        result = belief_shares_list(direction="sideways")

        assert result["success"] is False
        assert "Invalid direction" in result["error"]
        mock_get_cursor.execute.assert_not_called()

    def test_revoke_then_list_excludes_revoked(self, mock_get_cursor):
        """After revoking, the share doesn't appear in default list."""
        mock_get_cursor.fetchall.return_value = []  # no active shares

        result = belief_shares_list(direction="outgoing", include_revoked=False)

        assert result["success"] is True
        sql = mock_get_cursor.execute.call_args[0][0]
        assert "revoked = false" in sql

    def test_share_response_includes_all_fields(self, mock_get_cursor):
        """Successful share response includes all expected fields."""
        belief_id = uuid4()
        consent_chain_id = uuid4()
        share_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"id": belief_id, "content": "Complete response test"},
            {"id": consent_chain_id},
            {"id": share_id},
        ]

        result = belief_share(belief_id=str(belief_id), recipient_did="did:key:alice", intent="know_me")

        assert result["success"] is True
        assert "share_id" in result
        assert "consent_chain_id" in result
        assert "recipient" in result
        assert "intent" in result
        assert "policy" in result
        assert "belief_content" in result
        assert result["intent"] == "know_me"
        assert result["belief_content"] == "Complete response test"

    def test_list_shares_response_format(self, mock_get_cursor):
        """Each share in list response has the correct field format."""
        share_id = uuid4()
        belief_id = uuid4()
        now = datetime.now()
        mock_get_cursor.fetchall.return_value = [
            {
                "share_id": share_id,
                "recipient_did": "did:key:alice",
                "intent": "learn_from_me",
                "belief_id": belief_id,
                "created_at": now,
                "access_count": 3,
                "origin_sharer": "did:valence:local",
                "revoked": False,
                "revoked_at": None,
                "revocation_reason": None,
                "belief_content": "Test content",
            }
        ]

        result = belief_shares_list(direction="outgoing")

        share = result["shares"][0]
        assert share["share_id"] == str(share_id)
        assert share["belief_id"] == str(belief_id)
        assert share["belief_content"] == "Test content"
        assert share["access_count"] == 3
        assert share["revoked"] is False
        assert share["revoked_at"] is None
        assert share["revocation_reason"] is None

    def test_revoke_without_reason_succeeds(self, mock_get_cursor):
        """Revocation without a reason is allowed."""
        share_id = uuid4()
        mock_get_cursor.fetchone.return_value = {
            "id": share_id,
            "consent_chain_id": uuid4(),
            "origin_sharer": "did:valence:local",
            "revoked": False,
        }

        result = belief_share_revoke(share_id=str(share_id))

        assert result["success"] is True
        assert result["reason"] is None

    def test_share_all_intent_types(self, mock_get_cursor):
        """All four intent types produce valid shares."""
        for intent in ["know_me", "work_with_me", "learn_from_me", "use_this"]:
            belief_id = uuid4()
            mock_get_cursor.fetchone.side_effect = [
                {"id": belief_id, "content": f"Test {intent}"},
                {"id": uuid4()},
                {"id": uuid4()},
            ]

            result = belief_share(belief_id=str(belief_id), recipient_did="did:key:alice", intent=intent)
            assert result["success"] is True, f"Failed for intent={intent}"
            assert result["intent"] == intent
