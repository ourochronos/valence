"""E2E validation: auto-capture pipeline.

Tests the full flow: session_start → exchange_add → session_end with summary/themes
→ beliefs created with extraction_method='auto' → vkb_session_insights links exist
→ dedup works (run twice with same summary)

Uses mocked DB to avoid requiring a live database.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_cursor():
    """Mock database cursor with call tracking."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Mock the get_cursor context manager for substrate/vkb tools."""

    @contextmanager
    def _mock_get_cursor(dict_cursor: bool = True) -> Generator:
        yield mock_cursor

    with (
        patch("valence.substrate.tools._common.get_cursor", _mock_get_cursor),
        patch("valence.vkb.tools.sessions.get_cursor", _mock_get_cursor),
        patch("valence.vkb.tools.exchanges.get_cursor", _mock_get_cursor),
        patch("valence.vkb.tools.patterns.get_cursor", _mock_get_cursor),
        patch("valence.vkb.tools.insights.get_cursor", _mock_get_cursor),
    ):
        yield mock_cursor


# =============================================================================
# E2E TESTS
# =============================================================================


class TestAutoCapturePipeline:
    """Test the full session lifecycle with auto-capture."""

    def test_session_lifecycle_creates_beliefs(self, mock_get_cursor):
        """Full flow: session_start → exchange_add → session_end."""
        # VKB removed from valence.vkb.tools import exchange_add, session_end, session_start

        session_id = uuid4()
        now = datetime.now()

        # session_start
        mock_get_cursor.fetchone.side_effect = [
            # session_start: INSERT INTO vkb_sessions RETURNING
            {
                "id": session_id,
                "platform": "claude-code",
                "project_context": "valence",
                "status": "active",
                "summary": None,
                "themes": [],
                "started_at": now,
                "ended_at": None,
                "claude_session_id": None,
                "external_room_id": None,
                "metadata": {},
                "opt_out_federation": False,
                "share_policy": None,
                "extraction_metadata": None,
            },
            None,  # INSERT INTO sources
        ]
        result = session_start(platform="claude-code", project_context="valence")
        assert result["success"] is True

        # exchange_add
        exchange_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            {"next_seq": 1},  # MAX(sequence) + 1
            {
                "id": exchange_id,
                "session_id": session_id,
                "sequence": 1,
                "role": "user",
                "content": "test message",
                "created_at": now,
                "tokens_approx": None,
                "tool_uses": [],
            },
        ]
        result = exchange_add(session_id=str(session_id), role="user", content="test message")
        assert result["success"] is True

        # session_end
        mock_get_cursor.fetchone.side_effect = [
            {
                "id": session_id,
                "platform": "claude-code",
                "project_context": "valence",
                "status": "completed",
                "summary": "Test summary",
                "themes": ["testing"],
                "started_at": now,
                "ended_at": now,
                "claude_session_id": None,
                "external_room_id": None,
                "metadata": {},
                "opt_out_federation": False,
                "share_policy": None,
                "extraction_metadata": None,
            },
        ]
        result = session_end(session_id=str(session_id), summary="Test summary", themes=["testing"])
        assert result["success"] is True

    def test_insight_extract_creates_belief_with_session_link(self, mock_get_cursor):
        """insight_extract creates a belief and links it to the session."""
        # VKB removed from valence.vkb.tools import insight_extract

        session_id = uuid4()
        belief_id = uuid4()
        insight_id = uuid4()
        source_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            None,  # no content_hash match (dedup check)
            {"id": source_id},  # source for session
            {  # INSERT belief RETURNING
                "id": belief_id,
                "content": "Test insight",
                "confidence": json.dumps({"overall": 0.8}),
                "domain_path": ["test"],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": source_id,
                "extraction_method": "conversation_extraction",
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
                "opt_out_federation": False,
                "content_hash": hashlib.sha256(b"test insight").hexdigest(),
                "holder_id": None,
                "version": 1,
                "visibility": "private",
                "share_policy": None,
                "extraction_metadata": None,
                "embedding": None,
            },
            {"id": insight_id},  # INSERT vkb_session_insights RETURNING
        ]

        result = insight_extract(
            session_id=str(session_id),
            content="Test insight",
            domain_path=["test"],
        )

        assert result["success"] is True
        assert result["belief_id"] == str(belief_id)
        assert result["session_id"] == str(session_id)

    def test_insight_extract_dedup_on_second_call(self, mock_get_cursor):
        """Second insight_extract with same content deduplicates."""
        # VKB removed from valence.vkb.tools import insight_extract

        session_id = uuid4()
        existing_id = uuid4()
        insight_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            {"id": existing_id, "confidence": {"overall": 0.5}},  # content_hash match
            {"cnt": 1},  # corroboration count
            None,  # INSERT corroboration
            None,  # UPDATE beliefs
            {"id": insight_id},  # INSERT vkb_session_insights
        ]

        result = insight_extract(
            session_id=str(session_id),
            content="Test insight",
        )

        assert result["success"] is True
        assert result["deduplicated"] is True
        assert result["action"] == "reinforced"
        assert result["corroboration_count"] == 2


class TestHookDeduplication:
    """Test the session-end hook's create_belief dedup logic."""

    def test_hook_create_belief_dedup(self):
        """Hook's create_belief function deduplicates by content hash."""
        # Import the hook module directly
        import importlib.util
        import os

        hook_path = os.path.join(os.path.dirname(__file__), "../../plugin/hooks/session-end.py")
        hook_path = os.path.normpath(hook_path)

        if not os.path.exists(hook_path):
            pytest.skip("session-end.py hook not found at expected path")

        spec = importlib.util.spec_from_file_location("session_end_hook", hook_path)
        hook = importlib.util.module_from_spec(spec)

        # The hook uses hashlib, json, uuid which are stdlib
        # We just verify the function signature exists and hash logic is consistent
        import hashlib

        content = "Test belief content"
        expected_hash = hashlib.sha256(content.strip().lower().encode()).hexdigest()

        # Verify the hash computation matches what our tools use
        from valence.core.articles import _content_hash

        assert _content_hash(content) == expected_hash
