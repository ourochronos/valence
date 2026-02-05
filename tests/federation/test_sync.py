"""Tests for belief synchronization.

Tests cover:
- Sync state management (get_sync_state, update_sync_state)
- Outbound queue operations (queue_belief_for_sync, get_pending_sync_items, mark_sync_item_*)
- SyncManager lifecycle and sync operations
- Vector clock operations (update_vector_clock, compare_vector_clocks)
- Convenience functions (trigger_sync, get_sync_status)
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from valence.federation.models import SyncStatus
from valence.federation.sync import (
    SyncManager,
    compare_vector_clocks,
    get_pending_sync_items,
    get_sync_state,
    get_sync_status,
    mark_sync_item_failed,
    mark_sync_item_sent,
    queue_belief_for_sync,
    trigger_sync,
    update_sync_state,
    update_vector_clock,
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

    with patch("valence.federation.sync.get_cursor", _mock_get_cursor):
        yield mock_cursor


@pytest.fixture
def sample_sync_state_row():
    """Create a sample sync state row."""

    def _factory(node_id: UUID | None = None, **kwargs):
        now = datetime.now()
        return {
            "id": uuid4(),
            "node_id": node_id or uuid4(),
            "last_received_cursor": kwargs.get("last_received_cursor"),
            "last_sent_cursor": kwargs.get("last_sent_cursor"),
            "vector_clock": kwargs.get("vector_clock", {}),
            "status": kwargs.get("status", "idle"),
            "beliefs_sent": kwargs.get("beliefs_sent", 0),
            "beliefs_received": kwargs.get("beliefs_received", 0),
            "last_sync_duration_ms": kwargs.get("last_sync_duration_ms"),
            "last_error": kwargs.get("last_error"),
            "error_count": kwargs.get("error_count", 0),
            "last_sync_at": kwargs.get("last_sync_at"),
            "next_sync_scheduled": kwargs.get("next_sync_scheduled"),
            "created_at": kwargs.get("created_at", now),
            "modified_at": kwargs.get("modified_at", now),
        }

    return _factory


@pytest.fixture
def sample_queue_item_row():
    """Create a sample sync queue item row."""

    def _factory(id: UUID | None = None, belief_id: UUID | None = None, **kwargs):
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "target_node_id": kwargs.get("target_node_id"),
            "operation": kwargs.get("operation", "share_belief"),
            "belief_id": belief_id or uuid4(),
            "priority": kwargs.get("priority", 5),
            "status": kwargs.get("status", "pending"),
            "scheduled_for": kwargs.get("scheduled_for", now),
            "attempts": kwargs.get("attempts", 0),
            "max_attempts": kwargs.get("max_attempts", 3),
            "last_attempt_at": kwargs.get("last_attempt_at"),
            "last_error": kwargs.get("last_error"),
            "created_at": kwargs.get("created_at", now),
        }

    return _factory


# =============================================================================
# SYNC STATE MANAGEMENT TESTS
# =============================================================================


class TestGetSyncState:
    """Tests for get_sync_state function."""

    def test_get_sync_state_found(self, mock_get_cursor, sample_sync_state_row):
        """Test getting sync state when record exists."""
        node_id = uuid4()
        row = sample_sync_state_row(node_id=node_id, status="syncing", beliefs_sent=100)
        mock_get_cursor.fetchone.return_value = row

        result = get_sync_state(node_id)

        assert result is not None
        assert result.node_id == node_id
        assert result.status == SyncStatus.SYNCING
        assert result.beliefs_sent == 100

    def test_get_sync_state_not_found(self, mock_get_cursor):
        """Test getting sync state when record doesn't exist."""
        mock_get_cursor.fetchone.return_value = None

        result = get_sync_state(uuid4())

        assert result is None

    def test_get_sync_state_error(self, mock_get_cursor):
        """Test getting sync state when database error occurs."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = get_sync_state(uuid4())

        assert result is None


class TestUpdateSyncState:
    """Tests for update_sync_state function."""

    def test_update_sync_state_status(self, mock_get_cursor):
        """Test updating sync state status."""
        node_id = uuid4()

        result = update_sync_state(node_id, status=SyncStatus.SYNCING)

        assert result is True
        mock_get_cursor.execute.assert_called_once()
        call_args = mock_get_cursor.execute.call_args
        assert "status = %s" in call_args[0][0]

    def test_update_sync_state_cursors(self, mock_get_cursor):
        """Test updating sync state cursors."""
        node_id = uuid4()

        result = update_sync_state(node_id, last_received_cursor="cursor123", last_sent_cursor="cursor456")

        assert result is True
        call_args = mock_get_cursor.execute.call_args
        assert "last_received_cursor = %s" in call_args[0][0]
        assert "last_sent_cursor = %s" in call_args[0][0]

    def test_update_sync_state_beliefs_delta(self, mock_get_cursor):
        """Test updating sync state with beliefs delta."""
        node_id = uuid4()

        result = update_sync_state(node_id, beliefs_sent_delta=5, beliefs_received_delta=10)

        assert result is True
        call_args = mock_get_cursor.execute.call_args
        assert "beliefs_sent = beliefs_sent + %s" in call_args[0][0]
        assert "beliefs_received = beliefs_received + %s" in call_args[0][0]

    def test_update_sync_state_set_error(self, mock_get_cursor):
        """Test updating sync state with error."""
        node_id = uuid4()

        result = update_sync_state(node_id, last_error="Connection timeout")

        assert result is True
        call_args = mock_get_cursor.execute.call_args
        assert "last_error = %s" in call_args[0][0]
        assert "error_count = error_count + 1" in call_args[0][0]

    def test_update_sync_state_clear_error(self, mock_get_cursor):
        """Test clearing sync state error."""
        node_id = uuid4()

        result = update_sync_state(node_id, clear_error=True)

        assert result is True
        call_args = mock_get_cursor.execute.call_args
        assert "last_error = NULL" in call_args[0][0]
        assert "error_count = 0" in call_args[0][0]

    def test_update_sync_state_error(self, mock_get_cursor):
        """Test updating sync state when database error occurs."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = update_sync_state(uuid4(), status=SyncStatus.ERROR)

        assert result is False


# =============================================================================
# OUTBOUND QUEUE TESTS
# =============================================================================


class TestQueueBeliefForSync:
    """Tests for queue_belief_for_sync function."""

    def test_queue_belief_success(self, mock_get_cursor):
        """Test queuing belief for sync."""
        belief_id = uuid4()

        result = queue_belief_for_sync(belief_id)

        assert result is True
        mock_get_cursor.execute.assert_called_once()
        call_args = mock_get_cursor.execute.call_args
        assert "INSERT INTO sync_outbound_queue" in call_args[0][0]

    def test_queue_belief_with_target_node(self, mock_get_cursor):
        """Test queuing belief for specific node."""
        belief_id = uuid4()
        target_node_id = uuid4()

        result = queue_belief_for_sync(belief_id, target_node_id=target_node_id, priority=1)

        assert result is True
        call_args = mock_get_cursor.execute.call_args
        # Check that target_node_id is in params
        assert target_node_id in call_args[0][1]

    def test_queue_belief_error(self, mock_get_cursor):
        """Test queuing belief when database error occurs."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = queue_belief_for_sync(uuid4())

        assert result is False


class TestGetPendingSyncItems:
    """Tests for get_pending_sync_items function."""

    def test_get_pending_items_all(self, mock_get_cursor, sample_queue_item_row):
        """Test getting all pending sync items."""
        items = [sample_queue_item_row() for _ in range(3)]
        mock_get_cursor.fetchall.return_value = items

        result = get_pending_sync_items()

        assert len(result) == 3
        mock_get_cursor.execute.assert_called_once()

    def test_get_pending_items_for_node(self, mock_get_cursor, sample_queue_item_row):
        """Test getting pending items for specific node."""
        target_node_id = uuid4()
        items = [sample_queue_item_row(target_node_id=target_node_id)]
        mock_get_cursor.fetchall.return_value = items

        result = get_pending_sync_items(target_node_id=target_node_id, limit=50)

        assert len(result) == 1
        call_args = mock_get_cursor.execute.call_args
        assert "target_node_id = %s" in call_args[0][0]

    def test_get_pending_items_empty(self, mock_get_cursor):
        """Test getting pending items when none exist."""
        mock_get_cursor.fetchall.return_value = []

        result = get_pending_sync_items()

        assert result == []

    def test_get_pending_items_error(self, mock_get_cursor):
        """Test getting pending items when database error occurs."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = get_pending_sync_items()

        assert result == []


class TestMarkSyncItemSent:
    """Tests for mark_sync_item_sent function."""

    def test_mark_sent_success(self, mock_get_cursor):
        """Test marking sync item as sent."""
        item_id = uuid4()

        result = mark_sync_item_sent(item_id)

        assert result is True
        call_args = mock_get_cursor.execute.call_args
        assert "status = 'sent'" in call_args[0][0]

    def test_mark_sent_error(self, mock_get_cursor):
        """Test marking sync item as sent when error occurs."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = mark_sync_item_sent(uuid4())

        assert result is False


class TestMarkSyncItemFailed:
    """Tests for mark_sync_item_failed function."""

    def test_mark_failed_success(self, mock_get_cursor):
        """Test marking sync item as failed."""
        item_id = uuid4()

        result = mark_sync_item_failed(item_id, "Connection refused")

        assert result is True
        call_args = mock_get_cursor.execute.call_args
        assert "attempts = attempts + 1" in call_args[0][0]
        assert "last_error = %s" in call_args[0][0]

    def test_mark_failed_error(self, mock_get_cursor):
        """Test marking sync item as failed when error occurs."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = mark_sync_item_failed(uuid4(), "error")

        assert result is False


# =============================================================================
# SYNC MANAGER TESTS
# =============================================================================


class TestSyncManager:
    """Tests for SyncManager class."""

    def test_sync_manager_init(self):
        """Test SyncManager initialization."""
        with patch("valence.federation.sync.get_federation_config") as mock_settings:
            mock_settings.return_value = MagicMock()
            manager = SyncManager()
            assert manager._running is False
            assert manager._sync_task is None

    @pytest.mark.asyncio
    async def test_sync_manager_start_stop(self):
        """Test SyncManager start and stop."""
        with patch("valence.federation.sync.get_federation_config") as mock_settings:
            mock_settings.return_value = MagicMock(federation_sync_interval_seconds=0.1)
            manager = SyncManager()

            # Mock the sync methods to avoid actual work
            manager._process_outbound_queue = AsyncMock()
            manager._sync_with_peers = AsyncMock()

            await manager.start()
            assert manager._running is True
            assert manager._sync_task is not None

            # Let it run briefly
            await asyncio.sleep(0.05)

            await manager.stop()
            assert manager._running is False

    @pytest.mark.asyncio
    async def test_sync_manager_already_running(self):
        """Test that start() does nothing if already running."""
        with patch("valence.federation.sync.get_federation_config") as mock_settings:
            mock_settings.return_value = MagicMock(federation_sync_interval_seconds=0.1)
            manager = SyncManager()
            manager._running = True

            await manager.start()
            # Should not create a new task
            assert manager._sync_task is None

    @pytest.mark.asyncio
    async def test_sync_manager_process_outbound_queue(self):
        """Test processing outbound queue."""
        with (
            patch("valence.federation.sync.get_federation_config") as mock_settings,
            patch("valence.federation.sync.get_cursor") as mock_cursor_ctx,
            patch("valence.federation.sync.get_pending_sync_items"),
            patch("valence.federation.sync.get_node_trust"),
        ):
            mock_settings.return_value = MagicMock(federation_sync_interval_seconds=60)

            # Mock cursor
            cursor = MagicMock()
            cursor.fetchall.return_value = []  # No active nodes
            mock_cursor_ctx.return_value.__enter__ = MagicMock(return_value=cursor)
            mock_cursor_ctx.return_value.__exit__ = MagicMock(return_value=False)

            manager = SyncManager()
            await manager._process_outbound_queue()

            # Should query for active nodes
            cursor.execute.assert_called()

    @pytest.mark.asyncio
    async def test_sync_manager_belief_to_federated(self):
        """Test converting belief to federated format."""
        with (
            patch("valence.federation.sync.get_federation_config") as mock_settings,
            patch("valence.federation.identity.sign_belief_content") as mock_sign,
        ):
            mock_settings.return_value = MagicMock(
                federation_node_did="did:vkb:web:test.example.com",
                federation_private_key="0123456789abcdef",
                port=8080,
            )
            mock_sign.return_value = "mock_signature"

            manager = SyncManager()
            row = {
                "id": uuid4(),
                "federation_id": None,
                "content": "Test belief",
                "confidence": {"overall": 0.8},
                "domain_path": ["test"],
                "visibility": "federated",
                "share_level": "belief_only",
                "valid_from": datetime.now(),
                "valid_until": None,
            }

            result = manager._belief_to_federated(row)

            assert result is not None
            assert result["content"] == "Test belief"
            assert result["origin_node_did"] == "did:vkb:web:test.example.com"
            assert "origin_signature" in result


# =============================================================================
# VECTOR CLOCK TESTS
# =============================================================================


class TestVectorClock:
    """Tests for vector clock operations."""

    def test_update_vector_clock(self, mock_get_cursor):
        """Test updating vector clock."""
        node_id = uuid4()
        mock_get_cursor.fetchone.return_value = {"vector_clock": {"peer1": 5}}

        result = update_vector_clock(node_id, "peer1", 10)

        assert result == {"peer1": 10}

    def test_update_vector_clock_new_peer(self, mock_get_cursor):
        """Test updating vector clock with new peer."""
        node_id = uuid4()
        mock_get_cursor.fetchone.return_value = {"vector_clock": {"peer1": 5}}

        result = update_vector_clock(node_id, "peer2", 3)

        assert result == {"peer1": 5, "peer2": 3}

    def test_update_vector_clock_no_existing_state(self, mock_get_cursor):
        """Test updating vector clock when no state exists."""
        node_id = uuid4()
        mock_get_cursor.fetchone.return_value = None

        result = update_vector_clock(node_id, "peer1", 1)

        assert result == {"peer1": 1}

    def test_update_vector_clock_error(self, mock_get_cursor):
        """Test updating vector clock when error occurs."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = update_vector_clock(uuid4(), "peer1", 1)

        assert result == {}


class TestCompareVectorClocks:
    """Tests for compare_vector_clocks function."""

    def test_compare_equal(self):
        """Test comparing equal clocks."""
        clock_a = {"p1": 1, "p2": 2}
        clock_b = {"p1": 1, "p2": 2}

        result = compare_vector_clocks(clock_a, clock_b)

        assert result == "equal"

    def test_compare_a_before_b(self):
        """Test when A happened before B."""
        clock_a = {"p1": 1, "p2": 2}
        clock_b = {"p1": 2, "p2": 3}

        result = compare_vector_clocks(clock_a, clock_b)

        assert result == "a_before_b"

    def test_compare_b_before_a(self):
        """Test when B happened before A."""
        clock_a = {"p1": 3, "p2": 4}
        clock_b = {"p1": 1, "p2": 2}

        result = compare_vector_clocks(clock_a, clock_b)

        assert result == "b_before_a"

    def test_compare_concurrent(self):
        """Test concurrent clocks (conflict)."""
        clock_a = {"p1": 2, "p2": 1}
        clock_b = {"p1": 1, "p2": 2}

        result = compare_vector_clocks(clock_a, clock_b)

        assert result == "concurrent"

    def test_compare_different_keys(self):
        """Test clocks with different keys."""
        clock_a = {"p1": 1}
        clock_b = {"p2": 1}

        result = compare_vector_clocks(clock_a, clock_b)

        assert result == "concurrent"

    def test_compare_empty_clocks(self):
        """Test comparing empty clocks."""
        result = compare_vector_clocks({}, {})

        assert result == "equal"


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestTriggerSync:
    """Tests for trigger_sync function."""

    @pytest.mark.asyncio
    async def test_trigger_sync_all_nodes(self):
        """Test triggering sync for all nodes."""
        with (
            patch("valence.federation.sync.get_federation_config") as mock_settings,
            patch("valence.federation.sync.SyncManager") as MockManager,  # noqa: N806
        ):
            mock_settings.return_value = MagicMock()
            mock_manager = MagicMock()
            mock_manager._sync_with_peers = AsyncMock()
            MockManager.return_value = mock_manager

            result = await trigger_sync(None)

            assert result["success"] is True
            mock_manager._sync_with_peers.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_sync_specific_node(self):
        """Test triggering sync for specific node."""
        node_id = uuid4()

        with (
            patch("valence.federation.sync.get_federation_config") as mock_settings,
            patch("valence.federation.sync.get_node_by_id") as mock_get_node,
            patch("valence.federation.sync.get_node_trust") as mock_get_trust,
            patch("valence.federation.sync.get_sync_state") as mock_get_state,
            patch("valence.federation.sync.SyncManager") as MockManager,  # noqa: N806
        ):
            mock_settings.return_value = MagicMock()
            mock_node = MagicMock()
            mock_node.did = "did:vkb:web:test.example.com"
            mock_node.federation_endpoint = "https://test.example.com/federation"
            mock_get_node.return_value = mock_node
            mock_get_trust.return_value = MagicMock(overall=0.5)
            mock_get_state.return_value = MagicMock(last_received_cursor=None, last_sync_at=None)

            mock_manager = MagicMock()
            mock_manager._pull_from_node = AsyncMock()
            MockManager.return_value = mock_manager

            result = await trigger_sync(node_id)

            assert result["success"] is True
            assert result["synced_with"] == mock_node.did

    @pytest.mark.asyncio
    async def test_trigger_sync_node_not_found(self):
        """Test triggering sync when node not found."""
        with (
            patch("valence.federation.sync.get_federation_config") as mock_settings,
            patch("valence.federation.sync.get_node_by_id") as mock_get_node,
        ):
            mock_settings.return_value = MagicMock()
            mock_get_node.return_value = None

            result = await trigger_sync(uuid4())

            assert result["success"] is False
            assert "not found" in result["error"].lower()


class TestGetSyncStatus:
    """Tests for get_sync_status function."""

    def test_get_sync_status_success(self, mock_get_cursor):
        """Test getting sync status."""
        mock_get_cursor.fetchone.return_value = {
            "total_nodes": 5,
            "syncing": 1,
            "errors": 0,
            "total_sent": 100,
            "total_received": 50,
            "last_sync": datetime.now(),
        }

        result = get_sync_status()

        assert result["total_nodes"] == 5
        assert result["syncing"] == 1
        assert result["errors"] == 0
        assert result["total_beliefs_sent"] == 100
        assert result["total_beliefs_received"] == 50

    def test_get_sync_status_empty(self, mock_get_cursor):
        """Test getting sync status when no nodes."""
        mock_get_cursor.fetchone.return_value = {
            "total_nodes": None,
            "syncing": None,
            "errors": None,
            "total_sent": None,
            "total_received": None,
            "last_sync": None,
        }

        result = get_sync_status()

        assert result["total_nodes"] == 0
        assert result["last_sync"] is None

    def test_get_sync_status_error(self, mock_get_cursor):
        """Test getting sync status when error occurs."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = get_sync_status()

        assert "error" in result


# =============================================================================
# VECTOR CLOCK INTEGRATION TESTS (Issue #234)
# =============================================================================


class TestVectorClockIntegration:
    """Tests for vector clock conflict detection in sync operations."""

    @pytest.mark.asyncio
    async def test_pull_from_peer_detects_concurrent_clocks(self):
        """Test that _pull_from_peer detects concurrent vector clocks."""
        node_id = uuid4()

        with (
            patch("valence.federation.sync.get_federation_config") as mock_settings,
            patch("valence.federation.sync.get_sync_state") as mock_get_state,
            patch("valence.federation.sync.update_sync_state"),
            patch("valence.federation.sync.mark_node_active"),
            patch("valence.federation.sync.get_cursor"),
            patch("aiohttp.ClientSession") as mock_session_class,
        ):
            mock_settings.return_value = MagicMock(
                federation_sync_interval_seconds=60,
                federation_node_did="did:vkb:web:local",
                federation_private_key=None,
            )

            # Local clock: p1=2, p2=1
            mock_state = MagicMock()
            mock_state.vector_clock = {"p1": 2, "p2": 1}
            mock_get_state.return_value = mock_state

            # Peer clock: p1=1, p2=2 (concurrent!)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "changes": [],
                    "cursor": None,
                    "vector_clock": {"p1": 1, "p2": 2},
                }
            )

            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock()

            manager = SyncManager()

            # Capture logging
            with patch("valence.federation.sync.logger") as mock_logger:
                await manager._pull_from_node(
                    node_id=node_id,
                    federation_endpoint="https://peer.example.com/federation",
                    trust_level=0.5,
                    cursor=None,
                    last_sync=None,
                )

                # Should log warning about concurrent clocks
                mock_logger.warning.assert_called()
                warning_msg = str(mock_logger.warning.call_args)
                assert "concurrent" in warning_msg.lower() or "Concurrent" in warning_msg

    @pytest.mark.asyncio
    async def test_pull_from_peer_no_conflict_when_clocks_ordered(self):
        """Test that _pull_from_peer doesn't flag conflict for ordered clocks."""
        node_id = uuid4()

        with (
            patch("valence.federation.sync.get_federation_config") as mock_settings,
            patch("valence.federation.sync.get_sync_state") as mock_get_state,
            patch("valence.federation.sync.update_sync_state"),
            patch("valence.federation.sync.mark_node_active"),
            patch("valence.federation.sync.get_cursor"),
            patch("aiohttp.ClientSession") as mock_session_class,
        ):
            mock_settings.return_value = MagicMock(
                federation_sync_interval_seconds=60,
                federation_node_did="did:vkb:web:local",
                federation_private_key=None,
            )

            # Local clock: p1=1, p2=1
            mock_state = MagicMock()
            mock_state.vector_clock = {"p1": 1, "p2": 1}
            mock_get_state.return_value = mock_state

            # Peer clock: p1=2, p2=2 (peer is ahead, no conflict)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "changes": [],
                    "cursor": None,
                    "vector_clock": {"p1": 2, "p2": 2},
                }
            )

            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock()

            manager = SyncManager()

            with patch("valence.federation.sync.logger") as mock_logger:
                await manager._pull_from_node(
                    node_id=node_id,
                    federation_endpoint="https://peer.example.com/federation",
                    trust_level=0.5,
                    cursor=None,
                    last_sync=None,
                )

                # Should NOT log warning about concurrent clocks
                for call in mock_logger.warning.call_args_list:
                    assert "concurrent" not in str(call).lower()

    @pytest.mark.asyncio
    async def test_process_sync_changes_logs_conflict_flag(self):
        """Test that _process_sync_changes handles conflict_detected flag."""
        node_id = uuid4()

        with (
            patch("valence.federation.sync.get_federation_config") as mock_settings,
            patch("valence.federation.sync.handle_share_belief") as mock_handler,
        ):
            mock_settings.return_value = MagicMock(
                federation_node_did="did:vkb:web:local",
                federation_private_key=None,
            )
            mock_handler.return_value = MagicMock(accepted=1)

            manager = SyncManager()
            changes = [
                {"type": "belief_created", "belief": {"content": "test", "confidence": {"overall": 0.8}}},
            ]

            with patch("valence.federation.sync.logger") as mock_logger:
                result = await manager._process_sync_changes(
                    node_id=node_id,
                    trust_level=0.5,
                    changes=changes,
                    conflict_detected=True,
                )

                # Should log about processing with conflict detected
                mock_logger.info.assert_called()
                info_msg = str(mock_logger.info.call_args)
                assert "conflict_detected=True" in info_msg

                assert result == 1  # One belief processed
