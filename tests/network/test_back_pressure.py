"""Tests for Router Back-Pressure Handling (Issue #110).

These tests verify:
- Router load tracking and calculation
- Back-pressure activation when load exceeds threshold
- Back-pressure release when load drops
- Back-pressure signaling to connected nodes
- Node handling of back-pressure signals
- Router selection avoiding back-pressured routers
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valence.network.router import Connection, QueuedMessage, RouterNode
from valence.network.messages import BackPressureMessage


# =============================================================================
# Unit Tests - BackPressureMessage
# =============================================================================


class TestBackPressureMessage:
    """Tests for the BackPressureMessage dataclass."""

    def test_back_pressure_message_creation(self):
        """Test creating a BackPressureMessage instance."""
        msg = BackPressureMessage(
            active=True,
            load_pct=85.5,
            retry_after_ms=2000,
            reason="High connection load",
        )
        assert msg.type == "back_pressure"
        assert msg.active is True
        assert msg.load_pct == 85.5
        assert msg.retry_after_ms == 2000
        assert msg.reason == "High connection load"

    def test_back_pressure_message_defaults(self):
        """Test BackPressureMessage default values."""
        msg = BackPressureMessage()
        assert msg.type == "back_pressure"
        assert msg.active is True
        assert msg.load_pct == 0.0
        assert msg.retry_after_ms == 1000
        assert msg.reason == ""

    def test_back_pressure_message_to_dict(self):
        """Test serialization to dict."""
        msg = BackPressureMessage(
            active=True,
            load_pct=90.0,
            retry_after_ms=5000,
            reason="Queue full",
        )
        d = msg.to_dict()
        assert d["type"] == "back_pressure"
        assert d["active"] is True
        assert d["load_pct"] == 90.0
        assert d["retry_after_ms"] == 5000
        assert d["reason"] == "Queue full"

    def test_back_pressure_message_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "type": "back_pressure",
            "active": False,
            "load_pct": 45.0,
            "retry_after_ms": 500,
            "reason": "Load normal",
        }
        msg = BackPressureMessage.from_dict(d)
        assert msg.active is False
        assert msg.load_pct == 45.0
        assert msg.retry_after_ms == 500
        assert msg.reason == "Load normal"


# =============================================================================
# Unit Tests - RouterNode Load Tracking
# =============================================================================


class TestRouterLoadTracking:
    """Tests for RouterNode load calculation."""

    def test_get_load_pct_empty(self):
        """Test load calculation with no connections or queues."""
        router = RouterNode()
        load = router.get_load_pct()
        assert load == 0.0

    def test_get_load_pct_connections_only(self):
        """Test load calculation based on connections."""
        router = RouterNode(max_connections=100)
        
        # Add 50 connections (50% of max)
        for i in range(50):
            ws = MagicMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        load = router.get_load_pct()
        # 50% connection load * 0.7 weight = 35%
        assert load == pytest.approx(35.0, rel=0.01)

    def test_get_load_pct_queue_only(self):
        """Test load calculation based on queued messages."""
        router = RouterNode()
        router.MAX_QUEUE_SIZE = 1000
        
        # Add 500 queued messages across multiple nodes
        for i in range(50):
            router.offline_queues[f"node-{i}"] = [
                QueuedMessage(f"msg-{i}-{j}", "payload", time.time(), 5)
                for j in range(10)
            ]
        
        load = router.get_load_pct()
        # 500/1000 = 50% queue load * 0.3 weight = 15%
        assert load == pytest.approx(15.0, rel=0.01)

    def test_get_load_pct_combined(self):
        """Test combined load from connections and queues."""
        router = RouterNode(max_connections=100)
        router.MAX_QUEUE_SIZE = 1000
        
        # 80 connections (80% of max)
        for i in range(80):
            ws = MagicMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        # 600 queued messages (60% of max)
        for i in range(60):
            router.offline_queues[f"offline-{i}"] = [
                QueuedMessage(f"msg-{i}-{j}", "payload", time.time(), 5)
                for j in range(10)
            ]
        
        load = router.get_load_pct()
        # (80% * 0.7) + (60% * 0.3) = 56 + 18 = 74%
        assert load == pytest.approx(74.0, rel=0.01)


class TestRouterBackPressureSettings:
    """Tests for RouterNode back-pressure settings."""

    def test_default_back_pressure_settings(self):
        """Test default back-pressure thresholds."""
        router = RouterNode()
        assert router.back_pressure_threshold == 80.0
        assert router.back_pressure_release_threshold == 60.0
        assert router.back_pressure_retry_ms == 1000
        assert router._back_pressure_active is False

    def test_custom_back_pressure_settings(self):
        """Test custom back-pressure thresholds."""
        router = RouterNode(
            back_pressure_threshold=90.0,
            back_pressure_release_threshold=70.0,
            back_pressure_retry_ms=2000,
        )
        assert router.back_pressure_threshold == 90.0
        assert router.back_pressure_release_threshold == 70.0
        assert router.back_pressure_retry_ms == 2000

    def test_is_back_pressure_active_property(self):
        """Test is_back_pressure_active property."""
        router = RouterNode()
        assert router.is_back_pressure_active is False
        
        router._back_pressure_active = True
        assert router.is_back_pressure_active is True


# =============================================================================
# Unit Tests - RouterNode Back-Pressure Activation
# =============================================================================


class TestRouterBackPressureActivation:
    """Tests for RouterNode back-pressure activation/release."""

    @pytest.mark.asyncio
    async def test_check_back_pressure_activates(self):
        """Test that back-pressure activates when load exceeds threshold."""
        # With connection weight of 0.7, need >80/0.7 = 114% connections
        # So we use a lower threshold
        router = RouterNode(max_connections=10, back_pressure_threshold=60.0)
        
        # Add 9 connections (90% connection load * 0.7 weight = 63% total load)
        for i in range(9):
            ws = AsyncMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        assert router._back_pressure_active is False
        
        await router._check_back_pressure()
        
        assert router._back_pressure_active is True

    @pytest.mark.asyncio
    async def test_check_back_pressure_releases(self):
        """Test that back-pressure releases when load drops below threshold."""
        router = RouterNode(
            max_connections=10,
            back_pressure_threshold=80.0,
            back_pressure_release_threshold=50.0,
        )
        router._back_pressure_active = True
        
        # Add 3 connections (30% load from connections alone)
        for i in range(3):
            ws = AsyncMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        await router._check_back_pressure()
        
        assert router._back_pressure_active is False

    @pytest.mark.asyncio
    async def test_check_back_pressure_hysteresis(self):
        """Test that back-pressure has hysteresis (doesn't oscillate)."""
        # Adjusted for weighted load calculation
        # 7 connections = 70% * 0.7 = 49% load
        # Set thresholds accordingly
        router = RouterNode(
            max_connections=10,
            back_pressure_threshold=55.0,  # Will activate at >55%
            back_pressure_release_threshold=35.0,  # Will release at <35%
        )
        
        # 5 connections = 50% * 0.7 = 35% load (exactly at release threshold)
        for i in range(5):
            ws = AsyncMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        # Should NOT activate (35% < 55% threshold)
        await router._check_back_pressure()
        assert router._back_pressure_active is False
        
        # Manually activate
        router._back_pressure_active = True
        
        # Add one more connection to be clearly above release threshold
        ws = AsyncMock()
        router.connections["node-5"] = Connection(
            node_id="node-5",
            websocket=ws,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        # Now 6 connections = 60% * 0.7 = 42% load, above 35% release threshold
        
        # Should NOT release (42% > 35% release threshold)
        await router._check_back_pressure()
        assert router._back_pressure_active is True

    @pytest.mark.asyncio
    async def test_activate_back_pressure_broadcasts(self):
        """Test that activating back-pressure broadcasts to all nodes."""
        router = RouterNode(max_connections=10)
        
        # Add 3 connected nodes
        for i in range(3):
            ws = AsyncMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        await router._activate_back_pressure(85.0)
        
        # All 3 nodes should have been notified
        for conn in router.connections.values():
            conn.websocket.send_json.assert_called_once()
            call_args = conn.websocket.send_json.call_args[0][0]
            assert call_args["type"] == "back_pressure"
            assert call_args["active"] is True
            assert call_args["load_pct"] == 85.0
        
        # Nodes should be tracked
        assert len(router._back_pressure_nodes) == 3

    @pytest.mark.asyncio
    async def test_release_back_pressure_notifies_tracked_nodes(self):
        """Test that releasing back-pressure notifies previously notified nodes."""
        router = RouterNode(max_connections=10)
        
        # Set up some connected nodes
        for i in range(3):
            ws = AsyncMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        # Simulate previous activation
        router._back_pressure_active = True
        router._back_pressure_nodes = {"node-0", "node-1", "node-2"}
        
        await router._release_back_pressure(55.0)
        
        # All 3 nodes should have been notified of release
        for conn in router.connections.values():
            conn.websocket.send_json.assert_called_once()
            call_args = conn.websocket.send_json.call_args[0][0]
            assert call_args["type"] == "back_pressure"
            assert call_args["active"] is False
            assert call_args["load_pct"] == 55.0
        
        # Tracked nodes should be cleared
        assert len(router._back_pressure_nodes) == 0


# =============================================================================
# Unit Tests - RouterNode Endpoints with Back-Pressure
# =============================================================================


class TestRouterEndpointsBackPressure:
    """Tests for RouterNode HTTP endpoints with back-pressure info."""

    @pytest.mark.asyncio
    async def test_handle_health_includes_back_pressure(self):
        """Test health endpoint includes back-pressure status."""
        router = RouterNode()
        router._back_pressure_active = True
        
        # Add some load
        for i in range(5):
            ws = MagicMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        request = MagicMock()
        response = await router.handle_health(request)
        
        data = json.loads(response.body)
        assert "back_pressure" in data
        assert data["back_pressure"]["active"] is True
        assert data["back_pressure"]["load_pct"] > 0

    @pytest.mark.asyncio
    async def test_handle_status_includes_back_pressure_details(self):
        """Test status endpoint includes detailed back-pressure info."""
        router = RouterNode(
            back_pressure_threshold=80.0,
            back_pressure_release_threshold=60.0,
        )
        router._back_pressure_active = True
        router._back_pressure_nodes = {"node-1", "node-2"}
        
        # Mock circuit stats to avoid unrelated initialization issues
        with patch.object(router, 'get_circuit_stats', return_value={"circuits_active": 0}):
            request = MagicMock()
            response = await router.handle_status(request)
            
            data = json.loads(response.body)
            assert "back_pressure" in data
            bp = data["back_pressure"]
            assert bp["active"] is True
            assert bp["threshold"] == 80.0
            assert bp["release_threshold"] == 60.0
            assert bp["nodes_notified"] == 2


# =============================================================================
# Integration Tests - Back-Pressure During Relay
# =============================================================================


class TestBackPressureDuringRelay:
    """Tests for back-pressure triggering during message relay."""

    @pytest.mark.asyncio
    async def test_relay_triggers_back_pressure_check(self):
        """Test that handling a relay checks back-pressure status."""
        # 9 connections = 90% * 0.7 = 63% load, so threshold must be <= 63%
        router = RouterNode(max_connections=10, back_pressure_threshold=60.0)
        
        # Add enough connections to trigger back-pressure
        for i in range(9):
            ws = AsyncMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        assert router._back_pressure_active is False
        
        # Handle a relay message
        await router._handle_relay({
            "message_id": "msg-1",
            "next_hop": "offline-node",
            "payload": "encrypted_data",
            "ttl": 10,
        })
        
        # Back-pressure should now be active
        assert router._back_pressure_active is True

    @pytest.mark.asyncio
    async def test_relay_releases_back_pressure_when_load_drops(self):
        """Test that back-pressure releases when connections drop."""
        router = RouterNode(
            max_connections=10,
            back_pressure_threshold=80.0,
            back_pressure_release_threshold=50.0,
        )
        router._back_pressure_active = True
        
        # Only 3 connections now (30% load)
        for i in range(3):
            ws = AsyncMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        # Handle a relay message
        await router._handle_relay({
            "message_id": "msg-1",
            "next_hop": "offline-node",
            "payload": "encrypted_data",
            "ttl": 10,
        })
        
        # Back-pressure should be released
        assert router._back_pressure_active is False


# =============================================================================
# Unit Tests - Node Back-Pressure Handling
# =============================================================================


class TestNodeBackPressureHandling:
    """Tests for NodeClient handling of back-pressure signals."""

    def test_router_connection_back_pressure_defaults(self):
        """Test RouterConnection back-pressure defaults."""
        from valence.network.node import RouterConnection
        from valence.network.discovery import RouterInfo
        
        router_info = RouterInfo(
            router_id="test-router",
            endpoints=["127.0.0.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        
        # Create minimal connection (can't use real websocket in unit test)
        # Just test the dataclass defaults
        conn = RouterConnection(
            router=router_info,
            websocket=MagicMock(),
            session=MagicMock(),
            connected_at=time.time(),
            last_seen=time.time(),
        )
        
        assert conn.back_pressure_active is False
        assert conn.back_pressure_until == 0.0
        assert conn.back_pressure_retry_ms == 1000

    def test_router_connection_is_under_back_pressure(self):
        """Test is_under_back_pressure property."""
        from valence.network.node import RouterConnection
        from valence.network.discovery import RouterInfo
        
        router_info = RouterInfo(
            router_id="test-router",
            endpoints=["127.0.0.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        
        conn = RouterConnection(
            router=router_info,
            websocket=MagicMock(),
            session=MagicMock(),
            connected_at=time.time(),
            last_seen=time.time(),
        )
        
        # Not under back-pressure by default
        assert conn.is_under_back_pressure is False
        
        # Activate back-pressure
        conn.back_pressure_active = True
        conn.back_pressure_until = time.time() + 10  # 10 seconds from now
        assert conn.is_under_back_pressure is True
        
        # Expired back-pressure
        conn.back_pressure_until = time.time() - 1  # 1 second ago
        assert conn.is_under_back_pressure is False


# =============================================================================
# Integration Tests - Full Back-Pressure Flow
# =============================================================================


class TestBackPressureIntegration:
    """Integration tests for full back-pressure flow."""

    @pytest.mark.asyncio
    async def test_full_back_pressure_flow(self):
        """Test complete back-pressure activation and release cycle."""
        # Adjusted thresholds for weighted load calculation
        # 5 connections = 50% * 0.7 = 35% load
        # 9 connections = 90% * 0.7 = 63% load
        # 3 connections = 30% * 0.7 = 21% load
        router = RouterNode(
            max_connections=10,
            back_pressure_threshold=50.0,  # Activates at 50% load
            back_pressure_release_threshold=25.0,  # Releases at 25% load
        )
        
        # Add initial connections (below threshold: 5 * 0.7 = 35% < 50%)
        for i in range(5):
            ws = AsyncMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        # Verify not under back-pressure
        await router._check_back_pressure()
        assert router._back_pressure_active is False
        
        # Add more connections (above threshold: 9 * 0.7 = 63% > 50%)
        for i in range(5, 9):
            ws = AsyncMock()
            router.connections[f"node-{i}"] = Connection(
                node_id=f"node-{i}",
                websocket=ws,
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        # Check back-pressure - should activate
        await router._check_back_pressure()
        assert router._back_pressure_active is True
        assert len(router._back_pressure_nodes) == 9
        
        # Verify all nodes received back-pressure signal
        for conn in router.connections.values():
            conn.websocket.send_json.assert_called()
            last_call = conn.websocket.send_json.call_args[0][0]
            assert last_call["type"] == "back_pressure"
            assert last_call["active"] is True
        
        # Remove connections (below release threshold)
        # Reset mocks first
        for conn in router.connections.values():
            conn.websocket.reset_mock()
        
        # Remove 6 connections, leaving only 3 (30% * 0.7 = 21% load < 25%)
        for i in range(6):
            del router.connections[f"node-{i}"]
        
        # Check back-pressure - should release
        await router._check_back_pressure()
        assert router._back_pressure_active is False
        assert len(router._back_pressure_nodes) == 0
        
        # Verify remaining nodes received release signal
        for conn in router.connections.values():
            conn.websocket.send_json.assert_called()
            last_call = conn.websocket.send_json.call_args[0][0]
            assert last_call["type"] == "back_pressure"
            assert last_call["active"] is False
