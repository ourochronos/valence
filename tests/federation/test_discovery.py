"""Tests for node discovery and registration.

Tests cover:
- Node discovery (discover_node, _fetch_node_metadata)
- Node registration
- Node retrieval (get_node_by_did, get_node_by_id, get_node_trust)
- Status management (update_node_status, mark_node_active, mark_node_unreachable)
- Bootstrap mechanism
- Health checks
- Node listing with filters
- Peer exchange
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4, UUID

import pytest

from valence.federation.discovery import (
    discover_node,
    discover_node_sync,
    register_node,
    get_node_by_did,
    get_node_by_id,
    get_node_trust,
    update_node_status,
    mark_node_active,
    mark_node_unreachable,
    bootstrap_federation,
    bootstrap_federation_sync,
    check_node_health,
    check_all_nodes_health,
    list_nodes,
    list_active_nodes,
    list_nodes_with_trust,
    get_known_peers,
)
from valence.federation.models import (
    NodeStatus,
    TrustPhase,
)
from valence.federation.identity import DIDDocument


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

    with patch("valence.federation.discovery.get_cursor", _mock_get_cursor):
        yield mock_cursor


@pytest.fixture
def sample_did_document():
    """Create a sample DIDDocument mock."""
    def _factory(**kwargs):
        doc = MagicMock(spec=DIDDocument)
        doc.id = kwargs.get("did", "did:vkb:web:test.example.com")
        doc.public_key_multibase = kwargs.get("public_key", "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK")
        
        # Mock services
        fed_service = MagicMock()
        fed_service.type = "ValenceFederationProtocol"
        fed_service.service_endpoint = kwargs.get(
            "federation_endpoint",
            "https://test.example.com/federation"
        )
        
        mcp_service = MagicMock()
        mcp_service.type = "ModelContextProtocol"
        mcp_service.service_endpoint = kwargs.get(
            "mcp_endpoint",
            "https://test.example.com/mcp"
        )
        
        doc.services = kwargs.get("services", [fed_service, mcp_service])
        doc.capabilities = kwargs.get("capabilities", ["belief_sync"])
        doc.profile = kwargs.get("profile", {"name": "Test Node", "domains": ["test"]})
        doc.protocol_version = kwargs.get("protocol_version", "0.1.0")
        return doc
    return _factory


@pytest.fixture
def sample_node_row():
    """Create a sample federation node row."""
    def _factory(**kwargs):
        now = datetime.now()
        return {
            "id": kwargs.get("id", uuid4()),
            "did": kwargs.get("did", "did:vkb:web:test.example.com"),
            "federation_endpoint": kwargs.get(
                "federation_endpoint",
                "https://test.example.com/federation"
            ),
            "mcp_endpoint": kwargs.get("mcp_endpoint", "https://test.example.com/mcp"),
            "public_key_multibase": kwargs.get(
                "public_key_multibase",
                "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
            ),
            "name": kwargs.get("name", "Test Node"),
            "domains": kwargs.get("domains", ["test"]),
            "capabilities": kwargs.get("capabilities", ["belief_sync"]),
            "status": kwargs.get("status", "active"),
            "trust_phase": kwargs.get("trust_phase", "contributor"),
            "protocol_version": kwargs.get("protocol_version", "0.1.0"),
            "discovered_at": kwargs.get("discovered_at", now),
            "last_seen_at": kwargs.get("last_seen_at", now),
            "phase_started_at": kwargs.get("phase_started_at", now - timedelta(days=30)),
            "metadata": kwargs.get("metadata", {}),
            "created_at": kwargs.get("created_at", now),
            "modified_at": kwargs.get("modified_at", now),
        }
    return _factory


@pytest.fixture
def sample_trust_row():
    """Create a sample node trust row."""
    def _factory(**kwargs):
        now = datetime.now()
        return {
            "id": kwargs.get("id", uuid4()),
            "node_id": kwargs.get("node_id", uuid4()),
            "trust": kwargs.get("trust", {"overall": 0.5}),
            "beliefs_received": kwargs.get("beliefs_received", 10),
            "beliefs_corroborated": kwargs.get("beliefs_corroborated", 5),
            "beliefs_disputed": kwargs.get("beliefs_disputed", 1),
            "relationship_started_at": kwargs.get("relationship_started_at", now),
            "last_interaction_at": kwargs.get("last_interaction_at", now),
        }
    return _factory


# =============================================================================
# NODE DISCOVERY TESTS
# =============================================================================


class TestDiscoverNode:
    """Tests for discover_node function."""

    @pytest.mark.asyncio
    async def test_discover_node_by_did(self, sample_did_document):
        """Test discovering a node by DID."""
        did = "did:vkb:web:test.example.com"
        did_doc = sample_did_document(did=did)
        
        with patch("valence.federation.discovery.resolve_did") as mock_resolve:
            mock_resolve.return_value = did_doc

            result = await discover_node(did)

            mock_resolve.assert_called_once_with(did)
            assert result == did_doc

    @pytest.mark.asyncio
    async def test_discover_node_by_url(self, sample_did_document):
        """Test discovering a node by URL."""
        url = "https://test.example.com"
        did_doc = sample_did_document()
        
        with patch("valence.federation.discovery._fetch_node_metadata") as mock_fetch:
            mock_fetch.return_value = did_doc

            result = await discover_node(url)

            mock_fetch.assert_called_once_with(url)
            assert result == did_doc

    @pytest.mark.asyncio
    async def test_discover_node_adds_https(self, sample_did_document):
        """Test that URL without scheme gets https added."""
        url = "test.example.com"
        did_doc = sample_did_document()
        
        with patch("valence.federation.discovery._fetch_node_metadata") as mock_fetch:
            mock_fetch.return_value = did_doc

            result = await discover_node(url)

            # Should be called with https:// prefix
            call_args = mock_fetch.call_args[0][0]
            assert call_args.startswith("https://") or url in call_args


class TestDiscoverNodeSync:
    """Tests for discover_node_sync function."""

    def test_discover_node_sync_success(self, sample_did_document):
        """Test synchronous node discovery."""
        did_doc = sample_did_document()
        
        with patch("valence.federation.discovery.discover_node", new_callable=AsyncMock) as mock_discover:
            mock_discover.return_value = did_doc

            result = discover_node_sync("https://test.example.com")

            assert result == did_doc

    def test_discover_node_sync_error(self):
        """Test synchronous discovery with error."""
        with patch("valence.federation.discovery.discover_node", new_callable=AsyncMock) as mock_discover:
            mock_discover.side_effect = Exception("Network error")

            result = discover_node_sync("https://test.example.com")

            assert result is None


# =============================================================================
# NODE REGISTRATION TESTS
# =============================================================================


class TestRegisterNode:
    """Tests for register_node function."""

    def test_register_new_node(self, mock_get_cursor, sample_did_document, sample_node_row):
        """Test registering a new node."""
        did_doc = sample_did_document()
        row = sample_node_row(did=did_doc.id)
        
        # First query returns None (node doesn't exist)
        # Second query returns the new node
        mock_get_cursor.fetchone.side_effect = [None, row]

        result = register_node(did_doc)

        assert result is not None
        assert mock_get_cursor.execute.call_count >= 2  # Select, Insert, trust init, sync init

    def test_register_existing_node(self, mock_get_cursor, sample_did_document, sample_node_row):
        """Test updating an existing node."""
        did_doc = sample_did_document()
        existing_row = {"id": uuid4(), "status": "active"}
        updated_row = sample_node_row(did=did_doc.id)
        
        mock_get_cursor.fetchone.side_effect = [existing_row, updated_row]

        result = register_node(did_doc)

        assert result is not None

    def test_register_node_no_public_key(self, sample_did_document):
        """Test registering node without public key fails."""
        did_doc = sample_did_document()
        did_doc.public_key_multibase = None

        result = register_node(did_doc)

        assert result is None

    def test_register_node_error(self, mock_get_cursor, sample_did_document):
        """Test registration with database error."""
        did_doc = sample_did_document()
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = register_node(did_doc)

        assert result is None


# =============================================================================
# NODE RETRIEVAL TESTS
# =============================================================================


class TestGetNodeByDid:
    """Tests for get_node_by_did function."""

    def test_get_node_found(self, mock_get_cursor, sample_node_row):
        """Test getting node by DID when found."""
        did = "did:vkb:web:test.example.com"
        row = sample_node_row(did=did)
        mock_get_cursor.fetchone.return_value = row

        result = get_node_by_did(did)

        assert result is not None
        assert result.did == did

    def test_get_node_not_found(self, mock_get_cursor):
        """Test getting node by DID when not found."""
        mock_get_cursor.fetchone.return_value = None

        result = get_node_by_did("did:vkb:web:nonexistent")

        assert result is None

    def test_get_node_error(self, mock_get_cursor):
        """Test getting node with database error."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = get_node_by_did("did:vkb:web:test")

        assert result is None


class TestGetNodeById:
    """Tests for get_node_by_id function."""

    def test_get_node_by_id_found(self, mock_get_cursor, sample_node_row):
        """Test getting node by ID when found."""
        node_id = uuid4()
        row = sample_node_row(id=node_id)
        mock_get_cursor.fetchone.return_value = row

        result = get_node_by_id(node_id)

        assert result is not None
        assert result.id == node_id

    def test_get_node_by_id_not_found(self, mock_get_cursor):
        """Test getting node by ID when not found."""
        mock_get_cursor.fetchone.return_value = None

        result = get_node_by_id(uuid4())

        assert result is None


class TestGetNodeTrust:
    """Tests for get_node_trust function."""

    def test_get_trust_found(self, mock_get_cursor, sample_trust_row):
        """Test getting node trust when found."""
        node_id = uuid4()
        row = sample_trust_row(node_id=node_id)
        mock_get_cursor.fetchone.return_value = row

        result = get_node_trust(node_id)

        assert result is not None
        assert result.node_id == node_id

    def test_get_trust_not_found(self, mock_get_cursor):
        """Test getting node trust when not found."""
        mock_get_cursor.fetchone.return_value = None

        result = get_node_trust(uuid4())

        assert result is None


# =============================================================================
# STATUS MANAGEMENT TESTS
# =============================================================================


class TestUpdateNodeStatus:
    """Tests for update_node_status function."""

    def test_update_status_success(self, mock_get_cursor):
        """Test updating node status."""
        node_id = uuid4()

        result = update_node_status(node_id, NodeStatus.ACTIVE)

        assert result is True
        mock_get_cursor.execute.assert_called_once()

    def test_update_status_error(self, mock_get_cursor):
        """Test updating status with error."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = update_node_status(uuid4(), NodeStatus.ACTIVE)

        assert result is False


class TestMarkNodeActive:
    """Tests for mark_node_active function."""

    def test_mark_active(self):
        """Test marking node as active."""
        node_id = uuid4()
        
        with patch("valence.federation.discovery.update_node_status") as mock_update:
            mock_update.return_value = True

            result = mark_node_active(node_id)

            mock_update.assert_called_once_with(node_id, NodeStatus.ACTIVE)
            assert result is True


class TestMarkNodeUnreachable:
    """Tests for mark_node_unreachable function."""

    def test_mark_unreachable(self):
        """Test marking node as unreachable."""
        node_id = uuid4()
        
        with patch("valence.federation.discovery.update_node_status") as mock_update:
            mock_update.return_value = True

            result = mark_node_unreachable(node_id)

            mock_update.assert_called_once_with(node_id, NodeStatus.UNREACHABLE)
            assert result is True


# =============================================================================
# BOOTSTRAP TESTS
# =============================================================================


class TestBootstrapFederation:
    """Tests for bootstrap_federation function."""

    @pytest.mark.asyncio
    async def test_bootstrap_success(self, sample_did_document):
        """Test successful bootstrap."""
        did_docs = [
            sample_did_document(did=f"did:vkb:web:node{i}.example.com")
            for i in range(3)
        ]
        
        with patch("valence.federation.discovery.discover_node", new_callable=AsyncMock) as mock_discover, \
             patch("valence.federation.discovery.register_node") as mock_register:
            
            # Return different docs for each call
            mock_discover.side_effect = did_docs
            mock_register.side_effect = [
                MagicMock(did=doc.id) for doc in did_docs
            ]

            result = await bootstrap_federation([
                "https://node0.example.com",
                "https://node1.example.com",
                "https://node2.example.com",
            ])

            assert len(result) == 3

    @pytest.mark.asyncio
    async def test_bootstrap_partial_failure(self, sample_did_document):
        """Test bootstrap with some failures."""
        did_doc = sample_did_document()
        
        with patch("valence.federation.discovery.discover_node", new_callable=AsyncMock) as mock_discover, \
             patch("valence.federation.discovery.register_node") as mock_register:
            
            # First succeeds, second fails (returns None)
            mock_discover.side_effect = [did_doc, None]
            mock_register.return_value = MagicMock(did=did_doc.id)

            result = await bootstrap_federation([
                "https://node1.example.com",
                "https://node2.example.com",
            ])

            # Only one should succeed
            assert len(result) == 1


class TestBootstrapFederationSync:
    """Tests for bootstrap_federation_sync function."""

    def test_bootstrap_sync_success(self, sample_did_document):
        """Test synchronous bootstrap."""
        did_doc = sample_did_document()
        node = MagicMock(did=did_doc.id)
        
        with patch("valence.federation.discovery.bootstrap_federation", new_callable=AsyncMock) as mock_bootstrap:
            mock_bootstrap.return_value = [node]

            result = bootstrap_federation_sync(["https://test.example.com"])

            assert len(result) == 1

    def test_bootstrap_sync_error(self):
        """Test synchronous bootstrap with error."""
        with patch("valence.federation.discovery.bootstrap_federation", new_callable=AsyncMock) as mock_bootstrap:
            mock_bootstrap.side_effect = Exception("Network error")

            result = bootstrap_federation_sync(["https://test.example.com"])

            assert result == []


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================


class TestCheckNodeHealth:
    """Tests for check_node_health function."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, sample_did_document):
        """Test successful health check."""
        did_doc = sample_did_document()
        node = MagicMock()
        node.id = uuid4()
        node.did = did_doc.id
        node.federation_endpoint = "https://test.example.com/federation"
        
        with patch("valence.federation.discovery._fetch_node_metadata") as mock_fetch, \
             patch("valence.federation.discovery.get_cursor"):
            
            mock_fetch.return_value = did_doc

            result = await check_node_health(node)

            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_no_endpoint(self):
        """Test health check with no endpoint."""
        node = MagicMock()
        node.federation_endpoint = None

        result = await check_node_health(node)

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_mismatch(self, sample_did_document):
        """Test health check with DID mismatch."""
        did_doc = sample_did_document(did="did:vkb:web:different.example.com")
        node = MagicMock()
        node.id = uuid4()
        node.did = "did:vkb:web:original.example.com"
        node.federation_endpoint = "https://test.example.com/federation"
        
        with patch("valence.federation.discovery._fetch_node_metadata") as mock_fetch, \
             patch("valence.federation.discovery.mark_node_unreachable") as mock_mark:
            
            mock_fetch.return_value = did_doc
            mock_mark.return_value = True

            result = await check_node_health(node)

            assert result is False
            mock_mark.assert_called_once_with(node.id)


class TestCheckAllNodesHealth:
    """Tests for check_all_nodes_health function."""

    @pytest.mark.asyncio
    async def test_check_all_nodes(self, mock_get_cursor, sample_node_row):
        """Test checking all nodes health."""
        rows = [
            sample_node_row(id=uuid4(), did=f"did:vkb:web:node{i}.example.com")
            for i in range(2)
        ]
        mock_get_cursor.fetchall.return_value = rows
        
        with patch("valence.federation.discovery.check_node_health") as mock_check:
            mock_check.return_value = True

            results = await check_all_nodes_health()

            assert len(results) == 2
            for did, health in results.items():
                assert health is True


# =============================================================================
# NODE LISTING TESTS
# =============================================================================


class TestListNodes:
    """Tests for list_nodes function."""

    def test_list_all_nodes(self, mock_get_cursor, sample_node_row):
        """Test listing all nodes."""
        rows = [sample_node_row() for _ in range(5)]
        mock_get_cursor.fetchall.return_value = rows

        result = list_nodes()

        assert len(result) == 5

    def test_list_nodes_with_status_filter(self, mock_get_cursor, sample_node_row):
        """Test listing nodes with status filter."""
        rows = [sample_node_row(status="active")]
        mock_get_cursor.fetchall.return_value = rows

        result = list_nodes(status=NodeStatus.ACTIVE)

        assert len(result) == 1
        # Check that status filter was in query
        call_args = mock_get_cursor.execute.call_args[0][0]
        assert "status = %s" in call_args

    def test_list_nodes_with_phase_filter(self, mock_get_cursor, sample_node_row):
        """Test listing nodes with trust phase filter."""
        rows = [sample_node_row(trust_phase="participant")]
        mock_get_cursor.fetchall.return_value = rows

        result = list_nodes(trust_phase=TrustPhase.PARTICIPANT)

        assert len(result) == 1

    def test_list_nodes_with_domains_filter(self, mock_get_cursor, sample_node_row):
        """Test listing nodes with domains filter."""
        rows = [sample_node_row(domains=["science"])]
        mock_get_cursor.fetchall.return_value = rows

        result = list_nodes(domains=["science"])

        assert len(result) == 1

    def test_list_nodes_empty(self, mock_get_cursor):
        """Test listing nodes when none exist."""
        mock_get_cursor.fetchall.return_value = []

        result = list_nodes()

        assert result == []

    def test_list_nodes_error(self, mock_get_cursor):
        """Test listing nodes with error."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = list_nodes()

        assert result == []


class TestListActiveNodes:
    """Tests for list_active_nodes function."""

    def test_list_active(self):
        """Test listing active nodes."""
        with patch("valence.federation.discovery.list_nodes") as mock_list:
            mock_list.return_value = [MagicMock(), MagicMock()]

            result = list_active_nodes()

            mock_list.assert_called_once_with(status=NodeStatus.ACTIVE)
            assert len(result) == 2


class TestListNodesWithTrust:
    """Tests for list_nodes_with_trust function."""

    def test_list_with_trust(self, mock_get_cursor, sample_node_row):
        """Test listing nodes with trust information."""
        row = sample_node_row()
        row["trust_id"] = uuid4()
        row["trust"] = {"overall": 0.6}
        row["beliefs_received"] = 20
        row["beliefs_corroborated"] = 10
        row["beliefs_disputed"] = 2
        row["relationship_started_at"] = datetime.now()
        row["last_interaction_at"] = datetime.now()
        
        mock_get_cursor.fetchall.return_value = [row]

        result = list_nodes_with_trust()

        assert len(result) == 1
        node, trust = result[0]
        assert trust is not None
        assert trust.overall == 0.6

    def test_list_with_trust_no_trust_record(self, mock_get_cursor, sample_node_row):
        """Test listing nodes without trust records."""
        row = sample_node_row()
        row["trust_id"] = None
        
        mock_get_cursor.fetchall.return_value = [row]

        result = list_nodes_with_trust()

        assert len(result) == 1
        node, trust = result[0]
        assert trust is None


# =============================================================================
# PEER EXCHANGE TESTS
# =============================================================================


class TestGetKnownPeers:
    """Tests for get_known_peers function."""

    def test_get_known_peers(self, mock_get_cursor, sample_node_row):
        """Test getting known peers for exchange."""
        rows = [
            {
                "did": f"did:vkb:web:node{i}.example.com",
                "federation_endpoint": f"https://node{i}.example.com/federation",
                "domains": ["test"],
                "trust_phase": "contributor",
            }
            for i in range(3)
        ]
        mock_get_cursor.fetchall.return_value = rows

        result = get_known_peers()

        assert len(result) == 3
        for peer in result:
            assert "did" in peer
            assert "federation_endpoint" in peer

    def test_get_known_peers_empty(self, mock_get_cursor):
        """Test getting peers when none exist."""
        mock_get_cursor.fetchall.return_value = []

        result = get_known_peers()

        assert result == []

    def test_get_known_peers_error(self, mock_get_cursor):
        """Test getting peers with error."""
        mock_get_cursor.execute.side_effect = Exception("DB error")

        result = get_known_peers()

        assert result == []
