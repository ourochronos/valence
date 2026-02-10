"""Tests for valence-federation CLI."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Import the CLI module
from valence.cli.federation import (
    cmd_discover,
    cmd_list,
    cmd_sync,
    cmd_trust,
    create_parser,
    federation_request,
    get_local_did,
    get_private_key,
    get_public_key_multibase,
    main,
    sign_request,
)
from valence.core.config import clear_config_cache

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_did_document():
    """Return a mock DID document response."""
    return {
        "@context": [
            "https://www.w3.org/ns/did/v1",
            "https://valence.dev/ns/vfp/v1",
        ],
        "id": "did:vkb:web:valence.example.com",
        "verificationMethod": [
            {
                "id": "did:vkb:web:valence.example.com#keys-1",
                "type": "Ed25519VerificationKey2020",
                "controller": "did:vkb:web:valence.example.com",
                "publicKeyMultibase": "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            }
        ],
        "authentication": ["did:vkb:web:valence.example.com#keys-1"],
        "assertionMethod": ["did:vkb:web:valence.example.com#keys-1"],
        "service": [
            {
                "id": "did:vkb:web:valence.example.com#vfp",
                "type": "ValenceFederationProtocol",
                "serviceEndpoint": "https://valence.example.com/federation",
            },
            {
                "id": "did:vkb:web:valence.example.com#mcp",
                "type": "ModelContextProtocol",
                "serviceEndpoint": "https://valence.example.com/mcp",
            },
        ],
        "vfp:capabilities": ["belief_sync", "aggregation_participate"],
        "vfp:profile": {
            "name": "Example Valence Node",
            "domains": ["research", "ai"],
        },
        "vfp:protocolVersion": "1.0",
    }


@pytest.fixture
def mock_node_list():
    """Return a mock node list response."""
    return {
        "success": True,
        "nodes": [
            {
                "node": {
                    "id": str(uuid4()),
                    "did": "did:vkb:web:node1.example.com",
                    "status": "active",
                    "trust_phase": "participant",
                    "name": "Node 1",
                },
                "trust": {
                    "trust": {"overall": 0.75},
                },
            },
            {
                "node": {
                    "id": str(uuid4()),
                    "did": "did:vkb:web:node2.example.com",
                    "status": "discovered",
                    "trust_phase": "observer",
                    "name": "Node 2",
                },
                "trust": {
                    "trust": {"overall": 0.1},
                },
            },
        ],
        "count": 2,
    }


@pytest.fixture
def mock_federation_status():
    """Return mock federation status data."""
    return {
        "nodes_by_status": {"active": 3, "discovered": 2, "unreachable": 1},
        "sync_stats": {
            "total_peers": 3,
            "beliefs_sent": 150,
            "beliefs_received": 200,
            "last_sync": datetime.now(),
        },
        "belief_stats": {
            "local_beliefs": 500,
            "federated_beliefs": 300,
        },
    }


# =============================================================================
# PARSER TESTS
# =============================================================================


class TestParser:
    """Tests for CLI argument parsing."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "valence-federation"

    def test_discover_command(self):
        """Test discover command parsing."""
        parser = create_parser()
        args = parser.parse_args(["discover", "https://example.com"])

        assert args.command == "discover"
        assert args.endpoint == "https://example.com"
        assert args.register is True

    def test_discover_no_register(self):
        """Test discover with --no-register flag."""
        parser = create_parser()
        args = parser.parse_args(["discover", "--no-register", "https://example.com"])

        assert args.command == "discover"
        assert args.register is False

    def test_list_command(self):
        """Test list command parsing."""
        parser = create_parser()
        args = parser.parse_args(["list"])

        assert args.command == "list"
        assert args.status is None
        assert args.limit == 50

    def test_list_with_filters(self):
        """Test list command with filters."""
        parser = create_parser()
        args = parser.parse_args(["list", "--status", "active", "--trust-phase", "participant", "-n", "10"])

        assert args.command == "list"
        assert args.status == "active"
        assert args.trust_phase == "participant"
        assert args.limit == 10

    def test_status_command(self):
        """Test status command parsing."""
        parser = create_parser()
        args = parser.parse_args(["status"])

        assert args.command == "status"

    def test_trust_command(self):
        """Test trust command parsing."""
        parser = create_parser()
        node_id = str(uuid4())
        args = parser.parse_args(["trust", node_id, "elevated", "--reason", "Trusted partner"])

        assert args.command == "trust"
        assert args.node_id == node_id
        assert args.level == "elevated"
        assert args.reason == "Trusted partner"

    def test_trust_with_score(self):
        """Test trust command with manual score."""
        parser = create_parser()
        node_id = str(uuid4())
        args = parser.parse_args(["trust", node_id, "elevated", "--score", "0.9"])

        assert args.command == "trust"
        assert args.score == 0.9

    def test_sync_command_all(self):
        """Test sync command without node_id."""
        parser = create_parser()
        args = parser.parse_args(["sync"])

        assert args.command == "sync"
        assert args.node_id is None

    def test_sync_command_specific_node(self):
        """Test sync command with specific node."""
        parser = create_parser()
        node_id = str(uuid4())
        args = parser.parse_args(["sync", node_id, "--wait"])

        assert args.command == "sync"
        assert args.node_id == node_id
        assert args.wait is True

    def test_json_flag(self):
        """Test --json flag on subcommand."""
        parser = create_parser()
        # Each subcommand has its own --json flag (after the subcommand)
        args = parser.parse_args(["status", "--json"])

        assert args.json is True


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestConfiguration:
    """Tests for configuration functions."""

    def test_get_private_key_from_env(self, monkeypatch):
        """Test getting private key from environment."""
        clear_config_cache()
        test_key = "0" * 64  # 32 bytes in hex
        monkeypatch.setenv("VALENCE_FEDERATION_PRIVATE_KEY", test_key)

        key = get_private_key()
        assert key is not None
        assert len(key) == 32

    def test_get_private_key_missing(self, monkeypatch):
        """Test missing private key."""
        clear_config_cache()
        monkeypatch.delenv("VALENCE_FEDERATION_PRIVATE_KEY", raising=False)

        key = get_private_key()
        assert key is None

    def test_get_public_key_multibase(self, monkeypatch):
        """Test getting public key from environment."""
        clear_config_cache()
        test_key = "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
        monkeypatch.setenv("VALENCE_FEDERATION_PUBLIC_KEY", test_key)

        key = get_public_key_multibase()
        assert key == test_key

    def test_get_local_did(self, monkeypatch):
        """Test getting local DID from environment."""
        clear_config_cache()
        test_did = "did:vkb:web:localhost"
        monkeypatch.setenv("VALENCE_FEDERATION_DID", test_did)

        did = get_local_did()
        assert did == test_did


# =============================================================================
# REQUEST SIGNING TESTS
# =============================================================================


class TestRequestSigning:
    """Tests for VFP request signing."""

    @pytest.mark.skipif(
        not os.environ.get("VALENCE_FEDERATION_PRIVATE_KEY"),
        reason="Requires cryptography library and private key",
    )
    def test_sign_request(self, monkeypatch):
        """Test request signing produces valid headers."""
        # Generate a test key pair
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )

            private_key = Ed25519PrivateKey.generate()
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )

            monkeypatch.setenv("VALENCE_FEDERATION_DID", "did:vkb:web:test")

            headers = sign_request(
                method="POST",
                path="/federation/protocol",
                body=b'{"type": "SYNC_REQUEST"}',
                private_key=private_bytes,
            )

            assert "X-VFP-DID" in headers
            assert "X-VFP-Signature" in headers
            assert "X-VFP-Timestamp" in headers
            assert "X-VFP-Nonce" in headers
            assert headers["X-VFP-DID"] == "did:vkb:web:test"

        except ImportError:
            pytest.skip("cryptography library not installed")


# =============================================================================
# COMMAND TESTS
# =============================================================================


class TestDiscoverCommand:
    """Tests for the discover command."""

    @pytest.mark.asyncio
    async def test_discover_success(self, mock_did_document, capsys):
        """Test successful node discovery."""
        args = argparse.Namespace(
            endpoint="https://valence.example.com",
            register=False,
            json=False,
        )

        with patch("valence.cli.federation.federation_request") as mock_request:
            mock_request.return_value = mock_did_document

            result = await cmd_discover(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Discovered node" in captured.out
            assert "did:vkb:web:valence.example.com" in captured.out

    @pytest.mark.asyncio
    async def test_discover_failure(self, capsys):
        """Test failed node discovery."""
        args = argparse.Namespace(
            endpoint="https://invalid.example.com",
            register=False,
            json=False,
        )

        with patch("valence.cli.federation.federation_request") as mock_request:
            mock_request.return_value = {"error": "Connection refused"}

            result = await cmd_discover(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Discovery failed" in captured.err

    @pytest.mark.asyncio
    async def test_discover_json_output(self, mock_did_document, capsys):
        """Test discover with JSON output."""
        args = argparse.Namespace(
            endpoint="https://valence.example.com",
            register=False,
            json=True,
        )

        with patch("valence.cli.federation.federation_request") as mock_request:
            mock_request.return_value = mock_did_document

            result = await cmd_discover(args)

            assert result == 0
            captured = capsys.readouterr()
            output = json.loads(captured.out)  # Full output is JSON when --json flag
            assert output["id"] == "did:vkb:web:valence.example.com"


class TestListCommand:
    """Tests for the list command."""

    @pytest.mark.asyncio
    async def test_list_success(self, mock_node_list, capsys):
        """Test successful node listing."""
        args = argparse.Namespace(
            status=None,
            trust_phase=None,
            limit=50,
            json=False,
        )

        with patch("oro_federation.tools.federation_node_list") as mock_list:
            mock_list.return_value = mock_node_list

            result = await cmd_list(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Federation Nodes (2)" in captured.out

    @pytest.mark.asyncio
    async def test_list_empty(self, capsys):
        """Test listing with no nodes."""
        args = argparse.Namespace(
            status=None,
            trust_phase=None,
            limit=50,
            json=False,
        )

        with patch("oro_federation.tools.federation_node_list") as mock_list:
            mock_list.return_value = {"success": True, "nodes": [], "count": 0}

            result = await cmd_list(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "No federation nodes found" in captured.out

    @pytest.mark.asyncio
    async def test_list_json_output(self, mock_node_list, capsys):
        """Test list with JSON output."""
        args = argparse.Namespace(
            status=None,
            trust_phase=None,
            limit=50,
            json=True,
        )

        with patch("oro_federation.tools.federation_node_list") as mock_list:
            mock_list.return_value = mock_node_list

            result = await cmd_list(args)

            assert result == 0
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["success"] is True
            assert len(output["nodes"]) == 2


class TestTrustCommand:
    """Tests for the trust command."""

    @pytest.mark.asyncio
    async def test_trust_set_elevated(self, capsys):
        """Test setting elevated trust."""
        node_id = str(uuid4())
        args = argparse.Namespace(
            node_id=node_id,
            level="elevated",
            score=None,
            reason="Trusted partner",
            json=False,
        )

        with (
            patch("oro_federation.tools.federation_trust_get") as mock_get,
            patch("oro_federation.tools.federation_trust_set_preference") as mock_set,
        ):
            mock_get.return_value = {"success": True, "effective_trust": 0.5}
            mock_set.return_value = {"success": True, "effective_trust": 0.75}

            result = await cmd_trust(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Trust preference updated" in captured.out

    @pytest.mark.asyncio
    async def test_trust_invalid_level(self, capsys):
        """Test invalid trust level."""
        args = argparse.Namespace(
            node_id=str(uuid4()),
            level="invalid_level",
            score=None,
            reason=None,
            json=False,
        )

        result = await cmd_trust(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid trust level" in captured.err

    @pytest.mark.asyncio
    async def test_trust_node_not_found(self, capsys):
        """Test trust command with non-existent node."""
        args = argparse.Namespace(
            node_id=str(uuid4()),
            level="elevated",
            score=None,
            reason=None,
            json=False,
        )

        with patch("oro_federation.tools.federation_trust_get") as mock_get:
            mock_get.return_value = {"success": False, "error": "Node not found"}

            result = await cmd_trust(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Node not found" in captured.err


class TestSyncCommand:
    """Tests for the sync command."""

    @pytest.mark.asyncio
    async def test_sync_all_nodes(self, capsys):
        """Test sync with all nodes."""
        args = argparse.Namespace(
            node_id=None,
            wait=False,
            json=False,
        )

        with patch("oro_federation.tools.federation_sync_trigger") as mock_trigger:
            mock_trigger.return_value = {
                "success": True,
                "queued_nodes": 3,
                "beliefs_queued": 10,
            }

            result = await cmd_sync(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Sync triggered" in captured.out

    @pytest.mark.asyncio
    async def test_sync_specific_node(self, capsys):
        """Test sync with specific node."""
        node_id = str(uuid4())
        args = argparse.Namespace(
            node_id=node_id,
            wait=False,
            json=False,
        )

        with patch("oro_federation.tools.federation_sync_trigger") as mock_trigger:
            mock_trigger.return_value = {"success": True}

            result = await cmd_sync(args)

            assert result == 0
            mock_trigger.assert_called_once_with(node_id=node_id)

    @pytest.mark.asyncio
    async def test_sync_failure(self, capsys):
        """Test sync failure."""
        args = argparse.Namespace(
            node_id=None,
            wait=False,
            json=False,
        )

        with patch("oro_federation.tools.federation_sync_trigger") as mock_trigger:
            mock_trigger.return_value = {"success": False, "error": "No active nodes"}

            result = await cmd_sync(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Sync failed" in captured.err


# =============================================================================
# HTTP CLIENT TESTS
# =============================================================================


class TestFederationRequest:
    """Tests for the HTTP client."""

    @pytest.mark.asyncio
    async def test_federation_request_success(self):
        """Test successful HTTP request."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            # Response mock - needs async json() method
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "ok"})

            # Request context manager (async with session.request(...))
            mock_request_ctx = MagicMock()
            mock_request_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_request_ctx.__aexit__ = AsyncMock(return_value=None)

            # Session mock - request() returns context manager directly (not a coroutine)
            mock_session = MagicMock()
            mock_session.request = MagicMock(return_value=mock_request_ctx)

            # Session class context manager (async with ClientSession(...))
            mock_session_ctx = MagicMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session_ctx

            result = await federation_request("GET", "https://example.com/test")

            assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_federation_request_error(self):
        """Test HTTP request error handling."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            # Response mock for error case
            mock_response = MagicMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")

            # Request context manager
            mock_request_ctx = MagicMock()
            mock_request_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_request_ctx.__aexit__ = AsyncMock(return_value=None)

            # Session mock
            mock_session = MagicMock()
            mock_session.request = MagicMock(return_value=mock_request_ctx)

            # Session class context manager
            mock_session_ctx = MagicMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session_ctx

            result = await federation_request("GET", "https://example.com/test")

            assert "error" in result
            assert "HTTP 500" in result["error"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI (requires database)."""

    def test_main_no_args(self, capsys):
        """Test main with no arguments shows help."""
        with patch("sys.argv", ["valence-federation"]):
            result = main()

            assert result == 0
            captured = capsys.readouterr()
            assert "valence-federation" in captured.out

    def test_main_help(self, capsys):
        """Test main with --help."""
        with patch("sys.argv", ["valence-federation", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0
