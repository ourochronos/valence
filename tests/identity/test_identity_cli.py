"""Tests for the identity CLI commands.

Issue #277: ``valence identity {list,link,revoke}``
"""

from __future__ import annotations

import json
from pathlib import Path

from valence.cli.commands.identity import (
    _load_manager,
    _save_store,
    cmd_identity,
    register_identity_commands,
)
from valence.identity.did_manager import DIDManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs):
    """Build a minimal argparse-like namespace."""
    import argparse

    return argparse.Namespace(**kwargs)


# ---------------------------------------------------------------------------
# Store persistence
# ---------------------------------------------------------------------------


class TestStorePersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        store_path = str(tmp_path / "identity.json")

        # Create and populate
        mgr = DIDManager()
        n1, k1 = mgr.create_node_did(label="laptop")
        n2, k2 = mgr.create_node_did(label="phone")
        mgr.link_dids(n1.did, k1, n2.did, k2)

        # Save
        _save_store(mgr._store, store_path)

        # Load into fresh manager
        mgr2, store2 = _load_manager(store_path)
        nodes = mgr2.list_nodes()
        assert len(nodes) == 2

        dids = {n.did for n in nodes}
        assert n1.did in dids
        assert n2.did in dids

    def test_load_nonexistent_creates_empty(self, tmp_path: Path):
        store_path = str(tmp_path / "does_not_exist.json")
        mgr, store = _load_manager(store_path)
        assert len(mgr.list_nodes()) == 0


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


class TestCmdIdentity:
    def test_dispatch_list(self, capsys):
        args = _make_args(identity_command="list", json=False, store=None)
        ret = cmd_identity(args)
        assert ret == 0
        output = capsys.readouterr().out
        assert "No DIDs registered" in output

    def test_dispatch_unknown_subcommand(self, capsys):
        args = _make_args(identity_command="foobar")
        ret = cmd_identity(args)
        assert ret == 1

    def test_dispatch_no_subcommand(self, capsys):
        args = _make_args(identity_command=None)
        ret = cmd_identity(args)
        assert ret == 1

    def test_dispatch_revoke_not_found(self, capsys, tmp_path: Path):
        store_path = str(tmp_path / "empty.json")
        args = _make_args(
            identity_command="revoke",
            did="did:valence:nonexistent",
            reason="test",
            store=store_path,
        )
        ret = cmd_identity(args)
        assert ret == 1

    def test_dispatch_link_not_found(self, capsys, tmp_path: Path):
        store_path = str(tmp_path / "empty.json")
        args = _make_args(
            identity_command="link",
            did_a="did:valence:a",
            did_b="did:valence:b",
            store=store_path,
        )
        ret = cmd_identity(args)
        assert ret == 1

    def test_list_with_json(self, capsys, tmp_path: Path):
        store_path = str(tmp_path / "store.json")

        # Pre-populate store
        mgr = DIDManager()
        n1, _ = mgr.create_node_did(label="test")
        _save_store(mgr._store, store_path)

        args = _make_args(identity_command="list", json=True, store=store_path)
        ret = cmd_identity(args)
        assert ret == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert len(data["nodes"]) == 1


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegisterIdentityCommands:
    def test_registers_subparser(self):
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        register_identity_commands(subparsers)

        # Should parse without error
        args = parser.parse_args(["identity", "list"])
        assert args.identity_command == "list"

        args = parser.parse_args(["identity", "revoke", "did:valence:abc"])
        assert args.identity_command == "revoke"
        assert args.did == "did:valence:abc"

        args = parser.parse_args(["identity", "link", "did:valence:a", "did:valence:b"])
        assert args.identity_command == "link"
        assert args.did_a == "did:valence:a"
        assert args.did_b == "did:valence:b"
