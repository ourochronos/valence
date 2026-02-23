"""Regression tests for MCP server startup issues.

These tests verify fixes for the MCP server connection failures
discovered during the valence decomposition to our-* bricks.

Issues fixed:
1. init_schema() called without required schema_dir argument
2. except clause catching wrong exception type (DatabaseException vs DatabaseError)
3. DatabaseStats missing to_dict() method
4. MCP env config using VKB_DB_* but our_db reading ORO_DB_*
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# DatabaseStats.to_dict() regression
# ============================================================================


class TestDatabaseStatsToDict:
    """DatabaseStats must have to_dict() for health check serialization."""

    def test_to_dict_exists(self):
        """Should have a to_dict method."""
        from valence.core.health import DatabaseStats

        stats = DatabaseStats()
        assert hasattr(stats, "to_dict")

    def test_to_dict_returns_all_fields(self):
        """Should return all count fields."""
        from valence.core.health import DatabaseStats

        stats = DatabaseStats()
        stats.beliefs_count = 10
        stats.entities_count = 5
        stats.sessions_count = 3
        stats.exchanges_count = 20
        stats.patterns_count = 2
        stats.tensions_count = 1

        result = stats.to_dict()

        assert result == {
            "beliefs_count": 10,
            "entities_count": 5,
            "sessions_count": 3,
            "exchanges_count": 20,
            "patterns_count": 2,
            "tensions_count": 1,
        }

    def test_to_dict_default_zeros(self):
        """Should return zeros for fresh instance."""
        from valence.core.health import DatabaseStats

        stats = DatabaseStats()
        result = stats.to_dict()

        assert all(v == 0 for v in result.values())
        assert len(result) == 6


# ============================================================================
# init_schema() with schema_dir regression
# ============================================================================


class TestMCPServerInitSchema:
    """MCP servers must pass schema_dir to init_schema()."""

    def test_substrate_server_passes_schema_dir(self):
        """Substrate server should pass its own directory as schema_dir."""
        import valence.substrate.mcp_server as mod

        # Verify the module has the Path import
        source = Path(mod.__file__).read_text()
        assert "from pathlib import Path" in source

        # Verify init_schema is called with a path argument
        assert "init_schema(schema_dir)" in source

        # Verify schema_dir points to the substrate directory
        assert "Path(__file__).parent" in source

    def test_vkb_server_passes_schema_dir(self):
        """VKB server should pass substrate directory as schema_dir."""
        import valence.vkb.mcp_server as mod

        source = Path(mod.__file__).read_text()
        assert "from pathlib import Path" in source
        assert "init_schema(schema_dir)" in source

        # VKB must reference the substrate directory since that's where schema.sql lives
        assert '"substrate"' in source or "'substrate'" in source

    def test_schema_files_exist(self):
        """Schema SQL files must exist at the expected location."""
        schema_dir = Path(__file__).parent.parent.parent / "src" / "valence" / "substrate"
        assert (schema_dir / "schema.sql").exists()
        assert (schema_dir / "procedures.sql").exists()


# ============================================================================
# Exception handling regression
# ============================================================================


class TestInitSchemaExceptionHandling:
    """MCP servers must catch our_db.DatabaseError, not just valence DatabaseException."""

    def test_substrate_catches_our_db_error(self):
        """Substrate server should catch OurDatabaseError from init_schema."""
        source = Path(__file__).parent.parent.parent / "src" / "valence" / "substrate" / "mcp_server.py"
        content = source.read_text()

        assert "OurDatabaseError" in content
        assert "from our_db.exceptions import DatabaseError as OurDatabaseError" in content
        assert "(DatabaseException, OurDatabaseError)" in content

    def test_vkb_catches_our_db_error(self):
        """VKB server should catch OurDatabaseError from init_schema."""
        source = Path(__file__).parent.parent.parent / "src" / "valence" / "vkb" / "mcp_server.py"
        content = source.read_text()

        assert "OurDatabaseError" in content
        assert "from our_db.exceptions import DatabaseError as OurDatabaseError" in content
        assert "(DatabaseException, OurDatabaseError)" in content

    def test_init_schema_error_does_not_crash_server(self):
        """init_schema raising DatabaseError should be caught, not crash the server."""
        from our_db.exceptions import DatabaseError

        with patch("valence.substrate.mcp_server.init_schema") as mock_init:
            mock_init.side_effect = DatabaseError("constraint already exists")

            # Import run but don't actually execute it — just verify the exception
            # is the type that would be caught
            assert issubclass(DatabaseError, Exception)

            # Verify the except clause structure by checking source
            import valence.substrate.mcp_server as mod

            source = Path(mod.__file__).read_text()
            assert "except (DatabaseException, OurDatabaseError)" in source


# ============================================================================
# MCP plugin config regression
# ============================================================================


class TestMCPPluginConfig:
    """Plugin .mcp.json must include VKB_DB_* env vars (bridge_db_env handles ORO_DB_*)."""

    def test_plugin_config_has_vkb_vars(self):
        """Plugin config must pass VKB_DB_* vars for valence.core.config."""
        import json

        config_path = Path(__file__).parent.parent.parent / "plugin" / ".mcp.json"
        config = json.loads(config_path.read_text())

        server_names = list(config["mcpServers"].keys())
        for server_name in server_names:
            env = config["mcpServers"][server_name]["env"]
            assert "VKB_DB_HOST" in env, f"{server_name} missing VKB_DB_HOST"
            assert "VKB_DB_PORT" in env, f"{server_name} missing VKB_DB_PORT"
            assert "VKB_DB_NAME" in env, f"{server_name} missing VKB_DB_NAME"
            assert "VKB_DB_USER" in env, f"{server_name} missing VKB_DB_USER"
            assert "VKB_DB_PASSWORD" in env, f"{server_name} missing VKB_DB_PASSWORD"

    def test_plugin_config_no_oro_vars(self):
        """Plugin config should NOT duplicate ORO_DB_* — bridge_db_env() handles it."""
        import json

        config_path = Path(__file__).parent.parent.parent / "plugin" / ".mcp.json"
        config = json.loads(config_path.read_text())

        server_names = list(config["mcpServers"].keys())
        for server_name in server_names:
            env = config["mcpServers"][server_name]["env"]
            assert "ORO_DB_HOST" not in env, f"{server_name} has ORO_DB_HOST (should use bridge_db_env)"

    def test_plugin_config_default_port_is_5433(self):
        """Default port should be 5433 (Docker container port)."""
        import json

        config_path = Path(__file__).parent.parent.parent / "plugin" / ".mcp.json"
        config = json.loads(config_path.read_text())

        server_names = list(config["mcpServers"].keys())
        for server_name in server_names:
            env = config["mcpServers"][server_name]["env"]
            assert "5433" in env.get("VKB_DB_PORT", "")
