"""Tests for config_cmd command module (WU-18)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from valence.cli.commands.config_cmd import (
    cmd_config_inference_cerebras,
    cmd_config_inference_gemini,
    cmd_config_inference_ollama,
    cmd_config_inference_show,
    cmd_config_set,
    cmd_config_show_all,
)
from valence.cli.http_client import ValenceAPIError, ValenceConnectionError


class TestConfigShowAll:
    """Test config show-all command."""

    @patch("valence.core.db.get_cursor")
    def test_show_all_success(self, mock_get_cursor):
        """Display all configuration from database."""
        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {"key": "inference", "value": {"provider": "gemini"}, "updated_at": "2024-01-01"},
            {"key": "retention", "value": '{"days": 30}', "updated_at": "2024-01-02"},
        ]

        args = MagicMock()
        result = cmd_config_show_all(args)

        assert result == 0
        mock_cursor.execute.assert_called_once()

    @patch("valence.core.db.get_cursor")
    def test_show_all_empty(self, mock_get_cursor):
        """Handle empty configuration."""
        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        args = MagicMock()
        result = cmd_config_show_all(args)

        assert result == 0

    @patch("valence.core.db.get_cursor")
    def test_show_all_masks_api_keys(self, mock_get_cursor):
        """Mask sensitive api_key values in display."""
        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {"key": "inference", "value": {"provider": "cerebras", "api_key": "secret123"}, "updated_at": "2024-01-01"},
        ]

        args = MagicMock()
        result = cmd_config_show_all(args)

        assert result == 0

    @patch("valence.core.db.get_cursor")
    def test_show_all_db_error(self, mock_get_cursor):
        """Handle database errors."""
        mock_get_cursor.return_value.__enter__.side_effect = Exception("DB connection failed")

        args = MagicMock()
        result = cmd_config_show_all(args)

        assert result == 1


class TestConfigSet:
    """Test config set command."""

    @patch("valence.core.db.get_cursor")
    def test_set_json_value(self, mock_get_cursor):
        """Set a configuration value with JSON."""
        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        args = MagicMock(key="test_key", value='{"setting": "value"}')
        result = cmd_config_set(args)

        assert result == 0
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        assert "INSERT INTO system_config" in call_args[0]
        assert call_args[1][0] == "test_key"
        parsed = json.loads(call_args[1][1])
        assert parsed == {"setting": "value"}

    @patch("valence.core.db.get_cursor")
    def test_set_string_value(self, mock_get_cursor):
        """Set a plain string value."""
        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        args = MagicMock(key="simple", value="plain_text")
        result = cmd_config_set(args)

        assert result == 0
        call_args = mock_cursor.execute.call_args[0]
        assert call_args[1][1] == '"plain_text"'

    @patch("valence.core.db.get_cursor")
    def test_set_db_error(self, mock_get_cursor):
        """Handle database errors during set."""
        mock_get_cursor.return_value.__enter__.side_effect = Exception("Write failed")

        args = MagicMock(key="test", value="value")
        result = cmd_config_set(args)

        assert result == 1


class TestConfigInferenceShow:
    """Test config inference show command."""

    @patch("valence.cli.commands.config_cmd.get_client")
    def test_show_via_api(self, mock_get_client):
        """Show inference config via REST API."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {"provider": "gemini", "model": "gemini-2.5-flash"}

        args = MagicMock()
        result = cmd_config_inference_show(args)

        assert result == 0
        mock_client.get.assert_called_with("/v2/config/inference")

    @patch("valence.cli.commands.config_cmd.get_client")
    @patch("valence.cli.commands.config_cmd._show_config_direct")
    def test_show_fallback_to_direct(self, mock_direct, mock_get_client):
        """Fallback to direct DB read when server unavailable."""
        mock_get_client.return_value.get.side_effect = ValenceConnectionError("Connection refused")
        mock_direct.return_value = 0

        args = MagicMock()
        result = cmd_config_inference_show(args)

        assert result == 0
        mock_direct.assert_called_once()

    @patch("valence.cli.commands.config_cmd.get_client")
    def test_show_api_error(self, mock_get_client):
        """Handle API errors."""
        mock_get_client.return_value.get.side_effect = ValenceAPIError(500, "error", "Internal error")

        args = MagicMock()
        result = cmd_config_inference_show(args)

        assert result == 1


class TestConfigInferenceGemini:
    """Test config inference gemini command."""

    @patch("valence.cli.commands.config_cmd.get_client")
    def test_gemini_default_model(self, mock_get_client):
        """Configure Gemini with default model."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True}

        args = MagicMock(model="gemini-2.5-flash")
        result = cmd_config_inference_gemini(args)

        assert result == 0
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/v2/config/inference"
        body = call_args[1]["body"]
        assert body["provider"] == "gemini"
        assert body["model"] == "gemini-2.5-flash"

    @patch("valence.cli.commands.config_cmd.get_client")
    def test_gemini_custom_model(self, mock_get_client):
        """Configure Gemini with custom model."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True}

        args = MagicMock(model="gemini-pro")
        result = cmd_config_inference_gemini(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["model"] == "gemini-pro"

    @patch("valence.cli.commands.config_cmd.get_client")
    @patch("valence.cli.commands.config_cmd._write_config_direct")
    def test_gemini_fallback_to_direct(self, mock_direct, mock_get_client):
        """Fallback to direct DB write when server unavailable."""
        mock_get_client.return_value.post.side_effect = ValenceConnectionError("Connection refused")
        mock_direct.return_value = 0

        args = MagicMock(model="gemini-2.5-flash")
        result = cmd_config_inference_gemini(args)

        assert result == 0
        mock_direct.assert_called_once()
        call_args = mock_direct.call_args[0]
        assert call_args[0] == "inference"
        assert call_args[1]["provider"] == "gemini"


class TestConfigInferenceCerebras:
    """Test config inference cerebras command."""

    @patch("valence.cli.commands.config_cmd.get_client")
    def test_cerebras_with_api_key(self, mock_get_client):
        """Configure Cerebras with API key."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True}

        args = MagicMock(api_key="csk_test123", model="llama-4-scout-17b-16e-instruct")
        result = cmd_config_inference_cerebras(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["provider"] == "cerebras"
        assert body["api_key"] == "csk_test123"
        assert body["model"] == "llama-4-scout-17b-16e-instruct"

    @patch("valence.cli.commands.config_cmd.get_client")
    def test_cerebras_custom_model(self, mock_get_client):
        """Configure Cerebras with custom model."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True}

        args = MagicMock(api_key="csk_key", model="custom-model")
        result = cmd_config_inference_cerebras(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["model"] == "custom-model"

    @patch("valence.cli.commands.config_cmd.get_client")
    def test_cerebras_api_error(self, mock_get_client):
        """Handle API errors."""
        mock_get_client.return_value.post.side_effect = ValenceAPIError(400, "invalid_api_key", "Invalid key")

        args = MagicMock(api_key="bad_key", model="llama-4-scout-17b-16e-instruct")
        result = cmd_config_inference_cerebras(args)

        assert result == 1


class TestConfigInferenceOllama:
    """Test config inference ollama command."""

    @patch("valence.cli.commands.config_cmd.get_client")
    def test_ollama_default_host(self, mock_get_client):
        """Configure Ollama with default host."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True}

        args = MagicMock(host="http://localhost:11434", model="qwen3:30b")
        result = cmd_config_inference_ollama(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["provider"] == "ollama"
        assert body["host"] == "http://localhost:11434"
        assert body["model"] == "qwen3:30b"

    @patch("valence.cli.commands.config_cmd.get_client")
    def test_ollama_custom_host(self, mock_get_client):
        """Configure Ollama with custom host."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True}

        args = MagicMock(host="http://derptop:11434", model="llama3")
        result = cmd_config_inference_ollama(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["host"] == "http://derptop:11434"
        assert body["model"] == "llama3"

    @patch("valence.cli.commands.config_cmd.get_client")
    @patch("valence.cli.commands.config_cmd._write_config_direct")
    def test_ollama_connection_error(self, mock_direct, mock_get_client):
        """Fallback to direct write on connection error."""
        mock_get_client.return_value.post.side_effect = ValenceConnectionError("Connection refused")
        mock_direct.return_value = 0

        args = MagicMock(host="http://localhost:11434", model="qwen3:30b")
        result = cmd_config_inference_ollama(args)

        assert result == 0
        mock_direct.assert_called_once()


class TestConfigDirectFallback:
    """Test direct database fallback functions."""

    @patch("valence.core.db.get_cursor")
    def test_show_config_direct_success(self, mock_get_cursor):
        """Read config directly from database."""
        from valence.cli.commands.config_cmd import _show_config_direct

        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {
            "value": {"provider": "gemini", "api_key": "secret"},
            "updated_at": "2024-01-01",
        }

        result = _show_config_direct()

        assert result == 0
        mock_cursor.execute.assert_called_once()

    @patch("valence.core.db.get_cursor")
    def test_show_config_direct_not_found(self, mock_get_cursor):
        """Handle missing config gracefully."""
        from valence.cli.commands.config_cmd import _show_config_direct

        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        result = _show_config_direct()

        assert result == 0

    @patch("valence.core.db.get_cursor")
    def test_write_config_direct_success(self, mock_get_cursor):
        """Write config directly to database."""
        from valence.cli.commands.config_cmd import _write_config_direct

        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        config_value = {"provider": "gemini", "model": "flash"}
        display_value = {"provider": "gemini", "model": "flash"}
        result = _write_config_direct("inference", config_value, display_value)

        assert result == 0
        mock_cursor.execute.assert_called_once()

    @patch("valence.core.db.get_cursor")
    def test_write_config_direct_error(self, mock_get_cursor):
        """Handle database write errors."""
        from valence.cli.commands.config_cmd import _write_config_direct

        mock_get_cursor.return_value.__enter__.side_effect = Exception("DB write failed")

        result = _write_config_direct("inference", {}, {})

        assert result == 1


class TestDisplayValueMasking:
    """Test _display_value helper."""

    def test_masks_api_key(self):
        """Mask api_key in config display."""
        from valence.cli.commands.config_cmd import _display_value

        config = {"provider": "cerebras", "api_key": "secret123", "model": "llama"}
        masked = _display_value(config)

        assert masked["api_key"] == "***"
        assert masked["provider"] == "cerebras"
        assert masked["model"] == "llama"

    def test_no_api_key(self):
        """Pass through config without api_key."""
        from valence.cli.commands.config_cmd import _display_value

        config = {"provider": "gemini", "model": "flash"}
        masked = _display_value(config)

        assert masked == config
