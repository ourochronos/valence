"""Tests for CLI config loading with precedence: flags > env > file > defaults."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from valence.cli.config import CLIConfig, get_cli_config, reset_cli_config, set_cli_config


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config singleton between tests."""
    reset_cli_config()
    yield
    reset_cli_config()


class TestCLIConfigDefaults:
    def test_defaults(self):
        config = CLIConfig()
        assert config.server_url == "http://127.0.0.1:8420"
        assert config.token == ""
        assert config.output == "text"
        assert config.timeout == 30.0

    def test_load_defaults_when_no_file(self, tmp_path):
        config = CLIConfig.load(config_path=tmp_path / "nonexistent.toml")
        assert config.server_url == "http://127.0.0.1:8420"
        assert config.token == ""
        assert config.output == "text"


class TestCLIConfigFile:
    def test_load_from_toml(self, tmp_path):
        config_file = tmp_path / "cli.toml"
        config_file.write_text('server_url = "http://remote:9000"\ntoken = "vt_file_token"\noutput = "json"\ntimeout = 60.0\n')
        config = CLIConfig.load(config_path=config_file)
        assert config.server_url == "http://remote:9000"
        assert config.token == "vt_file_token"
        assert config.output == "json"
        assert config.timeout == 60.0

    def test_partial_toml(self, tmp_path):
        config_file = tmp_path / "cli.toml"
        config_file.write_text('token = "vt_partial"\n')
        config = CLIConfig.load(config_path=config_file)
        assert config.server_url == "http://127.0.0.1:8420"  # default
        assert config.token == "vt_partial"

    def test_invalid_output_in_toml_ignored(self, tmp_path):
        config_file = tmp_path / "cli.toml"
        config_file.write_text('output = "csv"\n')
        config = CLIConfig.load(config_path=config_file)
        assert config.output == "text"  # default, csv is not valid


class TestCLIConfigEnv:
    def test_env_overrides_file(self, tmp_path):
        config_file = tmp_path / "cli.toml"
        config_file.write_text('server_url = "http://file:1111"\ntoken = "vt_file"\n')

        env = {
            "VALENCE_SERVER_URL": "http://env:2222",
            "VALENCE_TOKEN": "vt_env",
            "VALENCE_OUTPUT": "table",
        }
        with patch.dict(os.environ, env, clear=False):
            config = CLIConfig.load(config_path=config_file)
        assert config.server_url == "http://env:2222"
        assert config.token == "vt_env"
        assert config.output == "table"

    def test_env_invalid_output_ignored(self):
        with patch.dict(os.environ, {"VALENCE_OUTPUT": "xml"}, clear=False):
            config = CLIConfig.load(config_path=Path("/nonexistent"))
        assert config.output == "text"


class TestCLIConfigFlags:
    def test_flags_override_env(self):
        env = {"VALENCE_SERVER_URL": "http://env:2222", "VALENCE_TOKEN": "vt_env"}
        with patch.dict(os.environ, env, clear=False):
            config = CLIConfig.load(
                config_path=Path("/nonexistent"),
                server_url="http://flag:3333",
                token="vt_flag",
                output="json",
                timeout=120.0,
            )
        assert config.server_url == "http://flag:3333"
        assert config.token == "vt_flag"
        assert config.output == "json"
        assert config.timeout == 120.0

    def test_partial_flags_keep_env(self):
        with patch.dict(os.environ, {"VALENCE_TOKEN": "vt_env"}, clear=False):
            config = CLIConfig.load(config_path=Path("/nonexistent"), server_url="http://flag:3333")
        assert config.server_url == "http://flag:3333"
        assert config.token == "vt_env"


class TestCLIConfigSingleton:
    def test_get_set_singleton(self):
        config = CLIConfig(server_url="http://test:9999", token="vt_test")
        set_cli_config(config)
        assert get_cli_config().server_url == "http://test:9999"
        assert get_cli_config().token == "vt_test"

    def test_get_creates_default_if_unset(self):
        config = get_cli_config()
        assert config.server_url == "http://127.0.0.1:8420"
