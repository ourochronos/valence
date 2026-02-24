# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""CLI configuration — server URL, auth token, output format.

Loads from ~/.valence/cli.toml with environment variable and flag overrides.
Precedence: CLI flags > env vars > config file > defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path.home() / ".valence" / "cli.toml"
_DEFAULT_SERVER_URL = "http://127.0.0.1:8420"
_DEFAULT_OUTPUT = "text"
_DEFAULT_TIMEOUT = 30.0
_ADMIN_TIMEOUT = 300.0


@dataclass
class CLIConfig:
    """CLI configuration loaded from file, env, and flags."""

    server_url: str = _DEFAULT_SERVER_URL
    token: str = ""
    output: str = _DEFAULT_OUTPUT
    timeout: float = _DEFAULT_TIMEOUT

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        server_url: str | None = None,
        token: str | None = None,
        output: str | None = None,
        timeout: float | None = None,
    ) -> CLIConfig:
        """Load config with precedence: flags > env > file > defaults."""
        config = cls()

        # 1. Load from file
        path = config_path or _DEFAULT_CONFIG_PATH
        if path.exists():
            config._load_from_file(path)

        # 2. Override from env
        if url := os.environ.get("VALENCE_SERVER_URL"):
            config.server_url = url
        if tok := os.environ.get("VALENCE_TOKEN"):
            config.token = tok
        if out := os.environ.get("VALENCE_OUTPUT"):
            if out in ("json", "text", "table"):
                config.output = out

        # 3. Override from flags (highest precedence)
        if server_url is not None:
            config.server_url = server_url
        if token is not None:
            config.token = token
        if output is not None:
            config.output = output
        if timeout is not None:
            config.timeout = timeout

        return config

    def _load_from_file(self, path: Path) -> None:
        """Parse TOML config file."""
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        if "server_url" in data:
            self.server_url = str(data["server_url"])
        if "token" in data:
            self.token = str(data["token"])
        if "output" in data and data["output"] in ("json", "text", "table"):
            self.output = str(data["output"])
        if "timeout" in data:
            self.timeout = float(data["timeout"])


_config: CLIConfig | None = None


def get_cli_config() -> CLIConfig:
    """Get the current CLI config singleton."""
    global _config
    if _config is None:
        _config = CLIConfig.load()
    return _config


def set_cli_config(config: CLIConfig) -> None:
    """Set the CLI config singleton (called from main after parsing args)."""
    global _config
    _config = config


def reset_cli_config() -> None:
    """Reset the config singleton (for testing)."""
    global _config
    _config = None
