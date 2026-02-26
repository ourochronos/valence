"""Tests for server config security - Issue #27 JWT secret requirement."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from valence.server.config import ServerSettings


class TestJWTSecretRequirement:
    """Test that JWT secret is required in production environments."""

    def test_development_mode_generates_secret(self):
        """In development mode (localhost), a random secret is generated if not provided."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env vars
            for key in list(os.environ.keys()):
                if key.startswith("VALENCE_"):
                    del os.environ[key]

            settings = ServerSettings(
                host="127.0.0.1",
                oauth_enabled=True,
            )

            # Should have generated a secret
            assert settings.oauth_jwt_secret is not None
            assert len(settings.oauth_jwt_secret) >= 64  # 32 bytes = 64 hex chars

    def test_production_mode_disables_oauth_without_secret(self):
        """In production mode without JWT secret, ValueError is raised."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                ServerSettings(
                    external_url="https://valence.example.com",
                    oauth_enabled=True,
                    oauth_jwt_secret=None,
                )

            assert "VALENCE_OAUTH_JWT_SECRET is required" in str(exc_info.value)

    def test_production_mode_via_non_localhost_host(self):
        """Non-localhost host triggers production mode, raises ValueError without secret."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                ServerSettings(
                    host="192.168.1.100",  # External IP = production
                    oauth_enabled=True,
                    oauth_jwt_secret=None,
                )
            assert "VALENCE_OAUTH_JWT_SECRET is required" in str(exc_info.value)

        # Also test with a public IP
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                ServerSettings(
                    host="203.0.113.50",  # Public IP = definitely production
                    oauth_enabled=True,
                    oauth_jwt_secret=None,
                )
            assert "VALENCE_OAUTH_JWT_SECRET is required" in str(exc_info.value)

    def test_production_mode_via_env_var(self):
        """VALENCE_PRODUCTION=true triggers production mode, raises ValueError without secret."""
        with patch.dict(os.environ, {"VALENCE_PRODUCTION": "true"}, clear=False):
            with pytest.raises(ValueError) as exc_info:
                ServerSettings(
                    host="127.0.0.1",  # Even localhost
                    oauth_enabled=True,
                    oauth_jwt_secret=None,
                )
            assert "VALENCE_OAUTH_JWT_SECRET is required" in str(exc_info.value)

    def test_weak_secret_rejected(self):
        """Weak (short) secrets should be rejected in production."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                ServerSettings(
                    external_url="https://valence.example.com",
                    oauth_enabled=True,
                    oauth_jwt_secret="tooshort",  # Less than 32 chars
                )

            assert "at least 32 characters" in str(exc_info.value)

    def test_valid_production_config(self):
        """Valid production config with proper secret should work."""
        import secrets

        strong_secret = secrets.token_hex(32)  # 64 chars

        with patch.dict(os.environ, {}, clear=True):
            settings = ServerSettings(
                external_url="https://valence.example.com",
                oauth_enabled=True,
                oauth_jwt_secret=strong_secret,
            )

            assert settings.oauth_jwt_secret == strong_secret

    def test_oauth_disabled_skips_validation(self):
        """When OAuth is disabled, JWT secret is not required."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ServerSettings(
                external_url="https://valence.example.com",
                oauth_enabled=False,  # OAuth disabled
                oauth_jwt_secret=None,
            )

            # Should still have a generated secret for consistency
            assert settings.oauth_jwt_secret is not None

    def test_secret_from_environment(self):
        """JWT secret can be set via environment variable."""
        import secrets

        strong_secret = secrets.token_hex(32)

        with patch.dict(
            os.environ,
            {"VALENCE_OAUTH_JWT_SECRET": strong_secret},
            clear=False,
        ):
            settings = ServerSettings(
                external_url="https://valence.example.com",
                oauth_enabled=True,
            )

            assert settings.oauth_jwt_secret == strong_secret

    def test_localhost_variations_are_development(self):
        """Various localhost representations should be treated as development."""
        development_hosts = ["localhost", "127.0.0.1"]

        for host in development_hosts:
            with patch.dict(os.environ, {}, clear=True):
                # Should not raise - development mode allows missing secret
                settings = ServerSettings(
                    host=host,
                    oauth_enabled=True,
                    oauth_jwt_secret=None,
                )
                # Secret should be generated
                assert settings.oauth_jwt_secret is not None

    def test_auto_generate_jwt_secret_logs_warning(self, caplog):
        """Auto-generating JWT secret should persist to disk and log info."""
        import logging

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.INFO, logger="valence.server.config"):
                settings = ServerSettings(
                    host="127.0.0.1",
                    oauth_enabled=True,
                    oauth_jwt_secret=None,
                )

                # Secret should be generated
                assert settings.oauth_jwt_secret is not None
                # Info about loading or saving should be logged
                assert any("JWT secret" in record.message for record in caplog.records)
