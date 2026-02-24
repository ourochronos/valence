"""Tests for server configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from valence.server.config import get_package_version


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings between tests."""
    import valence.server.config as config_module

    config_module._settings = None
    yield
    config_module._settings = None


class TestServerSettings:
    """Tests for ServerSettings class."""

    def test_default_values(self, clean_env):
        """Test default configuration values."""
        from valence.server.config import ServerSettings

        settings = ServerSettings()

        assert settings.host == "127.0.0.1"
        assert settings.port == 8420
        assert settings.server_name == "valence"
        # server_version now dynamically reads from package metadata
        assert isinstance(settings.server_version, str)
        assert len(settings.server_version) > 0
        assert settings.rate_limit_rpm == 60
        assert settings.oauth_enabled is True
        assert settings.oauth_username == "admin"
        assert settings.oauth_password == ""  # Not set by default
        assert settings.allowed_origins == []

    def test_env_override(self, monkeypatch, clean_env):
        """Test environment variable override."""
        from valence.server.config import ServerSettings

        monkeypatch.setenv("VALENCE_HOST", "0.0.0.0")
        monkeypatch.setenv("VALENCE_PORT", "9000")
        monkeypatch.setenv("VALENCE_SERVER_NAME", "test-server")
        monkeypatch.setenv("VALENCE_RATE_LIMIT_RPM", "120")

        settings = ServerSettings()

        assert settings.host == "0.0.0.0"
        assert settings.port == 9000
        assert settings.server_name == "test-server"
        assert settings.rate_limit_rpm == 120

    def test_base_url_without_external(self, clean_env):
        """Test base_url property without external URL."""
        from valence.server.config import ServerSettings

        settings = ServerSettings()
        settings.external_url = None

        assert settings.base_url == "http://127.0.0.1:8420"

    def test_base_url_with_external(self, monkeypatch, clean_env):
        """Test base_url property with external URL."""
        # Reset settings to pick up new env
        import valence.server.config as config_module

        config_module._settings = None

        monkeypatch.setenv("VALENCE_EXTERNAL_URL", "https://pod.example.com/")
        # JWT secret required in production (external URL)
        monkeypatch.setenv(
            "VALENCE_OAUTH_JWT_SECRET",
            "test-secret-for-jwt-testing-must-be-at-least-32-chars",
        )

        from valence.server.config import ServerSettings

        settings = ServerSettings()

        assert settings.base_url == "https://pod.example.com"  # Trailing slash stripped

    def test_issuer_url(self, clean_env):
        """Test issuer_url property."""
        from valence.server.config import ServerSettings

        settings = ServerSettings()

        assert settings.issuer_url == settings.base_url

    def test_mcp_resource_url(self, monkeypatch, clean_env):
        """Test mcp_resource_url property."""
        # Ensure no external URL so we get predictable localhost URL
        monkeypatch.delenv("VALENCE_EXTERNAL_URL", raising=False)

        from valence.server.config import ServerSettings

        settings = ServerSettings()

        # Should be base_url + /api/v1/mcp
        assert settings.mcp_resource_url.endswith("/mcp")
        assert settings.mcp_resource_url == f"{settings.base_url}/api/v1/mcp"

    def test_database_url(self, monkeypatch, clean_env):
        """Test database_url construction."""
        from valence.server.config import ServerSettings

        monkeypatch.setenv("VALENCE_DB_HOST", "db.example.com")
        monkeypatch.setenv("VALENCE_DB_NAME", "mydb")
        monkeypatch.setenv("VALENCE_DB_USER", "myuser")
        monkeypatch.setenv("VALENCE_DB_PASSWORD", "secret")

        settings = ServerSettings()

        assert settings.database_url == "postgresql://myuser:secret@db.example.com:5433/mydb"

    def test_oauth_jwt_secret_auto_generated(self, clean_env):
        """Test JWT secret auto-generation."""
        from valence.server.config import ServerSettings

        settings1 = ServerSettings()
        settings2 = ServerSettings()

        # Each should generate a unique secret
        assert len(settings1.oauth_jwt_secret) == 64  # 32 bytes hex encoded
        assert len(settings2.oauth_jwt_secret) == 64
        # They should be different (random)
        assert settings1.oauth_jwt_secret != settings2.oauth_jwt_secret

    def test_oauth_jwt_secret_from_env(self, monkeypatch, clean_env):
        """Test JWT secret from environment."""
        from valence.server.config import ServerSettings

        monkeypatch.setenv("VALENCE_OAUTH_JWT_SECRET", "my-custom-secret")

        settings = ServerSettings()

        assert settings.oauth_jwt_secret == "my-custom-secret"

    def test_federation_settings(self, monkeypatch, clean_env):
        """Test federation configuration."""
        from valence.server.config import ServerSettings

        monkeypatch.setenv("VALENCE_FEDERATION_ENABLED", "true")
        monkeypatch.setenv("VALENCE_FEDERATION_NODE_NAME", "MyNode")
        monkeypatch.setenv("VALENCE_FEDERATION_DID", "did:vkb:web:example.com")

        settings = ServerSettings()

        assert settings.federation_enabled is True
        assert settings.federation_node_name == "MyNode"
        assert settings.federation_did == "did:vkb:web:example.com"
        assert settings.federation_capabilities == ["belief_sync"]

    def test_federation_defaults(self, clean_env):
        """Test federation default values."""
        from valence.server.config import ServerSettings

        settings = ServerSettings()

        assert settings.federation_enabled is False
        assert settings.federation_publish_trust_anchors is False
        assert settings.federation_sync_interval_seconds == 300
        assert settings.federation_max_hop_count == 3
        assert settings.federation_default_visibility == "private"


class TestGetPackageVersion:
    """Tests for get_package_version function."""

    def test_returns_string(self):
        """Test that get_package_version returns a string."""
        version = get_package_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_returns_dev_version_when_not_installed(self):
        """Test fallback when package is not installed."""
        from importlib.metadata import PackageNotFoundError

        with patch("valence.server.config.version") as mock_version:
            mock_version.side_effect = PackageNotFoundError("valence")
            # Need to re-import to pick up the mock
            from valence.server import config

            result = config.get_package_version()
            assert result == "0.0.0-dev"

    def test_returns_installed_version(self):
        """Test returns actual version when package is installed."""
        with patch("valence.server.config.version") as mock_version:
            mock_version.return_value = "0.2.1"
            from valence.server import config

            result = config.get_package_version()
            assert result == "0.2.1"


class TestGetSettings:
    """Tests for get_settings function."""

    def test_singleton_pattern(self, clean_env):
        """Test that get_settings returns singleton."""
        from valence.server.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_returns_server_settings(self, clean_env):
        """Test that get_settings returns ServerSettings instance."""
        from valence.server.config import ServerSettings, get_settings

        settings = get_settings()

        assert isinstance(settings, ServerSettings)
