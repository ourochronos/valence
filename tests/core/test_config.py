"""Tests for valence.core.config - CoreSettings and global config management.

Tests cover:
- Settings loading with defaults
- Environment variable overrides
- Singleton behavior (get_config / clear_config_cache)
- Computed properties (database_url, connection_params, pool_config)
- All setting categories: db, embedding, logging, cache, federation/seed
"""

from __future__ import annotations

from valence.core.config import (
    CoreSettings,
    clear_config_cache,
    get_config,
)

# ============================================================================
# CoreSettings - Default Values
# ============================================================================


class TestCoreSettingsDefaults:
    """Test that CoreSettings loads with correct default values."""

    def test_database_defaults(self, clean_env):
        """Test database settings have correct defaults."""
        clear_config_cache()
        settings = CoreSettings()

        assert settings.db_host == "localhost"
        assert settings.db_port == 5432
        assert settings.db_name == "valence"
        assert settings.db_user == "valence"
        assert settings.db_password == ""
        assert settings.db_pool_min == 5
        assert settings.db_pool_max == 20

    def test_embedding_defaults(self, clean_env):
        """Test embedding settings have correct defaults."""
        settings = CoreSettings()

        assert settings.embedding_provider == "local"
        assert settings.embedding_model_path == "BAAI/bge-small-en-v1.5"
        assert settings.embedding_device == "cpu"
        assert settings.openai_api_key == ""

    def test_logging_defaults(self, clean_env):
        """Test logging settings have correct defaults."""
        settings = CoreSettings()

        assert settings.log_level == "INFO"
        assert settings.log_format == ""
        assert settings.log_file is None

    def test_cache_defaults(self, clean_env):
        """Test cache settings have correct defaults."""
        settings = CoreSettings()

        assert settings.cache_max_size == 1000

    def test_seed_node_defaults(self, clean_env):
        """Test seed node settings have correct defaults."""
        settings = CoreSettings()

        assert settings.seed_host is None
        assert settings.seed_port == 8470
        assert settings.seed_id is None
        assert settings.seed_peers is None

    def test_federation_identity_defaults(self, clean_env):
        """Test federation identity settings have correct defaults."""
        settings = CoreSettings()

        assert settings.federation_private_key is None
        assert settings.federation_public_key is None
        assert settings.federation_did is None
        assert settings.trust_registry_path is None


# ============================================================================
# CoreSettings - Environment Variable Overrides
# ============================================================================


class TestCoreSettingsEnvOverrides:
    """Test that environment variables properly override settings."""

    def test_database_env_overrides(self, monkeypatch, clean_env):
        """Test database settings from VKB_ prefixed env vars."""
        monkeypatch.setenv("VKB_DB_HOST", "db.example.com")
        monkeypatch.setenv("VKB_DB_PORT", "5433")
        monkeypatch.setenv("VKB_DB_NAME", "test_db")
        monkeypatch.setenv("VKB_DB_USER", "test_user")
        monkeypatch.setenv("VKB_DB_PASSWORD", "secret123")

        clear_config_cache()
        settings = CoreSettings()

        assert settings.db_host == "db.example.com"
        assert settings.db_port == 5433
        assert settings.db_name == "test_db"
        assert settings.db_user == "test_user"
        assert settings.db_password == "secret123"

    def test_pool_env_overrides(self, monkeypatch, clean_env):
        """Test connection pool settings from VALENCE_ prefixed env vars."""
        monkeypatch.setenv("VALENCE_DB_POOL_MIN", "10")
        monkeypatch.setenv("VALENCE_DB_POOL_MAX", "50")

        settings = CoreSettings()

        assert settings.db_pool_min == 10
        assert settings.db_pool_max == 50

    def test_embedding_env_overrides(self, monkeypatch, clean_env):
        """Test embedding settings from env vars."""
        monkeypatch.setenv("VALENCE_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("VALENCE_EMBEDDING_MODEL_PATH", "custom/model")
        monkeypatch.setenv("VALENCE_EMBEDDING_DEVICE", "cuda")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        settings = CoreSettings()

        assert settings.embedding_provider == "openai"
        assert settings.embedding_model_path == "custom/model"
        assert settings.embedding_device == "cuda"
        assert settings.openai_api_key == "sk-test-key"

    def test_logging_env_overrides(self, monkeypatch, clean_env):
        """Test logging settings from env vars."""
        monkeypatch.setenv("VALENCE_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("VALENCE_LOG_FORMAT", "json")
        monkeypatch.setenv("VALENCE_LOG_FILE", "/var/log/valence.log")

        settings = CoreSettings()

        assert settings.log_level == "DEBUG"
        assert settings.log_format == "json"
        assert settings.log_file == "/var/log/valence.log"

    def test_cache_env_overrides(self, monkeypatch, clean_env):
        """Test cache settings from env vars."""
        monkeypatch.setenv("VALENCE_CACHE_MAX_SIZE", "5000")

        settings = CoreSettings()

        assert settings.cache_max_size == 5000

    def test_seed_node_env_overrides(self, monkeypatch, clean_env):
        """Test seed node settings from env vars."""
        monkeypatch.setenv("VALENCE_SEED_HOST", "seed.example.com")
        monkeypatch.setenv("VALENCE_SEED_PORT", "9000")
        monkeypatch.setenv("VALENCE_SEED_ID", "seed-node-1")
        monkeypatch.setenv("VALENCE_SEED_PEERS", "peer1:8470,peer2:8470")

        settings = CoreSettings()

        assert settings.seed_host == "seed.example.com"
        assert settings.seed_port == 9000
        assert settings.seed_id == "seed-node-1"
        assert settings.seed_peers == "peer1:8470,peer2:8470"

    def test_federation_identity_env_overrides(self, monkeypatch, clean_env):
        """Test federation identity settings from env vars."""
        monkeypatch.setenv("VALENCE_FEDERATION_PRIVATE_KEY", "abcd1234")
        monkeypatch.setenv("VALENCE_FEDERATION_PUBLIC_KEY", "z6Mk...")
        monkeypatch.setenv("VALENCE_FEDERATION_DID", "did:key:z6Mk...")
        monkeypatch.setenv("VALENCE_TRUST_REGISTRY", "/etc/valence/trust.json")

        settings = CoreSettings()

        assert settings.federation_private_key == "abcd1234"
        assert settings.federation_public_key == "z6Mk..."
        assert settings.federation_did == "did:key:z6Mk..."
        assert settings.trust_registry_path == "/etc/valence/trust.json"


# ============================================================================
# CoreSettings - Computed Properties
# ============================================================================


class TestCoreSettingsComputedProperties:
    """Test computed properties on CoreSettings."""

    def test_database_url(self, clean_env):
        """Test database_url property constructs correct URL."""
        settings = CoreSettings()
        expected = "postgresql://valence:@localhost:5432/valence"
        assert settings.database_url == expected

    def test_database_url_with_password(self, monkeypatch, clean_env):
        """Test database_url includes password when set."""
        monkeypatch.setenv("VKB_DB_PASSWORD", "secret")
        settings = CoreSettings()
        expected = "postgresql://valence:secret@localhost:5432/valence"
        assert settings.database_url == expected

    def test_database_url_custom_values(self, monkeypatch, clean_env):
        """Test database_url with fully custom settings."""
        monkeypatch.setenv("VKB_DB_HOST", "db.prod.example.com")
        monkeypatch.setenv("VKB_DB_PORT", "5433")
        monkeypatch.setenv("VKB_DB_NAME", "prod_db")
        monkeypatch.setenv("VKB_DB_USER", "prod_user")
        monkeypatch.setenv("VKB_DB_PASSWORD", "prod_pass")

        settings = CoreSettings()
        expected = "postgresql://prod_user:prod_pass@db.prod.example.com:5433/prod_db"
        assert settings.database_url == expected

    def test_connection_params(self, clean_env):
        """Test connection_params returns correct dict."""
        settings = CoreSettings()
        params = settings.connection_params

        assert params == {
            "host": "localhost",
            "port": 5432,
            "dbname": "valence",
            "user": "valence",
            "password": "",
        }

    def test_connection_params_custom(self, monkeypatch, clean_env):
        """Test connection_params with custom values."""
        monkeypatch.setenv("VKB_DB_HOST", "custom.host")
        monkeypatch.setenv("VKB_DB_PORT", "6543")
        monkeypatch.setenv("VKB_DB_NAME", "custom_db")
        monkeypatch.setenv("VKB_DB_USER", "custom_user")
        monkeypatch.setenv("VKB_DB_PASSWORD", "custom_pass")

        settings = CoreSettings()
        params = settings.connection_params

        assert params == {
            "host": "custom.host",
            "port": 6543,
            "dbname": "custom_db",
            "user": "custom_user",
            "password": "custom_pass",
        }

    def test_pool_config(self, clean_env):
        """Test pool_config returns correct dict."""
        settings = CoreSettings()
        config = settings.pool_config

        assert config == {
            "minconn": 5,
            "maxconn": 20,
        }

    def test_pool_config_custom(self, monkeypatch, clean_env):
        """Test pool_config with custom values."""
        monkeypatch.setenv("VALENCE_DB_POOL_MIN", "2")
        monkeypatch.setenv("VALENCE_DB_POOL_MAX", "100")

        settings = CoreSettings()
        config = settings.pool_config

        assert config == {
            "minconn": 2,
            "maxconn": 100,
        }


# ============================================================================
# Singleton Behavior - get_config / clear_config_cache
# ============================================================================


class TestGetConfigSingleton:
    """Test singleton behavior of get_config."""

    def test_get_config_returns_settings(self, clean_env):
        """Test get_config returns a CoreSettings instance."""
        clear_config_cache()
        config = get_config()
        assert isinstance(config, CoreSettings)

    def test_get_config_returns_same_instance(self, clean_env):
        """Test get_config returns the same instance on repeated calls."""
        clear_config_cache()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_clear_config_cache_resets_singleton(self, clean_env):
        """Test clear_config_cache allows new instance creation."""
        clear_config_cache()
        config1 = get_config()
        clear_config_cache()
        config2 = get_config()
        assert config1 is not config2

    def test_get_config_picks_up_env_changes_after_clear(self, monkeypatch, clean_env):
        """Test that env changes are picked up after clearing cache."""
        clear_config_cache()
        config1 = get_config()
        assert config1.log_level == "INFO"

        monkeypatch.setenv("VALENCE_LOG_LEVEL", "DEBUG")
        clear_config_cache()
        config2 = get_config()
        assert config2.log_level == "DEBUG"

    def test_get_config_ignores_env_changes_without_clear(self, monkeypatch, clean_env):
        """Test that env changes are NOT picked up without clearing cache."""
        clear_config_cache()
        config1 = get_config()
        assert config1.log_level == "INFO"

        monkeypatch.setenv("VALENCE_LOG_LEVEL", "DEBUG")
        config2 = get_config()
        # Same instance, no reload
        assert config2.log_level == "INFO"
        assert config1 is config2


# ============================================================================
# Extra Configuration
# ============================================================================


class TestCoreSettingsExtraIgnored:
    """Test that extra/unknown settings are ignored."""

    def test_unknown_env_vars_ignored(self, monkeypatch, clean_env):
        """Test that unknown VALENCE_ env vars don't cause errors."""
        monkeypatch.setenv("VALENCE_UNKNOWN_SETTING", "some_value")
        monkeypatch.setenv("VKB_RANDOM_THING", "another_value")

        # Should not raise
        settings = CoreSettings()
        assert settings.db_host == "localhost"  # defaults still work

    def test_model_config_extra_ignore(self, clean_env):
        """Test that model_config has extra='ignore' set."""
        settings = CoreSettings()
        model_config = settings.model_config

        assert model_config.get("extra") == "ignore"


# ============================================================================
# Type Coercion
# ============================================================================


class TestCoreSettingsTypeCoercion:
    """Test that environment variables are coerced to correct types."""

    def test_int_coercion_db_port(self, monkeypatch, clean_env):
        """Test that db_port string is coerced to int."""
        monkeypatch.setenv("VKB_DB_PORT", "5433")
        settings = CoreSettings()
        assert settings.db_port == 5433
        assert isinstance(settings.db_port, int)

    def test_int_coercion_pool_settings(self, monkeypatch, clean_env):
        """Test that pool settings are coerced to int."""
        monkeypatch.setenv("VALENCE_DB_POOL_MIN", "3")
        monkeypatch.setenv("VALENCE_DB_POOL_MAX", "25")

        settings = CoreSettings()
        assert settings.db_pool_min == 3
        assert settings.db_pool_max == 25
        assert isinstance(settings.db_pool_min, int)
        assert isinstance(settings.db_pool_max, int)

    def test_int_coercion_cache_size(self, monkeypatch, clean_env):
        """Test that cache_max_size is coerced to int."""
        monkeypatch.setenv("VALENCE_CACHE_MAX_SIZE", "2000")
        settings = CoreSettings()
        assert settings.cache_max_size == 2000
        assert isinstance(settings.cache_max_size, int)

    def test_int_coercion_seed_port(self, monkeypatch, clean_env):
        """Test that seed_port is coerced to int."""
        monkeypatch.setenv("VALENCE_SEED_PORT", "9000")
        settings = CoreSettings()
        assert settings.seed_port == 9000
        assert isinstance(settings.seed_port, int)
