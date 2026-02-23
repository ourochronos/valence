"""Server-specific test fixtures."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def clean_server_settings():
    """Reset server settings between tests."""
    import valence.server.config as config_module

    config_module._settings = None
    yield
    config_module._settings = None


@pytest.fixture
def clean_token_store():
    """Reset token store between tests."""
    import valence.server.auth as auth_module

    auth_module._token_store = None
    yield
    auth_module._token_store = None


@pytest.fixture
def clean_oauth_stores():
    """Reset OAuth stores between tests."""
    import valence.server.oauth_models as oauth_module

    oauth_module._client_store = None
    oauth_module._code_store = None
    oauth_module._refresh_store = None
    yield
    oauth_module._client_store = None
    oauth_module._code_store = None
    oauth_module._refresh_store = None


@pytest.fixture
def temp_token_file() -> Generator[Path, None, None]:
    """Create a temporary token file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"tokens": []}, f)
        f.flush()
        yield Path(f.name)


@pytest.fixture
def temp_clients_file() -> Generator[Path, None, None]:
    """Create a temporary OAuth clients file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"clients": []}, f)
        f.flush()
        yield Path(f.name)


@pytest.fixture
def server_env(monkeypatch, temp_token_file, temp_clients_file):
    """Set up server environment variables."""
    monkeypatch.setenv("VALENCE_HOST", "127.0.0.1")
    monkeypatch.setenv("VALENCE_PORT", "8420")
    monkeypatch.setenv("VALENCE_TOKEN_FILE", str(temp_token_file))
    monkeypatch.setenv("VALENCE_OAUTH_ENABLED", "true")
    monkeypatch.setenv("VALENCE_OAUTH_CLIENTS_FILE", str(temp_clients_file))
    monkeypatch.setenv("VALENCE_OAUTH_JWT_SECRET", "test-jwt-secret-for-testing-only")
    monkeypatch.setenv("VALENCE_OAUTH_USERNAME", "admin")
    monkeypatch.setenv("VALENCE_OAUTH_PASSWORD", "testpass")
    monkeypatch.setenv("VALENCE_FEDERATION_ENABLED", "false")


@pytest.fixture
def mock_db_for_server():
    """Mock database calls for server testing."""
    mock_cursor = MagicMock()

    def mock_context(*args, **kwargs):
        class CM:
            def __enter__(self):
                return mock_cursor

            def __exit__(self, *args):
                pass

        return CM()

    with patch("valence.core.db.get_cursor", mock_context):
        with patch("valence.server.app.get_settings") as mock_get_settings:
            from valence.server.config import ServerSettings

            # Create mock settings
            settings = MagicMock(spec=ServerSettings)
            settings.host = "127.0.0.1"
            settings.port = 8420
            settings.server_name = "valence"
            settings.server_version = "1.0.0"
            settings.rate_limit_rpm = 60
            settings.base_url = "http://127.0.0.1:8420"
            settings.mcp_resource_url = "http://127.0.0.1:8420/mcp"
            settings.oauth_enabled = False
            settings.allowed_origins = ["*"]
            settings.token_file = Path("/tmp/test-tokens.json")
            settings.federation_enabled = False

            mock_get_settings.return_value = settings
            yield {
                "cursor": mock_cursor,
                "settings": settings,
            }


@pytest.fixture
def test_token() -> str:
    """Return a test token value."""
    return "vt_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


@pytest.fixture
def test_client_with_mocks(
    server_env,
    clean_server_settings,
    clean_token_store,
    clean_oauth_stores,
    mock_db_for_health,
) -> TestClient:
    """Create a test client with mocked dependencies."""
    # Import after patching settings
    from valence.server.app import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_db_for_health():
    """Mock database for health check."""
    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = None

    def mock_context(*args, **kwargs):
        class CM:
            def __enter__(self):
                return mock_cursor

            def __exit__(self, *args):
                pass

        return CM()

    with patch("valence.server.app.get_cursor", mock_context):
        yield mock_cursor


@pytest.fixture
def mock_substrate_tools():
    """Mock substrate tools."""
    with patch("valence.server.app.SUBSTRATE_TOOLS", []) as mock_st:
        with patch("valence.server.app.handle_substrate_tool") as mock_handler:
            mock_handler.return_value = {"success": True, "data": "test"}
            yield {"tools": mock_st, "handler": mock_handler}


@pytest.fixture
def mock_vkb_tools():
    """Mock VKB tools."""
    with patch("valence.server.app.VKB_TOOLS", []) as mock_vt:
        with patch("valence.server.app.handle_vkb_tool") as mock_handler:
            mock_handler.return_value = {"success": True, "data": "test"}
            yield {"tools": mock_vt, "handler": mock_handler}
