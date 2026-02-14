"""Tests for federation discovery endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset all stores between tests."""
    import valence.server.auth as auth_module
    import valence.server.config as config_module
    import valence.server.oauth_models as oauth_module

    config_module._settings = None
    oauth_module._client_store = None
    oauth_module._code_store = None
    oauth_module._refresh_store = None
    auth_module._token_store = None

    yield

    config_module._settings = None
    oauth_module._client_store = None
    oauth_module._code_store = None
    oauth_module._refresh_store = None
    auth_module._token_store = None


@pytest.fixture
def federation_env(monkeypatch, tmp_path):
    """Set up federation-enabled environment."""
    token_file = tmp_path / "tokens.json"
    clients_file = tmp_path / "clients.json"

    token_file.write_text('{"tokens": []}')
    clients_file.write_text('{"clients": []}')

    monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("VALENCE_OAUTH_ENABLED", "true")
    monkeypatch.setenv("VALENCE_OAUTH_CLIENTS_FILE", str(clients_file))
    # JWT secret must be at least 32 characters
    monkeypatch.setenv(
        "VALENCE_OAUTH_JWT_SECRET",
        "test-jwt-secret-for-testing-must-be-at-least-32-chars",
    )
    monkeypatch.setenv("VALENCE_OAUTH_USERNAME", "admin")
    monkeypatch.setenv("VALENCE_OAUTH_PASSWORD", "testpass")
    monkeypatch.setenv("VALENCE_EXTERNAL_URL", "https://valence.example.com")
    monkeypatch.setenv("VALENCE_FEDERATION_ENABLED", "true")
    monkeypatch.setenv("VALENCE_FEDERATION_NODE_DID", "did:vkb:web:valence.example.com")
    monkeypatch.setenv("VALENCE_FEDERATION_NODE_NAME", "TestNode")
    monkeypatch.setenv(
        "VALENCE_FEDERATION_PUBLIC_KEY",
        "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
    )

    return {
        "token_file": token_file,
        "clients_file": clients_file,
    }


@pytest.fixture
def non_federation_env(monkeypatch, tmp_path):
    """Set up non-federation environment."""
    token_file = tmp_path / "tokens.json"
    clients_file = tmp_path / "clients.json"

    token_file.write_text('{"tokens": []}')
    clients_file.write_text('{"clients": []}')

    monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("VALENCE_OAUTH_ENABLED", "true")
    monkeypatch.setenv("VALENCE_OAUTH_CLIENTS_FILE", str(clients_file))
    # JWT secret must be at least 32 characters
    monkeypatch.setenv(
        "VALENCE_OAUTH_JWT_SECRET",
        "test-jwt-secret-for-testing-must-be-at-least-32-chars",
    )
    monkeypatch.setenv("VALENCE_OAUTH_USERNAME", "admin")
    monkeypatch.setenv("VALENCE_OAUTH_PASSWORD", "testpass")
    monkeypatch.setenv("VALENCE_EXTERNAL_URL", "https://valence.example.com")
    monkeypatch.setenv("VALENCE_FEDERATION_ENABLED", "false")

    return {
        "token_file": token_file,
        "clients_file": clients_file,
    }


@pytest.fixture
def mock_db():
    """Mock database for endpoints."""
    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = {
        "total_syncing": 0,
        "total_beliefs_sent": 0,
        "total_beliefs_received": 0,
        "local_beliefs": 0,
        "federated_beliefs": 0,
    }

    def mock_context(*args, **kwargs):
        class CM:
            def __enter__(self):
                return mock_cursor

            def __exit__(self, *args):
                pass

        return CM()

    with patch("our_db.get_cursor", mock_context):
        yield mock_cursor


# API prefix for v1 endpoints
API_V1 = "/api/v1"


@pytest.fixture
def fed_client(federation_env, mock_db) -> TestClient:
    """Create test client with federation enabled."""
    from valence.server.app import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def non_fed_client(non_federation_env, mock_db) -> TestClient:
    """Create test client with federation disabled."""
    from valence.server.app import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


# ============================================================================
# VFP Node Metadata Tests
# ============================================================================


class TestVFPNodeMetadata:
    """Tests for VFP node metadata endpoint."""

    def test_returns_did_document(self, fed_client):
        """Test endpoint returns DID document."""
        response = fed_client.get("/.well-known/vfp-node-metadata")

        assert response.status_code == 200
        data = response.json()

        assert "@context" in data
        assert "id" in data
        assert data["id"] == "did:vkb:web:valence.example.com"

    def test_includes_verification_methods(self, fed_client):
        """Test DID document includes verification methods."""
        response = fed_client.get("/.well-known/vfp-node-metadata")
        data = response.json()

        # Should have verification method since public key is configured
        assert "verificationMethod" in data
        assert len(data["verificationMethod"]) > 0

        vm = data["verificationMethod"][0]
        assert "Ed25519" in vm["type"]
        assert "publicKeyMultibase" in vm

    def test_includes_service_endpoints(self, fed_client):
        """Test DID document includes service endpoints."""
        response = fed_client.get("/.well-known/vfp-node-metadata")
        data = response.json()

        assert "service" in data

        # Should have VFP and MCP endpoints
        service_types = [s["type"] for s in data["service"]]
        assert "ValenceFederationProtocol" in service_types
        assert "ModelContextProtocol" in service_types

    def test_includes_vfp_extensions(self, fed_client):
        """Test DID document includes VFP extensions."""
        response = fed_client.get("/.well-known/vfp-node-metadata")
        data = response.json()

        assert "vfp:capabilities" in data
        assert "vfp:protocolVersion" in data

    def test_includes_profile(self, fed_client):
        """Test DID document includes profile."""
        response = fed_client.get("/.well-known/vfp-node-metadata")
        data = response.json()

        assert "vfp:profile" in data
        assert data["vfp:profile"]["name"] == "TestNode"

    def test_content_type(self, fed_client):
        """Test correct content type header."""
        response = fed_client.get("/.well-known/vfp-node-metadata")

        assert "application/did+ld+json" in response.headers.get("content-type", "")

    def test_cache_control(self, fed_client):
        """Test cache control header."""
        response = fed_client.get("/.well-known/vfp-node-metadata")

        assert "max-age=3600" in response.headers.get("cache-control", "")

    def test_404_when_federation_disabled(self, non_fed_client):
        """Test 404 when federation is disabled."""
        response = non_fed_client.get("/.well-known/vfp-node-metadata")

        assert response.status_code == 404


# ============================================================================
# VFP Trust Anchors Tests
# ============================================================================


class TestVFPTrustAnchors:
    """Tests for VFP trust anchors endpoint."""

    def test_404_when_federation_disabled(self, non_fed_client):
        """Test 404 when federation is disabled."""
        response = non_fed_client.get("/.well-known/vfp-trust-anchors")

        assert response.status_code == 404

    def test_404_when_trust_anchors_not_published(self, federation_env, mock_db, monkeypatch):
        """Test 404 when trust anchors not published."""
        monkeypatch.setenv("VALENCE_FEDERATION_PUBLISH_TRUST_ANCHORS", "false")

        # Reset settings
        import valence.server.config as config_module

        config_module._settings = None

        from valence.server.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/.well-known/vfp-trust-anchors")

        assert response.status_code == 404

    def test_returns_trust_anchors_when_published(self, federation_env, mock_db, monkeypatch):
        """Test returns trust anchors when published."""
        monkeypatch.setenv("VALENCE_FEDERATION_PUBLISH_TRUST_ANCHORS", "true")

        # Reset settings
        import valence.server.config as config_module

        config_module._settings = None

        from valence.server.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/.well-known/vfp-trust-anchors")

        assert response.status_code == 200
        data = response.json()

        assert "trust_anchors" in data
        assert "updated_at" in data


# ============================================================================
# Federation Status Tests
# ============================================================================


class TestFederationStatus:
    """Tests for federation status endpoint."""

    def test_returns_status(self, fed_client):
        """Test returns federation status."""
        response = fed_client.get(f"{API_V1}/federation/status")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "node" in data
        assert "federation" in data

    def test_node_info(self, fed_client):
        """Test node info in status."""
        response = fed_client.get(f"{API_V1}/federation/status")
        data = response.json()

        node = data["node"]
        assert node["did"] == "did:vkb:web:valence.example.com"
        assert node["name"] == "TestNode"
        assert "capabilities" in node
        assert "protocol_version" in node

    def test_federation_stats(self, fed_client):
        """Test federation stats in status."""
        response = fed_client.get(f"{API_V1}/federation/status")
        data = response.json()

        federation = data["federation"]
        assert "nodes" in federation
        assert "sync" in federation
        assert "beliefs" in federation

    def test_404_when_disabled(self, non_fed_client):
        """Test 404 when federation is disabled."""
        response = non_fed_client.get(f"{API_V1}/federation/status")

        assert response.status_code == 404


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestBuildDIDDocument:
    """Tests for _build_did_document helper."""

    def test_builds_basic_document(self, federation_env):
        """Test building basic DID document."""
        import valence.server.config as config_module

        config_module._settings = None
        import valence.server.oauth_models as oauth_module

        oauth_module._client_store = None

        from valence.server.config import get_settings
        from valence.server.federation_endpoints import _build_did_document

        settings = get_settings()
        doc = _build_did_document(settings)

        assert "@context" in doc
        assert "id" in doc

    def test_derives_did_from_external_url(self, monkeypatch, tmp_path):
        """Test DID derivation from external URL."""
        token_file = tmp_path / "tokens.json"
        clients_file = tmp_path / "clients.json"
        token_file.write_text('{"tokens": []}')
        clients_file.write_text('{"clients": []}')

        monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))
        monkeypatch.setenv("VALENCE_OAUTH_CLIENTS_FILE", str(clients_file))
        monkeypatch.setenv("VALENCE_FEDERATION_ENABLED", "true")
        monkeypatch.setenv("VALENCE_EXTERNAL_URL", "https://mynode.example.org")
        # JWT secret required for production (external URL)
        monkeypatch.setenv(
            "VALENCE_OAUTH_JWT_SECRET",
            "test-jwt-secret-for-testing-must-be-at-least-32-chars",
        )
        monkeypatch.delenv("VALENCE_FEDERATION_NODE_DID", raising=False)

        import valence.server.config as config_module

        config_module._settings = None
        import valence.server.oauth_models as oauth_module

        oauth_module._client_store = None

        from valence.server.config import get_settings
        from valence.server.federation_endpoints import _build_did_document

        settings = get_settings()
        settings.federation_node_did = None  # Ensure not set
        doc = _build_did_document(settings)

        assert doc["id"] == "did:vkb:web:mynode.example.org"


class TestDeriveDID:
    """Tests for _derive_did helper."""

    def test_derives_from_external_url(self, federation_env):
        """Test DID derivation from external URL."""
        import valence.server.config as config_module

        config_module._settings = None
        import valence.server.oauth_models as oauth_module

        oauth_module._client_store = None

        from valence.server.config import get_settings
        from valence.server.federation_endpoints import _derive_did

        settings = get_settings()
        settings.federation_node_did = None  # Force derivation
        did = _derive_did(settings)

        assert "did:vkb:web:" in did

    def test_fallback_to_localhost(self, monkeypatch, tmp_path):
        """Test DID fallback to localhost."""
        token_file = tmp_path / "tokens.json"
        clients_file = tmp_path / "clients.json"
        token_file.write_text('{"tokens": []}')
        clients_file.write_text('{"clients": []}')

        monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))
        monkeypatch.setenv("VALENCE_OAUTH_CLIENTS_FILE", str(clients_file))
        monkeypatch.setenv("VALENCE_FEDERATION_ENABLED", "true")
        monkeypatch.delenv("VALENCE_EXTERNAL_URL", raising=False)

        import valence.server.config as config_module

        config_module._settings = None

        from valence.server.config import get_settings
        from valence.server.federation_endpoints import _derive_did

        settings = get_settings()
        settings.external_url = None
        did = _derive_did(settings)

        assert "localhost" in did


class TestGetFederationStats:
    """Tests for _get_federation_stats helper."""

    def test_returns_stats_structure(self, mock_db):
        """Test stats structure."""
        from valence.server.federation_endpoints import _get_federation_stats

        stats = _get_federation_stats()

        assert "nodes" in stats
        assert "sync" in stats
        assert "beliefs" in stats

    def test_handles_db_error(self):
        """Test handles database error gracefully."""

        def mock_context(*args, **kwargs):
            class CM:
                def __enter__(self):
                    raise Exception("DB error")

                def __exit__(self, *args):
                    pass

            return CM()

        with patch("our_db.get_cursor", mock_context):
            from valence.server.federation_endpoints import _get_federation_stats

            stats = _get_federation_stats()

            # Should return default structure even on error
            assert "nodes" in stats
            assert stats["nodes"]["total"] == 0


class TestGetTrustAnchors:
    """Tests for _get_trust_anchors helper."""

    def test_returns_list(self, mock_db):
        """Test returns list structure."""
        from valence.server.federation_endpoints import _get_trust_anchors

        anchors = _get_trust_anchors()

        assert isinstance(anchors, list)

    def test_sql_operator_precedence_uses_parentheses(self, mock_db):
        """Regression: OR clause must be parenthesized with AND.

        Without parens, `AND ... OR ...` matches inactive nodes with
        trust_preference='anchor' due to operator precedence.
        Fixes #389.
        """
        from valence.server.federation_endpoints import _get_trust_anchors

        _get_trust_anchors()

        executed_sql = mock_db.execute.call_args[0][0]
        assert "(fn.trust_phase = 'anchor'" in executed_sql
        assert "OR unt.trust_preference = 'anchor')" in executed_sql
        assert "AND (fn.trust_phase" in executed_sql

    def test_handles_db_error(self):
        """Test handles database error gracefully."""

        def mock_context(*args, **kwargs):
            class CM:
                def __enter__(self):
                    raise Exception("DB error")

                def __exit__(self, *args):
                    pass

            return CM()

        with patch("our_db.get_cursor", mock_context):
            from valence.server.federation_endpoints import _get_trust_anchors

            anchors = _get_trust_anchors()

            assert anchors == []
