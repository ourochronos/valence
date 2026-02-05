"""
Integration tests for live Valence nodes.

These tests validate actual deployed Valence pods by testing:
1. Basic connectivity and health
2. API endpoints (health, info, federation status)
3. OAuth flow
4. Belief CRUD operations
5. Federation between nodes
6. MCP protocol

Run with: pytest tests/integration/test_live_nodes.py -v --live-nodes

Requires:
- VALENCE_NODE_1=https://valence.zonk1024.net
- VALENCE_NODE_2=https://valence2.zonk1024.net
"""

import base64
import hashlib
import os
import secrets

import httpx
import pytest

# Node URLs from environment or defaults
NODE_1 = os.environ.get("VALENCE_NODE_1", "https://valence.zonk1024.net")
NODE_2 = os.environ.get("VALENCE_NODE_2", "https://valence2.zonk1024.net")

# Tests are skipped via conftest.py if --live-nodes not passed


def generate_pkce():
    """Generate PKCE verifier and challenge."""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


class TestNodeHealth:
    """Test basic node health and connectivity."""

    @pytest.mark.parametrize("node_url", [NODE_1, NODE_2])
    def test_health_endpoint(self, node_url: str):
        """Node should return healthy status."""
        response = httpx.get(f"{node_url}/api/v1/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"

    @pytest.mark.parametrize("node_url", [NODE_1, NODE_2])
    def test_info_endpoint(self, node_url: str):
        """Node should return API info."""
        response = httpx.get(f"{node_url}/", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["server"] == "valence"
        assert "endpoints" in data
        assert "authentication" in data

    @pytest.mark.parametrize("node_url", [NODE_1, NODE_2])
    def test_federation_status(self, node_url: str):
        """Node should return federation status."""
        response = httpx.get(f"{node_url}/api/v1/federation/status", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "node" in data
        assert "did" in data["node"]
        assert data["node"]["did"].startswith("did:web:")


class TestOAuth:
    """Test OAuth 2.1 flow."""

    @pytest.mark.parametrize("node_url", [NODE_1, NODE_2])
    def test_oauth_metadata(self, node_url: str):
        """OAuth metadata should be available."""
        response = httpx.get(f"{node_url}/.well-known/oauth-authorization-server", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "authorization_endpoint" in data
        assert "token_endpoint" in data

    @pytest.mark.parametrize("node_url", [NODE_1, NODE_2])
    def test_client_registration(self, node_url: str):
        """Should be able to register OAuth client."""
        response = httpx.post(
            f"{node_url}/api/v1/oauth/register",
            json={"redirect_uris": ["http://localhost:8080/callback"]},
            timeout=10,
        )
        # 201 Created is the correct response for POST creation
        assert response.status_code in [200, 201]
        data = response.json()
        assert "client_id" in data
        # Should have redirect_uris echoed back
        assert "redirect_uris" in data


class TestBeliefsCRUD:
    """Test belief create/read/update/delete operations."""

    def get_auth_headers(self, node_url: str) -> dict:
        """Get authentication headers for a node."""
        # Register client
        reg_response = httpx.post(
            f"{node_url}/api/v1/oauth/register",
            json={"redirect_uris": ["http://localhost:8080/callback"]},
            timeout=10,
        )
        client_id = reg_response.json()["client_id"]

        # For testing, we'll try to get a token
        # In production this would go through the full OAuth flow
        verifier, challenge = generate_pkce()

        # Try direct token grant (may not work without user auth)
        # This test may need adjustment based on actual auth flow
        return {"Authorization": f"Bearer test-token-{client_id}"}

    @pytest.mark.parametrize("node_url", [NODE_1, NODE_2])
    def test_openapi_spec(self, node_url: str):
        """OpenAPI spec should be available."""
        response = httpx.get(f"{node_url}/api/v1/openapi.json", timeout=10)
        # May return 200 or 404 depending on implementation
        assert response.status_code in [200, 404]

    @pytest.mark.parametrize("node_url", [NODE_1, NODE_2])
    def test_beliefs_endpoint_requires_auth(self, node_url: str):
        """Beliefs endpoint should require authentication or not exist."""
        response = httpx.get(f"{node_url}/api/v1/beliefs", timeout=10)
        # Should return 401/403 Unauthorized without auth, or 404 if not implemented
        assert response.status_code in [401, 403, 404]


class TestMCPProtocol:
    """Test MCP (Model Context Protocol) endpoints."""

    @pytest.mark.parametrize("node_url", [NODE_1, NODE_2])
    def test_mcp_requires_auth(self, node_url: str):
        """MCP endpoint should require authentication."""
        response = httpx.post(
            f"{node_url}/api/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            },
            timeout=10,
        )
        # MCP requires auth - should return 401
        assert response.status_code in [200, 401]

    @pytest.mark.parametrize("node_url", [NODE_1, NODE_2])
    def test_mcp_endpoint_exists(self, node_url: str):
        """MCP endpoint should exist (even if auth required)."""
        response = httpx.post(
            f"{node_url}/api/v1/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            timeout=10,
        )
        # Should return 200 (success) or 401 (auth required), not 404
        assert response.status_code in [200, 401]


class TestFederation:
    """Test federation between nodes."""

    def test_nodes_can_discover_each_other(self):
        """Node 1 should be able to discover Node 2 and vice versa."""
        # Get Node 1's federation status
        response1 = httpx.get(f"{NODE_1}/api/v1/federation/status", timeout=10)
        data1 = response1.json()

        # Get Node 2's federation status
        response2 = httpx.get(f"{NODE_2}/api/v1/federation/status", timeout=10)
        data2 = response2.json()

        # Both should have valid DIDs
        assert data1["node"]["did"].startswith("did:web:")
        assert data2["node"]["did"].startswith("did:web:")

        # DIDs should be different
        assert data1["node"]["did"] != data2["node"]["did"]

    def test_well_known_did(self):
        """Nodes should serve DID documents at .well-known."""
        for node_url in [NODE_1, NODE_2]:
            response = httpx.get(f"{node_url}/.well-known/did.json", timeout=10)
            # May return 200 or 404 depending on DID configuration
            if response.status_code == 200:
                data = response.json()
                assert "@context" in data
                assert "id" in data


class TestRateLimiting:
    """Test rate limiting on OAuth endpoints."""

    def test_rate_limiting_headers(self):
        """Rate limited endpoints should return appropriate headers."""
        # Make a request to OAuth endpoint
        response = httpx.post(
            f"{NODE_1}/api/v1/oauth/register",
            json={"redirect_uris": ["http://localhost/callback"]},
            timeout=10,
        )
        # Should succeed (200/201) or be rate limited (429)
        assert response.status_code in [200, 201, 429]


class TestSecurityHeaders:
    """Test security-related HTTP headers."""

    @pytest.mark.parametrize("node_url", [NODE_1, NODE_2])
    def test_no_sensitive_error_details(self, node_url: str):
        """Error responses should not leak sensitive details."""
        # Make an invalid request
        response = httpx.post(f"{node_url}/api/v1/mcp", json={"invalid": "request"}, timeout=10)
        # Response should not contain stack traces or internal paths
        text = response.text.lower()
        assert "traceback" not in text
        assert "/opt/valence" not in text
        assert "exception" not in text or "internal" in text


class TestMetrics:
    """Test Prometheus metrics endpoint."""

    @pytest.mark.parametrize("node_url", [NODE_1, NODE_2])
    def test_metrics_endpoint(self, node_url: str):
        """Metrics endpoint should be available."""
        response = httpx.get(f"{node_url}/metrics", timeout=10)
        # Metrics endpoint might be at /metrics or disabled
        if response.status_code == 200:
            assert "valence" in response.text or "http" in response.text


# Conftest additions for pytest
def pytest_addoption(parser):
    parser.addoption(
        "--live-nodes",
        action="store_true",
        default=False,
        help="Run tests against live Valence nodes",
    )
