"""Tests for TLS validation in federation sync."""

from unittest.mock import patch

import pytest

from valence.federation.sync import TLSRequiredError, validate_endpoint_tls


class TestTLSValidation:
    """Test TLS requirement enforcement."""

    def test_https_allowed_when_tls_required(self):
        """HTTPS endpoints should pass when TLS is required."""
        with patch("valence.federation.sync.get_config") as mock:
            mock.return_value.require_tls = True
            # Should not raise
            validate_endpoint_tls("https://example.com/api")

    def test_http_rejected_when_tls_required(self):
        """HTTP endpoints should be rejected when TLS is required."""
        with patch("valence.federation.sync.get_config") as mock:
            mock.return_value.require_tls = True
            with pytest.raises(TLSRequiredError) as exc_info:
                validate_endpoint_tls("http://example.com/api")
            assert "TLS required" in str(exc_info.value)

    def test_http_allowed_when_tls_not_required(self):
        """HTTP endpoints should be allowed when TLS is not required."""
        with patch("valence.federation.sync.get_config") as mock:
            mock.return_value.require_tls = False
            # Should not raise
            validate_endpoint_tls("http://example.com/api")

    def test_https_allowed_when_tls_not_required(self):
        """HTTPS endpoints should always be allowed."""
        with patch("valence.federation.sync.get_config") as mock:
            mock.return_value.require_tls = False
            # Should not raise
            validate_endpoint_tls("https://example.com/api")
