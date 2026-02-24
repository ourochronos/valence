# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""HTTP client for the Valence REST API.

All CLI commands use this module to communicate with the server.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .config import get_cli_config


class ValenceAPIError(Exception):
    """Raised when the API returns an error response."""

    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(f"[{status_code}] {code}: {message}")


class ValenceConnectionError(Exception):
    """Raised when unable to connect to the server."""

    def __init__(self, server_url: str, detail: str = ""):
        self.server_url = server_url
        msg = f"Cannot connect to Valence server at {server_url}"
        if detail:
            msg += f": {detail}"
        msg += "\n\nIs the server running? Start with: valence-server"
        super().__init__(msg)


class ValenceClient:
    """Thin HTTP client for the Valence REST API."""

    def __init__(self, server_url: str | None = None, token: str | None = None, timeout: float | None = None):
        config = get_cli_config()
        self.base_url = (server_url if server_url is not None else config.server_url).rstrip("/")
        self.token = token if token is not None else config.token
        self.timeout = timeout if timeout is not None else config.timeout

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1{path}"

    def _handle_response(self, resp: httpx.Response) -> dict[str, Any]:
        """Parse response, raise ValenceAPIError on failure."""
        if resp.status_code >= 400:
            try:
                body = resp.json()
                error = body.get("error", {})
                raise ValenceAPIError(
                    status_code=resp.status_code,
                    code=error.get("code", "UNKNOWN"),
                    message=error.get("message", resp.text),
                )
            except (json.JSONDecodeError, KeyError):
                raise ValenceAPIError(resp.status_code, "UNKNOWN", resp.text)

        # For text/table output modes, server may return plain text
        content_type = resp.headers.get("content-type", "")
        if "text/plain" in content_type:
            return {"formatted": resp.text}

        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"formatted": resp.text}

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute an HTTP request with connection error handling."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.request(
                    method,
                    self._url(path),
                    headers=self._headers(),
                    params=params,
                    json=body if method in ("POST", "PUT", "PATCH", "DELETE") and body is not None else None,
                )
            return self._handle_response(resp)
        except httpx.ConnectError:
            raise ValenceConnectionError(self.base_url)
        except httpx.TimeoutException:
            raise ValenceConnectionError(self.base_url, "request timed out")

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """HTTP GET request."""
        return self._request("GET", path, params=params)

    def post(self, path: str, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """HTTP POST request."""
        return self._request("POST", path, params=params, body=body)

    def delete(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """HTTP DELETE request."""
        return self._request("DELETE", path, params=params)


def get_client() -> ValenceClient:
    """Get a configured ValenceClient instance."""
    return ValenceClient()
