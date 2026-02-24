# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Valence HTTP MCP Server.

Provides a unified HTTP endpoint for all Valence MCP tools.
Supports MCP JSON-RPC protocol over HTTP with Bearer token authentication.

Usage:
    # Start the server
    valence-server

    # Or with uvicorn directly
    uvicorn valence.server.app:app --port 8420

    # Manage tokens
    valence-token create --client-id "my-client"
    valence-token list
    valence-token revoke --client-id "my-client"
"""

from .auth import TokenStore, get_token_store, verify_token
from .config import ServerSettings, get_settings

__all__ = [
    "ServerSettings",
    "get_settings",
    "TokenStore",
    "verify_token",
    "get_token_store",
]
