# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Token authentication for the HTTP MCP server."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Token prefix for identification
TOKEN_PREFIX = "vt_"


@dataclass
class Token:
    """A stored API token."""

    token_hash: str
    client_id: str
    scopes: list[str] = field(default_factory=lambda: ["mcp:access"])
    expires_at: float | None = None
    created_at: float = field(default_factory=time.time)
    description: str = ""

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def has_scope(self, scope: str) -> bool:
        """Check if token has a specific scope."""
        return scope in self.scopes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "token_hash": self.token_hash,
            "client_id": self.client_id,
            "scopes": self.scopes,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Token:
        """Create from dictionary."""
        return cls(
            token_hash=data["token_hash"],
            client_id=data["client_id"],
            scopes=data.get("scopes", ["mcp:access"]),
            expires_at=data.get("expires_at"),
            created_at=data.get("created_at", time.time()),
            description=data.get("description", ""),
        )


def hash_token(token: str) -> str:
    """Hash a token using SHA-256."""
    return hashlib.sha256(token.encode()).hexdigest()


def generate_token() -> str:
    """Generate a new secure token."""
    # Generate 32 bytes of randomness, encode as hex (64 chars)
    random_part = secrets.token_hex(32)
    return f"{TOKEN_PREFIX}{random_part}"


class TokenStore:
    """File-based token storage with hashed tokens."""

    def __init__(self, token_file: Path):
        self.token_file = token_file
        self._tokens: dict[str, Token] = {}  # hash -> Token
        self._load()

    def _load(self) -> None:
        """Load tokens from file."""
        if not self.token_file.exists():
            self._tokens = {}
            return

        try:
            with open(self.token_file) as f:
                data = json.load(f)
            self._tokens = {t["token_hash"]: Token.from_dict(t) for t in data.get("tokens", [])}
            logger.info(f"Loaded {len(self._tokens)} tokens from {self.token_file}")
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load tokens from {self.token_file}: {e}")
            self._tokens = {}

    def _save(self) -> None:
        """Save tokens to file."""
        # Ensure parent directory exists
        self.token_file.parent.mkdir(parents=True, exist_ok=True)

        data = {"tokens": [t.to_dict() for t in self._tokens.values()]}
        with open(self.token_file, "w") as f:
            json.dump(data, f, indent=2)
        # Set restrictive permissions
        self.token_file.chmod(0o600)
        logger.info(f"Saved {len(self._tokens)} tokens to {self.token_file}")

    def create(
        self,
        client_id: str,
        description: str = "",
        scopes: list[str] | None = None,
        expires_at: float | None = None,
    ) -> str:
        """Create a new token and return the raw token (shown only once).

        Args:
            client_id: Identifier for the client
            description: Human-readable description
            scopes: Permission scopes (default: ["mcp:access"])
            expires_at: Optional expiration timestamp

        Returns:
            The raw token string (must be saved by caller, not stored)
        """
        raw_token = generate_token()
        token_hash = hash_token(raw_token)

        token = Token(
            token_hash=token_hash,
            client_id=client_id,
            scopes=scopes or ["mcp:access"],
            expires_at=expires_at,
            description=description,
        )

        self._tokens[token_hash] = token
        self._save()

        logger.info(f"Created token for client '{client_id}'")
        return raw_token

    def verify(self, raw_token: str) -> Token | None:
        """Verify a token and return the Token object if valid.

        Args:
            raw_token: The raw token string to verify

        Returns:
            Token object if valid, None otherwise
        """
        if not raw_token:
            return None

        # Strip Bearer prefix if present
        if raw_token.startswith("Bearer "):
            raw_token = raw_token[7:]

        token_hash = hash_token(raw_token)
        token = self._tokens.get(token_hash)

        if token is None:
            logger.warning("Token not found")
            return None

        if token.is_expired():
            logger.debug(f"Token for client '{token.client_id}' is expired")
            return None

        return token

    def revoke(self, token_hash: str) -> bool:
        """Revoke a token by its hash.

        Args:
            token_hash: The hash of the token to revoke

        Returns:
            True if token was found and revoked
        """
        if token_hash in self._tokens:
            client_id = self._tokens[token_hash].client_id
            del self._tokens[token_hash]
            self._save()
            logger.info(f"Revoked token for client '{client_id}'")
            return True
        return False

    def list_tokens(self) -> list[Token]:
        """List all tokens (without revealing hashes for verification)."""
        return list(self._tokens.values())

    def get_by_client_id(self, client_id: str) -> list[Token]:
        """Get all tokens for a client ID."""
        return [t for t in self._tokens.values() if t.client_id == client_id]


# Global token store - lazy loaded
_token_store: TokenStore | None = None


def get_token_store(token_file: Path | None = None) -> TokenStore:
    """Get the global token store instance."""
    global _token_store
    if _token_store is None:
        if token_file is None:
            from .config import get_settings

            token_file = get_settings().token_file
        _token_store = TokenStore(token_file)
    return _token_store


def verify_token(raw_token: str, required_scope: str = "mcp:access") -> Token | None:
    """Verify a token and check for required scope.

    Args:
        raw_token: The raw token string
        required_scope: The scope required for this operation

    Returns:
        Token object if valid and has required scope, None otherwise
    """
    store = get_token_store()
    token = store.verify(raw_token)

    if token is None:
        return None

    if not token.has_scope(required_scope):
        logger.warning(f"Token for client '{token.client_id}' lacks required scope '{required_scope}'")
        return None

    return token
