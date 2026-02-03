"""OAuth 2.1 data models and storage for the Valence MCP server."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jwt

from .config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class OAuthClient:
    """An OAuth 2.1 client (dynamically registered or pre-configured)."""

    client_id: str
    client_name: str
    redirect_uris: list[str]
    grant_types: list[str] = field(default_factory=lambda: ["authorization_code", "refresh_token"])
    response_types: list[str] = field(default_factory=lambda: ["code"])
    scope: str = "mcp:tools mcp:resources"
    client_secret: str | None = None  # For confidential clients
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "redirect_uris": self.redirect_uris,
            "grant_types": self.grant_types,
            "response_types": self.response_types,
            "scope": self.scope,
            "client_secret": self.client_secret,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OAuthClient:
        """Create from dictionary."""
        return cls(
            client_id=data["client_id"],
            client_name=data["client_name"],
            redirect_uris=data["redirect_uris"],
            grant_types=data.get("grant_types", ["authorization_code", "refresh_token"]),
            response_types=data.get("response_types", ["code"]),
            scope=data.get("scope", "mcp:tools mcp:resources"),
            client_secret=data.get("client_secret"),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class AuthorizationCode:
    """An OAuth authorization code (short-lived)."""

    code: str
    client_id: str
    redirect_uri: str
    scope: str
    code_challenge: str  # PKCE
    code_challenge_method: str  # S256
    user_id: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 600)

    def is_expired(self) -> bool:
        """Check if code is expired."""
        return time.time() > self.expires_at


@dataclass
class RefreshToken:
    """An OAuth refresh token."""

    token_hash: str
    client_id: str
    user_id: str
    scope: str
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class OAuthClientStore:
    """File-based storage for OAuth clients."""

    def __init__(self, clients_file: Path):
        self.clients_file = clients_file
        self._clients: dict[str, OAuthClient] = {}
        self._load()

    def _load(self) -> None:
        """Load clients from file."""
        if not self.clients_file.exists():
            self._clients = {}
            return

        try:
            with open(self.clients_file) as f:
                data = json.load(f)
            self._clients = {
                c["client_id"]: OAuthClient.from_dict(c) for c in data.get("clients", [])
            }
            logger.info(f"Loaded {len(self._clients)} OAuth clients from {self.clients_file}")
        except Exception as e:
            logger.error(f"Failed to load OAuth clients: {e}")
            self._clients = {}

    def _save(self) -> None:
        """Save clients to file."""
        self.clients_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"clients": [c.to_dict() for c in self._clients.values()]}
        with open(self.clients_file, "w") as f:
            json.dump(data, f, indent=2)
        self.clients_file.chmod(0o600)

    def register(
        self,
        client_name: str,
        redirect_uris: list[str],
        grant_types: list[str] | None = None,
        response_types: list[str] | None = None,
        scope: str | None = None,
    ) -> OAuthClient:
        """Register a new OAuth client (Dynamic Client Registration)."""
        client_id = secrets.token_urlsafe(16)

        client = OAuthClient(
            client_id=client_id,
            client_name=client_name,
            redirect_uris=redirect_uris,
            grant_types=grant_types or ["authorization_code", "refresh_token"],
            response_types=response_types or ["code"],
            scope=scope or "mcp:tools mcp:resources",
        )

        self._clients[client_id] = client
        self._save()

        logger.info(f"Registered OAuth client: {client_name} ({client_id})")
        return client

    def get(self, client_id: str) -> OAuthClient | None:
        """Get a client by ID."""
        return self._clients.get(client_id)

    def validate_redirect_uri(self, client_id: str, redirect_uri: str) -> bool:
        """Validate that a redirect URI is registered for a client."""
        client = self.get(client_id)
        if not client:
            return False
        return redirect_uri in client.redirect_uris


class AuthorizationCodeStore:
    """In-memory storage for authorization codes (short-lived)."""

    def __init__(self):
        self._codes: dict[str, AuthorizationCode] = {}

    def create(
        self,
        client_id: str,
        redirect_uri: str,
        scope: str,
        code_challenge: str,
        code_challenge_method: str,
        user_id: str,
    ) -> str:
        """Create a new authorization code."""
        settings = get_settings()
        code = secrets.token_urlsafe(32)

        auth_code = AuthorizationCode(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            user_id=user_id,
            expires_at=time.time() + settings.oauth_code_expiry,
        )

        self._codes[code] = auth_code
        return code

    def consume(self, code: str) -> AuthorizationCode | None:
        """Consume (get and delete) an authorization code."""
        auth_code = self._codes.pop(code, None)
        if auth_code and auth_code.is_expired():
            return None
        return auth_code

    def cleanup_expired(self) -> None:
        """Remove expired codes."""
        now = time.time()
        expired = [c for c, ac in self._codes.items() if ac.expires_at < now]
        for c in expired:
            del self._codes[c]


class RefreshTokenStore:
    """In-memory storage for refresh tokens."""

    def __init__(self):
        self._tokens: dict[str, RefreshToken] = {}

    def create(self, client_id: str, user_id: str, scope: str) -> str:
        """Create a new refresh token."""
        settings = get_settings()
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        refresh_token = RefreshToken(
            token_hash=token_hash,
            client_id=client_id,
            user_id=user_id,
            scope=scope,
            expires_at=time.time() + settings.oauth_refresh_token_expiry,
        )

        self._tokens[token_hash] = refresh_token
        return raw_token

    def validate(self, raw_token: str) -> RefreshToken | None:
        """Validate a refresh token."""
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        token = self._tokens.get(token_hash)

        if token and token.is_expired():
            del self._tokens[token_hash]
            return None

        return token

    def revoke(self, raw_token: str) -> bool:
        """Revoke a refresh token."""
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        if token_hash in self._tokens:
            del self._tokens[token_hash]
            return True
        return False


def create_access_token(
    client_id: str,
    user_id: str,
    scope: str,
    audience: str,
) -> str:
    """Create a JWT access token."""
    settings = get_settings()
    now = time.time()

    payload = {
        "iss": settings.issuer_url,
        "sub": user_id,
        "aud": audience,
        "client_id": client_id,
        "scope": scope,
        "iat": int(now),
        "exp": int(now + settings.oauth_access_token_expiry),
        "jti": secrets.token_urlsafe(16),
    }

    return jwt.encode(payload, settings.oauth_jwt_secret, algorithm=settings.oauth_jwt_algorithm)


def verify_access_token(token: str, expected_audience: str) -> dict[str, Any] | None:
    """Verify a JWT access token.

    Returns the payload if valid, None otherwise.
    """
    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.oauth_jwt_secret,
            algorithms=[settings.oauth_jwt_algorithm],
            audience=expected_audience,
            issuer=settings.issuer_url,
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except jwt.InvalidAudienceError:
        logger.debug("Invalid audience")
        return None
    except jwt.InvalidIssuerError:
        logger.debug("Invalid issuer")
        return None
    except jwt.PyJWTError as e:
        logger.debug(f"JWT verification failed: {e}")
        return None


def verify_pkce(code_verifier: str, code_challenge: str, method: str = "S256") -> bool:
    """Verify PKCE code verifier against challenge."""
    if method != "S256":
        logger.warning(f"Unsupported PKCE method: {method}")
        return False

    # S256: BASE64URL(SHA256(code_verifier))
    import base64

    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    computed_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    return secrets.compare_digest(computed_challenge, code_challenge)


# Global stores - lazy loaded
_client_store: OAuthClientStore | None = None
_code_store: AuthorizationCodeStore | None = None
_refresh_store: RefreshTokenStore | None = None


def get_client_store() -> OAuthClientStore:
    """Get the global OAuth client store."""
    global _client_store
    if _client_store is None:
        settings = get_settings()
        _client_store = OAuthClientStore(settings.oauth_clients_file)
    return _client_store


def get_code_store() -> AuthorizationCodeStore:
    """Get the global authorization code store."""
    global _code_store
    if _code_store is None:
        _code_store = AuthorizationCodeStore()
    return _code_store


def get_refresh_store() -> RefreshTokenStore:
    """Get the global refresh token store."""
    global _refresh_store
    if _refresh_store is None:
        _refresh_store = RefreshTokenStore()
    return _refresh_store
