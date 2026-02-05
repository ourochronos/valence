"""Peer storage for Valence Federation MVP.

Simple in-memory peer storage for the federation demo.
For production, use the database-backed federation_nodes table.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Peer:
    """A federation peer node."""
    
    did: str
    endpoint: str
    public_key_multibase: str
    name: str | None = None
    
    # Trust tracking
    trust_score: float = 0.5  # Start neutral
    beliefs_received: int = 0
    beliefs_sent: int = 0
    queries_received: int = 0
    queries_sent: int = 0
    
    # Timestamps
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "did": self.did,
            "endpoint": self.endpoint,
            "public_key_multibase": self.public_key_multibase,
            "name": self.name,
            "trust_score": self.trust_score,
            "beliefs_received": self.beliefs_received,
            "beliefs_sent": self.beliefs_sent,
            "queries_received": self.queries_received,
            "queries_sent": self.queries_sent,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Peer:
        """Create from dictionary."""
        return cls(
            did=data["did"],
            endpoint=data["endpoint"],
            public_key_multibase=data["public_key_multibase"],
            name=data.get("name"),
            trust_score=data.get("trust_score", 0.5),
            beliefs_received=data.get("beliefs_received", 0),
            beliefs_sent=data.get("beliefs_sent", 0),
            queries_received=data.get("queries_received", 0),
            queries_sent=data.get("queries_sent", 0),
            first_seen=datetime.fromisoformat(data["first_seen"]) if "first_seen" in data else datetime.now(),
            last_seen=datetime.fromisoformat(data["last_seen"]) if "last_seen" in data else datetime.now(),
        )


class PeerStore:
    """In-memory storage for federation peers.
    
    Optionally persists to a JSON file.
    """
    
    def __init__(self, persist_path: str | Path | None = None):
        """Initialize peer store.
        
        Args:
            persist_path: Optional path to persist peers to disk
        """
        self._peers: dict[str, Peer] = {}  # DID -> Peer
        self._persist_path = Path(persist_path) if persist_path else None
        
        if self._persist_path and self._persist_path.exists():
            self._load()
    
    def _load(self) -> None:
        """Load peers from disk."""
        if not self._persist_path:
            return
            
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
                for peer_data in data.get("peers", []):
                    peer = Peer.from_dict(peer_data)
                    self._peers[peer.did] = peer
            logger.info(f"Loaded {len(self._peers)} peers from {self._persist_path}")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            # OSError: file issues, JSONDecodeError: invalid JSON, KeyError/TypeError: schema mismatch
            logger.warning(f"Failed to load peers: {e}")
    
    def _save(self) -> None:
        """Save peers to disk."""
        if not self._persist_path:
            return
            
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "w") as f:
                json.dump({
                    "peers": [p.to_dict() for p in self._peers.values()]
                }, f, indent=2)
        except OSError as e:
            # File system errors (permissions, disk full, etc.)
            logger.warning(f"Failed to save peers: {e}")
    
    def add_peer(
        self,
        did: str,
        endpoint: str,
        public_key_multibase: str,
        name: str | None = None,
    ) -> Peer:
        """Add or update a peer.
        
        Args:
            did: Peer's DID
            endpoint: Peer's federation endpoint URL
            public_key_multibase: Peer's public key
            name: Optional human-readable name
            
        Returns:
            The Peer object
        """
        if did in self._peers:
            # Update existing peer
            peer = self._peers[did]
            peer.endpoint = endpoint
            peer.public_key_multibase = public_key_multibase
            if name:
                peer.name = name
            peer.last_seen = datetime.now()
        else:
            # Create new peer
            peer = Peer(
                did=did,
                endpoint=endpoint,
                public_key_multibase=public_key_multibase,
                name=name,
            )
            self._peers[did] = peer
            logger.info(f"Added new peer: {did}")
        
        self._save()
        return peer
    
    def get_peer(self, did: str) -> Peer | None:
        """Get a peer by DID."""
        return self._peers.get(did)
    
    def list_peers(self) -> list[Peer]:
        """List all peers."""
        return list(self._peers.values())
    
    def remove_peer(self, did: str) -> bool:
        """Remove a peer.
        
        Returns:
            True if peer was removed, False if not found
        """
        if did in self._peers:
            del self._peers[did]
            self._save()
            return True
        return False
    
    def update_trust(self, did: str, delta: float) -> float | None:
        """Update a peer's trust score.
        
        Args:
            did: Peer's DID
            delta: Amount to change trust by (-1.0 to 1.0)
            
        Returns:
            New trust score, or None if peer not found
        """
        peer = self._peers.get(did)
        if not peer:
            return None
        
        # Clamp to 0.0-1.0
        peer.trust_score = max(0.0, min(1.0, peer.trust_score + delta))
        peer.last_seen = datetime.now()
        self._save()
        return peer.trust_score
    
    def record_belief_sent(self, did: str) -> None:
        """Record that we sent a belief to a peer."""
        peer = self._peers.get(did)
        if peer:
            peer.beliefs_sent += 1
            peer.last_seen = datetime.now()
            self._save()
    
    def record_belief_received(self, did: str) -> None:
        """Record that we received a belief from a peer."""
        peer = self._peers.get(did)
        if peer:
            peer.beliefs_received += 1
            peer.last_seen = datetime.now()
            # Small trust increase for sharing beliefs
            self.update_trust(did, 0.01)
    
    def record_query(self, did: str, direction: str = "received") -> None:
        """Record a query.
        
        Args:
            did: Peer's DID
            direction: "sent" or "received"
        """
        peer = self._peers.get(did)
        if peer:
            if direction == "sent":
                peer.queries_sent += 1
            else:
                peer.queries_received += 1
            peer.last_seen = datetime.now()
            self._save()


# Global peer store (for simple usage)
_global_peer_store: PeerStore | None = None


def get_peer_store(persist_path: str | Path | None = None) -> PeerStore:
    """Get or create the global peer store.
    
    Args:
        persist_path: Optional path for persistence (only used on first call)
        
    Returns:
        The global PeerStore instance
    """
    global _global_peer_store
    
    if _global_peer_store is None:
        _global_peer_store = PeerStore(persist_path)
    
    return _global_peer_store
