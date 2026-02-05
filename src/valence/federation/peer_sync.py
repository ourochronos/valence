"""Peer Sync for Valence Federation - Week 2 MVP.

Two trusted agents sharing beliefs via file exchange.
Simple "phone call" model - not discovery/gossip.

Key features:
- Trust registry: Local registry of trusted peer DIDs and trust levels
- Export: Package beliefs for sharing with a specific peer
- Import: Receive beliefs from a peer, applying trust weighting
- Source-aware queries: Show which beliefs came from which peer
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ..core.confidence import DimensionalConfidence

logger = logging.getLogger(__name__)


# =============================================================================
# TRUST REGISTRY
# =============================================================================


@dataclass
class TrustedPeer:
    """A trusted peer in the local registry."""
    
    did: str
    trust_level: float  # 0.0 to 1.0
    name: str | None = None
    notes: str | None = None
    added_at: datetime = field(default_factory=datetime.now)
    last_sync_at: datetime | None = None
    beliefs_received: int = 0
    beliefs_sent: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "did": self.did,
            "trust_level": self.trust_level,
            "name": self.name,
            "notes": self.notes,
            "added_at": self.added_at.isoformat(),
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "beliefs_received": self.beliefs_received,
            "beliefs_sent": self.beliefs_sent,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrustedPeer:
        """Create from dictionary."""
        return cls(
            did=data["did"],
            trust_level=data["trust_level"],
            name=data.get("name"),
            notes=data.get("notes"),
            added_at=datetime.fromisoformat(data["added_at"]) if "added_at" in data else datetime.now(),
            last_sync_at=datetime.fromisoformat(data["last_sync_at"]) if data.get("last_sync_at") else None,
            beliefs_received=data.get("beliefs_received", 0),
            beliefs_sent=data.get("beliefs_sent", 0),
        )


class TrustRegistry:
    """Local file-based registry of trusted peers.
    
    Stores trust relationships in a simple JSON file.
    Location: ~/.valence/trust_registry.json or VALENCE_TRUST_REGISTRY env var.
    """
    
    DEFAULT_PATH = Path.home() / ".valence" / "trust_registry.json"
    
    def __init__(self, path: str | Path | None = None):
        """Initialize trust registry.
        
        Args:
            path: Path to registry file. Defaults to ~/.valence/trust_registry.json
        """
        if path:
            self._path = Path(path)
        else:
            from ..core.config import get_config
            config = get_config()
            self._path = Path(config.trust_registry_path) if config.trust_registry_path else self.DEFAULT_PATH
        
        self._peers: dict[str, TrustedPeer] = {}
        self._local_did: str | None = None
        self._load()
    
    def _load(self) -> None:
        """Load registry from disk."""
        if not self._path.exists():
            return
        
        try:
            with open(self._path) as f:
                data = json.load(f)
            
            self._local_did = data.get("local_did")
            for peer_data in data.get("peers", []):
                peer = TrustedPeer.from_dict(peer_data)
                self._peers[peer.did] = peer
            
            logger.debug(f"Loaded {len(self._peers)} peers from {self._path}")
        except Exception as e:
            logger.warning(f"Failed to load trust registry: {e}")
    
    def _save(self) -> None:
        """Save registry to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump({
                    "local_did": self._local_did,
                    "peers": [p.to_dict() for p in self._peers.values()],
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save trust registry: {e}")
    
    @property
    def local_did(self) -> str | None:
        """Get local node DID."""
        return self._local_did
    
    def set_local_did(self, did: str) -> None:
        """Set local node DID."""
        self._local_did = did
        self._save()
    
    def add_peer(
        self,
        did: str,
        trust_level: float,
        name: str | None = None,
        notes: str | None = None,
    ) -> TrustedPeer:
        """Add or update a trusted peer.
        
        Args:
            did: Peer's DID (e.g., did:vkb:web:alice.example.com)
            trust_level: Trust level from 0.0 (no trust) to 1.0 (full trust)
            name: Optional human-readable name
            notes: Optional notes about this peer
            
        Returns:
            The TrustedPeer object
        """
        if not 0.0 <= trust_level <= 1.0:
            raise ValueError(f"Trust level must be 0.0-1.0, got {trust_level}")
        
        if did in self._peers:
            # Update existing
            peer = self._peers[did]
            peer.trust_level = trust_level
            if name is not None:
                peer.name = name
            if notes is not None:
                peer.notes = notes
            logger.info(f"Updated peer trust: {did} -> {trust_level}")
        else:
            # Add new
            peer = TrustedPeer(
                did=did,
                trust_level=trust_level,
                name=name,
                notes=notes,
            )
            self._peers[did] = peer
            logger.info(f"Added new peer: {did} (trust={trust_level})")
        
        self._save()
        return peer
    
    def get_peer(self, did: str) -> TrustedPeer | None:
        """Get a peer by DID."""
        return self._peers.get(did)
    
    def list_peers(self) -> list[TrustedPeer]:
        """List all trusted peers, sorted by trust level (highest first)."""
        return sorted(self._peers.values(), key=lambda p: -p.trust_level)
    
    def remove_peer(self, did: str) -> bool:
        """Remove a peer from the registry.
        
        Returns:
            True if peer was removed, False if not found
        """
        if did in self._peers:
            del self._peers[did]
            self._save()
            logger.info(f"Removed peer: {did}")
            return True
        return False
    
    def get_trust_level(self, did: str) -> float:
        """Get trust level for a peer.
        
        Returns:
            Trust level (0.0-1.0), or 0.0 if peer not found
        """
        peer = self._peers.get(did)
        return peer.trust_level if peer else 0.0
    
    def record_sync(self, did: str, beliefs_received: int = 0, beliefs_sent: int = 0) -> None:
        """Record a sync interaction with a peer."""
        peer = self._peers.get(did)
        if peer:
            peer.last_sync_at = datetime.now()
            peer.beliefs_received += beliefs_received
            peer.beliefs_sent += beliefs_sent
            self._save()


# Global registry instance
_global_registry: TrustRegistry | None = None


def get_trust_registry(path: str | Path | None = None) -> TrustRegistry:
    """Get or create the global trust registry.
    
    Args:
        path: Optional path (only used on first call)
    
    Returns:
        The global TrustRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = TrustRegistry(path)
    return _global_registry


# =============================================================================
# EXPORT FORMAT
# =============================================================================


@dataclass
class ExportedBelief:
    """A belief packaged for export to a peer."""
    
    federation_id: str
    content: str
    confidence: dict[str, Any]
    domain_path: list[str]
    origin_did: str
    created_at: str
    content_hash: str
    
    # Optional metadata
    valid_from: str | None = None
    valid_until: str | None = None
    source_type: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportedBelief:
        """Create from dictionary."""
        return cls(
            federation_id=data["federation_id"],
            content=data["content"],
            confidence=data["confidence"],
            domain_path=data.get("domain_path", []),
            origin_did=data["origin_did"],
            created_at=data["created_at"],
            content_hash=data["content_hash"],
            valid_from=data.get("valid_from"),
            valid_until=data.get("valid_until"),
            source_type=data.get("source_type"),
        )


@dataclass
class ExportPackage:
    """A package of beliefs for export to a peer."""
    
    format_version: str = "1.0"
    exporter_did: str = ""
    recipient_did: str | None = None
    created_at: str = ""
    beliefs: list[ExportedBelief] = field(default_factory=list)
    
    # Stats
    total_beliefs: int = 0
    domain_summary: dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "format_version": self.format_version,
            "exporter_did": self.exporter_did,
            "recipient_did": self.recipient_did,
            "created_at": self.created_at,
            "total_beliefs": len(self.beliefs),
            "domain_summary": self.domain_summary,
            "beliefs": [b.to_dict() for b in self.beliefs],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportPackage:
        """Create from dictionary."""
        return cls(
            format_version=data.get("format_version", "1.0"),
            exporter_did=data.get("exporter_did", ""),
            recipient_did=data.get("recipient_did"),
            created_at=data.get("created_at", ""),
            total_beliefs=data.get("total_beliefs", 0),
            domain_summary=data.get("domain_summary", {}),
            beliefs=[ExportedBelief.from_dict(b) for b in data.get("beliefs", [])],
        )
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> ExportPackage:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# EXPORT FUNCTION
# =============================================================================


def export_beliefs(
    recipient_did: str | None = None,
    domain_filter: list[str] | None = None,
    min_confidence: float = 0.0,
    limit: int = 1000,
    include_federated: bool = False,
) -> ExportPackage:
    """Export beliefs for sharing with a peer.
    
    Args:
        recipient_did: DID of the intended recipient (for trust-based filtering)
        domain_filter: Only export beliefs in these domains
        min_confidence: Minimum confidence threshold
        limit: Maximum beliefs to export
        include_federated: If True, include beliefs received from federation
                          (default False = only export local beliefs)
    
    Returns:
        ExportPackage ready to serialize and share
    """
    from ..core.db import get_cursor
    
    registry = get_trust_registry()
    local_did = registry.local_did or "did:vkb:web:localhost"
    
    # Build query
    conditions = ["status = 'active'", "superseded_by_id IS NULL"]
    params: list[Any] = []
    
    if not include_federated:
        conditions.append("is_local = TRUE")
    
    if domain_filter:
        conditions.append("domain_path && %s")
        params.append(domain_filter)
    
    if min_confidence > 0:
        conditions.append("(confidence->>'overall')::numeric >= %s")
        params.append(min_confidence)
    
    params.append(limit)
    
    # Query beliefs
    query = f"""
        SELECT 
            id, content, confidence, domain_path, 
            created_at, valid_from, valid_until,
            extraction_method, content_hash
        FROM beliefs
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
        LIMIT %s
    """
    
    beliefs = []
    domain_counts: dict[str, int] = {}
    
    with get_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        
        for row in rows:
            # Compute content hash if not present
            content = row["content"]
            content_hash = row.get("content_hash") or hashlib.sha256(content.encode()).hexdigest()[:16]
            
            # Track domain stats
            for domain in (row["domain_path"] or []):
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Parse confidence
            conf = row["confidence"]
            if isinstance(conf, str):
                conf = json.loads(conf)
            
            beliefs.append(ExportedBelief(
                federation_id=str(row["id"]),
                content=content,
                confidence=conf,
                domain_path=row["domain_path"] or [],
                origin_did=local_did,
                created_at=row["created_at"].isoformat(),
                content_hash=content_hash,
                valid_from=row["valid_from"].isoformat() if row.get("valid_from") else None,
                valid_until=row["valid_until"].isoformat() if row.get("valid_until") else None,
                source_type=row.get("extraction_method"),
            ))
    
    # Update registry with export count
    if recipient_did:
        registry.record_sync(recipient_did, beliefs_sent=len(beliefs))
    
    return ExportPackage(
        exporter_did=local_did,
        recipient_did=recipient_did,
        created_at=datetime.now().isoformat(),
        beliefs=beliefs,
        domain_summary=domain_counts,
    )


# =============================================================================
# IMPORT FUNCTION
# =============================================================================


@dataclass
class ImportResult:
    """Result of importing beliefs from a peer."""
    
    total_in_package: int
    imported: int
    skipped_duplicate: int
    skipped_low_trust: int
    skipped_error: int
    trust_level_applied: float
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def import_beliefs(
    package: ExportPackage,
    from_did: str,
    trust_override: float | None = None,
) -> ImportResult:
    """Import beliefs from a peer.
    
    Args:
        package: The export package to import
        from_did: DID of the peer we're importing from
        trust_override: Override trust level (otherwise uses registry)
    
    Returns:
        ImportResult with statistics
    """
    from ..core.db import get_cursor
    
    registry = get_trust_registry()
    
    # Get trust level
    if trust_override is not None:
        trust_level = trust_override
    else:
        trust_level = registry.get_trust_level(from_did)
    
    # Skip if trust is too low
    if trust_level < 0.01:
        return ImportResult(
            total_in_package=len(package.beliefs),
            imported=0,
            skipped_duplicate=0,
            skipped_low_trust=len(package.beliefs),
            skipped_error=0,
            trust_level_applied=trust_level,
            errors=["Trust level too low - add peer with 'valence peer add'"],
        )
    
    imported = 0
    skipped_dup = 0
    skipped_err = 0
    errors = []
    
    with get_cursor() as cur:
        for belief in package.beliefs:
            try:
                # Check for duplicate by federation_id
                cur.execute(
                    "SELECT id FROM beliefs WHERE federation_id = %s OR content_hash = %s",
                    (belief.federation_id, belief.content_hash)
                )
                if cur.fetchone():
                    skipped_dup += 1
                    continue
                
                # Apply trust weighting to confidence
                original_conf = belief.confidence
                if isinstance(original_conf, dict):
                    weighted_conf = dict(original_conf)
                    original_overall = weighted_conf.get("overall", 0.7)
                    # Trust-weighted confidence: peer's confidence * our trust in them
                    weighted_conf["overall"] = original_overall * trust_level
                    weighted_conf["_original_overall"] = original_overall
                    weighted_conf["_peer_trust"] = trust_level
                else:
                    weighted_conf = {"overall": 0.7 * trust_level}
                
                # Insert belief
                cur.execute("""
                    INSERT INTO beliefs (
                        content, confidence, domain_path,
                        valid_from, valid_until,
                        status, is_local, federation_id,
                        content_hash, extraction_method,
                        origin_node_did, origin_node_trust
                    ) VALUES (
                        %s, %s, %s,
                        %s, %s,
                        'active', FALSE, %s,
                        %s, %s,
                        %s, %s
                    )
                    RETURNING id
                """, (
                    belief.content,
                    json.dumps(weighted_conf),
                    belief.domain_path,
                    datetime.fromisoformat(belief.valid_from) if belief.valid_from else None,
                    datetime.fromisoformat(belief.valid_until) if belief.valid_until else None,
                    belief.federation_id,
                    belief.content_hash,
                    belief.source_type or 'peer_import',
                    from_did,
                    trust_level,
                ))
                
                imported += 1
                
            except Exception as e:
                skipped_err += 1
                errors.append(f"Error importing {belief.federation_id}: {e}")
                logger.warning(f"Error importing belief: {e}")
    
    # Update registry
    registry.record_sync(from_did, beliefs_received=imported)
    
    return ImportResult(
        total_in_package=len(package.beliefs),
        imported=imported,
        skipped_duplicate=skipped_dup,
        skipped_low_trust=0,
        skipped_error=skipped_err,
        trust_level_applied=trust_level,
        errors=errors,
    )


# =============================================================================
# FEDERATED QUERY
# =============================================================================


@dataclass
class FederatedQueryResult:
    """A belief result with source attribution."""
    
    id: str
    content: str
    confidence: dict[str, Any]
    domain_path: list[str]
    similarity: float
    created_at: str
    
    # Source attribution
    is_local: bool
    origin_did: str | None
    origin_trust: float | None
    effective_confidence: float  # Confidence * trust weighting
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def query_federated(
    query_text: str,
    scope: str = "local",
    domain_filter: list[str] | None = None,
    min_confidence: float = 0.0,
    limit: int = 10,
    threshold: float = 0.3,
) -> list[FederatedQueryResult]:
    """Query beliefs with source attribution.
    
    Args:
        query_text: Semantic search query
        scope: "local" (only local beliefs), "federated" (include peer beliefs)
        domain_filter: Only include beliefs from these domains
        min_confidence: Minimum confidence threshold
        limit: Maximum results
        threshold: Minimum similarity threshold
    
    Returns:
        List of FederatedQueryResult with source attribution
    """
    from ..core.db import get_cursor
    
    # Generate embedding
    try:
        from openai import OpenAI
        from ..core.config import get_config
        config = get_config()
        client = OpenAI(api_key=config.openai_api_key)
        response = client.embeddings.create(
            model='text-embedding-3-small',
            input=query_text
        )
        embedding = response.data[0].embedding
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
    except Exception as e:
        logger.warning(f"Embedding failed, falling back to text search: {e}")
        embedding_str = None
    
    # Build query conditions
    conditions = ["status = 'active'", "superseded_by_id IS NULL"]
    params: list[Any] = []
    
    if scope == "local":
        conditions.append("is_local = TRUE")
    # scope == "federated" includes both local and peer beliefs
    
    if domain_filter:
        conditions.append("domain_path && %s")
        params.append(domain_filter)
    
    if min_confidence > 0:
        conditions.append("(confidence->>'overall')::numeric >= %s")
        params.append(min_confidence)
    
    results = []
    
    with get_cursor() as cur:
        if embedding_str:
            # Semantic search
            params.extend([embedding_str, embedding_str, limit])
            cur.execute(f"""
                SELECT 
                    id, content, confidence, domain_path,
                    created_at, is_local,
                    origin_node_did, origin_node_trust,
                    1 - (embedding <=> %s::vector) as similarity
                FROM beliefs
                WHERE embedding IS NOT NULL
                  AND {' AND '.join(conditions)}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, params)
        else:
            # Fallback to text search
            params.extend([query_text, query_text, query_text, limit])
            cur.execute(f"""
                SELECT 
                    id, content, confidence, domain_path,
                    created_at, is_local,
                    origin_node_did, origin_node_trust,
                    ts_rank(content_tsv, websearch_to_tsquery('english', %s)) as similarity
                FROM beliefs
                WHERE content_tsv @@ websearch_to_tsquery('english', %s)
                  AND {' AND '.join(conditions)}
                ORDER BY ts_rank(content_tsv, websearch_to_tsquery('english', %s)) DESC
                LIMIT %s
            """, params)
        
        rows = cur.fetchall()
        
        for row in rows:
            similarity = float(row["similarity"])
            if similarity < threshold:
                continue
            
            conf = row["confidence"]
            if isinstance(conf, str):
                conf = json.loads(conf)
            
            overall = conf.get("overall", 0.7)
            is_local = row["is_local"]
            origin_trust = row.get("origin_node_trust")
            
            # Effective confidence already accounts for trust weighting on import
            # For local beliefs, it's just the confidence
            effective = overall if is_local else overall
            
            results.append(FederatedQueryResult(
                id=str(row["id"]),
                content=row["content"],
                confidence=conf,
                domain_path=row["domain_path"] or [],
                similarity=similarity,
                created_at=row["created_at"].isoformat(),
                is_local=is_local,
                origin_did=row.get("origin_node_did"),
                origin_trust=origin_trust,
                effective_confidence=effective,
            ))
    
    # Sort by effective confidence * similarity
    results.sort(key=lambda r: r.effective_confidence * r.similarity, reverse=True)
    
    return results
