"""Data models for resilient storage.

These models represent shards, shard sets, and metadata for
erasure-coded storage of beliefs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class RedundancyLevel(Enum):
    """Predefined redundancy levels per spec.
    
    Each level specifies (data_shards, total_shards):
    - PERSONAL: 3 of 5 - 67% overhead, survives any 2 failures
    - FEDERATION: 5 of 9 - 80% overhead, survives any 4 failures
    - PARANOID: 7 of 15 - 114% overhead, survives any 8 failures
    - MINIMAL: 2 of 3 - 50% overhead, survives 1 failure (testing)
    """
    
    MINIMAL = (2, 3)      # 50% overhead, survives 1 failure
    PERSONAL = (3, 5)     # 67% overhead, survives 2 failures
    FEDERATION = (5, 9)   # 80% overhead, survives 4 failures
    PARANOID = (7, 15)    # 114% overhead, survives 8 failures
    
    @property
    def data_shards(self) -> int:
        """Number of data shards (k)."""
        return self.value[0]
    
    @property
    def total_shards(self) -> int:
        """Total number of shards (n)."""
        return self.value[1]
    
    @property
    def parity_shards(self) -> int:
        """Number of parity/recovery shards."""
        return self.total_shards - self.data_shards
    
    @property
    def max_failures(self) -> int:
        """Maximum number of shard failures that can be tolerated."""
        return self.parity_shards
    
    @property
    def overhead_percent(self) -> float:
        """Storage overhead as a percentage."""
        return ((self.total_shards - self.data_shards) / self.data_shards) * 100
    
    @classmethod
    def from_string(cls, name: str) -> RedundancyLevel:
        """Create from string name (case-insensitive)."""
        try:
            return cls[name.upper()]
        except KeyError:
            valid = [level.name.lower() for level in cls]
            raise ValueError(f"Unknown redundancy level: {name}. Valid: {valid}")
    
    @classmethod
    def custom(cls, data_shards: int, total_shards: int) -> tuple[int, int]:
        """Validate custom redundancy parameters.
        
        Returns (k, n) tuple if valid, raises ValueError otherwise.
        """
        if data_shards < 1:
            raise ValueError("data_shards must be at least 1")
        if total_shards <= data_shards:
            raise ValueError("total_shards must be greater than data_shards")
        if total_shards > 255:
            raise ValueError("total_shards cannot exceed 255 (Reed-Solomon limit)")
        return (data_shards, total_shards)


@dataclass
class ShardMetadata:
    """Metadata for a single shard."""
    
    shard_id: UUID = field(default_factory=uuid4)
    index: int = 0                    # Position in shard set (0 to n-1)
    is_parity: bool = False           # True if this is a parity shard
    size_bytes: int = 0               # Size of shard data
    checksum: str = ""                # SHA-256 hash of shard data
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    backend_id: str | None = None     # Which backend stores this shard
    location: str | None = None       # Backend-specific location
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shard_id": str(self.shard_id),
            "index": self.index,
            "is_parity": self.is_parity,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "backend_id": self.backend_id,
            "location": self.location,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardMetadata:
        """Create from dictionary."""
        return cls(
            shard_id=UUID(data["shard_id"]) if isinstance(data["shard_id"], str) else data["shard_id"],
            index=data["index"],
            is_parity=data.get("is_parity", False),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            backend_id=data.get("backend_id"),
            location=data.get("location"),
        )


@dataclass
class StorageShard:
    """A single shard of erasure-coded data."""
    
    data: bytes                       # The actual shard data
    metadata: ShardMetadata           # Shard metadata
    
    @property
    def index(self) -> int:
        """Shard index for convenience."""
        return self.metadata.index
    
    @property
    def is_valid(self) -> bool:
        """Check if shard data matches its checksum."""
        if not self.metadata.checksum:
            return True  # No checksum to verify
        import hashlib
        actual = hashlib.sha256(self.data).hexdigest()
        return actual == self.metadata.checksum
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (data as base64)."""
        import base64
        return {
            "data": base64.b64encode(self.data).decode("ascii"),
            "metadata": self.metadata.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StorageShard:
        """Create from dictionary."""
        import base64
        return cls(
            data=base64.b64decode(data["data"]),
            metadata=ShardMetadata.from_dict(data["metadata"]),
        )


@dataclass
class ShardSet:
    """A complete set of shards for a piece of data.
    
    Contains all shards (data + parity) plus metadata about
    the original data and encoding parameters.
    """
    
    set_id: UUID = field(default_factory=uuid4)
    shards: list[StorageShard | None] = field(default_factory=list)
    
    # Encoding parameters
    data_shards_k: int = 3            # Number of data shards
    total_shards_n: int = 5           # Total number of shards
    
    # Original data metadata
    original_size: int = 0            # Size of original data
    original_checksum: str = ""       # SHA-256 of original data
    content_type: str = "application/octet-stream"
    
    # Merkle tree for integrity
    merkle_root: str = ""             # Root hash of Merkle tree
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    verified_at: datetime | None = None
    
    # Optional reference to what this stores
    belief_id: UUID | None = None
    
    @property
    def available_shards(self) -> list[StorageShard]:
        """Return list of non-None shards."""
        return [s for s in self.shards if s is not None]
    
    @property
    def available_count(self) -> int:
        """Number of available shards."""
        return len(self.available_shards)
    
    @property
    def can_recover(self) -> bool:
        """Whether we have enough shards to recover original data."""
        return self.available_count >= self.data_shards_k
    
    @property
    def missing_indices(self) -> list[int]:
        """Indices of missing/corrupted shards."""
        available_indices = {s.index for s in self.available_shards}
        return [i for i in range(self.total_shards_n) if i not in available_indices]
    
    @property
    def redundancy_level(self) -> RedundancyLevel | None:
        """Return matching RedundancyLevel if one exists."""
        for level in RedundancyLevel:
            if level.data_shards == self.data_shards_k and level.total_shards == self.total_shards_n:
                return level
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "set_id": str(self.set_id),
            "shards": [s.to_dict() if s else None for s in self.shards],
            "data_shards_k": self.data_shards_k,
            "total_shards_n": self.total_shards_n,
            "original_size": self.original_size,
            "original_checksum": self.original_checksum,
            "content_type": self.content_type,
            "merkle_root": self.merkle_root,
            "created_at": self.created_at.isoformat(),
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "belief_id": str(self.belief_id) if self.belief_id else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardSet:
        """Create from dictionary."""
        shards = [
            StorageShard.from_dict(s) if s else None
            for s in data.get("shards", [])
        ]
        return cls(
            set_id=UUID(data["set_id"]) if isinstance(data["set_id"], str) else data["set_id"],
            shards=shards,
            data_shards_k=data["data_shards_k"],
            total_shards_n=data["total_shards_n"],
            original_size=data.get("original_size", 0),
            original_checksum=data.get("original_checksum", ""),
            content_type=data.get("content_type", "application/octet-stream"),
            merkle_root=data.get("merkle_root", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            verified_at=datetime.fromisoformat(data["verified_at"]) if data.get("verified_at") else None,
            belief_id=UUID(data["belief_id"]) if data.get("belief_id") else None,
        )


@dataclass
class RecoveryResult:
    """Result of a data recovery operation."""
    
    success: bool
    data: bytes | None = None
    shards_used: int = 0
    shards_available: int = 0
    shards_required: int = 0
    recovery_time_ms: float = 0.0
    error_message: str | None = None
    repaired_indices: list[int] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        import base64
        return {
            "success": self.success,
            "data": base64.b64encode(self.data).decode("ascii") if self.data else None,
            "shards_used": self.shards_used,
            "shards_available": self.shards_available,
            "shards_required": self.shards_required,
            "recovery_time_ms": self.recovery_time_ms,
            "error_message": self.error_message,
            "repaired_indices": self.repaired_indices,
        }


@dataclass
class IntegrityReport:
    """Report from integrity verification."""
    
    is_valid: bool
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_shards: int = 0
    valid_shards: int = 0
    corrupted_shards: list[int] = field(default_factory=list)
    missing_shards: list[int] = field(default_factory=list)
    merkle_valid: bool = True
    checksum_valid: bool = True
    can_recover: bool = True
    details: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "checked_at": self.checked_at.isoformat(),
            "total_shards": self.total_shards,
            "valid_shards": self.valid_shards,
            "corrupted_shards": self.corrupted_shards,
            "missing_shards": self.missing_shards,
            "merkle_valid": self.merkle_valid,
            "checksum_valid": self.checksum_valid,
            "can_recover": self.can_recover,
            "details": self.details,
        }
