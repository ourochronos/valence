"""Storage backend abstraction for resilient storage.

Provides a common interface for different storage backends, allowing
shards to be distributed across multiple storage providers for
redundancy.

Supported backends (per spec):
- Local file system
- Memory (for testing)
- S3-compatible (Backblaze B2, MinIO, AWS)
- IPFS (future)
- Decentralized (Sia, Filecoin - future)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from .models import StorageShard, ShardMetadata, ShardSet


class StorageBackendError(Exception):
    """Base exception for storage backend errors."""
    pass


class ShardNotFoundError(StorageBackendError):
    """Raised when a shard cannot be found."""
    pass


class StorageQuotaExceededError(StorageBackendError):
    """Raised when storage quota is exceeded."""
    pass


@dataclass
class StorageStats:
    """Statistics for a storage backend."""
    
    backend_id: str
    backend_type: str
    total_shards: int = 0
    total_bytes: int = 0
    quota_bytes: int | None = None
    available_bytes: int | None = None
    last_sync: datetime | None = None
    healthy: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend_id": self.backend_id,
            "backend_type": self.backend_type,
            "total_shards": self.total_shards,
            "total_bytes": self.total_bytes,
            "quota_bytes": self.quota_bytes,
            "available_bytes": self.available_bytes,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "healthy": self.healthy,
        }


class StorageBackend(ABC):
    """Abstract base class for storage backends.
    
    All storage backends must implement this interface to allow
    unified shard management across different storage providers.
    """
    
    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Unique identifier for this backend instance."""
        pass
    
    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Type of backend (e.g., 'local', 'memory', 's3')."""
        pass
    
    @abstractmethod
    async def store_shard(self, shard: StorageShard) -> str:
        """Store a shard and return its location identifier.
        
        Args:
            shard: StorageShard to store
        
        Returns:
            Location identifier (backend-specific)
        
        Raises:
            StorageBackendError: If storage fails
        """
        pass
    
    @abstractmethod
    async def retrieve_shard(self, location: str) -> StorageShard:
        """Retrieve a shard by its location.
        
        Args:
            location: Location identifier from store_shard
        
        Returns:
            The stored StorageShard
        
        Raises:
            ShardNotFoundError: If shard not found
            StorageBackendError: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def delete_shard(self, location: str) -> bool:
        """Delete a shard by its location.
        
        Args:
            location: Location identifier
        
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def shard_exists(self, location: str) -> bool:
        """Check if a shard exists at the given location.
        
        Args:
            location: Location identifier
        
        Returns:
            True if shard exists
        """
        pass
    
    @abstractmethod
    async def list_shards(self, prefix: str = "") -> list[str]:
        """List all shard locations, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter by
        
        Returns:
            List of location identifiers
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> StorageStats:
        """Get statistics for this backend.
        
        Returns:
            StorageStats with current backend state
        """
        pass
    
    async def health_check(self) -> bool:
        """Check if the backend is healthy and accessible.
        
        Returns:
            True if backend is operational
        """
        try:
            await self.get_stats()
            return True
        except Exception:
            return False
    
    async def store_shard_set(
        self,
        shard_set: ShardSet,
        indices: list[int] | None = None
    ) -> dict[int, str]:
        """Store multiple shards from a shard set.
        
        Args:
            shard_set: ShardSet containing shards
            indices: Optional list of shard indices to store (default: all)
        
        Returns:
            Dict mapping shard index to location
        """
        locations = {}
        shards_to_store = shard_set.shards
        
        if indices is not None:
            shards_to_store = [s for s in shards_to_store if s and s.index in indices]
        
        for shard in shards_to_store:
            if shard:
                location = await self.store_shard(shard)
                locations[shard.index] = location
                shard.metadata.backend_id = self.backend_id
                shard.metadata.location = location
        
        return locations
    
    async def retrieve_shard_set(
        self,
        locations: dict[int, str],
        shard_set_template: ShardSet
    ) -> ShardSet:
        """Retrieve shards and reconstruct a shard set.
        
        Args:
            locations: Dict mapping shard index to location
            shard_set_template: Template with metadata (k, n, etc.)
        
        Returns:
            ShardSet with retrieved shards
        """
        shards: list[StorageShard | None] = [None] * shard_set_template.total_shards_n
        
        for index, location in locations.items():
            try:
                shard = await self.retrieve_shard(location)
                shards[index] = shard
            except ShardNotFoundError:
                pass  # Leave as None
        
        return ShardSet(
            set_id=shard_set_template.set_id,
            shards=shards,
            data_shards_k=shard_set_template.data_shards_k,
            total_shards_n=shard_set_template.total_shards_n,
            original_size=shard_set_template.original_size,
            original_checksum=shard_set_template.original_checksum,
            content_type=shard_set_template.content_type,
            merkle_root=shard_set_template.merkle_root,
            created_at=shard_set_template.created_at,
            belief_id=shard_set_template.belief_id,
        )


class MemoryBackend(StorageBackend):
    """In-memory storage backend for testing.
    
    Stores shards in a dictionary. Not persistent.
    """
    
    def __init__(self, backend_id: str = "memory-default"):
        self._id = backend_id
        self._storage: dict[str, bytes] = {}
        self._metadata: dict[str, dict] = {}
    
    @property
    def backend_id(self) -> str:
        return self._id
    
    @property
    def backend_type(self) -> str:
        return "memory"
    
    async def store_shard(self, shard: StorageShard) -> str:
        """Store shard in memory."""
        location = f"{shard.metadata.shard_id}"
        self._storage[location] = shard.data
        self._metadata[location] = shard.metadata.to_dict()
        return location
    
    async def retrieve_shard(self, location: str) -> StorageShard:
        """Retrieve shard from memory."""
        if location not in self._storage:
            raise ShardNotFoundError(f"Shard not found: {location}")
        
        data = self._storage[location]
        metadata = ShardMetadata.from_dict(self._metadata[location])
        return StorageShard(data=data, metadata=metadata)
    
    async def delete_shard(self, location: str) -> bool:
        """Delete shard from memory."""
        if location in self._storage:
            del self._storage[location]
            del self._metadata[location]
            return True
        return False
    
    async def shard_exists(self, location: str) -> bool:
        """Check if shard exists in memory."""
        return location in self._storage
    
    async def list_shards(self, prefix: str = "") -> list[str]:
        """List all shard locations."""
        if prefix:
            return [k for k in self._storage.keys() if k.startswith(prefix)]
        return list(self._storage.keys())
    
    async def get_stats(self) -> StorageStats:
        """Get memory backend statistics."""
        total_bytes = sum(len(v) for v in self._storage.values())
        return StorageStats(
            backend_id=self._id,
            backend_type="memory",
            total_shards=len(self._storage),
            total_bytes=total_bytes,
            healthy=True,
        )
    
    def clear(self) -> None:
        """Clear all stored shards."""
        self._storage.clear()
        self._metadata.clear()


class LocalFileBackend(StorageBackend):
    """Local file system storage backend.
    
    Stores shards as individual files in a directory structure.
    """
    
    def __init__(
        self,
        base_path: str | Path,
        backend_id: str | None = None,
        quota_bytes: int | None = None
    ):
        """Initialize local file backend.
        
        Args:
            base_path: Base directory for shard storage
            backend_id: Optional unique ID (default: derived from path)
            quota_bytes: Optional storage quota
        """
        self._base_path = Path(base_path)
        self._id = backend_id or f"local-{hashlib.md5(str(base_path).encode()).hexdigest()[:8]}"
        self._quota_bytes = quota_bytes
        
        # Create base directory if it doesn't exist
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._shards_dir = self._base_path / "shards"
        self._metadata_dir = self._base_path / "metadata"
        self._shards_dir.mkdir(exist_ok=True)
        self._metadata_dir.mkdir(exist_ok=True)
    
    @property
    def backend_id(self) -> str:
        return self._id
    
    @property
    def backend_type(self) -> str:
        return "local"
    
    def _shard_path(self, location: str) -> Path:
        """Get file path for a shard."""
        # Use first 2 chars as subdirectory for better file distribution
        subdir = location[:2] if len(location) >= 2 else "00"
        return self._shards_dir / subdir / f"{location}.shard"
    
    def _metadata_path(self, location: str) -> Path:
        """Get file path for shard metadata."""
        subdir = location[:2] if len(location) >= 2 else "00"
        return self._metadata_dir / subdir / f"{location}.json"
    
    async def store_shard(self, shard: StorageShard) -> str:
        """Store shard to local file system."""
        location = str(shard.metadata.shard_id)
        
        # Check quota
        if self._quota_bytes:
            stats = await self.get_stats()
            if stats.total_bytes + len(shard.data) > self._quota_bytes:
                raise StorageQuotaExceededError(
                    f"Storage quota exceeded: {stats.total_bytes} + {len(shard.data)} > {self._quota_bytes}"
                )
        
        # Create subdirectories
        shard_path = self._shard_path(location)
        metadata_path = self._metadata_path(location)
        shard_path.parent.mkdir(exist_ok=True)
        metadata_path.parent.mkdir(exist_ok=True)
        
        # Write shard data
        shard_path.write_bytes(shard.data)
        
        # Write metadata
        metadata_path.write_text(json.dumps(shard.metadata.to_dict(), indent=2))
        
        return location
    
    async def retrieve_shard(self, location: str) -> StorageShard:
        """Retrieve shard from local file system."""
        shard_path = self._shard_path(location)
        metadata_path = self._metadata_path(location)
        
        if not shard_path.exists():
            raise ShardNotFoundError(f"Shard not found: {location}")
        
        data = shard_path.read_bytes()
        
        if metadata_path.exists():
            metadata_dict = json.loads(metadata_path.read_text())
            metadata = ShardMetadata.from_dict(metadata_dict)
        else:
            # Create basic metadata from file
            metadata = ShardMetadata(
                shard_id=UUID(location) if location else uuid4(),
                size_bytes=len(data),
                checksum=hashlib.sha256(data).hexdigest(),
            )
        
        return StorageShard(data=data, metadata=metadata)
    
    async def delete_shard(self, location: str) -> bool:
        """Delete shard from local file system."""
        shard_path = self._shard_path(location)
        metadata_path = self._metadata_path(location)
        
        deleted = False
        if shard_path.exists():
            shard_path.unlink()
            deleted = True
        if metadata_path.exists():
            metadata_path.unlink()
        
        return deleted
    
    async def shard_exists(self, location: str) -> bool:
        """Check if shard exists on file system."""
        return self._shard_path(location).exists()
    
    async def list_shards(self, prefix: str = "") -> list[str]:
        """List all shard locations."""
        locations = []
        for shard_file in self._shards_dir.rglob("*.shard"):
            location = shard_file.stem
            if not prefix or location.startswith(prefix):
                locations.append(location)
        return locations
    
    async def get_stats(self) -> StorageStats:
        """Get local backend statistics."""
        total_shards = 0
        total_bytes = 0
        
        for shard_file in self._shards_dir.rglob("*.shard"):
            total_shards += 1
            total_bytes += shard_file.stat().st_size
        
        # Get available disk space
        try:
            statvfs = os.statvfs(self._base_path)
            available_bytes = statvfs.f_frsize * statvfs.f_bavail
        except (OSError, AttributeError):
            available_bytes = None
        
        return StorageStats(
            backend_id=self._id,
            backend_type="local",
            total_shards=total_shards,
            total_bytes=total_bytes,
            quota_bytes=self._quota_bytes,
            available_bytes=available_bytes,
            healthy=True,
        )
    
    async def clear(self) -> None:
        """Remove all stored shards."""
        if self._shards_dir.exists():
            shutil.rmtree(self._shards_dir)
            self._shards_dir.mkdir()
        if self._metadata_dir.exists():
            shutil.rmtree(self._metadata_dir)
            self._metadata_dir.mkdir()


class BackendRegistry:
    """Registry for managing multiple storage backends.
    
    Allows distributing shards across multiple backends for redundancy.
    """
    
    def __init__(self):
        self._backends: dict[str, StorageBackend] = {}
    
    def register(self, backend: StorageBackend) -> None:
        """Register a storage backend.
        
        Args:
            backend: StorageBackend instance to register
        """
        self._backends[backend.backend_id] = backend
    
    def unregister(self, backend_id: str) -> bool:
        """Unregister a storage backend.
        
        Args:
            backend_id: ID of backend to remove
        
        Returns:
            True if backend was removed
        """
        if backend_id in self._backends:
            del self._backends[backend_id]
            return True
        return False
    
    def get(self, backend_id: str) -> StorageBackend | None:
        """Get a backend by ID.
        
        Args:
            backend_id: Backend ID to look up
        
        Returns:
            StorageBackend or None if not found
        """
        return self._backends.get(backend_id)
    
    def list_backends(self) -> list[str]:
        """List all registered backend IDs.
        
        Returns:
            List of backend IDs
        """
        return list(self._backends.keys())
    
    async def get_all_stats(self) -> dict[str, StorageStats]:
        """Get statistics for all backends.
        
        Returns:
            Dict mapping backend_id to StorageStats
        """
        stats = {}
        for backend_id, backend in self._backends.items():
            try:
                stats[backend_id] = await backend.get_stats()
            except Exception as e:
                stats[backend_id] = StorageStats(
                    backend_id=backend_id,
                    backend_type=backend.backend_type,
                    healthy=False,
                )
        return stats
    
    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all backends.
        
        Returns:
            Dict mapping backend_id to health status
        """
        health = {}
        for backend_id, backend in self._backends.items():
            health[backend_id] = await backend.health_check()
        return health
    
    async def distribute_shard_set(
        self,
        shard_set: ShardSet,
        distribution: dict[int, str] | None = None
    ) -> dict[int, tuple[str, str]]:
        """Distribute shards across backends.
        
        Args:
            shard_set: ShardSet to distribute
            distribution: Optional mapping of shard index to backend_id
                         (default: round-robin)
        
        Returns:
            Dict mapping shard index to (backend_id, location)
        """
        if not self._backends:
            raise StorageBackendError("No backends registered")
        
        backend_ids = list(self._backends.keys())
        result = {}
        
        for shard in shard_set.available_shards:
            if distribution and shard.index in distribution:
                backend_id = distribution[shard.index]
            else:
                # Round-robin distribution
                backend_id = backend_ids[shard.index % len(backend_ids)]
            
            backend = self._backends.get(backend_id)
            if not backend:
                raise StorageBackendError(f"Backend not found: {backend_id}")
            
            location = await backend.store_shard(shard)
            shard.metadata.backend_id = backend_id
            shard.metadata.location = location
            result[shard.index] = (backend_id, location)
        
        return result
    
    async def retrieve_distributed(
        self,
        locations: dict[int, tuple[str, str]],
        shard_set_template: ShardSet
    ) -> ShardSet:
        """Retrieve shards distributed across backends.
        
        Args:
            locations: Dict mapping index to (backend_id, location)
            shard_set_template: Template with metadata
        
        Returns:
            ShardSet with retrieved shards
        """
        shards: list[StorageShard | None] = [None] * shard_set_template.total_shards_n
        
        for index, (backend_id, location) in locations.items():
            backend = self._backends.get(backend_id)
            if backend:
                try:
                    shard = await backend.retrieve_shard(location)
                    shards[index] = shard
                except ShardNotFoundError:
                    pass
        
        return ShardSet(
            set_id=shard_set_template.set_id,
            shards=shards,
            data_shards_k=shard_set_template.data_shards_k,
            total_shards_n=shard_set_template.total_shards_n,
            original_size=shard_set_template.original_size,
            original_checksum=shard_set_template.original_checksum,
            content_type=shard_set_template.content_type,
            merkle_root=shard_set_template.merkle_root,
            created_at=shard_set_template.created_at,
            belief_id=shard_set_template.belief_id,
        )
