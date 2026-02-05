"""Reed-Solomon erasure coding for resilient storage.

Implements the erasure coding layer per SPEC.md using Reed-Solomon
error correction codes. Allows recovery of original data from any k
of n shards.

The implementation uses Galois Field GF(2^8) arithmetic, which limits
n to 255 maximum shards but provides excellent error correction properties.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Sequence
from uuid import UUID, uuid4

from .models import (
    RedundancyLevel,
    ShardMetadata,
    ShardSet,
    StorageShard,
    RecoveryResult,
)


class ErasureCodingError(Exception):
    """Base exception for erasure coding errors."""
    pass


class InsufficientShardsError(ErasureCodingError):
    """Raised when there aren't enough shards to recover data."""
    pass


class CorruptedDataError(ErasureCodingError):
    """Raised when recovered data doesn't match expected checksum."""
    pass


# Galois Field GF(2^8) arithmetic tables
# Using primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11d)
_GF_EXP = [0] * 512  # Anti-log table
_GF_LOG = [0] * 256  # Log table
_GF_INITIALIZED = False
_GF_LOCK = threading.Lock()  # Thread lock for Galois table initialization


def _init_galois_tables() -> None:
    """Initialize Galois Field lookup tables.
    
    Thread-safe initialization using double-checked locking pattern.
    """
    global _GF_INITIALIZED, _GF_EXP, _GF_LOG
    
    if _GF_INITIALIZED:
        return
    
    with _GF_LOCK:
        # Double-check after acquiring lock
        if _GF_INITIALIZED:
            return
        
        # Generate exp and log tables
        x = 1
        for i in range(255):
            _GF_EXP[i] = x
            _GF_LOG[x] = i
            x <<= 1
            if x & 0x100:
                x ^= 0x11d  # Primitive polynomial
        
        # Extend exp table for easier multiplication
        for i in range(255, 512):
            _GF_EXP[i] = _GF_EXP[i - 255]
        
        _GF_INITIALIZED = True


def _gf_mul(a: int, b: int) -> int:
    """Multiply two numbers in GF(2^8)."""
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _gf_div(a: int, b: int) -> int:
    """Divide two numbers in GF(2^8)."""
    if b == 0:
        raise ZeroDivisionError("Division by zero in GF(2^8)")
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] - _GF_LOG[b]) % 255]


def _gf_pow(x: int, power: int) -> int:
    """Raise x to power in GF(2^8)."""
    if power == 0:
        return 1
    if x == 0:
        return 0
    return _GF_EXP[(_GF_LOG[x] * power) % 255]


def _gf_inverse(x: int) -> int:
    """Multiplicative inverse in GF(2^8)."""
    if x == 0:
        raise ZeroDivisionError("Zero has no inverse")
    return _GF_EXP[255 - _GF_LOG[x]]


@dataclass
class ErasureCodec:
    """Reed-Solomon erasure codec for belief storage.
    
    Encodes data into n shards where any k shards can reconstruct
    the original data. Uses systematic encoding where the first k
    shards contain the original data.
    
    Args:
        level: Predefined RedundancyLevel or None for custom
        data_shards: Number of data shards (k) - only if level is None
        total_shards: Total number of shards (n) - only if level is None
    
    Example:
        # Using predefined level
        codec = ErasureCodec(RedundancyLevel.FEDERATION)
        
        # Using custom parameters
        codec = ErasureCodec(data_shards=4, total_shards=6)
        
        # Encode
        shard_set = codec.encode(data)
        
        # Decode (with some shards missing)
        recovered = codec.decode(partial_shards)
    """
    
    level: RedundancyLevel | None = None
    data_shards: int = 3
    total_shards: int = 5
    
    def __post_init__(self) -> None:
        """Initialize codec parameters."""
        _init_galois_tables()
        
        if self.level is not None:
            self.data_shards = self.level.data_shards
            self.total_shards = self.level.total_shards
        
        # Validate parameters
        if self.data_shards < 1:
            raise ValueError("data_shards must be at least 1")
        if self.total_shards <= self.data_shards:
            raise ValueError("total_shards must be greater than data_shards")
        if self.total_shards > 255:
            raise ValueError("total_shards cannot exceed 255")
        
        self._parity_shards = self.total_shards - self.data_shards
        self._encoding_matrix = self._build_encoding_matrix()
    
    def _build_encoding_matrix(self) -> list[list[int]]:
        """Build the encoding matrix for Reed-Solomon.
        
        Uses a Vandermonde matrix for systematic encoding.
        The first k rows are the identity matrix (data shards).
        The remaining rows generate parity shards.
        """
        matrix = []
        
        # Identity matrix for data shards (systematic encoding)
        for i in range(self.data_shards):
            row = [0] * self.data_shards
            row[i] = 1
            matrix.append(row)
        
        # Vandermonde matrix rows for parity shards
        for i in range(self._parity_shards):
            row = []
            for j in range(self.data_shards):
                # Vandermonde: element (i,j) = alpha^(i*j)
                # Use i+1 to avoid all-zeros first row
                row.append(_gf_pow(i + 1, j))
            matrix.append(row)
        
        return matrix
    
    def encode(self, data: bytes, belief_id: str | None = None) -> ShardSet:
        """Encode data into erasure-coded shards.
        
        Args:
            data: Raw bytes to encode
            belief_id: Optional belief ID to associate with shard set
        
        Returns:
            ShardSet containing all encoded shards
        """
        start_time = time.time()
        
        # Calculate shard size (pad data to multiple of k)
        data_len = len(data)
        shard_size = (data_len + self.data_shards - 1) // self.data_shards
        
        # Pad data to exact multiple of shard_size * data_shards
        padded_len = shard_size * self.data_shards
        padded_data = data + bytes(padded_len - data_len)
        
        # Split into data shards
        data_chunks = [
            padded_data[i * shard_size:(i + 1) * shard_size]
            for i in range(self.data_shards)
        ]
        
        # Create all shards (data + parity)
        shards: list[StorageShard] = []
        
        for shard_idx in range(self.total_shards):
            if shard_idx < self.data_shards:
                # Data shard - direct copy
                shard_data = data_chunks[shard_idx]
                is_parity = False
            else:
                # Parity shard - matrix multiplication
                shard_data = self._compute_parity_shard(
                    data_chunks,
                    shard_idx - self.data_shards
                )
                is_parity = True
            
            # Create metadata
            checksum = hashlib.sha256(shard_data).hexdigest()
            metadata = ShardMetadata(
                shard_id=uuid4(),
                index=shard_idx,
                is_parity=is_parity,
                size_bytes=len(shard_data),
                checksum=checksum,
            )
            
            shards.append(StorageShard(data=shard_data, metadata=metadata))
        
        # Create shard set with metadata
        original_checksum = hashlib.sha256(data).hexdigest()
        shard_set = ShardSet(
            set_id=uuid4(),
            shards=shards,
            data_shards_k=self.data_shards,
            total_shards_n=self.total_shards,
            original_size=data_len,
            original_checksum=original_checksum,
            belief_id=UUID(belief_id) if belief_id else None,
        )
        
        return shard_set
    
    def _compute_parity_shard(
        self,
        data_chunks: list[bytes],
        parity_idx: int
    ) -> bytes:
        """Compute a single parity shard.
        
        Args:
            data_chunks: List of data shard bytes
            parity_idx: Index of parity shard (0-indexed within parity shards)
        
        Returns:
            Bytes of the computed parity shard
        """
        shard_size = len(data_chunks[0])
        result = bytearray(shard_size)
        
        # Get the encoding row for this parity shard
        matrix_row = self._encoding_matrix[self.data_shards + parity_idx]
        
        for byte_idx in range(shard_size):
            value = 0
            for chunk_idx, chunk in enumerate(data_chunks):
                # GF(2^8) multiply and XOR accumulate
                value ^= _gf_mul(matrix_row[chunk_idx], chunk[byte_idx])
            result[byte_idx] = value
        
        return bytes(result)
    
    def decode(self, shard_set: ShardSet) -> RecoveryResult:
        """Decode/recover original data from available shards.
        
        Args:
            shard_set: ShardSet with at least k available shards
        
        Returns:
            RecoveryResult with recovered data or error details
        """
        start_time = time.time()
        
        available = shard_set.available_shards
        available_count = len(available)
        
        if available_count < self.data_shards:
            return RecoveryResult(
                success=False,
                shards_used=0,
                shards_available=available_count,
                shards_required=self.data_shards,
                error_message=f"Need {self.data_shards} shards, only {available_count} available",
            )
        
        # Verify shard integrity
        valid_shards = [s for s in available if s.is_valid]
        if len(valid_shards) < self.data_shards:
            return RecoveryResult(
                success=False,
                shards_used=0,
                shards_available=len(valid_shards),
                shards_required=self.data_shards,
                error_message=f"Only {len(valid_shards)} valid shards after integrity check",
            )
        
        # Use first k valid shards
        shards_to_use = valid_shards[:self.data_shards]
        indices_used = [s.index for s in shards_to_use]
        
        # Check if we have all data shards (fast path)
        data_shard_indices = set(range(self.data_shards))
        available_data_indices = set(s.index for s in shards_to_use if s.index < self.data_shards)
        
        if available_data_indices == data_shard_indices:
            # All data shards present - just concatenate
            sorted_shards = sorted(
                [s for s in shards_to_use if s.index < self.data_shards],
                key=lambda s: s.index
            )
            recovered = b"".join(s.data for s in sorted_shards)
        else:
            # Need matrix inversion to recover
            recovered = self._recover_with_matrix_inversion(shards_to_use)
        
        # Trim to original size
        recovered = recovered[:shard_set.original_size]
        
        # Verify checksum
        actual_checksum = hashlib.sha256(recovered).hexdigest()
        if shard_set.original_checksum and actual_checksum != shard_set.original_checksum:
            return RecoveryResult(
                success=False,
                shards_used=len(shards_to_use),
                shards_available=available_count,
                shards_required=self.data_shards,
                error_message=f"Checksum mismatch: expected {shard_set.original_checksum[:16]}..., got {actual_checksum[:16]}...",
            )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RecoveryResult(
            success=True,
            data=recovered,
            shards_used=len(shards_to_use),
            shards_available=available_count,
            shards_required=self.data_shards,
            recovery_time_ms=elapsed_ms,
        )
    
    def _recover_with_matrix_inversion(
        self,
        shards: list[StorageShard]
    ) -> bytes:
        """Recover data using matrix inversion when data shards are missing.
        
        Uses Gauss-Jordan elimination to invert the submatrix corresponding
        to the available shards, then multiplies by shard data to recover.
        
        The key insight: if M is the encoding matrix and D is the original data,
        then shard_data = M × D. To recover D from a subset of shards:
        D = M_sub^(-1) × shard_data_sub
        
        Gauss-Jordan applies the same row operations to both [M|I] to get [I|M^(-1)].
        Row swaps don't affect the final inverse - they're applied to both sides.
        """
        k = self.data_shards
        shard_size = len(shards[0].data)
        
        # Build submatrix from available shards (rows corresponding to shard indices)
        indices = [s.index for s in shards]
        submatrix = [self._encoding_matrix[i][:] for i in indices]
        
        # Build identity matrix - will become the inverse
        inverse = [[1 if i == j else 0 for j in range(k)] for i in range(k)]
        
        # Gauss-Jordan elimination: transform [submatrix | I] to [I | inverse]
        for col in range(k):
            # Find pivot (non-zero element in this column, at or below diagonal)
            pivot_row = None
            for row in range(col, k):
                if submatrix[row][col] != 0:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                raise ErasureCodingError(f"Matrix is singular, cannot recover from indices {indices}")
            
            # Swap rows if needed (same operation on both matrices)
            if pivot_row != col:
                submatrix[col], submatrix[pivot_row] = submatrix[pivot_row], submatrix[col]
                inverse[col], inverse[pivot_row] = inverse[pivot_row], inverse[col]
            
            # Scale pivot row to make pivot element = 1
            pivot_val = submatrix[col][col]
            pivot_inv = _gf_inverse(pivot_val)
            for j in range(k):
                submatrix[col][j] = _gf_mul(submatrix[col][j], pivot_inv)
                inverse[col][j] = _gf_mul(inverse[col][j], pivot_inv)
            
            # Eliminate this column in all other rows (above and below)
            for row in range(k):
                if row != col and submatrix[row][col] != 0:
                    factor = submatrix[row][col]
                    for j in range(k):
                        submatrix[row][j] ^= _gf_mul(factor, submatrix[col][j])
                        inverse[row][j] ^= _gf_mul(factor, inverse[col][j])
        
        # Now 'inverse' contains submatrix^(-1)
        # Recover original data: D = inverse × shard_data
        # D[i] = sum_j (inverse[i][j] × shard[j].data[byte])
        recovered_chunks = []
        for data_idx in range(k):
            chunk = bytearray(shard_size)
            for byte_idx in range(shard_size):
                value = 0
                for shard_idx, shard in enumerate(shards):
                    value ^= _gf_mul(inverse[data_idx][shard_idx], shard.data[byte_idx])
                chunk[byte_idx] = value
            recovered_chunks.append(bytes(chunk))
        
        return b"".join(recovered_chunks)
    
    def verify_integrity(self, shard_set: ShardSet) -> bool:
        """Verify integrity of all shards in a set.
        
        Checks individual shard checksums and overall recoverability.
        
        Args:
            shard_set: ShardSet to verify
        
        Returns:
            True if all shards are valid and data is recoverable
        """
        valid_count = sum(1 for s in shard_set.available_shards if s.is_valid)
        return valid_count >= self.data_shards
    
    def repair(self, shard_set: ShardSet) -> ShardSet:
        """Repair a shard set by regenerating missing/corrupted shards.
        
        First recovers the original data, then re-encodes to get
        a complete set of shards.
        
        Args:
            shard_set: ShardSet with some missing or corrupted shards
        
        Returns:
            New ShardSet with all shards regenerated
        
        Raises:
            InsufficientShardsError: If not enough shards to recover
        """
        # First, recover the original data
        result = self.decode(shard_set)
        if not result.success or result.data is None:
            raise InsufficientShardsError(
                result.error_message or "Cannot repair: insufficient valid shards"
            )
        
        # Re-encode to get complete shard set
        new_set = self.encode(result.data, str(shard_set.belief_id) if shard_set.belief_id else None)
        
        # Preserve original metadata
        new_set.set_id = shard_set.set_id
        new_set.created_at = shard_set.created_at
        new_set.belief_id = shard_set.belief_id
        
        return new_set
    
    def get_stats(self) -> dict:
        """Get codec statistics."""
        return {
            "data_shards": self.data_shards,
            "total_shards": self.total_shards,
            "parity_shards": self._parity_shards,
            "max_failures": self._parity_shards,
            "overhead_percent": ((self._parity_shards) / self.data_shards) * 100,
            "level": self.level.name if self.level else "custom",
        }
