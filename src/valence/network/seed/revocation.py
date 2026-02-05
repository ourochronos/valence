"""
Seed revocation management (Issue #121).

Handles tracking and verification of seed node revocations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

if TYPE_CHECKING:
    from .config import SeedConfig
    from valence.network.messages import SeedRevocationList

logger = logging.getLogger(__name__)


@dataclass
class SeedRevocationRecord:
    """Record of a seed revocation for storage."""
    seed_id: str
    revocation_id: str
    reason: str
    reason_detail: str
    timestamp: float
    effective_at: float
    issuer_id: str
    signature: str
    received_at: float = field(default_factory=time.time)
    source: str = "direct"  # "direct", "gossip", or "file"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "seed_id": self.seed_id,
            "revocation_id": self.revocation_id,
            "reason": self.reason,
            "reason_detail": self.reason_detail,
            "timestamp": self.timestamp,
            "effective_at": self.effective_at,
            "issuer_id": self.issuer_id,
            "signature": self.signature,
            "received_at": self.received_at,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedRevocationRecord":
        """Deserialize from dict."""
        return cls(
            seed_id=data["seed_id"],
            revocation_id=data["revocation_id"],
            reason=data.get("reason", ""),
            reason_detail=data.get("reason_detail", ""),
            timestamp=data.get("timestamp", 0),
            effective_at=data.get("effective_at", 0),
            issuer_id=data.get("issuer_id", ""),
            signature=data.get("signature", ""),
            received_at=data.get("received_at", time.time()),
            source=data.get("source", "direct"),
        )
    
    @property
    def is_effective(self) -> bool:
        """Check if the revocation is currently effective."""
        return time.time() >= self.effective_at


class SeedRevocationManager:
    """
    Manages seed revocations for a seed node (Issue #121).
    
    Implements:
    - Revocation storage and lookup
    - Signature verification for revocations
    - Out-of-band revocation list loading from file
    - Integration with gossip for revocation propagation
    
    Revocations can be:
    - Self-signed (seed revokes itself)
    - Authority-signed (trusted authority revokes seed)
    
    Nodes query the seed for revocations and honor them by
    excluding revoked seeds from discovery.
    """
    
    def __init__(self, config: "SeedConfig"):
        self.config = config
        self._revocations: Dict[str, SeedRevocationRecord] = {}  # seed_id -> record
        self._revocation_list_mtime: float = 0  # Last modification time of file
        self._check_task: Optional[asyncio.Task] = None
        self._running = False
    
    def is_seed_revoked(self, seed_id: str) -> bool:
        """
        Check if a seed is revoked.
        
        Args:
            seed_id: The seed ID to check
            
        Returns:
            True if the seed is revoked and the revocation is effective
        """
        if not self.config.seed_revocation_enabled:
            return False
        
        record = self._revocations.get(seed_id)
        if record is None:
            return False
        
        return record.is_effective
    
    def get_revocation(self, seed_id: str) -> Optional[SeedRevocationRecord]:
        """Get revocation record for a seed."""
        return self._revocations.get(seed_id)
    
    def get_all_revocations(self) -> List[SeedRevocationRecord]:
        """Get all revocation records."""
        return list(self._revocations.values())
    
    def get_revoked_seed_ids(self) -> set:
        """Get set of all currently revoked seed IDs."""
        now = time.time()
        return {
            seed_id for seed_id, record in self._revocations.items()
            if record.effective_at <= now
        }
    
    def add_revocation(
        self,
        revocation_data: Dict[str, Any],
        source: str = "direct",
        verify: bool = True,
    ) -> tuple[bool, Optional[str]]:
        """
        Add a seed revocation.
        
        Args:
            revocation_data: Revocation dict (from SeedRevocation.to_dict())
            source: Source of revocation ("direct", "gossip", "file")
            verify: Whether to verify the signature
            
        Returns:
            Tuple of (success: bool, error_reason: Optional[str])
        """
        if not self.config.seed_revocation_enabled:
            return False, "revocation_disabled"
        
        seed_id = revocation_data.get("seed_id")
        if not seed_id:
            return False, "missing_seed_id"
        
        revocation_id = revocation_data.get("revocation_id")
        if not revocation_id:
            return False, "missing_revocation_id"
        
        timestamp = revocation_data.get("timestamp", 0)
        issuer_id = revocation_data.get("issuer_id", "")
        signature = revocation_data.get("signature", "")
        
        # Check revocation age
        now = time.time()
        age = now - timestamp
        if age > self.config.seed_revocation_max_age_seconds:
            return False, f"revocation_too_old:age={int(age)}s"
        
        # Verify signature if enabled
        if verify and self.config.seed_revocation_verify_signatures:
            is_valid, error = self._verify_revocation_signature(revocation_data)
            if not is_valid:
                return False, f"invalid_signature:{error}"
        
        # Check if we already have this revocation
        existing = self._revocations.get(seed_id)
        if existing and existing.revocation_id == revocation_id:
            # Already have this exact revocation
            return True, None
        
        # If we have an older revocation for this seed, keep the newer one
        if existing and existing.timestamp > timestamp:
            # Existing is newer, ignore this one
            logger.debug(
                f"Ignoring older revocation for {seed_id[:20]}... "
                f"(existing={existing.timestamp}, new={timestamp})"
            )
            return True, None
        
        # Create and store the revocation record
        record = SeedRevocationRecord(
            seed_id=seed_id,
            revocation_id=revocation_id,
            reason=revocation_data.get("reason", ""),
            reason_detail=revocation_data.get("reason_detail", ""),
            timestamp=timestamp,
            effective_at=revocation_data.get("effective_at", timestamp),
            issuer_id=issuer_id,
            signature=signature,
            received_at=now,
            source=source,
        )
        
        self._revocations[seed_id] = record
        
        logger.info(
            f"Added seed revocation: {seed_id[:20]}... "
            f"reason={record.reason}, issuer={issuer_id[:20]}..., source={source}"
        )
        
        return True, None
    
    def _verify_revocation_signature(
        self,
        revocation_data: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Verify the signature on a revocation.
        
        A revocation is valid if:
        1. It's self-signed (issuer_id == seed_id) and signature is valid
        2. It's signed by a trusted authority in config.seed_revocation_trusted_authorities
        
        Args:
            revocation_data: Revocation dict
            
        Returns:
            Tuple of (is_valid: bool, error: Optional[str])
        """
        seed_id = revocation_data.get("seed_id", "")
        issuer_id = revocation_data.get("issuer_id", "")
        signature_hex = revocation_data.get("signature", "")
        
        if not signature_hex:
            return False, "missing_signature"
        
        if not issuer_id:
            return False, "missing_issuer_id"
        
        # Determine if this is self-signed or authority-signed
        is_self_signed = issuer_id == seed_id
        is_authority = issuer_id in self.config.seed_revocation_trusted_authorities
        
        if not is_self_signed and not is_authority:
            return False, "untrusted_issuer"
        
        # Build the signed data (same as SeedRevocation.get_signable_data())
        signed_data = {
            "type": revocation_data.get("type", "seed_revocation"),
            "revocation_id": revocation_data.get("revocation_id"),
            "seed_id": seed_id,
            "reason": revocation_data.get("reason", ""),
            "reason_detail": revocation_data.get("reason_detail", ""),
            "timestamp": revocation_data.get("timestamp"),
            "effective_at": revocation_data.get("effective_at"),
            "issuer_id": issuer_id,
        }
        message = json.dumps(signed_data, sort_keys=True, separators=(',', ':')).encode()
        
        try:
            # Parse public key from issuer_id (hex-encoded Ed25519 public key)
            public_key_bytes = bytes.fromhex(issuer_id)
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            
            # Parse and verify signature
            signature = bytes.fromhex(signature_hex)
            public_key.verify(signature, message)
            
            return True, None
            
        except InvalidSignature:
            return False, "signature_verification_failed"
        except (ValueError, TypeError) as e:
            return False, f"invalid_key_or_signature_format:{e}"
        except Exception as e:
            return False, f"verification_error:{e}"
    
    def load_revocation_list_from_file(self, file_path: str) -> tuple[int, List[str]]:
        """
        Load revocations from an out-of-band file.
        
        The file should contain a JSON SeedRevocationList.
        
        Args:
            file_path: Path to the revocation list file
            
        Returns:
            Tuple of (loaded_count: int, errors: List[str])
        """
        errors = []
        loaded_count = 0
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            errors.append(f"file_not_found:{file_path}")
            return 0, errors
        except json.JSONDecodeError as e:
            errors.append(f"invalid_json:{e}")
            return 0, errors
        except Exception as e:
            errors.append(f"read_error:{e}")
            return 0, errors
        
        # Import here to avoid circular imports
        from valence.network.messages import SeedRevocationList
        
        try:
            revocation_list = SeedRevocationList.from_dict(data)
        except Exception as e:
            errors.append(f"parse_error:{e}")
            return 0, errors
        
        # Verify list signature if we have trusted authorities
        if self.config.seed_revocation_verify_signatures:
            if revocation_list.authority_id:
                is_valid, error = self._verify_list_signature(revocation_list)
                if not is_valid:
                    errors.append(f"list_signature_invalid:{error}")
                    return 0, errors
        
        # Process each revocation
        for revocation in revocation_list.revocations:
            success, error = self.add_revocation(
                revocation.to_dict(),
                source="file",
                verify=self.config.seed_revocation_verify_signatures,
            )
            if success:
                loaded_count += 1
            else:
                errors.append(f"revocation_error:{revocation.seed_id}:{error}")
        
        logger.info(
            f"Loaded {loaded_count} revocations from {file_path} "
            f"(version={revocation_list.version}, errors={len(errors)})"
        )
        
        return loaded_count, errors
    
    def _verify_list_signature(
        self,
        revocation_list: "SeedRevocationList",
    ) -> tuple[bool, Optional[str]]:
        """Verify the signature on a revocation list."""
        authority_id = revocation_list.authority_id
        signature_hex = revocation_list.signature
        
        if not authority_id:
            return False, "missing_authority_id"
        
        if authority_id not in self.config.seed_revocation_trusted_authorities:
            return False, "untrusted_authority"
        
        if not signature_hex:
            return False, "missing_signature"
        
        message = revocation_list.get_signable_bytes()
        
        try:
            public_key_bytes = bytes.fromhex(authority_id)
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            signature = bytes.fromhex(signature_hex)
            public_key.verify(signature, message)
            return True, None
        except InvalidSignature:
            return False, "signature_verification_failed"
        except Exception as e:
            return False, f"verification_error:{e}"
    
    async def start(self) -> None:
        """Start the revocation manager (file watching loop)."""
        if self._running:
            return
        
        self._running = True
        
        # Load initial revocations from file if configured
        if self.config.seed_revocation_list_path:
            try:
                loaded, errors = self.load_revocation_list_from_file(
                    self.config.seed_revocation_list_path
                )
                if errors:
                    logger.warning(f"Revocation list load errors: {errors}")
            except Exception as e:
                logger.error(f"Failed to load revocation list: {e}")
        
        # Start file check loop if path is configured
        if self.config.seed_revocation_list_path:
            self._check_task = asyncio.create_task(self._file_check_loop())
        
        logger.info("Seed revocation manager started")
    
    async def stop(self) -> None:
        """Stop the revocation manager."""
        self._running = False
        
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
        
        logger.info("Seed revocation manager stopped")
    
    async def _file_check_loop(self) -> None:
        """Periodically check for revocation list file updates."""
        while self._running:
            await asyncio.sleep(self.config.seed_revocation_list_check_interval)
            
            if not self.config.seed_revocation_list_path:
                continue
            
            try:
                mtime = os.path.getmtime(self.config.seed_revocation_list_path)
                if mtime > self._revocation_list_mtime:
                    logger.info("Revocation list file updated, reloading...")
                    self.load_revocation_list_from_file(
                        self.config.seed_revocation_list_path
                    )
                    self._revocation_list_mtime = mtime
            except FileNotFoundError:
                pass  # File doesn't exist yet
            except Exception as e:
                logger.error(f"Error checking revocation list file: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get revocation manager statistics."""
        now = time.time()
        effective_count = sum(
            1 for r in self._revocations.values()
            if r.effective_at <= now
        )
        
        return {
            "enabled": self.config.seed_revocation_enabled,
            "total_revocations": len(self._revocations),
            "effective_revocations": effective_count,
            "revocation_list_path": self.config.seed_revocation_list_path,
            "revoked_seeds": list(self.get_revoked_seed_ids()),
        }
    
    def get_revocations_for_gossip(self) -> List[Dict[str, Any]]:
        """
        Get revocations to share in gossip exchange.
        
        Returns recent revocations that should be propagated.
        """
        if not self.config.seed_revocation_gossip_enabled:
            return []
        
        now = time.time()
        max_age = self.config.seed_revocation_max_age_seconds
        
        revocations = []
        for record in self._revocations.values():
            # Only share revocations within age limit
            if now - record.timestamp > max_age:
                continue
            
            revocations.append(record.to_dict())
        
        return revocations
    
    def process_gossip_revocations(self, revocations: List[Dict[str, Any]]) -> int:
        """
        Process revocations received via gossip.
        
        Args:
            revocations: List of revocation dicts from gossip
            
        Returns:
            Number of revocations added
        """
        added = 0
        
        for rev_data in revocations:
            success, error = self.add_revocation(
                rev_data,
                source="gossip",
                verify=self.config.seed_revocation_verify_signatures,
            )
            if success:
                added += 1
            elif error:
                logger.debug(f"Gossip revocation rejected: {error}")
        
        return added
