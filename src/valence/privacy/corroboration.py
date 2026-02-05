"""Corroboration-based auto-elevation for Valence privacy.

Implements Issue #96: Auto-elevate beliefs when multiple independent sources confirm.

When beliefs from different sources semantically match (via embedding distance),
the system tracks corroboration and can auto-propose elevation once a threshold
is met. Owners can opt-out of auto-elevation per belief or globally.
"""

from __future__ import annotations

import logging
import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .types import ShareLevel
from .elevation import (
    ElevationProposal,
    ElevationService,
    propose_elevation,
)

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_CORROBORATION_THRESHOLD = 3  # Sources needed for auto-elevation
DEFAULT_SIMILARITY_THRESHOLD = 0.85  # Minimum semantic similarity (0-1)
DEFAULT_TARGET_LEVEL = ShareLevel.BOUNDED  # Default elevation target


class CorroborationError(Exception):
    """Base exception for corroboration operations."""
    pass


class BeliefNotFoundError(CorroborationError):
    """Raised when a belief is not found."""
    pass


class DuplicateSourceError(CorroborationError):
    """Raised when trying to add corroboration from same source twice."""
    pass


class AutoElevationDisabledError(CorroborationError):
    """Raised when auto-elevation is disabled for a belief/owner."""
    pass


@dataclass
class CorroboratingSource:
    """A source that corroborates a belief.
    
    Attributes:
        source_did: DID of the corroborating source
        belief_id: ID of the corroborating belief from that source
        similarity: Semantic similarity score (0.0 - 1.0)
        corroborated_at: When corroboration was detected
        content_hash: Hash of the corroborating content (for verification)
        metadata: Additional source metadata
    """
    
    source_did: str
    belief_id: str
    similarity: float
    corroborated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_did": self.source_did,
            "belief_id": self.belief_id,
            "similarity": self.similarity,
            "corroborated_at": self.corroborated_at.isoformat(),
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorroboratingSource":
        """Deserialize from dictionary."""
        return cls(
            source_did=data["source_did"],
            belief_id=data["belief_id"],
            similarity=data["similarity"],
            corroborated_at=datetime.fromisoformat(data["corroborated_at"]),
            content_hash=data.get("content_hash"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CorroborationEvidence:
    """Evidence of corroboration for a belief.
    
    Tracks all independent sources that have confirmed similar beliefs
    and metadata about the corroboration process.
    
    Attributes:
        belief_id: ID of the belief being corroborated
        owner_did: DID of the belief owner
        sources: List of corroborating sources
        threshold_met: Whether auto-elevation threshold is met
        threshold_met_at: When threshold was first met
        auto_elevation_proposed: Whether auto-elevation has been proposed
        proposal_id: ID of the elevation proposal if created
        auto_elevation_enabled: Whether auto-elevation is enabled for this belief
    """
    
    belief_id: str
    owner_did: str
    sources: List[CorroboratingSource] = field(default_factory=list)
    threshold_met: bool = False
    threshold_met_at: Optional[datetime] = None
    auto_elevation_proposed: bool = False
    proposal_id: Optional[str] = None
    auto_elevation_enabled: bool = True
    
    @property
    def source_count(self) -> int:
        """Number of independent corroborating sources."""
        return len(self.sources)
    
    @property
    def source_dids(self) -> Set[str]:
        """Set of all source DIDs."""
        return {s.source_did for s in self.sources}
    
    @property
    def average_similarity(self) -> float:
        """Average similarity across all corroborating sources."""
        if not self.sources:
            return 0.0
        return sum(s.similarity for s in self.sources) / len(self.sources)
    
    @property
    def min_similarity(self) -> float:
        """Minimum similarity among corroborating sources."""
        if not self.sources:
            return 0.0
        return min(s.similarity for s in self.sources)
    
    @property
    def max_similarity(self) -> float:
        """Maximum similarity among corroborating sources."""
        if not self.sources:
            return 0.0
        return max(s.similarity for s in self.sources)
    
    def add_source(self, source: CorroboratingSource) -> bool:
        """Add a corroborating source.
        
        Returns:
            True if source was added, False if source already exists
        """
        if source.source_did in self.source_dids:
            return False
        
        self.sources.append(source)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "belief_id": self.belief_id,
            "owner_did": self.owner_did,
            "sources": [s.to_dict() for s in self.sources],
            "source_count": self.source_count,
            "average_similarity": self.average_similarity,
            "threshold_met": self.threshold_met,
            "threshold_met_at": self.threshold_met_at.isoformat() if self.threshold_met_at else None,
            "auto_elevation_proposed": self.auto_elevation_proposed,
            "proposal_id": self.proposal_id,
            "auto_elevation_enabled": self.auto_elevation_enabled,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorroborationEvidence":
        """Deserialize from dictionary."""
        return cls(
            belief_id=data["belief_id"],
            owner_did=data["owner_did"],
            sources=[CorroboratingSource.from_dict(s) for s in data.get("sources", [])],
            threshold_met=data.get("threshold_met", False),
            threshold_met_at=datetime.fromisoformat(data["threshold_met_at"]) if data.get("threshold_met_at") else None,
            auto_elevation_proposed=data.get("auto_elevation_proposed", False),
            proposal_id=data.get("proposal_id"),
            auto_elevation_enabled=data.get("auto_elevation_enabled", True),
        )


@dataclass
class SimilarBelief:
    """Result of a similarity search.
    
    Attributes:
        belief_id: ID of the similar belief
        owner_did: DID of the belief owner  
        content: Belief content (may be truncated)
        similarity: Semantic similarity score (0.0 - 1.0)
        current_level: Current privacy level
    """
    
    belief_id: str
    owner_did: str
    content: str
    similarity: float
    current_level: ShareLevel
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "belief_id": self.belief_id,
            "owner_did": self.owner_did,
            "content": self.content,
            "similarity": self.similarity,
            "current_level": self.current_level.value,
        }


# Type alias for similarity function
SimilarityFunc = Callable[[str, str], float]
EmbeddingFunc = Callable[[str], List[float]]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Cosine similarity (0.0 - 1.0, where 1.0 is identical)
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


class CorroborationDetector:
    """Detects and tracks belief corroboration from independent sources.
    
    Uses semantic similarity (embedding distance) to detect when beliefs
    from different sources convey similar information, and tracks
    corroboration evidence. When enough independent sources confirm,
    can auto-propose elevation of the belief.
    
    Configuration:
        - corroboration_threshold: Number of sources needed for auto-elevation (default: 3)
        - similarity_threshold: Minimum similarity for corroboration (default: 0.85)
        - target_level: Target privacy level for elevation (default: BOUNDED)
    
    Example:
        >>> detector = CorroborationDetector(
        ...     corroboration_threshold=3,
        ...     similarity_threshold=0.85,
        ... )
        >>> 
        >>> # Add corroboration from sources
        >>> result = detector.add_corroboration(
        ...     belief_id="belief-123",
        ...     owner_did="did:example:alice",
        ...     source_did="did:example:bob",
        ...     source_belief_id="bob-belief-456",
        ...     similarity=0.92,
        ... )
        >>> 
        >>> # Check if threshold met
        >>> if result.threshold_met and not result.auto_elevation_proposed:
        ...     proposal = detector.propose_auto_elevation(result.belief_id)
    """
    
    def __init__(
        self,
        corroboration_threshold: int = DEFAULT_CORROBORATION_THRESHOLD,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        target_level: ShareLevel = DEFAULT_TARGET_LEVEL,
        elevation_service: Optional[ElevationService] = None,
        embedding_func: Optional[EmbeddingFunc] = None,
    ):
        """Initialize the corroboration detector.
        
        Args:
            corroboration_threshold: Number of independent sources needed
                for auto-elevation (default: 3)
            similarity_threshold: Minimum semantic similarity (0.0-1.0)
                for considering beliefs corroborating (default: 0.85)
            target_level: Target privacy level for auto-elevation proposals
                (default: BOUNDED)
            elevation_service: Service for managing elevation proposals
                (creates new instance if not provided)
            embedding_func: Function to generate embeddings for content
                (optional, for similarity computation)
        """
        self.corroboration_threshold = corroboration_threshold
        self.similarity_threshold = similarity_threshold
        self.target_level = target_level
        self.elevation_service = elevation_service or ElevationService()
        self.embedding_func = embedding_func
        
        # In-memory storage (would be database-backed in production)
        self._evidence: Dict[str, CorroborationEvidence] = {}
        self._owner_opt_out: Set[str] = set()  # DIDs that opted out globally
        self._belief_opt_out: Set[str] = set()  # Belief IDs that opted out
        
        # Cache embeddings for similarity computation
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def add_corroboration(
        self,
        belief_id: str,
        owner_did: str,
        source_did: str,
        source_belief_id: str,
        similarity: float,
        current_level: ShareLevel = ShareLevel.PRIVATE,
        content_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CorroborationEvidence:
        """Add a corroborating source to a belief.
        
        Args:
            belief_id: ID of the belief being corroborated
            owner_did: DID of the belief owner
            source_did: DID of the corroborating source
            source_belief_id: ID of the belief from the source
            similarity: Semantic similarity score (0.0-1.0)
            current_level: Current privacy level of the belief
            content_hash: Hash of corroborating content (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Updated CorroborationEvidence
            
        Raises:
            DuplicateSourceError: If source already corroborated this belief
            ValueError: If similarity is below threshold
        """
        # Validate similarity threshold
        if similarity < self.similarity_threshold:
            raise ValueError(
                f"Similarity {similarity:.3f} below threshold {self.similarity_threshold}"
            )
        
        # Get or create evidence record
        evidence = self._evidence.get(belief_id)
        if evidence is None:
            evidence = CorroborationEvidence(
                belief_id=belief_id,
                owner_did=owner_did,
            )
            self._evidence[belief_id] = evidence
        
        # Check for duplicate source
        if source_did in evidence.source_dids:
            raise DuplicateSourceError(
                f"Source {source_did} already corroborated belief {belief_id}"
            )
        
        # Check opt-out status
        evidence.auto_elevation_enabled = self._is_auto_elevation_enabled(
            belief_id, owner_did
        )
        
        # Create and add source
        source = CorroboratingSource(
            source_did=source_did,
            belief_id=source_belief_id,
            similarity=similarity,
            content_hash=content_hash,
            metadata=metadata or {},
        )
        evidence.add_source(source)
        
        # Check if threshold now met
        if (
            not evidence.threshold_met
            and evidence.source_count >= self.corroboration_threshold
        ):
            evidence.threshold_met = True
            evidence.threshold_met_at = datetime.now(timezone.utc)
            logger.info(
                f"Corroboration threshold met for belief {belief_id} "
                f"({evidence.source_count} sources)"
            )
        
        return evidence
    
    def check_similarity(
        self,
        content1: str,
        content2: str,
    ) -> float:
        """Check semantic similarity between two belief contents.
        
        Uses embedding-based similarity if embedding_func is configured,
        otherwise returns 0.0.
        
        Args:
            content1: First belief content
            content2: Second belief content
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if self.embedding_func is None:
            logger.warning("No embedding function configured for similarity check")
            return 0.0
        
        # Get or compute embeddings
        emb1 = self._get_embedding(content1)
        emb2 = self._get_embedding(content2)
        
        return cosine_similarity(emb1, emb2)
    
    def find_similar_beliefs(
        self,
        content: str,
        source_did: str,
        beliefs: List[Dict[str, Any]],
        min_similarity: Optional[float] = None,
    ) -> List[SimilarBelief]:
        """Find beliefs similar to the given content.
        
        Args:
            content: Content to search for
            source_did: DID of the source (excluded from results)
            beliefs: List of belief records to search
            min_similarity: Minimum similarity threshold (defaults to configured)
            
        Returns:
            List of similar beliefs, sorted by similarity (highest first)
        """
        if self.embedding_func is None:
            return []
        
        threshold = min_similarity if min_similarity is not None else self.similarity_threshold
        source_embedding = self._get_embedding(content)
        
        results = []
        for belief in beliefs:
            # Skip beliefs from the same source
            if belief.get("owner_did") == source_did:
                continue
            
            belief_content = belief.get("content", "")
            belief_embedding = self._get_embedding(belief_content)
            
            similarity = cosine_similarity(source_embedding, belief_embedding)
            
            if similarity >= threshold:
                level_str = belief.get("share_level", "private")
                try:
                    level = ShareLevel(level_str)
                except ValueError:
                    level = ShareLevel.PRIVATE
                
                results.append(SimilarBelief(
                    belief_id=belief["id"],
                    owner_did=belief.get("owner_did", ""),
                    content=belief_content[:200],  # Truncate for privacy
                    similarity=similarity,
                    current_level=level,
                ))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results
    
    def propose_auto_elevation(
        self,
        belief_id: str,
        to_level: Optional[ShareLevel] = None,
        reason: Optional[str] = None,
    ) -> ElevationProposal:
        """Create an auto-elevation proposal for a corroborated belief.
        
        Args:
            belief_id: ID of the belief to elevate
            to_level: Target privacy level (defaults to configured target_level)
            reason: Reason for elevation (auto-generated if not provided)
            
        Returns:
            The created ElevationProposal
            
        Raises:
            BeliefNotFoundError: If belief has no corroboration evidence
            AutoElevationDisabledError: If auto-elevation is disabled
            ValueError: If threshold not met
        """
        evidence = self._evidence.get(belief_id)
        if evidence is None:
            raise BeliefNotFoundError(f"No corroboration evidence for belief {belief_id}")
        
        # Check opt-out
        if not evidence.auto_elevation_enabled:
            raise AutoElevationDisabledError(
                f"Auto-elevation disabled for belief {belief_id}"
            )
        
        if not self._is_auto_elevation_enabled(belief_id, evidence.owner_did):
            raise AutoElevationDisabledError(
                f"Auto-elevation disabled by owner {evidence.owner_did}"
            )
        
        # Check threshold
        if not evidence.threshold_met:
            raise ValueError(
                f"Corroboration threshold not met: {evidence.source_count} < "
                f"{self.corroboration_threshold}"
            )
        
        # Already proposed?
        if evidence.auto_elevation_proposed and evidence.proposal_id:
            # Return existing proposal
            existing = self.elevation_service.get(evidence.proposal_id)
            if existing:
                return existing
        
        # Generate reason
        target = to_level or self.target_level
        if reason is None:
            reason = (
                f"Auto-elevation proposed: {evidence.source_count} independent sources "
                f"corroborated this belief (avg similarity: {evidence.average_similarity:.2f}). "
                f"Sources: {', '.join(s.source_did for s in evidence.sources[:5])}"
            )
        
        # Create proposal
        proposal = self.elevation_service.propose(
            proposer="system:corroboration-detector",
            belief_id=belief_id,
            from_level=ShareLevel.PRIVATE,
            to_level=target,
            reason=reason,
            metadata={
                "auto_elevation": True,
                "source_count": evidence.source_count,
                "average_similarity": evidence.average_similarity,
                "sources": [s.source_did for s in evidence.sources],
                "threshold_met_at": evidence.threshold_met_at.isoformat() if evidence.threshold_met_at else None,
            },
        )
        
        # Update evidence
        evidence.auto_elevation_proposed = True
        evidence.proposal_id = proposal.proposal_id
        
        logger.info(
            f"Auto-elevation proposed for belief {belief_id}: {proposal.proposal_id}"
        )
        
        return proposal
    
    def get_evidence(self, belief_id: str) -> Optional[CorroborationEvidence]:
        """Get corroboration evidence for a belief.
        
        Args:
            belief_id: ID of the belief
            
        Returns:
            CorroborationEvidence if found, None otherwise
        """
        return self._evidence.get(belief_id)
    
    def get_pending_elevations(self) -> List[CorroborationEvidence]:
        """Get all beliefs that have met threshold but not been elevated.
        
        Returns:
            List of evidence records awaiting elevation decision
        """
        return [
            e for e in self._evidence.values()
            if e.threshold_met and e.auto_elevation_enabled
        ]
    
    def get_threshold_met(self) -> List[CorroborationEvidence]:
        """Get all beliefs that have met the corroboration threshold.
        
        Returns:
            List of evidence records that met threshold
        """
        return [e for e in self._evidence.values() if e.threshold_met]
    
    def opt_out_belief(self, belief_id: str) -> None:
        """Disable auto-elevation for a specific belief.
        
        Args:
            belief_id: ID of the belief
        """
        self._belief_opt_out.add(belief_id)
        
        # Update evidence if exists
        if belief_id in self._evidence:
            self._evidence[belief_id].auto_elevation_enabled = False
        
        logger.info(f"Auto-elevation disabled for belief {belief_id}")
    
    def opt_in_belief(self, belief_id: str) -> None:
        """Re-enable auto-elevation for a specific belief.
        
        Args:
            belief_id: ID of the belief
        """
        self._belief_opt_out.discard(belief_id)
        
        # Update evidence if exists
        evidence = self._evidence.get(belief_id)
        if evidence:
            # Check owner hasn't globally opted out
            if evidence.owner_did not in self._owner_opt_out:
                evidence.auto_elevation_enabled = True
        
        logger.info(f"Auto-elevation re-enabled for belief {belief_id}")
    
    def opt_out_owner(self, owner_did: str) -> None:
        """Disable auto-elevation globally for an owner.
        
        Args:
            owner_did: DID of the owner
        """
        self._owner_opt_out.add(owner_did)
        
        # Update all evidence for this owner
        for evidence in self._evidence.values():
            if evidence.owner_did == owner_did:
                evidence.auto_elevation_enabled = False
        
        logger.info(f"Auto-elevation disabled globally for owner {owner_did}")
    
    def opt_in_owner(self, owner_did: str) -> None:
        """Re-enable auto-elevation globally for an owner.
        
        Args:
            owner_did: DID of the owner
        """
        self._owner_opt_out.discard(owner_did)
        
        # Update all evidence for this owner (unless belief-level opt-out)
        for evidence in self._evidence.values():
            if evidence.owner_did == owner_did:
                if evidence.belief_id not in self._belief_opt_out:
                    evidence.auto_elevation_enabled = True
        
        logger.info(f"Auto-elevation re-enabled globally for owner {owner_did}")
    
    def is_owner_opted_out(self, owner_did: str) -> bool:
        """Check if owner has globally opted out of auto-elevation.
        
        Args:
            owner_did: DID of the owner
            
        Returns:
            True if opted out
        """
        return owner_did in self._owner_opt_out
    
    def is_belief_opted_out(self, belief_id: str) -> bool:
        """Check if belief has opted out of auto-elevation.
        
        Args:
            belief_id: ID of the belief
            
        Returns:
            True if opted out
        """
        return belief_id in self._belief_opt_out
    
    def process_incoming_belief(
        self,
        source_did: str,
        source_belief_id: str,
        content: str,
        local_beliefs: List[Dict[str, Any]],
        content_hash: Optional[str] = None,
    ) -> List[CorroborationEvidence]:
        """Process an incoming belief and check for corroboration.
        
        This is the main entry point for federation sync to call when
        receiving beliefs from peers. Checks if the incoming belief
        corroborates any local beliefs.
        
        Args:
            source_did: DID of the source node
            source_belief_id: ID of the belief from source
            content: Belief content
            local_beliefs: List of local belief records to check against
            content_hash: Hash of the content (optional)
            
        Returns:
            List of evidence records that were updated
        """
        updated = []
        
        # Find similar local beliefs
        similar = self.find_similar_beliefs(
            content=content,
            source_did=source_did,
            beliefs=local_beliefs,
        )
        
        for match in similar:
            try:
                evidence = self.add_corroboration(
                    belief_id=match.belief_id,
                    owner_did=match.owner_did,
                    source_did=source_did,
                    source_belief_id=source_belief_id,
                    similarity=match.similarity,
                    current_level=match.current_level,
                    content_hash=content_hash,
                )
                updated.append(evidence)
                
                # Auto-propose elevation if threshold just met
                if (
                    evidence.threshold_met
                    and evidence.auto_elevation_enabled
                    and not evidence.auto_elevation_proposed
                ):
                    try:
                        self.propose_auto_elevation(match.belief_id)
                    except (AutoElevationDisabledError, ValueError) as e:
                        logger.debug(f"Skipping auto-elevation: {e}")
                        
            except DuplicateSourceError:
                logger.debug(
                    f"Source {source_did} already corroborated {match.belief_id}"
                )
            except ValueError as e:
                logger.debug(f"Corroboration rejected: {e}")
        
        return updated
    
    def _is_auto_elevation_enabled(self, belief_id: str, owner_did: str) -> bool:
        """Check if auto-elevation is enabled for belief/owner."""
        if belief_id in self._belief_opt_out:
            return False
        if owner_did in self._owner_opt_out:
            return False
        return True
    
    def _get_embedding(self, content: str) -> List[float]:
        """Get or compute embedding for content."""
        if content in self._embedding_cache:
            return self._embedding_cache[content]
        
        if self.embedding_func is None:
            raise ValueError("No embedding function configured")
        
        embedding = self.embedding_func(content)
        self._embedding_cache[content] = embedding
        return embedding
    
    def clear_embedding_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()


# Module-level singleton for convenience
_default_detector: Optional[CorroborationDetector] = None
_default_detector_lock = threading.Lock()


def get_corroboration_detector() -> CorroborationDetector:
    """Get or create the default corroboration detector.
    
    Thread-safe initialization using double-checked locking pattern.
    """
    global _default_detector
    if _default_detector is None:
        with _default_detector_lock:
            # Double-check after acquiring lock
            if _default_detector is None:
                _default_detector = CorroborationDetector()
    return _default_detector


def set_corroboration_detector(detector: Optional[CorroborationDetector]) -> None:
    """Set the default corroboration detector.
    
    Thread-safe setter using lock.
    """
    global _default_detector
    with _default_detector_lock:
        _default_detector = detector


def add_corroboration(
    belief_id: str,
    owner_did: str,
    source_did: str,
    source_belief_id: str,
    similarity: float,
    **kwargs: Any,
) -> CorroborationEvidence:
    """Add corroboration using the default detector.
    
    See CorroborationDetector.add_corroboration for full documentation.
    """
    return get_corroboration_detector().add_corroboration(
        belief_id=belief_id,
        owner_did=owner_did,
        source_did=source_did,
        source_belief_id=source_belief_id,
        similarity=similarity,
        **kwargs,
    )


def get_evidence(belief_id: str) -> Optional[CorroborationEvidence]:
    """Get corroboration evidence using the default detector."""
    return get_corroboration_detector().get_evidence(belief_id)


def propose_auto_elevation(
    belief_id: str,
    to_level: Optional[ShareLevel] = None,
    reason: Optional[str] = None,
) -> ElevationProposal:
    """Propose auto-elevation using the default detector.
    
    See CorroborationDetector.propose_auto_elevation for full documentation.
    """
    return get_corroboration_detector().propose_auto_elevation(
        belief_id=belief_id,
        to_level=to_level,
        reason=reason,
    )


def opt_out_belief(belief_id: str) -> None:
    """Disable auto-elevation for a belief using the default detector."""
    get_corroboration_detector().opt_out_belief(belief_id)


def opt_out_owner(owner_did: str) -> None:
    """Disable auto-elevation for an owner using the default detector."""
    get_corroboration_detector().opt_out_owner(owner_did)
