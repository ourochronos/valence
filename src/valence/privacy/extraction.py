"""AI-assisted insight extraction for Valence privacy.

Implements Issue #95: Extract shareable insights from private content using AI.
Supports multiple extraction levels with human review before sharing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import hashlib
import uuid


class ExtractionLevel(Enum):
    """Level of detail for insight extraction.
    
    Controls how much of the original content is preserved vs abstracted.
    Higher abstraction levels are safer for wider sharing.
    """
    
    THEMES = "themes"           # High-level themes only (most abstract)
    KEY_POINTS = "key_points"   # Main points without detail
    SUMMARY = "summary"         # Condensed summary preserving key info
    ANONYMIZED = "anonymized"   # Full content with PII/identifiers removed


class ExtractionStatus(Enum):
    """Status of an extraction in the review workflow."""
    
    PENDING_REVIEW = "pending_review"  # Awaiting human approval
    APPROVED = "approved"               # Approved for sharing
    REJECTED = "rejected"               # Rejected, not shareable
    MODIFIED = "modified"               # Approved with human modifications


class ExtractionError(Exception):
    """Base exception for extraction operations."""
    pass


class ExtractorNotAvailableError(ExtractionError):
    """Raised when no AI extractor is configured."""
    pass


class ExtractionNotFoundError(ExtractionError):
    """Raised when an extraction record is not found."""
    pass


class ExtractionAlreadyReviewedError(ExtractionError):
    """Raised when trying to review an already-reviewed extraction."""
    pass


@dataclass
class ExtractionProvenance:
    """Tracks the relationship between extracted content and its source.
    
    Records that the extracted content was "derived from" the source,
    allowing provenance chains to indicate transformation.
    """
    
    source_id: str              # ID of source belief/content
    source_hash: str            # Hash of source content for integrity
    extracted_at: datetime
    extraction_level: ExtractionLevel
    extractor_id: str           # Identifier for the extractor used
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_id": self.source_id,
            "source_hash": self.source_hash,
            "extracted_at": self.extracted_at.isoformat(),
            "extraction_level": self.extraction_level.value,
            "extractor_id": self.extractor_id,
            "extraction_metadata": self.extraction_metadata,
            "relationship": "extracted_from",
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionProvenance":
        """Deserialize from dictionary."""
        return cls(
            source_id=data["source_id"],
            source_hash=data["source_hash"],
            extracted_at=datetime.fromisoformat(data["extracted_at"]),
            extraction_level=ExtractionLevel(data["extraction_level"]),
            extractor_id=data["extractor_id"],
            extraction_metadata=data.get("extraction_metadata", {}),
        )


@dataclass
class ExtractedInsight:
    """Result of an insight extraction operation.
    
    Contains the extracted content along with provenance and review status.
    """
    
    extraction_id: str
    content: str                          # The extracted insight text
    level: ExtractionLevel
    provenance: ExtractionProvenance
    status: ExtractionStatus = ExtractionStatus.PENDING_REVIEW
    
    # Review tracking
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None
    
    # If modified during review
    original_extraction: Optional[str] = None  # Original AI output if modified
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "extraction_id": self.extraction_id,
            "content": self.content,
            "level": self.level.value,
            "provenance": self.provenance.to_dict(),
            "status": self.status.value,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewed_by": self.reviewed_by,
            "review_notes": self.review_notes,
            "original_extraction": self.original_extraction,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedInsight":
        """Deserialize from dictionary."""
        return cls(
            extraction_id=data["extraction_id"],
            content=data["content"],
            level=ExtractionLevel(data["level"]),
            provenance=ExtractionProvenance.from_dict(data["provenance"]),
            status=ExtractionStatus(data["status"]),
            reviewed_at=datetime.fromisoformat(data["reviewed_at"]) if data.get("reviewed_at") else None,
            reviewed_by=data.get("reviewed_by"),
            review_notes=data.get("review_notes"),
            original_extraction=data.get("original_extraction"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )
    
    @property
    def is_pending(self) -> bool:
        """Check if extraction is pending review."""
        return self.status == ExtractionStatus.PENDING_REVIEW
    
    @property
    def is_approved(self) -> bool:
        """Check if extraction is approved for sharing."""
        return self.status in (ExtractionStatus.APPROVED, ExtractionStatus.MODIFIED)
    
    @property
    def is_rejected(self) -> bool:
        """Check if extraction was rejected."""
        return self.status == ExtractionStatus.REJECTED
    
    @property
    def was_modified(self) -> bool:
        """Check if extraction was modified during review."""
        return self.status == ExtractionStatus.MODIFIED


class InsightExtractor(ABC):
    """Abstract base class for AI insight extractors.
    
    Implementations should extract insights from content at various
    abstraction levels. The mock implementation is provided for testing.
    """
    
    @property
    @abstractmethod
    def extractor_id(self) -> str:
        """Unique identifier for this extractor."""
        pass
    
    @abstractmethod
    def extract(
        self,
        content: str,
        level: ExtractionLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Extract insights from content at the specified level.
        
        Args:
            content: The source content to extract from
            level: The desired abstraction level
            context: Optional context for the extraction (domain, intent, etc.)
            
        Returns:
            Extracted insight text
        """
        pass
    
    def supports_level(self, level: ExtractionLevel) -> bool:
        """Check if this extractor supports a given level."""
        return True  # Default: support all levels


class MockInsightExtractor(InsightExtractor):
    """Mock AI extractor for testing.
    
    Simulates AI extraction with deterministic, template-based outputs.
    Useful for testing the extraction workflow without real AI calls.
    """
    
    def __init__(
        self,
        extractor_id: str = "mock-extractor-v1",
        custom_extractors: Optional[Dict[ExtractionLevel, Callable[[str], str]]] = None,
    ):
        """Initialize the mock extractor.
        
        Args:
            extractor_id: Identifier for this extractor instance
            custom_extractors: Optional custom extraction functions per level
        """
        self._extractor_id = extractor_id
        self._custom_extractors = custom_extractors or {}
    
    @property
    def extractor_id(self) -> str:
        return self._extractor_id
    
    def extract(
        self,
        content: str,
        level: ExtractionLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Extract insights using mock/template logic."""
        # Use custom extractor if provided
        if level in self._custom_extractors:
            return self._custom_extractors[level](content)
        
        # Default mock extraction behavior
        word_count = len(content.split())
        content_preview = content[:100] + "..." if len(content) > 100 else content
        
        if level == ExtractionLevel.THEMES:
            # Extract mock themes (very abstract)
            return self._extract_themes(content)
        
        elif level == ExtractionLevel.KEY_POINTS:
            # Extract key points
            return self._extract_key_points(content)
        
        elif level == ExtractionLevel.SUMMARY:
            # Create summary
            return self._extract_summary(content)
        
        elif level == ExtractionLevel.ANONYMIZED:
            # Remove identifying information
            return self._extract_anonymized(content)
        
        return f"[Extracted at {level.value} level from {word_count} word content]"
    
    def _extract_themes(self, content: str) -> str:
        """Mock theme extraction."""
        words = content.lower().split()
        # Simulate finding themes based on word frequency
        word_freq: Dict[str, int] = {}
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                     "to", "of", "and", "in", "that", "it", "for", "on", "with"}
        for word in words:
            word = word.strip(".,!?;:'\"")
            if word and len(word) > 3 and word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top themes
        themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        if themes:
            theme_list = ", ".join(t[0] for t in themes)
            return f"Themes: {theme_list}"
        return "Themes: general content"
    
    def _extract_key_points(self, content: str) -> str:
        """Mock key points extraction."""
        sentences = content.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "Key points: No clear points identified"
        
        # Take first few sentences as "key points"
        key_sentences = sentences[:min(3, len(sentences))]
        points = "\n".join(f"• {s}" for s in key_sentences)
        return f"Key points:\n{points}"
    
    def _extract_summary(self, content: str) -> str:
        """Mock summary extraction."""
        words = content.split()
        if len(words) <= 20:
            return content
        
        # Simple truncation-based summary
        summary_words = words[:max(20, len(words) // 3)]
        return " ".join(summary_words) + "..."
    
    def _extract_anonymized(self, content: str) -> str:
        """Mock anonymization (remove potential PII patterns)."""
        import re
        
        result = content
        
        # Replace email-like patterns
        result = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', result)
        
        # Replace phone-like patterns
        result = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', result)
        
        # Replace potential names (capitalized words not at sentence start)
        # This is intentionally simplistic for the mock
        result = re.sub(r'(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+\b', '[NAME]', result)
        
        # Replace SSN-like patterns
        result = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', result)
        
        return result


def _compute_content_hash(content: str) -> str:
    """Compute a hash of content for provenance integrity."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def extract_insights(
    content: str,
    level: ExtractionLevel,
    source_id: str,
    extractor: Optional[InsightExtractor] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ExtractedInsight:
    """Extract shareable insights from content using AI.
    
    The extracted insight is created in PENDING_REVIEW status and must
    be approved by a human before sharing.
    
    Args:
        content: The source content to extract from
        level: The desired abstraction level
        source_id: ID of the source belief/content for provenance
        extractor: The AI extractor to use (defaults to MockInsightExtractor)
        context: Optional context for the extraction
        
    Returns:
        ExtractedInsight in PENDING_REVIEW status
        
    Raises:
        ExtractorNotAvailableError: If no extractor is available
    """
    if extractor is None:
        extractor = MockInsightExtractor()
    
    # Perform extraction
    extracted_content = extractor.extract(content, level, context)
    
    # Create provenance record
    provenance = ExtractionProvenance(
        source_id=source_id,
        source_hash=_compute_content_hash(content),
        extracted_at=datetime.now(timezone.utc),
        extraction_level=level,
        extractor_id=extractor.extractor_id,
        extraction_metadata=context or {},
    )
    
    # Create insight record
    return ExtractedInsight(
        extraction_id=str(uuid.uuid4()),
        content=extracted_content,
        level=level,
        provenance=provenance,
        status=ExtractionStatus.PENDING_REVIEW,
        metadata={"source_length": len(content)},
    )


def approve_extraction(
    insight: ExtractedInsight,
    reviewer: str,
    notes: Optional[str] = None,
) -> ExtractedInsight:
    """Approve an extracted insight for sharing.
    
    Args:
        insight: The insight to approve
        reviewer: DID of the reviewer approving
        notes: Optional review notes
        
    Returns:
        Updated insight with APPROVED status
        
    Raises:
        ExtractionAlreadyReviewedError: If already reviewed
    """
    if not insight.is_pending:
        raise ExtractionAlreadyReviewedError(
            f"Extraction {insight.extraction_id} is already {insight.status.value}"
        )
    
    return ExtractedInsight(
        extraction_id=insight.extraction_id,
        content=insight.content,
        level=insight.level,
        provenance=insight.provenance,
        status=ExtractionStatus.APPROVED,
        reviewed_at=datetime.now(timezone.utc),
        reviewed_by=reviewer,
        review_notes=notes,
        original_extraction=None,
        created_at=insight.created_at,
        metadata=insight.metadata,
    )


def reject_extraction(
    insight: ExtractedInsight,
    reviewer: str,
    reason: Optional[str] = None,
) -> ExtractedInsight:
    """Reject an extracted insight (not suitable for sharing).
    
    Args:
        insight: The insight to reject
        reviewer: DID of the reviewer rejecting
        reason: Optional rejection reason
        
    Returns:
        Updated insight with REJECTED status
        
    Raises:
        ExtractionAlreadyReviewedError: If already reviewed
    """
    if not insight.is_pending:
        raise ExtractionAlreadyReviewedError(
            f"Extraction {insight.extraction_id} is already {insight.status.value}"
        )
    
    return ExtractedInsight(
        extraction_id=insight.extraction_id,
        content=insight.content,
        level=insight.level,
        provenance=insight.provenance,
        status=ExtractionStatus.REJECTED,
        reviewed_at=datetime.now(timezone.utc),
        reviewed_by=reviewer,
        review_notes=reason,
        original_extraction=None,
        created_at=insight.created_at,
        metadata=insight.metadata,
    )


def modify_extraction(
    insight: ExtractedInsight,
    reviewer: str,
    modified_content: str,
    notes: Optional[str] = None,
) -> ExtractedInsight:
    """Approve with modifications (human edits the AI extraction).
    
    Args:
        insight: The insight to modify and approve
        reviewer: DID of the reviewer
        modified_content: The human-edited content
        notes: Optional notes about the modifications
        
    Returns:
        Updated insight with MODIFIED status and new content
        
    Raises:
        ExtractionAlreadyReviewedError: If already reviewed
    """
    if not insight.is_pending:
        raise ExtractionAlreadyReviewedError(
            f"Extraction {insight.extraction_id} is already {insight.status.value}"
        )
    
    return ExtractedInsight(
        extraction_id=insight.extraction_id,
        content=modified_content,
        level=insight.level,
        provenance=insight.provenance,
        status=ExtractionStatus.MODIFIED,
        reviewed_at=datetime.now(timezone.utc),
        reviewed_by=reviewer,
        review_notes=notes,
        original_extraction=insight.content,  # Preserve original AI output
        created_at=insight.created_at,
        metadata=insight.metadata,
    )


class ExtractionService:
    """Service for managing the extraction workflow.
    
    Provides storage and retrieval of extractions with support for
    the human review process.
    """
    
    def __init__(self, extractor: Optional[InsightExtractor] = None):
        """Initialize the extraction service.
        
        Args:
            extractor: The AI extractor to use (defaults to MockInsightExtractor)
        """
        self._extractor = extractor or MockInsightExtractor()
        self._extractions: Dict[str, ExtractedInsight] = {}
        self._by_source: Dict[str, List[str]] = {}  # source_id -> [extraction_ids]
        self._by_reviewer: Dict[str, List[str]] = {}  # reviewer -> [extraction_ids]
    
    @property
    def extractor(self) -> InsightExtractor:
        """Get the configured extractor."""
        return self._extractor
    
    def set_extractor(self, extractor: InsightExtractor) -> None:
        """Set a new extractor."""
        self._extractor = extractor
    
    def extract(
        self,
        content: str,
        level: ExtractionLevel,
        source_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractedInsight:
        """Extract insights and store for review.
        
        Returns:
            ExtractedInsight in PENDING_REVIEW status
        """
        insight = extract_insights(
            content=content,
            level=level,
            source_id=source_id,
            extractor=self._extractor,
            context=context,
        )
        
        self._store_extraction(insight)
        return insight
    
    def approve(
        self,
        extraction_id: str,
        reviewer: str,
        notes: Optional[str] = None,
    ) -> ExtractedInsight:
        """Approve an extraction by ID.
        
        Raises:
            ExtractionNotFoundError: If extraction doesn't exist
            ExtractionAlreadyReviewedError: If already reviewed
        """
        insight = self.get(extraction_id)
        if insight is None:
            raise ExtractionNotFoundError(f"Extraction {extraction_id} not found")
        
        updated = approve_extraction(insight, reviewer, notes)
        self._extractions[extraction_id] = updated
        self._index_reviewer(updated)
        return updated
    
    def reject(
        self,
        extraction_id: str,
        reviewer: str,
        reason: Optional[str] = None,
    ) -> ExtractedInsight:
        """Reject an extraction by ID.
        
        Raises:
            ExtractionNotFoundError: If extraction doesn't exist
            ExtractionAlreadyReviewedError: If already reviewed
        """
        insight = self.get(extraction_id)
        if insight is None:
            raise ExtractionNotFoundError(f"Extraction {extraction_id} not found")
        
        updated = reject_extraction(insight, reviewer, reason)
        self._extractions[extraction_id] = updated
        self._index_reviewer(updated)
        return updated
    
    def modify(
        self,
        extraction_id: str,
        reviewer: str,
        modified_content: str,
        notes: Optional[str] = None,
    ) -> ExtractedInsight:
        """Modify and approve an extraction by ID.
        
        Raises:
            ExtractionNotFoundError: If extraction doesn't exist
            ExtractionAlreadyReviewedError: If already reviewed
        """
        insight = self.get(extraction_id)
        if insight is None:
            raise ExtractionNotFoundError(f"Extraction {extraction_id} not found")
        
        updated = modify_extraction(insight, reviewer, modified_content, notes)
        self._extractions[extraction_id] = updated
        self._index_reviewer(updated)
        return updated
    
    def get(self, extraction_id: str) -> Optional[ExtractedInsight]:
        """Get an extraction by ID."""
        return self._extractions.get(extraction_id)
    
    def get_for_source(self, source_id: str) -> List[ExtractedInsight]:
        """Get all extractions from a specific source."""
        extraction_ids = self._by_source.get(source_id, [])
        return [self._extractions[eid] for eid in extraction_ids if eid in self._extractions]
    
    def get_by_reviewer(self, reviewer: str) -> List[ExtractedInsight]:
        """Get all extractions reviewed by a specific reviewer."""
        extraction_ids = self._by_reviewer.get(reviewer, [])
        return [self._extractions[eid] for eid in extraction_ids if eid in self._extractions]
    
    def get_pending(self) -> List[ExtractedInsight]:
        """Get all extractions pending review."""
        return [e for e in self._extractions.values() if e.is_pending]
    
    def get_approved(self) -> List[ExtractedInsight]:
        """Get all approved extractions (including modified)."""
        return [e for e in self._extractions.values() if e.is_approved]
    
    def _store_extraction(self, insight: ExtractedInsight) -> None:
        """Store extraction and update indices."""
        self._extractions[insight.extraction_id] = insight
        
        source_id = insight.provenance.source_id
        if source_id not in self._by_source:
            self._by_source[source_id] = []
        self._by_source[source_id].append(insight.extraction_id)
    
    def _index_reviewer(self, insight: ExtractedInsight) -> None:
        """Index extraction by reviewer after review."""
        if insight.reviewed_by:
            if insight.reviewed_by not in self._by_reviewer:
                self._by_reviewer[insight.reviewed_by] = []
            if insight.extraction_id not in self._by_reviewer[insight.reviewed_by]:
                self._by_reviewer[insight.reviewed_by].append(insight.extraction_id)


# Module-level singleton for convenient access
_extraction_service: Optional[ExtractionService] = None


def get_extraction_service() -> ExtractionService:
    """Get the global extraction service singleton."""
    global _extraction_service
    if _extraction_service is None:
        _extraction_service = ExtractionService()
    return _extraction_service


def set_extraction_service(service: ExtractionService) -> None:
    """Set the global extraction service singleton."""
    global _extraction_service
    _extraction_service = service
