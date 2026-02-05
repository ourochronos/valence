"""Tests for AI-assisted insight extraction (Issue #95)."""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from valence.privacy.extraction import (
    # Enums
    ExtractionLevel,
    ExtractionStatus,
    # Errors
    ExtractionError,
    ExtractorNotAvailableError,
    ExtractionNotFoundError,
    ExtractionAlreadyReviewedError,
    # Data classes
    ExtractionProvenance,
    ExtractedInsight,
    # Extractors
    InsightExtractor,
    MockInsightExtractor,
    # Functions
    extract_insights,
    approve_extraction,
    reject_extraction,
    modify_extraction,
    # Service
    ExtractionService,
    get_extraction_service,
    set_extraction_service,
)


# =============================================================================
# ExtractionLevel Enum Tests
# =============================================================================

class TestExtractionLevel:
    """Tests for ExtractionLevel enum."""
    
    def test_all_levels_defined(self):
        """Verify all required extraction levels exist."""
        assert ExtractionLevel.THEMES.value == "themes"
        assert ExtractionLevel.KEY_POINTS.value == "key_points"
        assert ExtractionLevel.SUMMARY.value == "summary"
        assert ExtractionLevel.ANONYMIZED.value == "anonymized"
    
    def test_level_from_string(self):
        """Can create level from string value."""
        assert ExtractionLevel("themes") == ExtractionLevel.THEMES
        assert ExtractionLevel("key_points") == ExtractionLevel.KEY_POINTS
        assert ExtractionLevel("summary") == ExtractionLevel.SUMMARY
        assert ExtractionLevel("anonymized") == ExtractionLevel.ANONYMIZED
    
    def test_invalid_level_raises(self):
        """Invalid level string raises ValueError."""
        with pytest.raises(ValueError):
            ExtractionLevel("invalid_level")


class TestExtractionStatus:
    """Tests for ExtractionStatus enum."""
    
    def test_all_statuses_defined(self):
        """Verify all statuses exist."""
        assert ExtractionStatus.PENDING_REVIEW.value == "pending_review"
        assert ExtractionStatus.APPROVED.value == "approved"
        assert ExtractionStatus.REJECTED.value == "rejected"
        assert ExtractionStatus.MODIFIED.value == "modified"


# =============================================================================
# ExtractionProvenance Tests
# =============================================================================

class TestExtractionProvenance:
    """Tests for ExtractionProvenance dataclass."""
    
    def test_create_provenance(self):
        """Can create provenance with required fields."""
        now = datetime.now(timezone.utc)
        prov = ExtractionProvenance(
            source_id="belief-123",
            source_hash="abc123",
            extracted_at=now,
            extraction_level=ExtractionLevel.SUMMARY,
            extractor_id="mock-v1",
        )
        
        assert prov.source_id == "belief-123"
        assert prov.source_hash == "abc123"
        assert prov.extracted_at == now
        assert prov.extraction_level == ExtractionLevel.SUMMARY
        assert prov.extractor_id == "mock-v1"
        assert prov.extraction_metadata == {}
    
    def test_provenance_with_metadata(self):
        """Can create provenance with metadata."""
        prov = ExtractionProvenance(
            source_id="belief-123",
            source_hash="abc123",
            extracted_at=datetime.now(timezone.utc),
            extraction_level=ExtractionLevel.THEMES,
            extractor_id="mock-v1",
            extraction_metadata={"domain": "test", "intent": "share"},
        )
        
        assert prov.extraction_metadata["domain"] == "test"
        assert prov.extraction_metadata["intent"] == "share"
    
    def test_provenance_serialization(self):
        """Provenance can be serialized and deserialized."""
        now = datetime.now(timezone.utc)
        prov = ExtractionProvenance(
            source_id="belief-123",
            source_hash="abc123",
            extracted_at=now,
            extraction_level=ExtractionLevel.KEY_POINTS,
            extractor_id="mock-v1",
            extraction_metadata={"key": "value"},
        )
        
        data = prov.to_dict()
        assert data["source_id"] == "belief-123"
        assert data["relationship"] == "extracted_from"
        assert data["extraction_level"] == "key_points"
        
        restored = ExtractionProvenance.from_dict(data)
        assert restored.source_id == prov.source_id
        assert restored.source_hash == prov.source_hash
        assert restored.extraction_level == prov.extraction_level
        assert restored.extractor_id == prov.extractor_id


# =============================================================================
# ExtractedInsight Tests
# =============================================================================

class TestExtractedInsight:
    """Tests for ExtractedInsight dataclass."""
    
    @pytest.fixture
    def sample_provenance(self):
        """Create sample provenance for tests."""
        return ExtractionProvenance(
            source_id="belief-456",
            source_hash="def456",
            extracted_at=datetime.now(timezone.utc),
            extraction_level=ExtractionLevel.SUMMARY,
            extractor_id="mock-v1",
        )
    
    def test_create_insight(self, sample_provenance):
        """Can create an extracted insight."""
        insight = ExtractedInsight(
            extraction_id="ext-001",
            content="This is the extracted summary.",
            level=ExtractionLevel.SUMMARY,
            provenance=sample_provenance,
        )
        
        assert insight.extraction_id == "ext-001"
        assert insight.content == "This is the extracted summary."
        assert insight.level == ExtractionLevel.SUMMARY
        assert insight.status == ExtractionStatus.PENDING_REVIEW
        assert insight.reviewed_at is None
        assert insight.reviewed_by is None
    
    def test_insight_pending_properties(self, sample_provenance):
        """Pending insight has correct property values."""
        insight = ExtractedInsight(
            extraction_id="ext-001",
            content="Content",
            level=ExtractionLevel.THEMES,
            provenance=sample_provenance,
        )
        
        assert insight.is_pending is True
        assert insight.is_approved is False
        assert insight.is_rejected is False
        assert insight.was_modified is False
    
    def test_insight_approved_properties(self, sample_provenance):
        """Approved insight has correct property values."""
        insight = ExtractedInsight(
            extraction_id="ext-001",
            content="Content",
            level=ExtractionLevel.THEMES,
            provenance=sample_provenance,
            status=ExtractionStatus.APPROVED,
        )
        
        assert insight.is_pending is False
        assert insight.is_approved is True
        assert insight.is_rejected is False
        assert insight.was_modified is False
    
    def test_insight_modified_properties(self, sample_provenance):
        """Modified insight is considered approved."""
        insight = ExtractedInsight(
            extraction_id="ext-001",
            content="Modified content",
            level=ExtractionLevel.THEMES,
            provenance=sample_provenance,
            status=ExtractionStatus.MODIFIED,
            original_extraction="Original content",
        )
        
        assert insight.is_pending is False
        assert insight.is_approved is True  # Modified counts as approved
        assert insight.is_rejected is False
        assert insight.was_modified is True
    
    def test_insight_rejected_properties(self, sample_provenance):
        """Rejected insight has correct property values."""
        insight = ExtractedInsight(
            extraction_id="ext-001",
            content="Content",
            level=ExtractionLevel.THEMES,
            provenance=sample_provenance,
            status=ExtractionStatus.REJECTED,
        )
        
        assert insight.is_pending is False
        assert insight.is_approved is False
        assert insight.is_rejected is True
        assert insight.was_modified is False
    
    def test_insight_serialization(self, sample_provenance):
        """Insight can be serialized and deserialized."""
        insight = ExtractedInsight(
            extraction_id="ext-001",
            content="Test content",
            level=ExtractionLevel.KEY_POINTS,
            provenance=sample_provenance,
            metadata={"test": True},
        )
        
        data = insight.to_dict()
        assert data["extraction_id"] == "ext-001"
        assert data["status"] == "pending_review"
        assert data["level"] == "key_points"
        assert "provenance" in data
        
        restored = ExtractedInsight.from_dict(data)
        assert restored.extraction_id == insight.extraction_id
        assert restored.content == insight.content
        assert restored.level == insight.level
        assert restored.status == insight.status


# =============================================================================
# MockInsightExtractor Tests
# =============================================================================

class TestMockInsightExtractor:
    """Tests for the mock AI extractor."""
    
    def test_extractor_has_id(self):
        """Extractor has an identifier."""
        extractor = MockInsightExtractor()
        assert extractor.extractor_id == "mock-extractor-v1"
    
    def test_custom_extractor_id(self):
        """Can customize extractor ID."""
        extractor = MockInsightExtractor(extractor_id="custom-v2")
        assert extractor.extractor_id == "custom-v2"
    
    def test_extract_themes(self):
        """Can extract themes from content."""
        extractor = MockInsightExtractor()
        content = "Machine learning algorithms process data to make predictions. Neural networks are a type of machine learning."
        
        result = extractor.extract(content, ExtractionLevel.THEMES)
        
        assert "Themes:" in result
        # Should identify relevant themes
        assert "learning" in result.lower() or "machine" in result.lower()
    
    def test_extract_key_points(self):
        """Can extract key points from content."""
        extractor = MockInsightExtractor()
        content = "First important point. Second key fact. Third notable item."
        
        result = extractor.extract(content, ExtractionLevel.KEY_POINTS)
        
        assert "Key points:" in result
        assert "•" in result  # Bullet points
    
    def test_extract_summary(self):
        """Can extract summary from content."""
        extractor = MockInsightExtractor()
        content = "This is a long piece of content. " * 20
        
        result = extractor.extract(content, ExtractionLevel.SUMMARY)
        
        # Summary should be shorter than original
        assert len(result) < len(content)
    
    def test_extract_anonymized_email(self):
        """Anonymization removes email addresses."""
        extractor = MockInsightExtractor()
        content = "Contact me at john.doe@example.com for more info."
        
        result = extractor.extract(content, ExtractionLevel.ANONYMIZED)
        
        assert "[EMAIL]" in result
        assert "john.doe@example.com" not in result
    
    def test_extract_anonymized_phone(self):
        """Anonymization removes phone numbers."""
        extractor = MockInsightExtractor()
        content = "Call me at 555-123-4567 tomorrow."
        
        result = extractor.extract(content, ExtractionLevel.ANONYMIZED)
        
        assert "[PHONE]" in result
        assert "555-123-4567" not in result
    
    def test_extract_anonymized_ssn(self):
        """Anonymization removes SSN-like patterns."""
        extractor = MockInsightExtractor()
        content = "My SSN is 123-45-6789."
        
        result = extractor.extract(content, ExtractionLevel.ANONYMIZED)
        
        assert "[SSN]" in result
        assert "123-45-6789" not in result
    
    def test_custom_extractor_function(self):
        """Can provide custom extraction function."""
        def custom_themes(content: str) -> str:
            return "Custom themes: AI, Technology"
        
        extractor = MockInsightExtractor(
            custom_extractors={ExtractionLevel.THEMES: custom_themes}
        )
        
        result = extractor.extract("any content", ExtractionLevel.THEMES)
        assert result == "Custom themes: AI, Technology"
    
    def test_supports_all_levels(self):
        """Mock extractor supports all levels by default."""
        extractor = MockInsightExtractor()
        
        for level in ExtractionLevel:
            assert extractor.supports_level(level) is True


# =============================================================================
# extract_insights Function Tests
# =============================================================================

class TestExtractInsights:
    """Tests for the extract_insights function."""
    
    def test_basic_extraction(self):
        """Can extract insights with default extractor."""
        result = extract_insights(
            content="This is test content for extraction.",
            level=ExtractionLevel.SUMMARY,
            source_id="belief-789",
        )
        
        assert isinstance(result, ExtractedInsight)
        assert result.status == ExtractionStatus.PENDING_REVIEW
        assert result.level == ExtractionLevel.SUMMARY
        assert result.provenance.source_id == "belief-789"
    
    def test_extraction_creates_provenance(self):
        """Extraction creates proper provenance record."""
        result = extract_insights(
            content="Test content",
            level=ExtractionLevel.THEMES,
            source_id="belief-001",
        )
        
        prov = result.provenance
        assert prov.source_id == "belief-001"
        assert prov.source_hash is not None
        assert len(prov.source_hash) == 16  # Truncated SHA256
        assert prov.extraction_level == ExtractionLevel.THEMES
        assert prov.extractor_id == "mock-extractor-v1"
    
    def test_extraction_with_custom_extractor(self):
        """Can use custom extractor."""
        extractor = MockInsightExtractor(extractor_id="custom-test-v1")
        
        result = extract_insights(
            content="Test content",
            level=ExtractionLevel.KEY_POINTS,
            source_id="belief-002",
            extractor=extractor,
        )
        
        assert result.provenance.extractor_id == "custom-test-v1"
    
    def test_extraction_with_context(self):
        """Can provide context for extraction."""
        result = extract_insights(
            content="Test content",
            level=ExtractionLevel.SUMMARY,
            source_id="belief-003",
            context={"domain": "technology", "audience": "public"},
        )
        
        assert result.provenance.extraction_metadata["domain"] == "technology"
        assert result.provenance.extraction_metadata["audience"] == "public"
    
    def test_extraction_records_source_length(self):
        """Extraction records source content length."""
        content = "A" * 1000
        result = extract_insights(
            content=content,
            level=ExtractionLevel.SUMMARY,
            source_id="belief-004",
        )
        
        assert result.metadata["source_length"] == 1000
    
    def test_extraction_generates_unique_ids(self):
        """Each extraction gets a unique ID."""
        ids = set()
        for _ in range(10):
            result = extract_insights(
                content="Test",
                level=ExtractionLevel.THEMES,
                source_id="belief-005",
            )
            ids.add(result.extraction_id)
        
        assert len(ids) == 10  # All unique


# =============================================================================
# Review Workflow Tests
# =============================================================================

class TestApproveExtraction:
    """Tests for approve_extraction function."""
    
    @pytest.fixture
    def pending_insight(self):
        """Create a pending insight for tests."""
        return extract_insights(
            content="Test content for approval",
            level=ExtractionLevel.SUMMARY,
            source_id="belief-approve-test",
        )
    
    def test_approve_pending(self, pending_insight):
        """Can approve a pending extraction."""
        approved = approve_extraction(
            insight=pending_insight,
            reviewer="did:example:reviewer",
        )
        
        assert approved.status == ExtractionStatus.APPROVED
        assert approved.reviewed_by == "did:example:reviewer"
        assert approved.reviewed_at is not None
        assert approved.is_approved is True
    
    def test_approve_with_notes(self, pending_insight):
        """Can add review notes when approving."""
        approved = approve_extraction(
            insight=pending_insight,
            reviewer="did:example:reviewer",
            notes="Looks good, safe to share",
        )
        
        assert approved.review_notes == "Looks good, safe to share"
    
    def test_cannot_approve_already_approved(self, pending_insight):
        """Cannot approve an already-approved extraction."""
        approved = approve_extraction(pending_insight, "reviewer1")
        
        with pytest.raises(ExtractionAlreadyReviewedError):
            approve_extraction(approved, "reviewer2")
    
    def test_cannot_approve_rejected(self, pending_insight):
        """Cannot approve a rejected extraction."""
        rejected = reject_extraction(pending_insight, "reviewer1")
        
        with pytest.raises(ExtractionAlreadyReviewedError):
            approve_extraction(rejected, "reviewer2")


class TestRejectExtraction:
    """Tests for reject_extraction function."""
    
    @pytest.fixture
    def pending_insight(self):
        """Create a pending insight for tests."""
        return extract_insights(
            content="Test content for rejection",
            level=ExtractionLevel.KEY_POINTS,
            source_id="belief-reject-test",
        )
    
    def test_reject_pending(self, pending_insight):
        """Can reject a pending extraction."""
        rejected = reject_extraction(
            insight=pending_insight,
            reviewer="did:example:reviewer",
        )
        
        assert rejected.status == ExtractionStatus.REJECTED
        assert rejected.reviewed_by == "did:example:reviewer"
        assert rejected.reviewed_at is not None
        assert rejected.is_rejected is True
    
    def test_reject_with_reason(self, pending_insight):
        """Can add rejection reason."""
        rejected = reject_extraction(
            insight=pending_insight,
            reviewer="did:example:reviewer",
            reason="Contains sensitive information",
        )
        
        assert rejected.review_notes == "Contains sensitive information"
    
    def test_cannot_reject_already_rejected(self, pending_insight):
        """Cannot reject an already-rejected extraction."""
        rejected = reject_extraction(pending_insight, "reviewer1")
        
        with pytest.raises(ExtractionAlreadyReviewedError):
            reject_extraction(rejected, "reviewer2")


class TestModifyExtraction:
    """Tests for modify_extraction function."""
    
    @pytest.fixture
    def pending_insight(self):
        """Create a pending insight for tests."""
        return extract_insights(
            content="Test content for modification",
            level=ExtractionLevel.ANONYMIZED,
            source_id="belief-modify-test",
        )
    
    def test_modify_pending(self, pending_insight):
        """Can modify and approve a pending extraction."""
        modified = modify_extraction(
            insight=pending_insight,
            reviewer="did:example:reviewer",
            modified_content="Human-edited content",
        )
        
        assert modified.status == ExtractionStatus.MODIFIED
        assert modified.content == "Human-edited content"
        assert modified.original_extraction == pending_insight.content
        assert modified.is_approved is True
        assert modified.was_modified is True
    
    def test_modify_with_notes(self, pending_insight):
        """Can add notes when modifying."""
        modified = modify_extraction(
            insight=pending_insight,
            reviewer="did:example:reviewer",
            modified_content="Edited",
            notes="Removed additional sensitive info",
        )
        
        assert modified.review_notes == "Removed additional sensitive info"
    
    def test_cannot_modify_approved(self, pending_insight):
        """Cannot modify an already-approved extraction."""
        approved = approve_extraction(pending_insight, "reviewer1")
        
        with pytest.raises(ExtractionAlreadyReviewedError):
            modify_extraction(approved, "reviewer2", "New content")


# =============================================================================
# ExtractionService Tests
# =============================================================================

class TestExtractionService:
    """Tests for ExtractionService."""
    
    @pytest.fixture
    def service(self):
        """Create a fresh extraction service."""
        return ExtractionService()
    
    def test_service_has_default_extractor(self, service):
        """Service has a default extractor."""
        assert service.extractor is not None
        assert isinstance(service.extractor, MockInsightExtractor)
    
    def test_service_extract(self, service):
        """Service can extract and store insights."""
        insight = service.extract(
            content="Test content",
            level=ExtractionLevel.THEMES,
            source_id="belief-svc-001",
        )
        
        assert insight.status == ExtractionStatus.PENDING_REVIEW
        assert service.get(insight.extraction_id) is not None
    
    def test_service_approve(self, service):
        """Service can approve stored extraction."""
        insight = service.extract("Content", ExtractionLevel.SUMMARY, "src-001")
        
        approved = service.approve(
            extraction_id=insight.extraction_id,
            reviewer="did:reviewer",
            notes="Approved",
        )
        
        assert approved.is_approved
        assert service.get(insight.extraction_id).is_approved
    
    def test_service_reject(self, service):
        """Service can reject stored extraction."""
        insight = service.extract("Content", ExtractionLevel.SUMMARY, "src-002")
        
        rejected = service.reject(
            extraction_id=insight.extraction_id,
            reviewer="did:reviewer",
            reason="Not suitable",
        )
        
        assert rejected.is_rejected
        assert service.get(insight.extraction_id).is_rejected
    
    def test_service_modify(self, service):
        """Service can modify stored extraction."""
        insight = service.extract("Content", ExtractionLevel.SUMMARY, "src-003")
        
        modified = service.modify(
            extraction_id=insight.extraction_id,
            reviewer="did:reviewer",
            modified_content="Better content",
        )
        
        assert modified.was_modified
        assert service.get(insight.extraction_id).content == "Better content"
    
    def test_service_get_nonexistent(self, service):
        """Getting nonexistent extraction returns None."""
        assert service.get("nonexistent-id") is None
    
    def test_service_approve_nonexistent_raises(self, service):
        """Approving nonexistent extraction raises error."""
        with pytest.raises(ExtractionNotFoundError):
            service.approve("nonexistent", "reviewer")
    
    def test_service_get_for_source(self, service):
        """Can get all extractions for a source."""
        # Create multiple extractions from same source
        service.extract("Content 1", ExtractionLevel.THEMES, "source-A")
        service.extract("Content 2", ExtractionLevel.SUMMARY, "source-A")
        service.extract("Content 3", ExtractionLevel.KEY_POINTS, "source-B")
        
        source_a_extractions = service.get_for_source("source-A")
        assert len(source_a_extractions) == 2
        
        source_b_extractions = service.get_for_source("source-B")
        assert len(source_b_extractions) == 1
    
    def test_service_get_by_reviewer(self, service):
        """Can get all extractions reviewed by a reviewer."""
        e1 = service.extract("Content 1", ExtractionLevel.THEMES, "src-1")
        e2 = service.extract("Content 2", ExtractionLevel.SUMMARY, "src-2")
        e3 = service.extract("Content 3", ExtractionLevel.KEY_POINTS, "src-3")
        
        service.approve(e1.extraction_id, "reviewer-A")
        service.reject(e2.extraction_id, "reviewer-A")
        service.approve(e3.extraction_id, "reviewer-B")
        
        reviewer_a = service.get_by_reviewer("reviewer-A")
        assert len(reviewer_a) == 2
        
        reviewer_b = service.get_by_reviewer("reviewer-B")
        assert len(reviewer_b) == 1
    
    def test_service_get_pending(self, service):
        """Can get all pending extractions."""
        e1 = service.extract("Content 1", ExtractionLevel.THEMES, "src-1")
        e2 = service.extract("Content 2", ExtractionLevel.SUMMARY, "src-2")
        e3 = service.extract("Content 3", ExtractionLevel.KEY_POINTS, "src-3")
        
        service.approve(e1.extraction_id, "reviewer")
        
        pending = service.get_pending()
        assert len(pending) == 2
        pending_ids = {p.extraction_id for p in pending}
        assert e2.extraction_id in pending_ids
        assert e3.extraction_id in pending_ids
    
    def test_service_get_approved(self, service):
        """Can get all approved extractions."""
        e1 = service.extract("Content 1", ExtractionLevel.THEMES, "src-1")
        e2 = service.extract("Content 2", ExtractionLevel.SUMMARY, "src-2")
        e3 = service.extract("Content 3", ExtractionLevel.KEY_POINTS, "src-3")
        
        service.approve(e1.extraction_id, "reviewer")
        service.modify(e2.extraction_id, "reviewer", "Modified")
        service.reject(e3.extraction_id, "reviewer")
        
        approved = service.get_approved()
        assert len(approved) == 2  # Both APPROVED and MODIFIED
    
    def test_service_set_extractor(self, service):
        """Can set a custom extractor."""
        custom = MockInsightExtractor(extractor_id="custom-svc")
        service.set_extractor(custom)
        
        insight = service.extract("Content", ExtractionLevel.THEMES, "src")
        assert insight.provenance.extractor_id == "custom-svc"


# =============================================================================
# Global Service Tests
# =============================================================================

class TestGlobalService:
    """Tests for global extraction service functions."""
    
    def test_get_extraction_service(self):
        """Can get global service singleton."""
        svc = get_extraction_service()
        assert svc is not None
        assert isinstance(svc, ExtractionService)
    
    def test_set_extraction_service(self):
        """Can set global service singleton."""
        custom = ExtractionService(
            extractor=MockInsightExtractor(extractor_id="global-custom")
        )
        set_extraction_service(custom)
        
        svc = get_extraction_service()
        assert svc.extractor.extractor_id == "global-custom"


# =============================================================================
# Custom Extractor Tests
# =============================================================================

class CustomTestExtractor(InsightExtractor):
    """Custom extractor implementation for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.last_content = None
        self.last_level = None
    
    @property
    def extractor_id(self) -> str:
        return "custom-test-extractor"
    
    def extract(
        self,
        content: str,
        level: ExtractionLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.call_count += 1
        self.last_content = content
        self.last_level = level
        return f"Custom extraction at {level.value}: {len(content)} chars"


class TestCustomExtractor:
    """Tests for custom extractor implementations."""
    
    def test_custom_extractor_integration(self):
        """Custom extractor can be used with extract_insights."""
        extractor = CustomTestExtractor()
        
        result = extract_insights(
            content="Test content here",
            level=ExtractionLevel.SUMMARY,
            source_id="custom-test",
            extractor=extractor,
        )
        
        assert extractor.call_count == 1
        assert extractor.last_content == "Test content here"
        assert extractor.last_level == ExtractionLevel.SUMMARY
        assert "17 chars" in result.content
    
    def test_custom_extractor_with_service(self):
        """Custom extractor works with ExtractionService."""
        extractor = CustomTestExtractor()
        service = ExtractionService(extractor=extractor)
        
        service.extract("Content", ExtractionLevel.THEMES, "src")
        service.extract("More content", ExtractionLevel.KEY_POINTS, "src")
        
        assert extractor.call_count == 2


# =============================================================================
# Provenance Integration Tests
# =============================================================================

class TestProvenanceIntegration:
    """Tests for provenance tracking."""
    
    def test_provenance_tracks_extraction_relationship(self):
        """Provenance correctly tracks 'extracted from' relationship."""
        insight = extract_insights(
            content="Original private content with details",
            level=ExtractionLevel.THEMES,
            source_id="private-belief-123",
        )
        
        prov_dict = insight.provenance.to_dict()
        
        assert prov_dict["relationship"] == "extracted_from"
        assert prov_dict["source_id"] == "private-belief-123"
        assert prov_dict["extraction_level"] == "themes"
    
    def test_provenance_hash_changes_with_content(self):
        """Different content produces different source hashes."""
        insight1 = extract_insights("Content A", ExtractionLevel.SUMMARY, "src")
        insight2 = extract_insights("Content B", ExtractionLevel.SUMMARY, "src")
        
        assert insight1.provenance.source_hash != insight2.provenance.source_hash
    
    def test_provenance_hash_same_for_same_content(self):
        """Same content produces same source hash."""
        content = "Identical content"
        insight1 = extract_insights(content, ExtractionLevel.SUMMARY, "src1")
        insight2 = extract_insights(content, ExtractionLevel.SUMMARY, "src2")
        
        assert insight1.provenance.source_hash == insight2.provenance.source_hash
    
    def test_provenance_preserved_through_review(self):
        """Provenance is preserved when extraction is reviewed."""
        insight = extract_insights(
            content="Test content",
            level=ExtractionLevel.KEY_POINTS,
            source_id="origin-123",
        )
        original_provenance = insight.provenance.to_dict()
        
        approved = approve_extraction(insight, "reviewer")
        
        assert approved.provenance.to_dict() == original_provenance
    
    def test_modified_extraction_preserves_provenance(self):
        """Modified extraction keeps original provenance."""
        insight = extract_insights(
            content="Original content",
            level=ExtractionLevel.ANONYMIZED,
            source_id="origin-456",
        )
        
        modified = modify_extraction(insight, "reviewer", "Human-modified content")
        
        # Provenance still points to original source
        assert modified.provenance.source_id == "origin-456"
        # But we can see both versions
        assert modified.original_extraction is not None
        assert modified.content == "Human-modified content"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_content_extraction(self):
        """Can extract from empty content."""
        insight = extract_insights(
            content="",
            level=ExtractionLevel.THEMES,
            source_id="empty-src",
        )
        
        assert insight is not None
        assert insight.metadata["source_length"] == 0
    
    def test_very_long_content(self):
        """Can extract from very long content."""
        long_content = "This is a test sentence. " * 10000
        
        insight = extract_insights(
            content=long_content,
            level=ExtractionLevel.SUMMARY,
            source_id="long-src",
        )
        
        # Summary should be shorter
        assert len(insight.content) < len(long_content)
    
    def test_special_characters_in_content(self):
        """Handles special characters in content."""
        content = "Test with émojis 🎉 and spëcial chars: <>&\"'"
        
        insight = extract_insights(
            content=content,
            level=ExtractionLevel.ANONYMIZED,
            source_id="special-src",
        )
        
        assert insight is not None
    
    def test_all_extraction_levels(self):
        """Can extract at all levels."""
        content = "Sample content for testing all extraction levels."
        
        for level in ExtractionLevel:
            insight = extract_insights(
                content=content,
                level=level,
                source_id=f"test-{level.value}",
            )
            assert insight.level == level


# =============================================================================
# MockInsightExtractor Production Guard Tests (Issue #175)
# =============================================================================

class TestMockExtractorProductionGuard:
    """Tests for MockInsightExtractor production guard.
    
    Issue #175: Ensure MockInsightExtractor cannot be used in production.
    """
    
    def test_mock_extractor_allowed_without_env(self, monkeypatch):
        """MockInsightExtractor works when VALENCE_ENV is not set."""
        monkeypatch.delenv("VALENCE_ENV", raising=False)
        
        # Should not raise
        extractor = MockInsightExtractor()
        assert extractor.extractor_id == "mock-extractor-v1"
    
    def test_mock_extractor_allowed_in_dev(self, monkeypatch):
        """MockInsightExtractor works when VALENCE_ENV=development."""
        monkeypatch.setenv("VALENCE_ENV", "development")
        
        # Should not raise
        extractor = MockInsightExtractor()
        assert extractor.extractor_id == "mock-extractor-v1"
    
    def test_mock_extractor_allowed_in_test(self, monkeypatch):
        """MockInsightExtractor works when VALENCE_ENV=test."""
        monkeypatch.setenv("VALENCE_ENV", "test")
        
        # Should not raise
        extractor = MockInsightExtractor()
        assert extractor.extractor_id == "mock-extractor-v1"
    
    def test_mock_extractor_blocked_in_production(self, monkeypatch):
        """MockInsightExtractor raises RuntimeError when VALENCE_ENV=production."""
        monkeypatch.setenv("VALENCE_ENV", "production")
        
        with pytest.raises(RuntimeError) as exc_info:
            MockInsightExtractor()
        
        assert "cannot be used in production" in str(exc_info.value)
        assert "VALENCE_ENV" in str(exc_info.value)
    
    def test_mock_extractor_blocked_case_insensitive(self, monkeypatch):
        """Production guard is case-insensitive."""
        monkeypatch.setenv("VALENCE_ENV", "PRODUCTION")
        
        with pytest.raises(RuntimeError) as exc_info:
            MockInsightExtractor()
        
        assert "cannot be used in production" in str(exc_info.value)
    
    def test_mock_extractor_blocked_mixed_case(self, monkeypatch):
        """Production guard handles mixed case."""
        monkeypatch.setenv("VALENCE_ENV", "Production")
        
        with pytest.raises(RuntimeError) as exc_info:
            MockInsightExtractor()
        
        assert "cannot be used in production" in str(exc_info.value)
    
    def test_allow_in_production_flag_for_testing(self, monkeypatch):
        """Internal _allow_in_production flag bypasses guard for testing."""
        monkeypatch.setenv("VALENCE_ENV", "production")
        
        # Should not raise with the internal flag
        extractor = MockInsightExtractor(_allow_in_production=True)
        assert extractor.extractor_id == "mock-extractor-v1"
    
    def test_mock_extractor_error_message_helpful(self, monkeypatch):
        """Error message provides guidance on what to do instead."""
        monkeypatch.setenv("VALENCE_ENV", "production")
        
        with pytest.raises(RuntimeError) as exc_info:
            MockInsightExtractor()
        
        error_msg = str(exc_info.value)
        # Should mention it's a mock
        assert "MockInsightExtractor" in error_msg
        # Should explain why it's blocked
        assert "production" in error_msg.lower()
        # Should suggest using a real implementation
        assert "real" in error_msg.lower() or "InsightExtractor" in error_msg
