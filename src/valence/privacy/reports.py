"""Self-service data report API for GDPR compliance (Issue #84).

Users can request a complete export of their data including:
- Beliefs they own
- Shares sent to others
- Shares received from others
- Trust edges (both directions)
- Audit events involving them

Supports JSON and CSV export formats with async generation for large reports.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats for data reports."""

    JSON = "json"
    CSV = "csv"


class ReportStatus(str, Enum):
    """Status of a data report generation job."""

    PENDING = "pending"  # Report requested, not yet started
    GENERATING = "generating"  # Currently being generated
    COMPLETED = "completed"  # Ready for download
    FAILED = "failed"  # Generation failed
    EXPIRED = "expired"  # Report expired and was deleted


@dataclass
class ReportScope:
    """Defines what data to include in a report.

    Attributes:
        include_beliefs: Include beliefs owned by the user
        include_shares_sent: Include shares the user has sent
        include_shares_received: Include shares the user has received
        include_trust_outgoing: Include trust edges from the user
        include_trust_incoming: Include trust edges to the user
        include_audit_events: Include audit events involving the user
        start_date: Only include data from this date (optional)
        end_date: Only include data until this date (optional)
        domains: Only include data from these domains (optional, empty = all)
    """

    include_beliefs: bool = True
    include_shares_sent: bool = True
    include_shares_received: bool = True
    include_trust_outgoing: bool = True
    include_trust_incoming: bool = True
    include_audit_events: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    domains: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "include_beliefs": self.include_beliefs,
            "include_shares_sent": self.include_shares_sent,
            "include_shares_received": self.include_shares_received,
            "include_trust_outgoing": self.include_trust_outgoing,
            "include_trust_incoming": self.include_trust_incoming,
            "include_audit_events": self.include_audit_events,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "domains": self.domains,
        }


@dataclass
class BeliefRecord:
    """A belief record for the data report."""

    belief_id: str
    content: str
    confidence: float
    domains: list[str]
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShareRecord:
    """A share record for the data report."""

    share_id: str
    belief_id: str
    sharer_did: str
    recipient_did: str
    created_at: datetime
    policy_level: str
    revoked: bool = False
    revoked_at: Optional[datetime] = None


@dataclass
class TrustRecord:
    """A trust edge record for the data report."""

    source_did: str
    target_did: str
    competence: float
    integrity: float
    confidentiality: float
    judgment: float
    domain: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


@dataclass
class AuditRecord:
    """An audit event record for the data report."""

    event_id: str
    event_type: str
    actor_did: str
    target_did: Optional[str]
    resource: str
    action: str
    success: bool
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportMetadata:
    """Metadata about a generated report.

    Attributes:
        report_id: Unique identifier for the report
        user_did: DID of the user who requested the report
        requested_at: When the report was requested
        generated_at: When the report was completed (None if not yet done)
        scope: What data was included
        format: Export format used
        status: Current status of the report
        error_message: Error details if generation failed
        record_counts: Count of records by type
        size_bytes: Size of the generated report in bytes
        expires_at: When the report will be auto-deleted
    """

    report_id: str
    user_did: str
    requested_at: datetime
    scope: ReportScope
    format: ExportFormat
    status: ReportStatus = ReportStatus.PENDING
    generated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    record_counts: dict[str, int] = field(default_factory=dict)
    size_bytes: int = 0
    expires_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_id": self.report_id,
            "user_did": self.user_did,
            "requested_at": self.requested_at.isoformat(),
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "scope": self.scope.to_dict(),
            "format": self.format.value,
            "status": self.status.value,
            "error_message": self.error_message,
            "record_counts": self.record_counts,
            "size_bytes": self.size_bytes,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class DataReport:
    """Complete data report for a user.

    Contains all data associated with a user's DID, organized by type.
    """

    metadata: ReportMetadata
    beliefs: list[BeliefRecord] = field(default_factory=list)
    shares_sent: list[ShareRecord] = field(default_factory=list)
    shares_received: list[ShareRecord] = field(default_factory=list)
    trust_outgoing: list[TrustRecord] = field(default_factory=list)
    trust_incoming: list[TrustRecord] = field(default_factory=list)
    audit_events: list[AuditRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert entire report to dictionary for JSON export."""
        return {
            "metadata": self.metadata.to_dict(),
            "beliefs": [asdict(b) for b in self.beliefs],
            "shares_sent": [asdict(s) for s in self.shares_sent],
            "shares_received": [asdict(s) for s in self.shares_received],
            "trust_outgoing": [asdict(t) for t in self.trust_outgoing],
            "trust_incoming": [asdict(t) for t in self.trust_incoming],
            "audit_events": [asdict(a) for a in self.audit_events],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string."""

        def serialize(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        return json.dumps(self.to_dict(), default=serialize, indent=indent)

    def to_csv(self) -> dict[str, str]:
        """Export report as multiple CSV strings (one per data type).

        Returns:
            Dictionary mapping section name to CSV string.
        """
        csvs = {}

        # Beliefs CSV
        if self.beliefs:
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "belief_id",
                    "content",
                    "confidence",
                    "domains",
                    "created_at",
                    "updated_at",
                    "metadata",
                ],
            )
            writer.writeheader()
            for belief in self.beliefs:
                row = asdict(belief)
                row["domains"] = ";".join(row["domains"])
                row["metadata"] = json.dumps(row["metadata"])
                row["created_at"] = row["created_at"].isoformat() if row["created_at"] else ""
                row["updated_at"] = row["updated_at"].isoformat() if row["updated_at"] else ""
                writer.writerow(row)
            csvs["beliefs"] = output.getvalue()

        # Shares sent CSV
        if self.shares_sent:
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "share_id",
                    "belief_id",
                    "sharer_did",
                    "recipient_did",
                    "created_at",
                    "policy_level",
                    "revoked",
                    "revoked_at",
                ],
            )
            writer.writeheader()
            for share in self.shares_sent:
                row = asdict(share)
                row["created_at"] = row["created_at"].isoformat() if row["created_at"] else ""
                row["revoked_at"] = row["revoked_at"].isoformat() if row["revoked_at"] else ""
                writer.writerow(row)
            csvs["shares_sent"] = output.getvalue()

        # Shares received CSV
        if self.shares_received:
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "share_id",
                    "belief_id",
                    "sharer_did",
                    "recipient_did",
                    "created_at",
                    "policy_level",
                    "revoked",
                    "revoked_at",
                ],
            )
            writer.writeheader()
            for share in self.shares_received:
                row = asdict(share)
                row["created_at"] = row["created_at"].isoformat() if row["created_at"] else ""
                row["revoked_at"] = row["revoked_at"].isoformat() if row["revoked_at"] else ""
                writer.writerow(row)
            csvs["shares_received"] = output.getvalue()

        # Trust outgoing CSV
        if self.trust_outgoing:
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "source_did",
                    "target_did",
                    "competence",
                    "integrity",
                    "confidentiality",
                    "judgment",
                    "domain",
                    "created_at",
                    "updated_at",
                    "expires_at",
                ],
            )
            writer.writeheader()
            for trust in self.trust_outgoing:
                row = asdict(trust)
                row["created_at"] = row["created_at"].isoformat() if row["created_at"] else ""
                row["updated_at"] = row["updated_at"].isoformat() if row["updated_at"] else ""
                row["expires_at"] = row["expires_at"].isoformat() if row["expires_at"] else ""
                writer.writerow(row)
            csvs["trust_outgoing"] = output.getvalue()

        # Trust incoming CSV
        if self.trust_incoming:
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "source_did",
                    "target_did",
                    "competence",
                    "integrity",
                    "confidentiality",
                    "judgment",
                    "domain",
                    "created_at",
                    "updated_at",
                    "expires_at",
                ],
            )
            writer.writeheader()
            for trust in self.trust_incoming:
                row = asdict(trust)
                row["created_at"] = row["created_at"].isoformat() if row["created_at"] else ""
                row["updated_at"] = row["updated_at"].isoformat() if row["updated_at"] else ""
                row["expires_at"] = row["expires_at"].isoformat() if row["expires_at"] else ""
                writer.writerow(row)
            csvs["trust_incoming"] = output.getvalue()

        # Audit events CSV
        if self.audit_events:
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "event_id",
                    "event_type",
                    "actor_did",
                    "target_did",
                    "resource",
                    "action",
                    "success",
                    "timestamp",
                    "metadata",
                ],
            )
            writer.writeheader()
            for event in self.audit_events:
                row = asdict(event)
                row["timestamp"] = row["timestamp"].isoformat() if row["timestamp"] else ""
                row["metadata"] = json.dumps(row["metadata"])
                writer.writerow(row)
            csvs["audit_events"] = output.getvalue()

        return csvs


class ReportDataSource(Protocol):
    """Protocol for data sources used by the report generator.

    This abstraction allows the report generator to work with any
    database implementation that provides these methods.
    """

    async def get_beliefs_for_user(
        self,
        user_did: str,
        domains: Optional[list[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AsyncIterator[BeliefRecord]:
        """Get all beliefs owned by a user."""
        raise NotImplementedError
        yield  # Makes this an async generator

    async def get_shares_sent(
        self,
        user_did: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AsyncIterator[ShareRecord]:
        """Get all shares sent by a user."""
        raise NotImplementedError
        yield  # Makes this an async generator

    async def get_shares_received(
        self,
        user_did: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AsyncIterator[ShareRecord]:
        """Get all shares received by a user."""
        raise NotImplementedError
        yield  # Makes this an async generator

    async def get_trust_edges_from(
        self,
        user_did: str,
        domains: Optional[list[str]] = None,
    ) -> AsyncIterator[TrustRecord]:
        """Get all trust edges where user is the source."""
        raise NotImplementedError
        yield  # Makes this an async generator

    async def get_trust_edges_to(
        self,
        user_did: str,
        domains: Optional[list[str]] = None,
    ) -> AsyncIterator[TrustRecord]:
        """Get all trust edges where user is the target."""
        raise NotImplementedError
        yield  # Makes this an async generator

    async def get_audit_events_for_user(
        self,
        user_did: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AsyncIterator[AuditRecord]:
        """Get all audit events where user is actor or target."""
        raise NotImplementedError
        yield  # Makes this an async generator


class ReportStore(Protocol):
    """Protocol for storing report metadata and generated reports."""

    async def save_metadata(self, metadata: ReportMetadata) -> None:
        """Save report metadata."""
        ...

    async def get_metadata(self, report_id: str) -> Optional[ReportMetadata]:
        """Get report metadata by ID."""
        ...

    async def update_status(
        self,
        report_id: str,
        status: ReportStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """Update report status."""
        ...

    async def save_report(self, report_id: str, report: DataReport) -> None:
        """Save a generated report."""
        ...

    async def get_report(self, report_id: str) -> Optional[DataReport]:
        """Get a generated report by ID."""
        ...

    async def list_reports_for_user(self, user_did: str) -> list[ReportMetadata]:
        """List all reports for a user."""
        ...

    async def delete_report(self, report_id: str) -> bool:
        """Delete a report. Returns True if deleted."""
        ...


class InMemoryReportStore:
    """In-memory implementation of ReportStore for testing."""

    def __init__(self) -> None:
        self._metadata: dict[str, ReportMetadata] = {}
        self._reports: dict[str, DataReport] = {}

    async def save_metadata(self, metadata: ReportMetadata) -> None:
        """Save report metadata."""
        self._metadata[metadata.report_id] = metadata

    async def get_metadata(self, report_id: str) -> Optional[ReportMetadata]:
        """Get report metadata by ID."""
        return self._metadata.get(report_id)

    async def update_status(
        self,
        report_id: str,
        status: ReportStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """Update report status."""
        if report_id in self._metadata:
            self._metadata[report_id].status = status
            if error_message:
                self._metadata[report_id].error_message = error_message
            if status == ReportStatus.COMPLETED:
                self._metadata[report_id].generated_at = datetime.now(UTC)

    async def save_report(self, report_id: str, report: DataReport) -> None:
        """Save a generated report."""
        self._reports[report_id] = report
        # Update size estimate
        if report_id in self._metadata:
            json_str = report.to_json()
            self._metadata[report_id].size_bytes = len(json_str.encode("utf-8"))

    async def get_report(self, report_id: str) -> Optional[DataReport]:
        """Get a generated report by ID."""
        return self._reports.get(report_id)

    async def list_reports_for_user(self, user_did: str) -> list[ReportMetadata]:
        """List all reports for a user."""
        return [m for m in self._metadata.values() if m.user_did == user_did]

    async def delete_report(self, report_id: str) -> bool:
        """Delete a report. Returns True if deleted."""
        deleted = False
        if report_id in self._metadata:
            del self._metadata[report_id]
            deleted = True
        if report_id in self._reports:
            del self._reports[report_id]
            deleted = True
        return deleted


class InMemoryDataSource:
    """In-memory data source for testing."""

    def __init__(self) -> None:
        self.beliefs: list[BeliefRecord] = []
        self.shares: list[ShareRecord] = []
        self.trust_edges: list[TrustRecord] = []
        self.audit_events: list[AuditRecord] = []

    async def get_beliefs_for_user(
        self,
        user_did: str,
        domains: Optional[list[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AsyncIterator[BeliefRecord]:
        """Get beliefs for a user (in-memory, all beliefs assumed owned by queried user for testing)."""
        for belief in self.beliefs:
            # Filter by domain if specified
            if domains and not any(d in belief.domains for d in domains):
                continue
            # Filter by date range
            if start_date and belief.created_at < start_date:
                continue
            if end_date and belief.created_at > end_date:
                continue
            yield belief

    async def get_shares_sent(
        self,
        user_did: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AsyncIterator[ShareRecord]:
        """Get shares sent by user."""
        for share in self.shares:
            if share.sharer_did != user_did:
                continue
            if start_date and share.created_at < start_date:
                continue
            if end_date and share.created_at > end_date:
                continue
            yield share

    async def get_shares_received(
        self,
        user_did: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AsyncIterator[ShareRecord]:
        """Get shares received by user."""
        for share in self.shares:
            if share.recipient_did != user_did:
                continue
            if start_date and share.created_at < start_date:
                continue
            if end_date and share.created_at > end_date:
                continue
            yield share

    async def get_trust_edges_from(
        self,
        user_did: str,
        domains: Optional[list[str]] = None,
    ) -> AsyncIterator[TrustRecord]:
        """Get trust edges from user."""
        for edge in self.trust_edges:
            if edge.source_did != user_did:
                continue
            if domains and edge.domain not in domains:
                continue
            yield edge

    async def get_trust_edges_to(
        self,
        user_did: str,
        domains: Optional[list[str]] = None,
    ) -> AsyncIterator[TrustRecord]:
        """Get trust edges to user."""
        for edge in self.trust_edges:
            if edge.target_did != user_did:
                continue
            if domains and edge.domain not in domains:
                continue
            yield edge

    async def get_audit_events_for_user(
        self,
        user_did: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AsyncIterator[AuditRecord]:
        """Get audit events involving user."""
        for event in self.audit_events:
            if event.actor_did != user_did and event.target_did != user_did:
                continue
            if start_date and event.timestamp < start_date:
                continue
            if end_date and event.timestamp > end_date:
                continue
            yield event


class ReportError(Exception):
    """Base exception for report generation errors."""

    pass


class ReportNotFoundError(ReportError):
    """Raised when a requested report does not exist."""

    pass


class ReportGenerationError(ReportError):
    """Raised when report generation fails."""

    pass


class ReportService:
    """Service for generating and managing data reports.

    Provides async report generation with status tracking for large reports.
    """

    def __init__(
        self,
        data_source: ReportDataSource,
        report_store: ReportStore,
    ) -> None:
        self._data_source = data_source
        self._report_store = report_store

    async def request_report(
        self,
        user_did: str,
        scope: Optional[ReportScope] = None,
        format: ExportFormat = ExportFormat.JSON,
        expires_in_hours: int = 24,
    ) -> ReportMetadata:
        """Request a new data report.

        Creates the report metadata and returns immediately. The report
        will be generated asynchronously.

        Args:
            user_did: DID of the user requesting their data
            scope: What data to include (defaults to everything)
            format: Export format (JSON or CSV)
            expires_in_hours: Hours until report auto-expires

        Returns:
            ReportMetadata with report_id for tracking
        """
        if scope is None:
            scope = ReportScope()

        now = datetime.now(UTC)
        from datetime import timedelta

        metadata = ReportMetadata(
            report_id=str(uuid.uuid4()),
            user_did=user_did,
            requested_at=now,
            scope=scope,
            format=format,
            status=ReportStatus.PENDING,
            expires_at=now + timedelta(hours=expires_in_hours),
        )

        await self._report_store.save_metadata(metadata)
        logger.info(f"Report requested: {metadata.report_id} for {user_did}")

        return metadata

    async def generate_report(self, report_id: str) -> DataReport:
        """Generate a data report.

        This is the main generation method that collects all data.
        For large reports, this should be called in a background task.

        Args:
            report_id: ID of the report to generate

        Returns:
            The completed DataReport

        Raises:
            ReportNotFoundError: If report_id doesn't exist
            ReportGenerationError: If generation fails
        """
        metadata = await self._report_store.get_metadata(report_id)
        if metadata is None:
            raise ReportNotFoundError(f"Report {report_id} not found")

        # Update status to generating
        await self._report_store.update_status(report_id, ReportStatus.GENERATING)

        try:
            report = DataReport(metadata=metadata)
            scope = metadata.scope
            user_did = metadata.user_did

            # Collect beliefs
            if scope.include_beliefs:
                async for belief in self._data_source.get_beliefs_for_user(
                    user_did,
                    domains=scope.domains or None,
                    start_date=scope.start_date,
                    end_date=scope.end_date,
                ):
                    report.beliefs.append(belief)

            # Collect shares sent
            if scope.include_shares_sent:
                async for share in self._data_source.get_shares_sent(
                    user_did,
                    start_date=scope.start_date,
                    end_date=scope.end_date,
                ):
                    report.shares_sent.append(share)

            # Collect shares received
            if scope.include_shares_received:
                async for share in self._data_source.get_shares_received(
                    user_did,
                    start_date=scope.start_date,
                    end_date=scope.end_date,
                ):
                    report.shares_received.append(share)

            # Collect trust edges (outgoing)
            if scope.include_trust_outgoing:
                async for edge in self._data_source.get_trust_edges_from(
                    user_did,
                    domains=scope.domains or None,
                ):
                    report.trust_outgoing.append(edge)

            # Collect trust edges (incoming)
            if scope.include_trust_incoming:
                async for edge in self._data_source.get_trust_edges_to(
                    user_did,
                    domains=scope.domains or None,
                ):
                    report.trust_incoming.append(edge)

            # Collect audit events
            if scope.include_audit_events:
                async for event in self._data_source.get_audit_events_for_user(
                    user_did,
                    start_date=scope.start_date,
                    end_date=scope.end_date,
                ):
                    report.audit_events.append(event)

            # Update record counts
            metadata.record_counts = {
                "beliefs": len(report.beliefs),
                "shares_sent": len(report.shares_sent),
                "shares_received": len(report.shares_received),
                "trust_outgoing": len(report.trust_outgoing),
                "trust_incoming": len(report.trust_incoming),
                "audit_events": len(report.audit_events),
            }

            # Save the report
            await self._report_store.save_report(report_id, report)
            await self._report_store.update_status(report_id, ReportStatus.COMPLETED)

            logger.info(
                f"Report {report_id} generated with "
                f"{sum(metadata.record_counts.values())} total records"
            )

            return report

        except Exception as e:  # Intentionally broad: wrap all failures, update status, re-raise
            logger.error(f"Report generation failed for {report_id}: {e}")
            await self._report_store.update_status(report_id, ReportStatus.FAILED, str(e))
            raise ReportGenerationError(f"Failed to generate report: {e}") from e

    async def get_report_status(self, report_id: str) -> ReportMetadata:
        """Get the current status of a report.

        Args:
            report_id: ID of the report

        Returns:
            ReportMetadata with current status

        Raises:
            ReportNotFoundError: If report doesn't exist
        """
        metadata = await self._report_store.get_metadata(report_id)
        if metadata is None:
            raise ReportNotFoundError(f"Report {report_id} not found")
        return metadata

    async def get_report(self, report_id: str) -> DataReport:
        """Get a completed report.

        Args:
            report_id: ID of the report

        Returns:
            The DataReport if completed

        Raises:
            ReportNotFoundError: If report doesn't exist
            ReportGenerationError: If report is not completed
        """
        metadata = await self._report_store.get_metadata(report_id)
        if metadata is None:
            raise ReportNotFoundError(f"Report {report_id} not found")

        if metadata.status != ReportStatus.COMPLETED:
            raise ReportGenerationError(
                f"Report {report_id} is not completed (status: {metadata.status.value})"
            )

        report = await self._report_store.get_report(report_id)
        if report is None:
            raise ReportNotFoundError(f"Report data not found for {report_id}")

        return report

    async def list_reports(self, user_did: str) -> list[ReportMetadata]:
        """List all reports for a user.

        Args:
            user_did: DID of the user

        Returns:
            List of ReportMetadata for all user's reports
        """
        return await self._report_store.list_reports_for_user(user_did)

    async def delete_report(self, report_id: str, user_did: str) -> bool:
        """Delete a report.

        Args:
            report_id: ID of the report to delete
            user_did: DID of the requesting user (must own the report)

        Returns:
            True if deleted, False if not found

        Raises:
            ReportError: If user doesn't own the report
        """
        metadata = await self._report_store.get_metadata(report_id)
        if metadata is None:
            return False

        if metadata.user_did != user_did:
            raise ReportError("Cannot delete report owned by another user")

        return await self._report_store.delete_report(report_id)


# Module-level service instance for convenience
_default_service: Optional[ReportService] = None


def get_report_service() -> Optional[ReportService]:
    """Get the default report service instance."""
    return _default_service


def set_report_service(service: Optional[ReportService]) -> None:
    """Set the default report service instance."""
    global _default_service
    _default_service = service


async def generate_data_report(
    user_did: str,
    scope: Optional[ReportScope] = None,
    format: ExportFormat = ExportFormat.JSON,
) -> DataReport:
    """Convenience function to generate a data report synchronously.

    This is the primary API for generating reports. It requests a report,
    generates it immediately, and returns the result.

    For very large reports, use ReportService.request_report() followed by
    ReportService.generate_report() in a background task.

    Args:
        user_did: DID of the user requesting their data
        scope: What data to include (defaults to everything)
        format: Export format (JSON or CSV)

    Returns:
        The completed DataReport

    Raises:
        ReportError: If no report service is configured
        ReportGenerationError: If generation fails
    """
    service = get_report_service()
    if service is None:
        raise ReportError("No report service configured. Call set_report_service() first.")

    metadata = await service.request_report(user_did, scope, format)
    return await service.generate_report(metadata.report_id)
