"""Tests for trust concentration warnings.

Tests cover:
- Single node dominance detection (>30% threshold)
- Top 3 nodes dominance detection (>50% threshold)
- Few trusted sources warning (<3 sources)
- Gini coefficient calculation
- Custom threshold support
- Warning severity levels
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock, patch
from uuid import uuid4, UUID

import pytest

from valence.federation.trust_policy import (
    TrustPolicy,
    CONCENTRATION_THRESHOLDS,
)
from valence.federation.models import (
    WarningSeverity,
    TrustConcentrationWarning,
    TrustConcentrationReport,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_cursor():
    """Mock database cursor."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Mock the get_cursor context manager."""
    @contextmanager
    def _mock_get_cursor(dict_cursor: bool = True) -> Generator:
        yield mock_cursor

    with patch("valence.federation.trust_policy.get_cursor", _mock_get_cursor):
        yield mock_cursor


@pytest.fixture
def mock_registry():
    """Mock TrustRegistry."""
    return MagicMock()


@pytest.fixture
def trust_policy(mock_registry):
    """Create a TrustPolicy instance with mocked registry."""
    return TrustPolicy(registry=mock_registry)


def make_node_row(
    node_id: UUID | None = None,
    name: str | None = None,
    status: str = "active",
    trust_score: float = 0.5,
) -> dict:
    """Create a mock node row."""
    return {
        "id": node_id or uuid4(),
        "name": name,
        "status": status,
        "trust_score": trust_score,
    }


# =============================================================================
# UNIT TESTS - TrustConcentrationWarning
# =============================================================================


class TestTrustConcentrationWarning:
    """Tests for TrustConcentrationWarning dataclass."""

    def test_warning_creation(self):
        """Test basic warning creation."""
        warning = TrustConcentrationWarning(
            warning_type="single_node_dominant",
            severity=WarningSeverity.WARNING,
            message="Test warning message",
        )
        
        assert warning.warning_type == "single_node_dominant"
        assert warning.severity == WarningSeverity.WARNING
        assert warning.message == "Test warning message"
        assert warning.node_id is None
        assert warning.trust_share is None

    def test_warning_with_node_details(self):
        """Test warning with node information."""
        node_id = uuid4()
        warning = TrustConcentrationWarning(
            warning_type="single_node_dominant",
            severity=WarningSeverity.CRITICAL,
            message="Node dominates trust",
            node_id=node_id,
            node_name="alice",
            trust_share=0.55,
            recommendation="Reduce trust concentration",
        )
        
        assert warning.node_id == node_id
        assert warning.node_name == "alice"
        assert warning.trust_share == 0.55
        assert warning.recommendation == "Reduce trust concentration"

    def test_warning_to_dict(self):
        """Test warning serialization."""
        node_id = uuid4()
        warning = TrustConcentrationWarning(
            warning_type="few_sources",
            severity=WarningSeverity.WARNING,
            message="Few sources",
            node_id=node_id,
            trust_share=0.33,
        )
        
        data = warning.to_dict()
        
        assert data["warning_type"] == "few_sources"
        assert data["severity"] == "warning"
        assert data["node_id"] == str(node_id)
        assert data["trust_share"] == 0.33

    def test_warning_str_representation(self):
        """Test human-readable string format."""
        warning = TrustConcentrationWarning(
            warning_type="single_node_dominant",
            severity=WarningSeverity.CRITICAL,
            message="Single node holds 55% of trust",
        )
        
        result = str(warning)
        assert "CRITICAL" in result
        assert "55%" in result
        assert "🚨" in result  # Critical icon

    def test_warning_severity_icons(self):
        """Test different severity icons."""
        info_warning = TrustConcentrationWarning(
            warning_type="info",
            severity=WarningSeverity.INFO,
            message="Info message",
        )
        warning_warning = TrustConcentrationWarning(
            warning_type="warning",
            severity=WarningSeverity.WARNING,
            message="Warning message",
        )
        
        assert "ℹ️" in str(info_warning)
        assert "⚠️" in str(warning_warning)


# =============================================================================
# UNIT TESTS - TrustConcentrationReport
# =============================================================================


class TestTrustConcentrationReport:
    """Tests for TrustConcentrationReport dataclass."""

    def test_empty_report(self):
        """Test report with no warnings."""
        report = TrustConcentrationReport(
            total_nodes=10,
            active_nodes=8,
            total_trust=5.0,
        )
        
        assert report.total_nodes == 10
        assert report.has_warnings is False
        assert report.has_critical_warnings is False
        assert report.max_severity is None

    def test_report_with_warnings(self):
        """Test report with multiple warnings."""
        warnings = [
            TrustConcentrationWarning(
                warning_type="single_node_dominant",
                severity=WarningSeverity.WARNING,
                message="Warning 1",
            ),
            TrustConcentrationWarning(
                warning_type="few_sources",
                severity=WarningSeverity.CRITICAL,
                message="Warning 2",
            ),
        ]
        
        report = TrustConcentrationReport(
            warnings=warnings,
            total_nodes=5,
            active_nodes=5,
        )
        
        assert report.has_warnings is True
        assert report.has_critical_warnings is True
        assert report.max_severity == WarningSeverity.CRITICAL

    def test_report_to_dict(self):
        """Test report serialization."""
        report = TrustConcentrationReport(
            total_nodes=10,
            active_nodes=8,
            total_trust=5.0,
            top_node_share=0.35,
            top_3_share=0.60,
            trusted_sources=5,
            gini_coefficient=0.45,
        )
        
        data = report.to_dict()
        
        assert data["metrics"]["total_nodes"] == 10
        assert data["metrics"]["top_node_share"] == 0.35
        assert data["metrics"]["gini_coefficient"] == 0.45
        assert data["has_warnings"] is False

    def test_report_max_severity_warning_only(self):
        """Test max severity with only warning level."""
        warnings = [
            TrustConcentrationWarning(
                warning_type="test",
                severity=WarningSeverity.INFO,
                message="Info",
            ),
            TrustConcentrationWarning(
                warning_type="test",
                severity=WarningSeverity.WARNING,
                message="Warning",
            ),
        ]
        
        report = TrustConcentrationReport(warnings=warnings)
        
        assert report.max_severity == WarningSeverity.WARNING


# =============================================================================
# UNIT TESTS - Gini Coefficient
# =============================================================================


class TestGiniCoefficient:
    """Tests for Gini coefficient calculation."""

    def test_gini_perfect_equality(self, trust_policy):
        """Test Gini = 0 for perfect equality."""
        # All equal values
        values = [1.0, 1.0, 1.0, 1.0, 1.0]
        gini = trust_policy._calculate_gini(values)
        
        assert gini is not None
        assert abs(gini - 0.0) < 0.01  # Should be close to 0

    def test_gini_high_inequality(self, trust_policy):
        """Test high Gini for concentrated distribution."""
        # One dominant value
        values = [0.1, 0.1, 0.1, 0.1, 10.0]
        gini = trust_policy._calculate_gini(values)
        
        assert gini is not None
        assert gini > 0.5  # Should be high

    def test_gini_empty_list(self, trust_policy):
        """Test Gini with too few values."""
        assert trust_policy._calculate_gini([]) is None
        assert trust_policy._calculate_gini([1.0]) is None

    def test_gini_all_zeros(self, trust_policy):
        """Test Gini with all zero values."""
        values = [0.0, 0.0, 0.0]
        gini = trust_policy._calculate_gini(values)
        
        assert gini == 0.0


# =============================================================================
# INTEGRATION TESTS - check_trust_concentration
# =============================================================================


class TestCheckTrustConcentration:
    """Tests for check_trust_concentration method."""

    def test_no_nodes(self, trust_policy, mock_get_cursor):
        """Test with empty network."""
        mock_get_cursor.fetchall.return_value = []
        
        report = trust_policy.check_trust_concentration()
        
        assert report.total_nodes == 0
        assert report.has_warnings is False

    def test_healthy_network(self, trust_policy, mock_get_cursor):
        """Test healthy network with distributed trust."""
        # 6 nodes with evenly distributed trust (each ~16.7%)
        # Top 3 will be ~50%, so just below threshold
        mock_get_cursor.fetchall.return_value = [
            make_node_row(name="node1", trust_score=0.35),
            make_node_row(name="node2", trust_score=0.33),
            make_node_row(name="node3", trust_score=0.32),
            make_node_row(name="node4", trust_score=0.35),
            make_node_row(name="node5", trust_score=0.33),
            make_node_row(name="node6", trust_score=0.32),
        ]
        # Total = 2.0, top 3 = 1.03 = 51.5% - still over 50%
        # Let's use 7 nodes to get under 50%
        mock_get_cursor.fetchall.return_value = [
            make_node_row(name="node1", trust_score=0.3),
            make_node_row(name="node2", trust_score=0.28),
            make_node_row(name="node3", trust_score=0.26),
            make_node_row(name="node4", trust_score=0.29),
            make_node_row(name="node5", trust_score=0.27),
            make_node_row(name="node6", trust_score=0.30),
            make_node_row(name="node7", trust_score=0.30),
        ]
        # Total = 2.0, top 3 = 0.3+0.30+0.30 = 0.90 = 45%
        
        report = trust_policy.check_trust_concentration()
        
        assert report.total_nodes == 7
        assert report.trusted_sources == 7
        assert report.has_warnings is False
        assert report.top_node_share < 0.30  # Under threshold

    def test_single_node_dominant_warning(self, trust_policy, mock_get_cursor):
        """Test warning when single node dominates."""
        # One node with 35% of trust (above 30% warning threshold)
        mock_get_cursor.fetchall.return_value = [
            make_node_row(name="dominant", trust_score=0.7),  # 35% of 2.0 total
            make_node_row(name="node2", trust_score=0.5),
            make_node_row(name="node3", trust_score=0.4),
            make_node_row(name="node4", trust_score=0.3),
            make_node_row(name="node5", trust_score=0.1),
        ]
        
        report = trust_policy.check_trust_concentration()
        
        assert report.has_warnings is True
        
        # Find the single node warning
        single_warning = next(
            (w for w in report.warnings if w.warning_type == "single_node_dominant"),
            None
        )
        assert single_warning is not None
        assert single_warning.severity == WarningSeverity.WARNING
        assert single_warning.node_name == "dominant"

    def test_single_node_dominant_critical(self, trust_policy, mock_get_cursor):
        """Test critical warning when single node is very dominant."""
        # One node with >50% of trust (critical threshold)
        mock_get_cursor.fetchall.return_value = [
            make_node_row(name="dominant", trust_score=1.5),  # 60% of 2.5 total
            make_node_row(name="node2", trust_score=0.5),
            make_node_row(name="node3", trust_score=0.3),
            make_node_row(name="node4", trust_score=0.1),
            make_node_row(name="node5", trust_score=0.1),
        ]
        
        report = trust_policy.check_trust_concentration()
        
        assert report.has_critical_warnings is True
        
        single_warning = next(
            (w for w in report.warnings if w.warning_type == "single_node_dominant"),
            None
        )
        assert single_warning is not None
        assert single_warning.severity == WarningSeverity.CRITICAL

    def test_top_3_dominant_warning(self, trust_policy, mock_get_cursor):
        """Test warning when top 3 nodes dominate."""
        # Top 3 nodes hold >50% of trust
        mock_get_cursor.fetchall.return_value = [
            make_node_row(name="node1", trust_score=0.5),   # 25%
            make_node_row(name="node2", trust_score=0.4),   # 20%
            make_node_row(name="node3", trust_score=0.35),  # 17.5% (total 62.5% for top 3)
            make_node_row(name="node4", trust_score=0.4),
            make_node_row(name="node5", trust_score=0.35),
        ]
        
        report = trust_policy.check_trust_concentration()
        
        top3_warning = next(
            (w for w in report.warnings if w.warning_type == "top_nodes_dominant"),
            None
        )
        # Top 3 share is 0.5+0.4+0.4 / 2.0 = 0.65 > 50%
        # Actually let me recalculate: 0.5+0.4+0.35 / (0.5+0.4+0.35+0.4+0.35) = 1.25/2.0 = 62.5%
        assert top3_warning is not None

    def test_few_sources_warning(self, trust_policy, mock_get_cursor):
        """Test warning when there are too few trusted sources."""
        # Only 2 nodes with trust above threshold
        mock_get_cursor.fetchall.return_value = [
            make_node_row(name="node1", trust_score=0.5),
            make_node_row(name="node2", trust_score=0.3),
            make_node_row(name="node3", trust_score=0.05),  # Below 0.1 threshold
            make_node_row(name="node4", trust_score=0.02),  # Below threshold
        ]
        
        report = trust_policy.check_trust_concentration()
        
        assert report.trusted_sources == 2
        
        few_sources_warning = next(
            (w for w in report.warnings if w.warning_type == "few_sources"),
            None
        )
        assert few_sources_warning is not None
        assert "2" in few_sources_warning.message

    def test_few_sources_critical(self, trust_policy, mock_get_cursor):
        """Test critical warning when only 1 trusted source."""
        mock_get_cursor.fetchall.return_value = [
            make_node_row(name="only_one", trust_score=0.8),
            make_node_row(name="node2", trust_score=0.05),
            make_node_row(name="node3", trust_score=0.01),
        ]
        
        report = trust_policy.check_trust_concentration()
        
        assert report.trusted_sources == 1
        
        few_sources_warning = next(
            (w for w in report.warnings if w.warning_type == "few_sources"),
            None
        )
        assert few_sources_warning is not None
        assert few_sources_warning.severity == WarningSeverity.CRITICAL

    def test_custom_thresholds(self, trust_policy, mock_get_cursor):
        """Test with custom threshold values."""
        # Node has 25% share - below default 30% but above custom 20%
        mock_get_cursor.fetchall.return_value = [
            make_node_row(name="node1", trust_score=0.5),   # 25%
            make_node_row(name="node2", trust_score=0.5),   # 25%
            make_node_row(name="node3", trust_score=0.5),   # 25%
            make_node_row(name="node4", trust_score=0.5),   # 25%
        ]
        
        # With default thresholds - no warning
        report = trust_policy.check_trust_concentration()
        single_warnings = [w for w in report.warnings if w.warning_type == "single_node_dominant"]
        assert len(single_warnings) == 0
        
        # With stricter custom threshold - should warn
        custom_thresholds = {
            "single_node_warning": 0.20,  # 20% instead of 30%
            "single_node_critical": 0.50,
            "top_3_warning": 0.50,
            "top_3_critical": 0.70,
            "min_trusted_sources": 3,
            "min_trust_to_count": 0.1,
        }
        report = trust_policy.check_trust_concentration(thresholds=custom_thresholds)
        single_warnings = [w for w in report.warnings if w.warning_type == "single_node_dominant"]
        assert len(single_warnings) == 1

    def test_db_error_handling(self, trust_policy, mock_get_cursor):
        """Test graceful handling of database errors."""
        mock_get_cursor.execute.side_effect = Exception("Database error")
        
        report = trust_policy.check_trust_concentration()
        
        assert len(report.warnings) == 1
        assert "error" in report.warnings[0].warning_type
        assert "Database error" in report.warnings[0].message

    def test_unreachable_nodes_counted(self, trust_policy, mock_get_cursor):
        """Test that unreachable nodes are counted but marked."""
        mock_get_cursor.fetchall.return_value = [
            make_node_row(name="active1", status="active", trust_score=0.5),
            make_node_row(name="active2", status="active", trust_score=0.4),
            make_node_row(name="unreachable", status="unreachable", trust_score=0.3),
        ]
        
        report = trust_policy.check_trust_concentration()
        
        assert report.total_nodes == 3
        assert report.active_nodes == 2


# =============================================================================
# INTEGRATION TESTS - CLI
# =============================================================================


class TestTrustCheckCLI:
    """Tests for trust check CLI command."""

    def test_cli_parses_check_command(self):
        """Test that CLI parses trust check command."""
        from valence.cli.main import app
        
        parser = app()
        args = parser.parse_args(['trust', 'check'])
        
        assert args.command == 'trust'
        assert args.trust_command == 'check'
        assert args.json is False

    def test_cli_parses_json_flag(self):
        """Test that CLI parses --json flag."""
        from valence.cli.main import app
        
        parser = app()
        args = parser.parse_args(['trust', 'check', '--json'])
        
        assert args.json is True

    def test_cli_parses_custom_thresholds(self):
        """Test that CLI parses custom threshold arguments."""
        from valence.cli.main import app
        
        parser = app()
        args = parser.parse_args([
            'trust', 'check',
            '--single-threshold', '0.25',
            '--top3-threshold', '0.45',
            '--min-sources', '5',
        ])
        
        assert args.single_threshold == 0.25
        assert args.top3_threshold == 0.45
        assert args.min_sources == 5


# =============================================================================
# CONSTANTS TESTS
# =============================================================================


class TestConcentrationThresholds:
    """Tests for concentration threshold constants."""

    def test_default_thresholds_exist(self):
        """Test that all required thresholds are defined."""
        assert "single_node_warning" in CONCENTRATION_THRESHOLDS
        assert "single_node_critical" in CONCENTRATION_THRESHOLDS
        assert "top_3_warning" in CONCENTRATION_THRESHOLDS
        assert "top_3_critical" in CONCENTRATION_THRESHOLDS
        assert "min_trusted_sources" in CONCENTRATION_THRESHOLDS
        assert "min_trust_to_count" in CONCENTRATION_THRESHOLDS

    def test_thresholds_are_sensible(self):
        """Test that threshold values make sense."""
        # Warning should be lower than critical
        assert CONCENTRATION_THRESHOLDS["single_node_warning"] < CONCENTRATION_THRESHOLDS["single_node_critical"]
        assert CONCENTRATION_THRESHOLDS["top_3_warning"] < CONCENTRATION_THRESHOLDS["top_3_critical"]
        
        # All should be between 0 and 1
        for key, value in CONCENTRATION_THRESHOLDS.items():
            if key != "min_trusted_sources":
                assert 0 <= value <= 1, f"{key} should be between 0 and 1"
