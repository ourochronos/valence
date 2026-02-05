"""Tests for gateway nodes for external federation sharing.

Tests cover:
- GatewayNode initialization and configuration
- Inbound share handling with validation and routing
- Outbound share handling
- Rate limiting
- Access control
- Audit logging
- Gateway registry

Issue #88: Implement gateway nodes for external sharing
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID

import pytest

from valence.federation.gateway import (
    # Enums
    GatewayCapability,
    GatewayStatus,
    AuditEventType,
    ShareDirection,
    ValidationResult,
    # Exceptions
    GatewayException,
    RateLimitException,
    AccessDeniedException,
    ValidationFailedException,
    # Data classes
    GatewayConfig,
    RateLimitState,
    AuditEntry,
    InboundShare,
    OutboundShare,
    ShareResult,
    # Main classes
    GatewayNode,
    GatewayRegistry,
    # Functions
    get_gateway_registry,
    create_gateway,
    # Constants
    DEFAULT_RATE_LIMIT_WINDOW,
    DEFAULT_RATE_LIMIT_MAX_REQUESTS,
    DEFAULT_RATE_LIMIT_MAX_BELIEFS,
    MIN_TRUST_FOR_GATEWAY,
)
from valence.federation.models import FederatedBelief, Visibility
from valence.federation.protocol import ErrorCode


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def federation_id() -> UUID:
    """Test federation ID."""
    return uuid4()


@pytest.fixture
def external_federation_id() -> UUID:
    """External federation ID."""
    return uuid4()


@pytest.fixture
def gateway_endpoint() -> str:
    """Test gateway endpoint."""
    return "https://gateway.example.com/v1/federation"


@pytest.fixture
def gateway_config() -> GatewayConfig:
    """Test gateway configuration."""
    return GatewayConfig(
        rate_limit_window=60,
        rate_limit_max_requests=10,
        rate_limit_max_beliefs=100,
        min_trust_for_access=0.3,
        audit_enabled=True,
    )


@pytest.fixture
def gateway(federation_id: UUID, gateway_endpoint: str, gateway_config: GatewayConfig) -> GatewayNode:
    """Test gateway node."""
    return GatewayNode(
        federation_id=federation_id,
        endpoint=gateway_endpoint,
        config=gateway_config,
    )


@pytest.fixture
def mock_belief() -> FederatedBelief:
    """Mock federated belief."""
    from valence.core.confidence import DimensionalConfidence
    return FederatedBelief(
        id=uuid4(),
        federation_id=uuid4(),
        origin_node_did="did:vkb:web:test.example.com",
        content="Test belief content",
        confidence=DimensionalConfidence(overall=0.8),
        visibility=Visibility.FEDERATED,
    )


@pytest.fixture
def inbound_share(external_federation_id: UUID, mock_belief: FederatedBelief) -> InboundShare:
    """Test inbound share."""
    return InboundShare(
        source_federation_id=external_federation_id,
        source_federation_did="did:vkb:web:external.example.com",
        source_gateway_endpoint="https://external.example.com/gateway",
        beliefs=[mock_belief],
    )


@pytest.fixture
def outbound_share(external_federation_id: UUID, mock_belief: FederatedBelief) -> OutboundShare:
    """Test outbound share."""
    return OutboundShare(
        target_federation_id=external_federation_id,
        target_federation_did="did:vkb:web:external.example.com",
        target_gateway_endpoint="https://external.example.com/gateway",
        beliefs=[mock_belief],
    )


@pytest.fixture
def mock_cursor():
    """Mock database cursor."""
    cursor = MagicMock()
    cursor.fetchone.return_value = {"trust_score": 0.7}
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Mock the get_cursor context manager."""
    from contextlib import contextmanager
    
    @contextmanager
    def _mock_get_cursor(dict_cursor: bool = True):
        yield mock_cursor
    
    with patch("valence.federation.gateway.get_cursor", _mock_get_cursor):
        yield mock_cursor


# =============================================================================
# GATEWAY CAPABILITY TESTS
# =============================================================================


class TestGatewayCapability:
    """Tests for GatewayCapability enum."""
    
    def test_all_capabilities_defined(self):
        """Test all expected capabilities are defined."""
        expected = [
            "INBOUND_SHARE",
            "OUTBOUND_SHARE",
            "BELIEF_RELAY",
            "TRUST_BRIDGE",
            "PRIVACY_AGGREGATE",
            "QUERY_FORWARD",
        ]
        for cap in expected:
            assert hasattr(GatewayCapability, cap)
    
    def test_capability_values(self):
        """Test capability string values."""
        assert GatewayCapability.INBOUND_SHARE.value == "inbound_share"
        assert GatewayCapability.OUTBOUND_SHARE.value == "outbound_share"


class TestGatewayStatus:
    """Tests for GatewayStatus enum."""
    
    def test_all_statuses_defined(self):
        """Test all expected statuses are defined."""
        expected = [
            "INITIALIZING",
            "ACTIVE",
            "DEGRADED",
            "RATE_LIMITED",
            "SUSPENDED",
            "OFFLINE",
        ]
        for status in expected:
            assert hasattr(GatewayStatus, status)


class TestAuditEventType:
    """Tests for AuditEventType enum."""
    
    def test_all_event_types_defined(self):
        """Test all expected event types are defined."""
        expected = [
            "INBOUND_SHARE",
            "OUTBOUND_SHARE",
            "ACCESS_DENIED",
            "RATE_LIMITED",
            "VALIDATION_FAILED",
            "ROUTE_SUCCESS",
            "ROUTE_FAILED",
            "TRUST_CHECK",
            "CONFIG_CHANGE",
        ]
        for event in expected:
            assert hasattr(AuditEventType, event)


# =============================================================================
# GATEWAY CONFIG TESTS
# =============================================================================


class TestGatewayConfig:
    """Tests for GatewayConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GatewayConfig()
        
        assert config.rate_limit_window == DEFAULT_RATE_LIMIT_WINDOW
        assert config.rate_limit_max_requests == DEFAULT_RATE_LIMIT_MAX_REQUESTS
        assert config.rate_limit_max_beliefs == DEFAULT_RATE_LIMIT_MAX_BELIEFS
        assert config.min_trust_for_access == MIN_TRUST_FOR_GATEWAY
        assert config.audit_enabled is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        blocked = {uuid4(), uuid4()}
        config = GatewayConfig(
            rate_limit_window=120,
            rate_limit_max_requests=50,
            blocked_federations=blocked,
        )
        
        assert config.rate_limit_window == 120
        assert config.rate_limit_max_requests == 50
        assert config.blocked_federations == blocked
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = GatewayConfig()
        d = config.to_dict()
        
        assert "rate_limit_window" in d
        assert "enabled_capabilities" in d
        assert isinstance(d["enabled_capabilities"], list)


# =============================================================================
# RATE LIMIT STATE TESTS
# =============================================================================


class TestRateLimitState:
    """Tests for RateLimitState dataclass."""
    
    def test_initial_state(self):
        """Test initial rate limit state."""
        fed_id = uuid4()
        state = RateLimitState(federation_id=fed_id)
        
        assert state.federation_id == fed_id
        assert state.request_count == 0
        assert state.belief_count == 0
    
    def test_record_request(self):
        """Test recording requests."""
        state = RateLimitState(federation_id=uuid4())
        
        state.record_request()
        assert state.request_count == 1
        
        state.record_request()
        assert state.request_count == 2
    
    def test_record_beliefs(self):
        """Test recording beliefs."""
        state = RateLimitState(federation_id=uuid4())
        
        state.record_beliefs(5)
        assert state.belief_count == 5
        
        state.record_beliefs(3)
        assert state.belief_count == 8
    
    def test_is_request_allowed(self):
        """Test request allowance check."""
        state = RateLimitState(federation_id=uuid4())
        state.request_count = 9
        
        assert state.is_request_allowed(10) is True
        
        state.request_count = 10
        assert state.is_request_allowed(10) is False
    
    def test_is_belief_allowed(self):
        """Test belief allowance check."""
        state = RateLimitState(federation_id=uuid4())
        state.belief_count = 95
        
        assert state.is_belief_allowed(100, 5) is True
        assert state.is_belief_allowed(100, 6) is False
    
    def test_reset_if_expired(self):
        """Test window expiration reset."""
        state = RateLimitState(federation_id=uuid4())
        state.window_start = time.time() - 120  # 2 minutes ago
        state.request_count = 50
        state.belief_count = 500
        
        state.reset_if_expired(60)  # 1 minute window
        
        assert state.request_count == 0
        assert state.belief_count == 0


# =============================================================================
# AUDIT ENTRY TESTS
# =============================================================================


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""
    
    def test_audit_entry_creation(self):
        """Test creating an audit entry."""
        source_id = uuid4()
        belief_ids = [uuid4(), uuid4()]
        
        entry = AuditEntry(
            event_type=AuditEventType.INBOUND_SHARE,
            source_federation_id=source_id,
            direction=ShareDirection.INBOUND,
            belief_ids=belief_ids,
            success=True,
        )
        
        assert entry.event_type == AuditEventType.INBOUND_SHARE
        assert entry.source_federation_id == source_id
        assert entry.direction == ShareDirection.INBOUND
        assert len(entry.belief_ids) == 2
        assert entry.success is True
    
    def test_audit_entry_to_dict(self):
        """Test audit entry serialization."""
        entry = AuditEntry(
            event_type=AuditEventType.ACCESS_DENIED,
            error_code=ErrorCode.TRUST_INSUFFICIENT,
            error_message="Trust below threshold",
        )
        
        d = entry.to_dict()
        
        assert d["event_type"] == "access_denied"
        assert d["error_code"] == "TRUST_INSUFFICIENT"
        assert d["success"] is True  # Default


# =============================================================================
# INBOUND/OUTBOUND SHARE TESTS
# =============================================================================


class TestInboundShare:
    """Tests for InboundShare dataclass."""
    
    def test_inbound_share_creation(self, mock_belief: FederatedBelief):
        """Test creating an inbound share."""
        source_id = uuid4()
        share = InboundShare(
            source_federation_id=source_id,
            source_federation_did="did:vkb:web:example.com",
            source_gateway_endpoint="https://example.com/gateway",
            beliefs=[mock_belief],
        )
        
        assert share.source_federation_id == source_id
        assert len(share.beliefs) == 1
        assert share.received_at is not None


class TestOutboundShare:
    """Tests for OutboundShare dataclass."""
    
    def test_outbound_share_creation(self, mock_belief: FederatedBelief):
        """Test creating an outbound share."""
        target_id = uuid4()
        share = OutboundShare(
            target_federation_id=target_id,
            target_gateway_endpoint="https://target.com/gateway",
            beliefs=[mock_belief],
        )
        
        assert share.target_federation_id == target_id
        assert len(share.beliefs) == 1
        assert share.sent_at is None


class TestShareResult:
    """Tests for ShareResult dataclass."""
    
    def test_success_result(self):
        """Test successful share result."""
        result = ShareResult(
            success=True,
            share_id=uuid4(),
            belief_count=5,
        )
        
        assert result.success is True
        assert result.belief_count == 5
        assert result.error_code is None
    
    def test_failure_result(self):
        """Test failed share result."""
        result = ShareResult(
            success=False,
            error_code=ErrorCode.RATE_LIMITED,
            error_message="Rate limit exceeded",
        )
        
        assert result.success is False
        assert result.error_code == ErrorCode.RATE_LIMITED
    
    def test_result_to_dict(self):
        """Test result serialization."""
        share_id = uuid4()
        routed = [uuid4(), uuid4()]
        result = ShareResult(
            success=True,
            share_id=share_id,
            routed_to=routed,
        )
        
        d = result.to_dict()
        assert d["success"] is True
        assert d["share_id"] == str(share_id)
        assert len(d["routed_to"]) == 2


# =============================================================================
# GATEWAY NODE TESTS
# =============================================================================


class TestGatewayNodeInit:
    """Tests for GatewayNode initialization."""
    
    def test_basic_init(self, federation_id: UUID, gateway_endpoint: str):
        """Test basic gateway initialization."""
        gateway = GatewayNode(
            federation_id=federation_id,
            endpoint=gateway_endpoint,
        )
        
        assert gateway.federation_id == federation_id
        assert gateway.endpoint == gateway_endpoint
        assert gateway.status == GatewayStatus.INITIALIZING
        assert GatewayCapability.INBOUND_SHARE in gateway.capabilities
        assert GatewayCapability.OUTBOUND_SHARE in gateway.capabilities
    
    def test_init_with_capabilities(self, federation_id: UUID, gateway_endpoint: str):
        """Test initialization with custom capabilities."""
        gateway = GatewayNode(
            federation_id=federation_id,
            endpoint=gateway_endpoint,
            capabilities=[GatewayCapability.INBOUND_SHARE],
        )
        
        assert GatewayCapability.INBOUND_SHARE in gateway.capabilities
        assert GatewayCapability.OUTBOUND_SHARE not in gateway.capabilities
    
    def test_init_with_config(
        self,
        federation_id: UUID,
        gateway_endpoint: str,
        gateway_config: GatewayConfig,
    ):
        """Test initialization with custom config."""
        gateway = GatewayNode(
            federation_id=federation_id,
            endpoint=gateway_endpoint,
            config=gateway_config,
        )
        
        assert gateway.config.rate_limit_max_requests == 10
    
    def test_has_capability(self, gateway: GatewayNode):
        """Test capability check."""
        assert gateway.has_capability(GatewayCapability.INBOUND_SHARE) is True
        assert gateway.has_capability(GatewayCapability.TRUST_BRIDGE) is False


class TestGatewayNodeLifecycle:
    """Tests for GatewayNode lifecycle methods."""
    
    @pytest.mark.asyncio
    async def test_start(self, gateway: GatewayNode):
        """Test starting a gateway."""
        assert gateway.status == GatewayStatus.INITIALIZING
        
        await gateway.start()
        
        assert gateway.status == GatewayStatus.ACTIVE
        assert gateway.is_active is True
    
    @pytest.mark.asyncio
    async def test_stop(self, gateway: GatewayNode):
        """Test stopping a gateway."""
        await gateway.start()
        await gateway.stop()
        
        assert gateway.status == GatewayStatus.OFFLINE
        assert gateway.is_active is False
    
    def test_set_handlers(self, gateway: GatewayNode):
        """Test setting share handlers."""
        inbound_handler = AsyncMock()
        outbound_handler = AsyncMock()
        
        gateway.set_inbound_handler(inbound_handler)
        gateway.set_outbound_handler(outbound_handler)
        
        assert gateway._inbound_handler == inbound_handler
        assert gateway._outbound_handler == outbound_handler
    
    def test_register_external_gateway(self, gateway: GatewayNode):
        """Test registering external gateways."""
        ext_id = uuid4()
        endpoint = "https://external.com/gateway"
        
        gateway.register_external_gateway(ext_id, endpoint)
        
        assert ext_id in gateway._external_gateways
        assert gateway._external_gateways[ext_id] == endpoint
    
    def test_unregister_external_gateway(self, gateway: GatewayNode):
        """Test unregistering external gateways."""
        ext_id = uuid4()
        gateway.register_external_gateway(ext_id, "https://external.com/gateway")
        
        gateway.unregister_external_gateway(ext_id)
        
        assert ext_id not in gateway._external_gateways


class TestGatewayNodeRateLimiting:
    """Tests for GatewayNode rate limiting."""
    
    def test_rate_limit_check_allowed(self, gateway: GatewayNode):
        """Test rate limit check when allowed."""
        fed_id = uuid4()
        
        allowed, retry_after = gateway._check_rate_limit(fed_id, belief_count=5)
        
        assert allowed is True
        assert retry_after is None
    
    def test_rate_limit_check_exceeded_requests(self, gateway: GatewayNode):
        """Test rate limit check when requests exceeded."""
        fed_id = uuid4()
        
        # Simulate max requests
        for _ in range(gateway.config.rate_limit_max_requests):
            gateway._record_rate_limit(fed_id)
        
        allowed, retry_after = gateway._check_rate_limit(fed_id)
        
        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0
    
    def test_rate_limit_check_exceeded_beliefs(self, gateway: GatewayNode):
        """Test rate limit check when beliefs exceeded."""
        fed_id = uuid4()
        
        # Simulate near-max beliefs
        gateway._record_rate_limit(fed_id, belief_count=95)
        
        # Try to add more than remaining
        allowed, retry_after = gateway._check_rate_limit(fed_id, belief_count=10)
        
        assert allowed is False
    
    def test_rate_limit_window_reset(self, gateway: GatewayNode):
        """Test rate limit window resets after expiry."""
        fed_id = uuid4()
        
        # Record some requests
        gateway._record_rate_limit(fed_id)
        gateway._record_rate_limit(fed_id)
        
        state = gateway._get_rate_limit_state(fed_id)
        assert state.request_count == 2
        
        # Simulate window expiry
        state.window_start = time.time() - gateway.config.rate_limit_window - 1
        
        # Next check should reset
        state = gateway._get_rate_limit_state(fed_id)
        assert state.request_count == 0


class TestGatewayNodeAccessControl:
    """Tests for GatewayNode access control."""
    
    @pytest.mark.asyncio
    async def test_access_blocked_federation(
        self,
        gateway: GatewayNode,
        external_federation_id: UUID,
    ):
        """Test access denied for blocked federation."""
        gateway.block_federation(external_federation_id)
        
        allowed, reason = await gateway._check_access(
            external_federation_id,
            GatewayCapability.INBOUND_SHARE,
        )
        
        assert allowed is False
        assert reason == "federation_blocked"
    
    @pytest.mark.asyncio
    async def test_access_disabled_capability(
        self,
        gateway: GatewayNode,
        external_federation_id: UUID,
    ):
        """Test access denied for disabled capability."""
        # Disable a capability
        gateway.config.enabled_capabilities.discard(GatewayCapability.INBOUND_SHARE)
        
        allowed, reason = await gateway._check_access(
            external_federation_id,
            GatewayCapability.INBOUND_SHARE,
        )
        
        assert allowed is False
        assert reason == "capability_disabled"
    
    @pytest.mark.asyncio
    async def test_access_low_trust(
        self,
        gateway: GatewayNode,
        external_federation_id: UUID,
        mock_get_cursor,
    ):
        """Test access denied for low trust."""
        # Set low trust in mock
        mock_get_cursor.fetchone.return_value = {"trust_score": 0.1}
        
        allowed, reason = await gateway._check_access(
            external_federation_id,
            GatewayCapability.INBOUND_SHARE,
        )
        
        assert allowed is False
        assert "trust_below_threshold" in reason
    
    @pytest.mark.asyncio
    async def test_access_allowed(
        self,
        gateway: GatewayNode,
        external_federation_id: UUID,
        mock_get_cursor,
    ):
        """Test access allowed with sufficient trust."""
        mock_get_cursor.fetchone.return_value = {"trust_score": 0.8}
        
        allowed, reason = await gateway._check_access(
            external_federation_id,
            GatewayCapability.INBOUND_SHARE,
        )
        
        assert allowed is True
        assert reason is None


class TestGatewayNodeValidation:
    """Tests for GatewayNode share validation."""
    
    def test_validate_valid_share(
        self,
        gateway: GatewayNode,
        mock_belief: FederatedBelief,
        external_federation_id: UUID,
    ):
        """Test validating a valid share."""
        result = gateway._validate_share([mock_belief], external_federation_id)
        
        assert result == ValidationResult.VALID
    
    def test_validate_blocked_federation(
        self,
        gateway: GatewayNode,
        mock_belief: FederatedBelief,
        external_federation_id: UUID,
    ):
        """Test validation fails for blocked federation."""
        gateway.block_federation(external_federation_id)
        
        result = gateway._validate_share([mock_belief], external_federation_id)
        
        assert result == ValidationResult.BLOCKED_FEDERATION
    
    def test_validate_size_exceeded(
        self,
        gateway: GatewayNode,
        external_federation_id: UUID,
    ):
        """Test validation fails for oversized belief."""
        from valence.core.confidence import DimensionalConfidence
        large_belief = FederatedBelief(
            id=uuid4(),
            federation_id=uuid4(),
            origin_node_did="did:vkb:web:test.example.com",
            content="x" * (gateway.config.max_belief_size + 1),
            confidence=DimensionalConfidence(overall=0.8),
            visibility=Visibility.FEDERATED,
        )
        
        result = gateway._validate_share([large_belief], external_federation_id)
        
        assert result == ValidationResult.SIZE_EXCEEDED


class TestGatewayNodeAuditLogging:
    """Tests for GatewayNode audit logging."""
    
    def test_audit_entry_created(self, gateway: GatewayNode):
        """Test audit entry creation."""
        source_id = uuid4()
        
        entry = gateway._audit(
            AuditEventType.INBOUND_SHARE,
            source_federation_id=source_id,
            direction=ShareDirection.INBOUND,
            success=True,
        )
        
        assert entry.event_type == AuditEventType.INBOUND_SHARE
        assert entry.source_federation_id == source_id
        assert entry.success is True
    
    def test_audit_log_stored(self, gateway: GatewayNode):
        """Test audit entries are stored."""
        gateway._audit(AuditEventType.INBOUND_SHARE, success=True)
        gateway._audit(AuditEventType.OUTBOUND_SHARE, success=True)
        
        log = gateway.get_audit_log()
        
        assert len(log) == 2
    
    def test_audit_log_filtered_by_type(self, gateway: GatewayNode):
        """Test audit log filtering by event type."""
        gateway._audit(AuditEventType.INBOUND_SHARE, success=True)
        gateway._audit(AuditEventType.OUTBOUND_SHARE, success=True)
        gateway._audit(AuditEventType.INBOUND_SHARE, success=True)
        
        log = gateway.get_audit_log(event_type=AuditEventType.INBOUND_SHARE)
        
        assert len(log) == 2
        assert all(e.event_type == AuditEventType.INBOUND_SHARE for e in log)
    
    def test_audit_log_filtered_by_time(self, gateway: GatewayNode):
        """Test audit log filtering by time."""
        # Create old entry
        old_entry = gateway._audit(AuditEventType.INBOUND_SHARE, success=True)
        old_entry.timestamp = datetime.now(UTC) - timedelta(hours=2)
        
        # Create recent entry
        gateway._audit(AuditEventType.OUTBOUND_SHARE, success=True)
        
        since = datetime.now(UTC) - timedelta(hours=1)
        log = gateway.get_audit_log(since=since)
        
        assert len(log) == 1
    
    def test_audit_disabled(self, gateway: GatewayNode):
        """Test audit logging when disabled."""
        gateway.config.audit_enabled = False
        
        gateway._audit(AuditEventType.INBOUND_SHARE, success=True)
        
        # Should not be stored
        assert len(gateway._audit_log) == 0


class TestGatewayNodeInboundOperations:
    """Tests for GatewayNode inbound share operations."""
    
    @pytest.mark.asyncio
    async def test_receive_share_inactive(
        self,
        gateway: GatewayNode,
        inbound_share: InboundShare,
    ):
        """Test receiving share when gateway inactive."""
        result = await gateway.receive_share(inbound_share)
        
        assert result.success is False
        assert result.error_code == ErrorCode.INTERNAL_ERROR
    
    @pytest.mark.asyncio
    async def test_receive_share_no_source_id(
        self,
        gateway: GatewayNode,
        mock_belief: FederatedBelief,
    ):
        """Test receiving share without source federation ID."""
        await gateway.start()
        
        share = InboundShare(beliefs=[mock_belief])
        result = await gateway.receive_share(share)
        
        assert result.success is False
        assert result.error_code == ErrorCode.INVALID_REQUEST
    
    @pytest.mark.asyncio
    async def test_receive_share_rate_limited(
        self,
        gateway: GatewayNode,
        inbound_share: InboundShare,
        mock_get_cursor,
    ):
        """Test receiving share when rate limited."""
        await gateway.start()
        
        # Exhaust rate limit
        for _ in range(gateway.config.rate_limit_max_requests):
            gateway._record_rate_limit(inbound_share.source_federation_id)
        
        with pytest.raises(RateLimitException):
            await gateway.receive_share(inbound_share)
    
    @pytest.mark.asyncio
    async def test_receive_share_access_denied(
        self,
        gateway: GatewayNode,
        inbound_share: InboundShare,
    ):
        """Test receiving share when access denied."""
        await gateway.start()
        gateway.block_federation(inbound_share.source_federation_id)
        
        with pytest.raises(AccessDeniedException):
            await gateway.receive_share(inbound_share)
    
    @pytest.mark.asyncio
    async def test_receive_share_validation_failed(
        self,
        gateway: GatewayNode,
        external_federation_id: UUID,
        mock_get_cursor,
    ):
        """Test receiving share with validation failure."""
        await gateway.start()
        
        # Create oversized belief
        from valence.core.confidence import DimensionalConfidence
        large_belief = FederatedBelief(
            id=uuid4(),
            federation_id=uuid4(),
            origin_node_did="did:vkb:web:test.example.com",
            content="x" * (gateway.config.max_belief_size + 1),
            confidence=DimensionalConfidence(overall=0.8),
            visibility=Visibility.FEDERATED,
        )
        share = InboundShare(
            source_federation_id=external_federation_id,
            beliefs=[large_belief],
        )
        
        with pytest.raises(ValidationFailedException):
            await gateway.receive_share(share)
    
    @pytest.mark.asyncio
    async def test_receive_share_success(
        self,
        gateway: GatewayNode,
        inbound_share: InboundShare,
        mock_get_cursor,
    ):
        """Test successful share reception."""
        await gateway.start()
        
        result = await gateway.receive_share(inbound_share)
        
        assert result.success is True
        assert result.belief_count == 1
    
    @pytest.mark.asyncio
    async def test_receive_share_with_handler(
        self,
        gateway: GatewayNode,
        inbound_share: InboundShare,
        mock_get_cursor,
    ):
        """Test share reception with custom handler."""
        await gateway.start()
        
        routed_ids = [uuid4(), uuid4()]
        handler = AsyncMock(return_value=ShareResult(
            success=True,
            routed_to=routed_ids,
        ))
        gateway.set_inbound_handler(handler)
        
        result = await gateway.receive_share(inbound_share)
        
        assert result.success is True
        assert result.routed_to == routed_ids
        handler.assert_called_once_with(inbound_share)


class TestGatewayNodeOutboundOperations:
    """Tests for GatewayNode outbound share operations."""
    
    @pytest.mark.asyncio
    async def test_send_share_inactive(
        self,
        gateway: GatewayNode,
        outbound_share: OutboundShare,
    ):
        """Test sending share when gateway inactive."""
        result = await gateway.send_share(outbound_share)
        
        assert result.success is False
        assert result.error_code == ErrorCode.INTERNAL_ERROR
    
    @pytest.mark.asyncio
    async def test_send_share_no_target_id(
        self,
        gateway: GatewayNode,
        mock_belief: FederatedBelief,
    ):
        """Test sending share without target federation ID."""
        await gateway.start()
        
        share = OutboundShare(beliefs=[mock_belief])
        result = await gateway.send_share(share)
        
        assert result.success is False
        assert result.error_code == ErrorCode.INVALID_REQUEST
    
    @pytest.mark.asyncio
    async def test_send_share_unknown_gateway(
        self,
        gateway: GatewayNode,
        external_federation_id: UUID,
        mock_belief: FederatedBelief,
    ):
        """Test sending share to unknown gateway."""
        await gateway.start()
        
        share = OutboundShare(
            target_federation_id=external_federation_id,
            beliefs=[mock_belief],
        )
        result = await gateway.send_share(share)
        
        assert result.success is False
        assert result.error_code == ErrorCode.NODE_NOT_FOUND
    
    @pytest.mark.asyncio
    async def test_send_share_access_denied(
        self,
        gateway: GatewayNode,
        outbound_share: OutboundShare,
    ):
        """Test sending share when access denied."""
        await gateway.start()
        gateway.register_external_gateway(
            outbound_share.target_federation_id,
            outbound_share.target_gateway_endpoint,
        )
        gateway.block_federation(outbound_share.target_federation_id)
        
        with pytest.raises(AccessDeniedException):
            await gateway.send_share(outbound_share)
    
    @pytest.mark.asyncio
    async def test_send_share_success(
        self,
        gateway: GatewayNode,
        outbound_share: OutboundShare,
        mock_get_cursor,
    ):
        """Test successful share sending."""
        await gateway.start()
        gateway.register_external_gateway(
            outbound_share.target_federation_id,
            outbound_share.target_gateway_endpoint,
        )
        
        result = await gateway.send_share(outbound_share)
        
        assert result.success is True
        assert result.belief_count == 1
    
    @pytest.mark.asyncio
    async def test_send_share_with_handler(
        self,
        gateway: GatewayNode,
        outbound_share: OutboundShare,
        mock_get_cursor,
    ):
        """Test share sending with custom handler."""
        await gateway.start()
        gateway.register_external_gateway(
            outbound_share.target_federation_id,
            outbound_share.target_gateway_endpoint,
        )
        
        handler = AsyncMock(return_value=ShareResult(
            success=True,
            acknowledged=True,
        ))
        gateway.set_outbound_handler(handler)
        
        result = await gateway.send_share(outbound_share)
        
        assert result.success is True
        assert result.acknowledged is True
        handler.assert_called_once()


class TestGatewayNodeAdminOperations:
    """Tests for GatewayNode admin operations."""
    
    def test_block_federation(self, gateway: GatewayNode):
        """Test blocking a federation."""
        fed_id = uuid4()
        
        gateway.block_federation(fed_id)
        
        assert fed_id in gateway.config.blocked_federations
    
    def test_unblock_federation(self, gateway: GatewayNode):
        """Test unblocking a federation."""
        fed_id = uuid4()
        gateway.block_federation(fed_id)
        
        gateway.unblock_federation(fed_id)
        
        assert fed_id not in gateway.config.blocked_federations
    
    def test_update_config(self, gateway: GatewayNode):
        """Test updating gateway configuration."""
        gateway.update_config(
            rate_limit_max_requests=50,
            min_trust_for_access=0.5,
        )
        
        assert gateway.config.rate_limit_max_requests == 50
        assert gateway.config.min_trust_for_access == 0.5
    
    def test_get_stats(self, gateway: GatewayNode):
        """Test getting gateway statistics."""
        # Generate some audit entries
        gateway._audit(AuditEventType.INBOUND_SHARE, success=True)
        gateway._audit(AuditEventType.OUTBOUND_SHARE, success=True)
        gateway._audit(AuditEventType.ACCESS_DENIED, success=False)
        
        stats = gateway.get_stats()
        
        assert "gateway_id" in stats
        assert "federation_id" in stats
        assert "status" in stats
        assert "stats" in stats
        assert stats["stats"]["last_hour"]["total_events"] == 3
    
    def test_to_dict(self, gateway: GatewayNode):
        """Test gateway serialization."""
        d = gateway.to_dict()
        
        assert "id" in d
        assert "federation_id" in d
        assert "endpoint" in d
        assert "capabilities" in d
        assert "status" in d
        assert "config" in d


# =============================================================================
# GATEWAY REGISTRY TESTS
# =============================================================================


class TestGatewayRegistry:
    """Tests for GatewayRegistry."""
    
    @pytest.fixture
    def registry(self) -> GatewayRegistry:
        """Fresh registry for each test."""
        return GatewayRegistry()
    
    def test_register(self, registry: GatewayRegistry, gateway: GatewayNode):
        """Test registering a gateway."""
        registry.register(gateway)
        
        assert registry.get(gateway.id) == gateway
        assert registry.get_by_federation(gateway.federation_id) == gateway
    
    def test_unregister(self, registry: GatewayRegistry, gateway: GatewayNode):
        """Test unregistering a gateway."""
        registry.register(gateway)
        
        removed = registry.unregister(gateway.id)
        
        assert removed == gateway
        assert registry.get(gateway.id) is None
        assert registry.get_by_federation(gateway.federation_id) is None
    
    def test_get_nonexistent(self, registry: GatewayRegistry):
        """Test getting nonexistent gateway."""
        assert registry.get(uuid4()) is None
        assert registry.get_by_federation(uuid4()) is None
    
    def test_list_gateways(self, registry: GatewayRegistry):
        """Test listing gateways."""
        # Create multiple gateways
        g1 = GatewayNode(uuid4(), "https://g1.example.com")
        g2 = GatewayNode(uuid4(), "https://g2.example.com")
        g2.status = GatewayStatus.ACTIVE
        
        registry.register(g1)
        registry.register(g2)
        
        # All gateways
        all_gateways = registry.list_gateways()
        assert len(all_gateways) == 2
        
        # Filter by status
        active = registry.list_gateways(status=GatewayStatus.ACTIVE)
        assert len(active) == 1
        assert active[0] == g2
    
    def test_list_gateways_by_capability(self, registry: GatewayRegistry):
        """Test listing gateways by capability."""
        g1 = GatewayNode(
            uuid4(),
            "https://g1.example.com",
            capabilities=[GatewayCapability.INBOUND_SHARE],
        )
        g2 = GatewayNode(
            uuid4(),
            "https://g2.example.com",
            capabilities=[GatewayCapability.OUTBOUND_SHARE],
        )
        
        registry.register(g1)
        registry.register(g2)
        
        inbound = registry.list_gateways(capability=GatewayCapability.INBOUND_SHARE)
        assert len(inbound) == 1
        assert inbound[0] == g1
    
    @pytest.mark.asyncio
    async def test_start_all(self, registry: GatewayRegistry):
        """Test starting all gateways."""
        g1 = GatewayNode(uuid4(), "https://g1.example.com")
        g2 = GatewayNode(uuid4(), "https://g2.example.com")
        
        registry.register(g1)
        registry.register(g2)
        
        await registry.start_all()
        
        assert g1.status == GatewayStatus.ACTIVE
        assert g2.status == GatewayStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_stop_all(self, registry: GatewayRegistry):
        """Test stopping all gateways."""
        g1 = GatewayNode(uuid4(), "https://g1.example.com")
        g2 = GatewayNode(uuid4(), "https://g2.example.com")
        
        registry.register(g1)
        registry.register(g2)
        
        await registry.start_all()
        await registry.stop_all()
        
        assert g1.status == GatewayStatus.OFFLINE
        assert g2.status == GatewayStatus.OFFLINE


class TestGlobalRegistry:
    """Tests for global gateway registry."""
    
    def test_get_gateway_registry(self):
        """Test getting global registry."""
        registry = get_gateway_registry()
        
        assert registry is not None
        assert isinstance(registry, GatewayRegistry)
    
    def test_get_same_registry(self):
        """Test global registry is singleton."""
        r1 = get_gateway_registry()
        r2 = get_gateway_registry()
        
        assert r1 is r2


class TestCreateGateway:
    """Tests for create_gateway convenience function."""
    
    def test_create_gateway_basic(self):
        """Test basic gateway creation."""
        fed_id = uuid4()
        endpoint = "https://gateway.example.com"
        
        gateway = create_gateway(
            federation_id=fed_id,
            endpoint=endpoint,
            register=False,
        )
        
        assert gateway.federation_id == fed_id
        assert gateway.endpoint == endpoint
    
    def test_create_gateway_with_capabilities(self):
        """Test gateway creation with capabilities."""
        gateway = create_gateway(
            federation_id=uuid4(),
            endpoint="https://gateway.example.com",
            capabilities=[GatewayCapability.TRUST_BRIDGE],
            register=False,
        )
        
        assert GatewayCapability.TRUST_BRIDGE in gateway.capabilities
    
    def test_create_gateway_with_config(self):
        """Test gateway creation with custom config."""
        config = GatewayConfig(rate_limit_max_requests=50)
        
        gateway = create_gateway(
            federation_id=uuid4(),
            endpoint="https://gateway.example.com",
            config=config,
            register=False,
        )
        
        assert gateway.config.rate_limit_max_requests == 50
    
    def test_create_gateway_auto_register(self):
        """Test gateway creation with auto-registration."""
        fed_id = uuid4()
        
        gateway = create_gateway(
            federation_id=fed_id,
            endpoint="https://gateway.example.com",
            register=True,
        )
        
        registry = get_gateway_registry()
        assert registry.get(gateway.id) == gateway
        
        # Cleanup
        registry.unregister(gateway.id)


# =============================================================================
# EXCEPTION TESTS
# =============================================================================


class TestGatewayExceptions:
    """Tests for gateway exceptions."""
    
    def test_rate_limit_exception(self):
        """Test RateLimitException."""
        fed_id = uuid4()
        exc = RateLimitException(
            "Rate limit exceeded",
            federation_id=fed_id,
            retry_after=30.0,
        )
        
        assert exc.federation_id == fed_id
        assert exc.retry_after == 30.0
        assert "rate limit" in exc.message.lower()
    
    def test_access_denied_exception(self):
        """Test AccessDeniedException."""
        fed_id = uuid4()
        exc = AccessDeniedException(
            "Access denied",
            federation_id=fed_id,
            reason="trust_below_threshold",
        )
        
        assert exc.federation_id == fed_id
        assert exc.reason == "trust_below_threshold"
    
    def test_validation_failed_exception(self):
        """Test ValidationFailedException."""
        exc = ValidationFailedException(
            "Validation failed",
            result=ValidationResult.SIZE_EXCEEDED,
        )
        
        assert exc.result == ValidationResult.SIZE_EXCEEDED
