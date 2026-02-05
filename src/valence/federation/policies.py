"""Federation admin threshold policies.

Implements multi-admin approval requirements for sensitive federation operations.
Ensures critical actions require consensus among administrators.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable
from uuid import UUID, uuid4


def _utc_now() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


# =============================================================================
# ENUMS
# =============================================================================


class Operation(str, Enum):
    """Sensitive federation operations requiring admin approval."""
    ADD_MEMBER = "add_member"           # Add new member to federation
    REMOVE_MEMBER = "remove_member"     # Remove member from federation
    UPDATE_POLICY = "update_policy"     # Modify federation policies
    DISSOLVE = "dissolve"               # Dissolve the federation


class ApprovalStatus(str, Enum):
    """Status of a pending approval request."""
    PENDING = "pending"       # Awaiting approvals
    APPROVED = "approved"     # Threshold met, executed
    REJECTED = "rejected"     # Explicitly rejected
    EXPIRED = "expired"       # Timed out without meeting threshold


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ThresholdPolicy:
    """Policy defining approval thresholds for sensitive operations.
    
    Attributes:
        operation: The operation this policy governs
        required_approvals: Number of admin approvals needed
        admin_dids: DIDs of admins authorized to approve
        timeout: Duration before pending approvals expire
        id: Unique identifier for this policy
    """
    operation: Operation
    required_approvals: int
    admin_dids: set[str]
    timeout: timedelta = field(default_factory=lambda: timedelta(hours=24))
    id: UUID = field(default_factory=uuid4)
    
    def __post_init__(self):
        """Validate policy configuration."""
        # Ensure admin_dids is a set first
        if not isinstance(self.admin_dids, set):
            self.admin_dids = set(self.admin_dids)
        
        if not self.admin_dids:
            raise ValueError("admin_dids cannot be empty")
        if self.required_approvals < 1:
            raise ValueError("required_approvals must be at least 1")
        if self.required_approvals > len(self.admin_dids):
            raise ValueError(
                f"required_approvals ({self.required_approvals}) cannot exceed "
                f"number of admins ({len(self.admin_dids)})"
            )
    
    def is_admin(self, did: str) -> bool:
        """Check if a DID is an authorized admin."""
        return did in self.admin_dids
    
    def can_approve(self, approval: PendingApproval, did: str) -> bool:
        """Check if a DID can approve a pending request."""
        if not self.is_admin(did):
            return False
        if did in approval.approved_by:
            return False  # Already approved
        if approval.status != ApprovalStatus.PENDING:
            return False
        return True


@dataclass
class PendingApproval:
    """Tracks pending approval for a sensitive operation.
    
    Attributes:
        operation: The operation requiring approval
        payload: Operation-specific data (e.g., member DID to add)
        approved_by: Set of admin DIDs who have approved
        created_at: When the approval request was created
        expires_at: When the approval request expires
        status: Current status of the approval
        id: Unique identifier for this approval request
        executed_at: When the operation was executed (if approved)
        result: Result of execution (if executed)
    """
    operation: Operation
    payload: dict[str, Any]
    approved_by: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=_utc_now)
    expires_at: datetime | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    id: UUID = field(default_factory=uuid4)
    executed_at: datetime | None = None
    result: Any = None
    
    def __post_init__(self):
        """Ensure approved_by is a set."""
        if not isinstance(self.approved_by, set):
            self.approved_by = set(self.approved_by)
    
    @property
    def approval_count(self) -> int:
        """Number of approvals received."""
        return len(self.approved_by)
    
    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if this approval has expired."""
        if self.expires_at is None:
            return False
        now = now or _utc_now()
        return now >= self.expires_at


# =============================================================================
# POLICY MANAGER
# =============================================================================


class PolicyManager:
    """Manages threshold policies and approval workflows.
    
    Handles policy registration, approval tracking, and automatic
    execution when thresholds are met.
    """
    
    def __init__(self):
        """Initialize the policy manager."""
        self._policies: dict[Operation, ThresholdPolicy] = {}
        self._pending: dict[UUID, PendingApproval] = {}
        self._handlers: dict[Operation, Callable[[dict[str, Any]], Any]] = {}
    
    # -------------------------------------------------------------------------
    # Policy Management
    # -------------------------------------------------------------------------
    
    def register_policy(self, policy: ThresholdPolicy) -> None:
        """Register a threshold policy for an operation.
        
        Args:
            policy: The policy to register
            
        Raises:
            ValueError: If a policy for this operation already exists
        """
        if policy.operation in self._policies:
            raise ValueError(f"Policy already registered for {policy.operation}")
        self._policies[policy.operation] = policy
    
    def update_policy(self, policy: ThresholdPolicy) -> None:
        """Update an existing policy.
        
        Note: This itself should typically go through approval for UPDATE_POLICY.
        
        Args:
            policy: The updated policy
        """
        self._policies[policy.operation] = policy
    
    def get_policy(self, operation: Operation) -> ThresholdPolicy | None:
        """Get the policy for an operation."""
        return self._policies.get(operation)
    
    def register_handler(
        self, 
        operation: Operation, 
        handler: Callable[[dict[str, Any]], Any]
    ) -> None:
        """Register an execution handler for an operation.
        
        Args:
            operation: The operation to handle
            handler: Callable that executes the operation, receives payload
        """
        self._handlers[operation] = handler
    
    # -------------------------------------------------------------------------
    # Approval Workflow
    # -------------------------------------------------------------------------
    
    def request_approval(
        self,
        operation: Operation,
        payload: dict[str, Any],
        requestor_did: str,
    ) -> PendingApproval:
        """Create a new approval request.
        
        Args:
            operation: The operation requiring approval
            payload: Operation-specific data
            requestor_did: DID of the admin requesting the operation
            
        Returns:
            The created PendingApproval
            
        Raises:
            ValueError: If no policy exists or requestor isn't an admin
        """
        policy = self._policies.get(operation)
        if not policy:
            raise ValueError(f"No policy registered for {operation}")
        
        if not policy.is_admin(requestor_did):
            raise ValueError(f"Requestor {requestor_did} is not an authorized admin")
        
        now = _utc_now()
        approval = PendingApproval(
            operation=operation,
            payload=payload,
            approved_by={requestor_did},  # Requestor auto-approves
            created_at=now,
            expires_at=now + policy.timeout,
        )
        
        self._pending[approval.id] = approval
        
        # Check if single approval meets threshold
        self._check_and_execute(approval)
        
        return approval
    
    def approve(
        self,
        approval_id: UUID,
        admin_did: str,
    ) -> PendingApproval:
        """Add an approval to a pending request.
        
        Args:
            approval_id: ID of the pending approval
            admin_did: DID of the approving admin
            
        Returns:
            The updated PendingApproval
            
        Raises:
            KeyError: If approval_id not found
            ValueError: If admin can't approve or approval is not pending
        """
        approval = self._pending.get(approval_id)
        if not approval:
            raise KeyError(f"Approval {approval_id} not found")
        
        # Check expiration
        if approval.is_expired():
            approval.status = ApprovalStatus.EXPIRED
            raise ValueError("Approval has expired")
        
        policy = self._policies.get(approval.operation)
        if not policy:
            raise ValueError(f"No policy for operation {approval.operation}")
        
        if not policy.can_approve(approval, admin_did):
            if not policy.is_admin(admin_did):
                raise ValueError(f"{admin_did} is not an authorized admin")
            if admin_did in approval.approved_by:
                raise ValueError(f"{admin_did} has already approved")
            raise ValueError("Cannot approve this request")
        
        approval.approved_by.add(admin_did)
        
        # Check if threshold is met
        self._check_and_execute(approval)
        
        return approval
    
    def reject(
        self,
        approval_id: UUID,
        admin_did: str,
    ) -> PendingApproval:
        """Reject a pending approval request.
        
        Any admin can reject, which cancels the request.
        
        Args:
            approval_id: ID of the pending approval
            admin_did: DID of the rejecting admin
            
        Returns:
            The updated PendingApproval
            
        Raises:
            KeyError: If approval_id not found
            ValueError: If admin isn't authorized
        """
        approval = self._pending.get(approval_id)
        if not approval:
            raise KeyError(f"Approval {approval_id} not found")
        
        policy = self._policies.get(approval.operation)
        if not policy:
            raise ValueError(f"No policy for operation {approval.operation}")
        
        if not policy.is_admin(admin_did):
            raise ValueError(f"{admin_did} is not an authorized admin")
        
        if approval.status != ApprovalStatus.PENDING:
            raise ValueError(f"Cannot reject: approval is {approval.status}")
        
        approval.status = ApprovalStatus.REJECTED
        return approval
    
    def get_pending(self, approval_id: UUID) -> PendingApproval | None:
        """Get a pending approval by ID."""
        return self._pending.get(approval_id)
    
    def list_pending(
        self,
        operation: Operation | None = None,
        include_expired: bool = False,
    ) -> list[PendingApproval]:
        """List pending approvals.
        
        Args:
            operation: Filter by operation type
            include_expired: Include expired approvals
            
        Returns:
            List of matching pending approvals
        """
        results = []
        now = _utc_now()
        
        for approval in self._pending.values():
            if approval.status != ApprovalStatus.PENDING:
                continue
            if operation and approval.operation != operation:
                continue
            if not include_expired and approval.is_expired(now):
                approval.status = ApprovalStatus.EXPIRED
                continue
            results.append(approval)
        
        return results
    
    def cleanup_expired(self) -> int:
        """Mark expired approvals and return count."""
        now = _utc_now()
        count = 0
        
        for approval in self._pending.values():
            if approval.status == ApprovalStatus.PENDING and approval.is_expired(now):
                approval.status = ApprovalStatus.EXPIRED
                count += 1
        
        return count
    
    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------
    
    def _check_and_execute(self, approval: PendingApproval) -> bool:
        """Check if threshold is met and execute if so.
        
        Returns:
            True if executed, False otherwise
        """
        policy = self._policies.get(approval.operation)
        if not policy:
            return False
        
        if approval.approval_count >= policy.required_approvals:
            return self._execute(approval)
        
        return False
    
    def _execute(self, approval: PendingApproval) -> bool:
        """Execute an approved operation.
        
        Returns:
            True if executed successfully
        """
        handler = self._handlers.get(approval.operation)
        
        approval.status = ApprovalStatus.APPROVED
        approval.executed_at = _utc_now()
        
        if handler:
            try:
                approval.result = handler(approval.payload)
            except Exception as e:
                approval.result = {"error": str(e)}
                return False
        
        return True


# =============================================================================
# MODULE-LEVEL CONVENIENCE
# =============================================================================

_default_manager: PolicyManager | None = None
_default_manager_lock = threading.Lock()


def get_policy_manager() -> PolicyManager:
    """Get the default policy manager singleton.
    
    Thread-safe initialization using double-checked locking pattern.
    """
    global _default_manager
    if _default_manager is None:
        with _default_manager_lock:
            # Double-check after acquiring lock
            if _default_manager is None:
                _default_manager = PolicyManager()
    return _default_manager


def reset_policy_manager() -> None:
    """Reset the default policy manager (for testing).
    
    Thread-safe reset using lock.
    """
    global _default_manager
    with _default_manager_lock:
        _default_manager = None
