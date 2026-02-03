"""Belief Synchronization for Valence Federation.

Manages synchronization of beliefs between federation nodes:
- Outbound sync: Queue and send beliefs to peers
- Inbound sync: Receive and store beliefs from peers
- Cursor-based incremental sync
- Vector clocks for conflict detection
- Sync scheduling and retry logic
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import aiohttp

from ..core.db import get_cursor
from ..server.config import get_settings
from .discovery import get_node_by_id, get_node_trust, mark_node_active, mark_node_unreachable
from .models import (
    FederationNode,
    NodeTrust,
    SyncState,
    SyncStatus,
    Visibility,
    ShareLevel,
)
from .protocol import (
    SyncRequest,
    SyncResponse,
    SyncChange,
    ShareBeliefRequest,
    ShareBeliefResponse,
    handle_share_belief,
    handle_sync_request,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SYNC STATE MANAGEMENT
# =============================================================================


def get_sync_state(node_id: UUID) -> SyncState | None:
    """Get sync state for a node.

    Args:
        node_id: The node's UUID

    Returns:
        SyncState if found, None otherwise
    """
    try:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM sync_state WHERE node_id = %s", (node_id,))
            row = cur.fetchone()
            if row:
                return SyncState(
                    id=row["id"],
                    node_id=row["node_id"],
                    last_received_cursor=row.get("last_received_cursor"),
                    last_sent_cursor=row.get("last_sent_cursor"),
                    vector_clock=row.get("vector_clock", {}),
                    status=SyncStatus(row.get("status", "idle")),
                    beliefs_sent=row.get("beliefs_sent", 0),
                    beliefs_received=row.get("beliefs_received", 0),
                    last_sync_duration_ms=row.get("last_sync_duration_ms"),
                    last_error=row.get("last_error"),
                    error_count=row.get("error_count", 0),
                    last_sync_at=row.get("last_sync_at"),
                    next_sync_scheduled=row.get("next_sync_scheduled"),
                    created_at=row.get("created_at", datetime.now()),
                    modified_at=row.get("modified_at", datetime.now()),
                )
            return None
    except Exception as e:
        logger.warning(f"Error getting sync state for {node_id}: {e}")
        return None


def update_sync_state(
    node_id: UUID,
    status: SyncStatus | None = None,
    last_received_cursor: str | None = None,
    last_sent_cursor: str | None = None,
    beliefs_sent_delta: int = 0,
    beliefs_received_delta: int = 0,
    last_error: str | None = None,
    clear_error: bool = False,
) -> bool:
    """Update sync state for a node.

    Args:
        node_id: The node's UUID
        status: New status (optional)
        last_received_cursor: New received cursor (optional)
        last_sent_cursor: New sent cursor (optional)
        beliefs_sent_delta: Increment beliefs sent counter
        beliefs_received_delta: Increment beliefs received counter
        last_error: Set error message (optional)
        clear_error: Clear error state

    Returns:
        True if updated
    """
    try:
        updates = ["modified_at = NOW()"]
        params: list[Any] = []

        if status is not None:
            updates.append("status = %s")
            params.append(status.value)

        if last_received_cursor is not None:
            updates.append("last_received_cursor = %s")
            params.append(last_received_cursor)

        if last_sent_cursor is not None:
            updates.append("last_sent_cursor = %s")
            params.append(last_sent_cursor)

        if beliefs_sent_delta:
            updates.append("beliefs_sent = beliefs_sent + %s")
            params.append(beliefs_sent_delta)

        if beliefs_received_delta:
            updates.append("beliefs_received = beliefs_received + %s")
            params.append(beliefs_received_delta)

        if last_error is not None:
            updates.append("last_error = %s")
            updates.append("error_count = error_count + 1")
            params.append(last_error)
        elif clear_error:
            updates.append("last_error = NULL")
            updates.append("error_count = 0")

        params.append(node_id)

        with get_cursor() as cur:
            cur.execute(f"""
                UPDATE sync_state
                SET {', '.join(updates)}
                WHERE node_id = %s
            """, params)
            return True

    except Exception as e:
        logger.warning(f"Error updating sync state: {e}")
        return False


# =============================================================================
# OUTBOUND QUEUE
# =============================================================================


def queue_belief_for_sync(
    belief_id: UUID,
    operation: str = "share_belief",
    target_node_id: UUID | None = None,
    priority: int = 5,
) -> bool:
    """Queue a belief for outbound synchronization.

    Args:
        belief_id: The belief to sync
        operation: Type of operation (share_belief, update_belief, etc.)
        target_node_id: Specific node to sync to (None = all active nodes)
        priority: Priority (1=highest, 10=lowest)

    Returns:
        True if queued
    """
    try:
        with get_cursor() as cur:
            cur.execute("""
                INSERT INTO sync_outbound_queue (
                    target_node_id, operation, belief_id, priority, status
                ) VALUES (%s, %s, %s, %s, 'pending')
            """, (target_node_id, operation, belief_id, priority))
            return True
    except Exception as e:
        logger.warning(f"Error queueing belief for sync: {e}")
        return False


def get_pending_sync_items(
    target_node_id: UUID | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get pending sync items from the queue.

    Args:
        target_node_id: Filter by target node (None = all pending)
        limit: Maximum items to return

    Returns:
        List of pending sync items
    """
    try:
        conditions = ["status = 'pending'", "scheduled_for <= NOW()"]
        params: list[Any] = []

        if target_node_id:
            conditions.append("(target_node_id = %s OR target_node_id IS NULL)")
            params.append(target_node_id)

        params.append(limit)

        with get_cursor() as cur:
            cur.execute(f"""
                SELECT * FROM sync_outbound_queue
                WHERE {' AND '.join(conditions)}
                ORDER BY priority ASC, created_at ASC
                LIMIT %s
            """, params)
            return cur.fetchall()
    except Exception as e:
        logger.warning(f"Error getting pending sync items: {e}")
        return []


def mark_sync_item_sent(item_id: UUID) -> bool:
    """Mark a sync item as sent."""
    try:
        with get_cursor() as cur:
            cur.execute("""
                UPDATE sync_outbound_queue
                SET status = 'sent', last_attempt_at = NOW()
                WHERE id = %s
            """, (item_id,))
            return True
    except Exception as e:
        logger.warning(f"Error marking sync item sent: {e}")
        return False


def mark_sync_item_failed(item_id: UUID, error: str) -> bool:
    """Mark a sync item as failed."""
    try:
        with get_cursor() as cur:
            cur.execute("""
                UPDATE sync_outbound_queue
                SET status = CASE
                        WHEN attempts >= max_attempts THEN 'failed'
                        ELSE 'pending'
                    END,
                    attempts = attempts + 1,
                    last_attempt_at = NOW(),
                    last_error = %s,
                    scheduled_for = NOW() + INTERVAL '5 minutes' * attempts
                WHERE id = %s
            """, (error, item_id))
            return True
    except Exception as e:
        logger.warning(f"Error marking sync item failed: {e}")
        return False


# =============================================================================
# SYNC MANAGER
# =============================================================================


class SyncManager:
    """Manages synchronization with federation peers."""

    def __init__(self):
        self.settings = get_settings()
        self._running = False
        self._sync_task: asyncio.Task | None = None

    async def start(self):
        """Start the sync manager."""
        if self._running:
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Sync manager started")

    async def stop(self):
        """Stop the sync manager."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info("Sync manager stopped")

    async def _sync_loop(self):
        """Main sync loop."""
        while self._running:
            try:
                await self._process_outbound_queue()
                await self._sync_with_peers()
            except Exception as e:
                logger.exception("Error in sync loop")

            # Wait for next sync interval
            await asyncio.sleep(self.settings.federation_sync_interval_seconds)

    async def _process_outbound_queue(self):
        """Process the outbound sync queue."""
        # Get all active nodes
        with get_cursor() as cur:
            cur.execute("""
                SELECT id, did, federation_endpoint
                FROM federation_nodes
                WHERE status = 'active'
            """)
            active_nodes = cur.fetchall()

        for node_row in active_nodes:
            node_id = node_row["id"]
            federation_endpoint = node_row["federation_endpoint"]

            if not federation_endpoint:
                continue

            # Get pending items for this node
            items = get_pending_sync_items(target_node_id=node_id, limit=50)

            if not items:
                continue

            # Get trust level
            trust = get_node_trust(node_id)
            if not trust or trust.overall < 0.2:
                continue  # Don't sync to untrusted nodes

            # Send beliefs
            await self._send_beliefs_to_node(
                node_id,
                federation_endpoint,
                items,
            )

    async def _send_beliefs_to_node(
        self,
        node_id: UUID,
        federation_endpoint: str,
        items: list[dict[str, Any]],
    ):
        """Send beliefs to a specific node.

        Args:
            node_id: Target node's UUID
            federation_endpoint: Node's federation endpoint URL
            items: Queue items to send
        """
        # Collect beliefs to send
        belief_ids = [item["belief_id"] for item in items if item.get("belief_id")]

        if not belief_ids:
            return

        # Get belief data
        with get_cursor() as cur:
            placeholders = ",".join(["%s"] * len(belief_ids))
            cur.execute(f"""
                SELECT * FROM beliefs
                WHERE id IN ({placeholders})
                  AND is_local = TRUE
                  AND visibility IN ('federated', 'public')
            """, belief_ids)
            belief_rows = cur.fetchall()

        if not belief_rows:
            # Mark items as sent (nothing to send)
            for item in items:
                mark_sync_item_sent(item["id"])
            return

        # Build federated beliefs
        beliefs_data = []
        for row in belief_rows:
            belief_dict = self._belief_to_federated(row)
            if belief_dict:
                beliefs_data.append(belief_dict)

        if not beliefs_data:
            for item in items:
                mark_sync_item_sent(item["id"])
            return

        # Send to node
        try:
            url = f"{federation_endpoint}/beliefs"
            async with aiohttp.ClientSession() as session:
                # TODO: Add authentication headers
                async with session.post(
                    url,
                    json={
                        "type": "SHARE_BELIEF",
                        "request_id": str(uuid4()),
                        "beliefs": beliefs_data,
                        "timestamp": datetime.now().isoformat(),
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(
                            f"Sent {len(beliefs_data)} beliefs to node {node_id}: "
                            f"accepted={result.get('accepted', 0)}, rejected={result.get('rejected', 0)}"
                        )
                        for item in items:
                            mark_sync_item_sent(item["id"])

                        update_sync_state(
                            node_id,
                            beliefs_sent_delta=result.get("accepted", len(beliefs_data)),
                            clear_error=True,
                        )
                        mark_node_active(node_id)
                    else:
                        error = f"HTTP {response.status}"
                        for item in items:
                            mark_sync_item_failed(item["id"], error)
                        update_sync_state(node_id, last_error=error)

        except Exception as e:
            error = str(e)
            logger.warning(f"Failed to send beliefs to {node_id}: {error}")
            for item in items:
                mark_sync_item_failed(item["id"], error)
            update_sync_state(node_id, last_error=error)
            mark_node_unreachable(node_id)

    def _belief_to_federated(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a belief row to federated format."""
        from .identity import sign_belief_content

        settings = self.settings
        did = settings.federation_node_did or f"did:vkb:web:localhost:{settings.port}"

        result = {
            "id": str(row["id"]),
            "federation_id": str(row.get("federation_id") or row["id"]),
            "origin_node_did": did,
            "content": row["content"],
            "confidence": row["confidence"],
            "domain_path": row.get("domain_path", []),
            "visibility": row.get("visibility", "federated"),
            "share_level": row.get("share_level", "belief_only"),
            "hop_count": 0,
            "federation_path": [],
        }

        if row.get("valid_from"):
            result["valid_from"] = row["valid_from"].isoformat()
        if row.get("valid_until"):
            result["valid_until"] = row["valid_until"].isoformat()

        # Sign the belief
        if settings.federation_private_key:
            signable = {
                "federation_id": result["federation_id"],
                "origin_node_did": result["origin_node_did"],
                "content": result["content"],
                "confidence": result["confidence"],
                "domain_path": result["domain_path"],
                "valid_from": result.get("valid_from"),
                "valid_until": result.get("valid_until"),
            }
            result["origin_signature"] = sign_belief_content(
                signable,
                bytes.fromhex(settings.federation_private_key),
            )
            result["signed_at"] = datetime.now().isoformat()

        return result

    async def _sync_with_peers(self):
        """Pull sync updates from active peers."""
        with get_cursor() as cur:
            cur.execute("""
                SELECT fn.*, ss.last_received_cursor, ss.last_sync_at
                FROM federation_nodes fn
                JOIN sync_state ss ON fn.id = ss.node_id
                WHERE fn.status = 'active'
                  AND (ss.next_sync_scheduled IS NULL OR ss.next_sync_scheduled <= NOW())
            """)
            nodes = cur.fetchall()

        for node_row in nodes:
            node_id = node_row["id"]
            federation_endpoint = node_row.get("federation_endpoint")

            if not federation_endpoint:
                continue

            trust = get_node_trust(node_id)
            if not trust or trust.overall < 0.1:
                continue

            await self._pull_from_node(
                node_id,
                federation_endpoint,
                trust.overall,
                node_row.get("last_received_cursor"),
                node_row.get("last_sync_at"),
            )

    async def _pull_from_node(
        self,
        node_id: UUID,
        federation_endpoint: str,
        trust_level: float,
        cursor: str | None,
        last_sync: datetime | None,
    ):
        """Pull updates from a specific node.

        Args:
            node_id: Node's UUID
            federation_endpoint: Node's federation endpoint
            trust_level: Current trust level
            cursor: Last sync cursor
            last_sync: Last sync timestamp
        """
        try:
            url = f"{federation_endpoint}/sync"
            since = last_sync or (datetime.now() - timedelta(days=7))

            async with aiohttp.ClientSession() as session:
                # TODO: Add authentication
                async with session.post(
                    url,
                    json={
                        "type": "SYNC_REQUEST",
                        "request_id": str(uuid4()),
                        "since": since.isoformat(),
                        "domains": [],  # All domains
                        "cursor": cursor,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        changes = result.get("changes", [])

                        if changes:
                            # Process changes
                            beliefs_received = await self._process_sync_changes(
                                node_id, trust_level, changes
                            )

                            update_sync_state(
                                node_id,
                                last_received_cursor=result.get("cursor"),
                                beliefs_received_delta=beliefs_received,
                                clear_error=True,
                            )

                        # Schedule next sync
                        with get_cursor() as cur:
                            interval = self.settings.federation_sync_interval_seconds
                            cur.execute("""
                                UPDATE sync_state
                                SET last_sync_at = NOW(),
                                    next_sync_scheduled = NOW() + INTERVAL '%s seconds'
                                WHERE node_id = %s
                            """, (interval, node_id))

                        mark_node_active(node_id)

        except Exception as e:
            logger.warning(f"Failed to pull from {node_id}: {e}")
            update_sync_state(node_id, last_error=str(e))

    async def _process_sync_changes(
        self,
        node_id: UUID,
        trust_level: float,
        changes: list[dict[str, Any]],
    ) -> int:
        """Process sync changes from a peer.

        Args:
            node_id: Source node's UUID
            trust_level: Node's trust level
            changes: List of change objects

        Returns:
            Number of beliefs received
        """
        beliefs_received = 0

        for change in changes:
            change_type = change.get("type")
            belief_data = change.get("belief")

            if not belief_data:
                continue

            if change_type in ("belief_created", "belief_superseded"):
                # Use the share belief handler
                request = ShareBeliefRequest(beliefs=[belief_data])
                response = handle_share_belief(request, node_id, trust_level)
                beliefs_received += response.accepted

        return beliefs_received


# =============================================================================
# VECTOR CLOCK
# =============================================================================


def update_vector_clock(
    node_id: UUID,
    peer_did: str,
    sequence: int,
) -> dict[str, int]:
    """Update the vector clock for a peer.

    Args:
        node_id: Our node's sync state
        peer_did: The peer's DID
        sequence: The sequence number to record

    Returns:
        Updated vector clock
    """
    try:
        with get_cursor() as cur:
            cur.execute("""
                SELECT vector_clock FROM sync_state WHERE node_id = %s
            """, (node_id,))
            row = cur.fetchone()

            if row:
                clock = row.get("vector_clock", {})
            else:
                clock = {}

            # Update clock
            clock[peer_did] = max(clock.get(peer_did, 0), sequence)

            # Save back
            cur.execute("""
                UPDATE sync_state
                SET vector_clock = %s, modified_at = NOW()
                WHERE node_id = %s
            """, (json.dumps(clock), node_id))

            return clock

    except Exception as e:
        logger.warning(f"Error updating vector clock: {e}")
        return {}


def compare_vector_clocks(
    clock_a: dict[str, int],
    clock_b: dict[str, int],
) -> str:
    """Compare two vector clocks.

    Returns:
        'equal' - clocks are identical
        'a_before_b' - a happened before b
        'b_before_a' - b happened before a
        'concurrent' - clocks are concurrent (conflict)
    """
    all_keys = set(clock_a.keys()) | set(clock_b.keys())

    a_less = False
    b_less = False

    for key in all_keys:
        val_a = clock_a.get(key, 0)
        val_b = clock_b.get(key, 0)

        if val_a < val_b:
            a_less = True
        elif val_b < val_a:
            b_less = True

    if a_less and not b_less:
        return "a_before_b"
    elif b_less and not a_less:
        return "b_before_a"
    elif not a_less and not b_less:
        return "equal"
    else:
        return "concurrent"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def trigger_sync(node_id: UUID | None = None) -> dict[str, Any]:
    """Manually trigger a sync operation.

    Args:
        node_id: Specific node to sync with (None = all active nodes)

    Returns:
        Sync result summary
    """
    manager = SyncManager()

    if node_id:
        # Sync with specific node
        node = get_node_by_id(node_id)
        if not node or not node.federation_endpoint:
            return {"success": False, "error": "Node not found or no endpoint"}

        trust = get_node_trust(node_id)
        if not trust:
            return {"success": False, "error": "No trust record for node"}

        state = get_sync_state(node_id)

        await manager._pull_from_node(
            node_id,
            node.federation_endpoint,
            trust.overall,
            state.last_received_cursor if state else None,
            state.last_sync_at if state else None,
        )

        return {"success": True, "synced_with": node.did}

    else:
        # Sync with all active nodes
        await manager._sync_with_peers()
        return {"success": True, "message": "Synced with all active nodes"}


def get_sync_status() -> dict[str, Any]:
    """Get overall sync status.

    Returns:
        Sync status summary
    """
    try:
        with get_cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_nodes,
                    COUNT(*) FILTER (WHERE ss.status = 'syncing') as syncing,
                    COUNT(*) FILTER (WHERE ss.status = 'error') as errors,
                    SUM(ss.beliefs_sent) as total_sent,
                    SUM(ss.beliefs_received) as total_received,
                    MAX(ss.last_sync_at) as last_sync
                FROM federation_nodes fn
                LEFT JOIN sync_state ss ON fn.id = ss.node_id
                WHERE fn.status = 'active'
            """)
            row = cur.fetchone()

            return {
                "total_nodes": row["total_nodes"] or 0,
                "syncing": row["syncing"] or 0,
                "errors": row["errors"] or 0,
                "total_beliefs_sent": row["total_sent"] or 0,
                "total_beliefs_received": row["total_received"] or 0,
                "last_sync": row["last_sync"].isoformat() if row["last_sync"] else None,
            }

    except Exception as e:
        logger.warning(f"Error getting sync status: {e}")
        return {"error": str(e)}
