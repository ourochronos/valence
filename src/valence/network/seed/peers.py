"""
Seed peering and gossip management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web

from valence.network.seed.router import RouterRecord

if TYPE_CHECKING:
    from valence.network.seed.seed_node import SeedNode

# Use cryptographically secure RNG for router sampling (prevents predictable patterns)
_secure_random = secrets.SystemRandom()

logger = logging.getLogger(__name__)


class SeedPeerManager:
    """
    Manages peering and gossip between seed nodes.

    Implements:
    - Periodic gossip exchanges with peer seeds
    - Router registry synchronization (subset-based, not full)
    - Deduplication of routers by ID
    - Filtering of stale/unhealthy routers before propagation

    Gossip protocol:
    - Each seed maintains a list of peer seed URLs
    - Periodically (configurable interval), seeds exchange router subsets
    - Only healthy, fresh routers are shared
    - Routers are deduplicated by router_id (newer wins)
    """

    def __init__(self, seed: SeedNode):
        self.seed = seed
        self._running = False
        self._gossip_task: asyncio.Task | None = None
        self._peer_states: dict[str, dict[str, Any]] = {}  # peer_url -> state

    @property
    def peer_seeds(self) -> list[str]:
        """Get list of peer seed URLs."""
        return self.seed.config.peer_seeds

    @property
    def gossip_interval(self) -> float:
        """Get gossip interval in seconds."""
        return self.seed.config.gossip_interval_seconds

    @property
    def batch_size(self) -> int:
        """Get max routers per gossip exchange."""
        return self.seed.config.gossip_batch_size

    async def start(self) -> None:
        """Start the gossip loop."""
        if self._running:
            return

        if not self.seed.config.gossip_enabled:
            logger.info("Gossip disabled by configuration")
            return

        if not self.peer_seeds:
            logger.info("No peer seeds configured, gossip not started")
            return

        self._running = True
        self._gossip_task = asyncio.create_task(self._gossip_loop())
        logger.info(f"Seed peer manager started with {len(self.peer_seeds)} peers, gossip interval={self.gossip_interval}s")

    async def stop(self) -> None:
        """Stop the gossip loop."""
        self._running = False
        if self._gossip_task:
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass
            self._gossip_task = None
        logger.info("Seed peer manager stopped")

    async def _gossip_loop(self) -> None:
        """Periodically gossip with peer seeds."""
        # Initial delay to let seed start up
        await asyncio.sleep(5)

        while self._running:
            try:
                await self._gossip_round()
            except Exception as e:  # noqa: BLE001 - gossip loop must not crash
                logger.error(f"Gossip round failed: {e}")

            await asyncio.sleep(self.gossip_interval)

    async def _gossip_round(self) -> None:
        """Execute one round of gossip with all peers."""
        if not self.peer_seeds:
            return

        logger.debug(f"Starting gossip round with {len(self.peer_seeds)} peers")

        # Gather routers to share
        routers_to_share = self._select_routers_for_gossip()

        if not routers_to_share:
            logger.debug("No routers to share in gossip")

        # Exchange with each peer concurrently
        tasks = [self._exchange_with_peer(peer_url, routers_to_share) for peer_url in self.peer_seeds]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if r is True)
        logger.info(f"Gossip round complete: {successful}/{len(self.peer_seeds)} peers exchanged, shared {len(routers_to_share)} routers")

    def _select_routers_for_gossip(self) -> list[dict[str, Any]]:
        """
        Select a subset of routers to share in gossip.

        Filters:
        - Only healthy routers (based on health monitor)
        - Only fresh routers (seen within max_router_age)
        - Limited to batch_size
        - Randomized selection for fairness
        """
        now = time.time()
        max_age = self.seed.config.gossip_max_router_age_seconds

        candidates = []
        for router_id, router in self.seed.router_registry.items():
            # Check freshness
            last_seen = router.health.get("last_seen", 0)
            if now - last_seen > max_age:
                continue

            # Check health via health monitor
            if not self.seed.health_monitor.is_healthy_for_discovery(router_id):
                continue

            # Check legacy health status
            if not self.seed._is_healthy(router, now):
                continue

            candidates.append(router)

        # Randomize and limit using cryptographically secure RNG
        if len(candidates) > self.batch_size:
            candidates = _secure_random.sample(candidates, self.batch_size)

        return [r.to_dict() for r in candidates]

    async def _exchange_with_peer(
        self,
        peer_url: str,
        routers_to_share: list[dict[str, Any]],
    ) -> bool:
        """
        Exchange router info with a single peer seed.

        Returns True if exchange was successful.
        """
        try:
            # Normalize URL
            if not peer_url.startswith("http"):
                peer_url = f"http://{peer_url}"

            exchange_url = f"{peer_url.rstrip('/')}/gossip/exchange"

            # Include revocations in gossip exchange (Issue #121)
            revocations_to_share = []
            if hasattr(self.seed, "revocation_manager"):
                revocations_to_share = self.seed.revocation_manager.get_revocations_for_gossip()

            payload = {
                "seed_id": self.seed.seed_id,
                "timestamp": time.time(),
                "routers": routers_to_share,
                "revocations": revocations_to_share,  # Issue #121
            }

            timeout = aiohttp.ClientTimeout(total=self.seed.config.gossip_timeout_seconds)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(exchange_url, json=payload) as resp:
                    if resp.status != 200:
                        logger.warning(f"Gossip exchange with {peer_url} failed: status {resp.status}")
                        self._update_peer_state(peer_url, success=False)
                        return False

                    data = await resp.json()

                    # Process received routers
                    received_routers = data.get("routers", [])
                    merged_count = self._merge_routers(received_routers, peer_url)

                    # Process received revocations (Issue #121)
                    revocations_merged = 0
                    received_revocations = data.get("revocations", [])
                    if received_revocations and hasattr(self.seed, "revocation_manager"):
                        revocations_merged = self.seed.revocation_manager.process_gossip_revocations(received_revocations)

                    self._update_peer_state(peer_url, success=True)

                    logger.debug(
                        f"Gossip with {peer_url}: sent {len(routers_to_share)} routers, "
                        f"received {len(received_routers)}, merged {merged_count}; "
                        f"sent {len(revocations_to_share)} revocations, "
                        f"received {len(received_revocations)}, merged {revocations_merged}"
                    )

                    return True

        except TimeoutError:
            logger.warning(f"Gossip exchange with {peer_url} timed out")
            self._update_peer_state(peer_url, success=False, error="timeout")
            return False
        except (OSError, aiohttp.ClientError) as e:
            logger.warning(f"Gossip exchange with {peer_url} failed: {e}")
            self._update_peer_state(peer_url, success=False, error=str(e))
            return False

    def _merge_routers(
        self,
        received_routers: list[dict[str, Any]],
        source_peer: str,
    ) -> int:
        """
        Merge received routers into local registry.

        Deduplication rules:
        - If router_id doesn't exist, add it
        - If router_id exists, keep the one with more recent last_seen

        Returns number of routers actually merged (new or updated).
        """
        now = time.time()
        max_age = self.seed.config.gossip_max_router_age_seconds
        merged = 0

        for router_data in received_routers:
            try:
                router_id = router_data.get("router_id")
                if not router_id:
                    continue

                # Check freshness of received router
                last_seen = router_data.get("health", {}).get("last_seen", 0)
                if now - last_seen > max_age:
                    logger.debug(f"Skipping stale router {router_id[:20]}... from gossip (age={now - last_seen:.0f}s)")
                    continue

                # Check if we should merge
                existing = self.seed.router_registry.get(router_id)

                if existing is None:
                    # New router - add it
                    router = RouterRecord.from_dict(router_data)
                    router.source_ip = f"gossip:{source_peer}"
                    self.seed.router_registry[router_id] = router

                    # Initialize health state
                    self.seed.health_monitor.record_heartbeat(router_id)

                    merged += 1
                    logger.debug(f"Added router {router_id[:20]}... from gossip (peer={source_peer})")
                else:
                    # Existing router - compare freshness
                    existing_last_seen = existing.health.get("last_seen", 0)

                    if last_seen > existing_last_seen:
                        # Received router is newer - update
                        router = RouterRecord.from_dict(router_data)
                        router.source_ip = existing.source_ip  # Preserve original source
                        router.registered_at = existing.registered_at  # Preserve registration time
                        self.seed.router_registry[router_id] = router

                        merged += 1
                        logger.debug(
                            f"Updated router {router_id[:20]}... from gossip (peer={source_peer}, delta={last_seen - existing_last_seen:.0f}s)"
                        )

            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Failed to merge router from gossip: {e}")
                continue

        return merged

    def _update_peer_state(
        self,
        peer_url: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Update state tracking for a peer."""
        now = time.time()

        if peer_url not in self._peer_states:
            self._peer_states[peer_url] = {
                "first_seen": now,
                "successful_exchanges": 0,
                "failed_exchanges": 0,
                "last_success": None,
                "last_failure": None,
                "last_error": None,
            }

        state = self._peer_states[peer_url]

        if success:
            state["successful_exchanges"] += 1
            state["last_success"] = now
        else:
            state["failed_exchanges"] += 1
            state["last_failure"] = now
            state["last_error"] = error

    def get_peer_stats(self) -> dict[str, Any]:
        """Get statistics about peer connections."""
        return {
            "peer_count": len(self.peer_seeds),
            "gossip_enabled": self.seed.config.gossip_enabled,
            "gossip_interval_seconds": self.gossip_interval,
            "peer_states": {
                url: {
                    "successful": state["successful_exchanges"],
                    "failed": state["failed_exchanges"],
                    "last_success": state["last_success"],
                    "last_error": state["last_error"],
                }
                for url, state in self._peer_states.items()
            },
        }

    async def handle_gossip_exchange(self, request: web.Request) -> web.Response:
        """
        Handle incoming gossip exchange from peer seed.

        POST /gossip/exchange
        {
            "seed_id": "peer-seed-001",
            "timestamp": 1706789012.345,
            "routers": [...]
        }

        Response:
        {
            "seed_id": "this-seed-id",
            "timestamp": 1706789012.456,
            "routers": [...]  // Our routers to share back
        }
        """
        try:
            data = await request.json()
        except (ValueError, json.JSONDecodeError) as e:
            return web.json_response(
                {"status": "error", "reason": "invalid_json", "detail": str(e)},
                status=400,
            )

        peer_seed_id = data.get("seed_id", "unknown")
        received_routers = data.get("routers", [])
        received_revocations = data.get("revocations", [])

        # Merge received routers
        merged_count = self._merge_routers(received_routers, peer_seed_id)

        # Process received revocations (Issue #121)
        revocations_merged = 0
        if received_revocations and hasattr(self.seed, "revocation_manager"):
            revocations_merged = self.seed.revocation_manager.process_gossip_revocations(received_revocations)

        # Select our routers to send back
        routers_to_share = self._select_routers_for_gossip()

        # Include revocations in response (Issue #121)
        revocations_to_share = []
        if hasattr(self.seed, "revocation_manager"):
            revocations_to_share = self.seed.revocation_manager.get_revocations_for_gossip()

        logger.debug(
            f"Gossip exchange from {peer_seed_id}: "
            f"received {len(received_routers)} routers, merged {merged_count}; "
            f"received {len(received_revocations)} revocations, merged {revocations_merged}"
        )

        return web.json_response(
            {
                "seed_id": self.seed.seed_id,
                "timestamp": time.time(),
                "routers": routers_to_share,
                "revocations": revocations_to_share,
            }
        )
