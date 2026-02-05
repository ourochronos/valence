"""
Message Handler - Handles message routing for NodeClient.

This module manages:
- Message sending (direct, via router)
- Message receiving and delivery
- ACK tracking and handling
- Message queuing
- Traffic analysis mitigations (batching, jitter)

Extracted from NodeClient as part of god class decomposition (Issue #128).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from .discovery import RouterInfo
    from .node import RouterConnection, PendingAck, PendingMessage
    from .config import TrafficAnalysisMitigationConfig

from .crypto import encrypt_message, decrypt_message
from .messages import AckMessage, pad_message

logger = logging.getLogger(__name__)


@dataclass
class MessageHandlerConfig:
    """Configuration for MessageHandler."""
    
    # ACK configuration
    default_ack_timeout_ms: int = 30000  # 30 seconds
    max_seen_messages: int = 10000
    
    # Queue limits
    max_queue_size: int = 1000
    max_queue_age: float = 3600.0  # 1 hour


class MessageHandler:
    """
    Handles message sending and receiving for a node.
    
    Responsible for:
    - Encrypting and sending messages via routers
    - Receiving and decrypting incoming messages
    - ACK tracking and retries
    - Message queuing during failover
    - Traffic analysis mitigations
    """
    
    def __init__(
        self,
        node_id: str,
        private_key: "Ed25519PrivateKey",
        encryption_private_key: "X25519PrivateKey",
        config: Optional[MessageHandlerConfig] = None,
        traffic_mitigation_config: Optional["TrafficAnalysisMitigationConfig"] = None,
        on_message: Optional[Callable[[str, bytes], None]] = None,
        on_ack_timeout: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize the MessageHandler.
        
        Args:
            node_id: This node's ID (Ed25519 public key hex)
            private_key: Ed25519 private key for signing
            encryption_private_key: X25519 private key for decryption
            config: Message handler configuration
            traffic_mitigation_config: Traffic analysis mitigation config
            on_message: Callback for received messages (sender_id, content)
            on_ack_timeout: Callback for ACK timeouts (message_id, recipient_id)
        """
        self.node_id = node_id
        self.private_key = private_key
        self.encryption_private_key = encryption_private_key
        self.config = config or MessageHandlerConfig()
        self.traffic_mitigation_config = traffic_mitigation_config
        
        # Callbacks
        self.on_message = on_message
        self.on_ack_timeout = on_ack_timeout
        
        # ACK tracking
        self.pending_acks: Dict[str, "PendingAck"] = {}
        self.seen_messages: Set[str] = set()
        
        # Message queue
        self.message_queue: List["PendingMessage"] = []
        
        # Traffic analysis mitigation state
        self._message_batch: List[Dict[str, Any]] = []
        self._batch_lock: asyncio.Lock = asyncio.Lock()
        self._pending_batch_event: asyncio.Event = asyncio.Event()
        self._last_batch_flush: float = 0.0
        self._last_constant_rate_send: float = 0.0
        self._last_real_message_time: float = 0.0
        
        # Statistics
        self._stats: Dict[str, int] = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_queued": 0,
            "messages_dropped": 0,
            "messages_deduplicated": 0,
            "ack_successes": 0,
            "ack_failures": 0,
            "acks_sent": 0,
            "batched_messages": 0,
            "batch_flushes": 0,
            "jitter_delays_applied": 0,
            "total_jitter_ms": 0,
            "messages_with_jitter": 0,
            "constant_rate_padding_sent": 0,
            "bytes_padded": 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message handler statistics."""
        return {
            **self._stats,
            "pending_acks": len(self.pending_acks),
            "queued_messages": len(self.message_queue),
            "seen_messages_cached": len(self.seen_messages),
        }
    
    async def send_message(
        self,
        recipient_id: str,
        recipient_public_key: "X25519PublicKey",
        content: bytes,
        router_selector: Callable[[], Optional["RouterInfo"]],
        send_via_router: Callable,
        require_ack: bool = True,
        ack_timeout_ms: Optional[int] = None,
        bypass_mitigations: bool = False,
    ) -> str:
        """
        Send an encrypted message to a recipient.
        
        Args:
            recipient_id: Recipient's node ID
            recipient_public_key: Recipient's X25519 public key
            content: Raw message bytes
            router_selector: Function to select a router
            send_via_router: Function to send via selected router
            require_ack: Whether to require acknowledgment
            ack_timeout_ms: ACK timeout in milliseconds
            bypass_mitigations: Skip batching/jitter
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        timeout_ms = ack_timeout_ms or self.config.default_ack_timeout_ms
        
        # Update real message time for cover traffic
        self._last_real_message_time = time.time()
        
        # Apply traffic analysis mitigations if configured
        tam = self.traffic_mitigation_config
        if tam:
            # Pad message if enabled
            if tam.batching.enabled or tam.constant_rate.enabled:
                if tam.constant_rate.enabled and tam.constant_rate.pad_to_size > 0:
                    original_size = len(content)
                    content = pad_message(content, tam.constant_rate.pad_to_size)
                    if len(content) > original_size:
                        self._stats["bytes_padded"] += len(content) - original_size
            
            # Route through batching if enabled
            if tam.batching.enabled and not bypass_mitigations:
                return await self._add_to_batch(
                    message_id=message_id,
                    recipient_id=recipient_id,
                    recipient_public_key=recipient_public_key,
                    content=content,
                    require_ack=require_ack,
                    timeout_ms=timeout_ms,
                    router_selector=router_selector,
                    send_via_router=send_via_router,
                )
            
            # Apply timing jitter if enabled
            if tam.jitter.enabled and not bypass_mitigations:
                jitter_delay = tam.jitter.get_jitter_delay()
                if jitter_delay > 0:
                    self._stats["jitter_delays_applied"] += 1
                    self._stats["total_jitter_ms"] += int(jitter_delay * 1000)
                    self._stats["messages_with_jitter"] += 1
                    await asyncio.sleep(jitter_delay)
        
        # Direct send
        return await self._send_message_direct(
            message_id=message_id,
            recipient_id=recipient_id,
            recipient_public_key=recipient_public_key,
            content=content,
            require_ack=require_ack,
            timeout_ms=timeout_ms,
            router_selector=router_selector,
            send_via_router=send_via_router,
        )
    
    async def _send_message_direct(
        self,
        message_id: str,
        recipient_id: str,
        recipient_public_key: "X25519PublicKey",
        content: bytes,
        require_ack: bool,
        timeout_ms: int,
        router_selector: Callable,
        send_via_router: Callable,
    ) -> str:
        """Send a message directly without traffic analysis mitigations."""
        from .node import PendingAck, PendingMessage, NoRoutersAvailableError
        
        router = router_selector()
        if not router:
            # Queue message for later delivery
            if len(self.message_queue) >= self.config.max_queue_size:
                self._stats["messages_dropped"] += 1
                raise NoRoutersAvailableError(
                    "No routers available and message queue full"
                )
            
            self.message_queue.append(PendingMessage(
                message_id=message_id,
                recipient_id=recipient_id,
                content=content,
                recipient_public_key=recipient_public_key,
                queued_at=time.time(),
            ))
            self._stats["messages_queued"] += 1
            logger.debug(f"Message {message_id} queued (no routers available)")
            return message_id
        
        # Send via selected router
        await send_via_router(
            router,
            message_id,
            recipient_id,
            recipient_public_key,
            content,
            require_ack=require_ack,
        )
        
        # Track pending ACK if required
        if require_ack:
            self.pending_acks[message_id] = PendingAck(
                message_id=message_id,
                recipient_id=recipient_id,
                content=content,
                recipient_public_key=recipient_public_key,
                sent_at=time.time(),
                router_id=router.router_id,
                timeout_ms=timeout_ms,
            )
            asyncio.create_task(self._wait_for_ack(message_id, send_via_router, router_selector))
        
        return message_id
    
    async def _add_to_batch(
        self,
        message_id: str,
        recipient_id: str,
        recipient_public_key: "X25519PublicKey",
        content: bytes,
        require_ack: bool,
        timeout_ms: int,
        router_selector: Callable,
        send_via_router: Callable,
    ) -> str:
        """Add a message to the current batch for delayed sending."""
        async with self._batch_lock:
            batch_entry = {
                "message_id": message_id,
                "recipient_id": recipient_id,
                "recipient_public_key": recipient_public_key,
                "content": content,
                "require_ack": require_ack,
                "timeout_ms": timeout_ms,
                "queued_at": time.time(),
                "router_selector": router_selector,
                "send_via_router": send_via_router,
            }
            self._message_batch.append(batch_entry)
            self._stats["batched_messages"] += 1
            
            # Check if batch is full
            if self.traffic_mitigation_config and len(self._message_batch) >= self.traffic_mitigation_config.batching.max_batch_size:
                self._pending_batch_event.set()
        
        return message_id
    
    async def flush_batch(self) -> None:
        """Flush all messages in the current batch."""
        async with self._batch_lock:
            if not self._message_batch:
                return
            
            batch = self._message_batch.copy()
            self._message_batch.clear()
        
        # Randomize order if configured
        if self.traffic_mitigation_config and self.traffic_mitigation_config.batching.randomize_order:
            random.shuffle(batch)
        
        self._stats["batch_flushes"] += 1
        self._last_batch_flush = time.time()
        
        logger.debug(f"Flushing batch of {len(batch)} messages")
        
        for entry in batch:
            try:
                # Apply inter-message jitter if enabled
                if self.traffic_mitigation_config and self.traffic_mitigation_config.jitter.enabled:
                    jitter_delay = self.traffic_mitigation_config.jitter.get_jitter_delay()
                    if jitter_delay > 0:
                        self._stats["jitter_delays_applied"] += 1
                        self._stats["total_jitter_ms"] += int(jitter_delay * 1000)
                        await asyncio.sleep(jitter_delay)
                
                await self._send_message_direct(
                    message_id=entry["message_id"],
                    recipient_id=entry["recipient_id"],
                    recipient_public_key=entry["recipient_public_key"],
                    content=entry["content"],
                    require_ack=entry["require_ack"],
                    timeout_ms=entry["timeout_ms"],
                    router_selector=entry["router_selector"],
                    send_via_router=entry["send_via_router"],
                )
            except Exception as e:
                logger.warning(f"Failed to send batched message {entry['message_id']}: {e}")
    
    async def _wait_for_ack(
        self,
        message_id: str,
        send_via_router: Callable,
        router_selector: Callable,
    ) -> None:
        """Wait for an E2E ACK with timeout, then retry if needed."""
        pending = self.pending_acks.get(message_id)
        if not pending:
            return
        
        await asyncio.sleep(pending.timeout_ms / 1000)
        
        if message_id not in self.pending_acks:
            return  # ACK received
        
        pending = self.pending_acks.get(message_id)
        if not pending:
            return
        
        if pending.retries < 1:
            pending.retries += 1
            logger.debug(f"ACK timeout for {message_id}, retrying")
            await self._retry_message(message_id, send_via_router, router_selector)
        else:
            await self._retry_via_different_router(message_id, send_via_router, router_selector)
    
    async def _retry_message(
        self,
        message_id: str,
        send_via_router: Callable,
        router_selector: Callable,
    ) -> None:
        """Retry sending a message."""
        pending = self.pending_acks.get(message_id)
        if not pending:
            return
        
        try:
            router = router_selector()
            if router:
                await send_via_router(
                    router,
                    message_id,
                    pending.recipient_id,
                    pending.recipient_public_key,
                    pending.content,
                    require_ack=True,
                )
                pending.sent_at = time.time()
                asyncio.create_task(self._wait_for_ack(message_id, send_via_router, router_selector))
            else:
                await self._retry_via_different_router(message_id, send_via_router, router_selector)
        except Exception as e:
            logger.warning(f"Retry failed for {message_id}: {e}")
            await self._retry_via_different_router(message_id, send_via_router, router_selector)
    
    async def _retry_via_different_router(
        self,
        message_id: str,
        send_via_router: Callable,
        router_selector: Callable,
    ) -> None:
        """Retry sending via a different router."""
        pending = self.pending_acks.get(message_id)
        if not pending:
            return
        
        router = router_selector()
        if not router:
            self._handle_ack_failure(message_id, pending)
            return
        
        try:
            pending.retries += 1
            pending.router_id = router.router_id
            
            await send_via_router(
                router,
                message_id,
                pending.recipient_id,
                pending.recipient_public_key,
                pending.content,
                require_ack=True,
            )
            pending.sent_at = time.time()
            asyncio.create_task(self._wait_for_ack(message_id, send_via_router, router_selector))
        except Exception as e:
            logger.warning(f"Retry via different router failed: {e}")
            self._handle_ack_failure(message_id, pending)
    
    def _handle_ack_failure(self, message_id: str, pending: "PendingAck") -> None:
        """Handle final ACK failure after all retries."""
        self.pending_acks.pop(message_id, None)
        self._stats["ack_failures"] += 1
        
        if self.on_ack_timeout:
            try:
                self.on_ack_timeout(message_id, pending.recipient_id)
            except Exception as e:
                logger.warning(f"on_ack_timeout callback error: {e}")
        
        logger.warning(
            f"Message {message_id} to {pending.recipient_id[:16]}... "
            f"failed after {pending.retries} retries"
        )
    
    def handle_ack(self, message_id: str, success: bool = True) -> None:
        """Handle acknowledgment for a sent message."""
        pending = self.pending_acks.pop(message_id, None)
        
        if success and pending:
            self._stats["ack_successes"] += 1
            logger.debug(f"ACK received for {message_id}")
        elif not success:
            self._stats["ack_failures"] += 1
    
    def handle_e2e_ack(self, ack: AckMessage) -> None:
        """Handle an E2E acknowledgment from the recipient."""
        message_id = ack.original_message_id
        
        if message_id not in self.pending_acks:
            logger.debug(f"Received ACK for unknown message {message_id}")
            return
        
        pending = self.pending_acks.pop(message_id)
        self._stats["ack_successes"] += 1
        
        latency_ms = (ack.received_at - pending.sent_at) * 1000
        logger.debug(
            f"E2E ACK received for {message_id} (latency: {latency_ms:.1f}ms)"
        )
    
    def is_duplicate_message(self, message_id: str) -> bool:
        """Check if we've already seen this message."""
        if message_id in self.seen_messages:
            return True
        
        self.seen_messages.add(message_id)
        
        # Prune if too large
        if len(self.seen_messages) > self.config.max_seen_messages:
            seen_list = list(self.seen_messages)
            self.seen_messages = set(seen_list[len(seen_list)//2:])
        
        return False
    
    async def process_queue(
        self,
        router_selector: Callable,
        send_via_router: Callable,
    ) -> int:
        """
        Process queued messages.
        
        Returns:
            Number of messages processed
        """
        if not self.message_queue:
            return 0
        
        now = time.time()
        processed = []
        count = 0
        
        for i, msg in enumerate(self.message_queue):
            # Check message age
            if now - msg.queued_at > self.config.max_queue_age:
                processed.append(i)
                self._stats["messages_dropped"] += 1
                continue
            
            router = router_selector()
            if not router:
                break
            
            try:
                await send_via_router(
                    router,
                    msg.message_id,
                    msg.recipient_id,
                    msg.recipient_public_key,
                    msg.content,
                )
                processed.append(i)
                count += 1
            except Exception as e:
                msg.retries += 1
                if msg.retries >= msg.max_retries:
                    processed.append(i)
                    self._stats["messages_dropped"] += 1
                    logger.warning(f"Dropped message {msg.message_id} after retries: {e}")
        
        # Remove processed messages
        for i in reversed(processed):
            self.message_queue.pop(i)
        
        return count
    
    def sign_ack(self, message_id: str) -> str:
        """Sign a message_id to prove we received it."""
        signature = self.private_key.sign(message_id.encode())
        return signature.hex()
