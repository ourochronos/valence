"""Lightweight Federation Server for MVP.

A minimal HTTP server demonstrating federation between Valence nodes.
This is for the MVP demo - production code uses the full server/app.py.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .identity import (
    KeyPair,
    generate_keypair,
    create_key_did,
    sign_belief_content,
    verify_belief_signature,
    canonical_json,
)
from .peers import Peer, PeerStore
from ..compliance.pii_scanner import check_federation_allowed, scan_for_pii

logger = logging.getLogger(__name__)


@dataclass
class NodeIdentity:
    """This node's identity."""
    
    did: str
    keypair: KeyPair
    name: str | None = None
    endpoint: str | None = None
    
    @property
    def public_key_multibase(self) -> str:
        return self.keypair.public_key_multibase
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "did": self.did,
            "name": self.name,
            "endpoint": self.endpoint,
            "public_key_multibase": self.public_key_multibase,
        }


@dataclass
class LocalBelief:
    """A belief stored locally."""
    
    id: str
    content: str
    confidence: float
    domains: list[str]
    created_at: datetime = field(default_factory=datetime.now)
    signature: str | None = None  # Signed by origin node
    origin_did: str | None = None  # Who created it
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "confidence": self.confidence,
            "domains": self.domains,
            "created_at": self.created_at.isoformat(),
            "signature": self.signature,
            "origin_did": self.origin_did,
        }


class FederationNode:
    """A lightweight federation node for the MVP demo."""
    
    def __init__(
        self,
        name: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        keypair: KeyPair | None = None,
    ):
        """Initialize the federation node.
        
        Args:
            name: Human-readable node name
            host: Host to bind to
            port: Port to bind to
            keypair: Optional existing keypair (generates new if not provided)
        """
        self.name = name
        self.host = host
        self.port = port
        self.endpoint = f"http://{host}:{port}"
        
        # Identity
        self.keypair = keypair or generate_keypair()
        did = create_key_did(self.keypair.public_key_multibase)
        self.identity = NodeIdentity(
            did=did.full,
            keypair=self.keypair,
            name=name,
            endpoint=self.endpoint,
        )
        
        # Storage
        self.peer_store = PeerStore()
        self.beliefs: dict[str, LocalBelief] = {}
        
        # Server
        self.app = self._create_app()
        self._server = None
    
    def _create_app(self) -> Starlette:
        """Create the Starlette app with federation routes."""
        routes = [
            Route("/", self._info, methods=["GET"]),
            Route("/federation/introduce", self._introduce, methods=["POST"]),
            Route("/federation/share", self._share, methods=["POST"]),
            Route("/federation/query", self._query, methods=["POST"]),
            Route("/federation/peers", self._list_peers, methods=["GET"]),
        ]
        return Starlette(routes=routes)
    
    async def _info(self, request: Request) -> JSONResponse:
        """Node info endpoint."""
        return JSONResponse({
            "node": self.identity.to_dict(),
            "beliefs_count": len(self.beliefs),
            "peers_count": len(self.peer_store.list_peers()),
        })
    
    async def _introduce(self, request: Request) -> JSONResponse:
        """Handle peer introduction.
        
        POST /federation/introduce
        {
            "did": "did:vkb:key:...",
            "endpoint": "http://...",
            "public_key_multibase": "z...",
            "name": "Node Name"
        }
        
        Returns our identity in response.
        """
        try:
            body = await request.json()
            
            # Validate required fields
            required = ["did", "endpoint", "public_key_multibase"]
            for field_name in required:
                if field_name not in body:
                    return JSONResponse(
                        {"error": f"Missing required field: {field_name}"},
                        status_code=400,
                    )
            
            # Register the peer
            peer = self.peer_store.add_peer(
                did=body["did"],
                endpoint=body["endpoint"],
                public_key_multibase=body["public_key_multibase"],
                name=body.get("name"),
            )
            
            logger.info(f"[{self.name}] Received introduction from: {body.get('name', body['did'])}")
            
            # Return our identity
            return JSONResponse({
                "success": True,
                "message": f"Welcome to {self.name}!",
                "node": self.identity.to_dict(),
            })
            
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            logger.exception("Error in introduce endpoint")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    async def _share(self, request: Request) -> JSONResponse:
        """Handle shared belief.
        
        POST /federation/share
        {
            "belief": {
                "id": "...",
                "content": "...",
                "confidence": 0.8,
                "domains": ["..."],
                "origin_did": "did:vkb:key:...",
                "signature": "base64..."
            },
            "sender_did": "did:vkb:key:..."
        }
        """
        try:
            body = await request.json()
            
            belief_data = body.get("belief")
            sender_did = body.get("sender_did")
            
            if not belief_data or not sender_did:
                return JSONResponse(
                    {"error": "Missing belief or sender_did"},
                    status_code=400,
                )
            
            # Verify sender is known peer
            peer = self.peer_store.get_peer(sender_did)
            if not peer:
                return JSONResponse(
                    {"error": f"Unknown sender: {sender_did}"},
                    status_code=403,
                )
            
            # Verify signature if present
            origin_did = belief_data.get("origin_did", sender_did)
            signature = belief_data.get("signature")
            
            if signature:
                # Get origin node's public key
                if origin_did == sender_did:
                    origin_key = peer.public_key_multibase
                else:
                    origin_peer = self.peer_store.get_peer(origin_did)
                    if not origin_peer:
                        return JSONResponse(
                            {"error": f"Unknown origin node: {origin_did}"},
                            status_code=403,
                        )
                    origin_key = origin_peer.public_key_multibase
                
                # Verify signature
                signable = {
                    "id": belief_data["id"],
                    "content": belief_data["content"],
                    "confidence": belief_data["confidence"],
                    "domains": belief_data.get("domains", []),
                    "origin_did": origin_did,
                }
                
                if not verify_belief_signature(signable, signature, origin_key):
                    logger.warning(f"[{self.name}] Invalid signature from {sender_did}")
                    return JSONResponse(
                        {"error": "Invalid signature"},
                        status_code=403,
                    )
            
            # Store the belief
            belief = LocalBelief(
                id=belief_data["id"],
                content=belief_data["content"],
                confidence=belief_data["confidence"],
                domains=belief_data.get("domains", []),
                signature=signature,
                origin_did=origin_did,
            )
            self.beliefs[belief.id] = belief
            
            # Update peer trust
            self.peer_store.record_belief_received(sender_did)
            
            logger.info(f"[{self.name}] Received belief from {peer.name or sender_did}: {belief.content[:50]}...")
            
            return JSONResponse({
                "success": True,
                "belief_id": belief.id,
                "message": "Belief accepted",
            })
            
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            logger.exception("Error in share endpoint")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    async def _query(self, request: Request) -> JSONResponse:
        """Handle belief query.
        
        POST /federation/query
        {
            "query": "search terms",
            "domains": ["optional", "domain", "filter"],
            "min_confidence": 0.5,
            "sender_did": "did:vkb:key:..."
        }
        """
        try:
            body = await request.json()
            
            query_text = body.get("query", "")
            domains = body.get("domains", [])
            min_confidence = body.get("min_confidence", 0.0)
            sender_did = body.get("sender_did")
            
            # Verify sender is known peer
            if sender_did:
                peer = self.peer_store.get_peer(sender_did)
                if not peer:
                    return JSONResponse(
                        {"error": f"Unknown sender: {sender_did}"},
                        status_code=403,
                    )
                self.peer_store.record_query(sender_did, "received")
            
            # Simple search (in production, use vector similarity)
            results = []
            query_lower = query_text.lower()
            
            for belief in self.beliefs.values():
                # Check confidence threshold
                if belief.confidence < min_confidence:
                    continue
                
                # Check domain filter
                if domains and not any(d in belief.domains for d in domains):
                    continue
                
                # Simple text match (MVP - no vector search)
                if query_lower and query_lower not in belief.content.lower():
                    continue
                
                results.append(belief.to_dict())
            
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            logger.info(f"[{self.name}] Query '{query_text}' returned {len(results)} results")
            
            return JSONResponse({
                "success": True,
                "query": query_text,
                "results": results[:20],  # Limit to 20
                "total": len(results),
            })
            
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            logger.exception("Error in query endpoint")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    async def _list_peers(self, request: Request) -> JSONResponse:
        """List known peers.
        
        GET /federation/peers
        """
        peers = [p.to_dict() for p in self.peer_store.list_peers()]
        return JSONResponse({
            "peers": peers,
            "count": len(peers),
        })
    
    # ==========================================================================
    # Client methods (for calling other nodes)
    # ==========================================================================
    
    async def introduce_to(self, endpoint: str) -> dict[str, Any]:
        """Introduce ourselves to another node.
        
        Args:
            endpoint: The other node's base URL
            
        Returns:
            Response with the other node's identity
        """
        import aiohttp
        
        url = f"{endpoint}/federation/introduce"
        payload = {
            "did": self.identity.did,
            "endpoint": self.endpoint,
            "public_key_multibase": self.identity.public_key_multibase,
            "name": self.name,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                
                if result.get("success"):
                    # Register the other node as a peer
                    node_info = result.get("node", {})
                    self.peer_store.add_peer(
                        did=node_info.get("did"),
                        endpoint=node_info.get("endpoint"),
                        public_key_multibase=node_info.get("public_key_multibase"),
                        name=node_info.get("name"),
                    )
                    logger.info(f"[{self.name}] Introduced to: {node_info.get('name')}")
                
                return result
    
    async def share_belief(
        self,
        peer_did: str,
        content: str,
        confidence: float = 0.8,
        domains: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Share a belief with a peer.
        
        Implements PII scanning per COMPLIANCE.md §1 (Issue #35):
        - Blocks L3+ content from auto-federation
        - Use force=True to override (with explicit confirmation)
        
        Args:
            peer_did: The peer's DID
            content: Belief content
            confidence: Confidence level (0.0-1.0)
            domains: Optional domain tags
            force: Override L3 (Personal) classification blocks
            
        Returns:
            Response from the peer
        """
        import aiohttp
        
        # PII check before federation (Issue #35)
        allowed, scan_result = check_federation_allowed(content, force=force)
        if not allowed:
            return {
                "success": False,
                "error": "Content blocked from federation due to PII",
                "pii_scan": scan_result.to_dict(),
                "hint": "Use force=True to override L3 blocks (requires explicit confirmation)",
            }
        
        peer = self.peer_store.get_peer(peer_did)
        if not peer:
            return {"success": False, "error": f"Unknown peer: {peer_did}"}
        
        # Create and sign the belief
        belief_id = str(uuid4())
        domains = domains or []
        
        signable = {
            "id": belief_id,
            "content": content,
            "confidence": confidence,
            "domains": domains,
            "origin_did": self.identity.did,
        }
        
        signature = sign_belief_content(signable, self.keypair.private_key_bytes)
        
        # Store locally
        local_belief = LocalBelief(
            id=belief_id,
            content=content,
            confidence=confidence,
            domains=domains,
            signature=signature,
            origin_did=self.identity.did,
        )
        self.beliefs[belief_id] = local_belief
        
        # Send to peer
        url = f"{peer.endpoint}/federation/share"
        payload = {
            "belief": {
                "id": belief_id,
                "content": content,
                "confidence": confidence,
                "domains": domains,
                "origin_did": self.identity.did,
                "signature": signature,
            },
            "sender_did": self.identity.did,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                
                if result.get("success"):
                    self.peer_store.record_belief_sent(peer_did)
                    logger.info(f"[{self.name}] Shared belief with {peer.name or peer_did}")
                
                return result
    
    async def query_peer(
        self,
        peer_did: str,
        query: str,
        domains: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> dict[str, Any]:
        """Query a peer for beliefs.
        
        Args:
            peer_did: The peer's DID
            query: Search query text
            domains: Optional domain filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            Query results from the peer
        """
        import aiohttp
        
        peer = self.peer_store.get_peer(peer_did)
        if not peer:
            return {"success": False, "error": f"Unknown peer: {peer_did}"}
        
        url = f"{peer.endpoint}/federation/query"
        payload = {
            "query": query,
            "domains": domains or [],
            "min_confidence": min_confidence,
            "sender_did": self.identity.did,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                
                if result.get("success"):
                    self.peer_store.record_query(peer_did, "sent")
                    logger.info(f"[{self.name}] Queried {peer.name or peer_did}: {len(result.get('results', []))} results")
                
                return result
    
    # ==========================================================================
    # Server lifecycle
    # ==========================================================================
    
    async def start(self) -> None:
        """Start the server."""
        import uvicorn
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()
    
    def start_background(self) -> asyncio.Task:
        """Start the server in the background.
        
        Returns:
            The asyncio task running the server
        """
        return asyncio.create_task(self.start())
    
    async def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.should_exit = True


async def create_node(
    name: str,
    port: int,
    host: str = "127.0.0.1",
) -> FederationNode:
    """Create and return a federation node.
    
    Args:
        name: Node name
        port: Port to run on
        host: Host to bind to
        
    Returns:
        A FederationNode instance (not started yet)
    """
    return FederationNode(name=name, host=host, port=port)
