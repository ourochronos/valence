"""
Router registry data model for seed nodes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RouterRecord:
    """Record of a registered router."""
    
    router_id: str  # Ed25519 public key (hex)
    endpoints: List[str]  # ["ip:port", ...]
    capacity: Dict[str, Any]  # {max_connections, current_load_pct, bandwidth_mbps}
    health: Dict[str, Any]  # {last_seen, uptime_pct, avg_latency_ms, status}
    regions: List[str]  # Geographic regions served
    features: List[str]  # Supported features/protocols
    registered_at: float  # Unix timestamp
    router_signature: str  # Signature of registration data
    proof_of_work: Optional[Dict[str, Any]] = None  # PoW proof
    source_ip: Optional[str] = None  # IP address that registered this router
    region: Optional[str] = None  # ISO 3166-1 alpha-2 country code (e.g., "US", "DE")
    coordinates: Optional[List[float]] = None  # [latitude, longitude]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "router_id": self.router_id,
            "endpoints": self.endpoints,
            "capacity": self.capacity,
            "health": self.health,
            "regions": self.regions,
            "features": self.features,
            "registered_at": self.registered_at,
            "router_signature": self.router_signature,
        }
        if self.region:
            result["region"] = self.region
        if self.coordinates:
            result["coordinates"] = self.coordinates
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouterRecord":
        """Create from dictionary."""
        coords = data.get("coordinates")
        if coords and isinstance(coords, (list, tuple)) and len(coords) == 2:
            coordinates = [float(coords[0]), float(coords[1])]
        else:
            coordinates = None
            
        return cls(
            router_id=data["router_id"],
            endpoints=data.get("endpoints", []),
            capacity=data.get("capacity", {}),
            health=data.get("health", {}),
            regions=data.get("regions", []),
            features=data.get("features", []),
            registered_at=data.get("registered_at", time.time()),
            router_signature=data.get("router_signature", ""),
            proof_of_work=data.get("proof_of_work"),
            source_ip=data.get("source_ip"),
            region=data.get("region"),
            coordinates=coordinates,
        )
