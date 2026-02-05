"""
Seed node configuration and regional routing utilities.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SeedConfig:
    """Configuration for seed node."""
    
    host: str = "0.0.0.0"
    port: int = 8470
    
    # Router health thresholds
    min_uptime_pct: float = 90.0  # Minimum uptime to be considered healthy
    max_stale_seconds: float = 600.0  # Max time since last heartbeat
    
    # Selection weights for load-aware balancing
    weight_health: float = 0.25
    weight_load: float = 0.35  # Primary load factor (connection ratio)
    weight_queue: float = 0.15  # Queue depth penalty
    weight_throughput: float = 0.10  # Message rate consideration
    weight_region: float = 0.10
    weight_random: float = 0.05
    
    # Thundering herd prevention
    recovery_ramp_duration: float = 300.0  # 5 min ramp-up period for recovered routers
    recovery_initial_weight: float = 0.2  # Start at 20% weight when coming back online
    
    # Legacy weight (for backward compatibility)
    weight_capacity: float = 0.3  # Deprecated, use weight_load
    
    # Seed node identity
    seed_id: Optional[str] = None
    
    # Known other seeds for redundancy
    known_seeds: List[str] = field(default_factory=list)
    
    # Proof-of-work difficulty (leading zero bits required)
    pow_difficulty_base: int = 16  # First router from IP
    pow_difficulty_second: int = 20  # Second router from same IP
    pow_difficulty_third_plus: int = 24  # Third+ router from same IP
    
    # Enable/disable signature and PoW verification
    verify_signatures: bool = True
    verify_pow: bool = True
    
    # Endpoint probing settings
    probe_endpoints: bool = True
    probe_timeout_seconds: float = 5.0
    
    # Seed peering / gossip settings
    peer_seeds: List[str] = field(default_factory=list)  # URLs of peer seeds
    gossip_enabled: bool = True
    gossip_interval_seconds: float = 300.0  # 5 minutes
    gossip_batch_size: int = 20  # Max routers per gossip exchange
    gossip_timeout_seconds: float = 10.0  # Timeout for gossip requests
    gossip_max_router_age_seconds: float = 1800.0  # Don't propagate routers older than 30 min
    
    # Misbehavior report settings (Issue #119)
    misbehavior_reports_enabled: bool = True
    misbehavior_min_reports_to_flag: int = 3  # Minimum unique reporters to flag a router
    misbehavior_report_window_seconds: float = 3600.0  # Time window for counting reports
    misbehavior_verify_reporter_signature: bool = True  # Verify reporter signatures
    misbehavior_max_reports_per_router: int = 100  # Limit stored reports per router
    misbehavior_flag_severity_threshold: float = 0.5  # Min avg severity to flag
    
    # ==========================================================================
    # Seed Revocation Settings (Issue #121)
    # ==========================================================================
    
    # Enable/disable seed revocation handling
    seed_revocation_enabled: bool = True
    
    # Verify revocation signatures (set False only for testing)
    seed_revocation_verify_signatures: bool = True
    
    # Path to out-of-band revocation list file (optional)
    # If set, the seed will load revocations from this file on startup
    # and periodically check for updates
    seed_revocation_list_path: Optional[str] = None
    
    # Interval to check for revocation list file updates (seconds)
    seed_revocation_list_check_interval: float = 3600.0  # 1 hour
    
    # Whether to propagate revocations via gossip
    seed_revocation_gossip_enabled: bool = True
    
    # Maximum age of revocation to accept (prevents replay of old revocations)
    seed_revocation_max_age_seconds: float = 86400.0 * 30  # 30 days
    
    # Trusted authority public keys for revocation list verification (hex-encoded)
    # If empty, only self-signed revocations are accepted
    seed_revocation_trusted_authorities: List[str] = field(default_factory=list)
    
    # ==========================================================================
    # Sybil Resistance Settings (Issue #117)
    # ==========================================================================
    
    # Rate limiting per IP/subnet
    rate_limit_enabled: bool = True
    rate_limit_window_seconds: float = 3600.0  # 1 hour window
    rate_limit_max_per_ip: int = 5  # Max registrations per IP per window
    rate_limit_max_per_subnet: int = 10  # Max registrations per /24 subnet per window
    rate_limit_cooldown_seconds: float = 300.0  # 5 min cooldown between registrations from same IP
    
    # Reputation system for new routers
    reputation_enabled: bool = True
    reputation_initial_score: float = 0.5  # New routers start at 50% trust (0.0 - 1.0)
    reputation_decay_period_hours: float = 24.0  # Hours to reach full trust
    reputation_min_score_for_discovery: float = 0.3  # Minimum score to be included in discovery
    reputation_boost_per_heartbeat: float = 0.01  # Score boost per successful heartbeat
    reputation_penalty_missed_heartbeat: float = 0.05  # Score penalty per missed heartbeat
    reputation_max_score: float = 1.0  # Maximum reputation score
    
    # Correlated behavior detection
    correlation_detection_enabled: bool = True
    correlation_heartbeat_window_seconds: float = 30.0  # Heartbeats within this window are suspicious
    correlation_min_suspicious_events: int = 5  # Min correlated events to flag
    correlation_endpoint_similarity_threshold: float = 0.8  # 80% endpoint similarity is suspicious
    correlation_penalty_score: float = 0.2  # Reputation penalty for correlated behavior
    
    # Adaptive PoW difficulty based on network-wide registration rate
    adaptive_pow_enabled: bool = True
    adaptive_pow_threshold_per_hour: int = 100  # If >100 registrations/hour, increase difficulty
    adaptive_pow_max_difficulty: int = 28  # Maximum difficulty bits
    adaptive_pow_difficulty_step: int = 2  # Increase difficulty by this many bits when threshold hit
    
    def __post_init__(self):
        if self.seed_id is None:
            self.seed_id = f"seed-{secrets.token_hex(8)}"


# =============================================================================
# REGIONAL ROUTING
# =============================================================================


# ISO 3166-1 alpha-2 country code to continent mapping
# Continents: AF (Africa), AN (Antarctica), AS (Asia), EU (Europe),
#             NA (North America), OC (Oceania), SA (South America)
COUNTRY_TO_CONTINENT: Dict[str, str] = {
    # North America
    "US": "NA", "CA": "NA", "MX": "NA", "GT": "NA", "BZ": "NA", "HN": "NA",
    "SV": "NA", "NI": "NA", "CR": "NA", "PA": "NA", "CU": "NA", "JM": "NA",
    "HT": "NA", "DO": "NA", "PR": "NA", "BS": "NA", "BB": "NA", "TT": "NA",
    
    # South America
    "BR": "SA", "AR": "SA", "CL": "SA", "CO": "SA", "PE": "SA", "VE": "SA",
    "EC": "SA", "BO": "SA", "PY": "SA", "UY": "SA", "GY": "SA", "SR": "SA",
    
    # Europe
    "GB": "EU", "DE": "EU", "FR": "EU", "IT": "EU", "ES": "EU", "PT": "EU",
    "NL": "EU", "BE": "EU", "CH": "EU", "AT": "EU", "SE": "EU", "NO": "EU",
    "DK": "EU", "FI": "EU", "IE": "EU", "PL": "EU", "CZ": "EU", "SK": "EU",
    "HU": "EU", "RO": "EU", "BG": "EU", "GR": "EU", "HR": "EU", "SI": "EU",
    "RS": "EU", "UA": "EU", "BY": "EU", "LT": "EU", "LV": "EU", "EE": "EU",
    "LU": "EU", "MT": "EU", "CY": "EU", "IS": "EU", "AL": "EU", "MK": "EU",
    "BA": "EU", "ME": "EU", "MD": "EU", "XK": "EU",
    
    # Asia
    "CN": "AS", "JP": "AS", "KR": "AS", "IN": "AS", "ID": "AS", "TH": "AS",
    "VN": "AS", "MY": "AS", "SG": "AS", "PH": "AS", "TW": "AS", "HK": "AS",
    "BD": "AS", "PK": "AS", "LK": "AS", "NP": "AS", "MM": "AS", "KH": "AS",
    "LA": "AS", "MN": "AS", "KZ": "AS", "UZ": "AS", "TM": "AS", "TJ": "AS",
    "KG": "AS", "AZ": "AS", "AM": "AS", "GE": "AS",
    
    # Middle East (part of Asia)
    "TR": "AS", "IR": "AS", "IQ": "AS", "SA": "AS", "AE": "AS", "IL": "AS",
    "JO": "AS", "LB": "AS", "SY": "AS", "YE": "AS", "OM": "AS", "KW": "AS",
    "QA": "AS", "BH": "AS", "PS": "AS", "AF": "AS",
    
    # Africa
    "ZA": "AF", "EG": "AF", "NG": "AF", "KE": "AF", "ET": "AF", "GH": "AF",
    "TZ": "AF", "UG": "AF", "MA": "AF", "DZ": "AF", "TN": "AF", "LY": "AF",
    "SD": "AF", "AO": "AF", "MZ": "AF", "ZW": "AF", "ZM": "AF", "BW": "AF",
    "NA": "AF", "CI": "AF", "CM": "AF", "SN": "AF", "ML": "AF", "NE": "AF",
    "BF": "AF", "MG": "AF", "MW": "AF", "RW": "AF", "SO": "AF", "CD": "AF",
    "CG": "AF", "GA": "AF", "MU": "AF", "SC": "AF", "CV": "AF",
    
    # Oceania
    "AU": "OC", "NZ": "OC", "FJ": "OC", "PG": "OC", "NC": "OC", "VU": "OC",
    "WS": "OC", "TO": "OC", "PF": "OC", "GU": "OC", "FM": "OC", "SB": "OC",
    
    # Russia spans Europe/Asia - categorize as Europe for routing
    "RU": "EU",
}


def get_continent(country_code: Optional[str]) -> Optional[str]:
    """
    Get continent code from ISO 3166-1 alpha-2 country code.
    
    Args:
        country_code: Two-letter country code (e.g., "US", "DE")
        
    Returns:
        Two-letter continent code or None if not found
    """
    if not country_code:
        return None
    return COUNTRY_TO_CONTINENT.get(country_code.upper())


def compute_region_score(
    router_region: Optional[str],
    preferred_region: Optional[str],
) -> float:
    """
    Compute region match score for router selection.
    
    Scoring tiers:
    - Same region (country): 1.0 (full match)
    - Same continent: 0.5 (partial match)
    - Different continent or unknown: 0.0 (no match)
    
    Args:
        router_region: Router's region (country code)
        preferred_region: Requested region preference (country code)
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not preferred_region or not router_region:
        return 0.0
    
    # Normalize to uppercase
    router_region = router_region.upper()
    preferred_region = preferred_region.upper()
    
    # Same country = full match
    if router_region == preferred_region:
        return 1.0
    
    # Same continent = partial match
    router_continent = get_continent(router_region)
    preferred_continent = get_continent(preferred_region)
    
    if router_continent and preferred_continent and router_continent == preferred_continent:
        return 0.5
    
    # Different continent or unknown
    return 0.0
