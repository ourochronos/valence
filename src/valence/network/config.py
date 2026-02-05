"""
Network configuration for traffic analysis mitigations.

This module provides configurable privacy settings to protect against
traffic analysis attacks. These mitigations complement cover traffic
(Issue #116) by focusing on timing patterns.

Privacy Levels:
- LOW: Minimal mitigations, best latency
- MEDIUM: Moderate batching and jitter, balanced tradeoff
- HIGH: Aggressive batching, significant jitter, reduced latency
- PARANOID: Constant-rate sending with padding, maximum privacy

Issue #120 - Traffic Analysis Mitigations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import secrets

# Use cryptographically secure RNG for timing jitter (traffic analysis resistance)
_secure_random = secrets.SystemRandom()


class PrivacyLevel(Enum):
    """
    Privacy level presets for traffic analysis protection.
    
    Each level provides a different privacy/latency tradeoff:
    
    LOW: For applications where latency is critical and traffic patterns
         are not sensitive. Minimal protection.
         - No batching
         - No jitter
         - No constant-rate
    
    MEDIUM: Balanced protection for general use. Some latency impact
            in exchange for reasonable privacy.
            - Small batches (2-4 messages)
            - Short batch windows (1-3 seconds)
            - Light jitter (0-500ms)
    
    HIGH: Strong protection for privacy-sensitive applications.
          Noticeable latency impact.
          - Larger batches (4-8 messages)
          - Longer batch windows (3-10 seconds)
          - Moderate jitter (0-2000ms)
          - Optional constant-rate
    
    PARANOID: Maximum protection at the cost of significant latency.
              For high-security scenarios.
              - Large batches (8-16 messages)
              - Long batch windows (10-30 seconds)
              - Heavy jitter (0-5000ms)
              - Constant-rate sending enabled
              - Mix network integration (when available)
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PARANOID = "paranoid"


@dataclass
class BatchingConfig:
    """
    Configuration for message batching.
    
    Message batching collects outgoing messages and sends them together
    in batches at regular intervals. This obscures when individual
    messages were actually composed.
    
    Attributes:
        enabled: Whether batching is active
        min_batch_size: Minimum messages before sending (unless timeout)
        max_batch_size: Maximum messages per batch
        batch_interval_ms: Send batch after this many milliseconds
        randomize_order: Shuffle message order within batch
    """
    enabled: bool = False
    min_batch_size: int = 2
    max_batch_size: int = 8
    batch_interval_ms: int = 2000  # 2 seconds
    randomize_order: bool = True
    
    def get_effective_interval(self) -> float:
        """Get batch interval in seconds."""
        return self.batch_interval_ms / 1000.0


@dataclass
class TimingJitterConfig:
    """
    Configuration for timing jitter.
    
    Timing jitter adds random delays to message sending to obscure
    the exact timing of communications. This prevents observers from
    correlating send times across the network.
    
    Attributes:
        enabled: Whether jitter is active
        min_delay_ms: Minimum additional delay
        max_delay_ms: Maximum additional delay
        distribution: "uniform" or "exponential"
    """
    enabled: bool = False
    min_delay_ms: int = 0
    max_delay_ms: int = 500
    distribution: str = "uniform"  # "uniform" or "exponential"
    
    def get_jitter_delay(self) -> float:
        """
        Calculate a random jitter delay.
        
        Returns:
            Delay in seconds
        """
        if not self.enabled:
            return 0.0
        
        if self.distribution == "exponential":
            # Exponential distribution with mean at (max-min)/2
            # Use secrets.SystemRandom for security-sensitive timing jitter
            mean = (self.max_delay_ms - self.min_delay_ms) / 2
            delay = self.min_delay_ms + _secure_random.expovariate(1.0 / mean)
            delay = min(delay, self.max_delay_ms)
        else:
            # Uniform distribution
            # Use secrets.SystemRandom for security-sensitive timing jitter
            delay = _secure_random.uniform(self.min_delay_ms, self.max_delay_ms)
        
        return delay / 1000.0  # Convert to seconds


@dataclass
class ConstantRateConfig:
    """
    Configuration for constant-rate sending.
    
    Constant-rate sending ensures messages are sent at a fixed rate,
    with padding messages when there's nothing real to send. This
    provides the strongest protection against traffic analysis but
    uses more bandwidth.
    
    Attributes:
        enabled: Whether constant-rate mode is active
        messages_per_minute: Target send rate
        pad_to_size: Size to pad all messages to (bytes)
        allow_burst: Allow bursting above rate for queued messages
        max_burst_size: Maximum burst size if allowed
    """
    enabled: bool = False
    messages_per_minute: float = 10.0
    pad_to_size: int = 4096  # 4KB standard size
    allow_burst: bool = True
    max_burst_size: int = 5
    
    def get_send_interval(self) -> float:
        """Get interval between sends in seconds."""
        if self.messages_per_minute <= 0:
            return 60.0
        return 60.0 / self.messages_per_minute


@dataclass
class MixNetworkConfig:
    """
    Configuration for mix network integration.
    
    Mix networks provide strong anonymity by routing messages through
    multiple nodes that mix (reorder and delay) messages. This is a
    placeholder for future integration with mix networks like Nym.
    
    Attributes:
        enabled: Whether mix network routing is enabled
        provider_url: URL of the mix network provider
        min_hops: Minimum number of mix hops
        max_hops: Maximum number of mix hops
        loop_cover_traffic: Generate loopback cover traffic
    """
    enabled: bool = False
    provider_url: Optional[str] = None
    min_hops: int = 3
    max_hops: int = 5
    loop_cover_traffic: bool = True
    
    # Placeholder for future mix network client
    # In production, this would integrate with Nym, Loopix, etc.


@dataclass
class TrafficAnalysisMitigationConfig:
    """
    Comprehensive configuration for traffic analysis mitigations.
    
    This aggregates all mitigation settings and provides preset
    configurations for different privacy levels.
    
    Example:
        # Use medium privacy preset
        config = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.MEDIUM)
        
        # Or customize
        config = TrafficAnalysisMitigationConfig(
            batching=BatchingConfig(enabled=True, batch_interval_ms=5000),
            jitter=TimingJitterConfig(enabled=True, max_delay_ms=2000),
        )
    
    Attributes:
        privacy_level: The active privacy level preset
        batching: Message batching configuration
        jitter: Timing jitter configuration
        constant_rate: Constant-rate sending configuration
        mix_network: Mix network integration configuration
        adaptive: Adjust settings based on network conditions
        metrics_enabled: Track timing metrics for debugging
    """
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    jitter: TimingJitterConfig = field(default_factory=TimingJitterConfig)
    constant_rate: ConstantRateConfig = field(default_factory=ConstantRateConfig)
    mix_network: MixNetworkConfig = field(default_factory=MixNetworkConfig)
    adaptive: bool = False  # Adjust settings based on conditions
    metrics_enabled: bool = True  # Track timing metrics
    
    @classmethod
    def from_privacy_level(cls, level: PrivacyLevel) -> "TrafficAnalysisMitigationConfig":
        """
        Create configuration from a privacy level preset.
        
        Args:
            level: The desired privacy level
            
        Returns:
            Configured TrafficAnalysisMitigationConfig
        """
        if level == PrivacyLevel.LOW:
            return cls(
                privacy_level=level,
                batching=BatchingConfig(enabled=False),
                jitter=TimingJitterConfig(enabled=False),
                constant_rate=ConstantRateConfig(enabled=False),
                mix_network=MixNetworkConfig(enabled=False),
            )
        
        elif level == PrivacyLevel.MEDIUM:
            return cls(
                privacy_level=level,
                batching=BatchingConfig(
                    enabled=True,
                    min_batch_size=2,
                    max_batch_size=4,
                    batch_interval_ms=2000,
                    randomize_order=True,
                ),
                jitter=TimingJitterConfig(
                    enabled=True,
                    min_delay_ms=0,
                    max_delay_ms=500,
                    distribution="uniform",
                ),
                constant_rate=ConstantRateConfig(enabled=False),
                mix_network=MixNetworkConfig(enabled=False),
            )
        
        elif level == PrivacyLevel.HIGH:
            return cls(
                privacy_level=level,
                batching=BatchingConfig(
                    enabled=True,
                    min_batch_size=4,
                    max_batch_size=8,
                    batch_interval_ms=5000,
                    randomize_order=True,
                ),
                jitter=TimingJitterConfig(
                    enabled=True,
                    min_delay_ms=100,
                    max_delay_ms=2000,
                    distribution="exponential",
                ),
                constant_rate=ConstantRateConfig(
                    enabled=False,  # Optional at HIGH
                    messages_per_minute=6.0,
                ),
                mix_network=MixNetworkConfig(enabled=False),
            )
        
        elif level == PrivacyLevel.PARANOID:
            return cls(
                privacy_level=level,
                batching=BatchingConfig(
                    enabled=True,
                    min_batch_size=8,
                    max_batch_size=16,
                    batch_interval_ms=15000,
                    randomize_order=True,
                ),
                jitter=TimingJitterConfig(
                    enabled=True,
                    min_delay_ms=500,
                    max_delay_ms=5000,
                    distribution="exponential",
                ),
                constant_rate=ConstantRateConfig(
                    enabled=True,
                    messages_per_minute=4.0,
                    pad_to_size=4096,
                    allow_burst=False,  # Strict rate limiting
                ),
                mix_network=MixNetworkConfig(
                    enabled=False,  # Enable when mix network is available
                    min_hops=3,
                    max_hops=5,
                ),
            )
        
        # Default to LOW
        return cls(privacy_level=PrivacyLevel.LOW)
    
    def estimate_latency_overhead(self) -> Dict[str, Any]:
        """
        Estimate the latency overhead introduced by these settings.
        
        Returns:
            Dict with estimated delays for different scenarios
        """
        min_delay_ms = 0
        max_delay_ms = 0
        avg_delay_ms = 0
        
        # Batching delay
        if self.batching.enabled:
            max_delay_ms += self.batching.batch_interval_ms
            avg_delay_ms += self.batching.batch_interval_ms // 2
        
        # Jitter delay
        if self.jitter.enabled:
            min_delay_ms += self.jitter.min_delay_ms
            max_delay_ms += self.jitter.max_delay_ms
            if self.jitter.distribution == "exponential":
                # Exponential mean is roughly (max-min)/2
                avg_delay_ms += (
                    self.jitter.max_delay_ms - self.jitter.min_delay_ms
                ) // 2
            else:
                avg_delay_ms += (
                    self.jitter.min_delay_ms + self.jitter.max_delay_ms
                ) // 2
        
        # Constant rate delay (worst case: waiting for next slot)
        if self.constant_rate.enabled:
            interval_ms = int(self.constant_rate.get_send_interval() * 1000)
            max_delay_ms += interval_ms
            avg_delay_ms += interval_ms // 2
        
        return {
            "privacy_level": self.privacy_level.value,
            "min_delay_ms": min_delay_ms,
            "max_delay_ms": max_delay_ms,
            "avg_delay_ms": avg_delay_ms,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "privacy_level": self.privacy_level.value,
            "batching": {
                "enabled": self.batching.enabled,
                "min_batch_size": self.batching.min_batch_size,
                "max_batch_size": self.batching.max_batch_size,
                "batch_interval_ms": self.batching.batch_interval_ms,
                "randomize_order": self.batching.randomize_order,
            },
            "jitter": {
                "enabled": self.jitter.enabled,
                "min_delay_ms": self.jitter.min_delay_ms,
                "max_delay_ms": self.jitter.max_delay_ms,
                "distribution": self.jitter.distribution,
            },
            "constant_rate": {
                "enabled": self.constant_rate.enabled,
                "messages_per_minute": self.constant_rate.messages_per_minute,
                "pad_to_size": self.constant_rate.pad_to_size,
                "allow_burst": self.constant_rate.allow_burst,
                "max_burst_size": self.constant_rate.max_burst_size,
            },
            "mix_network": {
                "enabled": self.mix_network.enabled,
                "provider_url": self.mix_network.provider_url,
                "min_hops": self.mix_network.min_hops,
                "max_hops": self.mix_network.max_hops,
            },
            "adaptive": self.adaptive,
            "metrics_enabled": self.metrics_enabled,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrafficAnalysisMitigationConfig":
        """Deserialize configuration from dictionary."""
        privacy_level = PrivacyLevel(data.get("privacy_level", "low"))
        
        batching_data = data.get("batching", {})
        batching = BatchingConfig(
            enabled=batching_data.get("enabled", False),
            min_batch_size=batching_data.get("min_batch_size", 2),
            max_batch_size=batching_data.get("max_batch_size", 8),
            batch_interval_ms=batching_data.get("batch_interval_ms", 2000),
            randomize_order=batching_data.get("randomize_order", True),
        )
        
        jitter_data = data.get("jitter", {})
        jitter = TimingJitterConfig(
            enabled=jitter_data.get("enabled", False),
            min_delay_ms=jitter_data.get("min_delay_ms", 0),
            max_delay_ms=jitter_data.get("max_delay_ms", 500),
            distribution=jitter_data.get("distribution", "uniform"),
        )
        
        constant_rate_data = data.get("constant_rate", {})
        constant_rate = ConstantRateConfig(
            enabled=constant_rate_data.get("enabled", False),
            messages_per_minute=constant_rate_data.get("messages_per_minute", 10.0),
            pad_to_size=constant_rate_data.get("pad_to_size", 4096),
            allow_burst=constant_rate_data.get("allow_burst", True),
            max_burst_size=constant_rate_data.get("max_burst_size", 5),
        )
        
        mix_network_data = data.get("mix_network", {})
        mix_network = MixNetworkConfig(
            enabled=mix_network_data.get("enabled", False),
            provider_url=mix_network_data.get("provider_url"),
            min_hops=mix_network_data.get("min_hops", 3),
            max_hops=mix_network_data.get("max_hops", 5),
        )
        
        return cls(
            privacy_level=privacy_level,
            batching=batching,
            jitter=jitter,
            constant_rate=constant_rate,
            mix_network=mix_network,
            adaptive=data.get("adaptive", False),
            metrics_enabled=data.get("metrics_enabled", True),
        )


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# Quick access to preset configurations
PRIVACY_LOW = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.LOW)
PRIVACY_MEDIUM = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.MEDIUM)
PRIVACY_HIGH = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.HIGH)
PRIVACY_PARANOID = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.PARANOID)


def get_recommended_config(
    latency_sensitive: bool = False,
    bandwidth_limited: bool = False,
    high_security: bool = False,
) -> TrafficAnalysisMitigationConfig:
    """
    Get recommended configuration based on requirements.
    
    Args:
        latency_sensitive: If True, minimize latency overhead
        bandwidth_limited: If True, avoid constant-rate traffic
        high_security: If True, prioritize privacy over performance
        
    Returns:
        Recommended TrafficAnalysisMitigationConfig
    """
    if high_security and not latency_sensitive:
        if bandwidth_limited:
            # HIGH without constant-rate
            config = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.HIGH)
            config.constant_rate.enabled = False
            return config
        else:
            return PRIVACY_PARANOID
    
    elif latency_sensitive:
        if high_security:
            # MEDIUM with reduced delays
            config = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.MEDIUM)
            config.jitter.max_delay_ms = 200
            config.batching.batch_interval_ms = 1000
            return config
        else:
            return PRIVACY_LOW
    
    else:
        # Default balanced
        return PRIVACY_MEDIUM
