"""
Tests for Regional Router Preference (Issue #113).

Tests cover:
- Country to continent mapping
- Region score computation (same country, same continent, different)
- RouterRecord with region/coordinates
- Seed selection with region preference
- Discovery client region scoring
- Fallback when preferred region unavailable
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, AsyncMock

import pytest

from valence.network.seed import (
    SeedNode,
    SeedConfig,
    RouterRecord,
    COUNTRY_TO_CONTINENT,
    get_continent,
    compute_region_score,
)
from valence.network.discovery import (
    DiscoveryClient,
    RouterInfo,
)


# =============================================================================
# CONTINENT MAPPING TESTS
# =============================================================================


class TestContinentMapping:
    """Tests for country to continent mapping."""
    
    def test_north_america(self):
        """North American countries map to NA."""
        assert get_continent("US") == "NA"
        assert get_continent("CA") == "NA"
        assert get_continent("MX") == "NA"
    
    def test_europe(self):
        """European countries map to EU."""
        assert get_continent("GB") == "EU"
        assert get_continent("DE") == "EU"
        assert get_continent("FR") == "EU"
        assert get_continent("NL") == "EU"
        assert get_continent("SE") == "EU"
    
    def test_asia(self):
        """Asian countries map to AS."""
        assert get_continent("JP") == "AS"
        assert get_continent("CN") == "AS"
        assert get_continent("KR") == "AS"
        assert get_continent("SG") == "AS"
        assert get_continent("IN") == "AS"
    
    def test_south_america(self):
        """South American countries map to SA."""
        assert get_continent("BR") == "SA"
        assert get_continent("AR") == "SA"
        assert get_continent("CL") == "SA"
    
    def test_oceania(self):
        """Oceanian countries map to OC."""
        assert get_continent("AU") == "OC"
        assert get_continent("NZ") == "OC"
    
    def test_africa(self):
        """African countries map to AF."""
        assert get_continent("ZA") == "AF"
        assert get_continent("EG") == "AF"
        assert get_continent("NG") == "AF"
    
    def test_case_insensitive(self):
        """Country codes should be case-insensitive."""
        assert get_continent("us") == "NA"
        assert get_continent("Us") == "NA"
        assert get_continent("US") == "NA"
    
    def test_unknown_country(self):
        """Unknown country codes return None."""
        assert get_continent("XX") is None
        assert get_continent("ZZ") is None
    
    def test_none_input(self):
        """None input returns None."""
        assert get_continent(None) is None
    
    def test_empty_string(self):
        """Empty string returns None."""
        assert get_continent("") is None


# =============================================================================
# REGION SCORE TESTS
# =============================================================================


class TestRegionScore:
    """Tests for region score computation."""
    
    def test_same_country_full_score(self):
        """Same country should return full score (1.0)."""
        assert compute_region_score("US", "US") == 1.0
        assert compute_region_score("DE", "DE") == 1.0
        assert compute_region_score("JP", "JP") == 1.0
    
    def test_same_country_case_insensitive(self):
        """Same country matching should be case-insensitive."""
        assert compute_region_score("us", "US") == 1.0
        assert compute_region_score("US", "us") == 1.0
        assert compute_region_score("De", "dE") == 1.0
    
    def test_same_continent_partial_score(self):
        """Same continent should return partial score (0.5)."""
        # Both in Europe
        assert compute_region_score("DE", "FR") == 0.5
        assert compute_region_score("GB", "NL") == 0.5
        
        # Both in Asia
        assert compute_region_score("JP", "KR") == 0.5
        assert compute_region_score("SG", "IN") == 0.5
        
        # Both in North America
        assert compute_region_score("US", "CA") == 0.5
    
    def test_different_continent_zero_score(self):
        """Different continents should return zero score."""
        # US (NA) vs DE (EU)
        assert compute_region_score("US", "DE") == 0.0
        
        # JP (AS) vs BR (SA)
        assert compute_region_score("JP", "BR") == 0.0
        
        # AU (OC) vs ZA (AF)
        assert compute_region_score("AU", "ZA") == 0.0
    
    def test_unknown_router_region(self):
        """Unknown router region returns zero."""
        assert compute_region_score("XX", "US") == 0.0
        assert compute_region_score(None, "US") == 0.0
    
    def test_unknown_preferred_region(self):
        """Unknown preferred region returns zero."""
        assert compute_region_score("US", "XX") == 0.0
        assert compute_region_score("US", None) == 0.0
    
    def test_both_none(self):
        """Both None returns zero."""
        assert compute_region_score(None, None) == 0.0


# =============================================================================
# ROUTER RECORD WITH REGION TESTS
# =============================================================================


class TestRouterRecordRegion:
    """Tests for RouterRecord with region/coordinates."""
    
    def test_router_record_with_region(self):
        """RouterRecord should store region."""
        now = time.time()
        router = RouterRecord(
            router_id="test-router",
            endpoints=["10.0.0.1:8471"],
            capacity={},
            health={"last_seen": now, "uptime_pct": 99.0},
            regions=["us-west"],
            features=[],
            registered_at=now,
            router_signature="sig",
            region="US",
        )
        
        assert router.region == "US"
    
    def test_router_record_with_coordinates(self):
        """RouterRecord should store coordinates."""
        now = time.time()
        router = RouterRecord(
            router_id="test-router",
            endpoints=["10.0.0.1:8471"],
            capacity={},
            health={"last_seen": now, "uptime_pct": 99.0},
            regions=[],
            features=[],
            registered_at=now,
            router_signature="sig",
            region="US",
            coordinates=[37.7749, -122.4194],  # San Francisco
        )
        
        assert router.coordinates == [37.7749, -122.4194]
    
    def test_to_dict_includes_region(self):
        """to_dict should include region when set."""
        now = time.time()
        router = RouterRecord(
            router_id="test-router",
            endpoints=["10.0.0.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
            registered_at=now,
            router_signature="sig",
            region="DE",
            coordinates=[52.52, 13.405],  # Berlin
        )
        
        d = router.to_dict()
        assert d["region"] == "DE"
        assert d["coordinates"] == [52.52, 13.405]
    
    def test_to_dict_omits_none_region(self):
        """to_dict should omit region when None."""
        now = time.time()
        router = RouterRecord(
            router_id="test-router",
            endpoints=[],
            capacity={},
            health={},
            regions=[],
            features=[],
            registered_at=now,
            router_signature="sig",
        )
        
        d = router.to_dict()
        assert "region" not in d
        assert "coordinates" not in d
    
    def test_from_dict_with_region(self):
        """from_dict should restore region and coordinates."""
        data = {
            "router_id": "test-router",
            "region": "JP",
            "coordinates": [35.6762, 139.6503],  # Tokyo
        }
        
        router = RouterRecord.from_dict(data)
        
        assert router.region == "JP"
        assert router.coordinates == [35.6762, 139.6503]
    
    def test_from_dict_without_region(self):
        """from_dict should handle missing region gracefully."""
        data = {"router_id": "test-router"}
        
        router = RouterRecord.from_dict(data)
        
        assert router.region is None
        assert router.coordinates is None


# =============================================================================
# SEED SELECTION WITH REGION PREFERENCE TESTS
# =============================================================================


@pytest.fixture
def seed_config():
    """Create a test seed config."""
    config = SeedConfig(
        host="127.0.0.1",
        port=18470,
        seed_id="test-seed",
        weight_region=0.2,  # 20% weight for region
    )
    # Add missing config attributes for sybil resistance compatibility
    config.reputation_decay_period_hours = 24
    return config


@pytest.fixture
def mock_sybil_resistance():
    """Create a mock sybil resistance for testing."""
    mock = MagicMock()
    mock.check_registration = MagicMock(return_value=(True, None))
    mock.on_registration_success = MagicMock()
    mock.on_registration_failure = MagicMock()
    mock.is_trusted_for_discovery = MagicMock(return_value=True)
    mock.get_trust_factor = MagicMock(return_value=1.0)
    mock.reputation = MagicMock()
    mock.reputation.get_reputation = MagicMock(return_value=MagicMock(score=1.0))
    return mock


@pytest.fixture
def seed_node(seed_config, mock_sybil_resistance):
    """Create a test seed node with mocked sybil resistance."""
    node = SeedNode(config=seed_config)
    node.sybil_resistance = mock_sybil_resistance
    return node


@pytest.fixture
def regional_routers():
    """Create routers in different regions for testing.
    
    Note: Each router uses a different /16 subnet to avoid IP diversity filtering.
    """
    now = time.time()
    
    # US router (San Francisco) - subnet 10.0.x.x
    us_router = RouterRecord(
        router_id="router-us-001",
        endpoints=["10.0.1.1:8471"],
        capacity={"max_connections": 1000, "current_load_pct": 30},
        health={"last_seen": now, "uptime_pct": 99.0},
        regions=["us-west"],
        features=[],
        registered_at=now,
        router_signature="sig-us",
        region="US",
        coordinates=[37.7749, -122.4194],
    )
    
    # Canadian router (same continent as US) - subnet 172.16.x.x
    ca_router = RouterRecord(
        router_id="router-ca-001",
        endpoints=["172.16.1.1:8471"],
        capacity={"max_connections": 1000, "current_load_pct": 30},
        health={"last_seen": now, "uptime_pct": 99.0},
        regions=["ca-central"],
        features=[],
        registered_at=now,
        router_signature="sig-ca",
        region="CA",
        coordinates=[43.6532, -79.3832],  # Toronto
    )
    
    # German router (different continent) - subnet 192.168.x.x
    de_router = RouterRecord(
        router_id="router-de-001",
        endpoints=["192.168.1.1:8471"],
        capacity={"max_connections": 1000, "current_load_pct": 30},
        health={"last_seen": now, "uptime_pct": 99.0},
        regions=["eu-central"],
        features=[],
        registered_at=now,
        router_signature="sig-de",
        region="DE",
        coordinates=[52.52, 13.405],  # Berlin
    )
    
    # Japanese router (different continent) - subnet 10.1.x.x
    jp_router = RouterRecord(
        router_id="router-jp-001",
        endpoints=["10.1.1.1:8471"],
        capacity={"max_connections": 1000, "current_load_pct": 30},
        health={"last_seen": now, "uptime_pct": 99.0},
        regions=["ap-northeast"],
        features=[],
        registered_at=now,
        router_signature="sig-jp",
        region="JP",
        coordinates=[35.6762, 139.6503],  # Tokyo
    )
    
    return {
        "us": us_router,
        "ca": ca_router,
        "de": de_router,
        "jp": jp_router,
    }


class TestSeedRegionalSelection:
    """Tests for seed node regional router selection."""
    
    def test_prefers_same_region(self, seed_node, regional_routers):
        """Selection should prefer routers in the same region (country)."""
        for r in regional_routers.values():
            seed_node.router_registry[r.router_id] = r
        
        # Request US region
        selected = seed_node.select_routers(
            count=4,
            preferences={"preferred_region": "US"},
        )
        
        # US router should be first (highest score)
        assert len(selected) == 4
        assert selected[0].region == "US"
    
    def test_same_continent_second_choice(self, seed_node, regional_routers):
        """Same continent routers should score higher than different continent."""
        for r in regional_routers.values():
            seed_node.router_registry[r.router_id] = r
        
        # Request US region
        selected = seed_node.select_routers(
            count=4,
            preferences={"preferred_region": "US"},
        )
        
        # US should be first, CA (same continent) should be second
        assert selected[0].region == "US"
        assert selected[1].region == "CA"
    
    def test_fallback_to_any_region(self, seed_node, regional_routers):
        """Should fall back to any region if preferred unavailable."""
        # Only add European routers
        seed_node.router_registry[regional_routers["de"].router_id] = regional_routers["de"]
        
        # Request US region (not available)
        selected = seed_node.select_routers(
            count=1,
            preferences={"preferred_region": "US"},
        )
        
        # Should still return something (the German router)
        assert len(selected) == 1
        assert selected[0].region == "DE"
    
    def test_legacy_region_preference(self, seed_node, regional_routers):
        """Should support legacy 'region' key for backward compatibility."""
        for r in regional_routers.values():
            seed_node.router_registry[r.router_id] = r
        
        # Use legacy 'region' key
        selected = seed_node.select_routers(
            count=4,
            preferences={"region": "US"},  # Legacy key
        )
        
        # US router should still be preferred
        assert selected[0].region == "US"
    
    def test_no_region_preference(self, seed_node, regional_routers):
        """Without region preference, selection should be based on other factors."""
        for r in regional_routers.values():
            seed_node.router_registry[r.router_id] = r
        
        # No region preference
        selected = seed_node.select_routers(
            count=4,
            preferences={},
        )
        
        # Should return all routers (order based on other scoring factors)
        assert len(selected) == 4
    
    def test_score_difference_same_region_vs_same_continent(self, seed_node, regional_routers):
        """Same region should score higher than same continent."""
        us_router = regional_routers["us"]
        ca_router = regional_routers["ca"]
        
        preferences = {"preferred_region": "US"}
        
        us_score = seed_node._score_router(us_router, preferences)
        ca_score = seed_node._score_router(ca_router, preferences)
        
        # US (same country) should score higher than CA (same continent)
        assert us_score > ca_score
        
        # The difference should be approximately half the region weight
        # (1.0 * weight vs 0.5 * weight)
        # Allow larger tolerance due to deterministic random component in scoring
        expected_diff = seed_node.config.weight_region * 0.5
        actual_diff = us_score - ca_score
        assert abs(actual_diff - expected_diff) < 0.06
    
    def test_score_difference_same_continent_vs_different(self, seed_node, regional_routers):
        """Same continent should score higher than different continent."""
        ca_router = regional_routers["ca"]  # CA, same continent as US
        de_router = regional_routers["de"]  # DE, different continent
        
        preferences = {"preferred_region": "US"}
        
        ca_score = seed_node._score_router(ca_router, preferences)
        de_score = seed_node._score_router(de_router, preferences)
        
        # CA (same continent) should score higher than DE (different)
        assert ca_score > de_score


# =============================================================================
# DISCOVERY CLIENT REGION TESTS
# =============================================================================


class TestDiscoveryClientRegion:
    """Tests for discovery client regional scoring."""
    
    def test_router_info_with_region(self):
        """RouterInfo should store region and coordinates."""
        router = RouterInfo(
            router_id="test",
            endpoints=["10.0.0.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
            region="US",
            coordinates=[37.7749, -122.4194],
        )
        
        assert router.region == "US"
        assert router.coordinates == [37.7749, -122.4194]
    
    def test_router_info_to_dict_with_region(self):
        """RouterInfo.to_dict should include region."""
        router = RouterInfo(
            router_id="test",
            endpoints=[],
            capacity={},
            health={},
            regions=[],
            features=[],
            region="DE",
            coordinates=[52.52, 13.405],
        )
        
        d = router.to_dict()
        assert d["region"] == "DE"
        assert d["coordinates"] == [52.52, 13.405]
    
    def test_router_info_from_dict_with_region(self):
        """RouterInfo.from_dict should parse region."""
        data = {
            "router_id": "test",
            "region": "JP",
            "coordinates": [35.6762, 139.6503],
        }
        
        router = RouterInfo.from_dict(data)
        
        assert router.region == "JP"
        assert router.coordinates == [35.6762, 139.6503]
    
    def test_client_score_same_region(self):
        """Client should give highest score to same region."""
        client = DiscoveryClient(verify_signatures=False)
        
        router = RouterInfo(
            router_id="test",
            endpoints=[],
            capacity={"current_load_pct": 30},
            health={"uptime_pct": 99},
            regions=[],
            features=[],
            region="US",
        )
        
        score_match = client._score_router(router, {"preferred_region": "US"})
        score_no_pref = client._score_router(router, {})
        
        # With matching region, score should be higher
        assert score_match > score_no_pref
    
    def test_client_score_same_continent(self):
        """Client should give partial score to same continent."""
        client = DiscoveryClient(verify_signatures=False)
        
        router_us = RouterInfo(
            router_id="us",
            endpoints=[],
            capacity={"current_load_pct": 30},
            health={"uptime_pct": 99},
            regions=[],
            features=[],
            region="US",
        )
        
        router_ca = RouterInfo(
            router_id="ca",
            endpoints=[],
            capacity={"current_load_pct": 30},
            health={"uptime_pct": 99},
            regions=[],
            features=[],
            region="CA",
        )
        
        router_de = RouterInfo(
            router_id="de",
            endpoints=[],
            capacity={"current_load_pct": 30},
            health={"uptime_pct": 99},
            regions=[],
            features=[],
            region="DE",
        )
        
        prefs = {"preferred_region": "US"}
        
        score_us = client._score_router(router_us, prefs)
        score_ca = client._score_router(router_ca, prefs)
        score_de = client._score_router(router_de, prefs)
        
        # US > CA > DE
        assert score_us > score_ca
        assert score_ca > score_de
    
    def test_client_continent_lookup(self):
        """Client should correctly look up continents."""
        client = DiscoveryClient()
        
        assert client._get_continent("US") == "NA"
        assert client._get_continent("CA") == "NA"
        assert client._get_continent("DE") == "EU"
        assert client._get_continent("JP") == "AS"
        assert client._get_continent("AU") == "OC"
        assert client._get_continent("XX") is None
    
    def test_client_region_score_computation(self):
        """Client should compute region scores correctly."""
        client = DiscoveryClient()
        
        # Same country
        assert client._compute_region_score("US", "US") == 1.0
        
        # Same continent
        assert client._compute_region_score("US", "CA") == 0.5
        
        # Different continent
        assert client._compute_region_score("US", "DE") == 0.0
        
        # Unknown
        assert client._compute_region_score("XX", "US") == 0.0
        assert client._compute_region_score(None, "US") == 0.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestRegionalPreferenceIntegration:
    """Integration tests for regional preference end-to-end."""
    
    @pytest.mark.asyncio
    async def test_discover_with_region_preference(self, seed_node, regional_routers):
        """Discovery request with region preference should be honored."""
        # Register all routers
        for r in regional_routers.values():
            seed_node.router_registry[r.router_id] = r
        
        # Create mock request with preferred_region
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "requested_count": 4,
            "preferences": {"preferred_region": "US"},
        })
        
        response = await seed_node.handle_discover(request)
        
        assert response.status == 200
        import json
        data = json.loads(response.text)
        
        # First router should be US
        assert len(data["routers"]) == 4
        assert data["routers"][0]["region"] == "US"
    
    @pytest.mark.asyncio
    async def test_register_with_region(self, seed_node):
        """Registration should accept and store region/coordinates."""
        seed_node.config.verify_signatures = False
        seed_node.config.verify_pow = False
        seed_node.config.probe_endpoints = False
        
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "router_id": "new-router-001",
            "endpoints": ["10.0.0.1:8471"],
            "region": "US",
            "coordinates": [37.7749, -122.4194],
            "router_signature": "sig",
        })
        request.remote = "127.0.0.1"
        
        response = await seed_node.handle_register(request)
        
        assert response.status == 200
        
        # Verify router was registered with region
        router = seed_node.router_registry["new-router-001"]
        assert router.region == "US"
        assert router.coordinates == [37.7749, -122.4194]
